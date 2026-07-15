# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import MHCPostOp
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
)
from vllm.transformers_utils.configs.dk_deepseek_v4_kda_utils import (
    build_dk_kimi_linear_config,
    get_dk_projection_dims,
    normalize_dk_kda_layer_indices,
)

from .model import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)


class DKKimiKDALayerAdapter(nn.Module):
    """Project DeepSeek V4 multi-carrier states into one Kimi KDA layer."""

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if tp_size != 1:
            raise NotImplementedError(
                "DK KDA projection adapter currently supports tensor_parallel_size=1"
            )

        deepseek_hidden, kimi_hidden, deepseek_flat_hidden = get_dk_projection_dims(
            config
        )
        self.deepseek_hidden = deepseek_hidden
        self.deepseek_flat_hidden = deepseek_flat_hidden
        self.hc_mult = deepseek_flat_hidden // deepseek_hidden
        params_dtype = vllm_config.model_config.dtype

        self.dk_in_proj = ReplicatedLinear(
            deepseek_flat_hidden,
            kimi_hidden,
            bias=True,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.dk_in_proj",
        )
        self.kda_layer = KimiGatedDeltaNetAttention(
            build_dk_kimi_linear_config(config, layer_idx),
            vllm_config,
            prefix=f"{prefix}.kda_layer",
        )
        self.dk_out_proj = ReplicatedLinear(
            kimi_hidden,
            deepseek_flat_hidden,
            bias=True,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.dk_out_proj",
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        if hidden_states.dim() != 3:
            raise ValueError(
                "DK KDA adapter expects DeepSeek V4 hidden states shaped "
                "[num_tokens, hc_mult, hidden_size]"
            )
        num_tokens, hc_mult, hidden_size = hidden_states.shape
        if hc_mult != self.hc_mult or hidden_size != self.deepseek_hidden:
            raise ValueError(
                "Unexpected DK hidden state shape "
                f"{tuple(hidden_states.shape)}; expected "
                f"[num_tokens, {self.hc_mult}, {self.deepseek_hidden}]"
            )

        flat_states = hidden_states.reshape(num_tokens, self.deepseek_flat_hidden)
        kimi_states = self.dk_in_proj(flat_states)[0]
        kda_output = torch.empty_like(kimi_states)
        self.kda_layer(
            hidden_states=kimi_states,
            positions=positions,
            output=kda_output,
        )
        deepseek_flat = self.dk_out_proj(kda_output)[0]
        return deepseek_flat.reshape(num_tokens, hc_mult, hidden_size)


class DKDeepseekV4KDADecoderLayer(nn.Module):
    def __new__(
        cls,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        config = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)
        if layer_idx not in normalize_dk_kda_layer_indices(config):
            return DeepseekV4DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        return super().__new__(cls)

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layer_idx = extract_layer_index(prefix)
        self.adapter = DKKimiKDALayerAdapter(
            config,
            vllm_config,
            self.layer_idx,
            prefix=prefix,
        )
        self.mhc_post = MHCPostOp()

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ):
        if residual is None:
            return x
        if post is None or comb is None:
            raise ValueError("Pending DeepSeek V4 MHC state is incomplete")
        return self.mhc_post(x, residual, post, comb)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        x = self.hc_post(x, residual, post_mix, res_mix)
        x = self.adapter(x, positions)
        return x, None, None, None


class DKDeepseekV4KDAModel(DeepseekV4Model):
    layer_cls = DKDeepseekV4KDADecoderLayer

    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "finalize_mega_moe_weights"):
                ffn.finalize_mega_moe_weights()


class DKDeepseekV4KDAForCausalLM(
    DeepseekV4ForCausalLM, HasInnerState, IsHybrid
):
    has_inner_state = True
    is_hybrid = True
    model_cls = DKDeepseekV4KDAModel

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        kimi_config = build_dk_kimi_linear_config(
            hf_config,
            normalize_dk_kda_layer_indices(hf_config)[0],
        )
        linear_attn_config = kimi_config.linear_attn_config
        assert linear_attn_config is not None
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            parallel_config.tensor_parallel_size,
            linear_attn_config["num_heads"],
            linear_attn_config["head_dim"],
            conv_kernel_size=linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[
        MambaStateCopyFunc,
        MambaStateCopyFunc,
        MambaStateCopyFunc,
        MambaStateCopyFunc,
    ]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_substrs = ["mtp."]
        for layer_idx in normalize_dk_kda_layer_indices(self.config):
            layer_prefix = f"model.layers.{layer_idx}."
            skip_substrs.extend(
                [
                    f"{layer_prefix}attn.",
                    f"{layer_prefix}ffn.",
                    f"{layer_prefix}attn_norm.",
                    f"{layer_prefix}ffn_norm.",
                    f"{layer_prefix}hc_attn_",
                    f"{layer_prefix}hc_ffn_",
                ]
            )
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        return loaded_params
