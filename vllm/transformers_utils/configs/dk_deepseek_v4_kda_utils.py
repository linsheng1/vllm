# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from copy import deepcopy
from typing import Any

from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

DEFAULT_DK_KDA_LAYERS_1_BASED = (11, 21, 31)


def normalize_dk_kda_layer_indices(config: Any) -> tuple[int, ...]:
    """Return DK KDA replacement layers as sorted 0-based indices."""
    if hasattr(config, "dk_kda_layer_indices"):
        layers = tuple(int(layer) for layer in config.dk_kda_layer_indices)
        one_based = False
    else:
        layers = tuple(
            int(layer)
            for layer in getattr(config, "dk_kda_layers", DEFAULT_DK_KDA_LAYERS_1_BASED)
        )
        one_based = bool(getattr(config, "dk_kda_layers_are_1_based", True))

    if one_based:
        if any(layer <= 0 for layer in layers):
            raise ValueError("1-based DK KDA layers must be positive")
        layers = tuple(layer - 1 for layer in layers)

    if any(layer < 0 for layer in layers):
        raise ValueError("0-based DK KDA layer indices must be non-negative")

    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and any(
        layer >= int(num_hidden_layers) for layer in layers
    ):
        raise ValueError(
            "DK KDA layer index is outside num_hidden_layers="
            f"{num_hidden_layers}: {layers}"
        )

    return tuple(sorted(set(layers)))


def get_dk_projection_dims(config: Any) -> tuple[int, int, int]:
    """Return (DeepSeek stream hidden, Kimi hidden, DeepSeek flat hidden)."""
    deepseek_hidden = int(config.hidden_size)
    hc_mult = int(getattr(config, "hc_mult", 1))
    deepseek_flat_hidden = deepseek_hidden * hc_mult
    kimi_hidden = int(_get_kimi_config_dict(config).get("hidden_size", 0))
    if kimi_hidden <= 0:
        raise ValueError("DK Kimi config must define a positive hidden_size")
    return deepseek_hidden, kimi_hidden, deepseek_flat_hidden


def get_dk_layers_block_type(config: Any) -> list[str]:
    """Build vLLM hybrid layer tags for DK."""
    num_hidden_layers = int(config.num_hidden_layers)
    kda_indices = set(normalize_dk_kda_layer_indices(config))
    return [
        "linear_attention" if layer_idx in kda_indices else "attention"
        for layer_idx in range(num_hidden_layers)
    ]


def build_dk_kimi_linear_config(config: Any, layer_idx: int) -> KimiLinearConfig:
    """Build a KimiLinearConfig scoped to a single DK KDA replacement layer."""
    kimi_config_dict = _get_kimi_config_dict(config)
    linear_attn_config = deepcopy(kimi_config_dict.get("linear_attn_config"))
    if linear_attn_config is None:
        raise ValueError("DK Kimi config must include linear_attn_config")

    required_keys = ("head_dim", "num_heads", "short_conv_kernel_size")
    missing = [key for key in required_keys if key not in linear_attn_config]
    if missing:
        raise ValueError(
            "DK Kimi linear_attn_config is missing required keys: "
            f"{', '.join(missing)}"
        )

    linear_attn_config["kda_layers"] = [layer_idx + 1]
    linear_attn_config.setdefault("full_attn_layers", [])
    kimi_config_dict["linear_attn_config"] = linear_attn_config
    kimi_config_dict.setdefault("num_hidden_layers", int(config.num_hidden_layers))
    kimi_config_dict.setdefault("hidden_act", getattr(config, "hidden_act", "silu"))
    kimi_config_dict.setdefault("rms_norm_eps", getattr(config, "rms_norm_eps", 1e-6))
    return KimiLinearConfig(**kimi_config_dict)


def _get_kimi_config_dict(config: Any) -> dict[str, Any]:
    kimi_config = getattr(config, "dk_kimi_linear_config", None)
    if kimi_config is None:
        kimi_config = getattr(config, "kimi_linear_config", None)
    if kimi_config is None:
        raise ValueError(
            "DK config must include dk_kimi_linear_config copied from "
            "moonshotai/Kimi-Linear-48B-A3B-Instruct"
        )
    if isinstance(kimi_config, KimiLinearConfig):
        kimi_config = kimi_config.to_dict()
    if not isinstance(kimi_config, dict):
        raise TypeError("dk_kimi_linear_config must be a dict or KimiLinearConfig")
    return deepcopy(kimi_config)
