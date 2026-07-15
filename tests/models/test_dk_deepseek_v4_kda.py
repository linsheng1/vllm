# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

pytestmark = pytest.mark.cpu_test

DK_ARCH = "DKDeepseekV4KDAForCausalLM"
TWO_M_TOKENS = 2 * 1024 * 1024


def _import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr_name, None)


def _find_first_attr(candidates: tuple[tuple[str, str], ...]) -> Any:
    for module_name, attr_name in candidates:
        attr = _import_attr(module_name, attr_name)
        if attr is not None:
            return attr
    pytest.xfail(
        "DK DeepSeek V4 KDA API is not registered yet. Expected one of: "
        + ", ".join(f"{module}.{attr}" for module, attr in candidates)
    )


def _shape_from(value: Any) -> tuple[int, int]:
    if isinstance(value, torch.Size):
        value = tuple(value)
    if isinstance(value, tuple | list):
        return (int(value[0]), int(value[1]))
    if hasattr(value, "shape"):
        return _shape_from(value.shape)
    if hasattr(value, "out_features") and hasattr(value, "in_features"):
        return (int(value.out_features), int(value.in_features))
    raise TypeError(f"Unsupported projection shape value: {value!r}")


def _extract_projection_shapes(value: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(value, dict):
        ds_to_kimi = (
            value.get("deepseek_to_kimi")
            or value.get("dk_to_kimi")
            or value.get("in_proj")
            or value.get("down_proj")
        )
        kimi_to_ds = (
            value.get("kimi_to_deepseek")
            or value.get("kimi_to_dk")
            or value.get("out_proj")
            or value.get("up_proj")
        )
        if ds_to_kimi is not None and kimi_to_ds is not None:
            return _shape_from(ds_to_kimi), _shape_from(kimi_to_ds)

    for first_name, second_name in (
        ("deepseek_to_kimi", "kimi_to_deepseek"),
        ("dk_to_kimi", "kimi_to_dk"),
        ("in_proj", "out_proj"),
        ("down_proj", "up_proj"),
    ):
        if hasattr(value, first_name) and hasattr(value, second_name):
            return (
                _shape_from(getattr(value, first_name)),
                _shape_from(getattr(value, second_name)),
            )

    if isinstance(value, tuple | list):
        if len(value) == 2:
            return _shape_from(value[0]), _shape_from(value[1])
        if len(value) == 3 and all(isinstance(item, int) for item in value):
            _, kimi_hidden_size, deepseek_flat_hidden = value
            return (
                (kimi_hidden_size, deepseek_flat_hidden),
                (deepseek_flat_hidden, kimi_hidden_size),
            )

    raise TypeError(f"Unsupported projection shape contract: {value!r}")


def test_dk_replace_layers_normalize_1_based_user_config_to_0_based():
    normalize_layers = _find_first_attr(
        (
            (
                "vllm.transformers_utils.configs.dk_deepseek_v4_kda_utils",
                "normalize_dk_kda_layer_indices",
            ),
            (
                "vllm.models.deepseek_v4.dk_kda",
                "normalize_dk_replace_layers",
            ),
            (
                "vllm.models.deepseek_v4.kda",
                "normalize_dk_replace_layers",
            ),
            (
                "vllm.models.deepseek_v4",
                "normalize_dk_replace_layers",
            ),
        )
    )
    config = SimpleNamespace(
        dk_kda_layers=[11, 21, 31],
        dk_kda_layers_are_1_based=True,
    )
    try:
        normalized_layers = normalize_layers(config)
    except TypeError:
        normalized_layers = normalize_layers([11, 21, 31])

    assert list(normalized_layers) == [10, 20, 30]


def test_dk_deepseek_v4_kda_arch_is_registered():
    try:
        from vllm.model_executor.models.registry import ModelRegistry, _VLLM_MODELS
    except Exception as exc:
        pytest.xfail(f"ModelRegistry cannot be imported in this environment: {exc}")

    if DK_ARCH not in ModelRegistry.get_supported_archs():
        pytest.xfail(f"{DK_ARCH} is not in ModelRegistry supported archs yet")

    assert DK_ARCH in _VLLM_MODELS
    model_cls = ModelRegistry._try_load_model_cls(DK_ARCH)
    assert model_cls is not None
    assert model_cls.__name__ == DK_ARCH


def test_dk_projection_shape_contract_uses_hc_expanded_deepseek_width():
    get_projection_shapes = _find_first_attr(
        (
            (
                "vllm.transformers_utils.configs.dk_deepseek_v4_kda_utils",
                "get_dk_projection_dims",
            ),
            (
                "vllm.models.deepseek_v4.dk_kda",
                "get_dk_projection_shapes",
            ),
            (
                "vllm.models.deepseek_v4.kda",
                "get_dk_projection_shapes",
            ),
            (
                "vllm.models.deepseek_v4",
                "get_dk_projection_shapes",
            ),
        )
    )
    deepseek_hidden_size = 7168
    kimi_hidden_size = 4096
    hc_mult = 4

    try:
        shapes = get_projection_shapes(
            deepseek_hidden_size=deepseek_hidden_size,
            kimi_hidden_size=kimi_hidden_size,
            hc_mult=hc_mult,
        )
    except TypeError:
        config = SimpleNamespace(
            hidden_size=deepseek_hidden_size,
            hc_mult=hc_mult,
            dk_kimi_linear_config={"hidden_size": kimi_hidden_size},
        )
        shapes = get_projection_shapes(config)
    ds_to_kimi, kimi_to_ds = _extract_projection_shapes(shapes)

    hc_hidden_size = deepseek_hidden_size * hc_mult
    assert ds_to_kimi == (kimi_hidden_size, hc_hidden_size)
    assert kimi_to_ds == (hc_hidden_size, kimi_hidden_size)


def test_kv_spec_kind_distinguishes_deepseek_mla_from_kda_mamba():
    mla_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
        cache_dtype_str="fp8_ds_mla",
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    kda_spec = MambaSpec(
        block_size=128,
        shapes=((2, 512), (3, 32, 32)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="all",
    )

    assert get_kv_cache_spec_kind(mla_spec) == KVCacheSpecKind.MLA_ATTENTION
    assert get_kv_cache_spec_kind(kda_spec) == KVCacheSpecKind.MAMBA
    assert not UniformTypeKVCacheSpecs.is_uniform_type(
        {
            "model.layers.0.self_attn.mla": mla_spec,
            "model.layers.10.kda": kda_spec,
        }
    )


def test_two_million_max_model_len_is_only_planned_not_allocated():
    mla_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
        cache_dtype_str="fp8_ds_mla",
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    kda_spec = MambaSpec(
        block_size=128,
        shapes=((2, 512), (3, 32, 32)),
        dtypes=(torch.float32, torch.float32),
        num_speculative_blocks=0,
        mamba_cache_mode="all",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=TWO_M_TOKENS),
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    expected_blocks = TWO_M_TOKENS // mla_spec.block_size

    with (
        patch("torch.empty", side_effect=AssertionError("unexpected allocation")),
        patch("torch.zeros", side_effect=AssertionError("unexpected allocation")),
    ):
        mla_bytes = mla_spec.max_memory_usage_bytes(vllm_config)
        kda_bytes = kda_spec.max_memory_usage_bytes(vllm_config)

    assert expected_blocks == 16_384
    assert mla_bytes == expected_blocks * mla_spec.page_size_bytes
    assert kda_bytes == expected_blocks * kda_spec.page_size_bytes
