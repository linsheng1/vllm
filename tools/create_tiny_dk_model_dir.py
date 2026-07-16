# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a tiny DK model directory for vLLM dummy-weight smoke tests.

The output is intentionally weightless.  Run vLLM with ``--load-format dummy``
so the model loader initializes random parameters from the instantiated module
shapes.  This keeps the A10 validation focused on config/model wiring, layer
replacement, KDA state allocation, and hybrid KV cache behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DK_KDA_LAYERS = (11, 21, 31)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def build_layers_block_type(num_hidden_layers: int, dk_kda_layers: tuple[int, ...]):
    replaced = {layer - 1 for layer in dk_kda_layers}
    return [
        "linear_attention" if layer_idx in replaced else "attention"
        for layer_idx in range(num_hidden_layers)
    ]


def build_kimi_linear_config(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    kda_num_heads: int,
    kda_head_dim: int,
    kda_conv_kernel: int,
    vocab_size: int,
) -> dict[str, Any]:
    return {
        "model_type": "kimi_linear",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "head_dim": hidden_size // num_attention_heads,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "tie_word_embeddings": False,
        "q_lora_rank": None,
        "kv_lora_rank": None,
        "qk_nope_head_dim": None,
        "qk_rope_head_dim": None,
        "v_head_dim": None,
        "mla_use_nope": False,
        "num_experts": None,
        "num_experts_per_token": None,
        "num_shared_experts": 0,
        "moe_intermediate_size": None,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 0,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [],
            "num_heads": kda_num_heads,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": kda_conv_kernel,
        },
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    dk_kda_layers = tuple(args.dk_kda_layers)
    min_layers = max(dk_kda_layers)
    if args.num_hidden_layers < min_layers:
        raise ValueError(
            f"--num-hidden-layers must be >= {min_layers} for DK KDA layers "
            f"{dk_kda_layers}"
        )
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-attention-heads")
    if args.o_groups > args.num_attention_heads:
        raise ValueError("--o-groups must be <= --num-attention-heads")
    if args.num_attention_heads % args.o_groups != 0:
        raise ValueError("--num-attention-heads must be divisible by --o-groups")

    kimi_config = build_kimi_linear_config(
        hidden_size=args.kimi_hidden_size,
        intermediate_size=args.kimi_intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.kimi_attention_heads,
        kda_num_heads=args.kda_num_heads,
        kda_head_dim=args.kda_head_dim,
        kda_conv_kernel=args.kda_conv_kernel,
        vocab_size=args.vocab_size,
    )

    return {
        "architectures": ["DKDeepseekV4KDAForCausalLM"],
        "model_type": "dk_deepseek_v4_kda",
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "moe_intermediate_size": args.moe_intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "head_dim": args.head_dim,
        "q_lora_rank": args.q_lora_rank,
        "o_lora_rank": args.o_lora_rank,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "kv_lora_rank": args.head_dim,
        "v_head_dim": args.head_dim,
        "o_groups": args.o_groups,
        "sliding_window": args.sliding_window,
        "compress_ratios": [1] * args.num_hidden_layers,
        "index_head_dim": args.index_head_dim,
        "index_n_heads": args.index_n_heads,
        "index_topk": args.index_topk,
        "num_hash_layers": 0,
        "n_routed_experts": args.n_routed_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "n_shared_experts": None,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sqrtsoftplus",
        "swiglu_limit": None,
        "topk_method": "greedy",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "hc_eps": 1e-6,
        "hc_mult": args.hc_mult,
        "hc_sinkhorn_iters": 0,
        "max_position_embeddings": args.max_position_embeddings,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {"rope_type": "default", "rope_theta": 10000.0},
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "quantization_config": {"scale_fmt": "e4m3"},
        "torch_dtype": args.torch_dtype,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "dk_kda_layers": list(dk_kda_layers),
        "dk_kda_layers_are_1_based": True,
        "dk_kda_layer_indices": [layer - 1 for layer in dk_kda_layers],
        "dk_kimi_linear_config": kimi_config,
        "layers_block_type": build_layers_block_type(
            args.num_hidden_layers, dk_kda_layers
        ),
    }


def build_tokenizer_json(vocab_size: int) -> dict[str, Any]:
    base_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    vocab = {token: idx for idx, token in enumerate(base_tokens)}
    for idx in range(len(base_tokens), vocab_size):
        vocab[f"tok_{idx}"] = idx
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": idx,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, idx in vocab.items()
            if token.startswith("<")
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }


def write_tokenizer_files(output_dir: Path, vocab_size: int) -> None:
    write_json(output_dir / "tokenizer.json", build_tokenizer_json(vocab_size))
    write_json(
        output_dir / "tokenizer_config.json",
        {
            "model_max_length": 1000000000000000019884624838656,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
    )
    write_json(
        output_dir / "special_tokens_map.json",
        {
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
    )
    write_json(
        output_dir / "generation_config.json",
        {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny DK config/tokenizer directory for dummy loads."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=32)
    parser.add_argument("--dk-kda-layers", type=int, nargs="+", default=list(DEFAULT_DK_KDA_LAYERS))
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--moe-intermediate-size", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-lora-rank", type=int, default=64)
    parser.add_argument("--o-lora-rank", type=int, default=64)
    parser.add_argument("--qk-rope-head-dim", type=int, default=32)
    parser.add_argument("--o-groups", type=int, default=1)
    parser.add_argument("--index-head-dim", type=int, default=16)
    parser.add_argument("--index-n-heads", type=int, default=2)
    parser.add_argument("--index-topk", type=int, default=4)
    parser.add_argument("--n-routed-experts", type=int, default=2)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--hc-mult", type=int, default=1)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--kimi-hidden-size", type=int, default=256)
    parser.add_argument("--kimi-intermediate-size", type=int, default=512)
    parser.add_argument("--kimi-attention-heads", type=int, default=4)
    parser.add_argument("--kda-num-heads", type=int, default=4)
    parser.add_argument("--kda-head-dim", type=int, default=64)
    parser.add_argument("--kda-conv-kernel", type=int, default=4)
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{args.output_dir} already exists and is not empty; pass --force"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    write_json(args.output_dir / "config.json", config)
    write_tokenizer_files(args.output_dir, args.vocab_size)
    print(f"Wrote tiny DK dummy-load model directory to {args.output_dir}")
    print("Run vLLM with: --load-format dummy")


if __name__ == "__main__":
    main()
