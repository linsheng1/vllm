# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a DK checkpoint from DeepSeek-V4-Flash and Kimi-Linear weights.

This script intentionally keeps the conversion policy simple and explicit:

* copy tokenizer/config side files from the DeepSeek checkpoint,
* stream-copy DeepSeek safetensors except replaced layers,
* stream-copy Kimi KDA tensors into the DK target layers,
* synthesize dk_in_proj/dk_out_proj bridge weights.

It is meant to be importable and structurally testable without downloading the
real models.  It does not try to train, distill, or make the hybrid accurate.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_REPLACE_LAYERS = (10, 20, 30)
DEFAULT_MAX_SHARD_SIZE = "5GB"
WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    filename: Path


@dataclass(frozen=True)
class KdaLayerMapping:
    source_layer: int
    target_layer: int
    source_prefix: str
    target_prefix: str


def parse_int_list(value: str | None) -> list[int]:
    if value is None or value == "":
        return []
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected a comma-separated integer list, got {value!r}"
            ) from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError(
                f"Layer indices must be 0-based non-negative integers: {value!r}"
            )
        result.append(parsed)
    return result


def parse_size_to_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?b?)?\s*", value.lower())
    if match is None:
        raise argparse.ArgumentTypeError(f"Invalid size: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2) or "b"
    multiplier = {
        "b": 1,
        "kb": 1000,
        "k": 1000,
        "mb": 1000**2,
        "m": 1000**2,
        "gb": 1000**3,
        "g": 1000**3,
        "tb": 1000**4,
        "t": 1000**4,
    }[unit]
    return int(amount * multiplier)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def ensure_runtime_dependencies() -> None:
    missing = [
        name
        for name in ("torch", "safetensors")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing required conversion dependencies: "
            f"{', '.join(missing)}. Install them in the vLLM environment before "
            "running a real checkpoint conversion."
        )


def is_weight_file(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in WEIGHT_FILE_PATTERNS)


def copy_side_files(src: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] in {".git", "__pycache__"}:
            continue
        if is_weight_file(path):
            continue
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied.append(str(rel))
    return copied


def discover_safetensors(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} is missing a weight_map object")
        names = sorted({str(name) for name in weight_map.values()})
        return [checkpoint_dir / name for name in names]

    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if files:
        return files

    raise FileNotFoundError(
        f"No safetensors weights found under {checkpoint_dir}. "
        "This tool only streams safetensors checkpoints."
    )


def iter_safetensor_records(checkpoint_dir: Path) -> Iterator[TensorRecord]:
    from safetensors import safe_open

    for filename in discover_safetensors(checkpoint_dir):
        if not filename.exists():
            raise FileNotFoundError(filename)
        with safe_open(filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield TensorRecord(name=key, filename=filename)


def layer_index_from_key(name: str) -> int | None:
    match = LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def should_skip_deepseek_key(
    name: str,
    replace_layers: set[int],
    replace_scope: str,
) -> bool:
    layer_idx = layer_index_from_key(name)
    if layer_idx not in replace_layers:
        return False
    if replace_scope == "layer":
        return True
    return f".layers.{layer_idx}.attn." in name


def infer_hidden_size(config: dict[str, Any], override: int | None, label: str) -> int:
    if override is not None:
        return override
    hidden_size = config.get("hidden_size")
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        hidden_size = text_config.get("hidden_size")
        if isinstance(hidden_size, int) and hidden_size > 0:
            return hidden_size
    raise ValueError(f"Could not infer {label} hidden_size; pass an override")


def select_kimi_layers(
    kimi_config: dict[str, Any],
    count: int,
    fallback_layers: Iterable[int],
    override_layers: list[int],
) -> list[int]:
    if override_layers:
        if len(override_layers) != count:
            raise ValueError(
                "--kimi-layers must have the same length as --replace-layers "
                f"({len(override_layers)} != {count})"
            )
        return override_layers

    linear_attn_config = kimi_config.get("linear_attn_config")
    if isinstance(linear_attn_config, dict):
        kda_layers = linear_attn_config.get("kda_layers")
        if isinstance(kda_layers, list) and len(kda_layers) >= count:
            selected: list[int] = []
            for item in kda_layers[:count]:
                if not isinstance(item, int) or item <= 0:
                    raise ValueError(
                        "Kimi linear_attn_config.kda_layers must contain "
                        "1-based positive integers"
                    )
                selected.append(item - 1)
            return selected

    fallback = list(fallback_layers)
    if len(fallback) != count:
        raise ValueError("fallback_layers length mismatch")
    return fallback


def build_kda_mappings(
    replace_layers: list[int],
    kimi_layers: list[int],
    source_prefix_template: str,
    target_prefix_template: str,
) -> list[KdaLayerMapping]:
    mappings: list[KdaLayerMapping] = []
    for source_layer, target_layer in zip(kimi_layers, replace_layers, strict=True):
        mappings.append(
            KdaLayerMapping(
                source_layer=source_layer,
                target_layer=target_layer,
                source_prefix=source_prefix_template.format(layer=source_layer),
                target_prefix=target_prefix_template.format(layer=target_layer),
            )
        )
    return mappings


def map_kimi_kda_key(name: str, mappings: Iterable[KdaLayerMapping]) -> str | None:
    for mapping in mappings:
        prefix = f"{mapping.source_prefix}."
        if name.startswith(prefix):
            return f"{mapping.target_prefix}.{name[len(prefix):]}"
    return None


def torch_dtype_from_name(name: str):
    import torch

    normalized = name.lower()
    aliases = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype {name!r}")
    return aliases[normalized]


def dtype_from_config(config: dict[str, Any]):
    torch_dtype = config.get("torch_dtype") or config.get("dtype")
    if isinstance(torch_dtype, str):
        try:
            return torch_dtype_from_name(torch_dtype)
        except ValueError:
            return None
    return None


def tensor_nbytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


class ShardedSafetensorWriter:
    def __init__(self, output_dir: Path, max_shard_size: int):
        self.output_dir = output_dir
        self.max_shard_size = max_shard_size
        self.current: dict[str, Any] = {}
        self.current_size = 0
        self.tmp_files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add_tensor(self, name: str, tensor: Any) -> None:
        if name in self.weight_map or name in self.current:
            raise ValueError(f"Duplicate output tensor name: {name}")
        size = tensor_nbytes(tensor)
        if self.current and self.current_size + size > self.max_shard_size:
            self.flush()
        self.current[name] = tensor.contiguous()
        self.current_size += size
        self.total_size += size

    def flush(self) -> None:
        if not self.current:
            return
        from safetensors.torch import save_file

        shard_id = len(self.tmp_files) + 1
        tmp_name = f"model-{shard_id:05d}.safetensors.tmp"
        tmp_path = self.output_dir / tmp_name
        save_file(self.current, tmp_path, metadata={"format": "pt"})
        for name in self.current:
            self.weight_map[name] = tmp_name
        self.tmp_files.append(tmp_path)
        self.current = {}
        self.current_size = 0

    def close(self) -> dict[str, Any]:
        self.flush()
        shard_count = len(self.tmp_files)
        if shard_count == 0:
            raise ValueError("No tensors were written")

        final_map: dict[str, str] = {}
        for shard_idx, tmp_path in enumerate(self.tmp_files, start=1):
            final_name = f"model-{shard_idx:05d}-of-{shard_count:05d}.safetensors"
            final_path = self.output_dir / final_name
            os.replace(tmp_path, final_path)
            tmp_name = tmp_path.name
            for tensor_name, shard_name in self.weight_map.items():
                if shard_name == tmp_name:
                    final_map[tensor_name] = final_name

        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": dict(sorted(final_map.items())),
        }
        write_json(self.output_dir / "model.safetensors.index.json", index)
        return index


def open_tensor(filename: Path, name: str):
    from safetensors import safe_open

    with safe_open(filename, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def stream_copy_deepseek_tensors(
    deepseek_src: Path,
    writer: ShardedSafetensorWriter,
    replace_layers: set[int],
    replace_scope: str,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for record in iter_safetensor_records(deepseek_src):
        if should_skip_deepseek_key(record.name, replace_layers, replace_scope):
            skipped += 1
            continue
        writer.add_tensor(record.name, open_tensor(record.filename, record.name))
        copied += 1
    return copied, skipped


def stream_copy_kimi_kda_tensors(
    kimi_src: Path,
    writer: ShardedSafetensorWriter,
    mappings: list[KdaLayerMapping],
    allow_missing_kimi_layer: bool,
) -> tuple[int, dict[int, int]]:
    copied = 0
    per_target = {mapping.target_layer: 0 for mapping in mappings}
    for record in iter_safetensor_records(kimi_src):
        target_name = map_kimi_kda_key(record.name, mappings)
        if target_name is None:
            continue
        writer.add_tensor(target_name, open_tensor(record.filename, record.name))
        target_layer = layer_index_from_key(target_name)
        if target_layer is not None and target_layer in per_target:
            per_target[target_layer] += 1
        copied += 1

    missing = [layer for layer, count in per_target.items() if count == 0]
    if missing and not allow_missing_kimi_layer:
        raise ValueError(
            "No Kimi KDA tensors found for target layer(s) "
            f"{missing}. Check --kimi-layers or prefix templates."
        )
    return copied, per_target


def make_projection_weight(
    out_features: int,
    in_features: int,
    init: str,
    dtype: Any,
    seed: int,
    std: float,
):
    import torch

    if init == "identity":
        weight = torch.zeros((out_features, in_features), dtype=torch.float32)
        diag = min(out_features, in_features)
        weight[torch.arange(diag), torch.arange(diag)] = 1.0
    elif init == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        weight = torch.empty((out_features, in_features), dtype=torch.float32)
        weight.normal_(mean=0.0, std=std, generator=generator)
    else:
        raise ValueError(f"Unsupported projection init: {init}")
    return weight.to(dtype=dtype)


def add_projection_tensors(
    writer: ShardedSafetensorWriter,
    replace_layers: Iterable[int],
    deepseek_hidden_size: int,
    kimi_hidden_size: int,
    init: str,
    dtype: Any,
    seed: int,
    std: float,
    in_proj_template: str,
    out_proj_template: str,
) -> list[str]:
    import torch

    written: list[str] = []
    for offset, layer in enumerate(replace_layers):
        layer_seed = seed + offset * 1009
        in_prefix = in_proj_template.format(layer=layer)
        out_prefix = out_proj_template.format(layer=layer)
        tensors = {
            f"{in_prefix}.weight": make_projection_weight(
                kimi_hidden_size,
                deepseek_hidden_size,
                init,
                dtype,
                layer_seed,
                std,
            ),
            f"{in_prefix}.bias": torch.zeros(kimi_hidden_size, dtype=dtype),
            f"{out_prefix}.weight": make_projection_weight(
                deepseek_hidden_size,
                kimi_hidden_size,
                init,
                dtype,
                layer_seed + 1,
                std,
            ),
            f"{out_prefix}.bias": torch.zeros(deepseek_hidden_size, dtype=dtype),
        }
        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor)
            written.append(name)
    return written


def patch_config(
    output_dir: Path,
    deepseek_config: dict[str, Any],
    kimi_config: dict[str, Any],
    replace_layers: list[int],
    kimi_layers: list[int],
    patch_output_config: bool,
) -> dict[str, Any]:
    if not patch_output_config:
        return {}

    config_path = output_dir / "config.json"
    config = dict(deepseek_config)
    config["architectures"] = ["DKDeepseekV4KDAForCausalLM"]
    config["model_type"] = "dk_deepseek_v4_kda"
    config["dk_kda_layers"] = [layer + 1 for layer in replace_layers]
    config["dk_kda_layers_are_1_based"] = True
    config["dk_kda_layer_indices"] = replace_layers
    config["dk_kimi_source_layers"] = kimi_layers
    config["dk_kimi_source_layers_1_based"] = [layer + 1 for layer in kimi_layers]
    config["dk_kimi_linear_config"] = kimi_config

    num_layers = config.get("num_hidden_layers")
    if isinstance(num_layers, int) and num_layers > 0:
        replaced = set(replace_layers)
        config["layers_block_type"] = [
            "linear_attention" if idx in replaced else "attention"
            for idx in range(num_layers)
        ]

    write_json(config_path, config)
    return config


def validate_args(args: argparse.Namespace) -> None:
    if not args.deepseek_src.is_dir():
        raise FileNotFoundError(
            f"--deepseek-src is not a directory: {args.deepseek_src}"
        )
    if not args.kimi_src.is_dir():
        raise FileNotFoundError(f"--kimi-src is not a directory: {args.kimi_src}")
    if not (args.deepseek_src / "config.json").exists():
        raise FileNotFoundError(f"Missing DeepSeek config: {args.deepseek_src}")
    if not (args.kimi_src / "config.json").exists():
        raise FileNotFoundError(f"Missing Kimi config: {args.kimi_src}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"--output-dir already exists and is not empty: {args.output_dir}. "
            "Use --force to write into it."
        )
    if len(set(args.replace_layers)) != len(args.replace_layers):
        raise ValueError("--replace-layers contains duplicates")
    if "{layer}" not in args.kimi_kda_prefix_template:
        raise ValueError("--kimi-kda-prefix-template must contain {layer}")
    if "{layer}" not in args.target_kda_prefix_template:
        raise ValueError("--target-kda-prefix-template must contain {layer}")
    if "{layer}" not in args.dk_in_proj_template:
        raise ValueError("--dk-in-proj-template must contain {layer}")
    if "{layer}" not in args.dk_out_proj_template:
        raise ValueError("--dk-out-proj-template must contain {layer}")
    if not math.isfinite(args.projection_std) or args.projection_std <= 0:
        raise ValueError("--projection-std must be a positive finite number")


def convert_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    ensure_runtime_dependencies()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deepseek_config = read_json(args.deepseek_src / "config.json")
    kimi_config = read_json(args.kimi_src / "config.json")

    side_files = copy_side_files(args.deepseek_src, args.output_dir)
    kimi_layers = select_kimi_layers(
        kimi_config,
        len(args.replace_layers),
        args.replace_layers,
        args.kimi_layers,
    )
    mappings = build_kda_mappings(
        args.replace_layers,
        kimi_layers,
        args.kimi_kda_prefix_template,
        args.target_kda_prefix_template,
    )

    deepseek_stream_hidden_size = infer_hidden_size(
        deepseek_config, args.deepseek_hidden_size, "DeepSeek"
    )
    hc_mult = int(args.hc_mult or deepseek_config.get("hc_mult", 1))
    if hc_mult <= 0:
        raise ValueError("hc_mult must be a positive integer")
    deepseek_flat_hidden_size = deepseek_stream_hidden_size * hc_mult
    kimi_hidden_size = infer_hidden_size(kimi_config, args.kimi_hidden_size, "Kimi")

    dtype = torch_dtype_from_name(args.projection_dtype)
    if dtype is None:
        dtype = dtype_from_config(deepseek_config)
    if dtype is None:
        import torch

        dtype = torch.bfloat16

    writer = ShardedSafetensorWriter(args.output_dir, args.max_shard_size)
    copied_deepseek, skipped_deepseek = stream_copy_deepseek_tensors(
        args.deepseek_src,
        writer,
        set(args.replace_layers),
        args.replace_scope,
    )
    copied_kimi, copied_kimi_by_layer = stream_copy_kimi_kda_tensors(
        args.kimi_src,
        writer,
        mappings,
        args.allow_missing_kimi_layer,
    )
    projection_names = add_projection_tensors(
        writer,
        args.replace_layers,
        deepseek_flat_hidden_size,
        kimi_hidden_size,
        args.projection_init,
        dtype,
        args.seed,
        args.projection_std,
        args.dk_in_proj_template,
        args.dk_out_proj_template,
    )
    weight_index = writer.close()
    patched_config = patch_config(
        args.output_dir,
        deepseek_config,
        kimi_config,
        args.replace_layers,
        kimi_layers,
        args.patch_config,
    )

    report = {
        "deepseek_src": str(args.deepseek_src),
        "kimi_src": str(args.kimi_src),
        "output_dir": str(args.output_dir),
        "replace_scope": args.replace_scope,
        "replace_layers": args.replace_layers,
        "kimi_layers": kimi_layers,
        "kda_mappings": [mapping.__dict__ for mapping in mappings],
        "deepseek_tensors_copied": copied_deepseek,
        "deepseek_tensors_skipped": skipped_deepseek,
        "kimi_kda_tensors_copied": copied_kimi,
        "kimi_kda_tensors_copied_by_layer": copied_kimi_by_layer,
        "projection_tensors": projection_names,
        "projection_init": args.projection_init,
        "projection_dtype": str(dtype).replace("torch.", ""),
        "projection_shapes": {
            "dk_in_proj.weight": [kimi_hidden_size, deepseek_flat_hidden_size],
            "dk_in_proj.bias": [kimi_hidden_size],
            "dk_out_proj.weight": [deepseek_flat_hidden_size, kimi_hidden_size],
            "dk_out_proj.bias": [deepseek_flat_hidden_size],
        },
        "deepseek_stream_hidden_size": deepseek_stream_hidden_size,
        "deepseek_hc_mult": hc_mult,
        "deepseek_flat_hidden_size": deepseek_flat_hidden_size,
        "side_files_copied": side_files,
        "weight_shards": sorted(set(weight_index["weight_map"].values())),
        "config_patched": bool(patched_config),
    }
    write_json(args.output_dir / "conversion_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a DK checkpoint from DeepSeek and Kimi safetensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--deepseek-src", type=Path, required=True)
    parser.add_argument("--kimi-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replace-layers",
        type=parse_int_list,
        default=list(DEFAULT_REPLACE_LAYERS),
        help="0-based DeepSeek layer indices to replace.",
    )
    parser.add_argument(
        "--kimi-layers",
        type=parse_int_list,
        default=[],
        help=(
            "0-based Kimi KDA source layers. If omitted, use Kimi "
            "linear_attn_config.kda_layers, then fall back to replace layers."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-shard-size",
        type=parse_size_to_bytes,
        default=parse_size_to_bytes(DEFAULT_MAX_SHARD_SIZE),
        help="Maximum output safetensors shard size, e.g. 5GB or 500MB.",
    )
    parser.add_argument(
        "--replace-scope",
        choices=("layer", "attention"),
        default="layer",
        help="Whether to skip all tensors in replaced DeepSeek layers or only attn.",
    )
    parser.add_argument(
        "--kimi-kda-prefix-template",
        default="model.layers.{layer}.self_attn",
        help="Source Kimi KDA tensor prefix template.",
    )
    parser.add_argument(
        "--target-kda-prefix-template",
        default="model.layers.{layer}.adapter.kda_layer",
        help="Target DK KDA tensor prefix template.",
    )
    parser.add_argument(
        "--dk-in-proj-template",
        default="model.layers.{layer}.adapter.dk_in_proj",
        help="Target dk_in_proj prefix template.",
    )
    parser.add_argument(
        "--dk-out-proj-template",
        default="model.layers.{layer}.adapter.dk_out_proj",
        help="Target dk_out_proj prefix template.",
    )
    parser.add_argument(
        "--projection-init",
        choices=("identity", "random"),
        default="identity",
        help="Initialization for dk_in_proj/dk_out_proj weights.",
    )
    parser.add_argument(
        "--projection-std",
        type=float,
        default=0.02,
        help="Normal std used when --projection-init=random.",
    )
    parser.add_argument(
        "--projection-dtype",
        default="auto",
        help="Projection dtype: auto, bf16, fp16, or fp32.",
    )
    parser.add_argument("--deepseek-hidden-size", type=int, default=None)
    parser.add_argument(
        "--hc-mult",
        type=int,
        default=None,
        help="DeepSeek V4 carrier count. Defaults to config.hc_mult or 1.",
    )
    parser.add_argument("--kimi-hidden-size", type=int, default=None)
    parser.add_argument(
        "--allow-missing-kimi-layer",
        action="store_true",
        help="Do not fail if a requested Kimi source layer has no matching tensors.",
    )
    parser.add_argument(
        "--no-patch-config",
        dest="patch_config",
        action="store_false",
        help="Leave copied config.json unchanged.",
    )
    parser.set_defaults(patch_config=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = convert_checkpoint(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
