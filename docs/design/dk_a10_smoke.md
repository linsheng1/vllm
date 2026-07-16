# DK A10 冒烟运行手册

本文记录 DK 实验模型在单张 NVIDIA A10 上的低成本冒烟流程。目标是验证
vLLM 能识别 DK 架构、构造混合 KV Cache、加载 tiny 或 reduced real 权重、
启动 OpenAI 兼容服务，并完成一次短请求。

本文不验证模型精度、真实 2M 上下文显存行为或生产性能。完整执行清单、
最新测试结论和 PR 文件组织见 `docs/design/dk_runbook_and_test_report.md`。

## 前置条件

- A10 实例上已经安装当前 vLLM 分支，`vllm` 和 `python` 指向该环境。
- 分支包含 `tools/create_tiny_dk_model_dir.py`。
- PyTorch 可以看到 CUDA。

所有命令默认从仓库根目录执行。除 `<DK_MODEL_DIR>` 等显式占位符外，文档中
的代码路径均为相对路径。

## tiny DK 服务启动

```bash
tools/run_dk_a10_smoke_server.sh
```

脚本默认创建 `.smoke/tiny-dk-a10`，并启动：

```bash
vllm serve .smoke/tiny-dk-a10 \
  --served-model-name tiny-dk \
  --host 127.0.0.1 \
  --port 8000 \
  --load-format dummy \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.60 \
  --enforce-eager
```

常用覆盖项：

```bash
MODEL_DIR=.smoke/tiny-dk-a10 \
HOST=127.0.0.1 \
PORT=8000 \
SERVED_MODEL_NAME=tiny-dk \
tools/run_dk_a10_smoke_server.sh
```

设置 `CREATE_TINY_MODEL=0` 可复用已生成目录。设置
`WRITE_WEIGHTS=1 LOAD_FORMAT=auto` 可生成 synthetic safetensors，并走非
dummy 权重加载路径。

预期日志信号：

- `Resolved architecture: DKDeepseekV4KDAForCausalLM`
- `Using DeepSeek's fp8_ds_mla KV cache format`
- 日志报告 GPU KV cache capacity

## completion 检查

另开一个 shell：

```bash
tools/run_dk_a10_completion_check.sh
```

默认请求：

```json
{
  "model": "tiny-dk",
  "prompt": "def add(a, b):",
  "max_tokens": 16,
  "temperature": 0
}
```

首次请求可能触发 KDA/Mamba Triton JIT 或 autotune，导致客户端超时。等服务
日志稳定后重试即可。

## synthetic safetensors 构造

```bash
python tools/create_tiny_dk_model_dir.py \
  --output-dir .smoke/tiny-dk-a10-safetensors \
  --force \
  --write-weights \
  --torch-dtype bfloat16
```

该命令生成：

- `model.safetensors`
- `model.safetensors.index.json`
- `tiny_dk_weights_manifest.json`

这些权重只保证 shape 兼容，不用于质量评估。

## sharded_state 精确冒烟

从 dummy-loaded DK 模型导出 vLLM 原生 `sharded_state`：

```bash
tools/export_dk_a10_sharded_state.sh
```

再从导出的 checkpoint 启动：

```bash
MODEL_DIR=.smoke/tiny-dk-a10-sharded-exact \
LOAD_FORMAT=sharded_state \
CREATE_TINY_MODEL=0 \
tools/run_dk_a10_smoke_server.sh
```

A10 上已验证该路径能返回 16 token completion，并报告
`DKDeepseekV4KDAForCausalLM` 架构与 `fp8_ds_mla` KV cache 格式。

## reduced runtime

为降低 A10 成本，保留原始 config，但通过环境变量只实例化、加载和执行选中
层：

```bash
VLLM_DK_REDUCED_LAYER_INDICES=auto \
MODEL_DIR=<DK_MODEL_DIR> \
LOAD_FORMAT=auto \
CREATE_TINY_MODEL=0 \
tools/run_dk_a10_smoke_server.sh
```

`auto` 选择策略：

- 默认选第一个 KDA 替换层；
- 保留第一层和最后一层；
- 保留该 KDA 层前后相邻 DeepSeek 层；
- 如果已选 DeepSeek 层没有 C4A，则额外加入一个 C4A 层；
- 如果已选 DeepSeek 层没有 C128A，则额外加入一个 C128A 层。

也可以显式指定 0-based 原始层号：

```bash
VLLM_DK_REDUCED_LAYER_INDICES=0,9,10,11,31 \
tools/run_dk_a10_smoke_server.sh
```

启用该环境变量后：

- 未选中 decoder 层会变成 no-op placeholder；
- 未选中层不分配 attention 或 linear-attention cache/state；
- 加载权重时跳过未选中 `model.layers.N.` tensor；
- forward 保留原始层编号，只执行选中层；
- 选中 KDA 层的 `adapter.dk_in_proj`、`adapter.kda_layer`、
  `adapter.dk_out_proj` 仍然真实执行。

这是 debug/smoke 模式，不代表完整 DK 模型质量。

## reduced real 权重切片

如果实例上已有 DeepSeek 和 Kimi 源 checkpoint：

```bash
DEEPSEEK_SRC=<DEEPSEEK_V4_FLASH_DIR> \
KIMI_SRC=<KIMI_LINEAR_DIR> \
OUTPUT_DIR=.smoke/dk-reduced-real-slice \
tools/build_dk_reduced_real_slice.sh
```

该脚本调用 `tools/convert_dk_checkpoint.py --reduced-layer-indices auto`，
输出：

- DeepSeek 顶层 tensor，例如 embedding、final norm、LM head、MHC head；
- auto slice 选中的 DeepSeek decoder 层；
- 一个 Kimi KDA source layer，映射到 DK 目标 KDA 层；
- `dk_in_proj` 和 `dk_out_proj`，默认使用 identity-like 初始化。

ModelScope 低成本准备路径见
`docs/design/dk_runbook_and_test_report.md`。

## A10 上的真实切片结论

ModelScope `deepseek-ai/DeepSeek-V4-Flash` config 已确认是 43 层，不是
43 层以下的 tiny 假设。题干中的 1-based 替换层 `11,21,31` 在运行时对应
0-based `10,20,30`。

真实 reduced slice 选择：

- DeepSeek 层：`0,9,11,42`
- DK/KDA 替换层：`10`
- Kimi KDA source layer：`0`
- converted slice：`.smoke/dk-reduced-real-slice-ms`
- 大小约 `19G`，共 `11` 个 shard

## A10 debug 开关取舍

A10 上部分 DeepSeek-V4-Flash kernel 不适配或成本过高，因此 reduced real
运行使用：

```bash
VLLM_DK_REDUCED_LAYER_INDICES=auto \
VLLM_DK_FAKE_ROUTED_EXPERTS=1 \
VLLM_DK_FP8_BLOCK_TORCH_FALLBACK=1 \
VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1 \
VLLM_DK_TORCH_KV_CACHE_INSERT=1 \
CREATE_TINY_MODEL=0 \
MODEL_DIR=.smoke/dk-reduced-real-slice-ms \
SERVED_MODEL_NAME=dk-real-ms \
LOAD_FORMAT=auto \
MAX_MODEL_LEN=128 \
GPU_MEMORY_UTILIZATION=0.70 \
CPU_OFFLOAD_GB=12 \
DTYPE=bfloat16 \
KV_CACHE_DTYPE=fp8 \
EXTRA_SERVE_ARGS="--linear-backend triton" \
tools/run_dk_a10_smoke_server.sh
```

这些开关保留：

- 真实 DeepSeek 顶层权重；
- 真实 selected DeepSeek 层权重加载；
- 真实 Kimi KDA 权重加载；
- 真实 `dk_in_proj` / `dk_out_proj`；
- 真实 KDA/Mamba state 路径；
- DeepSeek attention KV cache 写入。

这些开关绕过：

- routed expert 的真实 MoE 计算；
- DeepSeek attention 的完整生产 kernel 输出；
- A10 不适配的高性能 FP8/indexer kernel。

## 真实 KDA/Mamba 状态冒烟

当前 reduced A10 smoke 不再需要 `VLLM_DK_FAKE_KDA`。vLLM 的混合 KV
allocator 已扩展为保留 DK/KDA `MambaSpec` group，并把 DeepSeek MLA/SWA 与
KDA/Mamba page size 放入统一分组。

已观察到：

- real KDA tensor 被加载；
- `kda_gate_fwd_kernel` JIT；
- `_causal_conv1d_update_kernel` JIT；
- `fused_recurrent_gated_delta_rule_fwd_kernel` JIT；
- `/v1/completions` HTTP 200；
- 服务停止后 GPU 显存回到 0。

## KV Cache 写入路径

`VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1` 不是纯跳过 attention。该路径仍执行
attention input projection、Q/KV norm、RoPE 准备，然后通过
`VLLM_DK_TORCH_KV_CACHE_INSERT=1` 的 torch fallback 写入 KV cache，最后把
attention 输出置零。

覆盖的 cache 写入包括：

- SWA 512D `fp8_ds_mla` cache；
- C4A/C128A compressed 512D `fp8_ds_mla` cache；
- C4A indexer 128D FP8 cache 和 FP32 scale bytes；
- compressed K 构造所需 compressor state cache。
