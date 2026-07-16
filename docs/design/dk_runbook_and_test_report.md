# DK 运行教程与测试报告

本文是 DK 实验支持的执行手册。所有路径默认相对 vLLM 仓库根目录，除非使用
`<DK_REDUCED_MODEL_DIR>` 这类显式占位符。

## 范围

DK 被建模为 DeepSeek-V4-Flash 变种：0-based decoder 层 `10,20,30` 被
Kimi Linear KDA 层替换。实现保持原始完整 config；A10 低成本路径通过
`VLLM_DK_REDUCED_LAYER_INDICES` 只加载和执行选中子集。

当前验证目标是结构跑通，不是模型质量：

- 加载 DK config 并解析为 `DKDeepseekV4KDAForCausalLM`；
- 构造 DeepSeek MLA/SWA 与 KDA/Mamba 混合 cache spec；
- 在 reduced real 权重切片上运行真实 KDA/Mamba state；
- 通过 vLLM metrics 验证 local prefix cache 和 eviction pressure；
- 保留 native CPU KV offload 接线，但 strict offload 验证仍未闭环。

## 准备权重源

在 GPU 环境安装 ModelScope：

```bash
pip install modelscope
python tools/prepare_dk_modelscope_sources.py \
  --output-dir .smoke/modelscope-dk-sources \
  --reduced-layer-indices auto \
  --max-download-bytes 50GB \
  --force
```

该脚本只下载 reduced DK slice 需要的 side files 和 safetensor shards。
报告写入：

```text
.smoke/modelscope-dk-sources/dk_modelscope_prepare_report.json
```

构造 reduced real-weight DK checkpoint：

```bash
DEEPSEEK_SRC=.smoke/modelscope-dk-sources/deepseek \
KIMI_SRC=.smoke/modelscope-dk-sources/kimi \
OUTPUT_DIR=.smoke/dk-reduced-real-slice-ms \
REDUCED_LAYER_INDICES=auto \
REPLACE_LAYERS=10,20,30 \
KIMI_LAYERS=0 \
PROJECTION_INIT=identity \
PROJECTION_DTYPE=auto \
tools/build_dk_reduced_real_slice.sh
```

A10 上观察到的 reduced slice：完整 config 仍是 43 层，但实际只加载
DeepSeek 层 `0,9,11,42` 加 DK/KDA 层 `10`。转换后约 `19G`，共 `11` 个
shard。

## 启动服务

tiny synthetic smoke：

```bash
MODEL_DIR=.smoke/tiny-dk-a10 \
SERVED_MODEL_NAME=tiny-dk \
MAX_MODEL_LEN=512 \
KV_CACHE_DTYPE=fp8 \
tools/run_dk_a10_smoke_server.sh
```

reduced real KDA smoke：

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

completion 检查：

```bash
SERVED_MODEL_NAME=dk-real-ms \
MAX_TOKENS=4 \
tools/run_dk_a10_completion_check.sh
```

该路径故意不设置 `VLLM_DK_FAKE_KDA`。A10 日志已确认真实 KDA/Mamba 执行，
包括 `kda_gate_fwd_kernel`、`_causal_conv1d_update_kernel` 和
`fused_recurrent_gated_delta_rule_fwd_kernel`。

## Prefix Cache 与 Eviction 测试

DK hybrid cache 需要足够长的 prompt 才能看到 hit。已测配置下 prefix hit
粒度是 `4608` tokens，因此 `MAX_MODEL_LEN=512` 不能证明 prefix reuse。

启动服务：

```bash
VLLM_LOG_STATS_INTERVAL=2 \
VLLM_DK_REDUCED_LAYER_INDICES=auto \
VLLM_DK_FAKE_ROUTED_EXPERTS=1 \
VLLM_DK_FP8_BLOCK_TORCH_FALLBACK=1 \
VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1 \
VLLM_DK_TORCH_KV_CACHE_INSERT=1 \
CREATE_TINY_MODEL=0 \
MODEL_DIR=.smoke/dk-reduced-real-slice-ms \
SERVED_MODEL_NAME=dk-real-ms \
LOAD_FORMAT=auto \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
CPU_OFFLOAD_GB=16 \
DTYPE=bfloat16 \
KV_CACHE_DTYPE=fp8 \
ENABLE_PREFIX_CACHING=1 \
MAMBA_CACHE_MODE=all \
NUM_GPU_BLOCKS_OVERRIDE=1024 \
EXTRA_SERVE_ARGS="--linear-backend triton" \
tools/run_dk_a10_smoke_server.sh
```

运行 HTTP 与 metrics 检查：

```bash
python tools/run_dk_a10_kv_behavior_check.py \
  --model dk-real-ms \
  --prefix-lines 420 \
  --eviction-requests 10 \
  --eviction-lines 420 \
  --max-tokens 4 \
  --metric-settle-s 1 \
  --timeout-s 240 \
  --strict-prefix-hit \
  --output .smoke/dk-kv-prefix-report.json
```

已观察结果：

- cold、warm、alternate suffix、post-eviction completion 均返回 `4` tokens；
- same-prefix reuse 后 `prefix_cache_hits` 增加 `4608`；
- cross-request shared-prefix hit 也增加 `4608`；
- eviction pressure 额外发出 `10` 个长 prompt，共 `75660` prompt tokens；
- final prefix metrics 包含 `vllm:prefix_cache_hits=9216` 和
  `vllm:prefix_cache_queries=100885`；
- `MAX_MODEL_LEN=8192` 下 GPU KV cache capacity 为 `71,089` tokens，最大并发
  约 `8.68x`。

## Native KV Offload 测试

native offload 路径已接入，但 DK strict pass 仍未完成。

启动时加入：

```bash
VLLM_USE_SIMPLE_KV_OFFLOAD=1 \
KV_OFFLOADING_SIZE=4 \
KV_OFFLOADING_BACKEND=native \
VLLM_DK_REDUCED_LAYER_INDICES=auto \
VLLM_DK_FAKE_ROUTED_EXPERTS=1 \
VLLM_DK_FP8_BLOCK_TORCH_FALLBACK=1 \
VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1 \
VLLM_DK_TORCH_KV_CACHE_INSERT=1 \
CREATE_TINY_MODEL=0 \
MODEL_DIR=.smoke/dk-reduced-real-slice-ms \
SERVED_MODEL_NAME=dk-real-ms \
LOAD_FORMAT=auto \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
CPU_OFFLOAD_GB=16 \
DTYPE=bfloat16 \
KV_CACHE_DTYPE=fp8 \
ENABLE_PREFIX_CACHING=1 \
MAMBA_CACHE_MODE=all \
NUM_GPU_BLOCKS_OVERRIDE=1024 \
EXTRA_SERVE_ARGS="--linear-backend triton" \
tools/run_dk_a10_smoke_server.sh
```

strict 检查：

```bash
python tools/run_dk_a10_kv_behavior_check.py \
  --model dk-real-ms \
  --prefix-lines 420 \
  --eviction-requests 10 \
  --eviction-lines 420 \
  --max-tokens 4 \
  --metric-settle-s 1 \
  --timeout-s 240 \
  --strict-prefix-hit \
  --strict-offload \
  --output .smoke/dk-kv-offload-report.json
```

当前状态：

- vLLM 已选择 `SimpleCPUOffloadConnector`；
- worker-side CPU KV blocks 已分配；
- scheduler-side CPU KV blocks 已分配；
- external prefix query metrics 出现；
- strict offload 尚未通过，因为 external hit/offloaded bytes 未在完整报告中确认。

已完成的修复：

- scheduler 的 Mamba aligned split 可以处理 external KV hits；
- `SimpleCPUOffloadWorker` 在 metadata cleanup 后仍保留 load-event 到 req-id 的映射；
- hybrid group lookup 能识别 DK/KDA Mamba spec。

## 本地检查

无需 GPU：

```bash
python3 -m py_compile \
  tools/run_dk_a10_kv_behavior_check.py \
  vllm/v1/worker/mamba_utils.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/simple_kv_offload/worker.py \
  tests/models/test_dk_deepseek_v4_kda.py

bash -n tools/run_dk_a10_smoke_server.sh
bash -n tools/build_dk_reduced_real_slice.sh
bash -n tools/export_dk_a10_sharded_state.sh
bash -n tools/run_dk_a10_completion_check.sh

git diff --check
```

新增 pytest 覆盖：

- `tests/models/test_dk_deepseek_v4_kda.py`
- `tests/models/test_dk_deepseek_v4_kv_cache_insert.py`
- `tests/tools/test_create_tiny_dk_model_dir.py`
- `tests/tools/test_dk_modelscope_prepare.py`

在完整 vLLM dev 环境中运行：

```bash
pytest \
  tests/models/test_dk_deepseek_v4_kda.py \
  tests/models/test_dk_deepseek_v4_kv_cache_insert.py \
  tests/tools/test_create_tiny_dk_model_dir.py \
  tests/tools/test_dk_modelscope_prepare.py
```

## PR 整理

建议分支：

```text
dk-v4-flash-kda
```

建议 PR 标题：

```text
Add experimental DK DeepSeek-V4 KDA support
```

目标仓库：

```text
linsheng1/vllm
```

建议 PR 描述：

```markdown
## 摘要

- 增加实验性的 DK DeepSeek-V4-Flash + Kimi Linear KDA 架构支持。
- 增加 reduced-layer 执行和 checkpoint 转换工具，用于低成本 A10 验证。
- 扩展 hybrid KV cache，使 DeepSeek MLA/SWA group 与 DK KDA/Mamba state group 可以共存。
- 增加 smoke 脚本、ModelScope 准备工具和结构性 KV-cache 验证报告。

## 验证

- DK smoke 工具和修改过的 KV/offload 模块通过 `python3 -m py_compile`。
- DK A10 shell 脚本通过 `bash -n`。
- `git diff --check`.
- A10 reduced real-weight KDA smoke 使用真实 KDA/Mamba kernel 完成。
- A10 prefix-cache 与 eviction-pressure smoke 在 `max_model_len=8192` 下通过。

## 已知限制

- 尚未验证完整 43 层真实 DK 执行。
- 尚未验证 2M 上下文显存行为。
- A10 路径绕过完整 DeepSeek attention/indexer 和 routed expert 计算。
- Native CPU KV offload 已接入，但还没有 strict 通过报告。
```

PR 应包含：

- model/config：
  `vllm/models/deepseek_v4/nvidia/kda_model.py`,
  `vllm/models/deepseek_v4/nvidia/model.py`,
  `vllm/transformers_utils/configs/dk_deepseek_v4_kda_utils.py`
- DK KV/cache：
  `vllm/v1/core/kv_cache_utils.py`,
  `vllm/v1/worker/mamba_utils.py`,
  `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/simple_kv_offload/worker.py`,
  `vllm/v1/worker/gpu_model_runner.py`,
  `vllm/v1/attention/backends/mla/indexer.py`
- A10 fallback/debug：
  `vllm/models/deepseek_v4/compressor.py`,
  `vllm/models/deepseek_v4/nvidia/ops/attention.py`,
  `vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py`
- tools：
  `tools/create_tiny_dk_model_dir.py`,
  `tools/convert_dk_checkpoint.py`,
  `tools/prepare_dk_modelscope_sources.py`,
  `tools/build_dk_reduced_real_slice.sh`,
  `tools/export_dk_a10_sharded_state.sh`,
  `tools/run_dk_a10_smoke_server.sh`,
  `tools/run_dk_a10_completion_check.sh`,
  `tools/run_dk_a10_kv_behavior_check.py`
- tests/docs：
  `tests/models/test_dk_deepseek_v4_kda.py`,
  `tests/models/test_dk_deepseek_v4_kv_cache_insert.py`,
  `tests/tools/test_create_tiny_dk_model_dir.py`,
  `tests/tools/test_dk_modelscope_prepare.py`,
  `docs/design/dk_deepseek_v4_kda_design.md`,
  `docs/design/dk_a10_smoke.md`,
  `docs/design/dk_kv_cache_validation_report.md`,
  `docs/design/dk_runbook_and_test_report.md`

不要加入：

- `graphify-out/**`
- `**/__pycache__/**`
- `.smoke/**`
- 下载的 DeepSeek/Kimi checkpoint
- 本地 A10 原始日志，除非抽取短片段写入文档

建议提交命令：

```bash
git add \
  docs/design/dk_deepseek_v4_kda_design.md \
  docs/design/dk_a10_smoke.md \
  docs/design/dk_kv_cache_validation_report.md \
  docs/design/dk_runbook_and_test_report.md \
  tests/models/test_dk_deepseek_v4_kda.py \
  tests/models/test_dk_deepseek_v4_kv_cache_insert.py \
  tests/tools/test_create_tiny_dk_model_dir.py \
  tests/tools/test_dk_modelscope_prepare.py \
  tools/build_dk_reduced_real_slice.sh \
  tools/convert_dk_checkpoint.py \
  tools/create_tiny_dk_model_dir.py \
  tools/export_dk_a10_sharded_state.sh \
  tools/prepare_dk_modelscope_sources.py \
  tools/run_dk_a10_completion_check.sh \
  tools/run_dk_a10_kv_behavior_check.py \
  tools/run_dk_a10_smoke_server.sh \
  vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py \
  vllm/models/deepseek_v4/compressor.py \
  vllm/models/deepseek_v4/nvidia/kda_model.py \
  vllm/models/deepseek_v4/nvidia/model.py \
  vllm/models/deepseek_v4/nvidia/ops/attention.py \
  vllm/transformers_utils/configs/dk_deepseek_v4_kda_utils.py \
  vllm/v1/attention/backends/mla/indexer.py \
  vllm/v1/core/kv_cache_utils.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/simple_kv_offload/worker.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/worker/mamba_utils.py

git commit -m "Add experimental DK DeepSeek-V4 KDA support"
git push linsheng1 dk-v4-flash-kda
```

## 剩余差距

- 尚未验证完整 43 层真实 DK。
- 真实 2M 上下文显存行为不在 A10 smoke 范围内。
- A10 路径绕过完整 DeepSeek attention/indexer 和 routed expert 计算。
- Native CPU KV offload/paging 已接入但没有 strict end-to-end 通过报告。
- 模型质量未评估；reduced run 只验证加载、forward 和 cache 行为。
