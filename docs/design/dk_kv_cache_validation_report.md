# DK KV Cache 验证报告

日期：2026-07-16

本文记录当前 DK 在单机单卡 A10 上的 KV Cache 验证状态。它不是精度报告，
也不验证真实 2M 上下文显存行为。可复现命令和 PR 文件组织见
`docs/design/dk_runbook_and_test_report.md`。

## 验证范围

验证对象是一个单进程 vLLM server，运行 reduced real-weight DK slice。

- 模型目录占位符：`<DK_REDUCED_MODEL_DIR>`
- served model name：`dk-real-ms`
- 完整 DeepSeek-V4-Flash config 保持 43 个 decoder 层
- reduced runtime slice：DeepSeek 层 `0,9,11,42`，以及 DK/KDA 层 `10`
- Kimi source KDA layer：`0`
- 主要验证长度：`max_model_len=8192`
- KV cache dtype：`fp8`
- prefix caching：开启

A10 上使用以下 debug 开关绕过不适配或成本过高的 kernel：

```bash
VLLM_DK_REDUCED_LAYER_INDICES=auto
VLLM_DK_FAKE_ROUTED_EXPERTS=1
VLLM_DK_FP8_BLOCK_TORCH_FALLBACK=1
VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1
VLLM_DK_TORCH_KV_CACHE_INSERT=1
```

这些开关保留真实 reduced 权重、真实 KDA state、projection 路径和 KV cache
写入；但绕过完整 routed expert 计算和完整 DeepSeek attention 生产 kernel。

## 实现状态

### 真实 KDA/Mamba 状态

状态：A10 通过。

DK KDA reduced smoke 不再需要 `VLLM_DK_FAKE_KDA`。vLLM 会通过 hybrid KV
cache 路径为 KDA `MambaSpec` 分配 state page。

已观察到：

- real KDA tensor 被加载；
- `kda_gate_fwd_kernel` 在推理中 JIT；
- `_causal_conv1d_update_kernel` 在推理中 JIT；
- `fused_recurrent_gated_delta_rule_fwd_kernel` 在推理中 JIT；
- `/v1/completions` 返回 HTTP 200；
- 服务关闭后 GPU 显存回到 0。

### DeepSeek Attention KV 写入

状态：A10 functional debug path 通过。

A10 smoke 仍使用 `VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1`。该路径执行 attention
input projection、Q/KV normalization 和 RoPE 准备，然后通过慢速 torch
fallback 写入 KV cache，最后将 attention 输出置零。

覆盖的 cache 写入：

- SWA 512D `fp8_ds_mla` cache；
- C4A/C128A compressed 512D `fp8_ds_mla` cache；
- C4A indexer 128D FP8 cache 与 FP32 scale bytes；
- compressed K 构造所需 compressor state cache。

SWA、512D compressed、128D indexer cache 写入已有 unit-level torch fallback
覆盖。

### Hybrid KV 分组

状态：已实现，并在 A10 启动中被实际触发。

DeepSeek-V4 KV allocator 现在会在 grouped allocation 路径中保留 DK/KDA
Mamba groups。canonical page-size bucket 使用 MLA、SWA MLA 和 Mamba state
page size 的并集，因此 KDA Mamba state page 可以和 DeepSeek MLA/SWA group
一起被分配。

相关覆盖：

- `tests/models/test_dk_deepseek_v4_kda.py::test_deepseek_v4_kv_groups_keep_kda_mamba_spec`
- `tests/models/test_dk_deepseek_v4_kda.py::test_deepseek_v4_hybrid_hash_block_size_uses_gcd_for_mamba`
- `tests/models/test_dk_deepseek_v4_kda.py::test_mamba_buffers_find_dk_uniform_kda_group`

## Prefix Cache 与 Eviction 压力

状态：A10 通过。

服务启动形态：

```bash
VLLM_LOG_STATS_INTERVAL=2 \
VLLM_DK_REDUCED_LAYER_INDICES=auto \
VLLM_DK_FAKE_ROUTED_EXPERTS=1 \
VLLM_DK_FP8_BLOCK_TORCH_FALLBACK=1 \
VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1 \
VLLM_DK_TORCH_KV_CACHE_INSERT=1 \
CREATE_TINY_MODEL=0 \
MODEL_DIR=<DK_REDUCED_MODEL_DIR> \
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

检查命令：

```bash
python tools/run_dk_a10_kv_behavior_check.py \
  --model dk-real-ms \
  --prefix-lines 420 \
  --eviction-requests 10 \
  --eviction-lines 420 \
  --max-tokens 4 \
  --metric-settle-s 1 \
  --timeout-s 240 \
  --output .smoke/dk-kv-prefix-report.json
```

结果摘要：

- cold prompt tokens：`6304`
- warm same-prefix prompt tokens：`6307`
- cold completion tokens：`4`
- warm same-prefix completion tokens：`4`
- warm different-suffix completion tokens：`4`
- post-eviction completion tokens：`4`
- same-prefix hit delta：`4608`
- different-suffix shared-prefix hit delta：`4608`
- eviction fill 期间 prefix query delta：`75660`
- final `vllm:prefix_cache_hits`：`9216`
- final `vllm:prefix_cache_queries`：`100885`

解释：

- 当前 DK hybrid prefix cache hit 粒度是 `4608` tokens，因为 MLA、SWA 和
  KDA/Mamba group 必须按 hybrid LCM 对齐。
- `max_model_len=512` 太短，不能证明该 DK 分组下的 prefix hit；
  实际通过场景使用 `max_model_len=8192`。
- Eviction 压力通过多次长 unique prompt 触发。HTTP smoke 通过 completion 和
  Prometheus counter 验证行为；精确 LRU victim 身份留给 vLLM unit test。

## Native CPU KV Offload / Paging

状态：部分触发，尚未 strict pass。

服务形态是在 prefix-cache 场景上额外加入：

```bash
VLLM_USE_SIMPLE_KV_OFFLOAD=1 \
KV_OFFLOADING_SIZE=4 \
KV_OFFLOADING_BACKEND=native
```

已观察到：

- 选中了 `SimpleCPUOffloadConnector`；
- worker 为 `4.00 GB` 分配了 `1459` 个 CPU block；
- scheduler 为 `4.00 GB` 分配了 `750` 个 CPU block；
- local prefix cache 仍然有 hit；
- `vllm:external_prefix_cache_queries` 增长到 `85362`。

已修复的问题：

- scheduler 原有 `External KV connector is not verified yet` 断言会阻断 DK
  Mamba aligned split；
- `SimpleCPUOffloadWorker` 在 per-step metadata 清理后会丢失
  `load_event -> req_id` 映射。

剩余问题：

- strict offload smoke 尚未产生一个完成的 external prefix hit 报告；
- 当前证据证明 offload control-plane 创建、store/query 活动和 scheduler/worker
  修复，但尚不能证明 DK 的完整 CPU KV reload 命中闭环。

相关覆盖：

- `tests/models/test_dk_deepseek_v4_kda.py::test_mamba_aligned_split_accepts_external_kv_hits`
- `tests/models/test_dk_deepseek_v4_kda.py::test_simple_cpu_offload_worker_keeps_load_event_request_mapping`

## 剩余风险

- 尚未验证完整 43 层 DK 加全部真实 DeepSeek attention 和 routed expert 计算。
- A10 验证有意绕过部分生产 kernel。
- Native CPU KV offload/paging 仍需要 strict end-to-end 通过报告。
- 真实 2M 上下文是策略目标，不是本 A10 smoke 的验证目标。
- Mamba common-prefix/cascade attention 仍不适用：
  `MambaManager.get_num_common_prefix_blocks()` 按设计返回 `0`。
