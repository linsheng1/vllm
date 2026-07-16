# DK DeepSeek-V4 KDA 支持设计文档

## 背景

题干假设存在一个 `deepseek-ai/DeepSeek-V4-Flash` 变种模型 DK：

- 原始 DeepSeek-V4-Flash config 保持不变；
- 1-based 第 `11,21,31` 层被替换为
  `moonshotai/Kimi-Linear-48B-A3B-Instruct` 的 KDA layer；
- 运行时使用 0-based 层号 `10,20,30`；
- DeepSeek hidden states 进入 KDA 前后通过额外 projection 对齐维度；
- DK 目标是 agentic coding 模型，设计上支持最长 2M 上下文。

本实现只支持 vLLM，不做 vLLM-Ascend，不做 NPU custom op，不做模型精度优化，
不做训练或蒸馏流程。当前目标是让模型结构、权重加载、forward 和 KV Cache
路径可跑通。

## Graphify 后的代码落点

基于 Graphify 对 DeepSeek-V4、Mamba/KDA、vLLM V1 KV 管理路径的分析，修改点
集中在以下模块：

- 模型注册与 DK wrapper：
  `vllm/models/deepseek_v4/nvidia/kda_model.py`
- DeepSeek-V4 原模型集成点：
  `vllm/models/deepseek_v4/nvidia/model.py`
- DK config 归一化：
  `vllm/transformers_utils/configs/dk_deepseek_v4_kda_utils.py`
- Hybrid KV 分组与 hash block size：
  `vllm/v1/core/kv_cache_utils.py`
- Mamba/KDA state group 发现：
  `vllm/v1/worker/mamba_utils.py`
- scheduler 与 external KV/offload：
  `vllm/v1/core/sched/scheduler.py`
- simple CPU offload worker：
  `vllm/v1/simple_kv_offload/worker.py`
- A10 debug KV 写入与 fallback：
  `vllm/models/deepseek_v4/nvidia/ops/attention.py`,
  `vllm/models/deepseek_v4/compressor.py`,
  `vllm/v1/attention/backends/mla/indexer.py`

## 模型结构设计

DK 新增 `DKDeepseekV4KDAForCausalLM`，继承 DeepSeek-V4 CausalLM 的外部接口，
但 decoder layer class 替换为 DK-aware layer。

普通 DeepSeek 层：

- 未被替换的 selected DeepSeek 层继续走 DeepSeek-V4 原 layer；
- reduced runtime 未选中的层替换为 no-op placeholder；
- placeholder 保留原始层编号，但不分配参数和 cache state。

KDA 替换层：

- `DKDeepseekV4KDADecoderLayer` 接收 DeepSeek 多 carrier hidden states；
- `DKKimiKDALayerAdapter` 先执行 `dk_in_proj`，把 DeepSeek hidden size 投影到
  Kimi KDA hidden size；
- 调用 Kimi `KimiLinearDecoderLayer` 中的 KDA/linear-attention 结构；
- 再执行 `dk_out_proj` 回到 DeepSeek hidden size；
- 层号仍使用 DeepSeek 原始层号，因此外部 config 与权重 key 能保持稳定。

当前 tensor parallel 限制：

- KDA adapter 当前只支持 `tensor_parallel_size=1`；
- 这是为了先保证单机单卡结构跑通，避免 Kimi KDA 与 DeepSeek MHC/MoE 在
  TP 切分上的额外复杂度。

## 权重构造设计

本项目不训练、不蒸馏。为了跑通结构，权重构造分三档。

### tiny synthetic

`tools/create_tiny_dk_model_dir.py` 构造 tiny DK config，并可生成 shape-compatible
随机/simulated safetensors。

用途：

- 本地和 A10 快速验证架构注册；
- 验证 loader 能看到 DK 权重 key；
- 验证 reduced runtime 和 dummy/auto load format。

限制：

- 权重无语义，不评估质量。

### reduced real slice

`tools/prepare_dk_modelscope_sources.py` 只下载 reduced slice 需要的 ModelScope
文件和 shard。`tools/build_dk_reduced_real_slice.sh` 调用
`tools/convert_dk_checkpoint.py` 构造 DK checkpoint。

真实权重来源：

- DeepSeek 顶层 tensor、embedding、norm、LM head、MHC head 来自
  `deepseek-ai/DeepSeek-V4-Flash`；
- selected DeepSeek decoder 层来自 DeepSeek 原始权重；
- selected KDA layer 来自 `moonshotai/Kimi-Linear-48B-A3B-Instruct`；
- `dk_in_proj` 和 `dk_out_proj` 可以 identity-like、random 或指定 dtype
  初始化。

当前 A10 verified slice：

- DeepSeek 原始 config：43 层；
- selected DeepSeek 层：`0,9,11,42`；
- active DK/KDA 层：`10`；
- Kimi source KDA 层：`0`；
- checkpoint 大小约 `19G`，共 `11` 个 shard。

### full DK checkpoint

如果用户提供完整 DK 权重，预期目录结构应满足：

- config 标记 DK 架构和替换层；
- DeepSeek 未替换层 tensor 完整存在；
- `model.layers.10/20/30.adapter.kda_layer.*` 对应 Kimi KDA tensor；
- `model.layers.10/20/30.adapter.dk_in_proj.*` 与
  `model.layers.10/20/30.adapter.dk_out_proj.*` 存在；
- tokenizer、generation config、safetensors index 完整。

在设备 kernel 可用、显存/内存足够、并且不开启 reduced runtime 的前提下，当前
代码路径设计上可以加载完整 DK 目录并进入 forward。但完整 43 层真实 DK 尚未被
实测验证。

## Reduced Runtime 设计

环境变量 `VLLM_DK_REDUCED_LAYER_INDICES` 用于低成本 debug。它不修改 config，
只改变运行时实例化、加载和 forward。

`auto` 策略：

- 选第一个 KDA 替换层；
- 保留第一层、最后一层；
- 保留 KDA 前后相邻 DeepSeek 层；
- 如果没有 C4A，则补一个 C4A；
- 如果没有 C128A，则补一个 C128A。

作用：

- 保持原 config 层数和编号；
- 未选中层不加载权重、不分配 KV/state、不执行计算；
- 能用 A10 跑真实 KDA 和关键 DeepSeek KV cache 写入路径。

## Global KV Cache 策略

DK 同时包含 DeepSeek MLA/SWA cache 和 KDA/Mamba state，因此不能只按传统
attention KV block 处理。策略分为四层。

### 1. Local Hybrid KV Cache

本地 GPU cache 由 vLLM V1 KV manager 统一管理：

- DeepSeek MLA/SWA 使用 `fp8_ds_mla` cache；
- KDA/Mamba 使用 `MambaSpec` state page；
- hybrid group allocation 保留 MLA、SWA、Mamba 三类 spec；
- page size bucket 使用三类 spec 的并集；
- hash block size 使用可兼容所有 group 的 GCD/LCM 对齐策略。

当前实测中，DK prefix hit 粒度为 `4608` tokens。这意味着短上下文无法观察到
prefix hit；验证时需要 `max_model_len=8192` 级别。

### 2. Prefix Cache

prefix cache 使用 vLLM 现有 prefix hashing 机制，但必须保证：

- DeepSeek MLA/SWA block hash 与 KDA/Mamba state group 对齐；
- selected reduced 层不会污染 full config 的层号语义；
- cache hit 只在所有相关 group 都满足对齐后才成立。

A10 已通过：

- same-prefix reuse hit；
- cross-request shared-prefix hit；
- eviction pressure 后 completion 仍成功。

### 3. KV Eviction

eviction 继续使用 vLLM block manager 的策略。DK 侧新增要求是：

- eviction 不能只释放 attention KV，而漏掉 KDA/Mamba state；
- reduced skip layer 不应出现在 eviction group 中；
- metrics 需要能反映 prefix query/hit 变化。

当前 HTTP smoke 已触发 eviction pressure，但精确 victim 身份仍依赖 vLLM
unit test，而不是 A10 HTTP 测试。

### 4. Offload / Paging

设计上使用 vLLM native/simple KV offload：

- GPU cache 压力大时把 block page 下沉到 CPU；
- external prefix cache metrics 用于观察 query/hit；
- scheduler 的 Mamba aligned split 需要接受 external computed tokens；
- worker 需要在 load event 完成前保留 request id 映射。

当前状态：

- control-plane 已接入；
- `SimpleCPUOffloadConnector` 可被选择；
- CPU KV blocks 已能分配；
- external prefix query 可观察；
- strict external hit/reload 报告尚未通过。

## A10 当前取舍

为了单卡 A10 跑通，当前 reduced real 路径使用：

- `VLLM_DK_FAKE_ROUTED_EXPERTS=1`
- `VLLM_DK_FP8_BLOCK_TORCH_FALLBACK=1`
- `VLLM_DK_FAKE_DEEPSEEK_ATTENTION=1`
- `VLLM_DK_TORCH_KV_CACHE_INSERT=1`

保留内容：

- 真实 reduced 权重加载；
- 真实 KDA/Mamba state；
- projection 路径；
- DeepSeek attention 的 KV cache 写入；
- hybrid KV allocation、prefix cache、eviction pressure。

绕过内容：

- routed expert 真实计算；
- DeepSeek attention 生产 kernel 输出；
- A10 不适配或成本过高的高性能 FP8/indexer kernel。

## 验证矩阵

已完成：

- tiny DK dummy load completion；
- tiny DK synthetic safetensors load completion；
- sharded_state load completion；
- reduced real KDA/Mamba completion；
- DeepSeek attention torch fallback KV cache insert；
- hybrid KV group 保留 KDA/Mamba spec；
- prefix cache hit；
- cross-request prefix reuse；
- eviction pressure；
- native CPU KV offload control-plane 部分验证。

未完成：

- 完整 43 层 DK 真实 forward；
- 不带 A10 debug fake 开关的完整 DeepSeek attention/MoE；
- 2M 上下文真实显存行为；
- native CPU KV offload strict end-to-end hit/reload。

## PR 范围

目标仓库：`linsheng1/vllm`

建议分支：`dk-v4-flash-kda`

建议 PR 标题：`Add experimental DK DeepSeek-V4 KDA support`

PR 应包含：

- DK model/config 支持；
- reduced runtime；
- checkpoint 转换与 ModelScope shard 准备工具；
- A10 smoke scripts；
- hybrid KV/Mamba/offload 修复；
- 中文设计文档、运行教程和测试报告；
- DK 相关 unit tests。

PR 不应包含：

- `graphify-out/**`
- `**/__pycache__/**`
- `.smoke/**`
- 下载的真实 checkpoint
- 未整理的原始运行日志

## 当前结论

如果有人提供题干中完整、结构正确的 DK 权重，并且他们的设备支持 DeepSeek 与
Kimi KDA 所需 kernel，当前代码设计上具备加载和运行的主路径。但这仍需要在目标
设备上做完整 43 层验证；本地和 A10 目前证明的是结构正确性、reduced real
forward、真实 KDA/Mamba state、KV 写入、prefix cache 和 eviction pressure。
