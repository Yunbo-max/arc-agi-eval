# 2026-08-06 执行分批计划（与可运行性/资源/价值对应）

- 当前门禁：`lp.process-tree-resources` 与 `protocol_root` 尚未 `frozen`; 不建议在 gate 关闭前进入公开 benchmark。
- 分批原则：先最小可审计证据（smoke）→ challenge-only 单任务/子集（有固定随机与独立打分）→ benchmark → full/论文复现。

| 阶段 | 方法 | ARC-AGI 覆盖 | 当前repro状态(S/B/F) | 资源等级 | 证据范围 | 价值/入口 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- | --- |
1.1 | CompressARC | ARC-AGI-1 + ARC-AGI-2 | passed/not_started/not_started | single_gpu_long_running | solver_prediction / reduced_method_execution | trusted_locked / A full split is long because every task receives independent training. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
1.2 | ARC-VSA-2025 | ARC-AGI-1 + ARC-AGI-2 | blocked/blocked/blocked | single_gpu_long_running | blocker_audit / blocked_before_method_execution | trusted_locked / The released solver imports an unavailable and unidentified sspspace implementation.；No end-to-end local command or resource report is documented. | 修复 smoke 或替代来源/实现
1.3 | ARC_NCA | ARC-AGI-1 | passed/not_started/not_started | single_gpu_long_running | solver_prediction / reduced_method_execution | trusted_locked / No requirements file or single benchmark entry point exists.；Notebook parameters and environment need reconstruction. 已补充 evaluation 1-step 子任务证据（`00dbd492`） | 先做挑战集/固定子任务再判断是否可上公开 benchmark
1.4 | GridCoder2024 | ARC-AGI-1 | passed/not_started/not_started | single_gpu_24g | component / component_only | trusted_locked / The evaluator runs one named task at a time.；Paper experiments cover a declared DSL-solvable subset rather than all ARC tasks. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
1.5 | 2D nGPT | ARC-AGI-1 | passed/not_started/not_started | single_gpu_24g | component / component_only | trusted_locked / Training duration is not specified.；re-ARC generation and training data may exceed local storage. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
1.6 | LPN | ARC-AGI-1 | passed/not_started/not_started | multi_gpu_or_above_host | component / component_only | trusted_locked / No paper checkpoint is advertised.；Published accelerator topology is not matched by one RTX 5090. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
2.7 | ARChitects 2024 | ARC-AGI-1 | passed/not_started/not_started | single_gpu_24g | blocker_audit / blocked_before_method_execution | trusted_locked / Full retraining is storage-sensitive.；Exact historical dependency resolution may require repair. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
2.8 | BARC | ARC-AGI-1 | passed/not_started/not_started | single_gpu_24g | component / component_only | generated_untrusted / Paper-scale fine-tuning uses an eight-process ZeRO-3 recipe.；Full model and dataset footprint exceeds available storage. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
2.9 | arc-lang-public | ARC-AGI-1 + ARC-AGI-2 | passed/not_started/not_started | metered_api | component / component_only | api_network / API quota and spend are unresolved.；Mutable proprietary model snapshots prevent strict parity. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
2.10 | epang080516/arc_agi | ARC-AGI-1 + ARC-AGI-2 | passed/not_started/not_started | metered_api | component / component_only | generated_untrusted / Reported runs depend on paid frontier APIs.；Library ordering and provider model drift affect results. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
2.11 | ArcMemo | ARC-AGI-1 | passed/not_started/not_started | metered_api | component / component_only | generated_untrusted / Proprietary model snapshots and rollout spend are not fixed.；Continual-memory ordering must be pinned. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.12 | TinyRecursiveModels | ARC-AGI-1 + ARC-AGI-2 | passed/not_started/not_started | multi_gpu_or_above_host | component / component_only | trusted_locked / ARC runs are reported at about three days on four H100s.；No paper checkpoint is advertised. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.13 | SOAR | ARC-AGI-1 | passed/not_started/not_started | multi_gpu_or_above_host | component / component_only | generated_untrusted / Smallest model plus useful data is storage-tight.；Full evolutionary training and larger models require substantially more compute. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.14 | NVARC | ARC-AGI-2 主对比 | passed/not_started/not_started | multi_gpu_or_above_host | component / component_only | generated_untrusted / The 3.2M augmented corpus and full ensemble exceed 41 GiB.；Several independently complex systems and external Kaggle assets must be coordinated. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.15 | MARC | ARC-AGI-1 | passed/not_started/not_started | multi_gpu_or_above_host | component / component_only | trusted_locked / Upstream warns that repository cleanup is still in progress.；Incompatible training/inference stacks and model caches exceed the practical storage budget. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.16 | LatentMAS | non-ARC 适配 | passed/not_started/not_started | multi_gpu_or_above_host | component / component_only | trusted_locked / Official 14B hybrid path recommends two GPUs.；One-GPU 4B/8B runs are reduced configurations. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.17 | AgentPrimitives | non-ARC 适配 | passed/blocked/blocked | unknown | component / component_only | trusted_locked / README states that the end-to-end demo and detailed experiment pipelines are forthcoming. | 补齐 API/firewall/code provenance/容量门禁，再做固定子任务
3.18 | GraphPlanner | non-ARC 适配 | passed/not_started/blocked | metered_api | component / component_only | api_network / NVIDIA NIM access and exact interaction histories are required.；No trained router checkpoint is advertised. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.19 | RouteMoA | non-ARC 适配 | passed/not_started/blocked | multi_gpu_or_above_host | scorer_only / scorer_only | api_network / Local paper setup used five A800 80 GiB GPUs.；Large-pool reproduction requires several paid APIs and an API judge. | 先做挑战集/固定子任务再判断是否可上公开 benchmark
3.20 | MACA | non-ARC 适配 | passed/blocked/blocked | unknown | component / component_only | trusted_locked / No documented command, expected output, model, checkpoint, or hardware budget is published.；GRPO/VERL training requirements are unknown. | 补齐 API/firewall/code provenance/容量门禁，再做固定子任务
4.21 | Omni-ARC | ARC-AGI-1 | blocked/blocked/blocked | unavailable | unavailable / unavailable | unavailable / No verified public implementation or default branch. | 修复 smoke 或替代来源/实现
4.22 | Mini-ARC transformer implementation | ARC-AGI-1 | blocked/blocked/blocked | unavailable | unavailable / unavailable | unavailable / No verified public implementation or default branch. | 修复 smoke 或替代来源/实现
4.23 | NeuroMAS | non-ARC 适配 | blocked/blocked/blocked | unavailable | unavailable / unavailable | unavailable / No verified public implementation or default branch. | 修复 smoke 或替代来源/实现
4.24 | ReM-MoA | non-ARC 适配 | blocked/blocked/blocked | unavailable | unavailable / unavailable | unavailable / No verified public implementation or default branch. | 修复 smoke 或替代来源/实现

## ARC-AGI-1 与 ARC-AGI-2 分离对比（当前可审计层面）

| 分区 | 方法数 | 方法 | 状态 | 说明 |
| --- | ---: | --- | --- | --- |
ARC-AGI-1 | 16 | CompressARC, ARC-VSA-2025, ARC_NCA, GridCoder2024, 2D nGPT, LPN, ARChitects 2024, BARC, arc-lang-public, epang080516/arc_agi, ArcMemo, TinyRecursiveModels, SOAR, MARC, Omni-ARC, Mini-ARC transformer implementation | mostly smoke 已过，0/16 benchmark | 仅 ARC_NCA/CompressARC 等存在 method-specific 严格审计，但尚未可复现 benchmark
ARC-AGI-2 | 1 | NVARC | smoke 已过，benchmark 未启动 | 原始训练/结果与 ARC-AGI-1 交叉任务去重未闭环

### 资源敏感度
- metered_api: 4
- multi_gpu_or_above_host: 7
- single_gpu_24g: 4
- single_gpu_long_running: 3
- unavailable: 4
- unknown: 2
