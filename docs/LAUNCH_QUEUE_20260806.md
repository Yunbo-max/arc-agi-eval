# ARC-AGI 追踪方法长目标 Launch Queue（2026-08-06）

## 当前环境与协议前置（可复用快照）

- 主机观察：
  - GPU: `NVIDIA RTX 3090`（显存 24 GiB）
  - `python3 scripts/snapshot_host_capacity.py --output-directory reports/e0-resources/20260806-host-capacity-launch-current2`
    - 结论：`disk_reserve_currently_satisfied=true`，`ten_gib_free_vram_and_idle_on_every_visible_gpu=false`
  - `python3 scripts/audit_protocol_root.py --output-directory reports/e0-protocol/20260806-protocol-v1-draft-root-launch-current2`
    - 结论：`protocol_status=draft-not-frozen`，`required_unmet_gate_ids=["lp.process-tree-resources"]`
- `python3 scripts/audit_process_tree_resource_gate.py --output-directory reports/e0-resources/20260806-process-tree-resource-gate-launch-current2`
  - 结论：门禁 `lp.process-tree-resources` 仍 blocked（cgroup v2/delegation/NVIDIA accounting 及子进程含量计量都未闭环）
- `python3 scripts/snapshot_host_capacity.py --output-directory reports/e0-resources/20260806-host-capacity-launch-current3`
  - 最新补拍（2026-08-06T16:52:31Z）结论一致：`ten_gib_free_vram_and_idle_on_every_visible_gpu=false`
- `python3 scripts/audit_protocol_root.py --output-directory reports/e0-protocol/20260806-protocol-v1-draft-root-launch-current3`
  - 新快照仍为 `draft-not-frozen`，`required_unmet_gate_ids=["lp.process-tree-resources"]`
- `python3 scripts/audit_process_tree_resource_gate.py --output-directory reports/e0-resources/20260806-process-tree-resource-gate-launch-current3`
  - 新快照仍 blocked：cgroup v2、delegation 与 NVIDIA accounting/进程树计量尚未闭环
- `python3 scripts/calibrate_resources.py --output-directory reports/e0-resources/20260806-process-resource-calibration-current3`
  - 当前进程开销校准通过（`process/RSS 仅覆盖本进程`，GPU 计量仅抽样）  

在 `lp.process-tree-resources` 闭环前，不建议开始任何“锁定式公开 benchmark”。

## 方法分批（按可运行性 / 资源 / 实验价值）

说明：
- `smoke`：方法级 component/smoke evidence；
- `subset`：预声明子任务集/同形态任务子集；
- `full`：完整公开 benchmark；
- `repro`：论文级复现；
- 目标顺序以阻塞最少、门禁最清晰者优先。

## 如果只按时间片推进（推荐版本）

这不是按“arc-challenge-2024/2025/2026”论文年度来切，而是按**目标评测板块 + 当前门控成本**来切。先跑能形成可审计证据的最小闭环（smoke→subset），最后再谈 benchmark/repro。

- **轮次 1（当日优先）**：压缩到最小阻塞、已知可跑的子任务扩展
  - ARC-AGI-1：CompressARC、ARC_NCA、LPN、2D nGPT、GridCoder2024
  - 说明：优先保留“有可追溯 run.json 的最小子任务”路径；若同一方法缺少入口，先补充 loader/adapter 再下一个任务
- **轮次 2（低风险推进）**：已过 smoke 但有公开权重/环境约束的本地阶段
  - ARC-AGI-1：BARC、ARChitects 2024
  - 说明：目前更像 capacity/gate readiness（下载、许可、依赖、本地复现边界）优先于得分
- **轮次 3（需适配/API/组件分离）**：零/低成本 API 与组件验证
  - ARC-AGI/ARC-AGI-2：ArcMemo、arc-lang-public、epang-arc-agi、NVARC
  - Native non-ARC：LatentMAS、AgentPrimitives、GraphPlanner、RouteMoA、MACA
  - 说明：先修好 label firewall / 沙箱 / 外部调用计费和预算，当前不宜直接上 ARC-AGI benchmark
- **轮次 4（blocked / 无公开实现）**：
  - ARC-AGI-1：ARC-VSA-2025、TinyRecursiveModels、SOAR、MARC、Omni-ARC、Mini-ARC transformer
  - ARC-AGI-2：N/A
  - Native non-ARC：NeuroMAS、ReM-MoA
  - 说明：多数在该状态下先做源码核验与依赖补齐，短期不适合 benchmark

### 按 ARC-AGI 板块/时间片的先后顺序（你可以直接按这三层开跑）

1) **ARC-AGI-1 主线（可复现边界先行）**
   1. `compressarc`, `arc-nca`, `lpn`, `2d-ngpt`, `gridcoder2024`
   2. `barc`, `architects-2024`
   3. `soar`, `marc`

2) **ARC-AGI-1/2 同时相关（先做风险缓解）**
   1. `arc-vsa-2025`（当前 blocked）
   2. `arc-lang-public`, `epang-arc-agi`, `arcmemo`
   3. `nvarc`（ARC-AGI-2 主体）

3) **Non-ARC native（非 ARC-AGI 本地）**
   1. `latentmas`, `agent-primitives`, `graphplanner`, `routemoa`, `maca`
   2. `tiny-recursive-models`（兼容 ARC-AGI-1/2，但现在阻塞在数据/标签/容量）
   3. `omni-arc`, `mini-arc-transformer`, `neuromas`, `rem-moa`（blocked）

### Batch A（低成本接续，先做可复现边界）

| 阶段 | 方法 | 当前状态 | 下一步 |
|---|---|---|---|
| A1 | CompressARC | smoke 已通过 | 已补齐同形态子任务证据：`reports/compressarc/20260806-subset-eval-009d5c81-1step/run.json`（evaluation，1 step，task=009d5c81） |
| A2 | ARC_NCA | smoke 已通过 | 已补充同形态子任务证据：`reports/arc-nca/20260806-subset-eval-16b78196-1step/run.json`（evaluation，1 step，task=16b78196）、`reports/arc-nca/20260806-subset-training-045e512c-1step/run.json`（training，1 step，task=045e512c）、`reports/arc-nca/20260806-subset-eval-00dbd492-2step/run.json`（evaluation，2 step，task=00dbd492）、`reports/arc-nca/20260806-subset-train-025d127b-1step-goalrun/run.json`（training，1 step，task=025d127b） |
| A3 | GridCoder2024 | architecture-only 已过 | 重建可追溯的上游来源与 checkpoint 预检后，先做 challenge-only 架构/adapter 小批次 |
| A4 | 2D nGPT | architecture-only 已过 | 补齐 exp_50/54、固定形状重建与子任务集 provenance 后再做固定子集 |
| A5 | LPN | architecture smoke 已过 | 先绑定 7 个 W&B artifact 的哈希/许可证，再做最小 challenge-only adapter 再跑同形态子集 |

### Batch B（本地权重，容量门控）

| 阶段 | 方法 | 当前状态 | 下一步 |
|---|---|---|---|
| B1 | ARChitects 2024 | forward 受 VRAM 门禁阻断 | 先完成 cgroup/进程树计量，再做单卡 4-bit checkpoint 的固定子任务可重复 run |
| B2 | BARC | seeds smoke 已过 | 先完成 base/LoRA provenance+license、challenge-only direct-transduction adapter、容量与依赖核验 |

### Batch C（零消费 API / 低成本）

| 阶段 | 方法 | 当前状态 | 下一步 |
|---|---|---|---|
| C1 | ArcMemo | component/API smoke 通过 | 完成无标签网络/防外泄的 challenge-only 接口后定义本地 native/固定预算运行方案 |
| C2 | arc-lang-public | parser/component smoke 通过 | 完成 raw-key 防火墙+预算上限+无标签/离线策略再评估 |
| C3 | epang080516/arc_agi | pickle 路径未执行，辅助 scorer 通过 | 分离不可信 pickle 与生成代码路径，先做固定预算零标签实验 |

### Batch D（高风险/集成重）

| 阶段 | 方法 | 当前状态 | 下一步 |
|---|---|---|---|
| D1 | TinyRecursiveModels | dataset/组件层 evidence 已通过 | 先解决数据/checkpoint/依赖/容量/标签防火墙，再才考虑固定原生子集 |
| D2 | SOAR | challenge-only 风格 evidence 已过 | 建立 label-free loader 与代码隔离后再选最小 checkpoint 跑同形态子集 |
| D3 | NVARC | source-only gate 通过 | 已补充组件/固定 helper smoke：`reports/nvarc/20260806-zero-dollar-component-source-smoke-goalrun3/run.json`；继续绑定 artifact/provenance 与 contamination 控制再做真实组件/挑战子集 |
| D4 | MARC | ARC 组件 smoke 通过 | 初始化 torchtune 子模块后补齐 native benchmark adapter 与资源可行性 |
| D5 | LatentMAS | AI2 ARC-Challenge 提示 schema smoke | 明确 native ARC 适配计划（非 paper-parity）后再考虑跑 reduced native 子集 |
| D6 | AgentPrimitives | static smoke 已过 | 修复 case-sensitive 导入与启动器缺失后才可做组件外的执行尝试 |
| D7 | GraphPlanner | schema/component smoke 已过 | 先获得可运行 checkpoint/API/路由执行协议 |
| D8 | RouteMoA | 路由/预计算评分完整性 smoke 已补跑 | 使用同一脚本输出已形成 scorer-only 证据，但无 solver 推理；后续需 router+模型池+API judge 才可进入 solver 级 |
| D9 | MACA | GraphSpec component smoke 已过 | 需发布/绑定 checkpoint/后端与数据适配器后再做真实执行 |

### Batch E（阻塞观察）

| 阶段 | 方法 | 当前状态 | 下一步 |
|---|---|---|---|
| E1 | ARC-VSA-2025 | blocker | 依赖与标签路径补齐后再审 |
| E2 | Omni-ARC | blocked（无可运行仓库） | 待可验证 runnable 实现 |
| E3 | Mini-ARC transformer | blocked（无公开实现） | 同上 |
| E4 | NeuroMAS | blocked（无公开实现） | 同上 |
| E5 | ReM-MoA | blocked（无公开实现） | 同上 |

## 本轮新增可追溯 run

- `reports/routemoa/20260806-routing-routemoa-smoke-relaunch1/run.json`  
  - 状态：`passed`
  - 证据类型：zero-dollar precomputed-results integrity/scope；非 solver/非 ARC benchmark
- `reports/routemoa/20260806-routing-routemoa-smoke-recheck/run.json`  
  - 状态：`passed`
  - 证据类型：zero-dollar precomputed-results integrity/scope；非 solver/非 ARC benchmark（本轮复核）
- `reports/graphplanner/20260806-graphplanner-schema-smoke-env-ok2/run.json`  
  - 状态：`passed`
  - 证据类型：`zero-dollar-bundled-router-data-schema-and-prompt-component`；CPU-only、零成本、无外网调用
- `reports/maca/20260806-maca-component-smoke-env-ok2/run.json`  
  - 状态：`passed`
  - 证据类型：`zero-dollar-graphspec-random-weight-cpu-component-only`；`CUDA_VISIBLE_DEVICES` 空置、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONHASHSEED=0` 条件下通过
- `reports/architects-2024/20260806-4bit-forward-preflight-check-v8/run.json`  
  - 状态：`blocked`
  - 证据类型：`published-4bit-checkpoint-one-token-forward`；GPU 空闲门控不足（`minimum_free_vram_bytes=8GiB`，实测可用约 `6.45GiB`），未执行模型加载
- `reports/barc/20260806-barc-seed-smoke-recheck/run.json`  
  - 状态：`passed`
  - 证据类型：seed 合成器 smoke（`00d62c1b`），不加载模型，不参与 ARC-AGI 求解
- `reports/barc/20260806-barc-seed-smoke-0520fde7-run1/run.json`  
  - 状态：`failed`
  - 证据类型：seed 复核 smoke（`0520fde7`），输入/输出形状不匹配导致 claim 边界失败
- `reports/barc/20260806-barc-seed-smoke-05f2a901-run1/run.json`  
  - 状态：`passed`
  - 证据类型：seed 复核 smoke（`05f2a901`），`input_shape=[8,20]`, `output_shape=[8,20]`，未加载模型
- `reports/compressarc/20260806-subset-eval-00576224-2step-goalrun4/run.json`  
  - 状态：`passed`
  - 证据类型：evaluation 子任务 2 step（`00576224`，`output/task pass-at-2=0.0`，`cell_accuracy=0.25`，峰值 VRAM `81,200,128` 字节）
- `reports/arc-nca/20260806-subset-2-eval-arcagi2-0934a4d8-2step-goalrun/run.json`  
  - 状态：`failed`
  - 证据类型：ARC-AGI-2 任务 `0934a4d8` 2 step 机制尝试；失败于协议约束（`ignore-size-change protocol requires equal train input/output shapes`）
- `reports/architects-2024/20260806-4bit-forward-preflight-check-8gib/run.json`  
  - 状态：`blocked`
  - 证据类型：4bit one-token 前检（`minimum_free_vram_gib=8.0`，实测 `free_memory_bytes=6,928,990,208`，未执行模型加载）
- `reports/architects-2024/20260806-4bit-forward-preflight-check-10gib/run.json`  
  - 状态：`blocked`
  - 证据类型：4bit one-token 前检（`minimum_free_vram_gib=10.0`，实测 `minimum_free_vram_bytes=10,737,418,240`，未执行模型加载）
- `reports/compressarc/20260806-subset-eval-00dbd492-1step/run.json`  
  - 状态：`passed`
  - 证据类型：evaluation 子任务 1 step 机制证据（`0.0` 得分，task=00dbd492）
- `reports/arc-nca/20260806-subset-training-007bbfb7-1step/run.json`  
  - 状态：`failed`
  - 证据类型：`ignore-size-change protocol requires equal train input/output shapes`（阻塞证据）
- `reports/arc-nca/20260806-subset-eval-16b78196-1step/run.json`  
  - 状态：`passed`
  - 证据类型：evaluation 子任务 1 step 机制证据（`0.0` output/task pass-at-2，峰值 VRAM `991,253,504` 字节，task=16b78196）
- `reports/arc-nca/20260806-subset-training-045e512c-1step/run.json`  
  - 状态：`passed`
  - 证据类型：training 子任务 1 step 机制证据（`0.0` output/task pass-at-2，峰值 VRAM `495,230,976` 字节，task=045e512c）
- `reports/2d-ngpt/20260806-large-architecture-forward-smoke/run.json`  
  - 状态：`passed`
  - 证据类型：依赖修复后重跑通过；输出形状 `(1,30,30,10)`，参数量 `38,047,106`，峰值 VRAM `209,891,328` 字节
- `reports/architects-2024/20260806-4bit-forward-preflight-lower-threshold4/run.json`  
  - 状态：`passed`
  - 证据类型：补齐 `accelerate`/`bitsandbytes`/`transformers` 后的 1-token forward smoke；在 4GiB 预检门限下通过，峰值 VRAM 约 `3,799,676,928`
- `reports/architects-2024/20260806-4bit-forward-preflight-lower-threshold6/run.json`  
  - 状态：`passed`
  - 证据类型：同上 forward 流程在 6GiB 门限下复核通过，峰值 VRAM 约 `3,799,676,928`
- `reports/arc-nca/20260806-subset-training-025d127b-1step/run.json`  
  - 状态：`passed`
  - 证据类型：training 子任务 1 step（task=025d127b，`0.0` output/task pass-at-2，峰值 VRAM `150,512,640`）
- `reports/arc-nca/20260806-subset-train-025d127b-1step-goalrun/run.json`  
  - 状态：`passed`
  - 证据类型：training 子任务 1 step（task=025d127b，`0.0` output/task pass-at-2，`cell_accuracy=0.82`，峰值 VRAM `150,512,640`）
- `reports/arc-nca/20260806-subset-eval-00dbd492-1step-new/run.json`  
  - 状态：`passed`
  - 证据类型：evaluation 子任务 1 step（task=00dbd492，`0.0` output/task pass-at-2，峰值 VRAM `59,927,040`，输出任务共 1/1）
- `reports/arc-nca/20260806-subset-arcagi2-eval-135a2760-1step-goalrun2/run.json`  
  - 状态：`passed`
  - 证据类型：ARC-AGI-2 任务 `135a2760` 1 step 子任务，输出 `output/task pass-at-2=0.0`，`cell_accuracy=0.00118906`，峰值 VRAM `79,180,800` 字节，用于 ARC-AGI-1/2 分离比较
- `reports/compressarc/20260806-subset-arcagi2-train-00dbd492-1step-goalrun2/run.json`  
  - 状态：`passed`
  - 证据类型：ARC-AGI-2 训练子任务 `00dbd492` 1 step，`output/task pass-at-2=0.0`，`cell_accuracy=0.2175`，峰值 VRAM `599,622,144` 字节
- `reports/arc-nca/20260806-subset-arcagi2-train-00dbd492-1step-goalrun2/run.json`  
  - 状态：`passed`
  - 证据类型：ARC-AGI-2 训练子任务 `00dbd492` 1 step，`output/task pass-at-2=0.0`，`cell_accuracy=0.5475`，峰值 VRAM `261,460,480` 字节
- `reports/arc-nca/20260806-subset-eval-00dbd492-2step/run.json`  
  - 状态：`passed`
  - 证据类型：evaluation 子任务 2 step（task=00dbd492，`0.0` output/task pass-at-2，未加载标签；`run.json` 未记录峰值资源）
- `reports/nvarc/20260806-zero-dollar-component-source-smoke-goalrun3/run.json`  
  - 状态：`passed`
  - 证据类型：component/source wiring 与 fixed trusted helper smoke；`arc_data_loaded=false`、`model_or_checkpoint_loaded=false`、`benchmark_executed=false`、`predictions_generated=false`、`root_license_files=[]`
- `reports/e0-protocol/20260806-protocol-v1-draft-root-launch-20260806T182930Z/run.json`  
  - 状态：`passed`（审计通过）
  - 证据类型：protocol-root 审核（`protocol_status=draft-not-frozen`，`required_unmet_gate_ids=["lp.process-tree-resources"]`）
- `reports/e0-resources/20260806-process-tree-resource-gate-launch-20260806T182722Z/run.json`  
  - 状态：`passed`（审计通过）
  - 证据类型：`lp.process-tree-resources` 门禁仍 blocked（cgroup v2/委派/NVIDIA accounting 均未闭环，当前有 GPU 占用）
- `reports/e0-resources/20260806-host-capacity-goalrun5/run.json`  
  - 状态：`passed`
  - 证据类型：主机快照（`free_bytes=23344746496`、`maximum_admissible_incremental_bytes=14754811904`、`ten_gib_free_vram_and_idle_on_every_visible_gpu=false`）
- `reports/compressarc/20260806-subset-eval-136b0064-1step-goalrun5/run.json`  
  - 状态：`passed`
  - 证据类型：ARC-AGI-2 evaluation 子任务 1 step，`output/task pass-at-2=0.0`，`cell_accuracy=0.0`
- `reports/arc-nca/20260806-subset-arcagi2-eval-136b0064-1step-goalrun7/run.json`  
  - 状态：`failed`
  - 证据类型：ARC-AGI-2 evaluation 子任务 1 step；protocol 阻塞：`ignore-size-change protocol requires equal train input/output shapes`
- `reports/arc-lang-public/20260806-arc-lang-public-goalrun5/run.json`  
  - 状态：`passed`
  - 证据类型：`zero-dollar-import-config-challenge-parser`；`first_task_id=00576224`
- `reports/e0-resources/20260806-host-capacity-launch-current4c/run.json`  
  - 状态：`passed`
  - 证据类型：主机快照（`ten_gib_free_vram_and_idle_on_every_visible_gpu=false`）
- `reports/e0-protocol/20260806-protocol-v1-draft-root-goalrun15/run.json`  
  - 状态：`passed`
  - 证据类型：protocol-root 审核（`protocol_status=draft-not-frozen`，`required_unmet_gate_ids=["lp.process-tree-resources"]`）
- `reports/e0-resources/20260806-process-tree-resource-gate-goalrun12b/run.json`  
  - 状态：`passed`（audit passed；`gate` blocked）
  - 证据类型：`lp.process-tree-resources` 子进程树门禁仍 blocked（cgroup v2/unified/delegation/NVIDIA accounting/占用 GPU）
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-goalrun15/run.json`  
  - 状态：`passed`
  - 证据类型：reproduction-funnel 对齐；`smoke_passed_count=19`、`source_audit_passed=20`、`benchmark_passed=0`、`full_reproduction_passed=0`

- `reports/e0-resources/20260806-process-tree-resource-gate-goalrun11/run.json`  
  - 状态：`passed`（审计通过）
  - 证据类型：`lp.process-tree-resources` 门禁仍 blocked（`cgroup_v2_unified_unavailable`、`nvidia_accounting_not_enabled`、`gpu_currently_occupied`）
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-launch-current4/run.json`  
  - 状态：`passed`
  - 证据类型：reproduction-funnel 对齐；当前 `smoke_passed_count=19`、`source_audit_passed=20`、`benchmark_passed=0`、`full=0`
- `reports/routemoa/20260806-routemoa-source-audit-batch2/run.json`  
  - 状态：`failed`
  - 证据类型：source-audit（`scripts.audit_source`）；语法阻塞：`babilong_2k_gen.py` 包含全角逗号 `U+FF0C`
- `reports/maca/20260806-maca-source-audit-batch2/run.json`  
  - 状态：`failed`
  - 证据类型：source-audit（`scripts.audit_source`）；语法阻塞：`infer_graphspec.py` 非打印字符 `U+E63F`，`readers.py` 存在异常续行字符
- `reports/routemoa/20260806-routemoa-source-audit-batch4/run.json`  
  - 状态：`passed`
  - 证据类型：source-audit（`scripts.audit_source`）；`routemoa` 锁定修订通过（`8d07c48...`），语法通过
- `reports/maca/20260806-maca-source-audit-batch3/run.json`  
  - 状态：`passed`
  - 证据类型：source-audit（`scripts.audit_source`）；`maca` 锁定修订通过（`62bd012...`），语法通过
- `reports/maca/20260806-maca-component-smoke-env-ok2/run.json`  
  - 状态：`passed`
  - 证据类型：`zero-dollar-graphspec-random-weight-cpu-component-only`；在 `CUDA_VISIBLE_DEVICES` 空置、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONHASHSEED=0` 的可复现实验条件下通过
- `reports/barc/20260806-source-artifact-label-resource-gate-v3/run.json`  
  - 状态：`failed`
  - 证据类型：严格源/Artifact/label/resource 门禁（`scripts.audit_barc_gates`）；阻塞点：BARC 源码树包含未声明文件 `seeds/__pycache__/007bbfb7.cpython-310.pyc`
- `reports/barc/20260806-source-artifact-label-resource-gate-v6/run.json`  
  - 状态：`passed`（方法层级 `method_gate_status=blocked`）
  - 证据类型：`scripts.audit_barc_gates`
  - 关键阻塞：`root-license`、`dependency-lock`、`base/lora-artifact provenance`、`safe-offline-model-load`、`label-firewall`、`single-gpu-capacity`、`solver-parity`；静态关闭世界校验通过，表示已进入可复用 blocker 诊断层
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-continue-batch2/run.json`  
  - 状态：`passed`
  - 证据类型：repro-funnel 门禁与方法分层聚合；当前 24 方法中 smoke 通过 19、source-audit 通过 18、benchmark/论文复现均未开始；可见 4 个方法 source-audit 尚未可计入（omni-arc、mini-arc-transformer、neuromas、rem-moa）
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-launch-goal-run2/run.json`  
  - 状态：`passed`
  - 证据类型：manifest 与通过证据完整性复核；当前 `smoke_passed_count=19`、`source_audit_passed=20`、`benchmark_passed=0`、`full=0`
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-goalrun11/run.json`  
  - 状态：`passed`
  - 证据类型：最新 reproduction-funnel 快照；当前 `smoke_passed_count=19`、`source_audit_passed=20`、`benchmark_passed=0`、`full_reproduction_passed=0`
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-goalrun10/run.json`  
  - 状态：`passed`
  - 证据类型：最新 reproduction-funnel 快照；当前 `smoke_passed_count=19`、`source_audit_passed=20`、`benchmark_passed=0`、`full_reproduction_passed=0`

### 2D nGPT 依赖修复（可复用）

- 已安装执行所需依赖：`pandas`、`rotary-embedding-torch`

## 下一个可执行建议（先后）

1. 不改现有协议门禁前提提下，继续清理并闭环 `process-tree-resource` 门禁（需要可写 cgroup v2 后端与 GPU 进程树计量校准）。
2. 选择 Batch B1 的 ARChitects：先在 `GPU free >= 8GiB` 条件下重试 4-bit one-token 前检（当前新复核是 `blocked`），再并行推进许可/本地-only 与 checkpoint 适配。
3. 并行更新执行日志到 `configs/baselines.json` 对应 `reproduction` 字段（smoke/subset/full）与 `blockers`，确保所有成功/失败/阻塞可审计。
4. 在 BARC 与 routemoa/maca 分别做最小清理后重跑剩余门禁（BARC 已完成门禁静态层并暴露 blocker；routemoa/maca 已完成源码语法修复与最小 component smoke，下一步集中修补 license/provenance/离线能力及后端/API 适配）。
