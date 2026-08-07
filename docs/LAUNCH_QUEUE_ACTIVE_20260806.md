# ARC-AGI 方法执行队列（当前活动目标快照）

更新时间：`2026-08-06`（按你当前目标继续推进）

## 当前门禁与前置条件

- `scripts/audit_protocol_root.py --output-directory reports/e0-protocol/20260806-protocol-v1-draft-root-goalrun2`  
  -> `protocol_status=draft-not-frozen`，`freeze_ready=false`，未满足的唯一必填门禁：`lp.process-tree-resources`。
- `scripts/audit_process_tree_resource_gate.py --output-directory reports/e0-resources/20260806-process-tree-resource-gate-goalrun1`  
  -> `lp.process-tree-resources` 仍 blocked（当前环境为 `cgroup` v1，未形成可计量的子进程树资源闭环）。
- 当前建议：在 `lp.process-tree-resources` 闭环前，只做 source/smoke/子任务实验，不做锁定式公开 benchmark/full 级别运行。

## 官方评测划分（按比赛账本，不按方法名年份）

- `arc-prize-2024` 原生评测链路：  
  `barc`, `lpn`, `architects-2024`, `gridcoder2024`, `2d-ngpt`, `soar`, `marc`, `omni-arc`, `mini-arc-transformer`
- `arc-prize-2025` 官方主账本关联：  
  `nvarc`（官方 ARC-AGI-2，亦可做 ARC-AGI-1 迁移对比，但不可直接合并）
- 按方法声明的公开结果/目标：  
  `tiny-recursive-models`, `compressarc`, `arc-vsa-2025`, `arc-lang-public`, `epang-arc-agi`, `arc-nca`, `arcmemo`, `latentmas`
- 非 ARC-AGI 线（native/组件实验）：  
  `agent-primitives`, `graphplanner`, `routemoa`, `maca`, `neuromas`, `rem-moa`
- 当前无已验证 `arc-prize-2026` 条目。

## 24 方法执行序列（按可运行性/资源/价值）

- `phase`/`order` 引用 `configs/baselines.json` 的优先级；数值越小越先执行。
- 目标状态是按 `S/B/F` 逐级推进：  
  `smoke -> subset(固定子任务) -> benchmark -> full-repro`。

| phase | order | 方法 | Smoke | Benchmark | Full | 可执行层 | 说明 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 1 | compressarc | passed | not_started | not_started | subset→benchmark | CPU 1-step subset 已验证链路，可继续扩展可复核 subset。 |
| 1 | 2 | arc-vsa-2025 | blocked | blocked | blocked | blocker | 依赖缺失（`sspspace`）与标签写入风险先修。 |
| 1 | 3 | arc-nca | passed | not_started | not_started | subset→benchmark | 已有 1-step/2-step subset 证据，可继续固定-denom 子任务。 |
| 1 | 4 | gridcoder2024 | passed | not_started | not_started | subset准备 | 架构证据到位，需重建来源与数据/适配器先决条件。 |
| 1 | 5 | 2d-ngpt | passed | not_started | not_started | subset准备 | 需先补齐 `exp_50.pt`/`exp_54.pt`/`fixed_size.pkl` provenance。 |
| 1 | 6 | lpn | passed | not_started | not_started | blocker+subset | 先完成 W&B artifact 与许可证审计。 |
| 2 | 7 | architects-2024 | passed | not_started | not_started | subset blocked/待门禁 | 先补齐容量门禁与依赖；当前最小前置是进程树计量。 |
| 2 | 8 | barc | passed | not_started | not_started | adapter/容量清单 | seed smoke 已过，需许可+safe-load+artifact 预检。 |
| 2 | 9 | arc-lang-public | passed | not_started | not_started | adapter | 先完成 raw-key 防火墙+计费上限。 |
| 2 | 10 | epang-arc-agi | passed | not_started | not_started | adapter | 先做 pickle/生成代码隔离与去重策略。 |
| 2 | 11 | arcmemo | passed | not_started | not_started | adapter | 先做无标签网络路径与 rollout 成本边界。 |
| 3 | 12 | tiny-recursive-models | passed | not_started | not_started | blocked | 先补齐数据/ checkpoint/依赖/容量，再固定子任务。 |
| 3 | 13 | soar | passed | not_started | not_started | blocked | 先做 label-free loader 与代码隔离。 |
| 3 | 14 | nvarc | passed | not_started | not_started | blocked | 先做 artifact/provenance + overlap 控制。 |
| 3 | 15 | marc | passed | not_started | not_started | blocked/准备 | 先初始化 torchtune 子模块与 ARC adapter。 |
| 3 | 16 | latentmas | passed | not_started | not_started | blocked | 先固定 native ARC 适配实验，再评估 ARC-AGI 兼容子任务。 |
| 3 | 17 | agent-primitives | passed | blocked | blocked | blocked | 先修 runner 与 case-sensitive 导入。 |
| 3 | 18 | graphplanner | passed | not_started | blocked | blocked | 先获取可复现 checkpoint/API 交互记录。 |
| 3 | 19 | routemoa | passed | not_started | blocked | blocked | 已有 scorer-only；缺少 solver/router 与 judge。 |
| 3 | 20 | maca | passed | blocked | blocked | blocked | 缺公开命令与训练/推理文档。 |
| 4 | 21 | omni-arc | blocked | blocked | blocked | blocked | 无可验证 runnable 实现。 |
| 4 | 22 | mini-arc-transformer | blocked | blocked | blocked | blocked | 无可验证 runnable 实现。 |
| 4 | 23 | neuromas | blocked | blocked | blocked | blocked | 无可验证 runnable 实现。 |
| 4 | 24 | rem-moa | blocked | blocked | blocked | blocked | 无可验证 runnable 实现。 |

## 下一步建议（按时间片）

1. 继续补齐 `lp.process-tree-resources`（cgroup v2 委派 + NVIDIA accounting + 子进程树标定）；
2. 然后进入 Batch A 的 `compressarc` 与 `arc-nca`，复核 1~2 个稳定子任务并把每个 run 的 `run.json` 追加到 `reports/`；本轮新增：
   - `reports/compressarc/20260806-subset-eval-0607ce86-1step-goalrun2/run.json`（状态：`passed`，`metrics` 显示 task-level `output_exact_accuracy=0`，`cell_accuracy=0.17803`，top-k=2）；
   - `reports/arc-nca/20260806-subset-eval-0607ce86-1step-goalrun2/run.json`（状态：`passed`，`task 0607ce86`，`output_exact_accuracy=0`，`cell_accuracy=0.57008`）；
   - `reports/compressarc/20260806-subset-eval-00576224-2step-goalrun4/run.json`（状态：`passed`，`task 00576224`，`output_exact_accuracy=0`，`cell_accuracy=0.25`，`steps=2`，峰值 VRAM 约 81,200,128 字节）；
   - `reports/architects-2024/20260806-4bit-forward-preflight-check-8gib/run.json`（状态：`blocked`，4GiB 预检扩展到 8GiB 后仍 blocked：`free_memory_bytes=6,928,990,208`，未尝试模型加载）；
   - `reports/architects-2024/20260806-4bit-forward-preflight-check-10gib/run.json`（状态：`blocked`，在 10GiB 阈值下 blocked：`minimum_free_vram_bytes=10,737,418,240`）；
   - `reports/arc-nca/20260806-subset-2-eval-arcagi2-0934a4d8-2step-goalrun/run.json`（状态：`failed`，`ignore-size-change protocol requires equal train input/output shapes`，用于 ARC-AGI-2 兼容性记录）；
   - `reports/arc-nca/20260806-subset-train-025d127b-1step-goalrun/run.json`（状态：`passed`，`task 025d127b` training，`output_exact_accuracy=0`，`cell_accuracy=0.82`，峰值 VRAM `150,512,640` 字节）；
   - `reports/nvarc/20260806-zero-dollar-component-source-smoke-goalrun3/run.json`（状态：`passed`，`scope`: `locked source / component wiring / fixed trusted helper only`）；
   - `reports/compressarc/20260806-subset-arcagi2-train-00dbd492-1step-goalrun2/run.json`（状态：`passed`，`task 00dbd492`，`output/task pass-at-2=0.0`，`cell_accuracy=0.2175`，`split=training`，`ARC-AGI-2`）；
   - `reports/arc-nca/20260806-subset-arcagi2-train-00dbd492-1step-goalrun2/run.json`（状态：`passed`，`task 00dbd492`，`output/task pass-at-2=0.0`，`cell_accuracy=0.5475`，`split=training`，`ARC-AGI-2`）；
   - `reports/e0-reproduction-funnel/20260806-manifest-funnel-goalrun10/run.json`（状态：`passed`，`smoke_passed_count=19`，`source_audit_passed=20`，`benchmark_passed=0`，`full_reproduction_passed=0`）；
   - `reports/arc-nca/20260806-subset-arcagi2-eval-135a2760-1step-goalrun2/run.json`（状态：`passed`，ARC-AGI-2 任务 `135a2760`，`output_exact_accuracy=0`）；
   - `reports/e0-resources/20260806-process-tree-resource-gate-goalrun11/run.json`（状态：`blocked`，`lp.process-tree-resources` 未闭环；阻塞项包含 cgroup v2 未统一、委派不可写、NVIDIA accounting 不可用、GPU 仍有占用）；
   - `reports/e0-resources/20260806-host-capacity-goalrun5/run.json`（状态：`passed`，`disk_reserve_currently_satisfied=true`，`ten_gib_free_vram_and_idle_on_every_visible_gpu=false`）；
- `reports/e0-reproduction-funnel/20260806-manifest-funnel-goalrun11/run.json`（状态：`passed`，`smoke_passed_count=19`，`source_audit_passed=20`，`benchmark_passed=0`，`full_reproduction_passed=0`）；
- `reports/compressarc/20260806-subset-eval-136b0064-1step-goalrun5/run.json`（状态：`passed`，任务 `136b0064`，ARC-AGI-2 evaluation，1-step 子任务，`output_exact_accuracy=0`，`cell_accuracy=0.0`）
   - `reports/arc-nca/20260806-subset-arcagi2-eval-136b0064-1step-goalrun7/run.json`（状态：`failed`，任务 `136b0064`，ARC-AGI-2 evaluation，`ignore-size-change protocol requires equal train input/output shapes`，`subset` protocol 阻塞）
   - `reports/arc-lang-public/20260806-arc-lang-public-goalrun5/run.json`（状态：`passed`，scope：零花费 parser/import；`first_task_id=00576224`）
   - `reports/e0-protocol/20260806-protocol-v1-draft-root-goalrun15/run.json`（状态：`passed`，`protocol_status=draft-not-frozen`）
   - `reports/e0-resources/20260806-process-tree-resource-gate-goalrun12b/run.json`（状态：`passed`，`lp.process-tree-resources` 仍 blocked）
   - `reports/e0-resources/20260806-host-capacity-launch-current4c/run.json`（状态：`passed`，`ten_gib_free_vram_and_idle_on_every_visible_gpu=false`）
   - `reports/e0-reproduction-funnel/20260806-manifest-funnel-goalrun15/run.json`（状态：`passed`，`smoke_passed_count=19`，`source_audit_passed=20`，`benchmark_passed=0`，`full=0`）
   - `reports/compressarc/20260806-subset-eval-0a2355a6-1step-goalrun8/run.json`（状态：`passed`，任务 `0a2355a6`，`output_exact_accuracy=0`，`cell_accuracy=0.11373`）
   - `reports/arc-nca/20260806-subset-eval-0a2355a6-1step-goalrun8/run.json`（状态：`passed`，任务 `0a2355a6`，`output_exact_accuracy=0`，`cell_accuracy=0.64706`）
3. 紧接 Batch B 的容量/许可前置（ARChitects 4-bit + BARC base/LoRA provenance）；
4. 每个可行 run 维持 `S/B/F` 状态不混淆：  
   - 任何阻塞都写 `blocked`；  
   - 每次只在 `run.json` 记录真实失败、阻塞原因、资源消耗、输入规模。  
5. ARC-AGI-1 与 ARC-AGI-2 结果保持分表比较，禁止互混。
