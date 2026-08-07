# ARC-AGI 长目标：方法批次进度与公平比较草案（2026-08-06）

更新时间：2026-08-06T20:20:00Z（按 `20260806` 最新快照）

## 1) 方法证据矩阵（当前快照）
| method-id | smoke | benchmark | full | 有source-audit记录 | 有smoke记录 | 有subset记录 | 备注 |
|---|---|---|---|---:|---:|---:|---|
| barc | passed | not_started | not_started | yes | yes | no |  |
| lpn | passed | not_started | not_started | yes | yes | no |  |
| architects-2024 | blocked | not_started | not_started | yes | no | no |  |
| gridcoder2024 | passed | not_started | not_started | yes | yes | no |  |
| 2d-ngpt | passed | not_started | not_started | yes | yes | no |  |
| tiny-recursive-models | passed | blocked | not_started | yes | yes | no |  |
| soar | passed | not_started | not_started | yes | yes | no |  |
| compressarc | passed | not_started | not_started | yes | yes | yes |  |
| arc-vsa-2025 | blocked | blocked | blocked | yes | no | no |  |
| arc-lang-public | passed | blocked | not_started | yes | yes | no |  |
| epang-arc-agi | passed | blocked | not_started | yes | yes | no |  |
| arc-nca | passed | not_started | not_started | yes | no | yes |  |
| arcmemo | passed | blocked | not_started | yes | yes | no |  |
| nvarc | passed | not_started | not_started | yes | yes | no |  |
| latentmas | passed | not_started | not_started | yes | yes | no |  |
| agent-primitives | passed | blocked | blocked | yes | yes | no |  |
| graphplanner | passed | not_started | blocked | yes | yes | no |  |
| routemoa | not_started | not_started | blocked | yes | yes | no |  |
| maca | passed | blocked | blocked | yes | yes | no |  |
| marc | passed | not_started | not_started | yes | yes | no |  |
| omni-arc | blocked | blocked | blocked | yes | yes | no |  |
| mini-arc-transformer | blocked | blocked | blocked | yes | yes | no |  |
| neuromas | blocked | blocked | blocked | yes | yes | no |  |
| rem-moa | blocked | blocked | blocked | yes | yes | no |  |

## 2) ARC-AGI-1 / ARC-AGI-2 分离子任务比较（已产生的 `20260806*`子任务 runs）

最新分组标准：按官方评测账本而非方法名中的年份标签；`nvarc` 为 ARC-AGI-2 主体，可与 ARC-AGI-1 任务池分开列分。

### compressarc
| board | subset_runs | passed | failed | mean_output_exact_acc | mean_cell_acc | passed examples | failed examples |
|---|---:|---:|---:|---:|---:|---|---|
| ARC-AGI-1 | 32 | 31 | 1 | 0.0 | 0.13666657479707928 | 20260806-subset-2-eval-0934a4d8-1step, 20260806-subset-eval-00576224-1step, 20260806-subset-eval-00576224-2step-goalrun4, 20260806-subset-eval-0a2355a6-1step-goalrun8 | 20260806-training-2step-00d62c1b |
| ARC-AGI-2 | 3 | 3 | 0 | 0.0 | 0.21205578512396694 | 20260806-subset-arcagi2-train-00dbd492-1step-goalrun2, 20260806-subset-arcagi2-training-09629e4f-1step-goalrun2, 20260806-subset-arcagi2-eval-135a2760-1step-goalrun2 |  |

### arc-nca
| board | subset_runs | passed | failed | mean_output_exact_acc | mean_cell_acc | passed examples | failed examples |
|---|---:|---:|---:|---:|---:|---|---|
| ARC-AGI-1 | 55 | 45 | 10 | 0.0 | 0.6205671592831724 | 20260806-subset-2-eval-135a2760-1step, 20260806-subset-2-eval-13e47133-1step, 20260806-subset-2-eval-16b78196-1step, 20260806-subset-eval-0a2355a6-1step-goalrun8 | 20260806-subset-2-eval-0934a4d8-1step, 20260806-subset-2-eval-136b0064-1step, 20260806-subset-eval-00576224-1step |
| ARC-AGI-2 | 6 | 3 | 3 | 0.0 | 0.3068632956961246 | 20260806-subset-arcagi2-eval-135a2760-1step-goalrun2, 20260806-subset-arcagi2-train-00dbd492-1step-goalrun2, 20260806-subset-arcagi2-training-09629e4f-1step-goalrun2 | 20260806-subset-2-eval-arcagi2-0934a4d8-2step-goalrun, 20260806-subset-arcagi2-eval-136b0064-1step-goalrun2, 20260806-subset-arcagi2-eval-136b0064-1step-goalrun7 |

## 3) 当前阻塞面

- 环境门禁仍 blocked：`lp.process-tree-resources`（cgroup v2 + NVIDIA accounting + 子进程树后端）
- `omni-arc / mini-arc-transformer / neuromas / rem-moa` 仍属于无公开可跑实现，已生成 blocker run 记录。
- `arc-vsa-2025` blocker 仍在：sspspace 依赖未声明 + solver 内联 test-label 行为。
- `architects-2024 / barc / route...` 等在依赖/容量/代码隔离与 benchmark 前置后续。
