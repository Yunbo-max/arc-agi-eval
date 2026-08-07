# Execution Batches For The 24 Tracked Methods

Last classified: 2026-08-06. The observed host is one RTX 3090 with 24 GiB
VRAM and about 24.3 GiB free on the root filesystem after preparing the current
environments and ARChitects checkpoint. These batches describe the
next honest experiment for each method; they do not imply equal scientific
scope or paper reproduction. Seventeen of 24 methods currently have a passing
scope-limited smoke; two retain legacy solver-prediction smokes, and both
CompressARC and ARC_NCA have method-specific strict runtime promotions. None has a declared public
benchmark or paper-level reproduction.

## Competition year versus benchmark generation

The execution axis is the benchmark generation, not the calendar year in a
paper or repository name:

- [ARC Prize 2024](https://arcprize.org/competitions/2024) used the ARC-AGI-1
  format.
- [ARC Prize 2025](https://arcprize.org/competitions/2025) introduced
  ARC-AGI-2 for the competition. Its official results classify NVARC as first
  place on the ARC-AGI-2 private evaluation at 24.0%, and SOAR as second place
  in the Paper Award; the released SOAR method primarily targets ARC-AGI-1.
- [ARC Prize 2026](https://arcprize.org/competitions/2026) is a distinct,
  currently ongoing competition. Neither NVARC's 2025 private-evaluation result
  nor SOAR's 2025 Paper Award placement is a 2026 result, and the current
  tracking inventory contains no verified 2026 submission.

Accordingly, each method is first audited against its documented native or
published evaluation contract. ARC-AGI-2 coverage is tracked regardless of
paper year; an ARC-AGI-2 transfer of an ARC-AGI-1-only method is separately
labeled. Native non-ARC papers stay on their own benchmark unless an ARC-AGI
adaptation is explicitly declared. ARC-AGI-1 and ARC-AGI-2 scores are never
pooled into a single year-based leaderboard.

The NVARC and SOAR placements above are external official classifications and
results, not scores reproduced by this repository. The local static gates bind
only the locked workspace source and evidence described in their reports; they
do not validate the official private evaluation, rerun a solver, or reproduce
either placement.

The tracked-method cohort and native-benchmark ledgers are separate:

| Cohort | Count | Tracked methods |
| --- | ---: | --- |
| ARC Prize 2024 | 8 | BARC, LPN, ARChitects 2024, GridCoder2024, 2D nGPT, MARC, Omni-ARC, Mini-ARC transformer |
| 2025 paper/method year bucket | 9 | TinyRecursiveModels, SOAR, CompressARC, ARC-VSA-2025, arc-lang-public, epang080516/arc_agi, ARC_NCA, ArcMemo, NVARC |
| Verified ARC Prize 2026 submission | 0 | None in the current locked inventory; a 2026 paper date is not evidence of competition entry |
| Non-ARC-Prize multi-agent research | 7 | LatentMAS, AgentPrimitives, GraphPlanner, RouteMoA, MACA, NeuroMAS, ReM-MoA |

| Native/published benchmark group | Count | Tracked methods |
| --- | ---: | --- |
| ARC-AGI-1 main evaluation | 11 | BARC, LPN, ARChitects 2024, GridCoder2024, 2D nGPT, SOAR, ARC_NCA, ArcMemo, MARC, Omni-ARC, Mini-ARC transformer |
| Documented ARC-AGI-1 and ARC-AGI-2 coverage (support and/or published results) | 5 | TinyRecursiveModels, CompressARC, ARC-VSA-2025, arc-lang-public, epang080516/arc_agi |
| ARC-AGI-2 primary with ARC-AGI-1 backtest | 1 | NVARC |
| Native non-ARC-AGI | 7 | LatentMAS, AgentPrimitives, GraphPlanner, RouteMoA, MACA, NeuroMAS, ReM-MoA |

These are provenance classifications, not local execution claims. Membership
in the 2025 bucket alone does not establish official ARC Prize participation;
the official 2025 page separately establishes the NVARC and SOAR classifications
stated above, while TinyRecursiveModels has no such evidence in the locked
snapshot. Also,
unavailable Omni-ARC, Mini-ARC, NeuroMAS, and ReM-MoA remain blocked even though
their intended benchmark family can be identified. LPN belongs to the ARC Prize
2024 cohort despite its later NeurIPS 2025 publication. “Mini-ARC transformer”
is a method name here; its reported target is an ARC-AGI-1 public-evaluation
subset, not a separate MiniARC dataset.

## Batch A: deepen existing zero-cost local smokes (5)

| Method | Current evidence | Next admissible run |
| --- | --- | --- |
| CompressARC | Compatibility smoke, single-task experiments, and a compatibility-deviant CPU method-specific A/B firewall smoke on frozen dev task `3c9b0459` | Expand the code-only label-free wrapper to a predeclared development subset; public execution remains unauthorized |
| ARC_NCA | Reduced scripted 1-step/10-step runs plus a CPU-only method-specific A/B firewall smoke on frozen dev task `6150a2bd` | Expand the frozen label-free wrapper to a predeclared same-shape development subset; public execution remains unauthorized |
| GridCoder2024 | Synthetic-weight LVM architecture forward plus a no-import static gate audit that records seven blockers and does not count as a smoke | Reconstruct verifiable upstream provenance, resolve source licensing and the missing ARC_gym APIs, then build a challenge-only adapter; checkpoint retrieval still requires explicit storage approval and a separate hash/security audit |
| 2D nGPT | Official large architecture forward with synthetic weights plus a no-import hardened static gate that records seven blockers and is not another smoke | Reconstruct verifiable upstream provenance and the fixed-shape denominator; acquire and separately hash/security-audit `exp_50.pt`/`exp_54.pt`, re-ARC data, and `fixed_size.pkl`; replace the solution-bearing TTT path with a challenge-only adapter |
| LPN | Official generator tests and encoder/decoder self-test plus a hardened no-import gate that records seven source/artifact/data/label/runtime blockers and is not another smoke | Select one of the seven source-declared W&B artifacts only after hash/config/license verification, then stage code-only source and wrap `Evaluator.json_submission` in a separate challenge-only inference process |

Promotion gate: a strict report must match the exact method and configuration,
validate every declared file, run A/B challenge trees with hidden-label mutation,
and show byte-identical inference predictions before independent post-inference
scoring can load labels. Missing outputs, timeouts, and wrong-shape outputs stay
in the denominator. A method promotion still does not authorize public execution
until the overall protocol is frozen and a new nonempty input bundle admits it.

## Batch B: published local weights, capacity-gated (2)

| Method | Exact capacity fact | Fairness boundary | Order |
| --- | --- | --- | ---: |
| ARChitects 2024 | Published 4-bit checkpoint is 3,790,920,477 bytes (3.531 GiB), but the forward preflight has `<10 GiB` free VRAM; a hardened static gate freezes eight blockers | Its locked training runner uses ARC-AGI-1 evaluation solutions, so ARC-1 is contamination-aware only. ARC-AGI-2 is only a potential new transfer benchmark after runtime gates pass | 1 |
| BARC | One source-declared BF16 8B base is about 14.97 GiB; four bases plus two LoRAs exceed 59.88 GiB and none is provenance-verified locally | The hardened static gate freezes eight root-license, base/LoRA provenance, safe-load, label-firewall, dependency, capacity, and prediction/parity blockers. The challenge-only direct-transduction path remains a design candidate | 2 |

The ARChitects 4-bit checkpoint is downloaded and hash-verified. Its forward
preflight stopped before allocation because an unrelated process leaves only
about 5 GiB free VRAM; its separate static gate is blocker evidence, not a
smoke. BARC's source/seed component smoke remains passed, and its separate
metadata-first static gate read no ARC/answer or weight worktree leaf, loaded no
model, and produced no prediction. A BARC base remains deferred until artifact,
safe-load, label-firewall, dependency, storage, and VRAM gates all pass. See the
[ARChitects gate](../reports/architects-2024/20260806-source-artifact-label-runtime-gate-v1/run.json)
and [BARC gate](../reports/barc/20260806-source-artifact-label-resource-gate-v1/run.json).

## Batch C: API-backed methods, zero-dollar gates first (3)

| Method | First zero-dollar run | Required change before any future paid request | Protocol-v1 spend authorization |
| --- | --- | --- | ---: |
| ArcMemo | Native no-memory generic-driver `dry_run=true` passed with dummy completions and network fail-closed; the memory mechanism did not run | Freeze the intended memory artifact/config, remove test labels and inline scoring, disable oracle continual updates, isolate generated code, and add fail-closed request/token/USD limits | USD 0 / not authorized |
| arc-lang-public | Import/config and Pydantic parser component passed with network disabled; this did not prove a raw-key label firewall | Freeze a challenge-only raw-key firewall/writer/scorer, make provider and logging egress lazy and closed, and enforce pre-request request/token/USD/timeout limits | USD 0 / not authorized |
| epang080516/arc_agi | Synthetic data-model plus auditor-written trusted-executor component passed; neither bundled pickle nor model-generated code ran | Remove eager truth/metrics, exclude ARC-2-library overlap from ARC-1, migrate pickle sacrificially, sandbox generated Python, and add a pre-request cost fuse | USD 0 / not authorized |

The frozen analysis plan caps API spend at USD 0 and does not authorize API
execution. Any future paid run requires explicit user authority and a prospective
protocol amendment, plus a provider/model snapshot, maximum calls, input/output
token limits, currency cap, timeout, and raw request identifiers. None of the
three root projects currently supplies a source license. Separate metadata-first
static audits freeze nine ArcMemo, eight arc-lang-public, and nine epang blockers
without importing or executing upstream code, opening restricted worktree data or
pickle leaves, making an API request, using the GPU, or producing a prediction.
They are auxiliary blocker evidence—not additional smokes or promotions. See the
[ArcMemo gate](../reports/arcmemo/20260806-source-label-memory-api-sandbox-gate-v1/run.json),
[arc-lang-public gate](../reports/arc-lang-public/20260806-source-label-api-egress-gate-v1/run.json),
and [epang gate](../reports/epang-arc-agi/20260806-source-label-pickle-sandbox-api-gate-v1/run.json).

## Batch D: heavy or integration-risk, reduced/native first (9)

| Method | Classification | First useful experiment |
| --- | --- | --- |
| TinyRecursiveModels | A random-weight 6,829,058-parameter CPU architecture forward passed; a separate static gate froze ten blockers and produced no prediction | Resolve dataset provenance/licensing, checkpoint/environment, label-firewall, selection-state, runtime, and capacity gates before any challenge-only run; full ARC training is not single-3090 parity |
| SOAR | Challenge-only ARC-AGI-1 data/helper smoke passed; a separate [formal static gate](../reports/soar/20260806-source-artifact-dataset-label-api-code-resource-gate-v1/run.json) preserved 13 blockers and produced no solver prediction | Build a label-free loader and obtain real code isolation before any candidate-program execution; only then review smallest-checkpoint capacity |
| NVARC | Gitlink/SFT-config/trusted-helper component smoke passed; a separate [formal static gate](../reports/nvarc/20260806-source-gitlink-artifact-dataset-label-code-resource-gate-v1/run.json) preserved 12 blockers and ran no ensemble component or solver | Resolve asset provenance and public-evaluation-as-validation contamination; isolate raw code execution before selecting one component |
| MARC | Synthetic ARC serialization, Top-2 submission, and fixed voting component smoke passed | Initialize the locked torchtune submodule and build a challenge-only adapter before a selected checkpoint inference |
| LatentMAS | Native AI2 ARC prompt/role schema smoke passed; no model, tokenizer, data, or latent KV loaded | Reproduce a frozen native 4B paper subset first; any ARC-AGI port is a separate new experiment |
| AgentPrimitives | Static AST/YAML/config smoke passed; upstream modules were not imported | Wait for/fix the missing runner and case-sensitive imports, then define a native component run; no end-to-end benchmark is currently admissible |
| GraphPlanner | Bundled router-data schema and pure prompt component smoke passed without API | Obtain a licensed checkpoint or predeclare a capped native AI2 ARC run; an ARC-AGI adaptation remains separate |
| RouteMoA | Repository syntax audit failed; labeled precomputed scorer-only auxiliary audit passed and is not counted as a smoke | Repair/lock an executable evaluator interface, obtain router/model pool, and predeclare native service budget before solver execution |
| MACA | Random-weight CPU GraphSpec component smoke passed; repository-wide syntax audit failed | A trained checkpoint, license, implemented backend/data adapter, and documented benchmark command are required |

These methods first answer whether the released artifact is internally
executable. Native non-ARC results and later ARC adaptations are separate
experiments. Reduced one-GPU runs never inherit a paper-level claim. The seven
new component smokes above count toward the 17/24 total; RouteMoA's scorer-only
auxiliary audit does not. No Batch D public benchmark has passed.

TinyRecursiveModels now also has a reproducible metadata-first static gate. It
byte-bound 28 retained text leaves while leaving ten ARC JSON and two image
leaves metadata-only, imported/executed no upstream code, loaded no checkpoint,
and produced no prediction. Its audit passed while the method remained blocked
on ten gates, including label persistence, cross-benchmark overlap, checkpoint
and RNG/evaluator state, dependency/egress, runtime, capacity, and no-prediction
requirements. The 45%/8% README values remain unverified self-reports, not
scores in this evaluation. See the
[TRM gate](../reports/tiny-recursive-models/20260806-source-artifact-dataset-label-resource-gate-v1/run.json).

SOAR and NVARC now also have formal metadata-first static-gate records. The
[SOAR gate](../reports/soar/20260806-source-artifact-dataset-label-api-code-resource-gate-v1/run.json)
passed its audit while leaving the method blocked on 13 source, artifact, data,
label, API, generated-code, resource, and parity gates. The
[NVARC gate](../reports/nvarc/20260806-source-gitlink-artifact-dataset-label-code-resource-gate-v1/run.json)
passed its audit while leaving the method blocked on 12 source/gitlink,
artifact, data, label, code-isolation, resource, and parity gates. Neither gate
is another smoke: neither imported or executed the upstream method, produced a
solver prediction, granted strict-runtime promotion, established performance
eligibility, or reproduced the external official 2025 result or award.

## Batch E: blocked implementation/watchlist (5)

| Method | Blocking fact | Re-entry condition |
| --- | --- | --- |
| ARC-VSA-2025 | A locked-source blocker audit confirmed missing/unimportable `sspspace`; the released solver constructor also reads `pair["output"]` for test examples | Authors release the dependency and a label-free path, or a clearly labeled new adaptation is designed |
| Omni-ARC | No verified public runnable implementation | Verifiable author source release |
| Mini-ARC transformer | No verified public runnable implementation | Verifiable implementation and data contract |
| NeuroMAS | No verified public implementation | Author code/checkpoint release |
| ReM-MoA | No verified public implementation | Author code/checkpoint release |

A blocked entry remains in every denominator and audit table. It is not replaced
with a similarly named third-party implementation.

## Global execution order

1. Preserve the implemented E0 scorer/firewall/resource/process gates and the
   recorded host-level isolation blocker. Use the frozen 94-task/97-output
   known-overlap-excluded dev-audit runtime for method development. Close the sole
   remaining required protocol-root gate—child-inclusive process-tree resource
   accounting—and freeze protocol v1 before new locked public evaluation.
2. Deepen Batch A only on development/training tasks.
3. Run ARChitects, then BARC, subject to the capacity and contamination gates.
4. Complete every Batch C zero-dollar smoke and label-free adapter. Protocol v1
   authorizes USD 0; do not request or spend an API budget without explicit user
   authority and a prospective amendment.
5. Process Batch D one repository at a time, starting with source-only and
   native-benchmark evidence.
6. Re-audit Batch E when upstream state changes.

ARC-AGI-1 and ARC-AGI-2 are always scored and reported separately. A source
audit, component smoke, solver-prediction smoke, single-task run, fixed-subset
benchmark, full public benchmark, and paper reproduction are seven distinct
evidence levels.

The cross-benchmark audit also found 376 shared task IDs between ARC-AGI-1
evaluation and ARC-AGI-2 training; 375 complete labeled tasks match after
normalizing example order, and all 376 test I/O sets match. A checkpoint trained
on ARC-AGI-2 training is therefore ineligible for a clean ARC-AGI-1 evaluation
score unless every overlapping task was excluded before training and that
exclusion is auditable.
