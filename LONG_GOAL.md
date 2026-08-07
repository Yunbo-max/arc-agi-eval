# Long Goal: Reproduce And Compare 24 Baselines

## Objective

Build a durable, auditable body of evidence for the 24 tracked methods without
conflating source availability, smoke execution, benchmark execution, and paper
reproduction. The active target is one NVIDIA RTX 3090 with 24 GiB VRAM and
about 24.3 GiB free storage as of the latest persisted check. Results must remain useful when upstream branches, hosted
models, APIs, and hardware change.

The source audit is captured in
[`docs/REPRODUCTION_MATRIX.md`](docs/REPRODUCTION_MATRIX.md), and the canonical
machine-readable inventory is [`configs/baselines.json`](configs/baselines.json).
The current five-batch execution split is
[`docs/EXECUTION_BATCHES.md`](docs/EXECUTION_BATCHES.md).

## Truthful Status

As of 2026-08-06:

- The local ARC-AGI evaluator validates all 1,920 vendored ARC-AGI-1/2 tasks.
  Challenge/reference scoring, IsoARC, process lifecycle, current-process
  resource monitoring/calibration, and malformed/timeout/label-mutation cases
  are implemented. The additive scorer contract now makes output-level exact
  pass@K primary and strict task exact secondary. Every public task now has an
  immutable per-file manifest, and the synthetic E0 contracts have terminal
  evidence. True namespace/container isolation is unavailable and protocol v1
  is not frozen.
- A deterministic training-only development draft now groups 1,400 ARC-AGI-1/2
  training records into 1,008 verified clusters and assigns 706/151/151 to
  dev-build/dev-select/dev-audit. The existing overlap ledger flags 376
  clusters, so this general view is contamination-aware for ARC-AGI-1. A
  derivative removes all 376 flagged clusters / 377 records without
  reallocation, leaving 632 clusters / 1,023 records. A deterministic 94-task /
  97-output dev-audit runtime is frozen from that view; this is a known-overlap
  exclusion rather than proof of absolute cleanliness
  ([evidence](reports/e0-development-split/20260806-frozen-known-overlap-excluded-dev-audit-v1/run.json)).
- Repository URLs and default branches have been audited for 19 public
  candidates and the partial/complex MARC candidate.
- Omni-ARC, the Mini-ARC transformer implementation, NeuroMAS, and ReM-MoA have
  no verified public runnable implementation in this audit.
- The deterministic local floor baseline has complete ARC-AGI-1/2 evaluation
  artifacts. It solves 0 exact tasks and reaches 0.74394/0.68541 cell accuracy.
- Seventeen of 24 methods have a passing but scope-limited compatibility,
  architecture, component, or dry-run smoke. The seven latest additions are
  SOAR, NVARC, MARC, LatentMAS, AgentPrimitives, GraphPlanner, and MACA. These
  are source/data/config, schema, serialization/voting, or random-weight
  component checks; several are native non-ARC-AGI methods. This count does not
  mean 17 ARC solvers have run. RouteMoA's labeled precomputed scorer audit is
  auxiliary evidence and is deliberately excluded.
- GridCoder2024, 2D nGPT, LPN, ARChitects, BARC, TinyRecursiveModels, SOAR,
  NVARC, ArcMemo, arc-lang-public, and epang080516/arc_agi now also have
  hardened static blocker audits. Each audit is explicitly excluded from the
  17-smoke count and from strict/runtime promotion. SOAR and NVARC preserve 13
  and 12 blockers respectively; their formal reports, configs, and runner
  manifests are anchored by input-bundle retry16.
  ARChitects records ARC-AGI-1 training contamination, solution-bearing local
  runners, runtime/dependency/capacity gaps, and no prediction. BARC separates
  its root-license, base/LoRA provenance, safe-load, label, dependency,
  capacity, and prediction/parity gates without reading ARC/answer or weight
  worktree leaves. The Batch C audits correct ArcMemo's old dry-run scope to a
  no-memory generic driver, show that arc-lang's parser smoke did not prove a
  raw-key firewall, and separate epang's synthetic trusted component from its
  untrusted pickle and generated-code paths
  ([GridCoder](reports/gridcoder2024/20260806-source-dependency-label-artifact-gate-v3/run.json),
  [2D nGPT](reports/2d-ngpt/20260806-source-artifact-label-runtime-gate-v1/run.json),
  [LPN](reports/lpn/20260806-source-artifact-data-label-gate-v1/run.json),
  [ARChitects](reports/architects-2024/20260806-source-artifact-label-runtime-gate-v1/run.json),
  [BARC](reports/barc/20260806-source-artifact-label-resource-gate-v1/run.json),
  [TRM](reports/tiny-recursive-models/20260806-source-artifact-dataset-label-resource-gate-v1/run.json),
  [SOAR](reports/soar/20260806-source-artifact-dataset-label-api-code-resource-gate-v1/run.json),
  [NVARC](reports/nvarc/20260806-source-gitlink-artifact-dataset-label-code-resource-gate-v1/run.json),
  [ArcMemo](reports/arcmemo/20260806-source-label-memory-api-sandbox-gate-v1/run.json),
  [arc-lang](reports/arc-lang-public/20260806-source-label-api-egress-gate-v1/run.json),
  [epang](reports/epang-arc-agi/20260806-source-label-pickle-sandbox-api-gate-v1/run.json)).
- The configuration-level audit finds only two legacy solver-prediction smokes
  (CompressARC and ARC_NCA). Both now have method-specific strict runtime
  promotions after reduced CPU-only A/B firewall smokes; zero of 24
  methods are eligible for a performance table
  ([eligibility](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry12/run.json),
  [CompressARC](reports/compressarc/20260806-cpu-dev-3c9b0459-strict-v1/run.json),
  [ARC_NCA](reports/arc-nca/20260806-cpu-dev-6150a2bd-strict-v1/run.json)).
- The exact published ARChitects 4-bit checkpoint is downloaded and
  hash-audited, but its forward smoke stopped at the free-VRAM preflight. Its
  model card also identifies ARC-AGI-1 public evaluation as training data, so
  ARC-AGI-1 use is contamination-aware or historical, not clean held-out
  evaluation.
- ARC-AGI-1 evaluation and ARC-AGI-2 training share 376 task IDs. Of these, 375
  are semantically identical complete labeled tasks and all 376 have
  semantically identical test I/O.
- Zero declared public benchmarks and zero paper-level reproductions have been
  completed. The machine-generated funnel audit validates all 17 primary
  passing evidence records plus ten non-promoting auxiliary records
  ([evidence](reports/e0-reproduction-funnel/20260806-manifest-funnel-audit-retry9/run.json)).
- A strict new-run schema now binds schema/protocol/source/artifact/data/config/
  environment/hardware digests to verified files. The refreshed non-circular
  prior-exposure cutoff inventories the current in-scope run records and four
  result artifacts. The latest protocol-v1 draft root passes its consistency audit and
  reports 14 passed, one required pending, two optional blocked, and
  `freeze_ready=false`.
  The associated zero-admit bundle binds 49 declared inputs and 125 native code
  files; the exposure inventory records 263 runs, four result artifacts, and no
  private workspace record
  ([protocol](reports/e0-protocol/20260806-protocol-v1-draft-root-retry16/run.json),
  [bundle](reports/e0-freeze/20260806-input-bundle-v1-retry16/run.json),
  [exposure](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry16/run.json)).

Future status updates must identify the run evidence that caused the change.
Do not mark a row `passed` from README inspection, a published score, or a
third-party claim.

## Acceptance Criteria

The long goal is complete only when all of the following are true:

1. All 24 entries remain represented in both the matrix and manifest with a
   stable ID, source or reference URL, availability state, and blocker state.
2. Every accessible repository used in a run is locked to a full commit SHA.
   External model and dataset artifacts have immutable revisions where the host
   supports them, plus file sizes and SHA-256 hashes for downloaded files.
3. Every public candidate has either a passing smoke run or a failure report
   containing the exact command, environment, logs, and actionable blocker.
4. Every feasible candidate has a benchmark run on a predeclared split or
   immutable task list. Infeasible candidates have a quantified resource,
   access, or missing-artifact justification.
5. Any result labeled `full-reproduction` matches the paper's data, model,
   adaptation/search budget, scoring, and compute class, and states a numerical
   tolerance before execution. Otherwise it is labeled `reduced`,
   `checkpoint-only`, `scorer-only`, `API-replication`, or `source-only`.
6. ARC result artifacts can be validated and rescored by the local evaluator.
   Native non-ARC papers are evaluated with their published native protocol;
   an ARC adaptation is reported separately as a new experiment.
7. Reports include successes, wrong answers, missing outputs, timeouts, OOMs,
   retries, filtered tasks, and manual interventions. Denominators are never
   reduced after seeing results.
8. At least 8 GiB remains free on the active filesystem throughout execution.
   Large artifacts are not committed to Git and are not duplicated when a
   suitable copy already exists under `/model`.
9. Secrets, API keys, provider credentials, and licensed/gated model files are
   absent from reports and Git history.
10. The final comparison separates measured values from estimates and compares
    only runs with compatible split, metric, attempt budget, and compute class.
11. A checkpoint exposed to ARC-AGI-2 training is not reported as clean on
    ARC-AGI-1 evaluation unless all overlapping tasks are excluded under a
    predeclared, auditable denominator. Otherwise the result is explicitly
    contamination-aware or historical.

## Status State Machine

Each level moves independently through:

`not-started -> running -> passed | failed | blocked`

- `source-audited` describes metadata only and is not an execution state.
- `failed` means the declared command ran and did not satisfy its acceptance
  check; preserve the evidence before fixing it.
- `blocked` means execution cannot start because an artifact, access grant,
  budget, platform, or instruction is absent.
- A repaired failure creates a new run record. Never overwrite the failed run.
- A status can regress if an upstream artifact disappears or a prior result is
  found invalid; record the reason and date.

## Foundation And Execution Batches

### Foundation Gate

- Validate the vendored ARC data and record its source revisions.
- Define immutable task lists for smoke, reduced benchmark, and full public
  benchmark runs. Include every test output for selected tasks.
- Freeze Top-K, exact-task accuracy, output accuracy, cell accuracy, timeout,
  and malformed-output behavior.
- Create report and artifact directories only when the first run is approved.
- Record host details: GPU model, driver, CUDA runtime, CPU, RAM, filesystems,
  free bytes, and container/runtime versions.

Challenge/reference separation, IsoARC, process lifecycle, current-process
resource monitoring/calibration, and malformed/timeout/label-mutation cases are
implemented. The output-level-primary score contract is also implemented and
independently audited against the canonical deterministic-floor predictions.
The full public data manifest, label-free 520-task public challenge view,
94-task/97-output development runtime, strict new-run schema, global trusted
challenge runtime, fixed64 IsoARC design, analysis plan, zero-admit input bundle,
prior-exposure cutoff, and protocol draft root are now materialized. Global
runtime evidence does not promote a method by itself; ARC_NCA and CompressARC
separately passed method/config-specific strict smokes, so strict promotion is 2/24 while
performance eligibility and public admission remain 0/24. The only unmet
required root gate is child-inclusive process-tree resource accounting. True
filesystem/network namespace isolation also remains unavailable and blocks
generated/untrusted execution, although it is optional for a trusted-code-only
protocol freeze.

### Batch A: Deepen Existing Low-Cost Smokes (5)

CompressARC, ARC_NCA, GridCoder2024, 2D nGPT, and LPN each have some passing
scope-limited evidence. Advance from compatibility/architecture checks to an
immutable task or declared subset only when the needed checkpoint/data exists
and outputs can be normalized independently.

Exit criterion: each method has a passing smoke or a preserved failure/blocker
report, measured disk/VRAM/runtime, and an output normalization decision.

### Batch B: Published Local Weights, Capacity-Gated (2)

ARChitects 2024 has an integrity-checked 4-bit checkpoint but is waiting for the
10 GiB free-VRAM gate; its hardened static audit freezes eight blockers and
confirms ARC-AGI-1 contamination. BARC's bundled seed/program component smoke
still passes, while a separate hardened static gate freezes eight source,
artifact, safe-load, label, dependency, capacity, and prediction/parity
blockers; its selected BF16 base remains deferred. Neither static audit is a
solver prediction, promotion, benchmark, or paper reproduction.

### Batch C: API-Backed, Zero-Dollar First (3)

ArcMemo, arc-lang-public, and epang080516/arc_agi have passed network-guarded,
zero-dollar component or dry-run checks. ArcMemo's pass was a no-memory generic
driver with dummy completions, arc-lang's was an import/config/Pydantic parser
component rather than a raw-key firewall, and epang's used synthetic data plus
auditor-written trusted code rather than its pickle or model-generated code.
Separate non-promoting static gates freeze their label, artifact, egress,
contamination, sandbox, dependency, native-contract, and no-prediction blockers.
Benchmark execution remains blocked until challenge-only adapters, explicit API
approval, immutable provider/model records, hard request/token/currency caps,
and generated-code isolation exist.

Protocol v1 currently freezes API spend at USD 0 and records API execution as
unauthorized. Any paid run requires explicit user authority and a prospective
protocol amendment before requests are made.

Exit criterion: each affordable method has a fixed-subset benchmark, and every
API report includes provider/model ID, request parameters, token usage, spend,
and UTC timestamps.

### Batch D: Heavy Or Integration-Risk (9)

Review TinyRecursiveModels, SOAR, NVARC, MARC, LatentMAS, AgentPrimitives,
GraphPlanner, RouteMoA, and MACA one at a time. Approve a reduced configuration
only when its scientific question is written first. Do not call a one-GPU port
of a multi-GPU paper setup a full reproduction.

All nine now have at least source or auxiliary evidence. Eight have a passing
scope-limited smoke: TinyRecursiveModels plus component checks for SOAR, NVARC,
MARC, LatentMAS, AgentPrimitives, GraphPlanner, and MACA. RouteMoA instead has a
preserved repository syntax failure and a separate labeled precomputed
scorer-only audit, so its smoke remains `not_started`. No Batch D public
benchmark or paper-level reproduction has passed.

TinyRecursiveModels also has a passing static blocker audit that is excluded
from the smoke count. It leaves the method blocked on ten gates and explicitly
produces no solver prediction, score, checkpoint inference, benchmark, or paper
reproduction. Its 2025 classification is a paper/method year bucket, the bound
asset snapshot is from 2026, and no official ARC Prize entry is verified.

Exit criterion: each entry has either a useful reduced benchmark with explicit
differences or a quantified blocker. Full reproduction is optional when the
paper budget cannot be matched; truthful infeasibility is an accepted outcome.

### Batch E: Blocked Watchlist (5)

ARC-VSA-2025, Omni-ARC, the Mini-ARC transformer implementation, NeuroMAS, and
ReM-MoA remain blocked by label dependence/missing dependency or unavailable
verified implementations. Recheck sources periodically, and keep blocked rows
rather than deleting them.

Exit criterion: all 24 entries satisfy the global acceptance criteria and the
comparison contains no unsupported paper-parity claim.

## Compute-Matched Reporting

Every run receives one comparison class:

- `paper-exact`: paper code/artifacts, data, budget, and accelerator topology.
- `paper-equivalent`: same effective model/data/search budget on equivalent
  hardware, with the equivalence argument recorded.
- `local-3090`: declared run designed for the active 24 GiB RTX 3090.
- `local-5090`: legacy evidence produced on the earlier 32 GiB RTX 5090; never
  pool it with `local-3090` as a repeated trial.
- `reduced`: smaller model, data, training, search, attempt, or time budget.
- `api-replication`: hosted model execution with provider and UTC date pinned.
- `scorer-only`: published predictions evaluated without rerunning inference.
- `source-only`: install/import/static checks without model execution.

Only compare headline scores within the same benchmark version, split,
denominator, metric, Top-K, and comparison class. If wall time is matched, also
report tokens, model calls, generated candidates, training steps, and energy
when available; equal wall time alone is not equal compute.

## Run Report Schema

Store one immutable `run.json` per attempt under
`reports/<baseline-id>/<run-id>/`. Logs, predictions, metrics, and environment
locks sit beside it and are referenced by relative path and SHA-256. A run
record contains at least:

| Field | Required content |
| --- | --- |
| `schema_version`, `run_id`, `created_at_utc` | Version, unique ID, and UTC timestamp |
| `baseline_id`, `level`, `comparison_class`, `status` | Manifest ID; smoke/benchmark/full; compute class; terminal state |
| `source` | Repository URL, commit SHA, dirty flag, submodule SHAs, patch hash |
| `paper_target` | Paper/table/row, expected metric, value, and predeclared tolerance, or `null` |
| `hardware` | GPU/VRAM, driver/CUDA, CPU, RAM, filesystem, container |
| `environment` | Python version, lockfile hash, installed-package export, relevant environment variables with secrets redacted |
| `artifacts` | Model/data IDs, immutable revisions, local paths, sizes, licenses/gates, SHA-256 |
| `dataset` | Benchmark generation, split, task-list hash, task/output counts, contamination policy |
| `method` | Model, precision, quantization, seed, batch size, adaptation/training/search configuration |
| `attempt_budget` | Top-K, samples, agents, rounds, candidates, API calls, token cap, timeout |
| `command` | Exact non-secret invocation and working directory |
| `resources` | Wall and GPU time, peak VRAM, RAM, disk delta, input/output tokens, API spend, energy if measured |
| `results` | Exact task/output/cell metrics with numerators and denominators, plus native paper metrics |
| `failures` | Missing/malformed outputs, timeouts, OOMs, retries, excluded-before-run tasks, manual interventions |
| `files` | Relative paths and hashes for stdout/stderr, predictions, scorer output, and checkpoints created by the run |
| `notes` | Known deviations and interpretation; never use this field to hide a protocol change |

For stochastic methods, report all seeds individually plus mean, standard
deviation, and confidence interval when enough runs exist. For APIs, save raw
responses where terms permit and always save request IDs and usage metadata.

## Storage Rules

The machine currently has only about 24.3 GiB free, so storage approval precedes
network access.

1. Check whether `/model` exists, is mounted, is writable, and already contains
   the exact model/dataset before downloading anything. Verify identity by
   revision and hash; do not assume matching filenames mean matching content.
2. Record `df` free bytes for `/model` and the workspace filesystem before and
   after every artifact operation. Keep a minimum 8 GiB reserve, so a plan that
   can add more than about 16.3 GiB requires additional capacity or explicit
   approval first.
3. If `/model` is suitable, use namespaced paths such as
   `/model/arc-agi/models/<provider>/<name>/<revision>` and
   `/model/arc-agi/datasets/<provider>/<name>/<revision>`. Point
   `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, Torch, JAX, and other
   caches there rather than duplicating artifacts in each checkout.
4. `/model` is currently absent. Do not duplicate the existing audited
   ARChitects checkpoint, and do not silently place another large model in the
   repository, home directory, or `/tmp`; every exception needs a recorded
   capacity preflight and immutable artifact identity.
5. Repository checkouts and per-method virtual environments are disposable;
   source locks and dependency exports are durable. One environment per method
   is preferred to resolving incompatible historical stacks into one mutable
   environment.
6. Never commit checkpoints, generated corpora, caches, raw API secrets, large
   predictions, or third-party repositories. Commit only small configs,
   manifests, reports, hashes, and summaries permitted by licenses.
7. Before deleting a run-created artifact, verify its hash is recorded and all
   derived metrics can be traced to it. Never delete or replace another user's
   artifact without explicit approval.

## Execution Gate

Before each smoke or benchmark, answer all of these in the run record:

- Is the source pinned to a commit and are submodules pinned?
- Is the artifact license/access condition acceptable?
- Was `/model` checked first, and is the projected disk delta below the reserve?
- Does the projected peak VRAM fit 24 GiB with a safety margin and the measured
  free VRAM at launch time?
- Are dataset split, task IDs, Top-K, seed, timeout, and score command frozen?
- For an API, are provider/model/date, expected tokens, estimated cost, and a
  hard cap recorded?
- Is the result label honest about every difference from the paper?

If any answer is no, the run remains `blocked` or `not-started`; reducing the
configuration requires a new declared run, not an undocumented workaround.
