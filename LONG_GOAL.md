# Long Goal: Reproduce And Compare 24 Baselines

## Objective

Build a durable, auditable body of evidence for the 24 tracked methods without
conflating source availability, smoke execution, benchmark execution, and paper
reproduction. The work targets one NVIDIA RTX 5090 with 32 GiB VRAM and about
41 GiB free storage. Results must remain useful when upstream branches, hosted
models, APIs, and hardware change.

The source audit is captured in
[`docs/REPRODUCTION_MATRIX.md`](docs/REPRODUCTION_MATRIX.md), and the canonical
machine-readable inventory is [`configs/baselines.json`](configs/baselines.json).

## Truthful Status

As of 2026-08-04:

- The local ARC-AGI evaluator validates all 1,920 vendored ARC-AGI-1/2 tasks,
  and its 29 tests pass.
- Repository URLs and default branches have been audited for 19 public
  candidates and the partial/complex MARC candidate.
- Omni-ARC, the Mini-ARC transformer implementation, NeuroMAS, and ReM-MoA have
  no verified public runnable implementation in this audit.
- Pinned, storage-filtered source snapshots exist for CompressARC and
  ARC-VSA-2025. No model checkpoint was downloaded.
- The deterministic local floor baseline has complete ARC-AGI-1/2 evaluation
  artifacts. It solves 0 exact tasks and reaches 0.74394/0.68541 cell accuracy.
- CompressARC has one passing single-task forward smoke in the existing
  Blackwell-compatible environment. ARC-VSA-2025 is blocked before solver
  import because upstream omits the required `sspspace` implementation.
- Zero paper benchmark or full-reproduction claims have been made. One method
  has passed a compatibility smoke only.

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

## Phases

### Phase 0: Freeze The Evaluation Contract

- Validate the vendored ARC data and record its source revisions.
- Define immutable task lists for smoke, reduced benchmark, and full public
  benchmark runs. Include every test output for selected tasks.
- Freeze Top-K, exact-task accuracy, output accuracy, cell accuracy, timeout,
  and malformed-output behavior.
- Create report and artifact directories only when the first run is approved.
- Record host details: GPU model, driver, CUDA runtime, CPU, RAM, filesystems,
  free bytes, and container/runtime versions.

Exit criterion: a prediction file can be scored twice with identical results,
and the run schema below has one synthetic example record.

### Phase 1: Low-Risk Local Smokes

Run CompressARC, ARC-VSA-2025, ARC_NCA, GridCoder2024, 2D nGPT, and LPN in the
matrix order. Start with source and environment checks, then one immutable task
or upstream self-test. Do not fetch a checkpoint until its expected size and
storage location have been approved.

Exit criterion: each method has a passing smoke or a preserved failure/blocker
report, measured disk/VRAM/runtime, and an output normalization decision.

### Phase 2: ARC Model And API Baselines

Run ARChitects 2024 and BARC with the smallest published checkpoint that
preserves the method. Then evaluate arc-lang-public, epang080516/arc_agi, and
ArcMemo under explicit API budgets. API methods require a dry-run cost estimate
and a hard cap before benchmark execution.

Exit criterion: each affordable method has a fixed-subset benchmark, and every
API report includes provider/model ID, request parameters, token usage, spend,
and UTC timestamps.

### Phase 3: Heavy And Complex Methods

Review TinyRecursiveModels, SOAR, NVARC, MARC, LatentMAS, AgentPrimitives,
GraphPlanner, RouteMoA, and MACA one at a time. Approve a reduced configuration
only when its scientific question is written first. Do not call a one-GPU port
of a multi-GPU paper setup a full reproduction.

Exit criterion: each entry has either a useful reduced benchmark with explicit
differences or a quantified blocker. Full reproduction is optional when the
paper budget cannot be matched; truthful infeasibility is an accepted outcome.

### Phase 4: Blocked Watchlist And Synthesis

Recheck unavailable sources quarterly. Consolidate results across compatible
compute classes, publish failure analysis, and update estimates with measured
resources. Keep blocked rows rather than deleting them.

Exit criterion: all 24 entries satisfy the global acceptance criteria and the
comparison contains no unsupported paper-parity claim.

## Compute-Matched Reporting

Every run receives one comparison class:

- `paper-exact`: paper code/artifacts, data, budget, and accelerator topology.
- `paper-equivalent`: same effective model/data/search budget on equivalent
  hardware, with the equivalence argument recorded.
- `local-5090`: declared run designed for one 32 GiB RTX 5090.
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

The machine currently has only about 41 GiB free, so storage approval precedes
network access.

1. Check whether `/model` exists, is mounted, is writable, and already contains
   the exact model/dataset before downloading anything. Verify identity by
   revision and hash; do not assume matching filenames mean matching content.
2. Record `df` free bytes for `/model` and the workspace filesystem before and
   after every artifact operation. Keep a minimum 8 GiB reserve, so a plan that
   can add more than 33 GiB requires cleanup or explicit approval first.
3. If `/model` is suitable, use namespaced paths such as
   `/model/arc-agi/models/<provider>/<name>/<revision>` and
   `/model/arc-agi/datasets/<provider>/<name>/<revision>`. Point
   `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, Torch, JAX, and other
   caches there rather than duplicating artifacts in each checkout.
4. If `/model` is absent or unsuitable, stop before large downloads. Do not
   silently place model weights in the repository, home directory, `/tmp`, or
   the 41 GiB workspace filesystem.
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
- Does the projected peak VRAM fit 32 GiB with a safety margin?
- Are dataset split, task IDs, Top-K, seed, timeout, and score command frozen?
- For an API, are provider/model/date, expected tokens, estimated cost, and a
  hard cap recorded?
- Is the result label honest about every difference from the paper?

If any answer is no, the run remains `blocked` or `not-started`; reducing the
configuration requires a new declared run, not an undocumented workaround.
