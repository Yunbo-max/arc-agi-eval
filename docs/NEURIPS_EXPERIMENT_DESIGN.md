# ARC-REBench: NeurIPS-Level Experimental Design

- Status: design version 0.2, not preregistered and not yet frozen
- Date: 2026-08-04
- Evaluation standard: NeurIPS Evaluations & Datasets (E&D); the next eligible
  venue cycle must be selected before protocol freeze
- Scope: experimental protocol only; this document makes no empirical claim

Operational status: this design draft is superseded by
[`ANALYSIS_PLAN_V1.md`](ANALYSIS_PLAN_V1.md) and the current
[`EXECUTION_BATCHES.md`](EXECUTION_BATCHES.md); it is retained as design
history. The current protocol remains draft-not-frozen because process-tree
resource accounting is pending, and locked-public execution is unauthorized.
CompressARC and ARC_NCA now have reduced method-specific strict runtime
promotions, but neither is performance-eligible.

## 1. Paper thesis

The paper must not be framed as a leaderboard assembled from incomparable
published numbers. Its central scientific question is:

> Which apparent differences between ARC solvers remain after evaluation is
> label-isolated, failure-complete, compute-matched, and stress-tested under
> semantics-preserving task isomorphisms?

The working title is:

> **ARC-REBench: Auditable, Compute-Normalized, and Isomorphism-Robust
> Evaluation of ARC Reasoning Systems**

The intended contribution is an evaluation study and executable protocol, not
a claim that a new solver achieves state of the art. This is aligned with the
NeurIPS 2026 E&D scope, which explicitly includes rigorous reproduction,
auditing, stress-testing, negative results, and studies showing how evaluation
assumptions alter scientific conclusions.

The paper is publishable only if it produces at least one conclusion that is
not recoverable from a conventional score table. The primary candidate is the
joint analysis of:

1. full-denominator pass@2 accuracy;
2. online compute and energy scaling;
3. robustness to task isomorphisms;
4. reproducibility level and failure behavior.

## 2. Intended contributions

### C1. Auditable reproduction census

Build a complete evidence-backed census of the 24 tracked methods. Separate
source availability, environment construction, smoke execution, benchmark
execution, checkpoint replication, and full paper reproduction. A README or
published score is not execution evidence.

### C2. Compute-matched ARC comparison

Compare eligible ARC-native systems under a fixed Top-2 protocol and common
online resource budgets. Offline training compute, per-task adaptation compute,
inference compute, API usage, and human intervention are reported separately.
Methods from incompatible compute classes are not collapsed into one ranking.

### C3. Isomorphism robustness evaluation

Measure whether a solver behaves consistently under task transformations that
preserve the abstract problem: grid dihedral transforms, non-background color
renaming, and demonstration ordering. This tests robustness to representation
without treating transformed public tasks as new private tasks.

### C4. Failure-complete and uncertainty-aware reporting

Count timeouts, OOMs, malformed outputs, missing tasks, API failures, and
manual interventions in the declared denominator. Report paired uncertainty,
seed sensitivity, and multiplicity-corrected comparisons.

## 3. Claims that are explicitly out of scope

- Public evaluation performance is not evidence of uncontaminated
  generalization.
- A source checkout is not a reproduced method.
- A forward smoke is not a benchmark result.
- A checkpoint run is not a training reproduction.
- A reduced one-GPU port is not paper-equivalent unless equivalence is
  justified before execution.
- Native non-ARC multi-agent papers are not compared on the ARC leaderboard
  unless a separately declared ARC adaptation is implemented.
- Isomorphism robustness is a diagnostic for representation sensitivity, not
  proof that a model memorized benchmark answers.
- Proprietary API reruns are timestamped replications, not immutable paper
  reproductions.

## 4. Research questions and prespecified hypotheses

The public ARC evaluation labels and some historical aggregate outcomes were
available before this protocol existed. The public analyses therefore cannot
be made retrospectively label-naive. The analysis plan must instead be frozen
before any **new solver outputs** are scored, and the paper must call these
prespecified, protocol-locked public analyses rather than pristine holdout
confirmation. Only a previously unseen evaluator-held or official private set
can provide external confirmation.

### RQ1. Does method ranking depend on online compute budget?

**H1:** Accuracy scaling differs by solver configuration. In a preregistered
task-clustered logistic model, the configuration-by-log-budget interaction is
non-zero. The prespecified primary test is the global interaction test;
pairwise slope contrasts are secondary and Holm-corrected. Mechanism-family
aggregation is descriptive unless at least two independently developed
configurations from each compared family are eligible.

This hypothesis can fail: rankings may be stable and scaling slopes may be
indistinguishable.

### RQ2. Are solver predictions robust to task isomorphisms?

**H2:** Accuracy is not invariant to the prespecified transformation condition.
The primary test is an omnibus task-clustered test of transformation and
method-by-transformation terms. A stronger claim of robustness loss additionally
requires at least one negative original-versus-transformed method contrast with
a task-clustered 95% confidence interval excluding zero after Holm correction.

The paper must report the result even if every method is robust or the
transformation effect is statistically inconclusive.

### RQ3. Does method diversity produce useful Top-2 complementarity?

**H3:** A cross-family two-attempt portfolio selected only on `dev-select`
outperforms the strongest same-family two-attempt portfolio at the same total
online compute budget. Each member receives half of budget `B` and contributes
its Top-1 prediction; both portfolios therefore produce exactly two attempts.
Selection, tie-breaking, and fallback order are frozen before evaluation. A
single-system control receives all of `B` and emits its native Top-2. The
primary comparison is paired at the base-task level.

H3 is declared infeasible before evaluation if no family has two eligible
systems. It is not repaired after outcomes are seen by changing the taxonomy
or unequal compute allocation.

An oracle union may be reported only as an upper bound. It is never a submitted
method and is never used for the prespecified H3 test.

### Descriptive research question: how reproducible is the literature?

The 24-method reproduction funnel is descriptive because the sample is a
curated finite set, not a random sample of all research. Report exact counts,
fractions, and blockers; do not attach sampling intervals or make
population-level causal claims.

## 5. Method taxonomy and eligibility

### 5.1 Fixed families

Taxonomy is assigned before benchmark outcomes are known.

| Family | Candidate examples | Defining mechanism |
|---|---|---|
| Deterministic floors | copy/input, dominant color, geometric/color baseline | no learned ARC solver |
| Symbolic/program search | GridCoder2024, ARC-VSA-2025, arc-lang-public, epang-arc-agi | explicit programs, DSLs, or symbolic search |
| Task-specific neural adaptation | CompressARC, ARC_NCA, LPN, 2D nGPT | optimization or adaptation on each task |
| Pretrained neural/LLM solver | ARChitects, BARC, TinyRecursiveModels, SOAR, MARC | pretrained representation or language model |
| Ensemble | NVARC | combines independently trained solver components |
| API-native ARC solver | ArcMemo and API configurations above | hosted model is essential to the method |

Methods may have hybrid mechanisms, but every reported configuration receives
one primary family label in the frozen manifest.

LatentMAS, AgentPrimitives, GraphPlanner, RouteMoA, MACA, NeuroMAS, and ReM-MoA
are evaluated first on their published native benchmarks. Any ARC adaptation is
an explicitly new experiment and is excluded from paper-parity comparisons.

### 5.2 Benchmark eligibility gate

A configuration enters the performance tables only if all answers are `yes`:

1. Is the exact source revision locked, including recursive submodule SHAs?
2. Is execution permitted by the code, data, and model licenses?
3. Are all required artifacts pinned by immutable revision and verified hash?
4. Does an isolated environment pass dependency and import checks?
5. Does a no-label smoke produce a schema-valid prediction?
6. Is the configuration class declared (`paper-exact`, `paper-equivalent`,
   `local-24g`, `local-32g`, `reduced`, `API-replication`, or
   `checkpoint-only`)?
7. Can the method obey the declared timeout and emit a best-so-far prediction?
8. Are task state, cache persistence, retry behavior, and seed behavior fixed?

Failure at this gate remains a scientifically useful reproduction result, but
the method stays out of the performance ranking. Missing implementations are
not replaced by unofficial third-party systems.

### 5.3 Minimum breadth gate for the headline comparison

The compute-matched comparison is a headline contribution only if at least six
ARC-native systems from at least four families pass the gate and cover at least
95% of the frozen task denominator without human correction. Otherwise the
paper is framed as a reproduction/evaluation audit, and performance comparisons
are explicitly exploratory.

## 6. Data protocol and contamination controls

### 6.1 Dataset roles

| Dataset | Role | Permitted use |
|---|---|---|
| ARC-AGI-1 training (400) | development | implementation, hyperparameter selection, ablation design |
| ARC-AGI-2 training (1,000) | development | implementation and model selection after content deduplication |
| ARC-AGI-1 public evaluation (400) | locked public audit benchmark | one frozen evaluation per protocol version; not label-naive |
| ARC-AGI-2 public evaluation (120) | primary locked public benchmark | one frozen evaluation per protocol version; not label-naive |
| ARC-AGI-2 semi-private/private (120 each) | external confirmation | one submission from a frozen commit if official access is available and the set was unseen by the team |
| IsoARC transformed suite | robustness stress test | never described as a private or independent benchmark |

ARC-AGI-2 training contains ARC-AGI-1 material. Development pools must therefore
be deduplicated by canonicalized task content, not filename. Isomorphic near
duplicates are clustered before any development split is made.

### 6.2 Development split

Create a machine-readable development manifest from training tasks only:

- `dev-build`: implementation and unrestricted debugging;
- `dev-select`: hyperparameter and model selection;
- `dev-audit`: final rehearsal with labels hidden from the run process.

Assign clusters, rather than individual files, by SHA-256 of canonicalized task
content and detected D4/color isomorphisms. Use a public deterministic seed.
Freeze all IDs before running method selection.

No public evaluation task may be moved into development. Public evaluation
labels already exist in the repository, so process-level isolation is required
even though perfect historical contamination control is impossible.

### 6.3 Label firewall

Inference and scoring run in separate processes and preferably separate
containers or Unix users.

The inference environment receives only:

- training demonstrations;
- test inputs;
- the frozen configuration;
- permitted model/source artifacts;
- a write-only run directory.

It must not receive evaluation outputs, solution files, prior predictions for
the evaluated task, or network access unless the configuration is explicitly an
API method. The challenge-only task tree is regenerated and hashed before each
protocol version.

The scorer receives persisted predictions only after inference terminates. A
test must prove that changing hidden test labels cannot change prediction bytes.
Any method that requires labels to choose a candidate is ineligible.

### 6.4 Freeze and submission policy

Before any new locked public or private evaluation, commit and hash:

- task manifests and challenge-only data;
- source and submodule revisions;
- artifact revisions and file hashes;
- container/environment locks;
- method configs and seeds;
- timeout, retry, and failure policies;
- analysis scripts and planned comparisons.

Evaluation output must not trigger a config change. A changed config is a new
protocol version and cannot replace the original result. Semi-private
leaderboard feedback is not used for tuning. Submit at most one frozen
configuration per declared method to the final private evaluation.

If private evaluation access is unavailable, the paper must say so prominently
and limit its claims to public-benchmark reproducibility and robustness.

### 6.5 Evidence tiers and prior exposure

Before protocol freeze, create a disclosure manifest listing every prior local
run, viewed score, leaderboard submission, and known use of ARC evaluation data
by the project. Evidence is labeled as:

1. `development`: training-only tasks and results used for decisions;
2. `locked-public`: public labels exist, but the new configuration and analysis
   were frozen before its predictions were scored;
3. `external-private`: the team had no labels or task-level feedback before the
   frozen submission;
4. `historical/published`: results imported from another report and not rerun.

The label firewall prevents direct answer leakage during execution; it does not
erase researcher exposure or model pretraining contamination. For pretrained
models, record release date, documented training-data cutoff, provider data
statements, and any available ARC-specific decontamination evidence. Unknown
contamination remains an explicit limitation rather than being inferred away
from good isomorphism performance.

## 7. IsoARC: semantics-preserving stress tests

### 7.1 Transform set

Every transform is applied consistently to all demonstration inputs,
demonstration outputs, test inputs, and hidden test outputs.

1. **D4 geometry:** identity, rotations by 90/180/270 degrees, horizontal and
   vertical reflection, transpose, and anti-transpose.
2. **Color renaming:** deterministic bijections among colors present in a task,
   with color 0 fixed in the primary analysis. A separate exploratory analysis
   may permute color 0 but must not be mixed with the primary result.
3. **Demonstration permutation:** reorder training pairs without changing their
   contents.
4. **Test-order permutation:** for multi-output tasks, reorder test inputs and
   restore canonical order before scoring.

Padding, cropping, translation, demonstration deletion, added distractors, and
rescaling are excluded from the primary isomorphism set because they can alter
the abstract task or the information available.

### 7.2 Variant construction

Use a balanced deterministic design rather than generating variants until a
desired result appears. For each selected base task, generate:

- all eight D4 variants for transformation-only analysis;
- four fixed-zero color permutations from published seeds;
- four D4-plus-color combinations from a Latin-square assignment;
- two demonstration-order permutations when at least three demonstrations
  exist.

The full generated set can be expensive. Therefore:

- a frozen 64-task curve/robustness set receives the complete design;
- the full public benchmark receives identity plus one balanced D4 and one
  balanced color variant per task;
- all variants remain clustered by base task in statistics.

The 64 base tasks are selected using input-visible features only, stratified by
benchmark generation, number of demonstrations, number of test inputs, input
area, color count, and whether demonstration shapes change. IDs are frozen
before any solver output is inspected.

### 7.3 Robustness metrics

For method `m`, report:

- original pass@2 accuracy;
- transformed pass@2 accuracy;
- paired accuracy drop;
- conditional retention: transformed correctness given original correctness;
- prediction equivariance: inverse-transform the prediction and compare it with
  the original prediction, independent of correctness;
- worst-transform accuracy;
- all-variants robust task accuracy;
- invalid-output and timeout rate by transform.

Prediction equivariance is useful but not sufficient: a consistently wrong
system can be perfectly equivariant. Accuracy and consistency must be shown
together.

## 8. Metrics and estimands

### 8.1 Primary ARC metric

The primary metric for ARC-AGI-2 is **output-level exact pass@2 with the full
declared denominator**, matching the current official competition scoring
description: each test input receives exactly two attempts, and a test output
is correct if either attempt exactly matches its target grid.

Because outputs from the same ARC task are dependent, uncertainty resamples
whole tasks even when the point estimate is output-weighted.

ARC-AGI-1 is reported separately using the same output-level pass@2 estimand for
cross-generation comparability and strict task accuracy for continuity with
this repository's existing reports. Neither metric is silently substituted for
the other, and ARC-AGI-1 and ARC-AGI-2 numerators are not pooled for the primary
claim.

### 8.2 Required secondary metrics

- strict task accuracy: every test output in the task is exact;
- Top-1 output and strict-task accuracy;
- output coverage and task coverage;
- invalid, missing, timeout, OOM, and retry counts;
- micro cell accuracy, clearly labeled diagnostic;
- output shape accuracy;
- wall time, device time, CPU-core seconds, peak RAM/VRAM, disk delta;
- GPU energy in joules and average/peak power when measurable;
- generated candidates, adaptation steps, model calls, input/output tokens;
- API cost and provider-reported usage;
- manual intervention count and duration.

Cell accuracy is never used to claim that an ARC task was solved.

### 8.3 Efficiency metrics

Report a performance frontier rather than one scalar rank:

- accuracy at each frozen online budget;
- area under accuracy versus log-online-compute on the curve set;
- joules and wall time per correct output;
- API dollars and tokens per correct output;
- offline training compute and artifact size as separate axes;
- Pareto dominance with uncertainty.

Offline training and online inference are not added into an arbitrary common
number. Published checkpoints with unknown training compute remain marked
unknown rather than estimated as zero.

## 9. Compute classes and budget accounting

### 9.1 Non-comparable execution classes

1. `local-24g`: one fixed 24 GiB GPU model, exclusive access;
2. `local-32g`: one fixed 32 GiB GPU model, exclusive access;
3. `cpu-local`: fixed CPU model and core count;
4. `api-replication`: hosted provider, model ID, date, tokens, requests, spend;
5. `paper-equivalent`: original/equivalent accelerator topology;
6. `reduced`: any smaller model, data, search, adaptation, or time budget.

Headline pairwise tests occur only within a compute class. Results from the
historical RTX 5090 and the current RTX A5000 must never be pooled as repeated
seeds.

### 9.2 Online budget definition

For local GPU methods, online time starts after shared model loading and ends
when the task's second persisted attempt is produced. It includes all
task-specific prompting, adaptation, search, validation, candidate selection,
and retries. Model cold-start and shared preprocessing are measured separately
and reported both raw and amortized.

The main public benchmark budget is provisionally **300 seconds per task**.
The frozen curve checkpoints are **30, 120, 300, and 900 seconds per task**.
These values remain provisional until a label-free development power/resource
simulation is completed; once frozen, they cannot be changed in response to
evaluation accuracy.

Anytime methods run once to the maximum budget and persist best-so-far attempts
at every checkpoint. Methods that cannot emit intermediate candidates run
separately at each checkpoint or are reported only at their supported budget.

Timeout setup/teardown receives a fixed grace period that cannot emit a new
prediction. A timeout without a valid persisted attempt is wrong, not missing
from the denominator.

### 9.3 State and amortization

- Main-track task state resets between tasks.
- Shared immutable model weights may remain resident in memory.
- Cross-task learning, memory, or library growth is prohibited in the standard
  track.
- Methods whose scientific contribution is continual learning run in a
  separate fixed-order track with at least three independently permuted task
  orders.
- Precomputed task-specific artifacts count as offline task leakage unless they
  were created solely from permitted training data.

### 9.4 API budgets

API methods are separate from local hardware comparisons. Freeze:

- provider and exact model identifier;
- UTC evaluation window;
- temperature and decoding parameters;
- calls, input/output tokens, and monetary cap per task;
- concurrency and retry schedule;
- response persistence policy and terms-of-service constraints.

At least three temporal replicates are desirable for mutable APIs. If cost
prevents this, report a single timestamped replication without seed-level
generalization claims.

### 9.5 Scheduling and machine drift

On a shared single-GPU campaign, method identity must not be confounded with
calendar time. Runs are scheduled in blocks with a frozen randomized method
order; the block size is chosen from development setup-overhead measurements.
Task order and seed schedules are fixed in the manifest. Each block records an
idle check and calibration workload, and an unapproved competing GPU process
invalidates the block under a rule written before evaluation.

## 10. Experimental matrix

### E0. Evaluator and firewall validation

Purpose: establish that the measurement system is correct before measuring
methods.

Required tests:

- official-format golden predictions with known Top-1/Top-2 scores;
- missing, unknown, malformed, wrong-shape, duplicate-key, and timeout cases;
- task/output weighting tests for multi-output tasks;
- scorer determinism and prediction hash stability;
- test-label mutation cannot change inference predictions;
- challenge-only trees contain no hidden output key;
- transform/inverse-transform round trips for all grids and predictions;
- transformed ground truth remains valid ARC JSON;
- resource monitor overhead calibration;
- container network and filesystem isolation checks.

Exit gate: two independent implementations or one implementation plus a
property-based test suite agree on every generated scorer case.

### E1. Twenty-four-method reproduction funnel

For every tracked method, record terminal evidence at each level:

1. reference identified;
2. source accessible and licensed;
3. source revision and submodules locked;
4. required artifacts accessible and hashed;
5. environment constructed;
6. import/static check passed;
7. no-label smoke passed;
8. fixed-subset benchmark passed;
9. full public benchmark passed;
10. paper-target reproduction passed.

Primary outputs are the funnel, time-to-level, disk/environment cost, manual
interventions, and blocker taxonomy. Preserve every failed attempt. Do not
overwrite failures with repaired runs.

### E2. Locked public compute-matched benchmark

Run every eligible ARC-native configuration on all 400 ARC-AGI-1 public
evaluation tasks and all 120 ARC-AGI-2 public evaluation tasks at the frozen
main budget.

- Top-K: exactly 2;
- task order: one public deterministic permutation per seed;
- local concurrency: one task process and exclusive GPU;
- method order: blocked randomization under Section 9.5;
- primary seed: one preregistered seed on all 520 tasks;
- stochastic replication: at least three seeds on all ARC-AGI-2 tasks and the
  64-task curve set;
- deterministic systems: one run plus a byte-identical replay audit;
- public generations: reported separately, never pooled.

Report ARC-AGI-1 and ARC-AGI-2 separately. A combined score may appear only as
an explicitly descriptive micro/macro aggregate.

These are exact results for the finite public benchmark. Confidence intervals
and hypothesis tests target variation across the broader task population that
the benchmark is intended to represent; they are not uncertainty about the
already observed finite-set score.

### E3. Online compute scaling

Run the 64-task frozen curve set to the maximum 900-second budget and persist
best-so-far attempts at 30, 120, 300, and 900 seconds. Use three seeds for
stochastic local systems.

Analyze:

- accuracy-compute curves and confidence bands;
- method-by-budget interaction;
- time-to-first-valid and time-to-first-correct output;
- marginal correct outputs per additional joule/minute;
- rank stability using Kendall's tau with task-bootstrap intervals;
- saturation and crossover points without extrapolating beyond measured
  budgets.

### E4. IsoARC robustness

Run the transformation design in Section 7 at the same online budget and matched
seed. Evaluate accuracy, consistency, conditional retention, and
method-by-transform interactions.

The primary paired unit is the base task. Variants are never treated as
independent samples. Show per-transform results so an aggregate cannot hide a
catastrophic failure on one symmetry.

### E5. Attempt and candidate-selection ablation

Using the exact same raw candidate streams, score:

- Top-1;
- Top-2;
- oracle best-of-all generated candidates (upper bound only);
- the method's declared candidate selector;
- a fixed development-selected cross-family two-attempt portfolio;
- a fixed development-selected same-family two-attempt portfolio;
- a single-system, full-budget Top-2 control.

This isolates gains from reasoning/search from gains due only to a larger
candidate budget. Official headline results remain Top-2.

For the portfolio comparison, the raw streams must come from runs made at half
of total budget `B`; re-scoring two full-budget systems would violate compute
matching. Each system contributes the candidate ranked first by its own frozen
selector. Portfolio membership is chosen using `dev-select` only, with a
lexicographic tie-break written into the manifest. The `dev-audit` split checks
the frozen selection procedure but cannot change its members.

### E6. Mechanism ablations

Select one runnable representative per family before locked evaluation.
The preregistered preferred set is:

- CompressARC: 0, 50, 300, and 1,500 task-optimization steps;
- GridCoder2024, if licensing and artifacts are resolved: learned heuristic
  versus the upstream non-learned search control;
- ARChitects: test-time adaptation on/off and voting on/off;
- NVARC, if runnable: full portfolio versus individual components.

If a preferred method fails the eligibility gate, use the next family member in
a fallback order frozen before evaluation. An unavailable ablation is reported
as unavailable; it is not silently replaced after seeing results.

### E7. Failure and capability analysis

Use only features computable from visible task information for quantitative
strata:

- number of demonstrations and test inputs;
- input dimensions and area;
- number and frequency distribution of colors;
- demonstration shape change;
- object count from a frozen, method-independent parser;
- transformation family;
- runtime/resource bucket.

Fit task-clustered models of correctness and failure. Do not call these ARC
"skills" unless the labels are validated independently.

For qualitative analysis, sample tasks by a frozen rule stratified over
success/failure patterns. Two annotators independently code failure mechanisms
using a preregistered rubric, remain blind to method identity where feasible,
and report agreement (Cohen's kappa plus raw agreement). Resolve disagreements
only after independent labels are saved.

### E8. Private confirmation, conditional on access

Submit exactly the frozen public-evaluation commit/configuration to the official
ARC-AGI-2 private evaluation under its current no-internet and compute rules.
No private or semi-private feedback is used to change the method.

Compare public and private results with calibrated uncertainty, but do not claim
the sets are identical in every machine-facing property. If access is denied or
the execution environment cannot host a method, report the gate failure rather
than substituting public accuracy.

## 11. Statistical analysis plan

### 11.1 Units and confidence intervals

- The finite-benchmark score is reported without pretending it has measurement
  error; inferential intervals target a task-generating population.
- Base ARC task is the resampling and clustering unit.
- Multi-output test pairs and IsoARC variants stay within their base-task
  cluster.
- Report point estimates and 95% task-cluster bootstrap intervals using 10,000
  deterministic resamples.
- For stochastic methods, show each seed and decompose between-task and
  between-seed variation.
- Never treat generated variants as increasing the number of independent ARC
  tasks.

### 11.2 Prespecified tests and confirmation status

1. H1: omnibus configuration-by-log-budget interaction in a task-clustered
   logistic model, with configuration-specific slopes only as follow-ups.
2. H2: omnibus transformation and method-by-transformation test, followed by
   paired original-versus-transformed method contrasts.
3. H3: paired cross-family versus same-family portfolio contrast at fixed total
   budget, with the single-system full-budget result as a required control.
4. Secondary: fixed-budget global method comparison, followed by pairwise
   contrasts only if the global test rejects.

Use two-sided alpha 0.05. The three primary omnibus p-values for H1-H3 form one
family and receive Holm correction. If H1 or H2 survives that gate, its
prespecified method-level follow-ups receive a second Holm correction within
that hypothesis. Fixed-budget pairwise comparisons form a separately labeled
secondary family. Report unadjusted effect sizes, unadjusted p-values, adjusted
p-values, and all planned contrasts regardless of significance.

The term `confirmatory` is reserved for a hypothesis evaluated on a genuinely
unseen external-private task set with the frozen procedure. On public tasks,
the identical tests are called prespecified locked-public analyses.

The binary observation is exact correctness for one test output. H1 uses a
marginal logistic GEE with configuration, centered log budget, their
interaction, and benchmark generation; H2 uses configuration, transformation
category, their interaction, and benchmark generation. Both use a robust
sandwich covariance clustered by base task, containing every output, variant,
budget, and seed for that task. H3 uses the corresponding paired marginal model
with portfolio type as the focal coefficient. Seeds receive equal weight
within a stochastic configuration; deterministic replays verify stability but
do not masquerade as independent seeds.

Before evaluation, simulation tests must cover separation, non-convergence, and
singular covariance. The frozen fallback is a base-task paired randomization
test with the same estimand and multiplicity hierarchy; the analysis may not
switch tests because one produces a more favorable p-value.

Exploratory subgroup analyses are labeled exploratory and emphasize intervals,
not binary significance.

### 11.3 Power and benchmark size

Headline experiments use the complete fixed public evaluation populations, so
task counts are 400 and 120 rather than power-selected samples. Before
evaluation, run a simulation-based minimum-detectable-effect analysis using
only development outcomes and the planned paired tests. Publish the simulation
code and its assumptions.

The 64-task curve/robustness subset supports estimation of large effects and
interactions; it is not used for small-difference leaderboard claims. If the
simulated power for a planned contrast is below 80% at a scientifically relevant
effect, label that contrast estimation-only before running it.

### 11.4 Missingness and retries

There is no outcome-dependent missing-data imputation:

- missing/malformed output: incorrect;
- task timeout: incorrect;
- method OOM: incorrect;
- API failure after the frozen retry count: incorrect;
- infrastructure failure external to the method: eligible for one adjudicated
  rerun under a blinded rule, while the failed run remains preserved;
- human correction of a prediction: prohibited.

## 12. Hyperparameter and configuration policy

- All method choices use development tasks only.
- Use upstream defaults unless a change is required for the declared compute
  class.
- Every deviation receives a rationale and an ablation when feasible.
- Search spaces, trial counts, and selection metrics are frozen before search.
- Hyperparameter search compute is reported as offline experimental compute.
- No method receives extra tuning because its initial evaluation score is low.
- The same visible task information is available to every solver, subject to
  the method's published protocol.
- Config files are schema-validated and content-hashed into each run record.

## 13. Run evidence contract

Each attempt writes an immutable directory:

`reports/<method>/<protocol-version>/<run-id>/`

Required files:

- `run.json` with schema version and terminal status;
- challenge manifest and hashes;
- exact command and working directory;
- source commit, dirty flag, submodule SHAs, and patch hash;
- container/environment lock and package export;
- artifact IDs, revisions, sizes, hashes, licenses, and local paths;
- stdout/stderr and structured event log;
- raw candidate stream with timestamps;
- canonical Top-2 predictions;
- scorer output and analysis-ready per-task records;
- hardware, power, timing, RAM/VRAM, disk, token, call, and cost traces;
- retry, timeout, OOM, malformed-output, and intervention records;
- a content manifest hashing every run file except the manifest itself, plus a
  signed or externally recorded root hash.

The final record is append-only after terminalization. Repairs produce a new
run ID. Logs must never contain credentials or hidden labels.

## 14. Resource and execution plan

### 14.1 Current pre-execution blockers

Observed on 2026-08-04:

- the container root overlay is 100 GiB with only about 7.3 GiB free;
- a physical approximately 232.9 GiB NVMe device is visible but not mounted as a
  writable workspace volume;
- `/model` is absent;
- the active GPU is an RTX A5000 with 24 GiB, not the RTX 5090 recorded by
  historical runs;
- the A5000 was idle at the latest check, but no exclusive campaign reservation
  or contention-rejection mechanism has been established;
- the reduced Qwen environment fails `pip check` because `pycairo` is absent.

No main experiment starts in this state.

### 14.2 Storage gate

Before execution:

1. mount or provision the intended approximately 200 GB workspace/model volume;
2. record its filesystem, quota, writable path, and free bytes;
3. place all model/dataset caches under a namespaced `/model/arc-rebench` (or
   equivalent mounted path), never the root home cache;
4. deduplicate existing Hugging Face blobs by immutable revision;
5. retain at least 20 GiB free throughout the campaign;
6. block a run whose projected maximum disk delta violates the reserve.

For a measured usable capacity of 200 GiB, the initial hard allocation is:

| Storage class | Cap |
|---|---:|
| Content-addressed model and dataset blobs | 108 GiB |
| Environments and container layers | 36 GiB |
| Immutable run records and resource traces | 22 GiB |
| Per-run scratch space | 14 GiB |
| Untouched emergency reserve | 20 GiB |

If the advertised 200 GB is decimal or the mounted quota is smaller, generate
the same allocation from measured bytes: reserve the larger of 20 GiB or 10%,
then allocate the usable remainder 60%/20%/12%/8% across the first four rows.
Artifact admission uses compressed and peak-extracted size; a model is not
downloaded merely because its compressed file fits.

No cache or user artifact is deleted without explicit approval and a verified
inventory.

### 14.3 Hardware gate

Choose the primary local compute class before benchmarking:

- reserve exclusive access to the exact GPU;
- lock driver, CUDA runtime, framework build, power mode, and clock policy;
- record idle utilization and reject runs with unapproved competing GPU work;
- run a calibration workload before and after each experiment block;
- never merge A5000 and 5090 results as repeated trials.

If both GPUs become available, treat hardware transfer as a separate systems
experiment with matched configs, not as extra seeds.

### 14.4 Estimated campaign size

The final estimate is generated from successful development smokes, not paper
claims. Before approval, produce a table with tasks x budgets x seeds x measured
seconds, projected GPU-hours, energy, API spend, disk delta, and 20% contingency.

A timeout-based upper bound is nevertheless required before work begins. With
six configurations in E2, four representatives selected by mechanism and
eligibility (not evaluation score) for E3/E4, and every run consuming its full
timeout, the current provisional design implies:

| Component | Upper-bound local GPU time |
|---|---:|
| E2 identity, 6 x 520 tasks x 300 s | 260 h |
| E2 two extra ARC-AGI-2 seeds, 6 x 2 x 120 x 300 s | 120 h |
| E3 curves, 4 x 3 x 64 x 900 s | 192 h |
| E4 two extra full-set variants, 4 x 2 x 520 x 300 s | 347 h |
| E4 remaining 15 variants on 64 tasks, 4 x 15 x 64 x 300 s | 320 h |
| **Subtotal before ablations** | **1,239 h (51.6 GPU-days)** |
| **With 20% contingency** | **1,487 h (62.0 GPU-days)** |

This is an upper bound, not a promise that all systems use the entire timeout;
overlap between anytime checkpoints may lower it. It excludes E6 ablations,
API spend, environment construction, failed runs, and paper-equivalent cluster
reproductions. At 80% single-GPU utilization it occupies about 78 calendar
days, so the expensive four-system diagnostic set and the total campaign cap
must be frozen before outcomes are inspected. If the measured projection does
not fit the calendar, reduce the number of prespecified transforms or budgets,
not the denominator or unsuccessful runs after seeing results.

The campaign is staged so that expensive work cannot begin before cheaper gates
pass:

1. evaluator/firewall/property tests;
2. source, license, and artifact audit;
3. environment and one-task smokes;
4. development timing measurements;
5. frozen 64-task curves and robustness;
6. full public benchmark;
7. private confirmation;
8. analysis replay from immutable records.

## 15. Stop, pivot, and exclusion rules

### Stop a configuration when

- it violates a license or access condition;
- it cannot be isolated from hidden labels;
- projected storage crosses the reserve;
- it repeatedly corrupts or omits run evidence;
- its declared resource class is impossible on available hardware;
- three independently repaired smokes fail for the same unresolved upstream
  blocker.

### Pivot the paper framing when

- fewer than six ARC systems or four families pass the benchmark gate;
- no private evaluation is available;
- transformed-task correctness cannot be validated unambiguously;
- compute logging is not comparable across the intended headline methods.

The fallback is a rigorous E&D reproduction audit with negative results and a
validated evaluation protocol, not an underpowered leaderboard paper.

### Never exclude after outcomes are known

- hard tasks;
- wrong answers;
- methods with low scores;
- seeds with failures;
- transformations that reduce accuracy;
- timeouts or OOMs within the declared method configuration.

## 16. Planned figures and tables

### Main paper

1. **Figure 1:** 24-method reproduction funnel with blocker taxonomy.
2. **Figure 2:** pass@2 versus online compute/energy Pareto curves.
3. **Figure 3:** original and IsoARC robustness by method and transform.
4. **Figure 4:** paired task-level method complementarity and portfolio gain.
5. **Table 1:** eligibility, source/artifact state, compute class, coverage.
6. **Table 2:** ARC-AGI-1/2 primary results with clustered confidence intervals.
7. **Table 3:** failures, invalid outputs, runtime, energy, and intervention rate.

### Appendix/artifact

- every per-seed score;
- every planned and exploratory statistical contrast;
- environment and artifact locks;
- task and transformation manifests;
- per-task outcomes and resource traces where licensing permits;
- all negative and blocked results;
- reproduction commands and expected checksums.

## 17. NeurIPS artifact readiness

Because the executable evaluator is a primary contribution, code must be
reviewer-accessible, documented, and runnable at submission. Provide:

- an anonymized repository for double-blind review;
- a CPU quickstart that reproduces scorer tests and the deterministic floor;
- one small end-to-end smoke runnable in under 30 minutes;
- exact environment and full-result replay commands;
- model/data cards or evaluation cards for every released artifact;
- license and provenance tables;
- limitations, maintenance, and deprecation policy;
- Croissant core and Responsible AI metadata if IsoARC is released as a dataset;
- a small inspectable sample if any released dataset exceeds 4 GB;
- the completed NeurIPS paper checklist with section-level pointers.

The artifact must let a reviewer distinguish reproduced results from blocked,
estimated, published-only, and exploratory results without reading raw logs.

## 18. Threats to validity

### Internal validity

- public labels are locally present and require a process-level firewall;
- historical environments may not resolve on current hardware;
- GPU contention and API drift can alter outcomes;
- method wrappers may change native timing or candidate selection;
- stochastic CUDA kernels may remain nondeterministic despite fixed seeds.

### Construct validity

- exact ARC accuracy measures output correctness, not a complete theory of
  intelligence;
- online time, energy, tokens, and offline compute capture different resources;
- fixed-zero color permutations test one abstraction assumption, not every
  legitimate use of color semantics;
- cell accuracy is background-sensitive and not task solution.

### External validity

- ARC-AGI public tasks are a finite curated benchmark;
- API systems may have benchmark contamination;
- one-GPU results do not generalize automatically to paper-scale clusters;
- blocked implementations bias performance tables toward reproducible methods;
- private ARC results are subject to competition environment constraints.

### Statistical conclusion validity

- 120 ARC-AGI-2 tasks limit detection of small differences;
- variants are correlated and cannot inflate sample size;
- multiple method/transform comparisons require correction;
- seed and temporal API variance may be expensive to estimate.

## 19. Historical immediate execution sequence (superseded)

The next actions are ordered by information value, not method popularity:

1. provision/mount usable storage and reserve the primary GPU;
2. update the host manifest and invalidate 5090 assumptions for new runs;
3. align the scorer's primary metric with official output-level pass@2 while
   retaining strict task accuracy;
4. implement the label firewall and challenge-only data generator;
5. add scorer property tests and an independent reference scorer;
6. implement and verify IsoARC transform/inverse-transform tooling;
7. repair preparation status, submodule, hash, and cache-location checks;
8. freeze method eligibility and family taxonomy;
9. run ARC_NCA and other low-cost smokes on development tasks only;
10. measure task-level runtime, then finalize the campaign power/resource
    calculation;
11. freeze protocol version 1 before any new public evaluation run;
12. execute curves, full public evaluation, and private confirmation in that
    order.

## 20. Success criterion

This design reaches NeurIPS level when another team can rerun the evaluator,
understand exactly which scientific claim every metric supports, reproduce the
main comparisons from immutable artifacts, and obtain the same conclusion even
when failures, resource use, and representation robustness are included.

A larger table of unnormalized scores does not satisfy this criterion.

## Official policy references

- NeurIPS 2026 Evaluations & Datasets call:
  <https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets>
- NeurIPS paper checklist:
  <https://neurips.cc/public/guides/PaperChecklist>
- NeurIPS 2026 E&D reviewer guidelines:
  <https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines>
- ARC-AGI-2 benchmark description:
  <https://arcprize.org/arc-agi/2>
- ARC Prize 2026 ARC-AGI-2 scoring and execution requirements:
  <https://arcprize.org/competitions/2026/arc-agi-2>
