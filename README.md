# ARC-AGI evaluation foundation

A small, dependency-free Python toolkit for validating, enumerating, and
scoring the public ARC-AGI-1 and ARC-AGI-2 benchmarks. Canonical public source
snapshots are vendored under `third_party/`; no model checkpoints are included.

## Project status

Last evidence update: **2026-08-04**. This section is an evidence ledger, not a
claim that all tracked papers have been reproduced.

| Area | Verified progress | Evidence |
| --- | --- | --- |
| Evaluator | All 1,920 vendored ARC-AGI-1/2 tasks validate; 29 tests pass | [`tests/`](tests/), [`third_party/SOURCES.md`](third_party/SOURCES.md) |
| Method census | 24 methods tracked: 19 public candidates, 1 partial/complex candidate, and 4 without a verified runnable implementation | [`configs/baselines.json`](configs/baselines.json), [`docs/REPRODUCTION_MATRIX.md`](docs/REPRODUCTION_MATRIX.md) |
| Method execution | 1 method has a passing compatibility smoke; 0 have passed a declared benchmark; 0 are full paper reproductions | [`reports/`](reports/), [`LONG_GOAL.md`](LONG_GOAL.md) |
| Deterministic floor | Complete Top-2 predictions and scores exist for all 400 ARC-AGI-1 and 120 ARC-AGI-2 public evaluation tasks | [`results/`](results/) |
| Research protocol | NeurIPS-level design v0.2 is written, but is not frozen, preregistered, or implemented | [`docs/NEURIPS_EXPERIMENT_DESIGN.md`](docs/NEURIPS_EXPERIMENT_DESIGN.md) |
| Durable artifacts | 13 baseline/CompressARC result files are mirrored and hash-verified in the private Hugging Face model repository; no run-produced checkpoint exists | [`scripts/hub_sync.py`](scripts/hub_sync.py) |

Evidence levels are deliberately separate:

`source-audited -> smoke -> single-task experiment -> benchmark -> paper reproduction`

A source checkout or published score does not advance execution status. Failed
runs remain in the history, and every status change must link to a run record.

## Measured results

All exact results below use Top-2. Cell accuracy is a diagnostic and never
means that an ARC task was solved.

| System | Scope | Output exact | Strict task exact | Cell accuracy | Wall time | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Deterministic floor | ARC-AGI-1 public evaluation, 400 tasks / 419 outputs | 0 / 419 | 0 / 400 | 73,289 / 98,515 (74.3937%) | 0.407 s | [`run.json`](results/arc-agi-1-evaluation-baseline-run.json) |
| Deterministic floor | ARC-AGI-2 public evaluation, 120 tasks / 167 outputs | 0 / 167 | 0 / 120 | 48,047 / 70,100 (68.5407%) | 0.216 s | [`run.json`](results/arc-agi-2-evaluation-baseline-run.json) |
| CompressARC | ARC-AGI-1 training task `007bbfb7`, 2 optimization steps | 0 / 1 | 0 / 1 | 11 / 81 (13.5802%) | 7.058 s | [`run.json`](reports/compressarc/20260804-training-2step/run.json) |
| CompressARC | ARC-AGI-1 training task `007bbfb7`, 1,500 optimization steps | 0 / 1 | 0 / 1 | 58 / 81 (71.6049%) | 1,817.725 s | [`run.json`](reports/compressarc/20260804-training-1500-007bbfb7/run.json) |

The CompressARC rows are post-hoc scores on one training task, not a benchmark
or paper reproduction. Its test output was unavailable to the optimizer. The
passing forward smoke is recorded separately in
[`20260804-smoke-forward-002`](reports/compressarc/20260804-smoke-forward-002/run.json).

## Project history

| Date | Milestone | Durable evidence |
| --- | --- | --- |
| 2026-08-04 | Created the ARC-AGI-1/2 validator, enumerator, scorer, deterministic floor, vendored data snapshots, source audit, and run evidence contract | [`f7e7935`](https://github.com/Yunbo-max/arc-agi-eval/commit/f7e7935382213b1712fb41a422c65ab8811d0ec4) |
| 2026-08-04 | Preserved the first CompressARC smoke failure: the local harness incorrectly assumed `ARCCompressor.eval()` existed | [`smoke-forward-001`](reports/compressarc/20260804-smoke-forward-001/run.json) |
| 2026-08-04 | Removed that invalid harness assumption and completed a forward-pass compatibility smoke | [`smoke-forward-002`](reports/compressarc/20260804-smoke-forward-002/run.json) |
| 2026-08-04 | Completed 2-step and 1,500-step CompressARC single-task runs; neither solved the task exactly | [`reports/compressarc/`](reports/compressarc/) |
| 2026-08-04 | Added isolated preparation contracts, source locks, asset manifests, and GitHub/Hugging Face persistence instructions for all 24 tracked methods | [`8a2632c`](https://github.com/Yunbo-max/arc-agi-eval/commit/8a2632cce8a39c9d665285574d00f155f87b49f8) |
| 2026-08-04 | Added the ARC-REBench NeurIPS-level protocol: compute matching, label isolation, IsoARC, clustered statistics, stop rules, and resource gates | [`7ba3087`](https://github.com/Yunbo-max/arc-agi-eval/commit/7ba30878bf5210e10ef057ab585c5c01c4d712c7) |

## Current execution gates

Latest host observation on 2026-08-04:

- A roughly 232.9 GiB physical NVMe device is visible, but it is not mounted as
  a general writable model/workspace volume. `/model` is absent.
- The container root is a 100 GiB overlay with less than 8 GiB free (7.3 GiB at
  the latest check). No main experiment or large model download starts there.
- The currently visible GPU is an idle 24 GiB RTX A5000. Historical
  CompressARC evidence was produced on a 32 GiB RTX 5090; those results are
  different compute classes and will not be pooled as repeated trials.
- The CLI currently treats strict whole-task exact accuracy as its primary
  score. The research protocol requires official ARC-AGI-2 output-level exact
  pass@2 as primary and strict task accuracy as secondary; migration tests are
  not implemented yet.
- The challenge-only data generator, label firewall, independent reference
  scorer, and IsoARC transform tests remain to be implemented.

The immediate order is: mount the intended storage, reserve the compute class,
complete evaluator/firewall experiment E0, run development-only low-cost
smokes, measure runtime, and freeze protocol v1 before another public
evaluation. See the full sequence in the
[`NeurIPS experiment design`](docs/NEURIPS_EXPERIMENT_DESIGN.md#19-immediate-execution-sequence).

## Data

| Benchmark | Path | Training | Evaluation | Data bytes |
| --- | --- | ---: | ---: | ---: |
| ARC-AGI-1 | `third_party/arc-agi-1/data` | 400 | 400 | 3,814,819 |
| ARC-AGI-2 | `third_party/arc-agi-2/data` | 1,000 | 120 | 6,063,256 |

The task data and upstream documentation are Apache-2.0 licensed. Exact source
URLs, revisions, retrieval date, and extracted snapshot sizes are recorded in
[`third_party/SOURCES.md`](third_party/SOURCES.md). Evaluation labels in these
repositories are public; do not tune against them when reporting held-out
results.

## Usage

Python 3.10 or newer is the only runtime requirement. Commands work directly
from the repository root; an editable install is optional.

```bash
python3 -m arc_agi_eval validate third_party/arc-agi-1/data third_party/arc-agi-2/data
python3 -m arc_agi_eval list third_party/arc-agi-2/data
python3 -m arc_agi_eval list third_party/arc-agi-2/data --tasks
python3 -m arc_agi_eval score predictions.json third_party/arc-agi-2/data/evaluation
python3 -m unittest discover -s tests -v
```

Install the `arc-agi-eval` command if desired:

```bash
python3 -m pip install -e .
arc-agi-eval --help
```

`validate` accepts task files, split directories, or dataset directories and
recursively validates every JSON task. It enforces the official task shape,
including the optional `name` metadata present in some current ARC-AGI-1
files, nonempty `train` and `test` lists, rectangular grids from 1x1 through
30x30, integer colors 0 through 9, and no duplicate JSON object keys. Use
`--allow-missing-test-outputs` only for unlabeled challenge inputs.

## Prediction format

Predictions use the common ARC submission format. There is one list item per
test input, and attempt numbers are one-based and contiguous:

```json
{
  "task_id": [
    {
      "attempt_1": [[1, 2], [3, 4]],
      "attempt_2": [[4, 3], [2, 1]]
    }
  ]
}
```

The scorer defaults to Top-2. Set any positive attempt budget with `--top-k`:

```bash
python3 -m arc_agi_eval score predictions.json TASK_SPLIT --top-k 1 --json
python3 -m arc_agi_eval score predictions.json TASK_SPLIT --top-k 3 --json
```

Scoring semantics are explicit:

- A test output is exact when any of its first K attempts equals the full
  expected grid, including dimensions.
- A task is exact only when every test output in that task is exact. This is
  the primary ARC task score.
- Cell accuracy is micro-averaged over all expected cells. For each test
  output, the first K attempts compete and the candidate with the most matching
  cells is used. A dimension mismatch receives zero matching cells.
- Missing tasks receive zero credit. Unknown task IDs and malformed supplied
  predictions are errors rather than silently ignored.

The command reports raw numerators and denominators alongside task, output, and
cell accuracies so results can be audited.

## Research protocol

The proposed NeurIPS-level study design is documented in
[`docs/NEURIPS_EXPERIMENT_DESIGN.md`](docs/NEURIPS_EXPERIMENT_DESIGN.md). It
defines evidence tiers, label isolation, compute-matched comparisons, IsoARC
stress tests, clustered statistics, resource gates, and immutable run records.
It is a design document rather than an implemented or completed experiment.

The design uses official output-level exact pass@2 as the ARC-AGI-2 primary
estimand and retains this evaluator's stricter whole-task exact score as a
required secondary metric. The current CLI behavior described above remains
unchanged until the scorer migration and compatibility tests are implemented.

## Per-paper preparation workspaces

The repository tracks 24 independent papers/methods. Each now has a standalone
model/data/environment/run contract under [`papers/`](papers/README.md). The
manifests are the source of truth:

- `configs/baselines.json`: scientific scope, feasibility, and execution state;
- `configs/source_locks.json`: exact upstream repository commits;
- `configs/paper_assets.json`: model, dataset, entry point, and storage policy.

Prepare one paper without changing this evaluator's environment:

```bash
python3 -m venv .venvs/preparation
.venvs/preparation/bin/pip install -r requirements/preparation.txt
export ARC_PAPER_ASSETS_ROOT=/usr/paper-assets/arc
.venvs/preparation/bin/python scripts/prepare_paper.py --paper latentmas --download-public-assets
```

Prepare all publicly accessible sources and capacity-approved assets while
preserving at least 10 GiB of free disk:

```bash
.venvs/preparation/bin/python scripts/prepare_paper.py --all --download-public-assets
```

The reduced Qwen3-based multi-agent papers use their own environment so they
cannot change EventTune or evaluator dependencies:

```bash
python3 -m venv --system-site-packages .venvs/reduced-qwen3
.venvs/reduced-qwen3/bin/pip install -r requirements/reduced-qwen3.txt
```

The preparation command never downloads paid-API, Kaggle-only, gated, or
unidentified artifacts automatically. Such papers receive a concrete blocker
instead of a false ready status. Every upstream environment remains isolated
from `arc_agi_eval`.

Code and READMEs are versioned in GitHub. Private durable manifests and run
artifacts are namespaced by paper in:

- `humanlong/ARC-AGI-Eval-Data`;
- `humanlong/ARC-AGI-Eval-Models`.

Synchronize approved material with:

```bash
.venvs/preparation/bin/python scripts/hub_sync.py push-metadata
.venvs/preparation/bin/python scripts/hub_sync.py push --paper compressarc
.venvs/preparation/bin/python scripts/hub_sync.py pull --paper compressarc
```
