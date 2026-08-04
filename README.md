# ARC-AGI evaluation foundation

A small, dependency-free Python toolkit for validating, enumerating, and
scoring the public ARC-AGI-1 and ARC-AGI-2 benchmarks. Canonical public source
snapshots are vendored under `third_party/`; no model checkpoints are included.

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
