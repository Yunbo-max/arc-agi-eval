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
