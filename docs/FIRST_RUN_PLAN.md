# First-run plan for storage-light official baselines

Status update (2026-08-06): this is a historical RTX 5090 acquisition plan,
not current execution authority. The active target is one RTX 3090; use
[`EXECUTION_BATCHES.md`](EXECUTION_BATCHES.md) and the current
[protocol root](../reports/e0-protocol/20260806-protocol-v1-draft-root-retry16/run.json).
The process-tree resource gate remains pending and no locked-public solver run
is authorized. Historical commands and measurements below are retained as
provenance.

This plan covers only the pinned source snapshots in `external/CompressARC`
and `external/ARC-VSA-2025`. No dependency was installed, no upstream file was
edited, no checkpoint was downloaded, and no training or solver experiment was
run during acquisition.

## Host and preflight status

| Item | Observed value |
| --- | --- |
| Python | CPython 3.12.11 |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB |
| Driver | 595.80 |
| System CUDA toolkit | 13.2 (`nvcc` 13.2.51) |
| Git | Not installed |
| Source syntax | All 24 retained `.py` files parse with Python 3.12 |

The system CUDA toolkit does not have to match a PyTorch wheel's bundled CUDA
runtime. The installed driver must be new enough for that runtime, which this
driver is for CUDA 12.8 wheels.

A dependency-free syntax check that does not create `__pycache__` is:

```bash
cd /workspace/arc-agi
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import ast
from pathlib import Path

files = sorted(Path("external").glob("**/*.py"))
for path in files:
    ast.parse(path.read_bytes(), filename=str(path))
print(f"parsed {len(files)} Python files")
PY
```

## CompressARC

### Actual upstream setup

There is no package metadata or test suite. The README says to create a venv,
install `requirements.txt`, change into the source root, and run
`analyze_example.py`. The requirements are exactly `matplotlib==3.10.0`,
`numpy==2.2.2`, `torch==2.5.1`, and `tqdm==4.66.6`.

Do not install that requirements file unchanged on this host. The standard
PyTorch 2.5.1 wheels predate Blackwell (`sm_120`) support, while the code sets
the global default device to CUDA at import time in `arc_compressor.py`. Use an
official CUDA 12.8 build with Blackwell support instead. PyTorch 2.7.0 is the
smallest stable release offered with CUDA 12.8 by the upstream selector; this
is an intentional compatibility deviation from the baseline's pin.

Minimal isolated environment commands, not executed during this audit:

```bash
cd /workspace/arc-agi
python3 -m venv .venvs/compressarc
. .venvs/compressarc/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib==3.10.0 numpy==2.2.2 tqdm==4.66.6
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip check
```

Confirm the wheel actually executes on the 5090 before importing the solver:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
x = torch.ones(1, device="cuda")
assert x.add(1).item() == 2
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

### Dataset and working directory

Every loader uses a path relative to the current directory. Run from
`/workspace/arc-agi/external/CompressARC`, where `dataset/` is already retained
in the required Kaggle aggregate format:

| Required path | Entries | Purpose |
| --- | ---: | --- |
| `dataset/arc-agi_training_challenges.json` | 400 | Training demonstrations and test inputs |
| `dataset/arc-agi_training_solutions.json` | 400 | Public training test outputs |
| `dataset/arc-agi_evaluation_challenges.json` | 400 | Evaluation demonstrations and test inputs |
| `dataset/arc-agi_evaluation_solutions.json` | 400 | Public evaluation test outputs |
| `dataset/arc-agi_test_challenges.json` | 100 | Competition test inputs |
| `dataset/sample_submission.json` | 100 | Competition submission skeleton |

The training and evaluation aggregates exactly match the canonical
`third_party/arc-agi-1/data/{training,evaluation}` tasks. A symlink to those
directories cannot replace `dataset/`, because their one-JSON-file-per-task
layout is not what `preprocessing.py` reads.

### Small smoke test

The README entry point always trains for 1,500 steps and writes plots, so it is
not a smoke test. After the CUDA check, use one task and one forward pass; this
does not optimize, save results, or edit source:

```bash
cd /workspace/arc-agi/external/CompressARC
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python - <<'PY'
import torch
import preprocessing
import arc_compressor

task = preprocessing.preprocess_tasks("training", ["007bbfb7"])[0]
model = arc_compressor.ARCCompressor(task)
with torch.no_grad():
    logits, x_mask, y_mask, kl, names = model.forward()
torch.cuda.synchronize()
print(task.task_name, tuple(logits.shape), tuple(x_mask.shape), len(kl))
PY
```

Do not use `analyze_example.py`, `train.py`, or `parallel_train.py` for the
first smoke. They respectively run 1,500 steps for one task, 2,000 steps for
each of 400 tasks, or a two-pass all-task GPU scheduler. `list_solved_puzzles.py`
also cannot be used because its large generated `.npz` inputs were deliberately
excluded.

### Compatibility notes

- CUDA is mandatory without changing upstream; importing `arc_compressor.py`
  sets `torch.set_default_device("cuda")`.
- Run from the source root because imports are flat modules and dataset/output
  paths are CWD-relative.
- Upgrading only Torch to 2.7.0 is expected to preserve the APIs used here, but
  it remains an unverified deviation until the forward smoke passes.
- Python 3.12 syntax parsing passed. The pinned NumPy and Matplotlib releases
  publish Python 3.12 support; runtime imports were not tested.
- Result files use pickled NumPy object arrays and the analysis script uses
  `eval()` on stored tensor names. Load only trusted artifacts.

## ARC-VSA-2025

### Actual upstream setup

The README gives no environment or execution command. There is no package
metadata or test suite. `requirements.txt` lists ten unpinned names:
`interruptingcow`, `matplotlib`, `natsort`, `nengo_spa`, `numpy`, `scipy`,
`scikit-learn`, `scikit-image`, `torch`, and `tqdm`.

The linked Kaggle submission currently identifies Python 3.11.13, CPU-only
execution, `nengo==4.1.0`, `nengo-spa==2.0.0`, and
`interruptingcow==0.8`. Its actual solver is imported from a separate Kaggle
dataset as `kaggle_solver`, not from this repository's `src/` tree.

This closest minimal environment can be staged, but it is not sufficient to
import the GitHub solver because of the blocker below. These commands were not
executed:

```bash
cd /workspace/arc-agi
python3 -m venv .venvs/arc-vsa-2025
. .venvs/arc-vsa-2025/bin/activate
python -m pip install --upgrade pip
python -m pip install nengo==4.1.0 nengo-spa==2.0.0 interruptingcow==0.8
python -m pip install matplotlib natsort numpy scipy scikit-learn scikit-image tqdm
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip check
```

CPU Torch is deliberate: the GitHub implementation only seeds CUDA when it is
available and never moves its networks or tensors to CUDA. The RTX 5090 will
not accelerate this baseline without an upstream code change.

### Blocking missing dependency

`src/vsa.py` directly imports `sspspace`, but it is absent from
`requirements.txt`, absent from the archive, and has no distribution named
`sspspace` on PyPI. Do not install a guessed package. The similarly named
<https://github.com/ctn-waterloo/sspspace> is not API-compatible: its
`RandomSSPSpace` does not accept this code's `domain_bounds` or `sampler`
arguments and its encoder does not provide `get_sample_pts_and_ssps`. Its
license also contains use restrictions despite `setup.py` describing it as
MIT, so it requires separate review.

An executable first run is blocked until the authors identify or publish the
exact `sspspace` implementation and version used by this source. The linked
Kaggle notebook's separate `arc-solver-v3` data source may contain the missing
implementation and a competition wrapper, but neither is part of this
MIT-licensed GitHub snapshot. No such asset was downloaded or vendored.

### Dataset symlink layout

The GitHub runner expects task-per-file data, which the canonical local
snapshots already provide. It also hardcodes the old repository name
`ARC-Development` when deriving `data/` and `runs/`. Running directly from
`external/ARC-VSA-2025` therefore constructs a nonexistent path such as
`ARC-VSA-2025data/`.

Once the missing dependency is resolved, use a temporary directory whose path
contains the expected old name. This avoids editing upstream and reuses data
through symlinks:

```bash
ROOT=/workspace/arc-agi
RUN_ROOT=/tmp/arc-vsa-run/ARC-Development
mkdir -p "$RUN_ROOT/data/arcagi1" "$RUN_ROOT/data/arcagi2" "$RUN_ROOT/runs"
ln -sfn "$ROOT/third_party/arc-agi-1/data/training" "$RUN_ROOT/data/arcagi1/training"
ln -sfn "$ROOT/third_party/arc-agi-1/data/evaluation" "$RUN_ROOT/data/arcagi1/evaluation"
ln -sfn "$ROOT/third_party/arc-agi-2/data/training" "$RUN_ROOT/data/arcagi2/training"
ln -sfn "$ROOT/third_party/arc-agi-2/data/evaluation" "$RUN_ROOT/data/arcagi2/evaluation"
```

A data-only smoke, valid before installing dependencies, is:

```bash
cd /tmp/arc-vsa-run/ARC-Development
python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/arcagi1/training/007bbfb7.json")
task = json.loads(path.read_text())
assert task["train"] and task["test"]
assert all("output" in pair for pair in task["test"])
print(path, len(task["train"]), len(task["test"]))
PY
```

After obtaining the exact `sspspace`, this import-and-construction check is the
smallest solver smoke. It does not call `solve_task`; allow substantial memory
because `vsa.py` constructs 4,096-dimensional spaces and fits a Ridge model at
import time:

```bash
cd /tmp/arc-vsa-run/ARC-Development
timeout 120s env PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  PYTHONPATH=/workspace/arc-agi/external/ARC-VSA-2025/src \
  python - <<'PY'
import json
from objobj_solver import ObjObjSolver

with open("data/arcagi1/training/007bbfb7.json") as stream:
    solver = ObjObjSolver(json.load(stream))
assert solver.n_demonstrations > 0 and solver.n_queries > 0
print(solver.n_demonstrations, solver.n_queries)
PY
```

Exit status 124 means the import exceeded the smoke limit, not that it passed.
Do not run the CLI solver until this construction check succeeds.

### Additional compatibility notes

- `src/test.py` permits `SceneObjSolver` in argparse but has no such entry in
  its solver map; only select `ObjObjSolver`.
- `ARCSolver` requires an `output` for every test pair, so the GitHub runner is
  suitable for the labeled local training/evaluation files, not unlabeled
  competition challenges. The linked Kaggle wrapper has a different API for
  those challenges.
- The runner catches broad exceptions, converts them to zero accuracy, and can
  still exit successfully. Inspect output; process exit code alone is not a
  valid smoke result.
- The internal timeout is 1,000 seconds per task. A real one-task run is not a
  short smoke and was not attempted.
- All dependency versions except those observed in the Kaggle notebook are
  unspecified, so a reproducible lock remains unavailable from upstream.

## Storage summary

The retained upstream files total 4,501,992 regular-file bytes: 4,323,833 for
CompressARC and 178,159 for ARC-VSA-2025. CompressARC contributes 24 files and
ARC-VSA contributes 13. The source tarballs totaled 78,102,614 compressed bytes
and were removed after checksum and manifest verification. Filtering omitted
141,992,993 uncompressed bytes of generated results, publication assets, and a
README banner. No model checkpoint or environment was added.

Acquisition details and verification hashes are in `external/SOURCES.md`.
