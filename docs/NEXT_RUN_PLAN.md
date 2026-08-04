# Next-run plan for ARC_NCA, GridCoder2024, and 2D nGPT

This plan covers the pinned, storage-filtered snapshots in `external/ARC_NCA`,
`external/GridCoder2024`, and `external/ARC-AGI-Challenge-2024`. Acquisition
installed no package, downloaded no model checkpoint, invoked no upstream
module, and used no CUDA device. Only CPU-side source, notebook, archive, and
canonical-data checks were run.

## Preflight status

| Item | Observed result |
| --- | --- |
| Host Python | CPython 3.12.11 |
| Python source syntax | All 36 retained `.py` files parse |
| Notebook structure | All 8 notebooks are valid nbformat 4 JSON |
| Notebook code | 58 of 61 code cells parse as plain Python; the other 3 start with valid Jupyter `%` or `!` syntax |
| Canonical data | 1,920 ARC-AGI-1/2 tasks validate |
| External files | No `.pt`, `.pth`, `.ckpt`, `.pdf`, archive, media, NumPy result, or Git LFS pointer retained |
| Retained source | 50 files and 2,801,767 regular-file bytes |

The target GPU documented in `FIRST_RUN_PLAN.md` is an RTX 5090 (`sm_120`). It
was not queried or used in this preflight because a CompressARC run was active.
For all three methods, first prove a one-tensor operation with the selected
Blackwell-capable PyTorch build before importing upstream code. The PyTorch 2.7
CUDA 12.8 wheel used for the prior CompressARC smoke is the earliest stable
upstream combination already identified for this host; reusing that version is
a compatibility experiment, not an upstream-exact environment.

## Recommended order

Run ARC_NCA next. It has no external model or synthetic corpus, its Apache-2.0
license is clear, and a one-grid forward pass can isolate environment and
Blackwell issues without entering a training notebook. Do not start its 3,000
step loop until that smoke passes and task-index ordering is made explicit.

GridCoder2024 is second only after licensing and dependency decisions. Its
official checkpoint is available but large. 2D nGPT is third: the checked-in
workflow does not provide the referenced pretrained checkpoint or exact TTT
source version, and paper-scale data generation is the largest storage risk.

## ARC_NCA

### Actual entry points

There is no package metadata, requirements file, CLI, or test suite. The README
identifies five notebooks:

- `Training_Ignore_Size_change.ipynb` filters out shape-changing tasks, then
  trains one `CA` per retained task for 3,000 optimizer steps.
- `Training_Padd_to_Size.ipynb` pads grids to 32x32 and defaults to three
  `EngramNCA_v3` tasks, again at 3,000 optimizer steps each.
- `Visualize.ipynb` and `Visualize_Padded.ipynb` require generated
  `TrainedARCModels/<class>/problem_<index>.pth` files and write video output.
- `Data_parsing.ipynb` analyzes generated loss JSON.

The actual imports require PyTorch, NumPy, Matplotlib, OpenCV (`cv2`), and
IPython/Jupyter. Nothing is version-pinned. An upstream notebook traceback
shows Python 3.12 and Matplotlib 3.9.2, but does not identify the Torch, NumPy,
OpenCV, or IPython versions and ends in a user `KeyboardInterrupt`; it is not a
record of a completed baseline run.

### Data reuse

The notebooks expect task-per-file JSON at CWD-relative
`ArcData/data/training` and `ArcData/data/evaluation`. Canonical ARC-AGI-1 can
therefore be reused without conversion by placing this symlink in a disposable
run overlay:

```bash
ln -s /workspace/arc-agi/third_party/arc-agi-1 /tmp/arc-nca-run/ArcData
```

There are three correctness hazards before a benchmark:

- Both training notebooks call `import_data(training_path)` only. The returned
  variables named `eval_in` and `eval_out` are the `test` examples from each
  training task; the declared `eval_path` is never read.
- `import_data` iterates unsorted `os.listdir()` output and checkpoints by
  integer `problem_<index>`. Index-to-task mapping is not reproducible unless
  the run captures or fixes the ordered filename list.
- Both training notebooks call `make_path("LossesData")` and then
  `create_empty_json(...)`. That helper writes a file only when its parent
  directory does not exist, so on a clean run the subsequent JSON read fails.
  Fix this in a disposable run copy before attempting optimization.

ARC-AGI-2 has the same task-per-file shape and could be symlinked for a new
experiment, but that would not reproduce the ARC_NCA workflow and must be
reported separately.

### Minimal smoke

After creating a dedicated environment with NumPy and a Blackwell-capable
Torch, this one-grid forward pass is the smallest useful smoke. It performs no
optimization and writes nothing. It cannot run on CPU without changing
upstream because `NCA.py`, `arc_agi_utils.py`, and `vft.py` hard-code
`cuda:0`, including tensors allocated at module import.

```bash
cd /workspace/arc-agi/external/ARC_NCA
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
from pathlib import Path

import torch
import arc_agi_utils as aau
from NCA import CA

path = Path("../../third_party/arc-agi-1/data/training/007bbfb7.json")
task = json.loads(path.read_text())
grid = torch.tensor(task["train"][0]["input"], device="cuda:0")
x = aau.arc_to_nca_space(10, grid, 50, 25, device="cuda:0").unsqueeze(0)
model = CA(50, 264).to("cuda:0").eval()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
assert y.shape == x.shape
print(path.stem, tuple(y.shape), sum(p.numel() for p in model.parameters()))
PY
```

Static parameter accounting gives 312,264 parameters for the notebook's
`CA(50, 264)`, or 1,249,056 raw FP32 parameter bytes. The padded notebook's
`EngramNCA_v3(50, 132, 132, 25)` has 223,112 parameters and 892,448 raw
FP32 parameter bytes. A state-dict checkpoint should be roughly 0.9-1.3 MB plus
small serialization overhead; upstream publishes no checkpoint.

Do not use either training notebook as the first smoke. Each allocates pools of
1,024 states per demonstration, unrolls 32-63 NCA steps per optimizer step,
evaluates every step, writes loss JSON and checkpoints, and uses task indices
rather than stable task IDs.

## GridCoder2024

### Actual entry points

`generate_training_data_full.py` writes CWD-relative
`training_data_atomic.csv` and `validation_data_atomic.csv` using 1,000,000 and
50,000 generated tasks. `train_full.py` reads those files, trains ten epochs on
CUDA with batches of 500, and writes `model_full.pth`. `test_gridcoder.py` is
the evaluator and imports `search/p_star_superposition.py`; it requires an
explicit `--task`, defaults to a 300-second search budget, and otherwise scans
the full evaluation loader until that filename appears.

The source directly requires PyTorch, NumPy, tqdm, and the separately installed
`SimonOuellette35/ARC_gym` package. ARC_gym in turn declares NumPy,
Matplotlib, tqdm, and SciPy. Neither repository version is pinned by
GridCoder2024, and the current ARC_gym repository also has no license file.
The historical Kaggle model bundle contains an ARC_gym copy, but retrieving the
bundle would also retrieve the prohibited 1.81 GB checkpoint; it was not
downloaded.

### Checkpoint and license

The README's official Kaggle model is
<https://www.kaggle.com/models/simonouellette/gridcoder-2024/PyTorch/default/1>.
Kaggle's read-only API reports:

| Artifact | Exact uncompressed bytes | License reported by Kaggle |
| --- | ---: | --- |
| `Kaggle_code/GridCoder_kaggle/model_full.pth` | 1,809,563,648 | CC0 model bundle |
| Complete model version | 1,813,720,624 | CC0 |

The checked-in architecture has 452,314,215 parameters by static accounting,
or 1,809,256,860 raw FP32 parameter bytes; the published checkpoint size is
consistent with a state dict plus buffers and serialization overhead. No
checkpoint was downloaded. The repository source itself remains unlicensed,
irrespective of the separate Kaggle model license. Obtain author clarification
before redistributing or modifying it.

### Data reuse

Both GridCoder's synthetic-data class and ARC_gym's evaluation loader use
CWD-relative `ARC/data/training` and `ARC/data/evaluation`. Canonical ARC-AGI-1
is directly reusable through a symlink in a disposable run overlay; no format
conversion is needed. All 49 task IDs claimed in GridCoder's README are present
in the canonical ARC-AGI-1 evaluation split.

```bash
ln -s /workspace/arc-agi/third_party/arc-agi-1 /tmp/gridcoder-run/ARC
```

ARC-AGI-2 is not a paper-parity replacement. Its current training set includes
ARC-AGI-1 material, but evaluating GridCoder there would be a separate protocol
and risks train/evaluation contamination if split provenance is ignored.

### Minimal smokes

After pinning and installing the exact ARC_gym revision to use, an
architecture-only smoke can run without data or checkpoint. It still constructs
a 452-million-parameter model and therefore is intentionally GPU-only here.

```bash
cd /workspace/arc-agi/external/GridCoder2024
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import torch
from model.LVM import LVM

model = LVM(13, 103, emb_dim=512, max_seq_length=40).to("cuda").eval()
x = torch.zeros((1, 13, 30, 30), device="cuda")
t = torch.tensor([[3]], device="cuda")
with torch.no_grad():
    p = model.predict(x, x, t)
torch.cuda.synchronize()
assert p.shape == (1, 103)
print(sum(v.numel() for v in model.parameters()), tuple(p.shape))
PY
```

Only after the author-license decision and explicit approval to spend 1.81 GB
on the checkpoint should the smallest method smoke be attempted from a run
overlay containing `ARC`, `model_full.pth`, and the pinned ARC_gym environment:

```bash
cd /tmp/gridcoder-run
PYTHONDONTWRITEBYTECODE=1 timeout 30s python \
  /workspace/arc-agi/external/GridCoder2024/test_gridcoder.py \
  --task 1990f7a8.json --dataset eval --time_budget 5
```

The evaluator moves the model and all task tensors to CUDA unconditionally. It
keeps the model in training mode because the author reports worse results in
evaluation mode, and the search includes random sampling; preserve its seed
state and stdout. A timeout or a process exit without `Success!` is not a pass.

## 2D nGPT

### Actual entry points

The executable files are monolithic `code/053.py` and `code/064.py`. Both load
a Python config given by `--cfg`, accept config overrides as unknown CLI pairs,
force CUDA in `run()`, and optionally start NCCL DDP. `053.py` is the training
version; `064.py` adds test-time tuning and prediction.

The README says the code ran in NVIDIA `nvcr.io/nvidia/pytorch:24.09-py3` with
`rotary_embedding_torch` added. Actual imports also require NumPy, pandas,
SciPy, tqdm, einops, and PyTorch. No versions are pinned. NVIDIA's 24.09 release
notes identify Python 3.10, CUDA 12.6.1, and a PyTorch 2.5.0 alpha build. That
historical stack predates Blackwell support and should not be used unchanged on
the RTX 5090. A Python 3.12 plus PyTorch 2.7/CUDA 12.8 environment is plausible
but is an unverified compatibility deviation, especially for
`rotary_embedding_torch` and the Torch parametrization/checkpoint keys.

The checked-in reproduction is internally inconsistent:

- `notebooks/train.ipynb` launches eight DDP processes with `code/053.py` and
  `cfg_053`, then records checkpoint `exp_50/gen10000_0.pt`.
- `notebooks/ttt.ipynb` launches four processes with absent `code/063.py` and
  absent `cfg/cfg_063.py`, loading a local `../checkpoints/ngc/exp_50.pt`.
- The README says version 64 should behave like version 63, but `cfg_064.py`
  instead defaults to absent `../checkpoints/ngc/exp_54.pt`.
- No checkpoint URL, checksum, exact re-ARC revision, requirements lock, or
  complete single-GPU reproduction command is published in this repository.

The default large config has 38,046,722 parameters, or 152,186,888 raw FP32
parameter bytes. The training notebook's `--task_embed_size 8` override reduces
that to about 37,861,634 parameters and 151,446,536 raw FP32 parameter bytes.
Expect an approximately 152 MB model-only checkpoint plus serialization
overhead. Optimizer state is not saved. This is a static estimate because the
referenced `exp_50.pt` is not published here.

### Data reuse

The code requires two distinct layouts:

- `cfg.input_path` defaults to `../input/arc-prize-2024/` and expects Kaggle
  aggregate challenge/solution JSON plus `fixed_size.pkl`.
- `cfg.data_path` defaults to `../re-arc/gen10000/tasks/` and expects one JSON
  array per generated training task, with 10,000 input/output examples.

Canonical ARC-AGI-1 can be converted losslessly to the four public training and
evaluation aggregate JSON files. The already-retained CompressARC copies were
checked against all 800 canonical tasks and can be symlinked instead of
duplicated. The current canonical split has 262 training and 270 evaluation
tasks whose input/output dimensions match for every example; an ordered
`fixed_size.pkl` can be derived from those criteria. The upstream training log
reports 259 generated task files, so do not assume that a freshly derived
262-key list reproduces `exp_50`; generator availability or the historical data
snapshot accounts for at least three tasks and must be resolved.

Canonical ARC data cannot replace the generated re-ARC corpus by symlink. The
generation notebook calls re-ARC's `generate_dataset(..., n_examples=10000)`
for all 400 training tasks and records 6:19:30 generation time. Its training log
reports 41,440,000 augmented samples from 259 generated tasks. Upstream gives no
byte size; treat it as a multi-GB artifact and measure a small sample before
approving full generation. The competition `arc-agi_test_challenges.json` is
also outside the canonical public ARC-AGI-1 repository and remains subject to
the applicable competition data terms.

ARC-AGI-2 would require a new aggregate conversion, task embedding policy, and
training corpus. It is not directly reusable for the 2024 checkpoint protocol.

### Minimal smoke

After installing an isolated, Blackwell-capable dependency lock, use a 2x2
architecture forward before obtaining a checkpoint or generating data. This
imports the retained training source but does not call its CLI, train, save, or
data paths.

```bash
cd /workspace/arc-agi/external/ARC-AGI-Challenge-2024
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cfg python - <<'PY'
import importlib.util

import torch
from cfg_053 import cfg

spec = importlib.util.spec_from_file_location("ngpt053", "code/053.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cfg.device = torch.device("cuda")
module.cfg = cfg
model = module.ARCModel(cfg).to(cfg.device).eval()
batch = {
    "input": torch.tensor([[[0, 1], [2, 3]]], device=cfg.device),
    "output": torch.tensor([[[0, 1], [2, 3]]], device=cfg.device),
    "task": torch.zeros((1, 1, 1), dtype=torch.long, device=cfg.device),
    "sym": torch.zeros((1, 1, 1), dtype=torch.long, device=cfg.device),
    "perm_idx": torch.zeros((1, 1, 1), dtype=torch.long, device=cfg.device),
}
with torch.no_grad():
    logits = model(batch)
torch.cuda.synchronize()
assert logits.shape == (1, 2, 2, 10)
print(sum(p.numel() for p in model.parameters()), tuple(logits.shape))
PY
```

Do not run either notebook command next. The training command requests eight
GPUs and 2.59 million raw generated examples before augmentation; the TTT
command requests missing source/config/checkpoint artifacts. A defensible next
step after the architecture smoke is to ask the author for the exact
`exp_50.pt` checksum, `063` source/config, re-ARC commit, dependency lock, and
single-GPU batch/accumulation settings.

## Static checks

The following CPU-only checks were completed during acquisition:

- Archive SHA-256, retained-tree SHA-256, file counts, and regular-file sizes
  match `external/SOURCES.md`.
- Every retained file was byte-compared with its extracted upstream archive
  member.
- All retained Python files parse under Python 3.12.11 without importing them.
- Notebook JSON and plain-Python cells parse; only the expected Jupyter magic
  and shell-command cells are non-Python syntax.
- `python3 -m arc_agi_eval validate third_party/arc-agi-1/data
  third_party/arc-agi-2/data` validates 1,920 tasks.
- ARC-AGI-1 training/evaluation aggregates match canonical task content for all
  400 tasks in each split, and every GridCoder-declared task ID exists.

No runtime, training, inference, checkpoint load, CUDA query, or GPU operation
was performed for these three candidates.
