# Reproduction Matrix

Audit date: 2026-08-04

This matrix turns the 24-paper review into a durable execution backlog. Source
snapshots for CompressARC and ARC-VSA-2025 have since been pinned and filtered.
CompressARC passed a one-task forward smoke; ARC-VSA-2025 is blocked by a
missing upstream dependency. No paper benchmark or full reproduction has been
completed.

The target host is one NVIDIA RTX 5090 with 32 GiB VRAM and approximately
41 GiB free storage. Resource ranges below are planning estimates, not upstream
measurements. They should be replaced with measured peak values after each run.

## Reproduction Levels

- **Smoke (S):** install an isolated environment, load the smallest practical
  configuration, and complete one task, one batch, or an upstream self-test.
  A smoke pass establishes that the entry point and required services work; it
  says nothing about benchmark quality.
- **Benchmark (B):** run a declared public split or a fixed, named subset with
  the upstream evaluation semantics and record predictions, exact inputs,
  seeds, resource use, and failures. A reduced model, sample count, or search
  budget must be labeled `reduced`, never presented as a paper result.
- **Full reproduction (F):** use the paper's code revision, model/data versions,
  preprocessing, search or training budget, and scoring protocol, then compare
  against a stated paper table within a predeclared tolerance. If the original
  hardware, proprietary model snapshot, API, private split, or budget cannot be
  matched, the result is not a full reproduction.

Status vocabulary is `source-audited`, `not-started`, `running`, `passed`,
`failed`, `blocked`, or `not-applicable`. Every S/B/F status in this document is
currently `not-started` unless the row is explicitly blocked.

## Artifact And Environment Audit

`External` means that an upstream link exists but the artifact is not in this
workspace. `None needed` means the method trains from scratch or uses an API and
does not publish a method checkpoint. Branch names were checked against the
linked public repositories on the audit date; an actual run must pin a commit.

| # | Candidate | Availability | Verified source and default branch | Code | Checkpoint | Data | API | Python/environment needs | Principal blocker |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BARC | Public candidate | [xu3kev/BARC](https://github.com/xu3kev/BARC), `master` | Available | External HF models and TTT LoRAs | Seeds bundled; synthetic and supplementary sets external on HF | Optional for generation; local inference possible | Python 3.10, PyTorch 2.4, FlashAttention; separate vLLM 0.6.0/0.5.4 paths | Paper-scale fine-tuning uses an 8-process ZeRO-3 recipe; model/data caches exceed the storage budget |
| 2 | LPN | Public candidate | [clement-bonnet/lpn](https://github.com/clement-bonnet/lpn), `main` | Available | No paper checkpoint advertised in README | Generated through re-ARC utilities; external inputs | HF and W&B credentials used by the training workflow | JAX, Hydra, Python environment from requirements; upstream also has TPU setup | Published training budget and accelerator topology are not a one-GPU match |
| 3 | ARChitects 2024 | Public candidate | [da-fr/arc-prize-2024](https://github.com/da-fr/arc-prize-2024), `main` | Available | External 4-bit 8B checkpoint on HF | ARC/Kaggle inputs external | None | Historical Unsloth/CUDA stack; upstream reports 24 GiB can work at batch size 2 | Exact 2024 package stack and full base-model retraining are storage-sensitive |
| 4 | GridCoder2024 | Public candidate | [SimonOuellette35/GridCoder2024](https://github.com/SimonOuellette35/GridCoder2024), `main` | Available, proof-of-concept | External `model_full.pth` on Kaggle | ARC plus separately installed `ARC_gym` | None | PyTorch; upstream does not pin a complete environment | Evaluator only runs one named task at a time and only claims the DSL-solvable subset |
| 5 | 2D nGPT | Public candidate | [jfpuget/ARC-AGI-Challenge-2024](https://github.com/jfpuget/ARC-AGI-Challenge-2024), `main` | Available | Repository checkpoint area exists; reusable artifact needs verification | ARC external; re-ARC generation required for training | None | NVIDIA PyTorch NGC 24.09, PyTorch, NumPy, einops, rotary-embedding-torch | Training duration and checkpoint provenance are not fully specified; generated data can exceed local storage |
| 6 | TinyRecursiveModels | Public candidate, archived | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), `main` | Available, read-only archive | No paper checkpoint advertised in README | Builders included; ARC and augmented data must be prepared | W&B optional | Python 3.10, CUDA 12.6, PyTorch nightly, `adam-atan2` | ARC runs are reported as about three days on four H100s, not one 5090 |
| 7 | SOAR | Public candidate | [flowersteam/SOAR](https://github.com/flowersteam/SOAR), `main` | Available | Five external HF models, 7B through 123B | External 5M-solution HF dataset | None for local models; W&B optional | Python 3.11; separate SGLang inference and Unsloth/CUDA training environments | Even the smallest model plus data approaches the disk limit; full evolutionary training is far beyond one GPU |
| 8 | CompressARC | Public candidate | [iliao2345/CompressARC](https://github.com/iliao2345/CompressARC), `master` | Available | None needed; per-task training from scratch | ARC-format dataset and prior result files bundled | None | Requirements-based PyTorch environment; upstream demonstrates RTX 4070 | Forward smoke passed; full splits still require many independent per-task runs |
| 9 | ARC-VSA-2025 | Public candidate | [ijoffe/ARC-VSA-2025](https://github.com/ijoffe/ARC-VSA-2025), `main` | Available | No checkpoint advertised | Solver source and competition layout available; verify input wiring | None | PyTorch, nengo-spa, NumPy/SciPy/scikit packages from requirements | Solver imports an unpublished/unidentified `sspspace` implementation |
| 10 | arc-lang-public | Public candidate | [jerber/arc-lang-public](https://github.com/jerber/arc-lang-public), `main` | Available | None needed; provider models | ARC data bundled | Required for selected provider; OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, or xAI | Python 3.12+, `uv`, provider keys, required `MAX_CONCURRENCY`; Postgres optional | Cost, quotas, and mutable proprietary model snapshots prevent strict paper parity |
| 11 | epang080516/arc_agi | Public candidate | [epang080516/arc_agi](https://github.com/epang080516/arc_agi), `main` | Available | Included saved program library; no local LLM checkpoint | ARC-AGI-1/2 layouts and test data included | Required for chosen frontier model, notably xAI for reported approach | Python 3.11 and pinned requirements; Dockerfile also supplied | Reported performance depends on paid, changing frontier APIs and ordering of an evolving library |
| 12 | ARC_NCA | Public candidate | [etimush/ARC_NCA](https://github.com/etimush/ARC_NCA), `main` | Available, notebook-centric | No checkpoint advertised | Parsing notebooks expect ARC data | None | Jupyter/PyTorch; code hard-codes `cuda:0`; package versions are not pinned | No requirements file or single benchmark entry point; notebooks need manual parameter capture |
| 13 | ArcMemo | Public candidate | [matt-seb-ho/arc_memo](https://github.com/matt-seb-ho/arc_memo), `main` | Available | None needed; uses provider models | Concept annotations and helper pipeline bundled | Required for reported o4-mini/GPT-4.1 experiments | Python 3.11, requirements, Hydra, provider credentials | Proprietary model snapshots, inference spend, and continual-memory ordering affect reproducibility |
| 14 | NVARC | Public candidate | [1ytic/NVARC](https://github.com/1ytic/NVARC), `main` | Available with seven submodules | External Kaggle notebooks/assets; component checkpoints vary | External 103k/3.2M synthetic sets plus ARC | Optional/phase-specific for synthetic generation | Multiple component environments: ARChitects/Unsloth and TRM stacks | Full ensemble and synthetic corpus exceed 41 GiB and combine several independently complex systems |
| 15 | LatentMAS | Public candidate | [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), `main` | Available | External Qwen/HF base models; method is training-free | HF benchmark loaders; one MedQA example bundled | None for HF path | Python 3.10, Transformers; optional modified vLLM path | Official 14B hybrid path recommends two GPUs; one-GPU HF runs are reduced configurations |
| 16 | AgentPrimitives | Public candidate, incomplete runner | [haibojin001/AgentPrimitives](https://github.com/haibojin001/AgentPrimitives), `main` | Partial reference implementation | External HF backbone; no method checkpoint | Data/config directories present | None for local HF path | Python 3.10 and requirements | README says the clean end-to-end demo and experiment pipelines are still forthcoming |
| 17 | GraphPlanner | Public candidate | [ulab-uiuc/GraphPlanner](https://github.com/ulab-uiuc/GraphPlanner), `main` | Available | No trained router checkpoint advertised | Data/config schemas and repository data present | NVIDIA NIM/OpenAI-compatible API required for agents | Python 3.10+, PyTorch, PyG/scatter, Transformers, W&B optional | API access and exact interaction history are part of training; documented CUDA wheels are old and need isolation |
| 18 | RouteMoA | Public candidate | [Jize-W/RouteMoA](https://github.com/Jize-W/RouteMoA), `main` | Available | External router weights and mDeBERTa; base LLMs external | OpenCompass and a self-contained 450-item result/eval set available | Large-pool route and some judging require APIs | Python 3.10, Linux x86_64, LMDeploy/OpenCompass | Local paper setup used five A800 80 GiB GPUs; API reproduction has material cost and model-version drift |
| 19 | MACA | Public candidate, sparse docs | [However-Li/Multi-Agent-Coordination-Adaptation-via-Structure-Guided-Orchestration](https://github.com/However-Li/Multi-Agent-Coordination-Adaptation-via-Structure-Guided-Orchestration), `main` | Available in one initial commit | No checkpoint advertised | Dataset configs and preparation script present | OpenAI package is present; exact service requirements undocumented | Unpinned requirements include PyTorch, Transformers, sentence-transformers, and `verl` | README contains only the title; no documented command, expected output, model, or hardware budget |
| 20 | MARC | Partial/complex | [ekinakyurek/marc](https://github.com/ekinakyurek/marc), `main` | Available with custom submodules/forks | External HF 8B bases, fine-tunes, LoRAs, and predictions | ARC Prize 2024 data external | None | Python 3.10; custom torchtune; nightly PyTorch/torchao cu121; separate custom-vLLM environments | Upstream warns the repo is still being cleaned; training and inference require incompatible environments and manual patches |
| 21 | Omni-ARC | Unavailable/blocked | [solution write-up](https://ironbar.github.io/arc24/05_Solution_Summary/); no verified public repository/default branch | Not found | Not found | Not packaged for this tracker | Unknown | Not specified | Paper/write-up is available, but the audited implementation is not |
| 22 | Mini-ARC transformer | Unavailable/blocked | [paper PDF](https://www.paulfletcherhill.com/mini-arc.pdf); no verified public repository/default branch | Not found | Not found | Not packaged for this tracker | Unknown | Not specified | Paper is available, but no runnable transformer implementation was verified |
| 23 | NeuroMAS | Unavailable/blocked | [arXiv:2605.16757](https://arxiv.org/abs/2605.16757); no verified public repository/default branch | Not found | Not found | Not found | Unknown | Not specified | No public implementation was verified during the audit |
| 24 | ReM-MoA | Unavailable/blocked | [arXiv:2606.24437](https://arxiv.org/abs/2606.24437); no verified public repository/default branch | Not found | Not found | Not found | Unknown | Not specified | No public implementation was verified during the audit |

## One-GPU Resource And Priority Matrix

Each resource cell is `incremental disk GiB / peak VRAM GiB / wall time` on
the target host. `API` means local VRAM is negligible but elapsed time, quota,
and spend still have to be reported. `>32` or `>41` is a host blocker unless the
method is reduced or artifacts are moved to approved storage after checking
`/model`. Runtime ranges assume the upstream path works; environment repair time
is excluded.

| Candidate | Smoke estimate and feasibility | Benchmark estimate and feasibility | Full-reproduction estimate and feasibility | Phase | Current S/B/F status |
| --- | --- | --- | --- | ---: | --- |
| CompressARC | `<2 / 2-8 / 0.3-1 h`, feasible | `2-10 / 8-20 / 2-7 d`, feasible but long | `5-20 / 8-24 / 4-14 d`, possible, protocol must be pinned | 1 | passed / not-started / not-started |
| ARC-VSA-2025 | `<2 / 0-8 / 0.5-2 h`, blocked by missing dependency | `2-8 / 4-16 / 1-3 d`, blocked | `5-20 / 8-24 / 2-7 d`, blocked | 1 | blocked / blocked / blocked |
| ARC_NCA | `<2 / 2-8 / 0.5-2 h`, likely after notebook repair | `2-10 / 8-24 / 1-3 d`, likely | `5-20 / 8-24 / 3-14 d`, under-specified | 1 | not-started / not-started / not-started |
| GridCoder2024 | `1-3 / 4-12 / <1 h`, feasible with external weight | `2-10 / 8-24 / 1-24 h`, fixed solvable subset only | `10-40 / 24-32 / 1-7 d`, training details incomplete | 1 | not-started / not-started / not-started |
| 2D nGPT | `1-5 / 8-16 / 1-6 h`, likely | `5-20 / 24-32 / 6-24 h`, tight | `20-80 / >=32 / 1-7 d`, disk/VRAM risk and under-specified | 1 | not-started / not-started / not-started |
| LPN | `1-3 / 4-12 / 0.5-2 h`, likely | `10-30 / 24-32 / 1-4 d`, reduced single-GPU run | `>41 / multi-device or TPU / unknown`, not compute-matched | 1 | not-started / not-started / not-started |
| ARChitects 2024 | `10-20 / 12-20 / 0.5-2 h`, feasible with 4-bit model | `18-30 / 20-32 / 6-12 h`, tight but plausible | `30-60 / 24-80 / 1-4 d`, storage and exact-stack risk | 2 | not-started / not-started / not-started |
| BARC | `1-5 / 0-8 / 0.5-2 h` without full model, feasible | `18-35 / 18-30 / 6-24 h`, selected 8B path only | `>100 / multi-GPU / days`, blocked on host | 2 | not-started / not-started / not-started |
| arc-lang-public | `<1 / 0 / 0.5-2 h + API`, feasible with key | `1-5 / 0 / 6-48 h + API`, quota/cost limited | Similar disk, but exact proprietary API snapshot unavailable | 2 | not-started / not-started / not-started |
| epang080516/arc_agi | `<1 / 0 / 0.5-2 h + API`, feasible with key | `1-5 / 0 / 2-24 h + API`, quota/cost limited | API and library-order parity not stable | 2 | not-started / not-started / not-started |
| ArcMemo | `<2 / 0 / 1-4 h + API`, feasible with key | `1-5 / 0 / 1-7 d + API`, spend limited | Proprietary model snapshot and full rollout spend block strict parity | 2 | not-started / not-started / not-started |
| TinyRecursiveModels | `<3 / 4-12 / 0.5-2 h`, feasible | `15-35 / 24-32 / 1-4 d`, reduced | `50-150 / >32 / ~3 d on 4 H100`, blocked on host | 3 | not-started / not-started / not-started |
| SOAR | `16-25 / 16-28 / 1-4 h`, smallest model only | `25-41 / 24-32 / 1-3 d`, storage-tight reduced search | `>500 / multi-GPU / weeks`, blocked on host | 3 | not-started / not-started / not-started |
| NVARC | `2-10 / 8-20 / 1-4 h`, component smoke only | `20-41 / 24-32 / 1-4 d`, one component at a time | `>100 / >32 / weeks`, ensemble/corpus blocked | 3 | not-started / not-started / not-started |
| MARC | `20-35 / 20-32 / 2-6 h`, storage/VRAM tight | `25-41 / 24-32 / 1-7 d`, selected tasks only | `>41 / >32 / multi-environment days`, blocked/complex | 3 | not-started / not-started / not-started |
| LatentMAS | `10-20 / 10-20 / 1-3 h`, 4B path | `15-35 / 18-32 / 6-48 h`, 4B/8B reduced | `35-80 / two GPUs / 1-5 d`, official hybrid path blocked | 3 | not-started / not-started / not-started |
| AgentPrimitives | `10-20 / 12-24 / 1-3 h`, source-level only | Unknown; blocked by missing end-to-end runner | Unknown; experiment pipeline not released | 3 | not-started / blocked / blocked |
| GraphPlanner | `1-5 / 2-8 / 1-4 h + API`, likely | `2-10 / 4-16 / 1-4 d + API`, cost/history limited | Unknown API spend and interaction corpus; not compute-matched | 3 | not-started / not-started / blocked |
| RouteMoA | `1-5 / 0-8 / 1-4 h`, precomputed eval/service smoke | `2-10 / 0-12 / 1-4 d + API`, or local models exceed disk | `>100 / five 80 GiB GPUs / days`, blocked on host | 3 | not-started / not-started / blocked |
| MACA | `1-5 / 0-16 / 1-4 h`, tests/source only | Unknown; no documented benchmark command | Unknown; GRPO/VERL budget and checkpoints absent | 3 | not-started / blocked / blocked |
| Omni-ARC | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |
| Mini-ARC transformer | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |
| NeuroMAS | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |
| ReM-MoA | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |

## Phased Execution Order

### Phase 0: Foundation And Source Locks

- Reconfirm free space and inspect `/model` before retrieving any artifact.
- Pin every selected repository to a commit and record repository size before
  checkout. A branch name alone is not a reproducible source lock.
- Validate the local ARC snapshots and freeze task-ID lists, Top-K semantics,
  seeds, and output normalization before comparing solvers.
- Create one isolated environment per candidate or explicitly documented group;
  do not upgrade the evaluator environment to satisfy a baseline.

### Phase 1: Local, Low-Artifact Smokes

Order: CompressARC, ARC-VSA-2025, ARC_NCA, GridCoder2024, 2D nGPT, then LPN.
These maximize information per downloaded GiB and establish PyTorch, notebook,
JAX, checkpoint, and program-search paths. Advance a candidate only after its
single-task output can be normalized and scored independently.

### Phase 2: ARC LLM And API Baselines

Order: ARChitects 2024, BARC, arc-lang-public, epang080516/arc_agi, then
ArcMemo. Prefer published quantized checkpoints. For API methods, set a hard
request/token/currency cap before the first request and store provider, exact
model identifier, request parameters, and returned request IDs.

### Phase 3: Heavy Or Integration-Risk Methods

TinyRecursiveModels, SOAR, NVARC, MARC, LatentMAS, AgentPrimitives,
GraphPlanner, RouteMoA, and MACA enter only after a written storage/compute
budget review. Reduced runs are useful engineering results but must be reported
as reduced. Full reproduction for several of these methods is impossible on the
target host without additional GPUs, storage, paid APIs, or missing upstream
assets.

### Phase 4: Watchlist

Omni-ARC, the Mini-ARC transformer implementation, NeuroMAS, and ReM-MoA stay
blocked. Re-audit quarterly or when authors announce code. Do not substitute an
unverified third-party repository and retain the original paper/write-up URL.

## Interpretation Rules

- ARC-AGI-1 and ARC-AGI-2 scores are separate results. Public, semi-private,
  private, training, and evaluation splits are never interchangeable.
- Native non-ARC methods such as LatentMAS, AgentPrimitives, GraphPlanner,
  RouteMoA, MACA, NeuroMAS, and ReM-MoA must first be reproduced on their paper
  benchmarks. Any later ARC adaptation is a new experiment, not paper parity.
- A published prediction file can validate a scorer but is not an inference
  reproduction. A published checkpoint run is not a training reproduction.
- API reruns are time-stamped replications unless the provider guarantees the
  same immutable model and decoding implementation.
- Missing task outputs score as missing under this repository's scorer; never
  report accuracy only over tasks for which a method returned a candidate.
- Record failed tasks, timeouts, OOMs, retries, filtered task IDs, and manual
  interventions. Excluding them after execution invalidates the benchmark.

The machine-readable counterpart to this document is
[`configs/baselines.json`](../configs/baselines.json). The execution policy and
acceptance criteria are in [`LONG_GOAL.md`](../LONG_GOAL.md).
