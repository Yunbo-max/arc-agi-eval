# Reproduction Matrix

Audit date: 2026-08-06

This matrix turns the 24-paper review into a durable execution backlog. The
current tally is **17/24 scope-limited smoke passes, 0/24 benchmark passes, and
0/24 full reproductions**. The smoke count includes compatibility,
architecture, static-schema, fixed-helper, source-data, and dry-run scopes; it
does not mean 17 entry points or ARC solvers have run. Nine passed auxiliary
records—RouteMoA's scorer, ARC-VSA-2025's blocker audit, the two Batch B gates,
the three Batch C gates, and the formal SOAR and NVARC static gates—are
explicitly excluded from the 17 smoke passes.
CompressARC and ARC_NCA have method-specific strict runtime
promotions (2/24), while performance eligibility remains 0/24. The current
five-batch split is in
[`EXECUTION_BATCHES.md`](EXECUTION_BATCHES.md).

The active target host is one NVIDIA RTX 3090 with 24 GiB VRAM and approximately
24.3 GiB free storage. Resource ranges below are planning estimates, not upstream
measurements. They should be replaced with measured peak values after each run.

## Reproduction Levels

- **Smoke (S):** complete one predeclared, bounded compatibility, architecture,
  component, schema, dry-run, task, batch, or upstream self-test scope. A pass
  establishes only the scope named in that row. In particular, a static/schema
  or fixed-helper pass does not establish that the solver entry point,
  dependencies, model, services, or benchmark path work.
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
`failed`, `blocked`, or `not-applicable`. The S/B/F columns are independent: a
component smoke can pass while benchmark execution remains blocked.

## Competition Classification Boundary

The [official ARC Prize 2025 results](https://arcprize.org/competitions/2025)
classify NVARC as first place on the ARC-AGI-2 private evaluation at 24.0%, and
SOAR as second place in the Paper Award; the released SOAR method primarily
targets ARC-AGI-1. These are external official classifications and results, not
scores reproduced by this repository. The formal local gates bind only their
locked workspace source and evidence, and neither gate ran a solver or validated
the official private evaluation or award placement.

Neither classification is a 2026 result. [ARC Prize
2026](https://arcprize.org/competitions/2026) is a separate, currently ongoing
competition, and this tracking inventory contains no verified 2026 submission.

## Artifact And Environment Audit

`External` means that an upstream link exists but the artifact is not in this
workspace. `None needed` means the method trains from scratch or uses an API and
does not publish a method checkpoint. Branch names were checked against the
linked public repositories on the audit date; an actual run must pin a commit.

| # | Candidate | Availability | Verified source and default branch | Code | Checkpoint | Data | API | Python/environment needs | Principal blocker |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BARC | Public candidate | [xu3kev/BARC](https://github.com/xu3kev/BARC), `master` | Available | Four external BF16 bases plus two TTT LoRAs; none is locally present or provenance-verified | Seeds bundled; synthetic and supplementary sets external on HF | Optional for generation; local inference possible | Python 3.10, PyTorch 2.4, FlashAttention; unpinned Git dependency and separate vLLM 0.6.0/0.5.4 paths | The bundled seed-generator/handwritten-program smoke passes but exercises no model ([smoke](../reports/barc/20260806-seed-00d62c1b-smoke/run.json)). A separate hardened static audit freezes eight root-license, base/LoRA provenance, safe-load, label-firewall, dependency, capacity, and prediction/parity blockers without reading ARC/answer or weight leaves or producing a prediction ([gate](../reports/barc/20260806-source-artifact-label-resource-gate-v1/run.json)) |
| 2 | LPN | Public candidate | [clement-bonnet/lpn](https://github.com/clement-bonnet/lpn), `main` | Available and revision-locked | Seven versioned W&B artifacts are named in source, but none is local or verified for bytes/hash/config/license | Generated through re-ARC utilities; challenge and solution JSON are colocated upstream | HF and W&B credentials used by the training workflow | JAX 0.4.26/Flax 0.8.4; upstream also has TPU setup | Official encoder/decoder and 400-generator tests pass locally. A hardened static gate parsed 21 locked Python files while reading no bundled ARC JSON, YAML, notebooks, bytecode, or checkpoints and executing no upstream code; it froze seven artifact/data/label/resource/staging blockers and produced no prediction ([evidence](../reports/lpn/20260806-source-artifact-data-label-gate-v1/run.json)). Test discovery exposes an upstream missing-`random` import. The official evaluation entry reads solutions after generation in the same process; `Evaluator.json_submission` is only a candidate challenge-only boundary until an adapter and exact artifact are audited |
| 3 | ARChitects 2024 | Public candidate | [da-fr/arc-prize-2024](https://github.com/da-fr/arc-prize-2024), `main` | Available | Exact 3,790,920,477-byte 4-bit checkpoint downloaded and hash-audited | ARC/Kaggle inputs external | None | Local compatibility stack differs from the historical Unsloth/PyTorch 2.4 stack | Checkpoint integrity passes ([integrity](../reports/architects-2024/20260806-4bit-checkpoint-integrity/run.json)), but the forward stopped before allocation because less than 10 GiB VRAM was free ([preflight](../reports/architects-2024/20260806-forward-preflight-gpu-occupied/run.json)). A separate hardened static audit freezes eight contamination, label-flow, code-isolation, dependency, capacity, license-review, and no-prediction blockers ([gate](../reports/architects-2024/20260806-source-artifact-label-runtime-gate-v1/run.json)); ARC-1 is contamination-aware only |
| 4 | GridCoder2024 | Public candidate | [SimonOuellette35/GridCoder2024](https://github.com/SimonOuellette35/GridCoder2024), `main` | Available as a byte-locked parent-repository snapshot; upstream commit chain is not locally verifiable | External `model_full.pth` on Kaggle; absent locally and no local SHA-256 lock | ARC plus ARC_gym pinned locally at `740b443a955cdb31ee8209ee4d74af87b027926e`; the default data overlay is absent | None | PyTorch/CUDA; pinned ARC_gym lacks `graphs.py` and `batching.py` | A synthetic-weight architecture forward passes (452.3M parameters, 1.88 GiB peak VRAM). A hardened tracked-file static audit read no ARC/checkpoint bytes, executed no upstream code, and produced no prediction, but froze seven blockers: source provenance, license, dependency/data, checkpoint, test-label flow into `yq`, CPU entrypoint, and 49-task subset coverage ([evidence](../reports/gridcoder2024/20260806-source-dependency-label-artifact-gate-v3/run.json)) |
| 5 | 2D nGPT | Public candidate | [jfpuget/ARC-AGI-Challenge-2024](https://github.com/jfpuget/ARC-AGI-Challenge-2024), `main` | Ten-file snapshot is byte-locked; upstream commit object is unavailable locally | Configs reference `exp_50.pt`/`exp_54.pt`, but neither checkpoint is present or hash-locked | ARC aggregates, re-ARC gen10000 data, and `fixed_size.pkl` are absent | None | NVIDIA PyTorch NGC 24.09, PyTorch, NumPy, einops, rotary-embedding-torch | Official large architecture forward passes with synthetic weights (38.05M parameters, 0.20 GiB peak VRAM). A hardened static audit executes no method and reads no forbidden artifact bytes, but freezes seven blockers: provenance, checkpoints, re-ARC, fixed-size/solution data, label flow, runtime portability, and reproduction contract ([evidence](../reports/2d-ngpt/20260806-source-artifact-label-runtime-gate-v1/run.json)). `064.py` injects evaluation solutions into test outputs that affect color augmentation, TTT, loss, and validation; it also uses direct `torch.load`, hard-coded CUDA, and global config |
| 6 | TinyRecursiveModels | Public candidate, archived | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), `main` | Available, read-only archive | No paper checkpoint advertised or present in the prepared asset status | Builders and bundled ARC/concept leaves exist, but dataset mapping/provenance and reuse terms are not cleared | W&B optional in source but no offline guard was detected | Python 3.10, CUDA 12.6, PyTorch nightly, `adam-atan2`; no hash-locked transitive closure | A random-weight 6,829,058-parameter CPU synthetic architecture forward passes with no ARC/checkpoint/training/GPU ([smoke](../reports/tiny-recursive-models/20260806-cpu-architecture-forward-smoke-retry2/run.json)). A separate metadata-first audit freezes ten artifact/data/label/overlap/dependency/isolation/runtime/selection/resource/no-prediction blockers and is auxiliary only ([gate](../reports/tiny-recursive-models/20260806-source-artifact-dataset-label-resource-gate-v1/run.json)). The method is a 2025 paper/method with a 2026 asset snapshot; no official ARC Prize entry is verified, and README 45%/8% values are unverified self-reports |
| 7 | SOAR | Public candidate; official ARC Prize 2025 Paper Award second place, chiefly ARC-AGI-1 | [flowersteam/SOAR](https://github.com/flowersteam/SOAR), `main` | Available | Five external HF models, 7B through 123B | External 5M-solution HF dataset | None for local models; W&B optional | Python 3.11; separate SGLang inference and Unsloth/CUDA training environments | A zero-dollar smoke opened only the bundled label-free ARC-AGI-1 evaluation challenges and called one fixed helper on synthetic trusted values ([smoke](../reports/soar/20260806-zero-dollar-source-data-smoke/run.json)). A separate [formal static gate](../reports/soar/20260806-source-artifact-dataset-label-api-code-resource-gate-v1/run.json) passed its audit while preserving 13 blockers; it imported/executed no upstream method, produced no solver prediction or accuracy, and granted no strict promotion or performance eligibility. The official 2025 award classification is external, not reproduced locally |
| 8 | CompressARC | Public candidate | [iliao2345/CompressARC](https://github.com/iliao2345/CompressARC), `master` | Available | None needed; per-task training from scratch | ARC-format dataset and prior result files bundled | None | Requirements-based PyTorch environment; upstream demonstrates RTX 4070 | Legacy single-task runs pass. A separate compatibility-deviant CPU method-specific strict smoke used an 11-file code-only stage, passed A/B hidden-label mutation with byte-identical predictions, and materialized run-local scoring payloads only after both inference processes exited ([evidence](../reports/compressarc/20260806-cpu-dev-3c9b0459-strict-v1/run.json)). It is trusted-code mechanism evidence, not environment/GPU parity, a benchmark, or a performance result |
| 9 | ARC-VSA-2025 | Public candidate | [ijoffe/ARC-VSA-2025](https://github.com/ijoffe/ARC-VSA-2025), `main` | Available | No checkpoint advertised | Solver source and competition layout available | None | PyTorch, nengo-spa, NumPy/SciPy/scikit packages from requirements | The source-only blocker audit found that `sspspace` is absent from requirements and PyPI and that upstream inference passes test outputs into object perception ([evidence](../reports/arc-vsa-2025/20260806-dependency-label-gate-audit-retry1/run.json)). No solver was imported or run, so this is not a smoke; a label-free adaptation would be a new experiment |
| 10 | arc-lang-public | Public candidate | [jerber/arc-lang-public](https://github.com/jerber/arc-lang-public), `main` | Available | None needed; provider models | ARC data bundled | Required for selected provider; OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, or xAI | Locked Python 3.12 `uv` environment; provider keys required for inference | A network-guarded import/config/Pydantic-parser component passed without an API call ([smoke](../reports/arc-lang-public/20260806-zero-dollar-import-smoke-retry1/run.json)); Pydantic's handling of extra raw keys means it did not prove a label firewall. A separate static gate freezes eight license, truth-flow/seam, provider-egress, budget, provenance, dependency, and no-prediction blockers ([gate](../reports/arc-lang-public/20260806-source-label-api-egress-gate-v1/run.json)). No provider run is authorized |
| 11 | epang080516/arc_agi | Public candidate | [epang080516/arc_agi](https://github.com/epang080516/arc_agi), `main` | Available | Included saved program library; no local LLM checkpoint | ARC-AGI-1/2 layouts and test data included | Required for chosen frontier model, notably xAI for reported approach | Python 3.11; full entry eagerly imports additional JAX/LPN/W&B/provider dependencies | A zero-dollar synthetic data-model/auditor-written trusted-executor component passed; it did not deserialize the pickle or execute model-generated code ([smoke](../reports/epang-arc-agi/20260806-zero-dollar-component-smoke/run.json)). A static gate freezes nine blockers, including eager labels/metrics, ARC-2-library overlap into ARC-1, pickle provenance, host-level generated-code execution, and absent cost fuse ([gate](../reports/epang-arc-agi/20260806-source-label-pickle-sandbox-api-gate-v1/run.json)) |
| 12 | ARC_NCA | Public candidate | [etimush/ARC_NCA](https://github.com/etimush/ARC_NCA), `main` | Available, notebook-centric | No checkpoint advertised | Parsing notebooks expect ARC data | None | Jupyter/PyTorch; code hard-codes `cuda:0`; package versions are not pinned | Reduced scripted smoke passes. A separate CPU-only method-specific strict smoke passed A/B hidden-label mutation with byte-identical predictions and post-inference independent scoring on one frozen dev task ([evidence](../reports/arc-nca/20260806-cpu-dev-6150a2bd-strict-v1/run.json)); analyst label exposure is disclosed, so this is mechanism evidence, not a benchmark. The adapter keeps the upstream CA architecture while placing the stochastic mask on the input device and importing only locked `arc_agi_utils.py` |
| 13 | ArcMemo | Public candidate | [matt-seb-ho/arc_memo](https://github.com/matt-seb-ho/arc_memo), `main` | Available | None needed; uses provider models | Concept annotations and helper pipeline bundled | Required for reported o4-mini/GPT-4.1 experiments | Pinned Python 3.11 dry-run environment; Hydra/provider credentials for inference | The two-problem pass was a no-memory generic-driver dry run with dummy completions—not a fixed-memory ArcMemo run or prediction ([smoke](../reports/arcmemo/20260806-native-dry-run-retry2/run.json)). A separate static gate freezes nine artifact/config, label, continual-oracle, budget, generated-code isolation, scoring, and no-prediction blockers ([gate](../reports/arcmemo/20260806-source-label-memory-api-sandbox-gate-v1/run.json)) |
| 14 | NVARC | Public candidate; official ARC Prize 2025 ARC-AGI-2 private-evaluation first place at 24.0% | [1ytic/NVARC](https://github.com/1ytic/NVARC), `main` | Available with seven submodules | External Kaggle notebooks/assets; component checkpoints vary | External 103k/3.2M synthetic sets plus ARC | Optional/phase-specific for synthetic generation | Multiple component environments: ARChitects/Unsloth and TRM stacks | A zero-dollar smoke verified locked component wiring/config, all seven gitlinks, and fixed helpers on synthetic trusted values ([smoke](../reports/nvarc/20260806-zero-dollar-component-source-smoke/run.json)). A separate [formal static gate](../reports/nvarc/20260806-source-gitlink-artifact-dataset-label-code-resource-gate-v1/run.json) passed its audit while preserving 12 blockers; it initialized no submodule or dataset, loaded no model/checkpoint, executed no returned code or solver, and granted no strict promotion or performance eligibility. The official 24.0% is external, not reproduced locally |
| 15 | LatentMAS | Public candidate | [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), `main` | Available | External Qwen/HF base models; method is training-free | HF benchmark loaders; one MedQA example bundled | None for HF path | Python 3.10, Transformers; optional modified vLLM path | The native role/prompt-schema smoke passes for AllenAI **AI2 ARC-Challenge multiple-choice QA**, not ARC-AGI grids ([evidence](../reports/latentmas/20260806-native-prompt-schema-smoke-retry1/run.json)). It loaded no dataset rows, answers, model, tokenizer, latent KV state, or inference path; any ARC-AGI adapter is a new experiment |
| 16 | AgentPrimitives | Public candidate, incomplete runner | [haibojin001/AgentPrimitives](https://github.com/haibojin001/AgentPrimitives), `main` | Partial reference implementation | External HF backbone; no method checkpoint | Data/config directories present | None for local HF path | Python 3.10 and requirements | A static YAML/AST smoke validates only generic MAS config and Organizer schema; its ARC reference is **AI2 ARC-Challenge**, not ARC-AGI ([evidence](../reports/agent-primitives/20260806-static-config-schema-smoke/run.json)). No upstream module or primitive ran. The root license and `run_demo.py` are absent, lowercase imports mismatch `Primitives/`, and Organizer construction requires Qwen3-8B |
| 17 | GraphPlanner | Public candidate | [ulab-uiuc/GraphPlanner](https://github.com/ulab-uiuc/GraphPlanner), `main` | Available | No trained router checkpoint advertised | Data/config schemas and repository data present | NVIDIA NIM/OpenAI-compatible API required for agents | Python 3.10+, PyTorch, PyG/scatter, Transformers, W&B optional | A zero-dollar smoke parses bundled router CSV schemas and one pure prompt formatter only ([evidence](../reports/graphplanner/20260806-zero-dollar-schema-smoke/run.json)). Its `arc_challenge` rows are **AI2 ARC science multiple choice**, not ARC-AGI; no router environment, weights, API, PPO, trained checkpoint, or ARC solver ran, and the root license is unresolved |
| 18 | RouteMoA | Public candidate | [Jize-W/RouteMoA](https://github.com/Jize-W/RouteMoA), `main` | Available | External router weights and mDeBERTa; base LLMs external | OpenCompass and a self-contained 450-item result/eval set available | Large-pool route and some judging require APIs | Python 3.10, Linux x86_64, LMDeploy/OpenCompass | A scorer-only auxiliary audit verifies labeled precomputed-file integrity and stored-score aggregation ([evidence](../reports/routemoa/20260806-zero-dollar-precomputed-scorer-audit/run.json)); it is not counted as a smoke and runs no inference, router, routed model, judge, API, or ARC solver. The source syntax audit also fails on one upstream file; paper setup used five A800 80 GiB GPUs |
| 19 | MACA | Public candidate, sparse docs | [However-Li/Multi-Agent-Coordination-Adaptation-via-Structure-Guided-Orchestration](https://github.com/However-Li/Multi-Agent-Coordination-Adaptation-via-Structure-Guided-Orchestration), `main` | Available in one initial commit | No checkpoint advertised | Dataset configs and preparation script present | OpenAI package is present; exact service requirements undocumented | Unpinned requirements include PyTorch, Transformers, sentence-transformers, and `verl` | A deterministic CPU smoke passes only through GraphSpec with random fallback embeddings and untrained weights ([evidence](../reports/maca/20260806-zero-dollar-graphspec-component-smoke/run.json)). It runs no GRPO, VERL, model/API, generated code, or benchmark; no checkpoint/license is bundled, most adapters are unimplemented, and two upstream files fail syntax audit |
| 20 | MARC | Partial/complex | [ekinakyurek/marc](https://github.com/ekinakyurek/marc), `main` | Available with custom submodules/forks | External HF 8B bases, fine-tunes, LoRAs, and predictions | ARC Prize 2024 data external | None | Python 3.10; custom torchtune; nightly PyTorch/torchao cu121; separate custom-vLLM environments | An ARC-native component smoke passes only task round-trip, two-attempt submission formatting, and fixed-candidate voting on one synthetic task ([evidence](../reports/marc/20260806-zero-cost-arc-components-smoke-retry1/run.json)). It loads no benchmark/test labels, model/checkpoint, torchtune, TTT, vLLM, or inference; the pinned torchtune submodule is uninitialized |
| 21 | Omni-ARC | Unavailable/blocked | [solution write-up](https://ironbar.github.io/arc24/05_Solution_Summary/); no verified public repository/default branch | Not found | Not found | Not packaged for this tracker | Unknown | Not specified | Paper/write-up is available, but the audited implementation is not |
| 22 | Mini-ARC transformer | Unavailable/blocked | [paper PDF](https://www.paulfletcherhill.com/mini-arc.pdf); no verified public repository/default branch | Not found | Not found | Not packaged for this tracker | Unknown | Not specified | Paper is available, but no runnable transformer implementation was verified |
| 23 | NeuroMAS | Unavailable/blocked | [arXiv:2605.16757](https://arxiv.org/abs/2605.16757); no verified public repository/default branch | Not found | Not found | Not found | Unknown | Not specified | No public implementation was verified during the audit |
| 24 | ReM-MoA | Unavailable/blocked | [arXiv:2606.24437](https://arxiv.org/abs/2606.24437); no verified public repository/default branch | Not found | Not found | Not found | Unknown | Not specified | No public implementation was verified during the audit |

## One-GPU Resource And Priority Matrix

Each resource cell is `incremental disk GiB / peak VRAM GiB / wall time` on
the target host. `API` means local VRAM is negligible but elapsed time, quota,
and spend still have to be reported. A plan above 24 GiB VRAM, or one that would
violate the 8 GiB reserve on the currently 24.3 GiB-free filesystem, is a host
blocker unless reduced or moved to approved storage. Runtime ranges assume the
upstream path works; environment repair time is excluded.

| Candidate | Smoke estimate and feasibility | Benchmark estimate and feasibility | Full-reproduction estimate and feasibility | Phase | Current S/B/F status |
| --- | --- | --- | --- | ---: | --- |
| CompressARC | compatibility-deviant CPU strict firewall smoke completed in 12.70 s; child-inclusive resources remain unmeasured | `2-10 / 8-20 / 2-7 d`, feasible but long | `5-20 / 8-24 / 4-14 d`, possible, protocol must be pinned | 1 | strict dev smoke passed / not-started / not-started |
| ARC-VSA-2025 | source-only dependency/label-gate audit passed, but no solver smoke; blocked by missing dependency and test-label dependence | `2-8 / 4-16 / 1-3 d`, ineligible upstream path | `5-20 / 8-24 / 2-7 d`, blocked | 1 | blocked (audit only; not a smoke) / blocked / blocked |
| ARC_NCA | reduced legacy run measured `<1 / 0.42 / <0.01 h`; CPU-only strict firewall smoke completed in 12.85 s with child-inclusive resources explicitly unmeasured | `2-10 / 8-24 / 1-3 d`, likely after wrapper expansion | `5-20 / 8-24 / 3-14 d`, under-specified | 1 | strict dev smoke passed / not-started / not-started |
| GridCoder2024 | architecture-only forward passed at `~1.88 GiB / 4.11 s`; static blocker audit passed without method execution; solver remains blocked | `2-10 / 8-24 / 1-24 h`, preselected 49-task subset only | `10-40 / 24-32 / 1-7 d`, training details incomplete | 1 | passed (architecture only) / not-started / not-started |
| 2D nGPT | architecture-only large forward passed at `~0.20 GiB / 2.13 s`; hardened static blocker audit passed without solver execution or forbidden artifact reads | `5-20 / 24-32 / 6-24 h`, unavailable until artifact/firewall/subset gates pass | `20-80 / >=32 / 1-7 d`, disk/VRAM risk and under-specified | 1 | passed (architecture only; audit not counted) / blocked / blocked |
| LPN | architecture/tests passed on CPU at `~3.89 GiB max RSS / 26.7 s`; hardened static gate passed without solver execution or restricted artifact/data reads | `10-30 / 24-32 / 1-4 d`, unknown until one exact W&B artifact/config is audited | `>41 / multi-device or TPU / unknown`, not compute-matched | 1 | passed (architecture/tests only; gate audit not counted) / blocked / blocked |
| ARChitects 2024 | checkpoint integrity passed; forward preflight blocked by `<10 GiB` free VRAM; static eight-blocker gate is not a smoke | `18-30 / 20-32 / 6-12 h`, tight and ARC-1 contaminated | `30-60 / 24-80 / 1-4 d`, storage and exact-stack risk | 2 | blocked (forward; gate audit not counted) / not-started / not-started |
| BARC | measured seed/program-only smoke, `0` model weights / `0` GPU / `0.287 s`; static eight-blocker gate is auxiliary only | `18-35 / 18-30 / 6-24 h`, selected 8B path deferred | `>100 / multi-GPU / days`, blocked on host | 2 | passed (seed/program only; gate audit not counted) / not-started / not-started |
| arc-lang-public | measured zero-dollar import/config/Pydantic component, no API; static eight-blocker gate is auxiliary only | `1-5 / 0 / 6-48 h + API`, blocked pending raw-key firewall/egress closure/approval/fuse | Exact proprietary API snapshot unavailable | 2 | passed (component only; gate not counted) / blocked / not-started |
| epang080516/arc_agi | measured synthetic trusted-code component, no pickle/generated code/API; static nine-blocker gate is auxiliary only | `1-5 / 0 / 2-24 h + API`, blocked by overlap/firewall/pickle/sandbox/cost gates | API and library-order parity not stable | 2 | passed (component only; gate not counted) / blocked / not-started |
| ArcMemo | measured two-problem no-memory dummy-completion dry-run, no API; static nine-blocker gate is auxiliary only | `1-5 / 0 / 1-7 d + API`, blocked pending memory artifact, fair adapter, sandbox, and budget | Proprietary model snapshot and full rollout spend block strict parity | 2 | passed (dry-run only; memory mechanism not run; gate not counted) / blocked / not-started |
| TinyRecursiveModels | measured CPU-only 6.83M-param random-weight synthetic architecture forward; static ten-blocker gate is auxiliary only | blocked pending dataset/checkpoint/firewall/runtime/selection/capacity closure; no admissible reduced solver config | `50-150 / >24 / ~3 d on 4 H100`, not host-parity | 3 | passed (architecture only; gate not counted) / blocked / not-started |
| SOAR | measured locked-source, bundled label-free challenge, and fixed-helper smoke; formal 13-blocker static gate passed without solutions/model/solver | `25-41 / 24-32 / 1-3 d`, storage-tight reduced search | `>500 / multi-GPU / weeks`, blocked on host | 3 | passed (source-data/helper only; gate not counted) / blocked / blocked |
| NVARC | measured locked wiring/config/gitlink and fixed-helper smoke; formal 12-blocker static gate passed without submodules/data/model/solver | `20-41 / 24-32 / 1-4 d`, one component at a time | `>100 / >32 / weeks`, ensemble/corpus blocked | 3 | passed (wiring/helper only; gate not counted) / blocked / blocked |
| MARC | measured synthetic ARC task/submission/voting component; no benchmark labels/model/TTT/inference | `25-41 / 24-32 / 1-7 d`, selected tasks only | `>41 / >32 / multi-environment days`, blocked/complex | 3 | passed (ARC-native component only) / not-started / not-started |
| LatentMAS | measured role/prompt schema for AI2 ARC-Challenge; no data/model/latent inference and not ARC-AGI | `15-35 / 18-32 / 6-48 h`, native benchmark with reduced 4B/8B | `35-80 / two GPUs / 1-5 d`, official hybrid path blocked | 3 | passed (AI2 ARC prompt schema only) / not-started / not-started |
| AgentPrimitives | measured static config/Organizer schema only; no import, primitive, model, or runnable pipeline | Unknown; blocked by missing end-to-end runner | Unknown; experiment pipeline not released | 3 | passed (static schema only) / blocked / blocked |
| GraphPlanner | measured bundled router-data schema and pure prompt component; AI2 ARC, no model/API/PPO | `2-10 / 4-16 / 1-4 d + API`, cost/history limited | Unknown API spend and interaction corpus; not compute-matched | 3 | passed (AI2 ARC schema/prompt only) / not-started / blocked |
| RouteMoA | labeled precomputed-result scorer audit only; auxiliary evidence with no inference or router | `2-10 / 0-12 / 1-4 d + API`, or local models exceed disk | `>100 / five 80 GiB GPUs / days`, blocked on host | 3 | not-started (scorer-only auxiliary; not counted) / not-started / blocked |
| MACA | measured deterministic random-weight CPU GraphSpec component only; no GRPO/VERL/model/task | Unknown; no documented benchmark command | Unknown; GRPO/VERL budget and checkpoints absent | 3 | passed (random-weight component only) / blocked / blocked |
| Omni-ARC | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |
| Mini-ARC transformer | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |
| NeuroMAS | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |
| ReM-MoA | Blocked: no verified implementation | Blocked | Blocked | 4 | blocked / blocked / blocked |

## Current Five-Batch Execution Order

The detailed gates and per-method membership are canonical in
[`EXECUTION_BATCHES.md`](EXECUTION_BATCHES.md). The five batches below are
mutually exclusive and cover all 24 methods.

### Phase 0: Foundation And Source Locks

- Reconfirm free space and inspect `/model` before retrieving any artifact.
- Pin every selected repository to a commit and record repository size before
  checkout. A branch name alone is not a reproducible source lock.
- Validate the local ARC snapshots and freeze task-ID lists, Top-K semantics,
  seeds, and output normalization before comparing solvers.
- The current training-only draft deterministically clusters 1,400 source
  records into 1,008 groups and assigns 706/151/151 to build/select/audit
  ([evidence](../reports/e0-development-split/20260806-training-only-deterministic-split-retry1/run.json)).
  It is not protocol v1. A derivative excludes all 376 flagged overlap clusters
  / 377 records without reallocation, leaving 632 clusters / 1,023 records;
  this controls only the locked overlap ledger and remains `draft-not-frozen`
  ([evidence](../reports/e0-development-split/20260806-arc1-clean-overlap-excluded-draft-view/run.json)).
- Create one isolated environment per candidate or explicitly documented group;
  do not upgrade the evaluator environment to satisfy a baseline.

### Batch A: Deepen Existing Low-Cost Smokes (5)

Order: CompressARC, ARC_NCA, GridCoder2024, 2D nGPT, then LPN.
These maximize information per downloaded GiB and establish PyTorch, notebook,
JAX, checkpoint, and program-search paths. Advance a candidate only after its
single-task output can be normalized and scored independently.

### Batch B: Published Local Weights, Capacity-Gated (2)

Order: ARChitects 2024, then BARC. Prefer published quantized checkpoints and
stop at disk/VRAM preflight failures.

### Batch C: API-Backed, Zero-Dollar First (3)

Order: ArcMemo, arc-lang-public, then epang080516/arc_agi. Set a hard
request/token/currency cap before the first request and store provider, exact
model identifier, request parameters, and returned request IDs. Their hardened
static gates are blocker inventories only: all three methods remain without a
solver prediction, strict promotion, benchmark, or performance eligibility.

### Batch D: Heavy Or Integration-Risk Methods (9)

TinyRecursiveModels, SOAR, NVARC, MARC, LatentMAS, AgentPrimitives,
GraphPlanner, RouteMoA, and MACA enter only after a written storage/compute
budget review. TinyRecursiveModels plus seven of these methods now have only the
scope-limited passes named above; TRM, SOAR, and NVARC additionally have formal
static gates that preserve ten, 13, and 12 blockers respectively, while RouteMoA
has scorer-only auxiliary evidence, not a smoke. None has a benchmark or full
reproduction. Reduced runs are useful
engineering results but must be reported as reduced. Full reproduction for
several methods is impossible on the target host without additional GPUs,
storage, paid APIs, or missing upstream assets.

### Batch E: Blocked Watchlist (5)

ARC-VSA-2025, Omni-ARC, the Mini-ARC transformer implementation, NeuroMAS, and
ReM-MoA stay blocked. Re-audit periodically or when authors announce code. Do
not substitute an unverified third-party repository and retain the original
paper/write-up URL.

## Interpretation Rules

- ARC-AGI-1 and ARC-AGI-2 scores are separate results. Public, semi-private,
  private, training, and evaluation splits are never interchangeable.
- External ARC Prize classifications, awards, and private-evaluation scores are
  provenance facts, not this repository's results, unless a separate local
  benchmark or reproduction report explicitly establishes them. The SOAR and
  NVARC static gates establish no such result.
- ARC-AGI-1 evaluation and ARC-AGI-2 training share 376 task IDs; 375 are
  semantically identical complete labeled tasks, and all 376 have semantically
  identical test I/O ([evidence](../reports/e0-overlap/20260806-arc1-eval-vs-arc2-train-retry1/run.json)). A checkpoint trained on ARC-AGI-2 training is not clean on ARC-AGI-1 evaluation unless overlapping tasks are excluded under a predeclared, auditable denominator; otherwise label the result contamination-aware or historical.
- Native non-ARC methods such as LatentMAS, AgentPrimitives, GraphPlanner,
  RouteMoA, MACA, NeuroMAS, and ReM-MoA must first be reproduced on their paper
  benchmarks. Any later ARC adaptation is a new experiment, not paper parity.
- `arc_challenge` in LatentMAS, AgentPrimitives, and GraphPlanner refers to the
  AllenAI AI2 ARC multiple-choice science benchmark, not ARC-AGI grid tasks.
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
