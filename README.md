# ARC-AGI evaluation foundation

A small, dependency-free Python toolkit for validating, enumerating, and
scoring the public ARC-AGI-1 and ARC-AGI-2 benchmarks. Canonical public source
snapshots are vendored under `third_party/`; no model checkpoints are included.

## Project status

Last evidence update: **2026-08-06**. This section is an evidence ledger, not a
claim that all tracked papers have been reproduced.

| Area | Verified progress | Evidence |
| --- | --- | --- |
| Evaluator | All 1,920 vendored ARC-AGI-1/2 tasks validate and have per-file hashes; challenge/reference scoring, IsoARC, process lifecycle, current-process resources, and malformed/timeout/label-mutation cases have terminal E0 evidence | [`data audit`](reports/e0-benchmark-data/20260806-public-snapshot-integrity-v1/run.json), [`contract audit`](reports/e0-contracts/20260806-firewall-isoarc-process-terminal/run.json) |
| Method census | 24 methods tracked: 19 public candidates, 1 partial/complex candidate, and 4 without a verified runnable implementation | [`configs/baselines.json`](configs/baselines.json), [`docs/REPRODUCTION_MATRIX.md`](docs/REPRODUCTION_MATRIX.md) |
| Method execution | 17/24 methods have a scope-limited compatibility, architecture, component, or dry-run pass, but only CompressARC and ARC_NCA retain legacy solver-prediction smokes. Both have method-specific strict runtime promotions (2/24); performance eligibility, declared public benchmarks, and paper reproductions remain 0/24. Eleven methods now also have hardened static blocker audits, all explicitly excluded from smoke and promotion counts | [`eligibility audit`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry12/run.json), [`GridCoder gate`](reports/gridcoder2024/20260806-source-dependency-label-artifact-gate-v3/run.json), [`2D nGPT gate`](reports/2d-ngpt/20260806-source-artifact-label-runtime-gate-v1/run.json), [`LPN gate`](reports/lpn/20260806-source-artifact-data-label-gate-v1/run.json), [`ARChitects gate`](reports/architects-2024/20260806-source-artifact-label-runtime-gate-v1/run.json), [`BARC gate`](reports/barc/20260806-source-artifact-label-resource-gate-v1/run.json), [`TRM gate`](reports/tiny-recursive-models/20260806-source-artifact-dataset-label-resource-gate-v1/run.json), [`SOAR gate`](reports/soar/20260806-source-artifact-dataset-label-api-code-resource-gate-v1/run.json), [`NVARC gate`](reports/nvarc/20260806-source-gitlink-artifact-dataset-label-code-resource-gate-v1/run.json), [`ArcMemo gate`](reports/arcmemo/20260806-source-label-memory-api-sandbox-gate-v1/run.json), [`arc-lang gate`](reports/arc-lang-public/20260806-source-label-api-egress-gate-v1/run.json), [`epang gate`](reports/epang-arc-agi/20260806-source-label-pickle-sandbox-api-gate-v1/run.json), [`CompressARC strict smoke`](reports/compressarc/20260806-cpu-dev-3c9b0459-strict-v1/run.json), [`ARC_NCA strict smoke`](reports/arc-nca/20260806-cpu-dev-6150a2bd-strict-v1/run.json) |
| Deterministic floor | Complete Top-2 predictions and scores exist for all 400 ARC-AGI-1 and 120 ARC-AGI-2 public evaluation tasks | [`results/`](results/) |
| Research protocol | The latest machine-readable protocol-v1 draft root binds scorer, data, overlap, frozen development/runtime/analysis/input evidence, strict-run schema, exposure, challenge views, resources, and eligibility. It remains deliberately unfrozen: 14 gates pass, 1 required gate is pending, and 2 optional gates are blocked. The sole required blocker is child-inclusive process-tree resource accounting; the 49-input/125-code-file bundle records two strict promotions, anchors the SOAR/NVARC static gates, and admits zero public configurations | [`protocol root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry16/run.json), [`input bundle`](reports/e0-freeze/20260806-input-bundle-v1-retry16/run.json), [`prior exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry16/run.json) |
| Durable artifacts | The published ARChitects 4-bit checkpoint is revision- and hash-audited locally; no run-produced checkpoint exists | [`run.json`](reports/architects-2024/20260806-4bit-checkpoint-integrity/run.json), [`scripts/hub_sync.py`](scripts/hub_sync.py) |

Evidence levels are deliberately separate:

`source-audited -> component smoke -> solver-prediction smoke -> single-task experiment -> fixed-subset benchmark -> full public benchmark -> paper reproduction`

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
| CompressARC | ARC-AGI-1 training task `00d62c1b`, 2 optimization steps, compatibility-deviant environment | 0 / 1 | 0 / 1 | 36 / 400 (9.0000%) | 9.283 s | [`run.json`](reports/compressarc/20260806-training-2step-00d62c1b-retry1/run.json) |
| CompressARC | Frozen development task `3c9b0459`, compatibility-deviant CPU two-step code-only A/B firewall smoke; current-process resources only | 0 / 1 | 0 / 1 | 2 / 9 (22.2222%) | 12.705 s | [`run.json`](reports/compressarc/20260806-cpu-dev-3c9b0459-strict-v1/run.json) |
| ARC_NCA | ARC-AGI-1 training task `00d62c1b`, reduced 10-step/32-rollout notebook adaptation | 0 / 1 | 0 / 1 | 342 / 400 (85.5000%) | 2.365 s | [`run.json`](reports/arc-nca/20260806-training-00d62c1b-10step/run.json) |
| ARC_NCA | Frozen development task `6150a2bd`, CPU-only two-step method-specific A/B firewall smoke; current-process resources only | 0 / 1 | 0 / 1 | 3 / 9 (33.3333%) | 12.849 s | [`run.json`](reports/arc-nca/20260806-cpu-dev-6150a2bd-strict-v1/run.json) |

The CompressARC rows are single-task evidence, not a benchmark or paper
reproduction. Test outputs were unavailable to the optimizer. The strict row
also delayed run-local scoring-payload materialization until both inference
processes exited, but remains a trusted-code rather than OS-isolated boundary. The
passing forward smoke is recorded separately in
[`20260804-smoke-forward-002`](reports/compressarc/20260804-smoke-forward-002/run.json).
The ARC_NCA strict row is one-task mechanism/firewall evidence with disclosed
analyst label exposure. It is not a benchmark, performance-table result, or
public-run authorization, and child inference/scoring resources are excluded.

## Project history

| Date | Milestone | Durable evidence |
| --- | --- | --- |
| 2026-08-04 | Created the ARC-AGI-1/2 validator, enumerator, scorer, deterministic floor, vendored data snapshots, source audit, and run evidence contract | [`f7e7935`](https://github.com/Yunbo-max/arc-agi-eval/commit/f7e7935382213b1712fb41a422c65ab8811d0ec4) |
| 2026-08-04 | Preserved the first CompressARC smoke failure: the local harness incorrectly assumed `ARCCompressor.eval()` existed | [`smoke-forward-001`](reports/compressarc/20260804-smoke-forward-001/run.json) |
| 2026-08-04 | Removed that invalid harness assumption and completed a forward-pass compatibility smoke | [`smoke-forward-002`](reports/compressarc/20260804-smoke-forward-002/run.json) |
| 2026-08-04 | Completed 2-step and 1,500-step CompressARC single-task runs; neither solved the task exactly | [`reports/compressarc/`](reports/compressarc/) |
| 2026-08-04 | Added isolated preparation contracts, source locks, asset manifests, and GitHub/Hugging Face persistence instructions for all 24 tracked methods | [`8a2632c`](https://github.com/Yunbo-max/arc-agi-eval/commit/8a2632cce8a39c9d665285574d00f155f87b49f8) |
| 2026-08-04 | Added the ARC-REBench NeurIPS-level protocol: compute matching, label isolation, IsoARC, clustered statistics, stop rules, and resource gates | [`7ba3087`](https://github.com/Yunbo-max/arc-agi-eval/commit/7ba30878bf5210e10ef057ab585c5c01c4d712c7) |
| 2026-08-06 | Added a hash-manifested challenge-only generator and an independent exact scorer cross-checked on 500 generated cases | [`tests/`](tests/) |
| 2026-08-06 | Migrated the additive score contract and CLI to output-level exact pass@K as primary, retained strict task exact as secondary and cell accuracy as diagnostic, and independently rechecked both deterministic-floor prediction files without rewriting historical results | [`run.json`](reports/e0-scoring/20260806-output-primary-contract/run.json) |
| 2026-08-06 | Preserved a missing-`matplotlib` CompressARC failure, then passed a second-task 2-step smoke in an isolated environment; this is not a benchmark | [`reports/compressarc/`](reports/compressarc/) |
| 2026-08-06 | Preserved an OpenCV import failure for ARC_NCA, then passed a label-isolated reduced smoke through a scripted notebook adaptation; this is not a benchmark | [`reports/arc-nca/`](reports/arc-nca/) |
| 2026-08-06 | Pinned ARC_gym and passed a synthetic-weight GridCoder LVM architecture forward; the missing Kaggle checkpoint still blocks a solver smoke | [`run.json`](reports/gridcoder2024/20260806-architecture-forward-smoke/run.json) |
| 2026-08-06 | Passed the official large 2D nGPT architecture forward with synthetic weights; missing `exp_54.pt`, generated re-ARC data, and `fixed_size.pkl` still block solver execution | [`run.json`](reports/2d-ngpt/20260806-large-architecture-forward-smoke/run.json) |
| 2026-08-06 | Pinned LPN, passed its 400-generator direct test and encoder/decoder self-test, and preserved its test-discovery-only missing-`random` failure; no checkpoint or dataset was used | [`run.json`](reports/lpn/20260806-official-tests-architecture-smoke/run.json) |
| 2026-08-06 | Completed a hardened 2D nGPT static source/artifact/label/runtime gate: no checkpoint, re-ARC, fixed-size, solution, ignored-bytecode, GPU, or network bytes were used. The audit freezes seven blockers and explicitly does not count as a solver smoke or promotion | [`gate audit`](reports/2d-ngpt/20260806-source-artifact-label-runtime-gate-v1/run.json), [`config`](configs/ngpt2d_gate_v1.json) |
| 2026-08-06 | Locked LPN at revision `0adfe56b`, parsed all 21 tracked Python files, and identified the root license without importing upstream code or opening any ARC JSON | [`source audit`](reports/lpn/20260806-locked-source-syntax-audit/run.json) |
| 2026-08-06 | Completed LPN's hardened source/artifact/data/label gate: the audit bound the exact Git tree and 21 Python files, read no bundled ARC JSON/YAML/notebook/bytecode/checkpoint bytes, executed no solver, and froze seven blockers without adding a smoke or promotion | [`gate audit`](reports/lpn/20260806-source-artifact-data-label-gate-v1/run.json), [`config`](configs/lpn_gate_v1.json) |
| 2026-08-06 | Calibrated the current-process resource monitor and recorded that the host cannot provide a usable mount/network namespace or sandbox runtime; the isolation gate remains false | [`calibration`](reports/e0-resources/20260806-process-monitor-calibration/run.json), [`isolation probe`](reports/e0-isolation/20260806-host-namespace-probe/run.json) |
| 2026-08-06 | Passed BARC's bundled seed-generator/handwritten-program smoke; no model or solver was exercised | [`run.json`](reports/barc/20260806-seed-00d62c1b-smoke/run.json) |
| 2026-08-06 | Downloaded and hash-audited the exact ARChitects 4-bit checkpoint, then stopped before allocation because the 10 GiB free-VRAM preflight failed | [`integrity`](reports/architects-2024/20260806-4bit-checkpoint-integrity/run.json), [`preflight`](reports/architects-2024/20260806-forward-preflight-gpu-occupied/run.json) |
| 2026-08-06 | Completed ARChitects' hardened static source/artifact/label/runtime gate: the audit froze eight blockers, including ARC-AGI-1 training contamination and in-process solution flows, without loading the checkpoint or producing a prediction | [`gate audit`](reports/architects-2024/20260806-source-artifact-label-runtime-gate-v1/run.json), [`config`](configs/architects_2024_gate_v1.json) |
| 2026-08-06 | Completed BARC's hardened metadata-first source/artifact/label/resource gate: 13 retained files were byte-bound, 1,464 tracked leaves remained metadata-only, and eight blockers were frozen without reading ARC/answer or weight leaves, loading a model, or adding a smoke | [`gate audit`](reports/barc/20260806-source-artifact-label-resource-gate-v1/run.json), [`config`](configs/barc_gate_v1.json) |
| 2026-08-06 | Refreshed eligibility, the reproduction funnel, the 40-input/117-code-file zero-admit bundle, prior exposure, and the protocol root after the two Batch-B static gates. Counts remain 17 scope-limited smokes, 2 strict promotions, 0 performance-eligible/admitted configurations, and one required protocol blocker | [`eligibility`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry9/run.json), [`funnel`](reports/e0-reproduction-funnel/20260806-manifest-funnel-audit-retry6/run.json), [`bundle`](reports/e0-freeze/20260806-input-bundle-v1-retry12/run.json), [`exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry12/run.json), [`root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry12/run.json) |
| 2026-08-06 | Passed zero-dollar, network-guarded component/dry-run checks for ArcMemo, arc-lang-public, and epang080516/arc_agi. Their exact scopes are no-memory dummy-completion driver, import/config/Pydantic parser, and synthetic data-model/auditor-written trusted executor; none made an API request or produced a solver prediction | [`ArcMemo`](reports/arcmemo/20260806-native-dry-run-retry2/run.json), [`arc-lang-public`](reports/arc-lang-public/20260806-zero-dollar-import-smoke-retry1/run.json), [`epang`](reports/epang-arc-agi/20260806-zero-dollar-component-smoke/run.json) |
| 2026-08-06 | Completed hardened metadata-first Batch C static gates: nine ArcMemo, eight arc-lang, and nine epang blockers were frozen without upstream execution, restricted worktree data/pickle reads, API/GPU/network use, or predictions. All three methods remain blocked and the gates are auxiliary only | [`ArcMemo gate`](reports/arcmemo/20260806-source-label-memory-api-sandbox-gate-v1/run.json), [`arc-lang gate`](reports/arc-lang-public/20260806-source-label-api-egress-gate-v1/run.json), [`epang gate`](reports/epang-arc-agi/20260806-source-label-pickle-sandbox-api-gate-v1/run.json), [`config`](configs/batch_c_static_gate_v1.json) |
| 2026-08-06 | Refreshed eligibility, the funnel, the 41-input/119-code-file zero-admit bundle, prior exposure, and protocol root after Batch C. Counts remain 17 smokes, 2 strict promotions, and 0 eligible/admitted; auxiliary evidence is now 7, and the sole required protocol blocker is unchanged | [`eligibility`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry10/run.json), [`funnel`](reports/e0-reproduction-funnel/20260806-manifest-funnel-audit-retry7/run.json), [`bundle`](reports/e0-freeze/20260806-input-bundle-v1-retry14/run.json), [`exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry14/run.json), [`root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry14/run.json) |
| 2026-08-06 | Passed a CPU-only TinyRecursiveModels synthetic 1x16 architecture forward with 6,829,058 trainable parameters; no ARC data, checkpoint, training, or GPU was used | [`run.json`](reports/tiny-recursive-models/20260806-cpu-architecture-forward-smoke-retry2/run.json) |
| 2026-08-06 | Completed TinyRecursiveModels' reproducible metadata-first static gate: 28 text leaves were byte-bound, 12 ARC/image leaves stayed metadata-only, and ten blockers were frozen without upstream execution, checkpoint loading, restricted leaf reads, GPU use, or a prediction. The audit passed but the method remains blocked; 45%/8% are unverified README self-reports | [`gate audit`](reports/tiny-recursive-models/20260806-source-artifact-dataset-label-resource-gate-v1/run.json), [`config`](configs/trm_gate_v1.json), [`runner manifest`](configs/trm_gate_runner_manifest_v1.json) |
| 2026-08-06 | Refreshed eligibility, the funnel, the 43-input/122-code-file zero-admit bundle, prior exposure, and protocol root after the TRM gate. Counts remain 17 smokes, 2 strict promotions, and 0 eligible/admitted; auxiliary evidence is now 8. The corrected bundle manifest directly covers all eight nested declared `run.json` leaves; the sole required protocol blocker is unchanged | [`eligibility`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry11/run.json), [`funnel`](reports/e0-reproduction-funnel/20260806-manifest-funnel-audit-retry8/run.json), [`bundle`](reports/e0-freeze/20260806-input-bundle-v1-retry15/run.json), [`exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry15/run.json), [`root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry15/run.json) |
| 2026-08-06 | Completed formal metadata-first SOAR and NVARC gates. The audits passed while the methods remained blocked on 13 and 12 gates, respectively; neither produced a prediction or reproduced an official result. External classification places SOAR in the 2025 Paper Award and NVARC first on the 2025 ARC-AGI-2 private evaluation, not in 2026 | [`SOAR`](reports/soar/20260806-source-artifact-dataset-label-api-code-resource-gate-v1/run.json), [`NVARC`](reports/nvarc/20260806-source-gitlink-artifact-dataset-label-code-resource-gate-v1/run.json), [`classification`](docs/EXECUTION_BATCHES.md) |
| 2026-08-06 | Refreshed eligibility, the funnel, the 49-input/125-code-file zero-admit bundle, prior exposure, and protocol root after the SOAR/NVARC gates. Counts remain 17 smokes, 2 strict promotions, and 0 eligible/admitted; auxiliary evidence is now 10. The bundle explicitly anchors both gate configs, runner manifests, and formal reports; the sole required protocol blocker is unchanged | [`eligibility`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry12/run.json), [`funnel`](reports/e0-reproduction-funnel/20260806-manifest-funnel-audit-retry9/run.json), [`bundle`](reports/e0-freeze/20260806-input-bundle-v1-retry16/run.json), [`exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry16/run.json), [`root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry16/run.json) |
| 2026-08-06 | Audited ARC-AGI-1 evaluation against ARC-AGI-2 training: 376 shared IDs, 375 semantically identical labeled tasks, and 376 semantically identical test input/output sets | [`run.json`](reports/e0-overlap/20260806-arc1-eval-vs-arc2-train-retry1/run.json) |
| 2026-08-06 | Passed zero-cost, source/data/config component smokes for SOAR and NVARC; neither loaded a model or ensemble checkpoint, and both generated-code paths remain blocked by the failed isolation gate | [`SOAR`](reports/soar/20260806-zero-dollar-source-data-smoke/run.json), [`NVARC`](reports/nvarc/20260806-zero-dollar-component-source-smoke/run.json) |
| 2026-08-06 | Passed scope-limited component/schema smokes for MARC, LatentMAS, AgentPrimitives, GraphPlanner, and MACA. LatentMAS, AgentPrimitives, and GraphPlanner use AI2 ARC or other native tasks rather than ARC-AGI; MACA used random weights | [`MARC`](reports/marc/20260806-zero-cost-arc-components-smoke-retry1/run.json), [`LatentMAS`](reports/latentmas/20260806-native-prompt-schema-smoke-retry1/run.json), [`AgentPrimitives`](reports/agent-primitives/20260806-static-config-schema-smoke/run.json), [`GraphPlanner`](reports/graphplanner/20260806-zero-dollar-schema-smoke/run.json), [`MACA`](reports/maca/20260806-zero-dollar-graphspec-component-smoke/run.json) |
| 2026-08-06 | Audited RouteMoA's bundled labeled predictions as scorer-only auxiliary evidence, preserving its repository-wide syntax failure; this is not a solver smoke or benchmark | [`scorer audit`](reports/routemoa/20260806-zero-dollar-precomputed-scorer-audit/run.json), [`source failure`](reports/routemoa/20260806-locked-source-syntax-audit/run.json) |
| 2026-08-06 | Confirmed ARC-VSA remains ineligible to run: `sspspace` is unavailable and the released solver reads test outputs during construction | [`run.json`](reports/arc-vsa-2025/20260806-dependency-label-gate-audit-retry1/run.json) |
| 2026-08-06 | Built a deterministic training-only development draft: 1,400 source records form 1,008 verified clusters split 706/151/151 across dev-build/select/audit. A locked overlap ledger flags 376 clusters, so the general draft is contamination-aware rather than ARC-1-clean | [`run.json`](reports/e0-development-split/20260806-training-only-deterministic-split-retry1/run.json), [`manifest.json`](reports/e0-development-split/20260806-training-only-deterministic-split-retry1/manifest.json) |
| 2026-08-06 | Materialized the known-overlap-excluded derivative without reallocation: 376 flagged clusters / 377 records were removed, leaving 632 clusters / 1,023 records; “clean” is limited to the locked overlap ledger | [`run.json`](reports/e0-development-split/20260806-arc1-clean-overlap-excluded-draft-view/run.json) |
| 2026-08-06 | Added a strict protocol-v1 terminal run schema with provenance-file and hardware binding, and classified all 24 methods by execution scope/trust; only 2 have solver-prediction smokes and 0 are performance-eligible | [`schema audit`](reports/e0-schema/20260806-protocol-v1-run-schema-self-audit/run.json), [`eligibility audit`](reports/e0-method-eligibility/20260806-eligibility-trust-audit/run.json) |
| 2026-08-06 | Locked a non-circular prior-exposure cutoff, materialized 520 label-free public challenge inputs, and generated the initial protocol-v1 draft root. At that stage the root was internally consistent but had 6 mandatory P0 gates and was not frozen | [`exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry1/run.json), [`challenge views`](reports/e0-challenge-data/20260806-locked-public-challenge-trees-draft/run.json), [`protocol root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry1/run.json) |
| 2026-08-06 | Froze the 94-task development runtime, trusted challenge-runtime core, fixed64 IsoARC and analysis plans, then emitted a zero-admit input bundle and refreshed the exposure/root attestations. At that stage the root had one required process-tree resource gate left and authorized no public solver run | [`development`](reports/e0-development-split/20260806-frozen-known-overlap-excluded-dev-audit-v1/run.json), [`input bundle`](reports/e0-freeze/20260806-input-bundle-v1-retry3/run.json), [`exposure`](reports/e0-prior-exposure/20260806-workspace-disclosure-draft-retry3/run.json), [`protocol root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry3/run.json) |
| 2026-08-06 | Passed ARC_NCA's CPU-only method-specific protocol-v1 A/B firewall smoke on frozen development task `6150a2bd`: both inference processes received zero test outputs and emitted byte-identical Top-2 predictions before independent scoring. Analyst label exposure is disclosed, GPU/network use was zero, and this is not a benchmark or performance-table result | [`strict run`](reports/arc-nca/20260806-cpu-dev-6150a2bd-strict-v1/run.json), [`eligibility`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry2/run.json), [`protocol root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry6/run.json) |
| 2026-08-06 | Passed CompressARC's compatibility-deviant CPU protocol-v1 A/B firewall smoke on frozen task `3c9b0459`: two inference-only processes received the same label-free challenge and code-only upstream stage, emitted byte-identical predictions, and exited before hidden scoring payloads were materialized. This promotes the mechanism contract, not performance eligibility or public execution | [`strict run`](reports/compressarc/20260806-cpu-dev-3c9b0459-strict-v1/run.json), [`eligibility`](reports/e0-method-eligibility/20260806-eligibility-trust-audit-retry4/run.json), [`protocol root`](reports/e0-protocol/20260806-protocol-v1-draft-root-retry8/run.json) |
| 2026-08-06 | Completed GridCoder2024's hardened static source/dependency/label/artifact audit with a tracked-file allowlist and fail-closed config: no upstream import, ARC/checkpoint byte read, GPU, network, or prediction. The audit freezes seven blockers and does not count as a method smoke or promotion | [`gate audit`](reports/gridcoder2024/20260806-source-dependency-label-artifact-gate-v3/run.json), [`config`](configs/gridcoder2024_gate_v3.json) |

## Current execution gates

Latest host observation on 2026-08-06:

- `/model` is absent. The container root is a 100 GiB overlay with 26,053,058,560
  bytes (about 24.3 GiB) free in the latest persisted snapshot; after the 8 GiB
  reserve, at most about 16.3 GiB is admissible before accounting for extraction
  peaks. Every large retrieval requires a fresh preflight
  ([evidence](reports/e0-resources/20260806-host-capacity-snapshot-0321/run.json)).
- The active target is one 24 GiB RTX 3090. An unrelated host process currently
  occupies about 19 GiB VRAM, leaving too little free memory for the ARChitects
  forward gate. Historical CompressARC evidence produced on a 32 GiB RTX 5090
  remains a different compute class and will not be pooled as repeated trials.
- The CLI and additive score schema now declare output-level exact pass@K as
  primary, strict whole-task exact as secondary, and micro cell accuracy as a
  diagnostic. Locked public benchmarks use K=2. The migration audit rechecked
  both deterministic-floor prediction files against an independent scorer and
  did not rewrite their historical result records.
- The challenge-only generator, independent exact scorer, IsoARC
  D4/color/permutation round trips, process lifecycle controls, current-process
  resource monitor/calibration, and E0 malformed/timeout/label-mutation cases
  are implemented. Current-process accounting does not include children. The
  host namespace/container probe ran, but its isolation gate is false; this
  blocks untrusted generated-code execution.
- ARC-AGI-1 evaluation overlaps ARC-AGI-2 training at 376 task IDs; 375 are
  semantically identical as complete labeled tasks and all 376 have semantically
  identical test I/O. A model trained on ARC-AGI-2 training cannot claim a clean
  ARC-AGI-1 evaluation result unless the overlapping tasks are excluded under a
  predeclared, auditable denominator; otherwise the result is contamination-aware
  or historical only.
- The deterministic training-only development split groups 400 ARC-AGI-1 and
  1,000 ARC-AGI-2 training records into 1,008 verified clusters, then assigns
  706/151/151 clusters to dev-build/dev-select/dev-audit using public seed
  `20260806`. A derivative removes all 376 flagged clusters / 377 source records
  without reallocating the remaining 632 clusters. A deterministic 94-task /
  97-output known-overlap-excluded dev-audit runtime is now frozen. The exclusion
  controls only the known overlap ledger and does not prove absence of renamed
  semantic overlap or pretrained-model exposure.
- The public execution view contains 400 ARC-AGI-1 and 120 ARC-AGI-2
  label-free challenge files (586 test inputs, zero test-output fields). Public
  labels remain locally accessible, and per-method mutation checks plus
  independent scoring are still mandatory.
- The latest protocol root reports 14 passed, 1 pending, and 2 optional blocked
  gates. The only unmet required gate is child-inclusive process-tree CPU/RSS/GPU
  accounting. The frozen input bundle records two strict runtime promotions,
  admits zero method configurations, and explicitly authorizes no
  locked-public solver run.

The 24 methods are now split into five mutually exclusive execution batches by
runnability, resource demand, and experiment value. See
[`docs/EXECUTION_BATCHES.md`](docs/EXECUTION_BATCHES.md) for the current order;
protocol v1 must still be frozen before another public evaluation.

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
python3 -m arc_agi_eval challenge third_party/arc-agi-2/data/evaluation /tmp/arc-agi-2-challenge
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

`challenge` creates a new label-free task tree for an inference process. It
removes every test output, preserves training demonstrations and test inputs,
rejects destinations inside the labeled source tree, and writes a hash ledger
to the suffix-free `MANIFEST` file. The output directory must be new or empty.

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
- Output-level exact pass@K is the primary metric. Locked public ARC-AGI-1/2
  comparisons use K=2 and the full declared output denominator.
- A task is exact only when every test output in that task is exact. Strict task
  accuracy is retained as a required secondary metric.
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

The design uses official output-level exact pass@2 as each benchmark's primary
estimand and retains the evaluator's stricter whole-task exact score as a
required secondary metric. That additive scorer/CLI migration is implemented
and audited. The 94-task known-overlap-excluded dev-audit runtime, global trusted
challenge-runtime core, fixed64 IsoARC design, analysis plan, and zero-admit input
bundle are frozen. The latest root has one required gate left—child-inclusive
process-tree resource accounting—so protocol v1 remains unfrozen and no new
locked-public solver result is authorized.

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
