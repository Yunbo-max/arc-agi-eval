# ARC-REBench Analysis Plan V1

- Scientific plan status: frozen v1 on 2026-08-06
- Target venue cycle: NeurIPS 2027 Evaluations & Datasets
- 2027 author CFP status at freeze: not published
- New protocol-locked public method scores inspected before freeze: zero
- Machine-readable authority: `configs/analysis_plan_v1.json`

This file freezes the decisions that could otherwise be tuned after seeing new
locked-public solver outputs. The older `NEURIPS_EXPERIMENT_DESIGN.md` remains
a broader design document; where it says “provisional,” this v1 plan and its
machine-readable config control.

## Venue boundary

NeurIPS 2026 E&D is closed to new submissions: its official abstract and full
paper deadlines were May 4 and May 6, 2026 (AoE). The target is therefore the
NeurIPS 2027 Evaluations & Datasets Track. NeurIPS has officially referenced
that cycle, but as of the freeze date has not published the 2027 author call,
portal, or dates. All those fields remain null; no 2027 date is inferred from
2026.

When official 2027 material appears, administrative and compliance fields may
be appended. Estimands, hypotheses, task sets, method eligibility, budgets,
stopping rules, missingness, retry policy, metric definitions, and analysis
choices cannot be changed in response to observed evaluation outcomes. If the
track is unavailable or materially excludes the work, the venue gate becomes
blocked and requires a prospective protocol amendment.

Official planning sources:

- <https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets>
- <https://neurips.cc/Conferences/2026/CallForCompetitions>
- <https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ>
- <https://neurips.cc/Conferences/2026/MainTrackHandbook>
- <https://neurips.cc/public/guides/PaperChecklist>

ARC Prize 2026 is an optional external milestone, not the archival venue. It
requires a linked Kaggle code submission; the current plan authorizes no paid
API use and no competition submission on the user's behalf.

## Estimands and inference

ARC-AGI-1 and ARC-AGI-2 are always reported separately. The primary estimand is
output-level exact pass@2 with every declared output in the denominator. Strict
whole-task exactness is secondary and micro cell accuracy is diagnostic only.
The base ARC task is the cluster and resampling unit; outputs, IsoARC variants,
budgets, and seeds never inflate the independent task count.

H1 tests the configuration-by-log-budget interaction. H2 tests transformation
and configuration-by-transformation terms. The primary implementation is a
logistic mean model with independence working correlation and base-task robust
sandwich covariance. Separation, rank deficiency, non-convergence, excessive
condition number, or singular/nonfinite covariance triggers the frozen paired
base-task randomization fallback. The analysis never switches tests because a
p-value is more favorable.

H3 is declared infeasible for protocol v1 at freeze because zero methods have
method-specific strict runtime promotion and no same-family pair is eligible.
It cannot be repaired after outcomes are observed by changing family labels or
compute allocation.

The H1–H3 omnibus p-values form one two-sided Holm family at alpha 0.05.
Prespecified follow-ups use a second within-hypothesis Holm correction only
after their omnibus survives. Fixed-budget pairwise comparisons are a separate
secondary Holm family. All planned contrasts are reported.

## Power, uncertainty, and missingness

Confidence intervals use 10,000 deterministic percentile bootstrap resamples
of whole base tasks. A synthetic, outcome-free power simulation evaluates the
exact paired fallback at 64, 120, and 400 tasks under every assumption recorded
in the config. It is a design diagnostic, not an empirical ARC result and not a
claim about GEE interaction power.

Missing, malformed, timed-out, OOM, and retry-exhausted outputs are incorrect in
the frozen denominator. Human correction is prohibited. One infrastructure
rerun is allowed only by a blinded pre-outcome rule, and the failed run remains
preserved.

## Campaign ceiling

The hard design ceiling is 1,500 local GPU-hours including contingency, at most
six headline and four diagnostic configurations, a 300-second main budget, and
30/120/300/900-second curve checkpoints. API spend is fixed at USD 0 and API
execution is unauthorized without a new explicit user authorization. Public
execution remains unauthorized until every required protocol gate passes.
This ceiling is not a GPU reservation and does not imply that any method is
currently eligible.
