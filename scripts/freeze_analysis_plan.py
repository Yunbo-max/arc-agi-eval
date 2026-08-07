#!/usr/bin/env python3
"""Freeze analysis-plan v1 and its deterministic statistical preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.challenge_runtime import (  # noqa: E402
    pretty_json_bytes,
    sha256_file,
    utc_now,
)
from arc_agi_eval.resources import ResourceMonitor  # noqa: E402
from arc_agi_eval.run_schema import validate_run_file  # noqa: E402
from arc_agi_eval.statistical_analysis import (  # noqa: E402
    AnalysisFallback,
    cluster_bootstrap_mean_interval,
    fit_cluster_robust_logistic,
    holm_adjust,
    minimum_detectable_effect_grid,
    paired_binary_exact_pvalue,
    paired_randomization_test,
    wald_block_test,
)


SCHEMA_PATH = ROOT / "schemas" / "protocol-v1-run.schema.json"
PROTOCOL_CONFIG = ROOT / "configs" / "protocol_v1_draft.json"


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, pretty_json_bytes(value))


def file_record(
    path: Path, *, role: str, required: bool = True
) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "required_for_claim": required,
    }


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown-cpu"


def load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("analysis config must be an object")
    if value.get("schema_version") != 1:
        raise ValueError("analysis config schema version must be 1")
    if value.get("plan_id") != "arc-rebench-analysis-plan-v1":
        raise ValueError("unexpected analysis plan id")
    if value.get("freeze_status") != "frozen-v1":
        raise ValueError("analysis plan is not frozen-v1")
    venue = value.get("venue_freeze")
    if not isinstance(venue, dict):
        raise ValueError("venue_freeze must be an object")
    if (venue.get("target_venue"), venue.get("target_track"), venue.get("target_cycle")) != (
        "NeurIPS",
        "Evaluations & Datasets",
        2027,
    ):
        raise ValueError("target venue/track/cycle mismatch")
    if any(
        venue.get(field) is not None
        for field in (
            "official_2027_author_cfp_url",
            "submission_portal_url",
            "abstract_deadline",
            "full_paper_deadline",
            "deadline_timezone",
        )
    ):
        raise ValueError("unpublished 2027 venue fields must remain null")
    campaign = value.get("campaign_cap")
    if not isinstance(campaign, dict) or campaign.get("api_spend_cap_usd") != 0:
        raise ValueError("API spend cap must remain zero")
    if campaign.get("public_execution_authorized_before_all_required_gates_pass"):
        raise ValueError("analysis plan cannot pre-authorize public execution")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list) or [item.get("id") for item in hypotheses] != [
        "h1-budget-interaction",
        "h2-isomorphism-interaction",
        "h3-family-portfolio",
    ]:
        raise ValueError("hypothesis roster mismatch")
    if hypotheses[2].get("status") != "declared_infeasible_for_protocol_v1_at_freeze":
        raise ValueError("H3 infeasibility decision is not frozen")
    return value


def statistical_self_test() -> dict[str, object]:
    checks: dict[str, object] = {}
    checks["holm_known_case"] = holm_adjust([0.04, 0.01, 0.03]) == [
        0.06,
        0.03,
        0.06,
    ]
    checks["paired_exact_known_case"] = abs(
        paired_binary_exact_pvalue([0] * 10, [1] * 10) - 2 / 1024
    ) < 1e-15
    differences = [1.0] * 25 + [-0.25] * 5
    randomization_a = paired_randomization_test(
        differences, seed=20260806, monte_carlo_resamples=4000
    )
    randomization_b = paired_randomization_test(
        differences, seed=20260806, monte_carlo_resamples=4000
    )
    checks["randomization_deterministic"] = randomization_a == randomization_b
    bootstrap_a = cluster_bootstrap_mean_interval(
        differences, seed=20260806, resamples=2000
    )
    bootstrap_b = cluster_bootstrap_mean_interval(
        differences, seed=20260806, resamples=2000
    )
    checks["cluster_bootstrap_deterministic"] = bootstrap_a == bootstrap_b

    rng = np.random.default_rng(20260806)
    design = []
    outcomes = []
    clusters = []
    for cluster in range(80):
        shift = rng.normal(0, 0.25)
        for condition in (0, 1):
            design.append([1.0, float(condition)])
            probability = 1 / (1 + np.exp(-(-0.7 + 0.8 * condition + shift)))
            outcomes.append(int(rng.random() < probability))
            clusters.append(cluster)
    fit = fit_cluster_robust_logistic(design, outcomes, clusters)
    wald = wald_block_test(fit, [1])
    checks["cluster_model_valid_case"] = (
        fit.converged
        and fit.cluster_count == 80
        and 0 <= wald["p_value"] <= 1
    )

    trigger_cases = [
        (
            "separation",
            [[1, 0], [1, 1], [1, 0], [1, 1]],
            [1, 1, 1, 1],
            [0, 1, 2, 3],
            {},
        ),
        (
            "singular-design",
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            [0, 1, 0, 1],
            [0, 1, 2, 3],
            {},
        ),
        (
            "non-convergence",
            [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1]],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 2, 3, 4, 5],
            {"max_iterations": 1, "tolerance": 1e-30},
        ),
    ]
    observed_triggers: dict[str, str | None] = {}
    for expected, case_design, case_outcomes, case_clusters, kwargs in trigger_cases:
        try:
            fit_cluster_robust_logistic(
                case_design, case_outcomes, case_clusters, **kwargs
            )
        except AnalysisFallback as error:
            observed_triggers[expected] = error.code
        else:
            observed_triggers[expected] = None
    checks["separation_trigger"] = observed_triggers["separation"] == "separation"
    checks["singular_design_trigger"] = (
        observed_triggers["singular-design"] == "singular-design"
    )
    checks["non_convergence_trigger"] = (
        observed_triggers["non-convergence"] == "non-convergence"
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "all_passed": all(checks.values()),
        "randomization_fixture": randomization_a,
        "bootstrap_fixture": bootstrap_a,
        "cluster_model_fixture": {
            "fit": fit.as_dict(),
            "wald": wald,
        },
        "observed_fallback_triggers": observed_triggers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "analysis_plan_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "reports" / "e0-analysis" / "20260806-analysis-plan-v1",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    try:
        output_directory.relative_to(ROOT)
    except ValueError as error:
        parser.error(f"output directory must remain inside repository: {error}")
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    monitor = ResourceMonitor(include_nvidia=False).start()
    status = "failed"
    failure: dict[str, str] | None = None
    record: dict[str, object] | None = None
    try:
        config = load_config(args.config)
        simulation_config = config["power_simulation"]
        simulation_kwargs = {
            "task_counts": simulation_config["task_counts"],
            "effects": simulation_config["effect_grid"],
            "baseline_probability": simulation_config["baseline_probability"],
            "discordance_probability": simulation_config[
                "discordance_probability"
            ],
            "simulations": simulation_config["simulations_per_cell"],
            "alpha": simulation_config["alpha"],
            "target_power": simulation_config["target_power"],
            "seed": simulation_config["seed"],
        }
        simulation_first = minimum_detectable_effect_grid(**simulation_kwargs)
        simulation_second = minimum_detectable_effect_grid(**simulation_kwargs)
        deterministic_simulation = simulation_first == simulation_second
        if not deterministic_simulation:
            raise ValueError("power simulation did not replay deterministically")
        power_path = output_directory / "power-simulation.json"
        atomic_json(
            power_path,
            {
                "schema_version": 1,
                "assumptions": simulation_config,
                "simulation": simulation_first,
                "deterministic_replay": deterministic_simulation,
                "uses_locked_public_outcomes": False,
                "interpretation": (
                    "Design sensitivity of the exact paired fallback under synthetic "
                    "assumptions; not an empirical ARC result or GEE interaction-power claim."
                ),
            },
        )
        self_test = statistical_self_test()
        if not self_test["all_passed"]:
            raise ValueError("statistical self-test failed")
        self_test_path = output_directory / "analysis-self-test.json"
        atomic_json(self_test_path, self_test)

        venue_path = output_directory / "venue-cycle.json"
        atomic_json(
            venue_path,
            {
                "schema_version": 1,
                "as_of": "2026-08-06",
                "venue_freeze": config["venue_freeze"],
                "expired_cycle_audit": config["expired_cycle_audit"],
                "optional_external_milestone": config[
                    "optional_external_milestone"
                ],
                "facts_verified_from_official_primary_urls": True,
                "network_used_by_freeze_execution": False,
            },
        )
        config_copy = output_directory / "config.json"
        atomic_bytes(config_copy, args.config.read_bytes())
        protocol_snapshot = output_directory / "protocol-manifest.json"
        atomic_json(
            protocol_snapshot,
            {
                "schema_version": 1,
                "protocol_id": "arc-rebench-protocol-v1-draft",
                "protocol_status": "draft-not-frozen",
                "gate_id": "lp.analysis-and-venue",
                "protocol_config_path": PROTOCOL_CONFIG.relative_to(ROOT).as_posix(),
                "protocol_config_sha256_at_execution": sha256_file(PROTOCOL_CONFIG),
            },
        )
        source_patch_path = output_directory / "source-patch-manifest.json"
        relevant_sources = [
            ROOT / "arc_agi_eval" / "statistical_analysis.py",
            ROOT / "scripts" / "freeze_analysis_plan.py",
            ROOT / "tests" / "test_statistical_analysis.py",
            ROOT / "tests" / "test_analysis_plan.py",
            ROOT / "docs" / "ANALYSIS_PLAN_V1.md",
            ROOT / "requirements" / "analysis.txt",
            args.config,
        ]
        atomic_json(
            source_patch_path,
            {
                "schema_version": 1,
                "scope": "frozen analysis plan and deterministic preflight",
                "files": [
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in relevant_sources
                ],
            },
        )
        source_lock_path = output_directory / "source-lock.json"
        dirty = git_dirty()
        atomic_json(
            source_lock_path,
            {
                "schema_version": 1,
                "revision": git_revision(),
                "dirty": dirty,
                "patch_manifest_path": source_patch_path.relative_to(ROOT).as_posix(),
                "patch_manifest_sha256": sha256_file(source_patch_path),
            },
        )
        artifact_manifest_path = output_directory / "artifact-manifest.json"
        atomic_json(
            artifact_manifest_path,
            {
                "schema_version": 1,
                "artifact_count": 0,
                "licenses_verified": True,
                "artifacts": [],
            },
        )
        data_manifest_path = output_directory / "data-manifest.json"
        atomic_json(
            data_manifest_path,
            {
                "schema_version": 1,
                "dataset_id": "analysis-plan-synthetic-assumptions",
                "split": "pre-evaluation-design",
                "task_count": 0,
                "locked_public_outcomes_used": False,
                "simulation_task_counts": simulation_config["task_counts"],
                "simulation_assumptions_sha256": sha256_file(power_path),
            },
        )
        environment_path = output_directory / "environment-lock.json"
        atomic_json(
            environment_path,
            {
                "schema_version": 1,
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "requirements_path": "requirements/analysis.txt",
                "requirements_sha256": sha256_file(
                    ROOT / "requirements" / "analysis.txt"
                ),
                "network_used": False,
            },
        )
        hardware_path = output_directory / "hardware-manifest.json"
        atomic_json(
            hardware_path,
            {
                "schema_version": 1,
                "profile_id": "host-20260806-cpu-analysis-preflight",
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
        )
        results_path = output_directory / "results-summary.json"
        minimum_effects = {
            task_count: result["minimum_grid_effect_at_target_power"]
            for task_count, result in simulation_first["results"].items()
        }
        atomic_json(
            results_path,
            {
                "schema_version": 1,
                "status": "passed",
                "plan_id": config["plan_id"],
                "freeze_status": config["freeze_status"],
                "target_venue": "NeurIPS",
                "target_track": "Evaluations & Datasets",
                "target_cycle": 2027,
                "official_2027_author_cfp_status": "not_published_as_of_2026-08-06",
                "statistical_self_test_passed": self_test["all_passed"],
                "power_simulation_deterministic": deterministic_simulation,
                "minimum_grid_effect_at_80_percent_power": minimum_effects,
                "campaign_gpu_hour_cap": config["campaign_cap"][
                    "local_gpu_hour_cap_including_contingency"
                ],
                "api_spend_cap_usd": 0,
                "h3_status": config["hypotheses"][2]["status"],
                "new_locked_public_method_scores_used": 0,
            },
        )
        content_manifest_path = output_directory / "content-manifest.json"
        content_entries = []
        for path in sorted(output_directory.rglob("*")):
            if path.is_file() and path.name not in {"run.json", "content-manifest.json"}:
                content_entries.append(
                    {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
        atomic_json(
            content_manifest_path,
            {
                "schema_version": 1,
                "file_count": len(content_entries),
                "files": content_entries,
            },
        )
        status = "passed"
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    usage = monitor.stop()

    if status == "passed":
        role_paths = {
            "protocol_manifest": protocol_snapshot,
            "source_lock": source_lock_path,
            "source_patch": source_patch_path,
            "artifact_manifest": artifact_manifest_path,
            "data_manifest": data_manifest_path,
            "config": config_copy,
            "environment_lock": environment_path,
            "hardware_manifest": hardware_path,
            "results": results_path,
            "content_manifest": content_manifest_path,
        }
        files = [file_record(SCHEMA_PATH, role="schema")]
        used_paths = {SCHEMA_PATH.resolve()}
        for role, path in role_paths.items():
            files.append(file_record(path, role=role))
            used_paths.add(path.resolve())
        for path in sorted(output_directory.rglob("*")):
            if (
                path.is_file()
                and path.name != "run.json"
                and path.resolve() not in used_paths
            ):
                files.append(file_record(path, role="other"))
        dirty = git_dirty()
        record = {
            "schema_version": "protocol-v1-run-1.0.0",
            "schema_digest_sha256": sha256_file(SCHEMA_PATH),
            "protocol_id": "arc-rebench-protocol-v1-draft",
            "protocol_digest_sha256": sha256_file(protocol_snapshot),
            "method_id": "e0-analysis-plan",
            "config_id": "analysis-plan-v1",
            "run_id": output_directory.name,
            "status": "passed",
            "evidence_scope": "e0_audit",
            "parity_class": "not_applicable",
            "resource_class": "local_cpu",
            "code_trust_class": "trusted_locked",
            "claim": (
                "Analysis plan v1 freezes the NeurIPS 2027 E&D target cycle, "
                "estimands, hypothesis and multiplicity hierarchy, clustered "
                "model and fallback triggers, missingness rules, and campaign cap "
                "before any new protocol-locked public method score; deterministic "
                "synthetic power and failure-mode preflights pass."
            ),
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "source": {
                "lock_digest_sha256": sha256_file(source_lock_path),
                "revision": git_revision(),
                "dirty": dirty,
                "patch_digest_sha256": sha256_file(source_patch_path)
                if dirty
                else None,
            },
            "artifacts": {
                "manifest_digest_sha256": sha256_file(artifact_manifest_path),
                "artifact_count": 0,
                "licenses_verified": True,
            },
            "data": {
                "manifest_digest_sha256": sha256_file(data_manifest_path),
                "dataset_id": "analysis-plan-synthetic-assumptions",
                "split": "pre-evaluation-design",
                "task_count": 0,
                "contamination_policy": "not_applicable",
            },
            "config": {
                "digest_sha256": sha256_file(config_copy),
                "seed": config["power_simulation"]["seed"],
                "deterministic": True,
            },
            "hardware": {
                "profile_id": "host-20260806-cpu-analysis-preflight",
                "manifest_digest_sha256": sha256_file(hardware_path),
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
            "execution": {
                "runner": "scripts.freeze_analysis_plan",
                "command": [
                    "python3",
                    "scripts/freeze_analysis_plan.py",
                    "--config",
                    args.config.relative_to(ROOT).as_posix(),
                    "--output-directory",
                    output_directory.relative_to(ROOT).as_posix(),
                ],
                "working_directory": ".",
                "environment_digest_sha256": sha256_file(environment_path),
                "claim_execution_started": True,
                "target_code_executed": True,
                "network_used": False,
                "gpu_requested": False,
            },
            "challenge_firewall": {
                "challenge_manifest_digest_sha256": None,
                "inference_received_test_labels": False,
                "inference_started": False,
                "scoring_after_inference": False,
                "label_mutation_check": "not_applicable",
                "network_policy": "not_applicable",
                "write_policy": "run_directory_only",
                "security_isolation": "trusted_process",
            },
            "attempt_budget": {
                "top_k": 0,
                "timeout_seconds": 300,
                "max_retries": 0,
                "max_candidates": 0,
                "api_call_cap": 0,
                "input_token_cap": 0,
                "output_token_cap": 0,
                "cost_cap_usd": 0,
            },
            "resources": {
                "accounting_scope": "current_process",
                "child_processes_observed": False,
                "children_included": False,
                "sampling": "sampled",
                "wall_time_seconds": usage.wall_time_seconds,
                "cpu_seconds": usage.process_cpu_seconds,
                "peak_rss_bytes": usage.sampled_peak_current_rss_bytes,
                "peak_vram_bytes": None,
                "energy_joules": None,
                "disk_delta_bytes": None,
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
            },
            "results": {
                "kind": "check",
                "predictions_path": None,
                "score_path": None,
                "primary_metric": None,
                "secondary_metrics": [],
                "checks": [
                    {
                        "name": "venue-cycle-freeze",
                        "status": "passed",
                        "detail": (
                            "NeurIPS 2027 E&D is selected; unpublished 2027 CFP, "
                            "portal, and date fields remain null and cannot be inferred."
                        ),
                    },
                    {
                        "name": "statistical-preflight",
                        "status": "passed",
                        "detail": (
                            "Holm, exact paired, cluster bootstrap, clustered logistic "
                            "Wald, and separation/singularity/non-convergence fallback "
                            "fixtures all passed."
                        ),
                    },
                    {
                        "name": "power-replay",
                        "status": "passed",
                        "detail": (
                            "Synthetic exact-paired power grids for 64, 120, and 400 "
                            "tasks replay byte-equivalently from the frozen seed."
                        ),
                    },
                    {
                        "name": "campaign-cap",
                        "status": "passed",
                        "detail": (
                            "The ceiling is 1,500 local GPU-hours and USD 0 API spend; "
                            "it is not a reservation or public-run authorization."
                        ),
                    },
                ],
            },
            "failures": {"count": 0, "items": []},
            "files": files,
            "limitations": [
                "NeurIPS has referenced a 2027 E&D cycle but has not published its author CFP, dates, portal, or final policies as of 2026-08-06; administrative compliance must be rechecked without changing scientific decisions after outcomes.",
                "The power grid uses synthetic paired assumptions and measures the exact fallback only; it is not an empirical ARC result or a claim about interaction-model power.",
                "H1 and H2 remain conditional on method eligibility, while H3 is declared infeasible for protocol v1 because zero methods have strict runtime promotion at freeze.",
                "The 1,500 GPU-hour cap is a maximum design ceiling, not hardware availability, allocation, or permission to displace another process.",
                "API spend and execution remain unauthorized at USD 0 absent explicit user approval.",
                "This analysis freeze does not authorize locked-public solver execution while other required protocol gates remain unmet.",
            ],
        }
        run_path = output_directory / "run.json"
        atomic_json(run_path, record)
        try:
            validation = validate_run_file(
                run_path,
                schema_path=SCHEMA_PATH,
                repo_root=ROOT,
                verify_files=True,
            )
        except BaseException as error:
            status = "failed"
            failure = {"type": type(error).__name__, "message": str(error)}
        else:
            print(json.dumps(validation.as_dict(), indent=2, sort_keys=True))
    if status != "passed":
        failed_record = {
            "schema_version": 1,
            "method_id": "e0-analysis-plan",
            "run_id": output_directory.name,
            "runner": "scripts.freeze_analysis_plan",
            "status": "failed",
            "scope": "analysis-plan-v1-freeze",
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "error": failure,
            "resources": usage.to_dict(),
        }
        atomic_json(output_directory / "run.json", failed_record)
        print(json.dumps(failed_record, indent=2, sort_keys=True))
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
