#!/usr/bin/env python3
"""Freeze and strictly audit the protocol-v1 input/configuration bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.challenge_runtime import (  # noqa: E402
    canonical_sha256,
    pretty_json_bytes,
    sha256_file,
    utc_now,
)
from arc_agi_eval.input_bundle import (  # noqa: E402
    build_code_inventory,
    build_public_task_orders,
    verify_challenge_view,
    verify_declared_inputs,
)
from arc_agi_eval.resources import ResourceMonitor  # noqa: E402
from arc_agi_eval.run_schema import validate_run_file  # noqa: E402


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


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


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


def bundle_leaf_paths(
    output_directory: Path, *, excluded_relative_paths: set[str]
) -> list[Path]:
    """Return every regular bundle leaf except exact root-relative exclusions.

    Exclusions are matched against the complete path relative to the bundle
    root.  In particular, excluding ``run.json`` never excludes a declared
    snapshot such as ``inputs/declared/reports/.../run.json``.
    """

    root = output_directory.resolve()
    normalized_exclusions: set[str] = set()
    for declared in excluded_relative_paths:
        if not isinstance(declared, str) or not declared:
            raise ValueError(f"invalid bundle leaf exclusion: {declared!r}")
        relative = Path(declared)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != declared
        ):
            raise ValueError(f"invalid bundle leaf exclusion: {declared!r}")
        normalized_exclusions.add(declared)

    leaves: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"bundle inventory refuses symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in normalized_exclusions:
            leaves.append(path)
    return leaves


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


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("input-freeze schema version must be 1")
    if config.get("bundle_id") != "arc-rebench-input-freeze-v1":
        raise ValueError("unexpected input-freeze bundle id")
    if config.get("freeze_status") != "frozen-no-admitted-method-configurations":
        raise ValueError("input-freeze status is not frozen")
    declared = config.get("declared_input_files")
    if not isinstance(declared, list) or not declared:
        raise ValueError("at least one declared input is required")
    code = config.get("code_inventory")
    if not isinstance(code, dict) or code.get("include_globs") != [
        "arc_agi_eval/**/*.py",
        "scripts/*.py",
        "tests/*.py",
    ]:
        raise ValueError("code inventory globs differ from protocol v1")
    admission = config.get("method_admission")
    expected_admission_keys = {
        "expected_entry_count",
        "expected_legacy_solver_smoke_passed",
        "expected_strict_runtime_passed",
        "expected_performance_eligible",
        "public_method_configurations_admitted",
        "strict_config_id_required_for_admission",
    }
    if not isinstance(admission, dict) or set(admission) != expected_admission_keys:
        raise ValueError("method-admission freeze has missing or unknown fields")
    if admission["expected_entry_count"] != 24:
        raise ValueError("method-admission roster must contain 24 methods")
    if admission["expected_legacy_solver_smoke_passed"] != 2:
        raise ValueError("legacy solver-smoke count must remain two")
    if type(admission["expected_strict_runtime_passed"]) is not int or not (
        0 <= admission["expected_strict_runtime_passed"] <= 24
    ):
        raise ValueError("expected strict-runtime count is invalid")
    if admission["expected_performance_eligible"] != 0:
        raise ValueError("this protocol version admits no performance-table methods")
    if admission["public_method_configurations_admitted"] != 0:
        raise ValueError("this protocol version authorizes no public method configs")
    if admission["strict_config_id_required_for_admission"] is not True:
        raise ValueError("strict config identity must remain mandatory")
    invalidation = config.get("invalidation_policy")
    required_invalidation = {
        "method_promotion_requires_new_bundle",
        "any_declared_input_hash_change_requires_new_bundle",
        "any_frozen_code_change_requires_new_bundle",
        "any_task_order_change_requires_new_bundle",
        "any_budget_seed_retry_change_requires_new_bundle",
        "new_bundle_never_overwrites_old_bundle",
    }
    if not isinstance(invalidation, dict) or set(invalidation) != required_invalidation:
        raise ValueError("input-bundle invalidation policy is incomplete")
    if not all(invalidation.values()):
        raise ValueError("every input-bundle invalidation rule must fail closed")
    ordering = config.get("public_task_order")
    if not isinstance(ordering, dict):
        raise ValueError("public task-order configuration is missing")
    if ordering.get("domain") != "arc-rebench-public-task-order-v1":
        raise ValueError("unexpected public task-order domain")
    if ordering.get("seeds") != [2026080601, 2026080602, 2026080603]:
        raise ValueError("public task-order seeds differ from the frozen sequence")
    views = ordering.get("views")
    if not isinstance(views, dict) or set(views) != {"arc_agi_1", "arc_agi_2"}:
        raise ValueError("public task-order views must contain ARC-AGI-1 and ARC-AGI-2")


def unique_role_path(
    inventory: list[dict[str, object]], role: str
) -> Path:
    matches = [item for item in inventory if item["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"declared input role {role!r} is not unique")
    return ROOT / str(matches[0]["path"])


def validate_methods(
    declared: list[dict[str, object]], config: dict[str, Any]
) -> dict[str, object]:
    baselines = load_object(unique_role_path(declared, "method-roster"))
    locks = load_object(unique_role_path(declared, "source-locks"))
    assets = load_object(unique_role_path(declared, "artifact-policy"))
    eligibility = load_object(unique_role_path(declared, "method-eligibility"))

    baseline_entries = baselines.get("entries")
    eligibility_entries = eligibility.get("entries")
    sources = locks.get("sources")
    papers = assets.get("papers")
    if not isinstance(baseline_entries, list) or not isinstance(
        eligibility_entries, list
    ):
        raise ValueError("baseline and eligibility entries must be arrays")
    if not isinstance(sources, dict) or not isinstance(papers, dict):
        raise ValueError("source-lock and paper-asset method maps are malformed")
    baseline_ids = [item.get("id") for item in baseline_entries]
    eligibility_ids = [item.get("id") for item in eligibility_entries]
    if len(baseline_ids) != 24 or len(set(baseline_ids)) != 24:
        raise ValueError("baseline roster must contain 24 unique method IDs")
    expected_set = set(baseline_ids)
    if eligibility_ids != baseline_ids:
        raise ValueError("eligibility order differs from baseline roster order")
    if set(sources) != expected_set or set(papers) != expected_set:
        raise ValueError("source-lock/paper-asset method sets differ from the roster")

    legacy_passed = sum(
        item.get("solver_prediction_smoke", {}).get("status") == "passed"
        for item in eligibility_entries
    )
    strict_passed = sum(
        item.get("strict_runtime_promotion", {}).get("status") == "passed"
        for item in eligibility_entries
    )
    performance_eligible = sum(
        item.get("performance_table_eligible") is True
        for item in eligibility_entries
    )
    nonnull_config_ids = [
        item["id"]
        for item in eligibility_entries
        if item.get("strict_runtime_promotion", {}).get("config_id") is not None
    ]
    admitted = [
        {
            "method_id": item["id"],
            "config_id": item["strict_runtime_promotion"]["config_id"],
        }
        for item in eligibility_entries
        if item.get("strict_runtime_promotion", {}).get("status") == "passed"
        and item.get("performance_table_eligible") is True
        and item.get("strict_runtime_promotion", {}).get("config_id") is not None
    ]
    expected = config["method_admission"]
    observed = {
        "entry_count": len(eligibility_entries),
        "legacy_solver_prediction_smoke_passed": legacy_passed,
        "strict_runtime_passed": strict_passed,
        "performance_eligible": performance_eligible,
        "admitted_configuration_count": len(admitted),
    }
    expected_observed = {
        "entry_count": expected["expected_entry_count"],
        "legacy_solver_prediction_smoke_passed": expected[
            "expected_legacy_solver_smoke_passed"
        ],
        "strict_runtime_passed": expected["expected_strict_runtime_passed"],
        "performance_eligible": expected["expected_performance_eligible"],
        "admitted_configuration_count": expected[
            "public_method_configurations_admitted"
        ],
    }
    if observed != expected_observed:
        raise ValueError(
            f"method-admission state changed; expected {expected_observed}, observed {observed}"
        )
    if len(nonnull_config_ids) != strict_passed:
        raise ValueError(
            "non-null strict config IDs must exactly match passed promotions"
        )
    summary = eligibility.get("summary", {})
    if summary.get("entry_count") != 24:
        raise ValueError("eligibility summary entry count differs from records")
    if (
        summary.get("strict_runtime_promotion_status", {}).get("passed")
        != strict_passed
    ):
        raise ValueError("eligibility summary strict-promotion count differs")
    if summary.get("performance_table_eligible", {}).get("eligible") != 0:
        raise ValueError("eligibility summary claims a performance-eligible method")
    source_lock_nonnull = sum(value is not None for value in sources.values())
    source_lock_null = sum(value is None for value in sources.values())
    if (source_lock_nonnull, source_lock_null) != (20, 4):
        raise ValueError("source-lock availability changed from frozen 20/4 state")
    return {
        **observed,
        "method_ids": baseline_ids,
        "method_id_order_sha256": canonical_sha256(baseline_ids),
        "strict_config_ids_nonnull": len(nonnull_config_ids),
        "admitted_configurations": admitted,
        "source_locks_nonnull": source_lock_nonnull,
        "source_locks_unavailable": source_lock_null,
        "source_lock_unavailable_method_ids": sorted(
            method_id for method_id, value in sources.items() if value is None
        ),
        "paper_asset_records": len(papers),
    }


def build_task_orders_and_data(
    config: dict[str, Any], declared: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    order_config = config["public_task_order"]
    audited_views: dict[str, dict[str, object]] = {}
    view_summaries: dict[str, dict[str, object]] = {}
    for benchmark, view in sorted(order_config["views"].items()):
        audit = verify_challenge_view(
            ROOT / view["directory"],
            expected_task_count=view["expected_task_count"],
        )
        if audit["test_input_count"] != view["expected_output_denominator"]:
            raise ValueError(f"{benchmark} output denominator changed")
        audited_views[benchmark] = audit
        view_summaries[benchmark] = {
            key: value for key, value in audit.items() if key != "records"
        }
        view_summaries[benchmark]["directory"] = view["directory"]

    first = build_public_task_orders(
        audited_views,
        domain=order_config["domain"],
        seeds=order_config["seeds"],
    )
    second = build_public_task_orders(
        audited_views,
        domain=order_config["domain"],
        seeds=order_config["seeds"],
    )
    if first != second:
        raise ValueError("public task orders did not replay deterministically")

    development_path = unique_role_path(declared, "development-manifest")
    development = load_object(development_path)
    development_ids = [
        Path(item["path"]).stem
        for item in development.get("inference_file_inventory", [])
        if item.get("path", "").endswith(".json")
    ]
    if len(development_ids) != 94 or len(set(development_ids)) != 94:
        raise ValueError("development representative order is not 94 unique tasks")
    if development.get("dev_audit", {}).get("test_output_denominator") != 97:
        raise ValueError("development output denominator changed")

    isoarc_path = unique_role_path(declared, "isoarc-manifest")
    isoarc = load_object(isoarc_path)
    assignments = isoarc.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 64:
        raise ValueError("fixed64 assignment count changed")
    fixed64_records = [
        {
            "benchmark_generation": item["benchmark_generation"],
            "task_id": item["task_id"],
        }
        for item in assignments
    ]
    if len({item["task_id"] for item in fixed64_records}) != 64:
        raise ValueError("fixed64 task IDs are not unique")

    task_orders: dict[str, object] = {
        **first,
        "deterministic_replay": True,
        "development": {
            "source_manifest_path": development_path.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": sha256_file(development_path),
            "task_count": 94,
            "test_output_denominator": 97,
            "task_ids": development_ids,
            "order_sha256": canonical_sha256(development_ids),
            "source_representative_order_sha256": development["dev_audit"][
                "representative_order_sha256"
            ],
        },
        "fixed64": {
            "source_manifest_path": isoarc_path.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": sha256_file(isoarc_path),
            "task_count": 64,
            "base_tasks": fixed64_records,
            "order_sha256": canonical_sha256(fixed64_records),
        },
    }
    task_orders["task_orders_sha256"] = canonical_sha256(task_orders)
    data = {
        "public_views": view_summaries,
        "public_task_count": sum(item["task_count"] for item in view_summaries.values()),
        "public_test_input_count": sum(
            item["test_input_count"] for item in view_summaries.values()
        ),
        "public_test_output_fields_present": sum(
            item["test_output_fields_present"] for item in view_summaries.values()
        ),
        "development_task_count": 94,
        "development_test_output_denominator": 97,
        "fixed64_base_task_count": 64,
    }
    if (data["public_task_count"], data["public_test_input_count"]) != (520, 586):
        raise ValueError("public challenge totals changed from 520 tasks / 586 inputs")
    return task_orders, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "input_freeze_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "e0-freeze"
            / "20260806-input-bundle-v1-retry10"
        ),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_directory = args.output_directory.resolve()
    try:
        config_path.relative_to(ROOT)
        output_directory.relative_to(ROOT)
    except ValueError as error:
        parser.error(f"config and output directory must remain inside repository: {error}")
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    monitor = ResourceMonitor(include_nvidia=False).start()
    status = "failed"
    failure: dict[str, str] | None = None
    record: dict[str, object] | None = None
    try:
        config = load_object(config_path)
        validate_config(config)
        declared = verify_declared_inputs(ROOT, config["declared_input_files"])
        methods = validate_methods(declared, config)
        task_orders, data_summary = build_task_orders_and_data(config, declared)

        declared_copies: list[dict[str, object]] = []
        for item in declared:
            source = ROOT / str(item["path"])
            target = output_directory / "inputs" / "declared" / str(item["path"])
            atomic_bytes(target, source.read_bytes())
            if sha256_file(target) != item["sha256"]:
                raise ValueError(f"declared input copy mismatch: {item['path']}")
            declared_copies.append(
                {
                    **item,
                    "snapshot_path": target.relative_to(ROOT).as_posix(),
                }
            )
        declared_inventory_path = output_directory / "declared-input-inventory.json"
        atomic_json(
            declared_inventory_path,
            {
                "schema_version": 1,
                "file_count": len(declared_copies),
                "files": declared_copies,
                "declared_input_inventory_sha256": canonical_sha256(declared_copies),
            },
        )

        code_records = build_code_inventory(
            ROOT, config["code_inventory"]["include_globs"]
        )
        code_copies: list[dict[str, object]] = []
        for item in code_records:
            source = ROOT / str(item["path"])
            target = output_directory / "inputs" / "code" / str(item["path"])
            atomic_bytes(target, source.read_bytes())
            if sha256_file(target) != item["sha256"]:
                raise ValueError(f"code snapshot mismatch: {item['path']}")
            code_copies.append(
                {
                    **item,
                    "snapshot_path": target.relative_to(ROOT).as_posix(),
                }
            )
        code_inventory_path = output_directory / "code-inventory.json"
        atomic_json(
            code_inventory_path,
            {
                "schema_version": 1,
                "include_globs": config["code_inventory"]["include_globs"],
                "file_count": len(code_copies),
                "files": code_copies,
                "code_inventory_sha256": canonical_sha256(code_copies),
            },
        )

        task_orders_path = output_directory / "task-orders.json"
        atomic_json(task_orders_path, task_orders)
        analysis = load_object(unique_role_path(declared, "analysis-plan"))
        bundle_manifest: dict[str, object] = {
            "schema_version": 1,
            "bundle_id": config["bundle_id"],
            "freeze_status": config["freeze_status"],
            "methods": methods,
            "authorization": {
                "locked_public_solver_run": False,
                "reason": (
                    f"{methods['strict_runtime_passed']} method configurations have "
                    "strict runtime promotion, but zero configurations are performance-"
                    "eligible or admitted; the required process-tree resource gate also "
                    "remains unmet."
                ),
                "new_protocol_locked_public_method_scores_authorized": 0,
            },
            "inputs": {
                "declared_input_count": len(declared_copies),
                "declared_input_inventory_path": declared_inventory_path.relative_to(
                    ROOT
                ).as_posix(),
                "declared_input_inventory_sha256": canonical_sha256(declared_copies),
                "code_file_count": len(code_copies),
                "code_inventory_path": code_inventory_path.relative_to(ROOT).as_posix(),
                "code_inventory_sha256": canonical_sha256(code_copies),
            },
            "task_orders": {
                "path": task_orders_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(task_orders_path),
                "task_orders_sha256": task_orders["task_orders_sha256"],
                "public_order_count_per_benchmark": len(
                    config["public_task_order"]["seeds"]
                ),
                "deterministic_replay": True,
            },
            "data": data_summary,
            "budget_seed_retry_freeze": {
                "analysis_plan_id": analysis["plan_id"],
                "campaign_cap": analysis["campaign_cap"],
                "metrics": analysis["metrics"],
                "confidence_interval_seed": analysis["confidence_intervals"][
                    "deterministic_seed"
                ],
                "task_order_seeds": config["public_task_order"]["seeds"],
                "infrastructure_rerun_rule": analysis["data_and_missingness"][
                    "infrastructure_rerun_rule"
                ],
                "post_retry_failure_rule": analysis["data_and_missingness"][
                    "malformed_missing_timeout_oom_after_frozen_retries"
                ],
            },
            "invalidation_policy": config["invalidation_policy"],
        }
        bundle_manifest["bundle_root_sha256"] = canonical_sha256(bundle_manifest)
        bundle_manifest_path = output_directory / "bundle-manifest.json"
        atomic_json(bundle_manifest_path, bundle_manifest)

        config_copy = output_directory / "config.json"
        atomic_bytes(config_copy, config_path.read_bytes())
        protocol_snapshot = output_directory / "protocol-manifest.json"
        atomic_json(
            protocol_snapshot,
            {
                "schema_version": 1,
                "protocol_id": "arc-rebench-protocol-v1-draft",
                "protocol_status": "draft-not-frozen",
                "gate_id": "lp.freeze-inputs",
                "protocol_config_path": PROTOCOL_CONFIG.relative_to(ROOT).as_posix(),
                "protocol_config_sha256_at_execution": sha256_file(PROTOCOL_CONFIG),
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
                "patch_manifest_path": code_inventory_path.relative_to(ROOT).as_posix(),
                "patch_manifest_sha256": sha256_file(code_inventory_path),
                "frozen_code_inventory_sha256": canonical_sha256(code_copies),
            },
        )
        artifact_manifest_path = output_directory / "artifact-manifest.json"
        atomic_json(
            artifact_manifest_path,
            {
                "schema_version": 1,
                "artifact_count": 0,
                "licenses_verified": True,
                "license_statement_scope": "empty admitted configuration set only",
                "all_roster_method_licenses_cleared": False,
                "artifacts": [],
            },
        )
        data_manifest_path = output_directory / "data-manifest.json"
        atomic_json(
            data_manifest_path,
            {
                "schema_version": 1,
                "dataset_id": "arc-rebench-public-challenges-v1",
                "split": "locked-public-inputs",
                "task_count": data_summary["public_task_count"],
                "test_input_count": data_summary["public_test_input_count"],
                "test_output_fields_present": data_summary[
                    "public_test_output_fields_present"
                ],
                "public_views": data_summary["public_views"],
                "development": {
                    "task_count": 94,
                    "test_output_denominator": 97,
                    "source_manifest_path": task_orders["development"][
                        "source_manifest_path"
                    ],
                    "source_manifest_sha256": task_orders["development"][
                        "source_manifest_sha256"
                    ],
                },
                "fixed64": {
                    "base_task_count": 64,
                    "source_manifest_path": task_orders["fixed64"][
                        "source_manifest_path"
                    ],
                    "source_manifest_sha256": task_orders["fixed64"][
                        "source_manifest_sha256"
                    ],
                },
                "historical_public_label_exposure_exists": True,
                "solver_inference_executed": False,
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
                "network_used": False,
                "gpu_used": False,
            },
        )
        hardware_path = output_directory / "hardware-manifest.json"
        atomic_json(
            hardware_path,
            {
                "schema_version": 1,
                "profile_id": "host-20260806-cpu-input-freeze",
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
        )
        results_path = output_directory / "results-summary.json"
        atomic_json(
            results_path,
            {
                "schema_version": 1,
                "status": "passed",
                "bundle_id": config["bundle_id"],
                "bundle_root_sha256": bundle_manifest["bundle_root_sha256"],
                "declared_input_count": len(declared_copies),
                "code_file_count": len(code_copies),
                "public_task_count": data_summary["public_task_count"],
                "public_test_input_count": data_summary["public_test_input_count"],
                "development_task_count": 94,
                "fixed64_base_task_count": 64,
                "method_entry_count": methods["entry_count"],
                "strict_runtime_passed": methods["strict_runtime_passed"],
                "admitted_configuration_count": methods[
                    "admitted_configuration_count"
                ],
                "locked_public_solver_run_authorized": False,
                "task_order_replay_passed": True,
            },
        )

        content_manifest_path = output_directory / "content-manifest.json"
        content_entries = []
        for path in bundle_leaf_paths(
            output_directory,
            excluded_relative_paths={"run.json", "content-manifest.json"},
        ):
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
            "source_patch": code_inventory_path,
            "artifact_manifest": artifact_manifest_path,
            "data_manifest": data_manifest_path,
            "config": config_copy,
            "environment_lock": environment_path,
            "hardware_manifest": hardware_path,
            "challenge_manifest": bundle_manifest_path,
            "results": results_path,
            "content_manifest": content_manifest_path,
        }
        files = [file_record(SCHEMA_PATH, role="schema")]
        used_paths = {SCHEMA_PATH.resolve()}
        for role, path in role_paths.items():
            files.append(file_record(path, role=role))
            used_paths.add(path.resolve())
        for path in bundle_leaf_paths(
            output_directory, excluded_relative_paths={"run.json"}
        ):
            if path.resolve() not in used_paths:
                files.append(file_record(path, role="other"))
        dirty = git_dirty()
        record = {
            "schema_version": "protocol-v1-run-1.0.0",
            "schema_digest_sha256": sha256_file(SCHEMA_PATH),
            "protocol_id": "arc-rebench-protocol-v1-draft",
            "protocol_digest_sha256": sha256_file(protocol_snapshot),
            "method_id": "e0-input-freeze",
            "config_id": "input-bundle-v1",
            "run_id": output_directory.name,
            "status": "passed",
            "evidence_scope": "e0_audit",
            "parity_class": "not_applicable",
            "resource_class": "local_cpu",
            "code_trust_class": "trusted_locked",
            "claim": (
                f"Input bundle v1 binds {len(declared_copies)} declared inputs, repository-native code, "
                "520 label-free locked-public tasks, three deterministic task orders "
                "per benchmark, development/fixed64 orders, and frozen budgets, seeds, "
                f"and retry policy while recording {methods['strict_runtime_passed']} strict runtime promotions, admitting zero method configurations, and "
                "authorizing no locked-public solver execution."
            ),
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "source": {
                "lock_digest_sha256": sha256_file(source_lock_path),
                "revision": git_revision(),
                "dirty": dirty,
                "patch_digest_sha256": sha256_file(code_inventory_path)
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
                "dataset_id": "arc-rebench-public-challenges-v1",
                "split": "locked-public-inputs",
                "task_count": 520,
                "contamination_policy": "historical",
            },
            "config": {
                "digest_sha256": sha256_file(config_copy),
                "seed": config["public_task_order"]["seeds"][0],
                "deterministic": True,
            },
            "hardware": {
                "profile_id": "host-20260806-cpu-input-freeze",
                "manifest_digest_sha256": sha256_file(hardware_path),
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
            "execution": {
                "runner": "scripts.freeze_input_bundle",
                "command": [
                    "python3",
                    "scripts/freeze_input_bundle.py",
                    "--config",
                    config_path.relative_to(ROOT).as_posix(),
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
                "challenge_manifest_digest_sha256": sha256_file(
                    bundle_manifest_path
                ),
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
                        "name": "declared-input-hashes",
                        "status": "passed",
                        "detail": f"All {len(declared_copies)} declared input hashes verified and were copied byte-for-byte into the immutable report.",
                    },
                    {
                        "name": "label-free-public-views",
                        "status": "passed",
                        "detail": "All 520 public challenge files and 586 test inputs verified against visible manifests; zero test output fields were present.",
                    },
                    {
                        "name": "deterministic-task-orders",
                        "status": "passed",
                        "detail": "Three orders per public benchmark replayed exactly; development has 94 representatives and fixed64 has 64 base tasks.",
                    },
                    {
                        "name": "method-admission",
                        "status": "passed",
                        "detail": f"The 24-method roster has two legacy solver smokes, {methods['strict_runtime_passed']} strict runtime promotions, zero eligible methods, and zero admitted configurations.",
                    },
                    {
                        "name": "public-run-authorization",
                        "status": "passed",
                        "detail": "Locked-public solver execution is explicitly false; promotion or any frozen-input/code/order/budget change requires a new immutable bundle.",
                    },
                ],
            },
            "failures": {"count": 0, "items": []},
            "files": files,
            "limitations": [
                "The empty admitted set makes artifact-license clearance vacuous for this bundle; it does not claim that all 24 roster methods have cleared licenses.",
                "Twenty methods have pinned source-lock metadata and four unavailable methods have null source locks; null locks are preserved rather than inferred.",
                "This bundle freezes metadata and code snapshots but is not a compute allocation, disk reservation, process-tree accounting mechanism, or GPU exclusivity grant.",
                "The required process-tree resource gate remains pending, so this passed input-freeze gate does not freeze the overall protocol or authorize a public solver run.",
                "Any method strict-runtime promotion, declared input or code hash change, task-order change, or budget/seed/retry change invalidates this bundle and requires a new immutable version.",
                "Public ARC labels have historical exposure in this workspace; this audit reads only sanitized challenge trees and produces no solver predictions or performance result.",
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
            "method_id": "e0-input-freeze",
            "run_id": output_directory.name,
            "runner": "scripts.freeze_input_bundle",
            "status": "failed",
            "scope": "protocol-v1-input-bundle-freeze",
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
