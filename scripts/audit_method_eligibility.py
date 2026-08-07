#!/usr/bin/env python3
"""Validate the per-method eligibility inventory and freeze an audit record."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.run_schema import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    PREDICTION_SCOPES,
    validate_run_file,
)


GLOBAL_RUNTIME_CORE_METHOD_IDS = frozenset({"deterministic-floor-runtime-core"})
GLOBAL_RUNTIME_CORE_CONFIG_IDS = frozenset({"challenge-runtime-core-v1"})

CATEGORY_ENUMS: dict[str, tuple[str, ...]] = {
    "native_benchmark_generation": (
        "arc_agi_1",
        "arc_agi_1_and_2",
        "arc_agi_2_primary_arc_agi_1_backtest",
        "native_non_arc_agi",
    ),
    "cohort": (
        "arc_prize_2024",
        "arc_prize_2025",
        "verified_arc_prize_2026",
        "non_arc_prize_research",
    ),
    "primary_mechanism_family": (
        "symbolic_program_search",
        "task_specific_neural_adaptation",
        "pretrained_neural_or_llm_solver",
        "ensemble",
        "api_native_arc_solver",
        "native_multi_agent",
        "native_multi_agent_router",
        "unknown",
    ),
    "evidence_scope": (
        "unavailable",
        "blocker_audit",
        "scorer_only",
        "component",
        "solver_prediction",
        "benchmark",
    ),
    "parity_class": (
        "unavailable",
        "scorer_only",
        "component_only",
        "blocked_before_method_execution",
        "reduced_method_execution",
        "native_method_execution",
        "benchmark_parity",
        "paper_parity",
    ),
    "resource_class": (
        "cpu_or_local_light",
        "single_gpu_24g",
        "single_gpu_long_running",
        "multi_gpu_or_above_host",
        "metered_api",
        "unknown",
        "unavailable",
    ),
    "code_trust_class": (
        "trusted_locked",
        "generated_untrusted",
        "unsafe_artifact",
        "api_network",
        "unavailable",
    ),
}

OTHER_ENUMS: dict[str, tuple[str, ...]] = {
    "solver_prediction_smoke_status": (
        "passed",
        "failed",
        "blocked",
        "not_run",
        "unavailable",
    ),
    "strict_runtime_promotion_status": (
        "passed",
        "failed",
        "blocked",
        "not_run",
        "unavailable",
    ),
    "blocking_gate_status": ("failed", "unknown"),
    "blocking_gate": (
        "source_revision_lock",
        "license_clearance",
        "artifact_provenance",
        "dataset_provenance",
        "dependency_environment",
        "label_firewall",
        "no_label_solver_prediction",
        "configuration_class",
        "timeout_best_effort",
        "state_retry_seed_contract",
        "resource_capacity",
        "code_isolation",
        "api_budget",
        "benchmark_coverage",
        "implementation_available",
        "native_benchmark_contract",
        "contamination_control",
        "runtime_portability",
    ),
}

CLASSIFICATION_EVIDENCE_KEYS = ("provenance", "taxonomy", "execution")
RISK_CLASSES = {"generated_untrusted", "unsafe_artifact", "api_network"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def immutable_json(path: Path, value: dict[str, object]) -> None:
    """Atomically create a JSON file and refuse to replace an existing record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable audit record: {path}"
            ) from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def error_record(
    kind: str,
    *,
    method_id: str | None = None,
    field: str | None = None,
    evidence: object = None,
    detail: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {"kind": kind}
    if method_id is not None:
        record["method_id"] = method_id
    if field is not None:
        record["field"] = field
    if evidence is not None:
        record["evidence"] = evidence
    if detail is not None:
        record["detail"] = detail
    return record


def resolve_repository_path(root: Path, declared: str) -> tuple[Path | None, str | None]:
    relative = Path(declared)
    if relative.is_absolute():
        return None, "path must be repository-relative"
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, "path escapes the repository root"
    return resolved, None


def _string_list(
    value: object,
    *,
    errors: list[dict[str, object]],
    method_id: str,
    field: str,
    nonempty: bool = True,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(
            error_record(
                "invalid-string-list", method_id=method_id, field=field, evidence=value
            )
        )
        return []
    strings = list(value)
    if nonempty and not strings:
        errors.append(error_record("empty-string-list", method_id=method_id, field=field))
    if len(strings) != len(set(strings)):
        errors.append(
            error_record("duplicate-string-list-item", method_id=method_id, field=field)
        )
    if any(not item for item in strings):
        errors.append(
            error_record("blank-string-list-item", method_id=method_id, field=field)
        )
    return strings


def _collect_declared_evidence(
    entry: dict[str, Any],
    *,
    errors: list[dict[str, object]],
    method_id: str,
) -> list[str]:
    declared: list[str] = []
    classification = entry.get("classification_evidence")
    if not isinstance(classification, dict):
        errors.append(
            error_record(
                "classification-evidence-not-object",
                method_id=method_id,
                field="classification_evidence",
            )
        )
    else:
        if set(classification) != set(CLASSIFICATION_EVIDENCE_KEYS):
            errors.append(
                error_record(
                    "classification-evidence-keys-mismatch",
                    method_id=method_id,
                    field="classification_evidence",
                    detail=(
                        f"expected {list(CLASSIFICATION_EVIDENCE_KEYS)!r}, "
                        f"observed {sorted(classification)!r}"
                    ),
                )
            )
        for key in CLASSIFICATION_EVIDENCE_KEYS:
            declared.extend(
                _string_list(
                    classification.get(key),
                    errors=errors,
                    method_id=method_id,
                    field=f"classification_evidence.{key}",
                )
            )

    smoke = entry.get("solver_prediction_smoke")
    if not isinstance(smoke, dict):
        errors.append(
            error_record(
                "solver-prediction-smoke-not-object",
                method_id=method_id,
                field="solver_prediction_smoke",
            )
        )
    else:
        declared.extend(
            _string_list(
                smoke.get("evidence"),
                errors=errors,
                method_id=method_id,
                field="solver_prediction_smoke.evidence",
            )
        )

    strict_runtime = entry.get("strict_runtime_promotion")
    if not isinstance(strict_runtime, dict):
        errors.append(
            error_record(
                "strict-runtime-promotion-not-object",
                method_id=method_id,
                field="strict_runtime_promotion",
            )
        )
    else:
        expected_keys = {"status", "config_id", "evidence"}
        if set(strict_runtime) != expected_keys:
            errors.append(
                error_record(
                    "strict-runtime-promotion-keys-mismatch",
                    method_id=method_id,
                    field="strict_runtime_promotion",
                    detail=(
                        f"expected {sorted(expected_keys)!r}; "
                        f"observed {sorted(strict_runtime)!r}"
                    ),
                )
            )
        declared.extend(
            _string_list(
                strict_runtime.get("evidence"),
                errors=errors,
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                nonempty=False,
            )
        )

    gates = entry.get("blocking_gates")
    if not isinstance(gates, list):
        errors.append(
            error_record(
                "blocking-gates-not-list", method_id=method_id, field="blocking_gates"
            )
        )
    else:
        for index, gate in enumerate(gates):
            if not isinstance(gate, dict):
                errors.append(
                    error_record(
                        "blocking-gate-not-object",
                        method_id=method_id,
                        field=f"blocking_gates[{index}]",
                    )
                )
                continue
            declared.extend(
                _string_list(
                    gate.get("evidence"),
                    errors=errors,
                    method_id=method_id,
                    field=f"blocking_gates[{index}].evidence",
                )
            )
    return declared


def _validate_evidence_paths(
    root: Path,
    declared: Iterable[tuple[str, str]],
    errors: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for method_id, evidence in declared:
        if evidence in manifest:
            continue
        resolved, path_error = resolve_repository_path(root, evidence)
        if path_error is not None or resolved is None:
            errors.append(
                error_record(
                    "invalid-evidence-path",
                    method_id=method_id,
                    evidence=evidence,
                    detail=path_error,
                )
            )
            continue
        if not resolved.is_file():
            errors.append(
                error_record(
                    "evidence-file-missing", method_id=method_id, evidence=evidence
                )
            )
            continue
        manifest[evidence] = {
            "path": evidence,
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return manifest


def _load_method_run(
    root: Path,
    method_id: str,
    evidence: str,
    errors: list[dict[str, object]],
    *,
    kind_prefix: str,
) -> dict[str, Any] | None:
    relative = Path(evidence)
    belongs = (
        len(relative.parts) >= 3
        and relative.parts[0] == "reports"
        and relative.parts[1] == method_id
        and relative.name == "run.json"
    )
    if not belongs:
        errors.append(
            error_record(
                f"{kind_prefix}-evidence-not-method-run-json",
                method_id=method_id,
                evidence=evidence,
            )
        )
        return None
    resolved, path_error = resolve_repository_path(root, evidence)
    if path_error is not None or resolved is None or not resolved.is_file():
        return None
    try:
        return load_json_object(resolved)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(
            error_record(
                f"{kind_prefix}-evidence-unreadable",
                method_id=method_id,
                evidence=evidence,
                detail=f"{type(error).__name__}: {error}",
            )
        )
        return None


def _validate_prediction_run(
    root: Path,
    method_id: str,
    evidence: str,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    run = _load_method_run(
        root,
        method_id,
        evidence,
        errors,
        kind_prefix="solver-prediction",
    )
    if run is None:
        return None

    if run.get("status") != "passed":
        errors.append(
            error_record(
                "solver-prediction-run-not-passed",
                method_id=method_id,
                evidence=evidence,
                detail=f"run status is {run.get('status')!r}",
            )
        )
    runner = run.get("runner")
    if not isinstance(runner, str) or not runner:
        errors.append(
            error_record(
                "solver-prediction-runner-not-declared",
                method_id=method_id,
                evidence=evidence,
            )
        )

    configuration = run.get("configuration")
    no_optimizer_labels = (
        isinstance(configuration, dict)
        and configuration.get("test_outputs_available_to_optimizer") is False
    )
    if not no_optimizer_labels:
        errors.append(
            error_record(
                "solver-prediction-label-exclusion-not-proven",
                method_id=method_id,
                evidence=evidence,
            )
        )

    metrics = run.get("metrics")
    counts = run.get("counts")
    tasks_predicted = None
    if isinstance(metrics, dict):
        tasks_predicted = metrics.get("tasks_predicted")
    if tasks_predicted is None and isinstance(counts, dict):
        tasks_predicted = counts.get("tasks_predicted")
    if not isinstance(tasks_predicted, int) or isinstance(tasks_predicted, bool) or tasks_predicted < 1:
        errors.append(
            error_record(
                "solver-prediction-task-count-invalid",
                method_id=method_id,
                evidence=evidence,
                detail=f"tasks_predicted is {tasks_predicted!r}",
            )
        )

    source = run.get("source")
    revision = source.get("revision") if isinstance(source, dict) else None
    if not isinstance(revision, str) or not revision:
        errors.append(
            error_record(
                "solver-prediction-source-revision-not-declared",
                method_id=method_id,
                evidence=evidence,
            )
        )

    artifacts = run.get("artifacts")
    prediction_name = artifacts.get("predictions") if isinstance(artifacts, dict) else None
    declared_hash = (
        artifacts.get("prediction_sha256") if isinstance(artifacts, dict) else None
    )
    observed_hash = None
    prediction_path = None
    if not isinstance(prediction_name, str) or not prediction_name:
        errors.append(
            error_record(
                "solver-prediction-artifact-not-declared",
                method_id=method_id,
                evidence=evidence,
            )
        )
    else:
        run_path, _ = resolve_repository_path(root, evidence)
        assert run_path is not None
        candidate = (run_path.parent / prediction_name).resolve()
        try:
            candidate.relative_to(run_path.parent.resolve())
        except ValueError:
            errors.append(
                error_record(
                    "solver-prediction-artifact-escapes-run-directory",
                    method_id=method_id,
                    evidence=prediction_name,
                )
            )
        else:
            prediction_path = candidate
            if not candidate.is_file():
                errors.append(
                    error_record(
                        "solver-prediction-artifact-missing",
                        method_id=method_id,
                        evidence=prediction_name,
                    )
                )
            else:
                observed_hash = sha256_file(candidate)
                if not isinstance(declared_hash, str) or declared_hash != observed_hash:
                    errors.append(
                        error_record(
                            "solver-prediction-artifact-hash-mismatch",
                            method_id=method_id,
                            evidence=prediction_name,
                            detail=(
                                f"declared {declared_hash!r}; observed {observed_hash!r}"
                            ),
                        )
                    )

    return {
        "run_id": run.get("run_id"),
        "runner": runner,
        "evidence": evidence,
        "tasks_predicted": tasks_predicted,
        "test_outputs_available_to_optimizer": (
            configuration.get("test_outputs_available_to_optimizer")
            if isinstance(configuration, dict)
            else None
        ),
        "prediction_artifact": (
            prediction_path.relative_to(root).as_posix()
            if prediction_path is not None
            and prediction_path.is_relative_to(root.resolve())
            else prediction_name
        ),
        "prediction_sha256": observed_hash,
        "source_revision": revision,
    }


def _validate_strict_runtime_run(
    root: Path,
    method_id: str,
    expected_config_id: str,
    evidence: str,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Validate one method/config-bound protocol-v1 promotion record fail-closed."""

    resolved, path_error = resolve_repository_path(root, evidence)
    if path_error is not None or resolved is None or not resolved.is_file():
        errors.append(
            error_record(
                "strict-runtime-evidence-missing-or-invalid",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=path_error,
            )
        )
        return None

    try:
        run = load_json_object(resolved)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(
            error_record(
                "strict-runtime-evidence-unreadable",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=f"{type(error).__name__}: {error}",
            )
        )
        return None

    observed_method_id = run.get("method_id")
    observed_config_id = run.get("config_id")
    if (
        observed_method_id in GLOBAL_RUNTIME_CORE_METHOD_IDS
        or observed_config_id in GLOBAL_RUNTIME_CORE_CONFIG_IDS
    ):
        errors.append(
            error_record(
                "strict-runtime-global-core-cannot-promote-method",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=(
                    f"global core identity {observed_method_id!r}/"
                    f"{observed_config_id!r} is infrastructure-only evidence"
                ),
            )
        )
        return None

    relative = Path(evidence)
    belongs = (
        len(relative.parts) >= 3
        and relative.parts[0] == "reports"
        and relative.parts[1] == method_id
        and relative.name == "run.json"
    )
    if not belongs:
        errors.append(
            error_record(
                "strict-runtime-evidence-not-method-run-json",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
            )
        )
        return None

    try:
        validation = validate_run_file(
            resolved,
            schema_path=DEFAULT_SCHEMA_PATH,
            repo_root=root,
            verify_files=True,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(
            error_record(
                "strict-runtime-protocol-v1-validation-failed",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=f"{type(error).__name__}: {error}",
            )
        )
        return None

    mismatch = False
    if observed_method_id != method_id:
        mismatch = True
        errors.append(
            error_record(
                "strict-runtime-method-id-mismatch",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=f"expected {method_id!r}; observed {observed_method_id!r}",
            )
        )
    if observed_config_id != expected_config_id:
        mismatch = True
        errors.append(
            error_record(
                "strict-runtime-config-id-mismatch",
                method_id=method_id,
                field="strict_runtime_promotion.config_id",
                evidence=evidence,
                detail=(
                    f"expected {expected_config_id!r}; "
                    f"observed {observed_config_id!r}"
                ),
            )
        )
    if run.get("status") != "passed":
        mismatch = True
        errors.append(
            error_record(
                "strict-runtime-run-not-passed",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=f"run status is {run.get('status')!r}",
            )
        )
    if run.get("evidence_scope") not in PREDICTION_SCOPES:
        mismatch = True
        errors.append(
            error_record(
                "strict-runtime-run-not-prediction-scope",
                method_id=method_id,
                field="strict_runtime_promotion.evidence",
                evidence=evidence,
                detail=f"evidence scope is {run.get('evidence_scope')!r}",
            )
        )
    if mismatch:
        return None

    firewall = run["challenge_firewall"]
    return {
        "run_id": run["run_id"],
        "method_id": observed_method_id,
        "config_id": observed_config_id,
        "evidence": evidence,
        "evidence_scope": run["evidence_scope"],
        "challenge_manifest_digest_sha256": firewall[
            "challenge_manifest_digest_sha256"
        ],
        "inference_received_test_labels": firewall[
            "inference_received_test_labels"
        ],
        "label_mutation_check": firewall["label_mutation_check"],
        "scoring_after_inference": firewall["scoring_after_inference"],
        "record_sha256": validation.record_sha256,
        "schema_sha256": validation.schema_sha256,
        "declared_file_count": validation.declared_file_count,
        "verified_file_count": validation.verified_file_count,
    }


def _zero_filled_counts(values: Iterable[object], enum_values: tuple[str, ...]) -> dict[str, int]:
    counter = Counter(value for value in values if isinstance(value, str))
    return {value: counter.get(value, 0) for value in enum_values}


def _validate_enum_manifest(
    inventory: dict[str, Any], errors: list[dict[str, object]]
) -> None:
    declared = inventory.get("enums")
    if not isinstance(declared, dict):
        errors.append(error_record("enums-not-object", field="enums"))
        return
    expected = {**CATEGORY_ENUMS, **OTHER_ENUMS}
    for key, values in expected.items():
        observed = declared.get(key)
        if observed != list(values):
            errors.append(
                error_record(
                    "enum-definition-mismatch",
                    field=f"enums.{key}",
                    detail=f"expected {list(values)!r}; observed {observed!r}",
                )
            )


def audit_inventory(
    root: Path,
    inventory_path: Path,
    baseline_path: Path,
) -> dict[str, object]:
    """Return a complete, non-mutating audit of one eligibility inventory."""

    started = utc_now()
    root = root.resolve()
    inventory_path = inventory_path if inventory_path.is_absolute() else root / inventory_path
    baseline_path = baseline_path if baseline_path.is_absolute() else root / baseline_path
    inventory = load_json_object(inventory_path)
    baseline = load_json_object(baseline_path)
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    _validate_enum_manifest(inventory, errors)

    baseline_entries_raw = baseline.get("entries")
    inventory_entries_raw = inventory.get("entries")
    if not isinstance(baseline_entries_raw, list):
        raise ValueError("baseline entries is not a list")
    if not isinstance(inventory_entries_raw, list):
        raise ValueError("inventory entries is not a list")
    baseline_entries = [entry for entry in baseline_entries_raw if isinstance(entry, dict)]
    inventory_entries = [entry for entry in inventory_entries_raw if isinstance(entry, dict)]
    if len(baseline_entries) != len(baseline_entries_raw):
        errors.append(error_record("baseline-entry-not-object", field="entries"))
    if len(inventory_entries) != len(inventory_entries_raw):
        errors.append(error_record("inventory-entry-not-object", field="entries"))

    baseline_ids = [entry.get("id") for entry in baseline_entries]
    inventory_ids = [entry.get("id") for entry in inventory_entries]
    for label, ids in (("baseline", baseline_ids), ("inventory", inventory_ids)):
        if any(not isinstance(method_id, str) or not method_id for method_id in ids):
            errors.append(error_record(f"{label}-method-id-invalid", field="entries.id"))
        duplicates = sorted(
            method_id
            for method_id, count in Counter(ids).items()
            if isinstance(method_id, str) and count > 1
        )
        if duplicates:
            errors.append(
                error_record(
                    f"{label}-method-id-duplicate",
                    field="entries.id",
                    detail=f"duplicates: {duplicates!r}",
                )
            )

    baseline_id_set = {method_id for method_id in baseline_ids if isinstance(method_id, str)}
    inventory_id_set = {method_id for method_id in inventory_ids if isinstance(method_id, str)}
    if baseline_id_set != inventory_id_set:
        errors.append(
            error_record(
                "baseline-inventory-id-set-mismatch",
                field="entries.id",
                detail=(
                    f"missing={sorted(baseline_id_set - inventory_id_set)!r}; "
                    f"extra={sorted(inventory_id_set - baseline_id_set)!r}"
                ),
            )
        )
    if baseline_ids != inventory_ids:
        errors.append(
            error_record(
                "baseline-inventory-order-mismatch",
                field="entries.id",
                detail="Inventory rows must retain baseline manifest order.",
            )
        )

    baseline_manifest_declared = inventory.get("baseline_manifest")
    try:
        baseline_relative = baseline_path.resolve().relative_to(root).as_posix()
    except ValueError:
        baseline_relative = str(baseline_path.resolve())
    if baseline_manifest_declared != baseline_relative:
        errors.append(
            error_record(
                "baseline-manifest-path-mismatch",
                field="baseline_manifest",
                detail=(
                    f"declared {baseline_manifest_declared!r}; audited {baseline_relative!r}"
                ),
            )
        )

    baseline_by_id = {
        entry["id"]: entry
        for entry in baseline_entries
        if isinstance(entry.get("id"), str)
    }
    all_declared_evidence: list[tuple[str, str]] = []
    method_audits: list[dict[str, object]] = []
    prediction_audits: dict[str, dict[str, object]] = {}
    strict_runtime_audits: dict[str, dict[str, object]] = {}

    for index, entry in enumerate(inventory_entries):
        method_id_value = entry.get("id")
        method_id = method_id_value if isinstance(method_id_value, str) else f"<row-{index}>"
        baseline_entry = baseline_by_id.get(method_id)
        if baseline_entry is not None and entry.get("name") != baseline_entry.get("name"):
            errors.append(
                error_record(
                    "baseline-inventory-name-mismatch",
                    method_id=method_id,
                    field="name",
                    detail=(
                        f"baseline {baseline_entry.get('name')!r}; "
                        f"inventory {entry.get('name')!r}"
                    ),
                )
            )

        for field, values in CATEGORY_ENUMS.items():
            if entry.get(field) not in values:
                errors.append(
                    error_record(
                        "invalid-category-value",
                        method_id=method_id,
                        field=field,
                        evidence=entry.get(field),
                    )
                )

        detail = entry.get("evidence_scope_detail")
        if not isinstance(detail, str) or not detail.strip():
            errors.append(
                error_record(
                    "evidence-scope-detail-missing",
                    method_id=method_id,
                    field="evidence_scope_detail",
                )
            )

        additional_risks = _string_list(
            entry.get("additional_code_trust_risks"),
            errors=errors,
            method_id=method_id,
            field="additional_code_trust_risks",
            nonempty=False,
        )
        for risk in additional_risks:
            if risk not in RISK_CLASSES:
                errors.append(
                    error_record(
                        "invalid-additional-code-trust-risk",
                        method_id=method_id,
                        field="additional_code_trust_risks",
                        evidence=risk,
                    )
                )
            if risk == entry.get("code_trust_class"):
                errors.append(
                    error_record(
                        "duplicate-primary-code-trust-risk",
                        method_id=method_id,
                        field="additional_code_trust_risks",
                        evidence=risk,
                    )
                )

        smoke = entry.get("solver_prediction_smoke")
        smoke_status = smoke.get("status") if isinstance(smoke, dict) else None
        if smoke_status not in OTHER_ENUMS["solver_prediction_smoke_status"]:
            errors.append(
                error_record(
                    "invalid-solver-prediction-smoke-status",
                    method_id=method_id,
                    field="solver_prediction_smoke.status",
                    evidence=smoke_status,
                )
            )

        strict_runtime = entry.get("strict_runtime_promotion")
        strict_status = (
            strict_runtime.get("status") if isinstance(strict_runtime, dict) else None
        )
        strict_config_id = (
            strict_runtime.get("config_id")
            if isinstance(strict_runtime, dict)
            else None
        )
        strict_evidence = (
            strict_runtime.get("evidence")
            if isinstance(strict_runtime, dict)
            and isinstance(strict_runtime.get("evidence"), list)
            else []
        )
        if strict_status not in OTHER_ENUMS["strict_runtime_promotion_status"]:
            errors.append(
                error_record(
                    "invalid-strict-runtime-promotion-status",
                    method_id=method_id,
                    field="strict_runtime_promotion.status",
                    evidence=strict_status,
                )
            )
        if strict_config_id is not None and (
            not isinstance(strict_config_id, str) or not strict_config_id.strip()
        ):
            errors.append(
                error_record(
                    "invalid-strict-runtime-config-id",
                    method_id=method_id,
                    field="strict_runtime_promotion.config_id",
                    evidence=strict_config_id,
                )
            )
        if strict_status == "not_run":
            if strict_config_id is not None:
                errors.append(
                    error_record(
                        "not-run-strict-runtime-has-config-id",
                        method_id=method_id,
                        field="strict_runtime_promotion.config_id",
                    )
                )
            if strict_evidence:
                errors.append(
                    error_record(
                        "not-run-strict-runtime-has-evidence",
                        method_id=method_id,
                        field="strict_runtime_promotion.evidence",
                    )
                )

        declared_evidence = _collect_declared_evidence(
            entry, errors=errors, method_id=method_id
        )
        all_declared_evidence.extend((method_id, path) for path in declared_evidence)

        gates = entry.get("blocking_gates")
        gate_names: list[str] = []
        if isinstance(gates, list):
            for gate_index, gate in enumerate(gates):
                if not isinstance(gate, dict):
                    continue
                gate_name = gate.get("gate")
                gate_status = gate.get("status")
                gate_names.append(gate_name) if isinstance(gate_name, str) else None
                if gate_name not in OTHER_ENUMS["blocking_gate"]:
                    errors.append(
                        error_record(
                            "invalid-blocking-gate",
                            method_id=method_id,
                            field=f"blocking_gates[{gate_index}].gate",
                            evidence=gate_name,
                        )
                    )
                if gate_status not in OTHER_ENUMS["blocking_gate_status"]:
                    errors.append(
                        error_record(
                            "invalid-blocking-gate-status",
                            method_id=method_id,
                            field=f"blocking_gates[{gate_index}].status",
                            evidence=gate_status,
                        )
                    )
                gate_detail = gate.get("detail")
                if not isinstance(gate_detail, str) or not gate_detail.strip():
                    errors.append(
                        error_record(
                            "blocking-gate-detail-missing",
                            method_id=method_id,
                            field=f"blocking_gates[{gate_index}].detail",
                        )
                    )
            if len(gate_names) != len(set(gate_names)):
                errors.append(
                    error_record(
                        "duplicate-blocking-gate", method_id=method_id, field="blocking_gates"
                    )
                )

        eligible = entry.get("performance_table_eligible")
        if not isinstance(eligible, bool):
            errors.append(
                error_record(
                    "performance-eligibility-not-boolean",
                    method_id=method_id,
                    field="performance_table_eligible",
                )
            )
        if eligible is False and (not isinstance(gates, list) or not gates):
            errors.append(
                error_record(
                    "ineligible-row-has-no-blocking-gate",
                    method_id=method_id,
                    field="blocking_gates",
                )
            )
        if eligible is True:
            if isinstance(gates, list) and gates:
                errors.append(
                    error_record(
                        "eligible-row-still-has-blocking-gates",
                        method_id=method_id,
                        field="blocking_gates",
                    )
                )
            if smoke_status != "passed":
                errors.append(
                    error_record(
                        "eligible-row-lacks-passed-solver-prediction",
                        method_id=method_id,
                        field="solver_prediction_smoke.status",
                    )
                )
            if strict_status != "passed":
                errors.append(
                    error_record(
                        "eligible-row-lacks-passed-strict-runtime",
                        method_id=method_id,
                        field="strict_runtime_promotion.status",
                    )
                )

        scope = entry.get("evidence_scope")
        parity = entry.get("parity_class")
        expected_scope_states: dict[str, tuple[set[str], set[str]]] = {
            "unavailable": ({"unavailable"}, {"unavailable"}),
            "blocker_audit": ({"blocked_before_method_execution"}, {"blocked"}),
            "scorer_only": ({"scorer_only"}, {"not_run"}),
            "component": ({"component_only"}, {"not_run"}),
            "solver_prediction": (
                {"reduced_method_execution", "native_method_execution"},
                {"passed"},
            ),
            "benchmark": ({"benchmark_parity", "paper_parity"}, {"passed"}),
        }
        if scope in expected_scope_states:
            allowed_parity, allowed_smoke = expected_scope_states[scope]
            if parity not in allowed_parity:
                errors.append(
                    error_record(
                        "evidence-scope-parity-mismatch",
                        method_id=method_id,
                        field="parity_class",
                        detail=f"scope {scope!r} does not allow parity {parity!r}",
                    )
                )
            if smoke_status not in allowed_smoke:
                errors.append(
                    error_record(
                        "evidence-scope-smoke-status-mismatch",
                        method_id=method_id,
                        field="solver_prediction_smoke.status",
                        detail=f"scope {scope!r} does not allow status {smoke_status!r}",
                    )
                )

        unavailable_fields = (
            scope == "unavailable",
            parity == "unavailable",
            entry.get("resource_class") == "unavailable",
            entry.get("code_trust_class") == "unavailable",
            smoke_status == "unavailable",
        )
        if any(unavailable_fields) and not all(unavailable_fields):
            errors.append(
                error_record(
                    "unavailable-state-inconsistent",
                    method_id=method_id,
                    detail="scope, parity, resource, trust, and prediction status must agree",
                )
            )
        if entry.get("code_trust_class") == "unavailable" and additional_risks:
            errors.append(
                error_record(
                    "unavailable-row-has-additional-trust-risks",
                    method_id=method_id,
                    field="additional_code_trust_risks",
                )
            )

        smoke_evidence = (
            smoke.get("evidence")
            if isinstance(smoke, dict) and isinstance(smoke.get("evidence"), list)
            else []
        )
        if smoke_status == "passed":
            if len(smoke_evidence) != 1:
                errors.append(
                    error_record(
                        "passed-solver-prediction-requires-one-primary-run",
                        method_id=method_id,
                        field="solver_prediction_smoke.evidence",
                    )
                )
            elif isinstance(smoke_evidence[0], str):
                prediction_audit = _validate_prediction_run(
                    root, method_id, smoke_evidence[0], errors
                )
                if prediction_audit is not None:
                    prediction_audits[method_id] = prediction_audit
        elif scope in {"component", "scorer_only"}:
            passing_run_found = False
            for evidence in smoke_evidence:
                if not isinstance(evidence, str):
                    continue
                run = _load_method_run(
                    root,
                    method_id,
                    evidence,
                    errors,
                    kind_prefix=scope.replace("_", "-"),
                )
                if run is not None and run.get("status") == "passed":
                    passing_run_found = True
            if not passing_run_found:
                errors.append(
                    error_record(
                        f"{scope.replace('_', '-')}-passing-evidence-not-found",
                        method_id=method_id,
                        field="solver_prediction_smoke.evidence",
                    )
                )

        if strict_status == "passed":
            if smoke_status != "passed":
                errors.append(
                    error_record(
                        "strict-runtime-promotion-lacks-passed-solver-smoke",
                        method_id=method_id,
                        field="solver_prediction_smoke.status",
                    )
                )
            if scope not in {"solver_prediction", "benchmark"}:
                errors.append(
                    error_record(
                        "strict-runtime-promotion-scope-mismatch",
                        method_id=method_id,
                        field="evidence_scope",
                        detail=f"evidence scope is {scope!r}",
                    )
                )
            if not isinstance(strict_config_id, str) or not strict_config_id.strip():
                errors.append(
                    error_record(
                        "passed-strict-runtime-requires-config-id",
                        method_id=method_id,
                        field="strict_runtime_promotion.config_id",
                    )
                )
            if len(strict_evidence) != 1:
                errors.append(
                    error_record(
                        "passed-strict-runtime-requires-one-primary-run",
                        method_id=method_id,
                        field="strict_runtime_promotion.evidence",
                    )
                )
            elif isinstance(strict_config_id, str) and strict_config_id.strip():
                strict_audit = _validate_strict_runtime_run(
                    root,
                    method_id,
                    strict_config_id,
                    strict_evidence[0],
                    errors,
                )
                if strict_audit is not None:
                    strict_runtime_audits[method_id] = strict_audit

        if scope in {"component", "scorer_only", "blocker_audit", "unavailable"} and eligible is True:
            errors.append(
                error_record(
                    "non-solver-evidence-promoted-to-performance-table",
                    method_id=method_id,
                    field="performance_table_eligible",
                    detail=f"evidence scope is {scope!r}",
                )
            )

        method_audits.append(
            {
                "id": method_id,
                "native_benchmark_generation": entry.get("native_benchmark_generation"),
                "cohort": entry.get("cohort"),
                "primary_mechanism_family": entry.get("primary_mechanism_family"),
                "evidence_scope": scope,
                "parity_class": parity,
                "resource_class": entry.get("resource_class"),
                "code_trust_class": entry.get("code_trust_class"),
                "additional_code_trust_risks": additional_risks,
                "solver_prediction_smoke_status": smoke_status,
                "strict_runtime_promotion_status": strict_status,
                "strict_runtime_config_id": strict_config_id,
                "performance_table_eligible": eligible,
                "blocking_gate_count": len(gates) if isinstance(gates, list) else None,
            }
        )

    evidence_manifest = _validate_evidence_paths(
        root, all_declared_evidence, errors
    )

    derived_summary: dict[str, object] = {"entry_count": len(inventory_entries)}
    for field, values in CATEGORY_ENUMS.items():
        derived_summary[field] = _zero_filled_counts(
            (entry.get(field) for entry in inventory_entries), values
        )
    derived_summary["solver_prediction_smoke_status"] = _zero_filled_counts(
        (
            entry.get("solver_prediction_smoke", {}).get("status")
            if isinstance(entry.get("solver_prediction_smoke"), dict)
            else None
            for entry in inventory_entries
        ),
        OTHER_ENUMS["solver_prediction_smoke_status"],
    )
    derived_summary["strict_runtime_promotion_status"] = _zero_filled_counts(
        (
            entry.get("strict_runtime_promotion", {}).get("status")
            if isinstance(entry.get("strict_runtime_promotion"), dict)
            else None
            for entry in inventory_entries
        ),
        OTHER_ENUMS["strict_runtime_promotion_status"],
    )
    eligible_count = sum(
        entry.get("performance_table_eligible") is True for entry in inventory_entries
    )
    derived_summary["performance_table_eligible"] = {
        "eligible": eligible_count,
        "ineligible": len(inventory_entries) - eligible_count,
    }
    declared_summary = inventory.get("summary")
    if declared_summary != derived_summary:
        errors.append(
            error_record(
                "declared-summary-mismatch",
                field="summary",
                detail=(
                    f"declared {declared_summary!r}; derived {derived_summary!r}"
                ),
            )
        )

    baseline_summary = baseline.get("summary")
    baseline_entry_count = (
        baseline_summary.get("entry_count") if isinstance(baseline_summary, dict) else None
    )
    baseline_smoke_passed = (
        baseline_summary.get("smoke_passed_count")
        if isinstance(baseline_summary, dict)
        else None
    )
    baseline_benchmark_passed = (
        baseline_summary.get("benchmark_passed_count")
        if isinstance(baseline_summary, dict)
        else None
    )
    scope_counts = derived_summary["evidence_scope"]
    manifest_smoke_passed = sum(
        baseline_entry.get("reproduction", {})
        .get("smoke", {})
        .get("status")
        == "passed"
        for baseline_entry in baseline_by_id.values()
        if baseline_entry.get("id") in (entry.get("id") for entry in inventory_entries)
    )
    asserted_smoke_like = manifest_smoke_passed
    if baseline_entry_count != len(inventory_entries):
        errors.append(
            error_record(
                "baseline-entry-summary-mismatch",
                field="baseline.summary.entry_count",
                detail=f"baseline {baseline_entry_count!r}; inventory {len(inventory_entries)!r}",
            )
        )
    if baseline_smoke_passed != asserted_smoke_like:
        errors.append(
            error_record(
                "baseline-smoke-count-mismatch",
                field="baseline.summary.smoke_passed_count",
                detail=(
                    f"baseline {baseline_smoke_passed!r}; "
                    f"manifest smoke-passed {asserted_smoke_like!r}"
                ),
            )
        )
    if baseline_benchmark_passed != eligible_count:
        errors.append(
            error_record(
                "baseline-benchmark-count-mismatch",
                field="baseline.summary.benchmark_passed_count",
                detail=f"baseline {baseline_benchmark_passed!r}; eligible {eligible_count!r}",
            )
        )

    try:
        inventory_relative = inventory_path.resolve().relative_to(root).as_posix()
    except ValueError:
        inventory_relative = str(inventory_path.resolve())
    report: dict[str, object] = {
        "schema_version": 1,
        "runner": "scripts.audit_method_eligibility",
        "runner_version": 2,
        "scope": "configuration_eligibility_trust_and_evidence_audit",
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "status": "passed" if not errors else "failed",
        "inputs": {
            "inventory": {
                "path": inventory_relative,
                "sha256": sha256_file(inventory_path),
            },
            "baseline_manifest": {
                "path": baseline_relative,
                "sha256": sha256_file(baseline_path),
            },
        },
        "summary": {
            **derived_summary,
            "baseline_smoke_passed_count": baseline_smoke_passed,
            "component_plus_solver_prediction_count": asserted_smoke_like,
            "solver_prediction_artifact_validated_count": len(prediction_audits),
            "strict_runtime_artifact_validated_count": len(strict_runtime_audits),
            "unique_evidence_file_count": len(evidence_manifest),
        },
        "methods": method_audits,
        "solver_prediction_evidence": prediction_audits,
        "strict_runtime_promotion_evidence": strict_runtime_audits,
        "evidence_manifest": [evidence_manifest[key] for key in sorted(evidence_manifest)],
        "validation": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "exact_baseline_id_set": baseline_id_set == inventory_id_set,
            "baseline_order_preserved": baseline_ids == inventory_ids,
            "component_scope_cannot_self_promote": True,
            "global_runtime_core_cannot_promote_method": True,
            "strict_runtime_requires_protocol_v1_file_validation": True,
            "all_declared_evidence_paths_exist": not any(
                error["kind"] in {"invalid-evidence-path", "evidence-file-missing"}
                for error in errors
            ),
        },
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("configs/method_eligibility.json"),
    )
    parser.add_argument(
        "--baselines", type=Path, default=Path("configs/baselines.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Create an immutable run.json audit record; existing files are refused.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_inventory(ROOT, args.inventory, args.baselines)
        if args.output is not None:
            output = args.output if args.output.is_absolute() else ROOT / args.output
            report["run_id"] = output.parent.name
            report["command"] = [
                "scripts/audit_method_eligibility.py",
                "--inventory",
                args.inventory.as_posix(),
                "--baselines",
                args.baselines.as_posix(),
                "--output",
                args.output.as_posix(),
            ]
            immutable_json(output, report)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"audit_method_eligibility: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": report["status"],
                "error_count": report["validation"]["error_count"],
                "summary": report["summary"],
                "output": args.output.as_posix() if args.output is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
