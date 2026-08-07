#!/usr/bin/env python3
"""Audit the manifest-declared reproduction funnel and its passed evidence."""

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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.resources import ResourceMonitor


REPRODUCTION_LEVELS = (
    ("smoke", "smoke"),
    ("public_benchmark", "benchmark"),
    ("full_reproduction", "full"),
)
SUMMARY_COUNT_FIELDS = {
    "entry_count": None,
    "public_candidate_count": ("availability", "public_candidate"),
    "partial_complex_count": ("availability", "partial_complex"),
    "unavailable_blocked_count": ("availability", "unavailable_blocked"),
    "smoke_passed_count": ("reproduction", "smoke"),
    "benchmark_passed_count": ("reproduction", "benchmark"),
    "full_reproduction_passed_count": ("reproduction", "full"),
}

PRIMARY_SMOKE_EXCLUSION_ERROR = "passed-smoke-evidence-explicitly-excluded"
BLOCKER_AUDIT_SCOPE = "source-dependency-label-artifact-gate-audit-only"


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


def atomic_json(path: Path, value: dict[str, object]) -> None:
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
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def error_record(
    kind: str,
    *,
    method_id: str | None = None,
    layer: str | None = None,
    evidence: object = None,
    detail: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"kind": kind}
    if method_id is not None:
        result["method_id"] = method_id
    if layer is not None:
        result["layer"] = layer
    if evidence is not None:
        result["evidence"] = evidence
    if detail is not None:
        result["detail"] = detail
    return result


def resolve_evidence(root: Path, declared: str) -> tuple[Path | None, str | None]:
    relative = Path(declared)
    if relative.is_absolute():
        return None, "evidence path must be repository-relative"
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, "evidence path escapes the repository root"
    return resolved, None


def validate_passed_evidence(
    root: Path,
    method_id: str,
    layer: str,
    state: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate evidence only when one manifest state declares ``passed``."""

    status = state.get("status")
    declared = state.get("evidence")
    validation: dict[str, object] = {
        "required": status == "passed",
        "declared_path": declared,
        "checked": False,
        "valid": None,
    }
    if status != "passed":
        validation["result"] = "not-required-for-non-passed-state"
        return validation, []

    errors: list[dict[str, object]] = []
    validation["checked"] = True
    if not isinstance(declared, str) or not declared:
        errors.append(
            error_record(
                "passed-evidence-not-declared",
                method_id=method_id,
                layer=layer,
                evidence=declared,
            )
        )
        validation["valid"] = False
        return validation, errors

    path, path_error = resolve_evidence(root, declared)
    if path_error is not None or path is None:
        errors.append(
            error_record(
                "invalid-evidence-path",
                method_id=method_id,
                layer=layer,
                evidence=declared,
                detail=path_error,
            )
        )
        validation["valid"] = False
        return validation, errors

    validation["exists"] = path.is_file()
    validation["is_run_json"] = path.name == "run.json"
    relative_parts = Path(declared).parts
    validation["belongs_to_method_report_directory"] = (
        len(relative_parts) >= 3
        and relative_parts[0] == "reports"
        and relative_parts[1] == method_id
    )
    if not path.is_file():
        errors.append(
            error_record(
                "passed-evidence-missing",
                method_id=method_id,
                layer=layer,
                evidence=declared,
            )
        )
    if path.name != "run.json":
        errors.append(
            error_record(
                "passed-evidence-is-not-run-json",
                method_id=method_id,
                layer=layer,
                evidence=declared,
            )
        )
    if not validation["belongs_to_method_report_directory"]:
        errors.append(
            error_record(
                "passed-evidence-outside-method-report-directory",
                method_id=method_id,
                layer=layer,
                evidence=declared,
            )
        )
    if errors:
        validation["valid"] = False
        return validation, errors

    try:
        run = load_json_object(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(
            error_record(
                "passed-evidence-unreadable",
                method_id=method_id,
                layer=layer,
                evidence=declared,
                detail=f"{type(error).__name__}: {error}",
            )
        )
        validation["valid"] = False
        return validation, errors

    run_status = run.get("status")
    run_method_id = run.get("method_id")
    run_counted_toward_smoke = run.get("counted_toward_smoke")
    run_method_gate_status = run.get("method_gate_status")
    fairness = run.get("fairness")
    run_evidence_scope = run.get("evidence_scope")
    if run_evidence_scope is None and isinstance(fairness, dict):
        run_evidence_scope = fairness.get("evidence_scope")
    run_fair_eligibility = (
        fairness.get("score_eligible_for_fair_main_board")
        if isinstance(fairness, dict)
        else None
    )
    validation.update(
        {
            "sha256": sha256_file(path),
            "run_status": run_status,
            "run_id": run.get("run_id"),
            "runner": run.get("runner"),
            "run_scope": run.get("scope"),
            "run_counted_toward_smoke": run_counted_toward_smoke,
            "run_method_gate_status": run_method_gate_status,
            "run_evidence_scope": run_evidence_scope,
            "run_score_eligible_for_fair_main_board": run_fair_eligibility,
            "run_method_id": run_method_id,
            "run_method_id_check": (
                "not-declared"
                if run_method_id is None
                else "matched"
                if run_method_id == method_id
                else "mismatched"
            ),
        }
    )
    if run_status != "passed":
        errors.append(
            error_record(
                "passed-evidence-run-status-mismatch",
                method_id=method_id,
                layer=layer,
                evidence=declared,
                detail=f"run.json status is {run_status!r}",
            )
        )
    if run_method_id is not None and run_method_id != method_id:
        errors.append(
            error_record(
                "passed-evidence-method-mismatch",
                method_id=method_id,
                layer=layer,
                evidence=declared,
                detail=f"run.json method_id is {run_method_id!r}",
            )
        )
    if layer == "smoke":
        exclusion_reasons: list[str] = []
        if run_counted_toward_smoke is False:
            exclusion_reasons.append("counted_toward_smoke=false")
        if run.get("scope") == BLOCKER_AUDIT_SCOPE:
            exclusion_reasons.append(f"scope={BLOCKER_AUDIT_SCOPE}")
        if (
            run_method_gate_status == "blocked"
            and run_evidence_scope == "blocker_audit"
        ):
            exclusion_reasons.append(
                "method_gate_status=blocked,evidence_scope=blocker_audit"
            )
        validation["primary_smoke_exclusion_reasons"] = exclusion_reasons
        if exclusion_reasons:
            errors.append(
                error_record(
                    PRIMARY_SMOKE_EXCLUSION_ERROR,
                    method_id=method_id,
                    layer=layer,
                    evidence=declared,
                    detail=(
                        "run.json cannot support a primary passing smoke: "
                        + "; ".join(exclusion_reasons)
                    ),
                )
            )
        validation["primary_smoke_accepted"] = not errors
    validation["valid"] = not errors
    return validation, errors


def passed_evidence_metadata_warnings(
    method_id: str,
    layer: str,
    validation: dict[str, object],
) -> list[dict[str, object]]:
    """Report legacy attribution gaps without invalidating otherwise valid evidence."""

    if validation.get("valid") is not True:
        return []
    checks = (
        ("run_method_id", "passed-evidence-method-id-not-declared"),
        ("runner", "passed-evidence-runner-not-declared"),
        ("run_scope", "passed-evidence-scope-not-declared"),
    )
    return [
        error_record(
            kind,
            method_id=method_id,
            layer=layer,
            evidence=validation.get("declared_path"),
            detail=(
                "Legacy run remains attributable through its manifest entry and "
                "reports/<method-id> path; the immutable record was not rewritten."
            ),
        )
        for field, kind in checks
        if validation.get(field) is None
    ]


def source_audit_layer(
    root: Path,
    method_id: str,
    allowed_statuses: set[str],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Discover source audits by exact runner identity, never by pass status."""

    report_root = root / "reports" / method_id
    attempts: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    if report_root.is_dir():
        for path in sorted(report_root.rglob("run.json")):
            try:
                run = load_json_object(path)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                warnings.append(
                    {
                        "kind": "unreadable-run-record-ignored-during-source-discovery",
                        "method_id": method_id,
                        "evidence": path.relative_to(root).as_posix(),
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if run.get("runner") != "scripts.audit_source":
                continue
            evidence = path.relative_to(root).as_posix()
            status = run.get("status")
            attempt = {
                "evidence": evidence,
                "sha256": sha256_file(path),
                "status": status,
                "run_id": run.get("run_id"),
                "started_at_utc": run.get("started_at_utc"),
                "ended_at_utc": run.get("ended_at_utc"),
                "method_id": run.get("method_id"),
                "observed_revision": (
                    run.get("source", {}).get("observed_revision")
                    if isinstance(run.get("source"), dict)
                    else None
                ),
            }
            attempts.append(attempt)
            if status not in allowed_statuses:
                errors.append(
                    error_record(
                        "source-audit-invalid-status",
                        method_id=method_id,
                        layer="source_audit",
                        evidence=evidence,
                        detail=f"status is {status!r}",
                    )
                )
            if run.get("method_id") != method_id:
                errors.append(
                    error_record(
                        "source-audit-method-mismatch",
                        method_id=method_id,
                        layer="source_audit",
                        evidence=evidence,
                        detail=f"run.json method_id is {run.get('method_id')!r}",
                    )
                )

    attempts.sort(
        key=lambda item: (
            str(item.get("ended_at_utc") or item.get("started_at_utc") or ""),
            str(item["evidence"]),
        )
    )
    selected = attempts[-1] if attempts else None
    return (
        {
            "authority": "reports/<method-id>/**/run.json with runner=scripts.audit_source",
            "status": "not_started" if selected is None else selected["status"],
            "evidence": None if selected is None else selected["evidence"],
            "evidence_paths": [attempt["evidence"] for attempt in attempts],
            "attempt_count": len(attempts),
            "attempts": attempts,
            "selection_rule": "latest ended_at_utc, then lexicographically greatest evidence path",
            "counted_toward_smoke": False,
        },
        errors,
        warnings,
    )


def status_counts(statuses: list[object], allowed: list[str]) -> dict[str, int]:
    observed = Counter(status for status in statuses if isinstance(status, str))
    return {status: observed.get(status, 0) for status in allowed}


def expected_summary_value(
    field: str, specification: tuple[str, str] | None, entries: list[dict[str, Any]]
) -> int:
    if field == "entry_count":
        return len(entries)
    if specification is None:
        raise AssertionError(field)
    category, value = specification
    if category == "availability":
        return sum(entry.get("availability") == value for entry in entries)
    return sum(
        entry.get("reproduction", {}).get(value, {}).get("status") == "passed"
        for entry in entries
    )


def audit_funnel(root: Path, manifest_path: Path) -> dict[str, object]:
    """Build a deterministic funnel audit without timestamps or output writes."""

    root = root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_json_object(manifest_path)
    entries_value = manifest.get("entries")
    if not isinstance(entries_value, list) or not all(
        isinstance(entry, dict) for entry in entries_value
    ):
        raise ValueError("manifest entries must be a list of objects")
    entries: list[dict[str, Any]] = entries_value
    enums = manifest.get("enums")
    if not isinstance(enums, dict) or not isinstance(
        enums.get("execution_status"), list
    ):
        raise ValueError("manifest must declare enums.execution_status")
    allowed_status_list = list(enums["execution_status"])
    if not all(isinstance(status, str) for status in allowed_status_list):
        raise ValueError("execution status enum must contain strings")
    allowed_statuses = set(allowed_status_list)

    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    ids = [entry.get("id") for entry in entries]
    if not all(isinstance(method_id, str) and method_id for method_id in ids):
        errors.append(error_record("invalid-or-missing-method-id"))
    duplicate_ids = sorted(
        method_id
        for method_id, count in Counter(ids).items()
        if isinstance(method_id, str) and count > 1
    )
    if duplicate_ids:
        errors.append(
            error_record("duplicate-method-ids", detail=", ".join(duplicate_ids))
        )

    methods: list[dict[str, object]] = []
    reproduction_passed_claims = 0
    valid_reproduction_passed_claims = 0
    valid_reproduction_passed_by_layer: Counter[str] = Counter()
    auxiliary_passed_claims = 0
    valid_auxiliary_passed_claims = 0

    for entry in entries:
        method_id_value = entry.get("id")
        method_id = method_id_value if isinstance(method_id_value, str) else "<invalid>"
        reproduction = entry.get("reproduction")
        if not isinstance(reproduction, dict):
            reproduction = {}
            errors.append(
                error_record("missing-reproduction-object", method_id=method_id)
            )

        source_layer, source_errors, source_warnings = source_audit_layer(
            root, method_id, allowed_statuses
        )
        errors.extend(source_errors)
        warnings.extend(source_warnings)
        layers: dict[str, object] = {"source_audit": source_layer}

        for output_name, manifest_name in REPRODUCTION_LEVELS:
            state_value = reproduction.get(manifest_name)
            if not isinstance(state_value, dict):
                state: dict[str, Any] = {"status": None}
                errors.append(
                    error_record(
                        "missing-reproduction-level",
                        method_id=method_id,
                        layer=output_name,
                    )
                )
            else:
                state = dict(state_value)
            status = state.get("status")
            if status not in allowed_statuses:
                errors.append(
                    error_record(
                        "invalid-reproduction-status",
                        method_id=method_id,
                        layer=output_name,
                        detail=f"status is {status!r}",
                    )
                )
            evidence_validation, evidence_errors = validate_passed_evidence(
                root, method_id, output_name, state
            )
            errors.extend(evidence_errors)
            warnings.extend(
                passed_evidence_metadata_warnings(
                    method_id, output_name, evidence_validation
                )
            )
            if status == "passed":
                reproduction_passed_claims += 1
                valid_reproduction_passed_claims += evidence_validation["valid"] is True
                if evidence_validation["valid"] is True:
                    valid_reproduction_passed_by_layer[output_name] += 1
            layers[output_name] = {
                "authority": f"configs/baselines.json:reproduction.{manifest_name}",
                "status": status,
                "feasibility": state.get("feasibility"),
                "scope": state.get("scope"),
                "evidence": state.get("evidence"),
                "evidence_paths": (
                    [state["evidence"]]
                    if isinstance(state.get("evidence"), str)
                    else []
                ),
                "evidence_validation": evidence_validation,
            }

        auxiliary_value = entry.get("auxiliary_evidence", [])
        if not isinstance(auxiliary_value, list) or not all(
            isinstance(item, dict) for item in auxiliary_value
        ):
            auxiliary: list[dict[str, Any]] = []
            errors.append(
                error_record("invalid-auxiliary-evidence-list", method_id=method_id)
            )
        else:
            auxiliary = auxiliary_value
        auxiliary_items: list[dict[str, object]] = []
        for index, item in enumerate(auxiliary):
            status = item.get("status")
            layer_name = f"auxiliary_evidence[{index}]"
            if status not in allowed_statuses:
                errors.append(
                    error_record(
                        "invalid-auxiliary-status",
                        method_id=method_id,
                        layer=layer_name,
                        detail=f"status is {status!r}",
                    )
                )
            validation, item_errors = validate_passed_evidence(
                root, method_id, layer_name, item
            )
            errors.extend(item_errors)
            if "score_eligible_for_fair_main_board" in item:
                declared_eligibility = item[
                    "score_eligible_for_fair_main_board"
                ]
                observed_eligibility = validation.get(
                    "run_score_eligible_for_fair_main_board"
                )
                validation["fair_main_board_eligibility_matches"] = (
                    declared_eligibility == observed_eligibility
                )
                if declared_eligibility != observed_eligibility:
                    validation["valid"] = False
                    errors.append(
                        error_record(
                            "auxiliary-fair-eligibility-mismatch",
                            method_id=method_id,
                            layer=layer_name,
                            evidence=item.get("evidence"),
                            detail=(
                                f"manifest declares {declared_eligibility!r}; "
                                f"run fairness declares {observed_eligibility!r}"
                            ),
                        )
                    )
            warnings.extend(
                passed_evidence_metadata_warnings(
                    method_id, layer_name, validation
                )
            )
            if status == "passed":
                auxiliary_passed_claims += 1
                valid_auxiliary_passed_claims += validation["valid"] is True
            auxiliary_items.append(
                {
                    **item,
                    "evidence_validation": validation,
                    "counted_toward_smoke": False,
                }
            )

        methods.append(
            {
                "id": method_id_value,
                "name": entry.get("name"),
                "availability": entry.get("availability"),
                "layers": layers,
                "auxiliary_evidence": {
                    "authority": "configs/baselines.json:auxiliary_evidence",
                    "items": auxiliary_items,
                    "passed_count": sum(
                        item.get("status") == "passed" for item in auxiliary
                    ),
                    "counted_toward_smoke": False,
                },
            }
        )

    manifest_summary = manifest.get("summary")
    if not isinstance(manifest_summary, dict):
        manifest_summary = {}
        errors.append(error_record("missing-manifest-summary"))
    summary_checks: dict[str, object] = {}
    for field, specification in SUMMARY_COUNT_FIELDS.items():
        expected = expected_summary_value(field, specification, entries)
        declared = manifest_summary.get(field)
        matched = declared == expected
        summary_checks[field] = {
            "declared": declared,
            "observed": expected,
            "matched": matched,
        }
        if not matched:
            errors.append(
                error_record(
                    "manifest-summary-count-mismatch",
                    layer=field,
                    detail=f"declared {declared!r}, observed {expected}",
                )
            )

    layer_counts: dict[str, dict[str, int]] = {}
    for layer_name in (
        "source_audit",
        "smoke",
        "public_benchmark",
        "full_reproduction",
    ):
        layer_counts[layer_name] = status_counts(
            [method["layers"][layer_name]["status"] for method in methods],
            allowed_status_list,
        )

    smoke_passed = layer_counts["smoke"].get("passed", 0)
    auxiliary_exclusion_check = smoke_passed == sum(
        entry.get("reproduction", {}).get("smoke", {}).get("status") == "passed"
        for entry in entries
    )
    if not auxiliary_exclusion_check:
        errors.append(error_record("auxiliary-evidence-counted-as-smoke"))

    return {
        "manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
            "schema_version": manifest.get("schema_version"),
            "generated_at": manifest.get("generated_at"),
        },
        "layer_semantics": {
            "source_audit": (
                "Auxiliary discovery by exact scripts.audit_source runner identity; "
                "never promotes or gates manifest smoke status."
            ),
            "smoke": (
                "configs/baselines.json reproduction.smoke declares the primary "
                "state; a passed declaration contributes to the main funnel count "
                "only when its run evidence is valid and not explicitly excluded."
            ),
            "public_benchmark": (
                "Only configs/baselines.json reproduction.benchmark is authoritative."
            ),
            "full_reproduction": (
                "Only configs/baselines.json reproduction.full is authoritative."
            ),
            "auxiliary_evidence": (
                "Validated when declared passed, but always excluded from smoke counts."
            ),
        },
        "summary": {
            "method_count": len(methods),
            "layer_status_counts": layer_counts,
            "main_reproduction_funnel_passed": {
                "smoke": valid_reproduction_passed_by_layer["smoke"],
                "public_benchmark": layer_counts["public_benchmark"].get(
                    "passed", 0
                ),
                "full_reproduction": layer_counts["full_reproduction"].get(
                    "passed", 0
                ),
            },
            "source_audit_passed": layer_counts["source_audit"].get("passed", 0),
            "auxiliary_passed_excluded_from_smoke": auxiliary_passed_claims,
            "manifest_passed_evidence": {
                "reproduction_claim_count": reproduction_passed_claims,
                "valid_reproduction_claim_count": valid_reproduction_passed_claims,
                "auxiliary_claim_count": auxiliary_passed_claims,
                "valid_auxiliary_claim_count": valid_auxiliary_passed_claims,
                "total_claim_count": reproduction_passed_claims
                + auxiliary_passed_claims,
                "valid_total_claim_count": valid_reproduction_passed_claims
                + valid_auxiliary_passed_claims,
            },
        },
        "methods": methods,
        "validation": {
            "manifest_summary_checks": summary_checks,
            "auxiliary_excluded_from_smoke": auxiliary_exclusion_check,
            "all_manifest_passed_evidence_valid": (
                reproduction_passed_claims == valid_reproduction_passed_claims
                and auxiliary_passed_claims == valid_auxiliary_passed_claims
            ),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs" / "baselines.json"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-reproduction-funnel"
        / "20260806-manifest-funnel-audit",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor()
    monitor.start()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-reproduction-funnel",
        "run_id": output_directory.name,
        "runner": "scripts.audit_reproduction_funnel",
        "status": "failed",
        "scope": "manifest-status-and-passed-evidence-integrity-only",
        "started_at_utc": utc_now(),
        "limitations": [
            "The manifest is authoritative for the 24-method set and all reproduction-layer statuses.",
            "Source audit status is auxiliary and discovered only from exact scripts.audit_source run records.",
            "A passed run.json proves only its declared scope; this audit does not reinterpret that scope as benchmark or full reproduction.",
            "Non-passed evidence paths are reported but are not required to contain a passed run record.",
            "No solver, model, checkpoint, benchmark data, or generated code is executed.",
        ],
    }
    try:
        audit = audit_funnel(ROOT, args.manifest)
        record.update(audit)
        record["status"] = (
            "passed" if audit["validation"]["error_count"] == 0 else "failed"
        )
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        usage = monitor.stop()
        record["started_at_utc"] = usage.started_at_utc
        record["ended_at_utc"] = usage.ended_at_utc
        record["resources"] = usage.to_dict()
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "run_json": str(output_directory / "run.json"),
                "summary": record.get("summary"),
                "validation": record.get("validation"),
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
