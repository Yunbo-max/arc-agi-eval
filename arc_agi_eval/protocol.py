"""Validate and materialize the single machine-readable protocol root.

The protocol root is deliberately stricter than an execution report.  It
records which gates are required for a freeze, why an unmet gate is blocking,
and hashes every local evidence file used by the decision.  A draft may have
open gates; a frozen protocol may not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


GATE_CLASSES = {
    "core_measurement",
    "trusted_execution",
    "untrusted_execution",
    "locked_public",
    "paper_parity",
}
GATE_STATUSES = {"passed", "pending", "blocked"}
REQUIRED_COMPARISON_AXES = {
    "evidence_scope",
    "parity_class",
    "resource_class",
    "code_trust_class",
}
REQUIRED_BENCHMARKS = {"arc_agi_1", "arc_agi_2"}
REQUIRED_BENCHMARK_DENOMINATORS = {
    "arc_agi_1": (400, 419),
    "arc_agi_2": (120, 167),
}
ARC_REBENCH_V1_REQUIRED_GATES = {
    "cm.scorer-contract",
    "cm.public-data-integrity",
    "cm.overlap-control",
    "cm.development-partition",
    "cm.e0-contracts",
    "cm.run-record-contract",
    "cm.prior-exposure",
    "te.challenge-runtime",
    "lp.method-eligibility",
    "lp.freeze-inputs",
    "lp.analysis-and-venue",
    "lp.fixed64-isoarc",
    "lp.process-tree-resources",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _nonempty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a nonempty string")
    return value


def _unique_strings(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a nonempty list")
    strings = [_nonempty_string(item, f"{where}[]") for item in value]
    if len(strings) != len(set(strings)):
        raise ValueError(f"{where} must contain unique values")
    return strings


def _repo_file(root: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute():
        raise ValueError(f"evidence path must be repository-relative: {declared}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"evidence path escapes repository: {declared}") from error
    if not resolved.is_file():
        raise ValueError(f"evidence file does not exist: {declared}")
    return resolved


def _evidence_record(root: Path, declared: str) -> dict[str, object]:
    relative = Path(declared)
    resolved = _repo_file(root, declared)
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _declared_evidence(value: object, where: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    paths = [_nonempty_string(item, f"{where}[]") for item in value]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{where} must not repeat a path")
    return paths


def _json_pointer(value: object, pointer: str, where: str) -> object:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"{where} must be an RFC 6901 JSON pointer")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"{where} does not exist: {pointer}")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as error:
                raise ValueError(f"{where} has a non-integer list index: {token}") from error
            if index < 0 or index >= len(current):
                raise ValueError(f"{where} list index is out of range: {token}")
            current = current[index]
        else:
            raise ValueError(f"{where} traverses a scalar at: {token}")
    return current


def _evaluate_gate_assertions(
    root: Path,
    gate_id: str,
    evidence: list[str],
    assertions_value: object,
    *,
    passed: bool,
) -> list[dict[str, object]]:
    if not isinstance(assertions_value, list):
        raise ValueError(f"{gate_id}.acceptance_assertions must be a list")
    if passed and not assertions_value:
        raise ValueError(f"passed gate requires acceptance assertions: {gate_id}")
    results: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, assertion in enumerate(assertions_value):
        where = f"{gate_id}.acceptance_assertions[{index}]"
        if not isinstance(assertion, dict) or set(assertion) != {"path", "pointer", "equals"}:
            raise ValueError(f"{where} must contain exactly path, pointer, and equals")
        declared = _nonempty_string(assertion["path"], f"{where}.path")
        pointer = assertion["pointer"]
        if not isinstance(pointer, str):
            raise ValueError(f"{where}.pointer must be a string")
        if declared not in evidence:
            raise ValueError(f"{where}.path is not gate evidence: {declared}")
        key = (declared, pointer)
        if key in seen:
            raise ValueError(f"duplicate gate assertion: {gate_id} {declared} {pointer}")
        seen.add(key)
        evidence_path = _repo_file(root, declared)
        try:
            document = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"assertion evidence is not valid JSON: {declared}") from error
        actual = _json_pointer(document, pointer, where)
        expected = assertion["equals"]
        if canonical_json_bytes(actual) != canonical_json_bytes(expected):
            raise ValueError(
                f"gate assertion failed: {gate_id} {declared}{pointer}; "
                f"expected {expected!r}, observed {actual!r}"
            )
        results.append(
            {
                "path": declared,
                "pointer": pointer,
                "equals": expected,
                "passed": True,
            }
        )
    if passed:
        for declared in evidence:
            if not declared.endswith("/run.json"):
                continue
            matching = [
                assertion
                for assertion in assertions_value
                if assertion["path"] == declared
                and assertion["pointer"] == "/status"
            ]
            if not matching or matching[0]["equals"] != "passed":
                raise ValueError(
                    f"passed gate must assert /status equals passed for run evidence: "
                    f"{gate_id} {declared}"
                )
        if not any(pointer != "/status" for _, pointer in seen):
            raise ValueError(
                f"passed gate requires a semantic assertion beyond /status: {gate_id}"
            )
    return results


def _validate_scoring_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("scoring_contract must be an object")
    primary = value.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("scoring_contract.primary must be an object")
    if primary.get("name") != "output_exact_pass_at_k":
        raise ValueError("primary score must be output_exact_pass_at_k")
    top_k = primary.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("scoring_contract.primary.top_k must be a positive integer")
    if primary.get("denominator_policy") != "all_declared_test_outputs":
        raise ValueError("primary denominator must contain all declared test outputs")
    if value.get("missing_output_policy") != "zero_credit_in_declared_denominator":
        raise ValueError("missing outputs must receive zero credit")
    evidence = _declared_evidence(value.get("evidence"), "scoring_contract.evidence")
    if not evidence:
        raise ValueError("scoring_contract requires evidence")
    return value


def _validate_benchmarks(value: object, top_k: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != REQUIRED_BENCHMARKS:
        raise ValueError("benchmark_contracts must contain exactly arc_agi_1 and arc_agi_2")
    for benchmark_id, contract in value.items():
        if not isinstance(contract, dict):
            raise ValueError(f"benchmark_contracts.{benchmark_id} must be an object")
        if contract.get("benchmark_generation") != benchmark_id:
            raise ValueError(f"benchmark generation mismatch: {benchmark_id}")
        if contract.get("top_k") != top_k:
            raise ValueError(f"benchmark top_k mismatch: {benchmark_id}")
        for field in ("declared_task_count", "declared_output_count"):
            count = contract.get(field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"{benchmark_id}.{field} must be a positive integer")
        if contract["declared_output_count"] < contract["declared_task_count"]:
            raise ValueError(f"{benchmark_id} output count cannot be below task count")
        expected_tasks, expected_outputs = REQUIRED_BENCHMARK_DENOMINATORS[
            benchmark_id
        ]
        if (
            contract["declared_task_count"],
            contract["declared_output_count"],
        ) != (expected_tasks, expected_outputs):
            raise ValueError(
                f"{benchmark_id} must declare the complete public denominator "
                f"({expected_tasks} tasks, {expected_outputs} outputs)"
            )
        evidence = _declared_evidence(
            contract.get("evidence", []), f"benchmark_contracts.{benchmark_id}.evidence"
        )
        if not evidence:
            raise ValueError(f"benchmark contract requires evidence: {benchmark_id}")
    return value


def _all_evidence_paths(config: dict[str, Any]) -> Iterable[str]:
    yield from config["scoring_contract"]["evidence"]
    for contract in config["benchmark_contracts"].values():
        yield from contract.get("evidence", [])
    for gate in config["freeze_gates"]:
        yield from gate["evidence"]


def build_protocol_manifest(root: Path, config_path: Path) -> dict[str, object]:
    """Return a validated protocol-root manifest with hashed local evidence."""

    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    protocol_status = config.get("protocol_status")
    if protocol_status not in {"draft-not-frozen", "frozen"}:
        raise ValueError("protocol_status must be draft-not-frozen or frozen")
    protocol_id = _nonempty_string(config.get("protocol_id"), "protocol_id")
    _nonempty_string(config.get("schema_version"), "schema_version")

    axes = config.get("comparison_axes")
    if not isinstance(axes, dict) or set(axes) != REQUIRED_COMPARISON_AXES:
        raise ValueError(
            "comparison_axes must contain exactly evidence_scope, parity_class, "
            "resource_class, and code_trust_class"
        )
    normalized_axes = {
        axis: _unique_strings(values, f"comparison_axes.{axis}")
        for axis, values in axes.items()
    }

    scoring = _validate_scoring_contract(config.get("scoring_contract"))
    benchmarks = _validate_benchmarks(
        config.get("benchmark_contracts"), scoring["primary"]["top_k"]
    )

    gates = config.get("freeze_gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("freeze_gates must be a nonempty list")
    gate_ids: list[str] = []
    required_unmet: list[str] = []
    gate_validations: list[dict[str, object]] = []
    gate_counts = {status: 0 for status in sorted(GATE_STATUSES)}
    for index, gate in enumerate(gates):
        where = f"freeze_gates[{index}]"
        if not isinstance(gate, dict):
            raise ValueError(f"{where} must be an object")
        gate_id = _nonempty_string(gate.get("id"), f"{where}.id")
        gate_ids.append(gate_id)
        if gate.get("class") not in GATE_CLASSES:
            raise ValueError(f"unknown gate class: {gate_id}")
        status = gate.get("status")
        if status not in GATE_STATUSES:
            raise ValueError(f"unknown gate status: {gate_id}")
        if not isinstance(gate.get("required_for_freeze"), bool):
            raise ValueError(f"{gate_id}.required_for_freeze must be boolean")
        evidence = _declared_evidence(gate.get("evidence"), f"{gate_id}.evidence")
        if status == "passed" and not evidence:
            raise ValueError(f"passed gate requires evidence: {gate_id}")
        assertion_results = _evaluate_gate_assertions(
            root,
            gate_id,
            evidence,
            gate.get("acceptance_assertions"),
            passed=status == "passed",
        )
        gate_validations.append(
            {
                "gate_id": gate_id,
                "status": status,
                "assertions": assertion_results,
                "all_assertions_passed": (
                    all(bool(result["passed"]) for result in assertion_results)
                    if assertion_results
                    else None
                ),
            }
        )
        if status != "passed":
            _nonempty_string(gate.get("blocker"), f"{gate_id}.blocker")
            if gate["required_for_freeze"]:
                required_unmet.append(gate_id)
        gate_counts[status] += 1
    if len(gate_ids) != len(set(gate_ids)):
        raise ValueError("freeze gate IDs must be unique")
    required_gate_ids = {
        gate["id"] for gate in gates if gate["required_for_freeze"]
    }
    if not required_gate_ids:
        raise ValueError("at least one freeze gate must be required_for_freeze")
    if protocol_id.startswith("arc-rebench-protocol-v1"):
        missing_roster = ARC_REBENCH_V1_REQUIRED_GATES - set(gate_ids)
        optional_roster = ARC_REBENCH_V1_REQUIRED_GATES - required_gate_ids
        if missing_roster:
            raise ValueError(
                "ARC-REBench protocol-v1 is missing mandatory gate IDs: "
                + ", ".join(sorted(missing_roster))
            )
        if optional_roster:
            raise ValueError(
                "ARC-REBench protocol-v1 mandatory gates cannot be optional: "
                + ", ".join(sorted(optional_roster))
            )

    unresolved = config.get("unresolved_p0")
    if not isinstance(unresolved, list):
        raise ValueError("unresolved_p0 must be a list")
    unresolved_gate_ids: list[str] = []
    for index, item in enumerate(unresolved):
        where = f"unresolved_p0[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        unresolved_gate_ids.append(_nonempty_string(item.get("gate_id"), f"{where}.gate_id"))
        _nonempty_string(item.get("resolution"), f"{where}.resolution")
    if len(unresolved_gate_ids) != len(set(unresolved_gate_ids)):
        raise ValueError("unresolved_p0 gate IDs must be unique")
    if set(unresolved_gate_ids) != set(required_unmet):
        raise ValueError(
            "unresolved_p0 must match every required, non-passed freeze gate exactly"
        )
    if protocol_status == "frozen" and required_unmet:
        raise ValueError("a frozen protocol cannot have an unmet required gate")

    evidence_by_path: dict[str, dict[str, object]] = {}
    for declared in _all_evidence_paths(config):
        record = _evidence_record(root, declared)
        evidence_by_path[str(record["path"])] = record
    evidence_inventory = [evidence_by_path[path] for path in sorted(evidence_by_path)]

    config_reference = {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file(config_path),
        "schema_version": config["schema_version"],
    }
    freeze_ready = protocol_status == "frozen" and not required_unmet
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_status": protocol_status,
        "config": config_reference,
        "comparison_axes": normalized_axes,
        "scoring_contract": scoring,
        "benchmark_contracts": benchmarks,
        "freeze_gates": gates,
        "gate_validations": gate_validations,
        "unresolved_p0": unresolved,
        "evidence_inventory": evidence_inventory,
        "readiness": {
            "freeze_ready": freeze_ready,
            "required_gate_count": len(required_gate_ids),
            "required_unmet_count": len(required_unmet),
            "required_unmet_gate_ids": sorted(required_unmet),
            "gate_status_counts": gate_counts,
        },
        "limitations": config.get("limitations", []),
    }
    manifest["protocol_root_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


__all__ = ["build_protocol_manifest", "canonical_json_bytes", "sha256_file"]
