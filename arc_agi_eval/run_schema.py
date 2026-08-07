"""Strict validation for new ARC-REBench protocol-v1 terminal run records.

This module intentionally does not migrate or reinterpret legacy ``run.json``
files.  It implements the JSON Schema keyword subset used by the pinned
protocol-v1 schema and adds cross-field, path, file-integrity, and secret checks
that JSON Schema alone cannot express clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "protocol-v1-run-1.0.0"
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "protocol-v1-run.schema.json"
)

PREDICTION_SCOPES = frozenset(
    {
        "solver_prediction_smoke",
        "single_task_experiment",
        "fixed_subset_benchmark",
        "full_public_benchmark",
        "paper_reproduction",
    }
)
UNTRUSTED_EXECUTION_CLASSES = frozenset(
    {"generated_untrusted", "unsafe_artifact"}
)
LOCAL_RESOURCE_CLASSES = frozenset(
    {"local_gpu", "local_cpu", "paper_equivalent_cluster"}
)

_UTC_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_SECRET_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:"
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|bearer[_-]?token|"
    r"client[_-]?secret|secret|password|passwd|credential|authorization|"
    r"private[_-]?key|cookie"
    r")(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE),
)
_SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "const",
        "description",
        "enum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)


class RunSchemaValidationError(ValueError):
    """Raised when a protocol-v1 run record is not valid."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        detail = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"protocol-v1 run validation failed:\n{detail}")


@dataclass(frozen=True)
class RunValidationResult:
    """Compact success result suitable for CLI and audit reports."""

    run_id: str
    record_sha256: str
    schema_sha256: str
    declared_file_count: int
    verified_file_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "declared_file_count": self.declared_file_count,
            "record_sha256": self.record_sha256,
            "run_id": self.run_id,
            "schema_sha256": self.schema_sha256,
            "status": "passed",
            "verified_file_count": self.verified_file_count,
        }


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number is prohibited: {token}")


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one strict JSON object, rejecting duplicate keys and NaN values."""

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot load strict JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    schema = load_json_object(path)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("protocol-v1 schema must declare JSON Schema draft 2020-12")
    if schema.get("type") != "object":
        raise ValueError("protocol-v1 schema root must declare object type")
    schema_version = (
        schema.get("properties", {}).get("schema_version", {}).get("const")
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            "protocol-v1 schema version constant does not match the validator"
        )
    _validate_schema_definition(schema)
    return schema


def _validate_schema_definition(
    schema: Mapping[str, Any],
    path: str = "$schema",
    root_schema: Mapping[str, Any] | None = None,
) -> None:
    selected_root = schema if root_schema is None else root_schema
    unsupported = sorted(set(schema) - _SUPPORTED_SCHEMA_KEYWORDS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"{path}: unsupported JSON Schema keyword(s): {joined}")

    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise ValueError(f"{path}.$ref: must be a string")
        siblings = set(schema) - {"$ref", "description", "title"}
        if siblings:
            joined = ", ".join(sorted(siblings))
            raise ValueError(
                f"{path}: $ref siblings are unsupported by this validator: {joined}"
            )
        _resolve_local_ref(selected_root, reference)

    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise ValueError(
            f"{path}.additionalProperties: only boolean values are supported"
        )

    declared_format = schema.get("format")
    if declared_format is not None and declared_format != "date-time":
        raise ValueError(f"{path}.format: unsupported format {declared_format!r}")

    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError(f"{path}.pattern: must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{path}.pattern: invalid regular expression: {exc}") from exc

    declared_type = schema.get("type")
    if declared_type is not None:
        declared_types = (
            [declared_type] if isinstance(declared_type, str) else declared_type
        )
        allowed_types = {
            "array",
            "boolean",
            "integer",
            "null",
            "number",
            "object",
            "string",
        }
        if not isinstance(declared_types, list) or not declared_types:
            raise ValueError(f"{path}.type: must be a string or non-empty list")
        if not all(item in allowed_types for item in declared_types):
            raise ValueError(f"{path}.type: contains an unsupported JSON type")

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise ValueError(f"{path}.required: must be a list of strings")
        if len(set(required)) != len(required):
            raise ValueError(f"{path}.required: duplicate field name")

    for container_name in ("$defs", "properties"):
        container = schema.get(container_name, {})
        if not isinstance(container, Mapping):
            raise ValueError(f"{path}.{container_name}: must be an object")
        for name, child in container.items():
            if not isinstance(child, Mapping):
                raise ValueError(f"{path}.{container_name}.{name}: must be an object")
            _validate_schema_definition(
                child,
                f"{path}.{container_name}.{name}",
                selected_root,
            )

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"{path}.items: must be an object")
        _validate_schema_definition(items, f"{path}.items", selected_root)


def _display_path(path: str, component: str | int) -> str:
    if isinstance(component, int):
        return f"{path}[{component}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", component):
        return f"{path}.{component}"
    return f"{path}[{component!r}]"


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _instance_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: object, declared: str) -> bool:
    if declared == "null":
        return value is None
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if declared == "string":
        return isinstance(value, str)
    if declared == "array":
        return isinstance(value, list)
    if declared == "object":
        return isinstance(value, dict)
    return False


def _resolve_local_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"only local JSON Schema references are supported: {reference}")
    current: object = root
    for encoded in reference[2:].split("/"):
        component = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"unresolvable JSON Schema reference: {reference}")
        current = current[component]
    if not isinstance(current, Mapping):
        raise ValueError(f"JSON Schema reference is not an object: {reference}")
    return current


def _parse_utc(value: str) -> datetime:
    if not _UTC_PATTERN.fullmatch(value):
        raise ValueError("must be an RFC 3339 UTC timestamp ending in Z")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _validate_schema_node(
    value: object,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    path: str,
    issues: list[str],
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            issues.append(f"{path}: schema $ref is not a string")
            return
        try:
            resolved = _resolve_local_ref(root_schema, reference)
        except ValueError as exc:
            issues.append(f"{path}: {exc}")
            return
        _validate_schema_node(value, resolved, root_schema, path, issues)
        return

    if "const" in schema and not _json_equal(value, schema["const"]):
        issues.append(f"{path}: expected constant {schema['const']!r}")

    if "enum" in schema:
        declared_values = schema["enum"]
        if not isinstance(declared_values, list) or not any(
            _json_equal(value, candidate) for candidate in declared_values
        ):
            issues.append(f"{path}: value {value!r} is not in the declared enum")

    declared_type = schema.get("type")
    if declared_type is not None:
        declared_types = (
            [declared_type] if isinstance(declared_type, str) else declared_type
        )
        if not isinstance(declared_types, list) or not all(
            isinstance(item, str) for item in declared_types
        ):
            issues.append(f"{path}: schema has an invalid type declaration")
            return
        if not any(_matches_type(value, item) for item in declared_types):
            expected = " or ".join(declared_types)
            issues.append(
                f"{path}: expected {expected}, received {_instance_type(value)}"
            )
            return

    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    issues.append(f"{_display_path(path, key)}: required field missing")

        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            issues.append(f"{path}: schema properties must be an object")
            return
        for key, child in value.items():
            child_path = _display_path(path, key)
            if key in properties:
                child_schema = properties[key]
                if isinstance(child_schema, Mapping):
                    _validate_schema_node(
                        child, child_schema, root_schema, child_path, issues
                    )
                else:
                    issues.append(f"{child_path}: invalid property schema")
            elif schema.get("additionalProperties") is False:
                issues.append(f"{child_path}: unknown field is prohibited")

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            issues.append(f"{path}: requires at least {minimum_items} item(s)")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            issues.append(f"{path}: permits at most {maximum_items} item(s)")
        if schema.get("uniqueItems") is True:
            observed: set[str] = set()
            for index, item in enumerate(value):
                marker = json.dumps(
                    item,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                if marker in observed:
                    issues.append(f"{path}[{index}]: duplicate array item")
                observed.add(marker)
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_node(
                    item,
                    item_schema,
                    root_schema,
                    _display_path(path, index),
                    issues,
                )

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        maximum_length = schema.get("maxLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            issues.append(f"{path}: shorter than minimum length {minimum_length}")
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            issues.append(f"{path}: longer than maximum length {maximum_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            issues.append(f"{path}: does not match required pattern {pattern!r}")
        if schema.get("format") == "date-time":
            try:
                _parse_utc(value)
            except ValueError as exc:
                issues.append(f"{path}: {exc}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            issues.append(f"{path}: must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            issues.append(f"{path}: must be at most {maximum}")


def _scan_for_secrets(value: object, path: str, issues: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = _display_path(path, key)
            if _SECRET_KEY_PATTERN.search(key):
                issues.append(f"{child_path}: secret-like field name is prohibited")
            _scan_for_secrets(child, child_path, issues)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_for_secrets(child, _display_path(path, index), issues)
    elif isinstance(value, str):
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                issues.append(f"{path}: value resembles a secret and is prohibited")
                break


def _repo_relative_path_issue(value: str, *, allow_dot: bool = False) -> str | None:
    if not value or "\x00" in value:
        return "path must be a non-empty text value"
    if "\\" in value:
        return "path must use POSIX separators"
    if value.startswith("/") or PurePosixPath(value).is_absolute():
        return "path must be repository-relative"
    if _WINDOWS_DRIVE_PATTERN.match(value):
        return "Windows drive paths are prohibited"
    if value == ".":
        return None if allow_dot else "path must name a file, not the repository root"
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return "path contains an empty, current-directory, or parent component"
    return None


def _resolved_repo_path(root: Path, value: str) -> Path:
    candidate = (root / PurePosixPath(value)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path escapes the repository root") from exc
    return candidate


def _validate_metric(metric: Mapping[str, Any], path: str, issues: list[str]) -> None:
    numerator = metric["numerator"]
    denominator = metric["denominator"]
    value = metric["value"]
    if numerator > denominator:
        issues.append(f"{path}: numerator cannot exceed denominator")
    expected = numerator / denominator
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        issues.append(
            f"{path}.value: expected numerator/denominator={expected:.17g}"
        )


def _require_digest_binding(
    files_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    digest: str,
    role: str,
    field_path: str,
    issues: list[str],
) -> None:
    candidates = files_by_role.get(role, ())
    if not any(
        item["sha256"] == digest and item["required_for_claim"]
        for item in candidates
    ):
        issues.append(
            f"{field_path}: no required {role} file has this SHA-256"
        )


def _validate_files_and_paths(
    record: Mapping[str, Any],
    *,
    repo_root: Path | None,
    verify_files: bool,
    issues: list[str],
) -> int:
    execution = record["execution"]
    working_directory = execution["working_directory"]
    path_issue = _repo_relative_path_issue(working_directory, allow_dot=True)
    if path_issue:
        issues.append(f"$.execution.working_directory: {path_issue}")
    elif repo_root is not None:
        try:
            resolved_working = _resolved_repo_path(repo_root, working_directory)
            if verify_files and not resolved_working.is_dir():
                issues.append(
                    "$.execution.working_directory: directory does not exist"
                )
        except ValueError as exc:
            issues.append(f"$.execution.working_directory: {exc}")

    results = record["results"]
    if isinstance(results, dict):
        for key in ("predictions_path", "score_path"):
            value = results[key]
            if isinstance(value, str):
                path_issue = _repo_relative_path_issue(value)
                if path_issue:
                    issues.append(f"$.results.{key}: {path_issue}")
                elif repo_root is not None:
                    try:
                        _resolved_repo_path(repo_root, value)
                    except ValueError as exc:
                        issues.append(f"$.results.{key}: {exc}")

    seen_paths: set[str] = set()
    verified_count = 0
    for index, file_record in enumerate(record["files"]):
        path = file_record["path"]
        issue_path = f"$.files[{index}].path"
        if path in seen_paths:
            issues.append(f"{issue_path}: duplicate declared file path")
        seen_paths.add(path)
        path_issue = _repo_relative_path_issue(path)
        if path_issue:
            issues.append(f"{issue_path}: {path_issue}")
            continue
        if repo_root is None:
            if verify_files:
                issues.append(f"{issue_path}: repo_root is required for file checks")
            continue
        try:
            resolved = _resolved_repo_path(repo_root, path)
        except ValueError as exc:
            issues.append(f"{issue_path}: {exc}")
            continue
        if not verify_files:
            continue
        if not resolved.is_file():
            issues.append(f"{issue_path}: declared file does not exist")
            continue
        observed_size = resolved.stat().st_size
        if observed_size != file_record["bytes"]:
            issues.append(
                f"$.files[{index}].bytes: declared {file_record['bytes']}, "
                f"observed {observed_size}"
            )
        observed_digest = sha256_file(resolved)
        if observed_digest != file_record["sha256"]:
            issues.append(
                f"$.files[{index}].sha256: declared digest does not match file"
            )
        if (
            observed_size == file_record["bytes"]
            and observed_digest == file_record["sha256"]
        ):
            verified_count += 1
    return verified_count


def _validate_semantics(record: Mapping[str, Any], issues: list[str]) -> None:
    started_at = _parse_utc(record["started_at_utc"])
    ended_at = _parse_utc(record["ended_at_utc"])
    if ended_at < started_at:
        issues.append("$.ended_at_utc: must not precede started_at_utc")

    status = record["status"]
    scope = record["evidence_scope"]
    execution = record["execution"]
    claim_started = execution["claim_execution_started"]
    target_executed = execution["target_code_executed"]
    if status in {"passed", "failed"} and not claim_started:
        issues.append(
            "$.execution.claim_execution_started: passed/failed requires true"
        )
    if status == "blocked" and claim_started:
        issues.append(
            "$.execution.claim_execution_started: blocked requires false"
        )
    if status == "blocked" and target_executed:
        issues.append("$.execution.target_code_executed: blocked requires false")

    failures = record["failures"]
    if failures["count"] != len(failures["items"]):
        issues.append("$.failures.count: must equal the number of failure items")
    if status == "failed" and failures["count"] == 0:
        issues.append("$.failures: failed terminal records require a failure")
    if status == "passed" and any(
        item["kind"] == "blocker" for item in failures["items"]
    ):
        issues.append("$.failures.items: passed status cannot contain a blocker")
    if status == "blocked":
        if failures["count"] == 0:
            issues.append("$.failures: blocked terminal records require a blocker")
        elif not any(item["kind"] == "blocker" for item in failures["items"]):
            issues.append("$.failures.items: blocked status requires kind=blocker")

    required_claim_files = [
        item for item in record["files"] if item["required_for_claim"]
    ]
    if status in {"passed", "failed"} and not required_claim_files:
        issues.append(
            "$.files: passed/failed terminal evidence requires a claim file"
        )

    files_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for file_record in record["files"]:
        files_by_role.setdefault(file_record["role"], []).append(file_record)

    provenance_bindings = (
        (
            "$.schema_digest_sha256",
            record["schema_digest_sha256"],
            "schema",
        ),
        (
            "$.protocol_digest_sha256",
            record["protocol_digest_sha256"],
            "protocol_manifest",
        ),
        (
            "$.source.lock_digest_sha256",
            record["source"]["lock_digest_sha256"],
            "source_lock",
        ),
        (
            "$.artifacts.manifest_digest_sha256",
            record["artifacts"]["manifest_digest_sha256"],
            "artifact_manifest",
        ),
        (
            "$.data.manifest_digest_sha256",
            record["data"]["manifest_digest_sha256"],
            "data_manifest",
        ),
        (
            "$.config.digest_sha256",
            record["config"]["digest_sha256"],
            "config",
        ),
        (
            "$.execution.environment_digest_sha256",
            record["execution"]["environment_digest_sha256"],
            "environment_lock",
        ),
        (
            "$.hardware.manifest_digest_sha256",
            record["hardware"]["manifest_digest_sha256"],
            "hardware_manifest",
        ),
    )
    for field_path, digest, role in provenance_bindings:
        _require_digest_binding(
            files_by_role,
            digest=digest,
            role=role,
            field_path=field_path,
            issues=issues,
        )

    source = record["source"]
    if source["dirty"] and source["patch_digest_sha256"] is None:
        issues.append("$.source.patch_digest_sha256: dirty source requires a digest")
    if not source["dirty"] and source["patch_digest_sha256"] is not None:
        issues.append(
            "$.source.patch_digest_sha256: clean source requires null"
        )
    if source["patch_digest_sha256"] is not None:
        _require_digest_binding(
            files_by_role,
            digest=source["patch_digest_sha256"],
            role="source_patch",
            field_path="$.source.patch_digest_sha256",
            issues=issues,
        )

    firewall = record["challenge_firewall"]
    if firewall["inference_started"] and not target_executed:
        issues.append(
            "$.challenge_firewall.inference_started: requires target code execution"
        )
    if (
        firewall["inference_started"]
        and firewall["challenge_manifest_digest_sha256"] is None
    ):
        issues.append(
            "$.challenge_firewall.challenge_manifest_digest_sha256: "
            "inference requires a challenge digest"
        )
    if firewall["challenge_manifest_digest_sha256"] is not None:
        _require_digest_binding(
            files_by_role,
            digest=firewall["challenge_manifest_digest_sha256"],
            role="challenge_manifest",
            field_path=(
                "$.challenge_firewall.challenge_manifest_digest_sha256"
            ),
            issues=issues,
        )
    if not firewall["inference_started"] and firewall["scoring_after_inference"]:
        issues.append(
            "$.challenge_firewall.scoring_after_inference: inference did not start"
        )

    trust_class = record["code_trust_class"]
    isolation = firewall["security_isolation"]
    if target_executed and trust_class in UNTRUSTED_EXECUTION_CLASSES:
        if isolation != "strict_sandbox":
            issues.append(
                "$.challenge_firewall.security_isolation: executed untrusted code "
                "requires strict_sandbox"
            )
    if target_executed and isolation == "no_execution":
        issues.append(
            "$.challenge_firewall.security_isolation: no_execution conflicts with "
            "target_code_executed"
        )
    if trust_class == "api_remote":
        if isolation != "api_remote":
            issues.append(
                "$.challenge_firewall.security_isolation: api_remote trust class "
                "requires api_remote isolation"
            )
        if firewall["network_policy"] != "api_allowlisted":
            issues.append(
                "$.challenge_firewall.network_policy: API execution must be allowlisted"
            )
        if record["resource_class"] != "api_hosted":
            issues.append("$.resource_class: API execution requires api_hosted")
    if execution["network_used"] and trust_class != "api_remote":
        issues.append("$.execution.network_used: only api_remote may use network")
    if trust_class == "no_code_execution" and target_executed:
        issues.append(
            "$.code_trust_class: no_code_execution conflicts with target execution"
        )

    resources = record["resources"]
    accounting_scope = resources["accounting_scope"]
    if accounting_scope == "current_process" and resources["children_included"]:
        issues.append(
            "$.resources.children_included: current_process scope excludes children"
        )
    if accounting_scope == "process_tree" and not resources["children_included"]:
        issues.append(
            "$.resources.children_included: process_tree scope includes children"
        )
    if accounting_scope in {"external_provider", "not_applicable"} and resources[
        "children_included"
    ]:
        issues.append(
            "$.resources.children_included: scope cannot include local children"
        )
    if claim_started and record["resource_class"] in LOCAL_RESOURCE_CLASSES:
        if accounting_scope not in {"current_process", "process_tree"}:
            issues.append(
                "$.resources.accounting_scope: started local execution requires "
                "current_process or process_tree"
            )
    if record["resource_class"] == "api_hosted" and claim_started:
        if accounting_scope != "external_provider":
            issues.append(
                "$.resources.accounting_scope: API execution requires external_provider"
            )
    if record["resource_class"] == "non_execution":
        if claim_started:
            issues.append("$.resource_class: non_execution cannot start a method")
        if accounting_scope != "not_applicable":
            issues.append(
                "$.resources.accounting_scope: non_execution requires not_applicable"
            )
    if execution["gpu_requested"] and record["resource_class"] not in {
        "local_gpu",
        "paper_equivalent_cluster",
    }:
        issues.append("$.execution.gpu_requested: resource class is not GPU-capable")

    hardware = record["hardware"]
    accelerator_kind = hardware["accelerator_kind"]
    accelerator_count = hardware["accelerator_count"]
    if accelerator_kind == "none":
        if accelerator_count != 0:
            issues.append("$.hardware.accelerator_count: none requires zero")
        if hardware["accelerator_model"] is not None:
            issues.append("$.hardware.accelerator_model: none requires null")
        if hardware["accelerator_uuid"] is not None:
            issues.append("$.hardware.accelerator_uuid: none requires null")
        if hardware["exclusive_accelerator"]:
            issues.append("$.hardware.exclusive_accelerator: none requires false")
    if accelerator_kind in {"nvidia_gpu", "other_gpu"}:
        if accelerator_count < 1:
            issues.append(
                "$.hardware.accelerator_count: local accelerator requires at least one"
            )
        if hardware["accelerator_model"] is None:
            issues.append(
                "$.hardware.accelerator_model: local accelerator requires a model"
            )
    if accelerator_kind == "remote_provider":
        if hardware["accelerator_model"] is None:
            issues.append(
                "$.hardware.accelerator_model: remote provider requires an identity"
            )
        if hardware["exclusive_accelerator"]:
            issues.append(
                "$.hardware.exclusive_accelerator: remote provider cannot claim local exclusivity"
            )
    if record["resource_class"] == "local_cpu" and accelerator_kind != "none":
        issues.append("$.hardware.accelerator_kind: local_cpu requires none")
    if record["resource_class"] in {"local_gpu", "paper_equivalent_cluster"}:
        if accelerator_kind not in {"nvidia_gpu", "other_gpu"}:
            issues.append(
                "$.hardware.accelerator_kind: local GPU resource requires a local accelerator"
            )
    if record["resource_class"] == "api_hosted" and accelerator_kind != "remote_provider":
        issues.append(
            "$.hardware.accelerator_kind: api_hosted requires remote_provider"
        )
    if execution["gpu_requested"] and accelerator_kind not in {
        "nvidia_gpu",
        "other_gpu",
    }:
        issues.append(
            "$.hardware.accelerator_kind: requested GPU is absent from hardware identity"
        )
    if (
        status == "passed"
        and scope in {"fixed_subset_benchmark", "full_public_benchmark"}
        and record["resource_class"]
        in {"local_gpu", "paper_equivalent_cluster"}
        and not hardware["exclusive_accelerator"]
    ):
        issues.append(
            "$.hardware.exclusive_accelerator: passed local benchmark requires true"
        )

    budget = record["attempt_budget"]
    for resource_key, cap_key in (
        ("api_calls", "api_call_cap"),
        ("input_tokens", "input_token_cap"),
        ("output_tokens", "output_token_cap"),
        ("cost_usd", "cost_cap_usd"),
    ):
        observed = resources[resource_key]
        if observed is not None and observed > budget[cap_key]:
            issues.append(
                f"$.resources.{resource_key}: exceeds attempt_budget.{cap_key}"
            )
    if status == "passed" and record["resource_class"] == "api_hosted":
        for key in ("api_calls", "input_tokens", "output_tokens", "cost_usd"):
            if resources[key] is None:
                issues.append(f"$.resources.{key}: passed API run must report usage")

    results = record["results"]
    if status == "passed" and results is None:
        issues.append("$.results: passed terminal records require results")
    if status == "blocked" and results is not None:
        issues.append("$.results: blocked terminal records cannot contain results")

    required_file_roles = {
        role: {
            item["path"] for item in items if item["required_for_claim"]
        }
        for role, items in files_by_role.items()
    }

    if isinstance(results, dict):
        primary = results["primary_metric"]
        metrics = ([primary] if isinstance(primary, dict) else []) + results[
            "secondary_metrics"
        ]
        for index, metric in enumerate(metrics):
            metric_path = (
                "$.results.primary_metric"
                if index == 0 and isinstance(primary, dict)
                else "$.results.secondary_metrics"
            )
            _validate_metric(metric, metric_path, issues)

        if results["kind"] == "check":
            if results["predictions_path"] is not None:
                issues.append("$.results.predictions_path: check result must be null")
            if results["score_path"] is not None:
                issues.append("$.results.score_path: check result must be null")
            if primary is not None or results["secondary_metrics"]:
                issues.append("$.results: check result cannot contain score metrics")
            if not results["checks"]:
                issues.append("$.results.checks: check result requires at least one check")
            if status == "passed" and any(
                check["status"] != "passed" for check in results["checks"]
            ):
                issues.append("$.results.checks: passed record contains a failed check")
            if status == "failed" and results["checks"] and not any(
                check["status"] == "failed" for check in results["checks"]
            ):
                issues.append("$.results.checks: failed record has no failed check")

        if results["kind"] == "arc_predictions":
            if status == "passed":
                predictions_path = results["predictions_path"]
                score_path = results["score_path"]
                if not isinstance(predictions_path, str):
                    issues.append(
                        "$.results.predictions_path: passed ARC result requires a path"
                    )
                elif predictions_path not in required_file_roles.get(
                    "predictions", set()
                ):
                    issues.append(
                        "$.results.predictions_path: no matching required predictions "
                        "file record"
                    )
                if not isinstance(score_path, str):
                    issues.append(
                        "$.results.score_path: passed ARC result requires a path"
                    )
                elif score_path not in required_file_roles.get("results", set()):
                    issues.append(
                        "$.results.score_path: no matching required results file record"
                    )
                if not isinstance(primary, dict):
                    issues.append(
                        "$.results.primary_metric: passed ARC result requires a metric"
                    )
                else:
                    if primary["name"] != "output_exact_pass_at_k":
                        issues.append(
                            "$.results.primary_metric.name: must be "
                            "output_exact_pass_at_k"
                        )
                    if primary["role"] != "primary":
                        issues.append(
                            "$.results.primary_metric.role: must be primary"
                        )
                    if primary["top_k"] != budget["top_k"]:
                        issues.append(
                            "$.results.primary_metric.top_k: must match attempt budget"
                        )

    if status == "passed" and scope in PREDICTION_SCOPES:
        if not isinstance(results, dict) or results["kind"] != "arc_predictions":
            issues.append("$.results: prediction scope requires arc_predictions")
        if budget["top_k"] < 1:
            issues.append("$.attempt_budget.top_k: prediction scope requires at least 1")
        if budget["max_candidates"] < budget["top_k"]:
            issues.append(
                "$.attempt_budget.max_candidates: cannot be less than top_k"
            )
        if record["data"]["task_count"] < 1:
            issues.append("$.data.task_count: prediction scope requires tasks")
        if not record["artifacts"]["licenses_verified"]:
            issues.append(
                "$.artifacts.licenses_verified: passed prediction requires true"
            )
        if source["revision"] is None:
            issues.append("$.source.revision: passed prediction requires revision")
        if not firewall["inference_started"]:
            issues.append(
                "$.challenge_firewall.inference_started: prediction scope requires true"
            )
        if firewall["inference_received_test_labels"]:
            issues.append(
                "$.challenge_firewall.inference_received_test_labels: prediction "
                "scope requires false"
            )
        if not firewall["scoring_after_inference"]:
            issues.append(
                "$.challenge_firewall.scoring_after_inference: prediction scope "
                "requires true"
            )
        if firewall["label_mutation_check"] != "passed":
            issues.append(
                "$.challenge_firewall.label_mutation_check: prediction scope "
                "requires passed"
            )
    elif status == "passed":
        if not isinstance(results, dict) or results["kind"] != "check":
            issues.append("$.results: non-prediction scope requires check results")


def validate_run_record(
    record: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    verify_files: bool = False,
    record_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> RunValidationResult:
    """Validate a new protocol-v1 record and return its integrity summary.

    ``verify_files`` is deliberately explicit for in-memory callers.  The CLI
    enables it by default for terminal evidence.
    """

    selected_schema = dict(schema) if schema is not None else load_schema()
    _validate_schema_definition(selected_schema)
    issues: list[str] = []
    _validate_schema_node(record, selected_schema, selected_schema, "$", issues)
    _scan_for_secrets(record, "$", issues)

    root = Path(repo_root).resolve() if repo_root is not None else None
    verified_file_count = 0
    if not issues:
        _validate_semantics(record, issues)
        if (
            schema_sha256 is not None
            and record["schema_digest_sha256"] != schema_sha256
        ):
            issues.append(
                "$.schema_digest_sha256: does not match the validator's schema file"
            )
        verified_file_count = _validate_files_and_paths(
            record,
            repo_root=root,
            verify_files=verify_files,
            issues=issues,
        )

    if issues:
        raise RunSchemaValidationError(issues)

    return RunValidationResult(
        run_id=str(record["run_id"]),
        record_sha256=record_sha256 or canonical_json_sha256(record),
        schema_sha256=schema_sha256 or canonical_json_sha256(selected_schema),
        declared_file_count=len(record["files"]),
        verified_file_count=verified_file_count,
    )


def validate_run_file(
    run_path: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    repo_root: str | Path | None = None,
    verify_files: bool = True,
) -> RunValidationResult:
    """Load and validate one terminal record without modifying it."""

    run_file = Path(run_path)
    schema_file = Path(schema_path)
    record = load_json_object(run_file)
    schema = load_schema(schema_file)
    selected_root = Path(repo_root) if repo_root is not None else run_file.parent
    return validate_run_record(
        record,
        schema=schema,
        repo_root=selected_root,
        verify_files=verify_files,
        record_sha256=sha256_file(run_file),
        schema_sha256=sha256_file(schema_file),
    )
