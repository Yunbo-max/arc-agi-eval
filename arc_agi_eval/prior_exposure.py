"""Build and validate a workspace-bounded prior-exposure disclosure manifest."""

from __future__ import annotations

from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "password",
    "secret",
}
CONTROL_PLANE_POLICIES = {
    "reports/e0-prior-exposure/**/run.json": {
        "method_id": "e0-prior-exposure",
        "runner": "scripts.audit_prior_exposure",
        "scope": "workspace-bounded-prior-exposure-disclosure-draft",
        "allowed_keys": {
            "claim_boundary",
            "ended_at_utc",
            "error",
            "limitations",
            "manifest",
            "method_id",
            "protocol_status",
            "resources",
            "run_id",
            "runner",
            "schema_version",
            "scope",
            "started_at_utc",
            "status",
            "summary",
        },
    },
    "reports/e0-protocol/**/run.json": {
        "method_id": "e0-protocol",
        "runner": "scripts.audit_protocol_root",
        "scope": "protocol-v1-draft-readiness-root",
        "allowed_keys": {
            "claim_boundary",
            "ended_at_utc",
            "environment",
            "error",
            "limitations",
            "manifest",
            "method_id",
            "protocol_status",
            "readiness",
            "resources",
            "run_id",
            "runner",
            "schema_version",
            "scope",
            "started_at_utc",
            "status",
        },
    },
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


def _secret_key_paths(value: object, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            child_path = f"{prefix}.{key}"
            if normalized in FORBIDDEN_SECRET_KEYS:
                found.append(child_path)
            found.extend(_secret_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_secret_key_paths(child, f"{prefix}[{index}]"))
    return found


def classify_path(path: str, rules: list[dict[str, str]]) -> dict[str, str]:
    matches = [rule for rule in rules if fnmatch.fnmatchcase(path, rule["glob"])]
    if not matches:
        raise ValueError(f"inventory path is not classified: {path}")
    return matches[0]


def exclusion_for_path(
    path: str, exclusions: list[dict[str, str]]
) -> dict[str, str] | None:
    for exclusion in exclusions:
        if fnmatch.fnmatchcase(path, exclusion["glob"]):
            return exclusion
    return None


def _validate_control_plane_record(
    relative: str, run: dict[str, Any], exclusion: dict[str, str]
) -> None:
    policy = CONTROL_PLANE_POLICIES.get(exclusion["glob"])
    if policy is None:
        raise ValueError(f"unsupported control-plane exclusion: {exclusion['glob']}")
    for field in ("method_id", "runner", "scope"):
        if run.get(field) != policy[field]:
            raise ValueError(
                f"excluded control-plane identity mismatch in {relative}: {field}"
            )
    if run.get("status") not in {"passed", "failed"}:
        raise ValueError(f"excluded control-plane status is not terminal: {relative}")
    unknown = set(run) - policy["allowed_keys"]
    if unknown:
        raise ValueError(
            f"excluded control-plane record has non-attestation fields in {relative}: "
            f"{sorted(unknown)}"
        )
    if Path(relative).parent.name != run.get("run_id"):
        raise ValueError(f"excluded control-plane run_id/path mismatch: {relative}")
    if run.get("status") == "passed" and not isinstance(run.get("manifest"), dict):
        raise ValueError(f"passed control-plane attestation lacks manifest: {relative}")


def _validate_repo_relative(root: Path, declared: str) -> Path:
    path = Path(declared)
    if path.is_absolute():
        raise ValueError(f"evidence path must be repository-relative: {declared}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"evidence path escapes repository: {declared}") from error
    if not resolved.exists():
        raise ValueError(f"evidence path does not exist: {declared}")
    return resolved


def _inventory_run(root: Path, path: Path, rules: list[dict[str, str]]) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    run = _load_object(path)
    secret_paths = _secret_key_paths(run)
    if secret_paths:
        raise ValueError(f"secret-like keys in {relative}: {secret_paths}")
    rule = classify_path(relative, rules)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tier": rule["tier"],
        "classification_rule": rule["glob"],
        "classification_reason": rule["reason"],
        "status": run.get("status"),
        "method_id": run.get("method_id", run.get("baseline_id")),
        "run_id": run.get("run_id"),
        "runner": run.get("runner"),
        "scope": run.get("scope", run.get("level")),
    }


def _inventory_result(root: Path, path: Path, rules: list[dict[str, str]]) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    rule = classify_path(relative, rules)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tier": rule["tier"],
        "classification_rule": rule["glob"],
        "classification_reason": rule["reason"],
        "kind": "run_record" if path.name.endswith("-run.json") else "predictions",
    }


def build_prior_exposure_manifest(
    root: Path,
    config_path: Path,
    *,
    inventory_cutoff_utc: str | None = None,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    if config.get("protocol_status") != "draft-not-frozen":
        raise ValueError("prior-exposure config must remain draft-not-frozen")
    tiers = config.get("evidence_tiers")
    if not isinstance(tiers, list) or len(tiers) != len(set(tiers)):
        raise ValueError("evidence_tiers must be a unique list")
    rules = config.get("classification_rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("classification_rules must be a nonempty list")
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"glob", "tier", "reason"}:
            raise ValueError("every classification rule needs glob, tier, and reason")
        if rule["tier"] not in tiers:
            raise ValueError(f"unknown rule tier: {rule['tier']}")
    exclusions = config.get("inventory_exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("inventory_exclusions must be a list")
    for exclusion in exclusions:
        if not isinstance(exclusion, dict) or set(exclusion) != {"glob", "reason"}:
            raise ValueError("every inventory exclusion needs glob and reason")
        if not exclusion["glob"].startswith("reports/e0-"):
            raise ValueError("inventory exclusions are restricted to E0 control-plane reports")
        if not exclusion["reason"]:
            raise ValueError("inventory exclusion reason must be nonempty")
    if {item["glob"] for item in exclusions} != set(CONTROL_PLANE_POLICIES):
        raise ValueError(
            "inventory exclusions must match the fixed prior/protocol control-plane policy"
        )

    disclosures = config.get("disclosures")
    if not isinstance(disclosures, list) or not disclosures:
        raise ValueError("disclosures must be a nonempty list")
    disclosure_ids: list[str] = []
    disclosure_evidence: list[dict[str, object]] = []
    for disclosure in disclosures:
        if not isinstance(disclosure, dict):
            raise ValueError("disclosure must be an object")
        disclosure_id = disclosure.get("id")
        if not isinstance(disclosure_id, str) or not disclosure_id:
            raise ValueError("disclosure id must be a nonempty string")
        disclosure_ids.append(disclosure_id)
        if disclosure.get("tier") not in tiers:
            raise ValueError(f"unknown disclosure tier: {disclosure_id}")
        evidence = disclosure.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"disclosure evidence must be nonempty: {disclosure_id}")
        items = []
        for declared in evidence:
            if not isinstance(declared, str):
                raise ValueError(f"non-string evidence path: {disclosure_id}")
            resolved = _validate_repo_relative(root, declared)
            items.append(
                {
                    "path": declared,
                    "kind": "directory" if resolved.is_dir() else "file",
                    "sha256": sha256_file(resolved) if resolved.is_file() else None,
                }
            )
        disclosure_evidence.append({"id": disclosure_id, "evidence": items})
    if len(disclosure_ids) != len(set(disclosure_ids)):
        raise ValueError("disclosure ids must be unique")

    run_paths = sorted((root / "reports").glob("**/run.json"))
    included_run_paths: list[Path] = []
    excluded_control_plane_count = 0
    for path in run_paths:
        relative = path.relative_to(root).as_posix()
        if exclusion_for_path(relative, exclusions) is None:
            included_run_paths.append(path)
            continue
        excluded_run = _load_object(path)
        secret_paths = _secret_key_paths(excluded_run)
        if secret_paths:
            raise ValueError(f"secret-like keys in excluded control-plane record: {relative}")
        exclusion = exclusion_for_path(relative, exclusions)
        assert exclusion is not None
        _validate_control_plane_record(relative, excluded_run, exclusion)
        excluded_control_plane_count += 1
    runs = [_inventory_run(root, path, rules) for path in included_run_paths]
    results = [
        _inventory_result(root, path, rules)
        for path in sorted((root / "results").glob("*"))
        if path.is_file()
    ]
    if not runs:
        raise ValueError("no local run records found")
    if not results:
        raise ValueError("no result artifacts found")
    inventoried_run_paths = [item["path"] for item in runs]
    result_paths = [item["path"] for item in results]
    if len(inventoried_run_paths) != len(set(inventoried_run_paths)) or len(result_paths) != len(set(result_paths)):
        raise ValueError("duplicate inventory path")

    tier_counts = Counter(str(item["tier"]) for item in [*runs, *results])
    included_paths = sorted(
        [str(item["path"]) for item in runs]
        + [str(item["path"]) for item in results]
    )
    for section_name in ("leaderboard_submissions", "external_private_evaluation"):
        section = config.get(section_name)
        if not isinstance(section, dict) or not isinstance(section.get("records"), list):
            raise ValueError(f"{section_name} must contain a records list")
        if section.get("workspace_record_count") != len(section["records"]):
            raise ValueError(f"{section_name} workspace_record_count mismatch")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "arc-rebench-prior-exposure-draft-20260806",
        "protocol_status": "draft-not-frozen",
        "scope": config["scope"],
        "inventory_scope": config["inventory_scope"],
        "inventory_exclusions": exclusions,
        "inventory_snapshot": {
            "cutoff_at_utc": inventory_cutoff_utc,
            "cutoff_recorded": inventory_cutoff_utc is not None,
            "semantics": (
                "The caller records the UTC start of the pre-attestation snapshot; "
                "the included path set and all file hashes below define the exact cutoff."
            ),
            "included_path_set_sha256": hashlib.sha256(
                canonical_json_bytes(included_paths)
            ).hexdigest(),
        },
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
            "schema_version": config["schema_version"],
        },
        "evidence_tiers": tiers,
        "classification_rules": rules,
        "disclosures": disclosures,
        "disclosure_evidence": disclosure_evidence,
        "leaderboard_submissions": config["leaderboard_submissions"],
        "external_private_evaluation": config["external_private_evaluation"],
        "inventory": {
            "run_records": runs,
            "result_artifacts": results,
        },
        "summary": {
            "disclosure_count": len(disclosures),
            "run_record_count": len(runs),
            "excluded_control_plane_record_count_at_cutoff": excluded_control_plane_count,
            "result_artifact_count": len(results),
            "inventory_tier_counts": dict(sorted(tier_counts.items())),
            "leaderboard_submission_workspace_record_count": config[
                "leaderboard_submissions"
            ]["workspace_record_count"],
            "external_private_workspace_record_count": config[
                "external_private_evaluation"
            ]["workspace_record_count"],
        },
        "limitations": config["limitations"],
    }
    digest_payload = {
        "config_sha256": manifest["config"]["sha256"],
        "disclosure_evidence": disclosure_evidence,
        "run_records": runs,
        "result_artifacts": results,
    }
    manifest["inventory_sha256"] = hashlib.sha256(
        canonical_json_bytes(digest_payload)
    ).hexdigest()
    return manifest


__all__ = [
    "build_prior_exposure_manifest",
    "canonical_json_bytes",
    "classify_path",
    "exclusion_for_path",
    "sha256_file",
]
