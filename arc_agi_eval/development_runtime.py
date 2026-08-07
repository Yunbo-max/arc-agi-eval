"""Freeze and materialize the known-overlap-excluded dev-audit runtime."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .firewall import challenge_only
from .scoring import score_predictions
from .validation import Grid, load_task


OFFICIAL_PROFILE_ID = "arc-rebench-development-partition-v1"


@dataclass(frozen=True)
class DevelopmentRuntimeBuild:
    manifest: dict[str, object]
    inference_files: dict[str, bytes]
    solution_files: dict[str, bytes]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def _repo_file(root: Path, declared: str) -> Path:
    path = Path(declared)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be repository-relative: {declared}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes repository: {declared}") from error
    if not resolved.is_file():
        raise ValueError(f"file does not exist: {declared}")
    return resolved


def build_development_runtime(
    root: Path, config_path: Path
) -> DevelopmentRuntimeBuild:
    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    if config.get("profile_id") != OFFICIAL_PROFILE_ID:
        raise ValueError("unexpected development partition profile")
    if config.get("freeze_status") != "frozen":
        raise ValueError("development partition declaration must be frozen")
    if config.get("claim_boundary") != "known-overlap-excluded-only":
        raise ValueError("development claim boundary cannot be broadened")
    source_declarations = config.get("source_manifests")
    if not isinstance(source_declarations, dict):
        raise ValueError("source_manifests must be an object")
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in ("general_training_only", "arc1_known_overlap_excluded"):
        declaration = source_declarations.get(name)
        if not isinstance(declaration, dict):
            raise ValueError(f"missing source declaration: {name}")
        path = _repo_file(root, str(declaration.get("path")))
        if sha256_file(path) != declaration.get("sha256"):
            raise ValueError(f"source manifest hash mismatch: {name}")
        loaded[name] = (path, _load_object(path))
    general_path, general = loaded["general_training_only"]
    clean_path, clean = loaded["arc1_known_overlap_excluded"]
    if general.get("summary", {}).get("deduplicated_cluster_count") != 1008:
        raise ValueError("general cluster denominator changed")
    if clean.get("summary", {}).get("excluded_cluster_count") != 376:
        raise ValueError("known-overlap exclusion denominator changed")
    if clean.get("summary", {}).get("remaining_cluster_count") != 632:
        raise ValueError("retained cluster denominator changed")
    if clean.get("source_manifest", {}).get("file_sha256") != sha256_file(general_path):
        raise ValueError("clean view no longer binds the declared general manifest")
    if clean.get("data_policy", {}).get("cluster_reallocation_performed") is not False:
        raise ValueError("clean view reallocated clusters")

    dev_config = config.get("dev_audit")
    if not isinstance(dev_config, dict):
        raise ValueError("dev_audit config must be an object")
    expected_contract = {
        "execution_unit": "one_frozen_representative_per_cluster",
        "representative_order": "source_clean_manifest_deduplicated_tasks_order",
        "expected_cluster_count": 94,
        "expected_representative_task_count": 94,
        "expected_source_record_count": 159,
        "expected_test_output_denominator": 97,
        "inference_visible_test_output_fields": 0,
        "scoring_data_location": "outside_inference_tree",
    }
    if dev_config != expected_contract:
        raise ValueError("official dev-audit runtime contract changed")
    split = clean.get("splits", {}).get("dev-audit")
    if not isinstance(split, dict):
        raise ValueError("clean dev-audit split is missing")
    if split.get("cluster_count") != 94 or split.get("source_record_count") != 159:
        raise ValueError("clean dev-audit split denominator changed")
    representatives = split.get("deduplicated_tasks")
    if not isinstance(representatives, list) or len(representatives) != 94:
        raise ValueError("dev-audit representative roster changed")
    if len({item["cluster_id"] for item in representatives}) != 94:
        raise ValueError("dev-audit representative cluster IDs are not unique")
    if len({item["task_id"] for item in representatives}) != 94:
        raise ValueError("dev-audit task IDs are not unique")
    cluster_by_id = {item["cluster_id"]: item for item in clean["clusters"]}
    task_record_by_id = {item["record_id"]: item for item in clean["task_records"]}
    source_roots = config.get("training_sources")
    if not isinstance(source_roots, dict):
        raise ValueError("training_sources must be an object")

    inference_files: dict[str, bytes] = {}
    solution_files: dict[str, bytes] = {}
    provenance_records: list[dict[str, object]] = []
    visible_task_records: list[dict[str, object]] = []
    solution_records: list[dict[str, object]] = []
    answers_original: dict[str, list[Grid]] = {}
    answers_mutated: dict[str, list[Grid]] = {}
    exact_predictions: dict[str, list[dict[str, Grid]]] = {}
    hidden_mutation_stable_count = 0
    test_output_denominator = 0
    for order_index, representative in enumerate(representatives):
        cluster_id = str(representative["cluster_id"])
        record_id = str(representative["record_id"])
        source_id = str(representative["source"])
        task_id = str(representative["task_id"])
        cluster = cluster_by_id.get(cluster_id)
        if cluster is None or cluster.get("representative_record_id") != record_id:
            raise ValueError(f"representative/cluster mismatch: {cluster_id}")
        task_record = task_record_by_id.get(record_id)
        if task_record is None or task_record.get("task_id") != task_id:
            raise ValueError(f"representative/task-record mismatch: {record_id}")
        source_root = source_roots.get(source_id)
        if not isinstance(source_root, str):
            raise ValueError(f"unknown training source: {source_id}")
        source_path = _repo_file(root, f"{source_root}/{task_record['relative_path']}")
        if sha256_file(source_path) != task_record.get("source_file_sha256"):
            raise ValueError(f"training source hash mismatch: {source_path}")
        labeled = load_task(source_path)
        challenge = challenge_only(labeled)
        mutated = copy.deepcopy(labeled)
        for pair in mutated["test"]:
            pair["output"] = [
                [(cell + 1) % 10 for cell in row] for row in pair["output"]
            ]
        if pretty_json_bytes(challenge) != pretty_json_bytes(challenge_only(mutated)):
            raise ValueError(f"hidden-label mutation changed dev challenge: {task_id}")
        hidden_mutation_stable_count += 1
        challenge_relative = f"inference/dev-audit/{task_id}.json"
        challenge_payload = pretty_json_bytes(challenge)
        inference_files[challenge_relative] = challenge_payload
        outputs = [pair["output"] for pair in labeled["test"]]
        solution_relative = f"scoring/dev-audit/{task_id}.json"
        solution_payload = pretty_json_bytes(
            {"task_id": task_id, "test_outputs": outputs}
        )
        solution_files[solution_relative] = solution_payload
        test_output_denominator += len(outputs)
        visible_task_records.append(
            {
                "order_index": order_index,
                "task_id": task_id,
                "cluster_id": cluster_id,
                "path": challenge_relative,
                "sha256": hashlib.sha256(challenge_payload).hexdigest(),
                "bytes": len(challenge_payload),
                "test_input_count": len(challenge["test"]),
                "test_output_fields_present": 0,
            }
        )
        solution_records.append(
            {
                "order_index": order_index,
                "task_id": task_id,
                "cluster_id": cluster_id,
                "path": solution_relative,
                "sha256": hashlib.sha256(solution_payload).hexdigest(),
                "bytes": len(solution_payload),
                "test_output_count": len(outputs),
            }
        )
        provenance_records.append(
            {
                "order_index": order_index,
                "cluster_id": cluster_id,
                "record_id": record_id,
                "source": source_id,
                "task_id": task_id,
                "source_path": source_path.relative_to(root).as_posix(),
                "source_sha256": task_record["source_file_sha256"],
                "challenge_path": challenge_relative,
                "challenge_sha256": hashlib.sha256(challenge_payload).hexdigest(),
                "solution_path": solution_relative,
                "solution_sha256": hashlib.sha256(solution_payload).hexdigest(),
            }
        )
        answers_original[task_id] = outputs
        answers_mutated[task_id] = [pair["output"] for pair in mutated["test"]]
        exact_predictions[task_id] = [
            {"attempt_1": output, "attempt_2": output} for output in outputs
        ]
    if test_output_denominator != 97:
        raise ValueError("dev-audit test-output denominator changed")

    visible_manifest = {
        "format": "arc-rebench-dev-audit-challenge-v1",
        "source_id": "known-overlap-excluded-training-only:dev-audit",
        "freeze_status": "frozen",
        "task_count": 94,
        "test_input_count": 97,
        "test_output_fields_present": 0,
        "files": visible_task_records,
    }
    visible_manifest_payload = pretty_json_bytes(visible_manifest)
    inference_files["inference/dev-audit/MANIFEST"] = visible_manifest_payload
    visible_inventory = [
        {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for path, payload in sorted(inference_files.items())
    ]
    solution_inventory = [
        {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for path, payload in sorted(solution_files.items())
    ]
    original_score = score_predictions(
        exact_predictions, answers_original, top_k=2, source="dev-audit-sentinel"
    )
    mutated_score = score_predictions(
        exact_predictions, answers_mutated, top_k=2, source="dev-audit-sentinel"
    )
    if original_score.outputs_exact != 97 or mutated_score.outputs_exact != 0:
        raise ValueError("dev-audit scorer label-sensitivity sentinel failed")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": OFFICIAL_PROFILE_ID,
        "freeze_status": "frozen",
        "claim_boundary": "known-overlap-excluded-only",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "source_manifests": {
            "general_training_only": {
                "path": general_path.relative_to(root).as_posix(),
                "sha256": sha256_file(general_path),
                "manifest_id": general["manifest_id"],
            },
            "arc1_known_overlap_excluded": {
                "path": clean_path.relative_to(root).as_posix(),
                "sha256": sha256_file(clean_path),
                "manifest_id": clean["manifest_id"],
            },
        },
        "partition": {
            "general_cluster_count": 1008,
            "known_overlap_excluded_cluster_count": 376,
            "remaining_cluster_count": 632,
            "no_cluster_reallocation": True,
            "split_roles": config["split_roles"],
        },
        "dev_audit": {
            "execution_unit": dev_config["execution_unit"],
            "cluster_count": 94,
            "representative_task_count": 94,
            "source_record_count": 159,
            "test_output_denominator": 97,
            "test_output_fields_present": 0,
            "representative_order_sha256": canonical_sha256(provenance_records),
            "challenge_inventory_sha256": canonical_sha256(visible_inventory),
            "solution_inventory_sha256": canonical_sha256(solution_inventory),
            "visible_manifest_sha256": hashlib.sha256(
                visible_manifest_payload
            ).hexdigest(),
            "inference_tree": "inference/dev-audit",
            "scoring_tree": "scoring/dev-audit",
        },
        "representatives": provenance_records,
        "inference_file_inventory": visible_inventory,
        "solution_file_inventory": solution_inventory,
        "checks": {
            "source_manifest_hashes_verified": True,
            "training_source_hashes_verified": True,
            "no_cluster_reallocation": True,
            "representative_cluster_ids_unique": True,
            "representative_task_ids_unique": True,
            "evaluation_task_files_read": 0,
            "evaluation_label_files_read": 0,
            "inference_visible_test_output_fields": 0,
            "hidden_label_mutation_task_count": hidden_mutation_stable_count,
            "hidden_label_mutation_stable": True,
            "scorer_label_sensitivity_passed": True,
            "scorer_original_outputs_exact": original_score.outputs_exact,
            "scorer_mutated_outputs_exact": mutated_score.outputs_exact,
            "scoring_data_outside_inference_tree": True,
        },
        "limitations": config["limitations"],
    }
    manifest["development_runtime_sha256"] = canonical_sha256(manifest)
    return DevelopmentRuntimeBuild(
        manifest=manifest,
        inference_files=inference_files,
        solution_files=solution_files,
    )


__all__ = [
    "DevelopmentRuntimeBuild",
    "OFFICIAL_PROFILE_ID",
    "build_development_runtime",
    "canonical_sha256",
    "sha256_file",
]
