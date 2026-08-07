"""Build an immutable, label-free inventory of vendored ARC benchmark files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .validation import load_task


BENCHMARK_IDS = {"arc_agi_1", "arc_agi_2"}
EXPECTED_SPLIT_COUNTS = {
    "arc_agi_1": {"evaluation": 400, "training": 400},
    "arc_agi_2": {"evaluation": 120, "training": 1000},
}
TASK_ID = re.compile(r"[0-9a-f]{8}\Z")


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


def _repo_file(root: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute():
        raise ValueError(f"path must be repository-relative: {declared}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {declared}") from error
    if not resolved.is_file():
        raise ValueError(f"file does not exist: {declared}")
    return resolved


def _repo_directory(root: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute():
        raise ValueError(f"path must be repository-relative: {declared}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {declared}") from error
    if not resolved.is_dir():
        raise ValueError(f"directory does not exist: {declared}")
    return resolved


def build_benchmark_manifest(root: Path, config_path: Path) -> dict[str, object]:
    """Validate all declared task files and return only metadata and hashes."""

    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != BENCHMARK_IDS:
        raise ValueError("benchmarks must contain exactly arc_agi_1 and arc_agi_2")

    benchmark_records: dict[str, object] = {}
    total_task_files = 0
    total_test_outputs = 0
    for benchmark_id in sorted(benchmarks):
        declaration = benchmarks[benchmark_id]
        if not isinstance(declaration, dict):
            raise ValueError(f"benchmark declaration must be an object: {benchmark_id}")
        revision = declaration.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"invalid full revision: {benchmark_id}")
        local_root = _repo_directory(root, declaration["local_path"])
        data_root = _repo_directory(root, declaration["data_path"])
        try:
            data_root.relative_to(local_root)
        except ValueError as error:
            raise ValueError(f"data path is outside local snapshot: {benchmark_id}") from error
        license_path = _repo_file(root, declaration["license_path"])
        provenance_path = _repo_file(root, declaration["provenance_evidence"])
        expected_splits = declaration.get("expected_splits")
        if not isinstance(expected_splits, dict) or not expected_splits:
            raise ValueError(f"expected_splits must be nonempty: {benchmark_id}")
        if expected_splits != EXPECTED_SPLIT_COUNTS[benchmark_id]:
            raise ValueError(
                f"declared split counts do not match the fixed public benchmark: "
                f"{benchmark_id}"
            )

        observed_split_names = {
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and any(path.glob("*.json"))
        }
        if observed_split_names != set(expected_splits):
            raise ValueError(
                f"split set mismatch for {benchmark_id}: "
                f"expected {sorted(expected_splits)}, observed {sorted(observed_split_names)}"
            )

        split_records: dict[str, object] = {}
        benchmark_tasks: list[dict[str, object]] = []
        benchmark_task_ids: set[str] = set()
        for split in sorted(expected_splits):
            expected_count = expected_splits[split]
            if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 1:
                raise ValueError(f"invalid expected split count: {benchmark_id}/{split}")
            split_root = data_root / split
            files = sorted(path for path in split_root.glob("*.json") if path.is_file())
            if len(files) != expected_count:
                raise ValueError(
                    f"task count mismatch for {benchmark_id}/{split}: "
                    f"expected {expected_count}, observed {len(files)}"
                )
            task_ids = [path.stem for path in files]
            if len(task_ids) != len(set(task_ids)):
                raise ValueError(f"duplicate task ID in {benchmark_id}/{split}")
            invalid_ids = [task_id for task_id in task_ids if TASK_ID.fullmatch(task_id) is None]
            if invalid_ids:
                raise ValueError(f"invalid task ID in {benchmark_id}/{split}: {invalid_ids[0]}")
            repeated_across_splits = benchmark_task_ids.intersection(task_ids)
            if repeated_across_splits:
                raise ValueError(
                    f"task ID appears in multiple {benchmark_id} splits: "
                    f"{sorted(repeated_across_splits)[0]}"
                )
            benchmark_task_ids.update(task_ids)

            tasks: list[dict[str, object]] = []
            split_test_outputs = 0
            split_train_examples = 0
            split_bytes = 0
            for path in files:
                resolved_path = path.resolve()
                try:
                    resolved_path.relative_to(split_root.resolve())
                except ValueError as error:
                    raise ValueError(f"task path escapes split directory: {path}") from error
                task = load_task(path, require_test_outputs=True)
                record = {
                    "task_id": path.stem,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "train_example_count": len(task["train"]),
                    "test_output_count": len(task["test"]),
                }
                tasks.append(record)
                split_test_outputs += len(task["test"])
                split_train_examples += len(task["train"])
                split_bytes += path.stat().st_size
            split_digest = hashlib.sha256(canonical_json_bytes(tasks)).hexdigest()
            split_records[split] = {
                "task_count": len(tasks),
                "test_output_count": split_test_outputs,
                "train_example_count": split_train_examples,
                "bytes": split_bytes,
                "task_inventory_sha256": split_digest,
                "tasks": tasks,
            }
            benchmark_tasks.extend(
                {"split": split, **record} for record in tasks
            )
            total_task_files += len(tasks)
            total_test_outputs += split_test_outputs

        benchmark_records[benchmark_id] = {
            "repository": declaration["repository"],
            "declared_revision": revision,
            "retrieved_at": declaration["retrieved_at"],
            "local_path": local_root.relative_to(root).as_posix(),
            "license": {
                "path": license_path.relative_to(root).as_posix(),
                "sha256": sha256_file(license_path),
            },
            "provenance_evidence": {
                "path": provenance_path.relative_to(root).as_posix(),
                "sha256": sha256_file(provenance_path),
            },
            "splits": split_records,
            "snapshot_inventory_sha256": hashlib.sha256(
                canonical_json_bytes(benchmark_tasks)
            ).hexdigest(),
        }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "arc-public-benchmark-snapshots-20260806",
        "scope": "schema-validated public benchmark task inventory; no solver execution",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
            "schema_version": config["schema_version"],
        },
        "benchmarks": benchmark_records,
        "summary": {
            "benchmark_count": len(benchmark_records),
            "task_file_count": total_task_files,
            "test_output_count": total_test_outputs,
        },
        "limitations": config["limitations"],
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


__all__ = ["build_benchmark_manifest", "canonical_json_bytes", "sha256_file"]
