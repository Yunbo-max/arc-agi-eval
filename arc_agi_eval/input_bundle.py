"""Build and verify immutable protocol input bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .challenge_runtime import canonical_sha256, sha256_file
from .validation import load_task


def _safe_repo_path(root: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe repository path: {declared}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {declared}") from error
    if not resolved.is_file():
        raise ValueError(f"declared input is not a file: {declared}")
    return resolved


def verify_declared_inputs(
    root: Path, records: Iterable[Mapping[str, Any]]
) -> list[dict[str, object]]:
    """Verify exact declared input hashes and return a normalized inventory."""

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if set(record) != {"role", "path", "sha256"}:
            raise ValueError(f"declared input {index} has unexpected fields")
        role = record["role"]
        declared = record["path"]
        expected = record["sha256"]
        if not isinstance(role, str) or not role:
            raise ValueError(f"declared input {index} has invalid role")
        if not isinstance(declared, str) or declared in seen:
            raise ValueError(f"declared input {index} has duplicate/invalid path")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"declared input {index} has invalid SHA-256")
        seen.add(declared)
        source = _safe_repo_path(root, declared)
        observed = sha256_file(source)
        if observed != expected:
            raise ValueError(
                f"declared input hash mismatch: {declared}; "
                f"expected {expected}, observed {observed}"
            )
        normalized.append(
            {
                "role": role,
                "path": Path(declared).as_posix(),
                "sha256": observed,
                "bytes": source.stat().st_size,
            }
        )
    return normalized


def build_code_inventory(
    root: Path, include_globs: Iterable[str]
) -> list[dict[str, object]]:
    """Inventory repository-native evaluator, runner, and test source files."""

    paths: set[Path] = set()
    for pattern in include_globs:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("code include globs must be nonempty strings")
        paths.update(path for path in root.glob(pattern) if path.is_file())
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError(f"code inventory refuses symlink: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"code path escapes repository: {path}") from error
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if not records:
        raise ValueError("code inventory is empty")
    return records


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def verify_challenge_view(
    directory: Path, *, expected_task_count: int
) -> dict[str, object]:
    """Verify an inference-visible MANIFEST, all files, and label absence."""

    manifest_path = directory / "MANIFEST"
    manifest = _load_object(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"visible manifest has no file list: {manifest_path}")
    records: list[dict[str, object]] = []
    declared_names: set[str] = set()
    output_fields = 0
    output_denominator = 0
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"visible manifest record {index} is malformed")
        name = item["path"]
        if not isinstance(name, str) or name in declared_names:
            raise ValueError("visible manifest contains duplicate/invalid path")
        declared_names.add(name)
        target = (directory / name).resolve()
        try:
            target.relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError(f"visible manifest path escapes view: {name}") from error
        if not target.is_file() or sha256_file(target) != item["sha256"]:
            raise ValueError(f"visible challenge hash mismatch: {target}")
        task = load_task(target, require_test_outputs=False)
        output_fields += sum("output" in pair for pair in task["test"])
        output_denominator += len(task["test"])
        records.append(
            {
                "task_id": target.stem,
                "path": name,
                "sha256": item["sha256"],
                "bytes": target.stat().st_size,
                "test_input_count": len(task["test"]),
            }
        )
    actual_json_names = {path.name for path in directory.glob("*.json")}
    if declared_names != actual_json_names:
        raise ValueError("visible manifest task set differs from directory task set")
    if len(records) != expected_task_count:
        raise ValueError(
            f"challenge task count mismatch: expected {expected_task_count}, "
            f"observed {len(records)}"
        )
    if output_fields:
        raise ValueError("challenge view contains hidden test outputs")
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "task_count": len(records),
        "test_input_count": output_denominator,
        "test_output_fields_present": output_fields,
        "task_id_set_sha256": canonical_sha256(sorted(item["task_id"] for item in records)),
        "task_inventory_sha256": canonical_sha256(records),
        "records": records,
    }


def deterministic_task_order(
    task_ids: Iterable[str], *, domain: str, benchmark: str, seed: int
) -> list[str]:
    """Order task IDs by a domain-separated SHA-256 rank."""

    identifiers = list(task_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("task IDs must be nonempty and unique")
    if not isinstance(domain, str) or not domain:
        raise ValueError("task-order domain must be nonempty")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("benchmark ID must be nonempty")
    if type(seed) is not int or seed < 0:
        raise ValueError("task-order seed must be a nonnegative integer")

    def rank(task_id: str) -> tuple[str, str]:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task IDs must be nonempty strings")
        payload = f"{domain}\0{benchmark}\0{seed}\0{task_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), task_id

    return sorted(identifiers, key=rank)


def build_public_task_orders(
    views: Mapping[str, Mapping[str, Any]], *, domain: str, seeds: Iterable[int]
) -> dict[str, object]:
    seeds_list = list(seeds)
    if not seeds_list or len(seeds_list) != len(set(seeds_list)):
        raise ValueError("task-order seeds must be nonempty and unique")
    benchmarks: dict[str, object] = {}
    for benchmark, view in sorted(views.items()):
        records = view["records"]
        task_ids = [record["task_id"] for record in records]
        orders = []
        for seed in seeds_list:
            ordered = deterministic_task_order(
                task_ids, domain=domain, benchmark=benchmark, seed=seed
            )
            orders.append(
                {
                    "seed": seed,
                    "task_ids": ordered,
                    "order_sha256": canonical_sha256(ordered),
                }
            )
        benchmarks[benchmark] = {
            "task_count": len(task_ids),
            "task_id_set_sha256": canonical_sha256(sorted(task_ids)),
            "orders": orders,
        }
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": domain,
        "seeds": seeds_list,
        "benchmarks": benchmarks,
    }
    result["public_task_orders_sha256"] = canonical_sha256(result)
    return result
