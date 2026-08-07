from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import DatasetError, task_files
from .validation import load_task, validate_task


class FirewallError(ValueError):
    """Raised when a challenge-only tree cannot be created safely."""


def challenge_only(task: dict[str, Any]) -> dict[str, Any]:
    """Return a validated ARC task with every test output removed."""
    result: dict[str, Any] = {
        "train": task["train"],
        "test": [{"input": pair["input"]} for pair in task["test"]],
    }
    if "name" in task:
        result["name"] = task["name"]
    return validate_task(result, require_test_outputs=False)


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def generate_challenge_tree(
    source: str | Path,
    destination: str | Path,
    *,
    source_id: str = "redacted",
) -> dict[str, Any]:
    """Create a label-free copy of a split or dataset and return its manifest."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path or _inside(destination_path, source_path):
        raise FirewallError("destination must be outside the labeled source tree")
    if destination_path.exists() and any(destination_path.iterdir()):
        raise FirewallError(f"{destination_path}: destination is not empty")

    recursive = any(path.is_dir() for path in source_path.iterdir()) if source_path.is_dir() else False
    files = task_files(source_path, recursive=recursive)
    records: list[dict[str, str]] = []
    for path in files:
        task = challenge_only(load_task(path))
        relative = path.relative_to(source_path) if source_path.is_dir() else Path(path.name)
        target = destination_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _encoded(task)
        target.write_bytes(payload)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    manifest: dict[str, Any] = {
        "format": "arc-agi-challenge-tree-v1",
        # This file is inference-visible. Never put a labeled filesystem path,
        # hidden-label digest, or another label locator in it.
        "source_id": source_id,
        "tasks_total": len(records),
        "files": records,
    }
    manifest_payload = _encoded(manifest)
    # Deliberately omit a .json suffix so task enumeration cannot mistake the
    # metadata for an ARC task.
    (destination_path / "MANIFEST").write_bytes(manifest_payload)

    # Re-read the generated artifacts and prove that no test pair has an output.
    for record in records:
        generated = load_task(
            destination_path / record["path"], require_test_outputs=False
        )
        if any("output" in pair for pair in generated["test"]):
            raise FirewallError(f"{record['path']}: generated test output leaked")
    return manifest
