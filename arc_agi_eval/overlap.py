"""Semantic overlap analysis for ARC task splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_task_split(path: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"task split is not a directory: {directory}")
    tasks: dict[str, dict[str, Any]] = {}
    for task_path in sorted(directory.glob("*.json")):
        task_id = task_path.stem
        if task_id in tasks:
            raise ValueError(f"duplicate task id: {task_id}")
        value = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"task must be an object: {task_path}")
        tasks[task_id] = value
    return tasks


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def normalized_task(
    task: dict[str, Any],
    *,
    normalize_train_order: bool = True,
    normalize_test_order: bool = True,
    include_test_outputs: bool = True,
) -> dict[str, object]:
    """Drop metadata and normalize scientifically irrelevant example ordering."""

    train = list(task.get("train", []))
    if normalize_train_order:
        train.sort(key=_canonical_bytes)
    test: list[object] = []
    for example in task.get("test", []):
        if include_test_outputs:
            test.append(example)
        else:
            test.append({"input": example["input"]})
    if normalize_test_order:
        test.sort(key=_canonical_bytes)
    return {"train": train, "test": test}


def fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def analyze_split_overlap(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, object]:
    overlap_ids = sorted(set(left) & set(right))
    raw_without_name_exact: list[str] = []
    semantic_labeled_exact: list[str] = []
    challenge_exact: list[str] = []
    ordered_test_io_exact: list[str] = []
    test_io_exact: list[str] = []
    ordered_test_inputs_exact: list[str] = []
    test_inputs_exact: list[str] = []
    for task_id in overlap_ids:
        left_task = left[task_id]
        right_task = right[task_id]
        if normalized_task(
            left_task,
            normalize_train_order=False,
            normalize_test_order=False,
        ) == normalized_task(
            right_task,
            normalize_train_order=False,
            normalize_test_order=False,
        ):
            raw_without_name_exact.append(task_id)
        if normalized_task(left_task) == normalized_task(right_task):
            semantic_labeled_exact.append(task_id)
        if normalized_task(
            left_task, include_test_outputs=False
        ) == normalized_task(right_task, include_test_outputs=False):
            challenge_exact.append(task_id)
        left_test = left_task.get("test", [])
        right_test = right_task.get("test", [])
        if left_test == right_test:
            ordered_test_io_exact.append(task_id)
        if sorted(left_test, key=_canonical_bytes) == sorted(
            right_test, key=_canonical_bytes
        ):
            test_io_exact.append(task_id)
        left_inputs = [example.get("input") for example in left_test]
        right_inputs = [example.get("input") for example in right_test]
        if left_inputs == right_inputs:
            ordered_test_inputs_exact.append(task_id)
        if sorted(left_inputs, key=_canonical_bytes) == sorted(
            right_inputs, key=_canonical_bytes
        ):
            test_inputs_exact.append(task_id)
    return {
        "left_task_count": len(left),
        "right_task_count": len(right),
        "id_overlap_count": len(overlap_ids),
        "id_overlap": overlap_ids,
        "raw_without_name_exact_count": len(raw_without_name_exact),
        "raw_without_name_exact": raw_without_name_exact,
        "semantic_labeled_exact_count": len(semantic_labeled_exact),
        "semantic_labeled_exact": semantic_labeled_exact,
        "challenge_semantic_exact_count": len(challenge_exact),
        "challenge_semantic_exact": challenge_exact,
        "ordered_test_io_exact_count": len(ordered_test_io_exact),
        "ordered_test_io_exact": ordered_test_io_exact,
        "test_io_exact_count": len(test_io_exact),
        "test_io_exact": test_io_exact,
        "ordered_test_inputs_exact_count": len(ordered_test_inputs_exact),
        "ordered_test_inputs_exact": ordered_test_inputs_exact,
        "test_inputs_exact_count": len(test_inputs_exact),
        "test_inputs_exact": test_inputs_exact,
        "overlap_digest": fingerprint(overlap_ids),
    }


__all__ = [
    "analyze_split_overlap",
    "fingerprint",
    "load_task_split",
    "normalized_task",
]
