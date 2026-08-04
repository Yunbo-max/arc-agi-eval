from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class DatasetError(ValueError):
    """Raised when a dataset path cannot be enumerated unambiguously."""


@dataclass(frozen=True)
class TaskRef:
    split: str
    task_id: str
    path: Path


def task_files(path: str | Path, *, recursive: bool = False) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() != ".json":
            raise DatasetError(f"{root}: expected a JSON file")
        return [root]
    if not root.is_dir():
        raise DatasetError(f"{root}: path does not exist or is not a directory")

    files = root.rglob("*.json") if recursive else root.glob("*.json")
    result = sorted(file for file in files if file.is_file())
    if not result:
        kind = "below" if recursive else "in"
        raise DatasetError(f"{root}: no JSON task files {kind} this path")
    return result


def enumerate_dataset(path: str | Path) -> list[TaskRef]:
    root = Path(path)
    if not root.is_dir():
        raise DatasetError(f"{root}: dataset path is not a directory")

    direct_files = sorted(file for file in root.glob("*.json") if file.is_file())
    if direct_files:
        return [TaskRef(root.name, file.stem, file) for file in direct_files]

    refs: list[TaskRef] = []
    for split_path in sorted(path for path in root.iterdir() if path.is_dir()):
        refs.extend(
            TaskRef(split_path.name, file.stem, file)
            for file in sorted(split_path.glob("*.json"))
            if file.is_file()
        )
    if not refs:
        raise DatasetError(f"{root}: no split directories containing JSON tasks")
    return refs


def group_by_split(refs: Iterable[TaskRef]) -> dict[str, list[TaskRef]]:
    grouped: dict[str, list[TaskRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.split, []).append(ref)
    return grouped
