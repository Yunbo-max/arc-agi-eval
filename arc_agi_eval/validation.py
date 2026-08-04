from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Grid = list[list[int]]


class TaskValidationError(ValueError):
    """Raised when JSON or an ARC task does not follow the expected schema."""


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError) as exc:
        raise TaskValidationError(f"{source}: cannot read JSON: {exc}") from exc
    except (json.JSONDecodeError, _DuplicateKeyError) as exc:
        raise TaskValidationError(f"{source}: invalid JSON: {exc}") from exc


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskValidationError(f"{where}: expected an object")
    return value


def validate_grid(value: Any, where: str = "grid") -> Grid:
    if not isinstance(value, list) or not value:
        raise TaskValidationError(f"{where}: expected a nonempty list of rows")
    if len(value) > 30:
        raise TaskValidationError(f"{where}: height {len(value)} exceeds 30")

    width: int | None = None
    for row_index, row in enumerate(value):
        row_where = f"{where}[{row_index}]"
        if not isinstance(row, list) or not row:
            raise TaskValidationError(f"{row_where}: expected a nonempty row")
        if width is None:
            width = len(row)
            if width > 30:
                raise TaskValidationError(f"{where}: width {width} exceeds 30")
        elif len(row) != width:
            raise TaskValidationError(
                f"{row_where}: width {len(row)} does not match width {width}"
            )

        for column_index, cell in enumerate(row):
            if type(cell) is not int or not 0 <= cell <= 9:
                raise TaskValidationError(
                    f"{row_where}[{column_index}]: expected an integer from 0 to 9"
                )

    return value


def _validate_pair(
    value: Any,
    where: str,
    *,
    output_required: bool,
) -> None:
    pair = _require_object(value, where)
    required = {"input"}
    allowed = {"input", "output"}
    if output_required:
        required.add("output")

    missing = required - pair.keys()
    extra = pair.keys() - allowed
    if missing:
        raise TaskValidationError(
            f"{where}: missing key(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise TaskValidationError(
            f"{where}: unknown key(s): {', '.join(sorted(extra))}"
        )

    validate_grid(pair["input"], f"{where}.input")
    if "output" in pair:
        validate_grid(pair["output"], f"{where}.output")


def validate_task(
    value: Any,
    *,
    source: str = "task",
    require_test_outputs: bool = True,
) -> dict[str, Any]:
    task = _require_object(value, source)
    required_keys = {"train", "test"}
    allowed_keys = required_keys | {"name"}
    missing = required_keys - task.keys()
    extra = task.keys() - allowed_keys
    if missing:
        raise TaskValidationError(
            f"{source}: missing key(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise TaskValidationError(
            f"{source}: unknown key(s): {', '.join(sorted(extra))}"
        )
    if "name" in task and (not isinstance(task["name"], str) or not task["name"]):
        raise TaskValidationError(f"{source}.name: expected a nonempty string")

    for split in ("train", "test"):
        pairs = task[split]
        if not isinstance(pairs, list) or not pairs:
            raise TaskValidationError(f"{source}.{split}: expected a nonempty list")
        for index, pair in enumerate(pairs):
            _validate_pair(
                pair,
                f"{source}.{split}[{index}]",
                output_required=split == "train" or require_test_outputs,
            )
    return task


def load_task(
    path: str | Path,
    *,
    require_test_outputs: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    task = validate_task(
        load_json(source),
        source=str(source),
        require_test_outputs=require_test_outputs,
    )
    if "name" in task and task["name"] != source.stem:
        raise TaskValidationError(
            f"{source}.name: {task['name']!r} does not match filename {source.stem!r}"
        )
    return task
