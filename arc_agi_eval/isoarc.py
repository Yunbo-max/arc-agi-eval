from __future__ import annotations

from typing import Any, Callable, Sequence

from .validation import Grid, validate_task


GridTransform = Callable[[Grid], Grid]


def identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def rotate_90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*reversed(grid))]


def rotate_180(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in reversed(grid)]


def rotate_270(grid: Grid) -> Grid:
    return [list(row) for row in reversed(list(zip(*grid)))]


def flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    return [row[:] for row in reversed(grid)]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def anti_transpose(grid: Grid) -> Grid:
    return rotate_180(transpose(grid))


D4: dict[str, GridTransform] = {
    "identity": identity,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}

INVERSE_D4 = {
    "identity": "identity",
    "rotate_90": "rotate_270",
    "rotate_180": "rotate_180",
    "rotate_270": "rotate_90",
    "flip_horizontal": "flip_horizontal",
    "flip_vertical": "flip_vertical",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def color_transform(mapping: dict[int, int]) -> GridTransform:
    if any(type(key) is not int or type(value) is not int for key, value in mapping.items()):
        raise ValueError("color mapping keys and values must be integers")
    if any(not 0 <= color <= 9 for color in (*mapping.keys(), *mapping.values())):
        raise ValueError("color mapping values must be between 0 and 9")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("color mapping must be injective")
    if set(mapping) != set(mapping.values()):
        raise ValueError(
            "partial color mapping must permute a closed color set; otherwise "
            "mapped values collide with unchanged colors"
        )

    def apply(grid: Grid) -> Grid:
        return [[mapping.get(color, color) for color in row] for row in grid]

    return apply


def inverse_color_mapping(mapping: dict[int, int]) -> dict[int, int]:
    color_transform(mapping)
    return {value: key for key, value in mapping.items()}


def transform_task(
    task: dict[str, Any],
    transform: GridTransform,
    *,
    train_order: Sequence[int] | None = None,
    test_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    def transform_pair(pair: dict[str, Grid]) -> dict[str, Grid]:
        result = {"input": transform(pair["input"])}
        if "output" in pair:
            result["output"] = transform(pair["output"])
        return result

    train = [transform_pair(pair) for pair in task["train"]]
    test = [transform_pair(pair) for pair in task["test"]]
    if train_order is not None:
        if sorted(train_order) != list(range(len(train))):
            raise ValueError("train_order must be a permutation of training indices")
        train = [train[index] for index in train_order]
    if test_order is not None:
        if sorted(test_order) != list(range(len(test))):
            raise ValueError("test_order must be a permutation of test indices")
        test = [test[index] for index in test_order]
    result: dict[str, Any] = {"train": train, "test": test}
    if "name" in task:
        result["name"] = task["name"]
    return validate_task(
        result,
        source="transformed task",
        require_test_outputs=all("output" in pair for pair in test),
    )


def transform_predictions(
    predictions: list[dict[str, Grid]], transform: GridTransform
) -> list[dict[str, Grid]]:
    return [
        {attempt: transform(grid) for attempt, grid in output.items()}
        for output in predictions
    ]


def restore_test_order(values: Sequence[Any], transformed_order: Sequence[int]) -> list[Any]:
    if sorted(transformed_order) != list(range(len(values))):
        raise ValueError("transformed_order must be a permutation of output indices")
    restored: list[Any] = [None] * len(values)
    for transformed_index, canonical_index in enumerate(transformed_order):
        restored[canonical_index] = values[transformed_index]
    return restored
