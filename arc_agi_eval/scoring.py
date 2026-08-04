from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import task_files
from .validation import Grid, TaskValidationError, load_json, load_task, validate_grid

_ATTEMPT_KEY = re.compile(r"attempt_([1-9][0-9]*)\Z")


class PredictionValidationError(ValueError):
    """Raised when a prediction file does not follow the submission schema."""


@dataclass(frozen=True)
class Score:
    top_k: int
    tasks_total: int
    tasks_predicted: int
    tasks_exact: int
    task_exact_accuracy: float
    outputs_total: int
    outputs_exact: int
    output_exact_accuracy: float
    cells_total: int
    cells_correct: int
    cell_accuracy: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _cell_matches(candidate: Grid, expected: Grid) -> int:
    if len(candidate) != len(expected):
        return 0
    if any(
        len(candidate_row) != len(expected_row)
        for candidate_row, expected_row in zip(candidate, expected)
    ):
        return 0
    return sum(
        candidate_cell == expected_cell
        for candidate_row, expected_row in zip(candidate, expected)
        for candidate_cell, expected_cell in zip(candidate_row, expected_row)
    )


def _validate_predictions(
    value: Any,
    answers: dict[str, list[Grid]],
    *,
    source: str,
) -> dict[str, list[dict[int, Grid]]]:
    if not isinstance(value, dict):
        raise PredictionValidationError(f"{source}: expected an object keyed by task ID")
    if any(not isinstance(task_id, str) for task_id in value):
        raise PredictionValidationError(f"{source}: task IDs must be strings")

    unknown = set(value) - set(answers)
    if unknown:
        raise PredictionValidationError(
            f"{source}: unknown task ID(s): {', '.join(sorted(unknown))}"
        )

    parsed: dict[str, list[dict[int, Grid]]] = {}
    for task_id, task_predictions in value.items():
        where = f"{source}.{task_id}"
        if not isinstance(task_predictions, list):
            raise PredictionValidationError(f"{where}: expected a list of test outputs")
        expected_count = len(answers[task_id])
        if len(task_predictions) != expected_count:
            raise PredictionValidationError(
                f"{where}: expected {expected_count} test output(s), got {len(task_predictions)}"
            )

        parsed_outputs: list[dict[int, Grid]] = []
        for output_index, output_predictions in enumerate(task_predictions):
            output_where = f"{where}[{output_index}]"
            if not isinstance(output_predictions, dict) or not output_predictions:
                raise PredictionValidationError(
                    f"{output_where}: expected a nonempty object of attempts"
                )

            attempts: dict[int, Grid] = {}
            for key, grid in output_predictions.items():
                match = _ATTEMPT_KEY.fullmatch(key) if isinstance(key, str) else None
                if match is None:
                    raise PredictionValidationError(
                        f"{output_where}: invalid attempt key {key!r}"
                    )
                number = int(match.group(1))
                try:
                    attempts[number] = validate_grid(grid, f"{output_where}.{key}")
                except TaskValidationError as exc:
                    raise PredictionValidationError(str(exc)) from exc

            numbers = sorted(attempts)
            if numbers != list(range(1, numbers[-1] + 1)):
                raise PredictionValidationError(
                    f"{output_where}: attempt numbers must be contiguous starting at 1"
                )
            parsed_outputs.append(attempts)
        parsed[task_id] = parsed_outputs
    return parsed


def _load_answers(task_dir: str | Path) -> dict[str, list[Grid]]:
    answers: dict[str, list[Grid]] = {}
    for path in task_files(task_dir):
        if path.stem in answers:
            raise PredictionValidationError(f"{task_dir}: duplicate task ID {path.stem!r}")
        task = load_task(path)
        answers[path.stem] = [pair["output"] for pair in task["test"]]
    return answers


def score_predictions(
    predictions: Any,
    answers: dict[str, list[Grid]],
    *,
    top_k: int = 2,
    source: str = "predictions",
) -> Score:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    parsed = _validate_predictions(predictions, answers, source=source)

    tasks_exact = 0
    outputs_total = 0
    outputs_exact = 0
    cells_total = 0
    cells_correct = 0

    for task_id, expected_outputs in answers.items():
        predicted_outputs = parsed.get(task_id)
        task_is_exact = True
        for output_index, expected in enumerate(expected_outputs):
            outputs_total += 1
            expected_cells = sum(len(row) for row in expected)
            cells_total += expected_cells
            attempts = predicted_outputs[output_index] if predicted_outputs is not None else {}

            output_is_exact = False
            best_matches = 0
            for attempt_number in range(1, top_k + 1):
                candidate = attempts.get(attempt_number)
                if candidate is None:
                    continue
                output_is_exact = output_is_exact or candidate == expected
                best_matches = max(best_matches, _cell_matches(candidate, expected))

            if output_is_exact:
                outputs_exact += 1
            else:
                task_is_exact = False
            cells_correct += best_matches

        if task_is_exact:
            tasks_exact += 1

    tasks_total = len(answers)
    return Score(
        top_k=top_k,
        tasks_total=tasks_total,
        tasks_predicted=len(parsed),
        tasks_exact=tasks_exact,
        task_exact_accuracy=tasks_exact / tasks_total if tasks_total else 0.0,
        outputs_total=outputs_total,
        outputs_exact=outputs_exact,
        output_exact_accuracy=outputs_exact / outputs_total if outputs_total else 0.0,
        cells_total=cells_total,
        cells_correct=cells_correct,
        cell_accuracy=cells_correct / cells_total if cells_total else 0.0,
    )


def score_prediction_file(
    prediction_path: str | Path,
    task_dir: str | Path,
    *,
    top_k: int = 2,
) -> Score:
    prediction_path = Path(prediction_path)
    try:
        predictions = load_json(prediction_path)
    except TaskValidationError as exc:
        raise PredictionValidationError(str(exc)) from exc
    return score_predictions(
        predictions,
        _load_answers(task_dir),
        top_k=top_k,
        source=str(prediction_path),
    )
