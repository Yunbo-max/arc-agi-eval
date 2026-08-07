from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .validation import Grid


@dataclass(frozen=True)
class ReferenceExactScore:
    """Minimal exact-match result from an implementation independent of scoring.py."""

    tasks_total: int
    tasks_predicted: int
    tasks_exact: int
    outputs_total: int
    outputs_exact: int


def reference_exact_score(
    predictions: dict[str, list[dict[str, Grid]]],
    answers: dict[str, list[Grid]],
    *,
    top_k: int = 2,
) -> ReferenceExactScore:
    """Compute full-denominator exact scores with intentionally simple control flow.

    Inputs to this reference are expected to have passed the production schema
    validator. Keeping parsing out of this function makes its scoring logic
    independent from the production implementation.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    task_hits = 0
    output_hits = 0
    output_count = sum(len(outputs) for outputs in answers.values())
    for task_id, expected_outputs in answers.items():
        supplied = predictions.get(task_id, [])
        all_outputs_hit = True
        for index, expected in enumerate(expected_outputs):
            attempts = supplied[index] if index < len(supplied) else {}
            hit = any(attempts.get(f"attempt_{number}") == expected for number in range(1, top_k + 1))
            output_hits += int(hit)
            all_outputs_hit = all_outputs_hit and hit
        task_hits += int(all_outputs_hit)

    return ReferenceExactScore(
        tasks_total=len(answers),
        tasks_predicted=sum(task_id in answers for task_id in predictions),
        tasks_exact=task_hits,
        outputs_total=output_count,
        outputs_exact=output_hits,
    )
