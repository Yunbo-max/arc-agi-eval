from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .dataset import task_files
from .scoring import score_prediction_file
from .validation import Grid, load_task

TrainPair = dict[str, Grid]
Transform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class Candidate:
    solver: str
    grid: Grid


def copy_input(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def dominant_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def constant_dominant_color(grid: Grid) -> Grid:
    color = dominant_color(grid)
    return [[color] * len(grid[0]) for _ in grid]


def _rotate_90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*reversed(grid))]


def _rotate_180(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in reversed(grid)]


def _rotate_270(grid: Grid) -> Grid:
    return [list(row) for row in reversed(list(zip(*grid)))]


def _flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_vertical(grid: Grid) -> Grid:
    return [row[:] for row in reversed(grid)]


def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def _anti_transpose(grid: Grid) -> Grid:
    return _rotate_180(_transpose(grid))


GEOMETRIC_TRANSFORMS: tuple[tuple[str, Transform], ...] = (
    ("identity", copy_input),
    ("rotate_90", _rotate_90),
    ("rotate_180", _rotate_180),
    ("rotate_270", _rotate_270),
    ("flip_horizontal", _flip_horizontal),
    ("flip_vertical", _flip_vertical),
    ("transpose", _transpose),
    ("anti_transpose", _anti_transpose),
)


def solve_geometric(train: Sequence[TrainPair], grid: Grid) -> list[Candidate]:
    candidates: list[Candidate] = []
    for name, transform in GEOMETRIC_TRANSFORMS:
        if all(transform(pair["input"]) == pair["output"] for pair in train):
            candidates.append(Candidate(f"geometric:{name}", transform(grid)))
    return candidates


def learn_color_mapping(train: Sequence[TrainPair]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for pair in train:
        source = pair["input"]
        target = pair["output"]
        if len(source) != len(target) or any(
            len(source_row) != len(target_row)
            for source_row, target_row in zip(source, target)
        ):
            return None
        for source_row, target_row in zip(source, target):
            for source_color, target_color in zip(source_row, target_row):
                previous = mapping.setdefault(source_color, target_color)
                if previous != target_color:
                    return None
    return mapping


def solve_color_mapping(
    train: Sequence[TrainPair], grid: Grid
) -> Candidate | None:
    mapping = learn_color_mapping(train)
    if mapping is None:
        return None
    return Candidate(
        "color_mapping",
        [[mapping.get(color, color) for color in row] for row in grid],
    )


def rank_candidates(
    train: Sequence[TrainPair], grid: Grid, *, top_k: int = 2
) -> list[Candidate]:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    ranked = solve_geometric(train, grid)
    color_candidate = solve_color_mapping(train, grid)
    if color_candidate is not None:
        ranked.append(color_candidate)
    ranked.extend(
        (
            Candidate("copy_input", copy_input(grid)),
            Candidate("dominant_color", constant_dominant_color(grid)),
        )
    )

    # Fill from the remaining constant colors so even uniform inputs get Top-K.
    counts = Counter(cell for row in grid for cell in row)
    for color in sorted(range(10), key=lambda value: (-counts[value], value)):
        ranked.append(
            Candidate(
                f"constant_color:{color}",
                [[color] * len(grid[0]) for _ in grid],
            )
        )

    result: list[Candidate] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for candidate in ranked:
        key = tuple(tuple(row) for row in candidate.grid)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) == top_k:
            return result
    raise ValueError(f"cannot produce {top_k} distinct candidates")


def predict_task(task: dict[str, Any], *, top_k: int = 2) -> list[dict[str, Grid]]:
    predictions: list[dict[str, Grid]] = []
    for pair in task["test"]:
        candidates = rank_candidates(task["train"], pair["input"], top_k=top_k)
        predictions.append(
            {
                f"attempt_{index}": candidate.grid
                for index, candidate in enumerate(candidates, start=1)
            }
        )
    return predictions


def generate_predictions(
    task_dir: str | Path, *, top_k: int = 2
) -> tuple[dict[str, list[dict[str, Grid]]], Counter[str]]:
    predictions: dict[str, list[dict[str, Grid]]] = {}
    solver_counts: Counter[str] = Counter()
    for path in task_files(task_dir):
        if path.stem in predictions:
            raise ValueError(f"{task_dir}: duplicate task ID {path.stem!r}")
        task = load_task(path, require_test_outputs=False)

        # Solvers receive only test inputs; public labels are reserved for scoring.
        prediction_task = {
            "train": task["train"],
            "test": [{"input": pair["input"]} for pair in task["test"]],
        }
        task_predictions: list[dict[str, Grid]] = []
        for pair in prediction_task["test"]:
            candidates = rank_candidates(
                prediction_task["train"], pair["input"], top_k=top_k
            )
            solver_counts.update(candidate.solver for candidate in candidates)
            task_predictions.append(
                {
                    f"attempt_{index}": candidate.grid
                    for index, candidate in enumerate(candidates, start=1)
                }
            )
        predictions[path.stem] = task_predictions
    return predictions, solver_counts


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_result_paths(task_dir: str | Path) -> tuple[Path, Path]:
    source = Path(task_dir)
    dataset = (
        source.parent.parent.name
        if source.parent.name == "data"
        else source.parent.name
    )
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{dataset}-{source.name}")
    result_dir = Path(__file__).resolve().parent.parent / "results"
    return (
        result_dir / f"{name}-baseline-predictions.json",
        result_dir / f"{name}-baseline-run.json",
    )


def run_baseline(
    task_dir: str | Path,
    prediction_path: str | Path,
    metadata_path: str | Path,
    *,
    score: bool = False,
    top_k: int = 2,
) -> dict[str, Any]:
    task_dir = Path(task_dir).resolve()
    prediction_path = Path(prediction_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()

    predictions, solver_counts = generate_predictions(task_dir, top_k=top_k)
    _write_json(prediction_path, predictions)
    prediction_elapsed = time.perf_counter() - started

    score_value = None
    scoring_elapsed = 0.0
    if score:
        scoring_started = time.perf_counter()
        score_value = score_prediction_file(
            prediction_path, task_dir, top_k=top_k
        ).as_dict()
        scoring_elapsed = time.perf_counter() - scoring_started

    prediction_bytes = prediction_path.read_bytes()
    outputs_total = sum(len(outputs) for outputs in predictions.values())
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "runner": "arc_agi_eval.deterministic_baseline",
        "runner_version": 1,
        "started_at_utc": started_at.isoformat(),
        "dataset_path": str(task_dir),
        "split": task_dir.name,
        "top_k": top_k,
        "prediction_path": str(prediction_path),
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "tasks_total": len(predictions),
        "test_outputs_total": outputs_total,
        "attempts_total": outputs_total * top_k,
        "solver_attempt_counts": dict(sorted(solver_counts.items())),
        "prediction_wall_time_seconds": round(prediction_elapsed, 6),
        "scoring_wall_time_seconds": round(scoring_elapsed, 6),
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "score": score_value,
    }
    _write_json(metadata_path, metadata)
    return metadata
