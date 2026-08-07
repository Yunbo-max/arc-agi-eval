#!/usr/bin/env python3
"""Persist the output-primary scoring-contract migration audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.dataset import task_files
from arc_agi_eval.reference_scoring import reference_exact_score
from arc_agi_eval.scoring import score_prediction_file, score_predictions
from arc_agi_eval.validation import load_json, load_task


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_answers(path: Path) -> dict[str, list[list[list[int]]]]:
    answers: dict[str, list[list[list[int]]]] = {}
    for task_path in task_files(path):
        task = load_task(task_path)
        answers[task_path.stem] = [pair["output"] for pair in task["test"]]
    return answers


def assert_reference_agreement(
    predictions: dict[str, object],
    answers: dict[str, list[list[list[int]]]],
    *,
    top_k: int,
) -> dict[str, object]:
    production = score_predictions(predictions, answers, top_k=top_k)
    reference = reference_exact_score(predictions, answers, top_k=top_k)
    compared = {
        "tasks_total": (production.tasks_total, reference.tasks_total),
        "tasks_predicted": (production.tasks_predicted, reference.tasks_predicted),
        "tasks_exact": (production.tasks_exact, reference.tasks_exact),
        "outputs_total": (production.outputs_total, reference.outputs_total),
        "outputs_exact": (production.outputs_exact, reference.outputs_exact),
    }
    disagreements = {
        key: values for key, values in compared.items() if values[0] != values[1]
    }
    if disagreements:
        raise RuntimeError(f"production/reference disagreement: {disagreements}")
    return production.as_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-scoring"
        / "20260806-output-primary-contract",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-scoring",
        "run_id": output_directory.name,
        "runner": "scripts.audit_scorer_contract",
        "status": "failed",
        "scope": "scoring-contract-migration",
        "started_at_utc": utc_now(),
        "historical_results_overwritten": False,
    }
    try:
        synthetic_answers = {
            "multi": [[[1]], [[2]]],
            "missing": [[[3]]],
        }
        synthetic_predictions = {
            "multi": [
                {"attempt_1": [[1]], "attempt_2": [[0]]},
                {"attempt_1": [[0]], "attempt_2": [[2]]},
            ]
        }
        synthetic = {
            f"top_{top_k}": assert_reference_agreement(
                synthetic_predictions, synthetic_answers, top_k=top_k
            )
            for top_k in (1, 2)
        }
        expected_synthetic = {
            "top_1": {"outputs_exact": 1, "outputs_total": 3, "tasks_exact": 0, "tasks_total": 2},
            "top_2": {"outputs_exact": 2, "outputs_total": 3, "tasks_exact": 1, "tasks_total": 2},
        }
        for case, expected in expected_synthetic.items():
            for field, expected_value in expected.items():
                if synthetic[case][field] != expected_value:
                    raise RuntimeError(
                        f"{case}.{field}={synthetic[case][field]} != {expected_value}"
                    )

        canonical: dict[str, object] = {}
        canonical_specs = {
            "arc_agi_1_evaluation": (
                ROOT / "results" / "arc-agi-1-evaluation-baseline-predictions.json",
                ROOT / "third_party" / "arc-agi-1" / "data" / "evaluation",
                (0, 419, 0, 400),
            ),
            "arc_agi_2_evaluation": (
                ROOT / "results" / "arc-agi-2-evaluation-baseline-predictions.json",
                ROOT / "third_party" / "arc-agi-2" / "data" / "evaluation",
                (0, 167, 0, 120),
            ),
        }
        for name, (prediction_path, task_dir, expected_counts) in canonical_specs.items():
            production = score_prediction_file(prediction_path, task_dir, top_k=2)
            predictions = load_json(prediction_path)
            reference = reference_exact_score(
                predictions, load_answers(task_dir), top_k=2
            )
            counts = (
                production.outputs_exact,
                production.outputs_total,
                production.tasks_exact,
                production.tasks_total,
            )
            if counts != expected_counts:
                raise RuntimeError(f"{name} counts {counts} != {expected_counts}")
            if counts != (
                reference.outputs_exact,
                reference.outputs_total,
                reference.tasks_exact,
                reference.tasks_total,
            ):
                raise RuntimeError(f"{name}: independent scorer disagreement")
            canonical[name] = {
                "predictions": str(prediction_path),
                "task_dir": str(task_dir),
                "score": production.as_dict(),
                "independent_reference_agreed": True,
            }

        record.update(
            {
                "status": "passed",
                "python_version": platform.python_version(),
                "contract": {
                    "primary": "output_exact_pass_at_k",
                    "locked_public_top_k": 2,
                    "secondary": "strict_task_exact_pass_at_k",
                    "diagnostic_only": "micro_cell_accuracy",
                    "missing_policy": "zero credit in full declared denominator",
                    "compatibility": "legacy flat score fields retained additively",
                },
                "synthetic_golden_cases": synthetic,
                "canonical_deterministic_floor_checks": canonical,
                "limitations": [
                    "This is a scorer-contract audit, not a solver benchmark.",
                    "Historical result files were read but not rewritten.",
                ],
            }
        )
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
            "ru_maxrss_before": rss_start,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "KiB on Linux",
            "memory_scope": "current process peak; children excluded",
        }
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "run_json": str(output_directory / "run.json"),
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
