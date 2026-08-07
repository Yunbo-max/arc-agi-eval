#!/usr/bin/env python3
"""Persist the ARC-AGI-1 evaluation versus ARC-AGI-2 training overlap audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.overlap import analyze_split_overlap, load_task_split


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left",
        type=Path,
        default=ROOT / "third_party" / "arc-agi-1" / "data" / "evaluation",
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=ROOT / "third_party" / "arc-agi-2" / "data" / "training",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "reports" / "e0-overlap" / "20260806-arc1-eval-vs-arc2-train",
    )
    args = parser.parse_args()

    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    started_at = utc_now()
    overlap = analyze_split_overlap(
        load_task_split(args.left), load_task_split(args.right)
    )
    expected = {
        "left_task_count": 400,
        "right_task_count": 1000,
        "id_overlap_count": 376,
        "semantic_labeled_exact_count": 375,
        "test_io_exact_count": 376,
    }
    passed = all(overlap[key] == value for key, value in expected.items())
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-overlap",
        "run_id": output_directory.name,
        "runner": "scripts.audit_benchmark_overlap",
        "status": "passed" if passed else "failed",
        "scope": "cross-benchmark-labeled-task-overlap",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "splits": {
            "left": str(args.left.resolve()),
            "right": str(args.right.resolve()),
        },
        "expected_core_counts": expected,
        "overlap": overlap,
        "policy": (
            "A checkpoint trained on ARC-AGI-2 training is ineligible for a "
            "label-clean ARC-AGI-1 public-evaluation score unless every overlapping "
            "task was excluded before training and that exclusion is auditable."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
