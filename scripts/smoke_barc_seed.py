#!/usr/bin/env python3
"""Run one deterministic BARC synthetic seed without loading model weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

import numpy as np


EXPECTED_REVISION = "a7b51a6b1ff969da3a78a71c533b6d79a93966e7"


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


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", default="00d62c1b")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    revision_result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    revision = revision_result.stdout.strip()
    task_path = source / "seeds" / f"{args.task_id}.py"
    if revision != EXPECTED_REVISION:
        parser.error(f"expected {EXPECTED_REVISION}, found {revision}")
    if not task_path.is_file():
        parser.error(f"missing seed task: {task_path}")

    started_at = utc_now()
    wall_start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, str(task_path.parent))
    try:
        specification = importlib.util.spec_from_file_location(
            f"barc_seed_{args.task_id}", task_path
        )
        if specification is None or specification.loader is None:
            raise RuntimeError("could not construct module specification")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        input_grid = np.asarray(module.generate_input())
        output_grid = np.asarray(module.main(input_grid.copy()))
    finally:
        sys.path.pop(0)
    wall_time = time.perf_counter() - wall_start

    valid = (
        input_grid.ndim == 2
        and output_grid.ndim == 2
        and input_grid.shape == output_grid.shape
        and input_grid.size > 0
        and input_grid.shape[0] <= 30
        and input_grid.shape[1] <= 30
        and np.issubdtype(input_grid.dtype, np.integer)
        and np.issubdtype(output_grid.dtype, np.integer)
        and bool(np.all((0 <= input_grid) & (input_grid <= 9)))
        and bool(np.all((0 <= output_grid) & (output_grid <= 9)))
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "barc",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_barc_seed",
        "status": "passed" if valid else "failed",
        "scope": "bundled-seed-generator-and-program-only",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "source": {
            "path": str(source),
            "revision": revision,
            "task_path": task_path.relative_to(source).as_posix(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "parameters": {"seed": args.seed, "task_id": args.task_id},
        "result": {
            "input_shape": list(input_grid.shape),
            "output_shape": list(output_grid.shape),
            "input_colors": sorted(int(value) for value in np.unique(input_grid)),
            "output_colors": sorted(int(value) for value in np.unique(output_grid)),
            "input_sha256": digest_array(input_grid),
            "output_sha256": digest_array(output_grid),
        },
        "resources": {"wall_time_seconds": wall_time},
        "model_checkpoint_loaded": False,
        "claim_boundary": (
            "This executes one bundled synthetic generator and handwritten solution; "
            "it does not load a BARC language model or solve an ARC benchmark task"
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
