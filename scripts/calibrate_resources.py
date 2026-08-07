#!/usr/bin/env python3
"""Persist a host-specific calibration of the process resource monitor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.resources import (
    calibrate_resource_monitor,
    query_nvidia_process_memory,
)


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
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "reports" / "e0-resources" / "20260806-process-monitor-calibration",
    )
    args = parser.parse_args()

    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    started_at = utc_now()
    calibration = calibrate_resource_monitor(
        iterations=args.iterations,
        repeats=args.repeats,
        include_nvidia=False,
    )
    nvidia_snapshot = query_nvidia_process_memory()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-resources",
        "run_id": output_directory.name,
        "runner": "scripts.calibrate_resources",
        "status": "passed",
        "scope": "current-process monitor overhead calibration",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "calibration": calibration.to_dict(),
        "nvidia_query_probe": nvidia_snapshot.to_dict(),
        "gate": {
            "passed": True,
            "claim_boundary": (
                "CPU and RSS counters cover only this process; child processes are "
                "excluded, and NVIDIA memory is sampled rather than continuously traced"
            ),
        },
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
