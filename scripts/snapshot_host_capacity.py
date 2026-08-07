#!/usr/bin/env python3
"""Persist a read-only host disk/GPU capacity snapshot for run admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MIB = 1024**2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found"}
    fields = [
        "index",
        "name",
        "uuid",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
    ]
    completed = subprocess.run(
        [
            executable,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return {
            "status": "error",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    devices = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            return {"status": "error", "reason": f"unexpected row: {line}"}
        total_mib, used_mib, free_mib, utilization = map(
            int, values[3:]
        )
        devices.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "memory_total_bytes": total_mib * MIB,
                "memory_used_bytes": used_mib * MIB,
                "memory_free_bytes": free_mib * MIB,
                "utilization_percent": utilization,
            }
        )
    return {"status": "ok", "devices": devices}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filesystem", type=Path, default=ROOT)
    parser.add_argument("--disk-reserve-gib", type=float, default=8.0)
    parser.add_argument("--forward-free-vram-gib", type=float, default=10.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-resources"
        / "20260806-host-capacity-snapshot-0321",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    if args.disk_reserve_gib < 0 or args.forward_free_vram_gib < 0:
        parser.error("capacity thresholds must be non-negative")

    filesystem = args.filesystem.resolve()
    usage = shutil.disk_usage(filesystem)
    reserve_bytes = int(args.disk_reserve_gib * GIB)
    gpu = gpu_snapshot()
    forward_required_bytes = int(args.forward_free_vram_gib * GIB)
    gpu_devices = gpu.get("devices", []) if gpu["status"] == "ok" else []
    forward_gate = bool(gpu_devices) and all(
        device["memory_free_bytes"] >= forward_required_bytes
        and device["utilization_percent"] == 0
        for device in gpu_devices
    )
    model_path = Path("/model")
    record: dict[str, Any] = {
        "schema_version": 1,
        "method_id": "e0-resources",
        "run_id": output_directory.name,
        "runner": "scripts.snapshot_host_capacity",
        "status": "passed",
        "scope": "read_only_host_capacity_and_run_admission_snapshot",
        "started_at_utc": utc_now(),
        "ended_at_utc": utc_now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "filesystem": {
            "path": str(filesystem),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "minimum_reserve_bytes": reserve_bytes,
            "maximum_admissible_incremental_bytes": max(
                0, usage.free - reserve_bytes
            ),
        },
        "model_mount": {
            "path": str(model_path),
            "exists": model_path.exists(),
            "is_directory": model_path.is_dir(),
            "writable_if_present": (
                os.access(model_path, os.W_OK) if model_path.exists() else False
            ),
        },
        "gpu": gpu,
        "gates": {
            "disk_reserve_currently_satisfied": usage.free >= reserve_bytes,
            "ten_gib_free_vram_and_idle_on_every_visible_gpu": forward_gate,
            "forward_required_free_vram_bytes": forward_required_bytes,
            "policy": (
                "This is an observation, not a reservation. A GPU run must repeat "
                "the preflight immediately before allocation and reject contention."
            ),
        },
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
