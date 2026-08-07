#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "LPN"
RUN_DIR = ROOT / "reports" / "lpn" / "20260806-official-tests-architecture-smoke"
REVISION = "0adfe56b86d2cba5ae5794edb02da6399a96d98a"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tree_hash() -> str:
    lines = []
    for path in sorted(SOURCE.rglob("*.py")):
        lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(SOURCE).as_posix()}\n"
        )
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def execute(arguments: list[str]) -> dict[str, object]:
    env = {**os.environ, "PYTHONPATH": str(SOURCE)}
    started = time.perf_counter()
    completed = subprocess.run(arguments, cwd=ROOT, env=env, text=True, capture_output=True)
    return {
        "command": arguments,
        "exit_code": completed.returncode,
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise SystemExit(f"run directory is not empty: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = now()
    python = sys.executable
    discovery = execute([python, "-m", "unittest", "discover", "-s", str(SOURCE / "src"), "-p", "*_test.py", "-v"])
    generator = execute([python, str(SOURCE / "src/datasets/task_gen/re_arc_generators_test.py"), "-v"])
    transformer = execute([python, str(SOURCE / "src/models/transformer.py")])
    expected_discovery_bug = (
        discovery["exit_code"] == 1
        and "NameError: name 'random' is not defined" in str(discovery["stderr"])
        and "Ran 6 tests" in str(discovery["stderr"])
    )
    passed = expected_discovery_bug and generator["exit_code"] == 0 and transformer["exit_code"] == 0
    record = {
        "schema_version": 1,
        "runner": "scripts.smoke_lpn",
        "run_id": RUN_DIR.name,
        "status": "passed" if passed else "failed",
        "started_at_utc": started_utc,
        "ended_at_utc": now(),
        "source": {"revision": REVISION, "tree_sha256": tree_hash()},
        "configuration": {
            "network_access": False,
            "checkpoint_used": False,
            "training_data_used": False,
            "device": "cpu",
        },
        "environment": {"python": platform.python_version()},
        "checks": {
            "discovery": discovery,
            "expected_upstream_discovery_bug": expected_discovery_bug,
            "direct_400_generator_test": generator,
            "transformer_self_test": transformer,
        },
        "peak_child_max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "timing": {"wall_time_seconds": round(time.perf_counter() - started, 6)},
    }
    atomic(RUN_DIR / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
