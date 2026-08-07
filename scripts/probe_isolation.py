#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "reports" / "e0-isolation" / "20260806-host-namespace-probe"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def probe(arguments: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(arguments, text=True, capture_output=True, check=False)
    return {
        "command": arguments,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_time_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> int:
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise SystemExit(f"run directory is not empty: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started_utc = now()
    unshare = shutil.which("unshare")
    tools = {name: shutil.which(name) for name in ("bwrap", "unshare", "firejail", "docker", "podman", "nsjail")}
    checks = {}
    if unshare:
        checks["user_mount_network_namespace"] = probe(
            [unshare, "--user", "--map-root-user", "--mount", "--net", "true"]
        )
        checks["network_namespace"] = probe([unshare, "--net", "true"])
        checks["mount_namespace"] = probe([unshare, "--mount", "true"])
    namespace_available = bool(checks) and all(
        check["exit_code"] == 0 for check in checks.values()
    )
    record = {
        "schema_version": 1,
        "runner": "scripts.probe_isolation",
        "run_id": RUN_DIR.name,
        "status": "passed",
        "started_at_utc": started_utc,
        "ended_at_utc": now(),
        "environment": {
            "platform": platform.platform(),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "tools": tools,
        },
        "checks": checks,
        "isolation_gate_passed": namespace_available,
        "blocker": None if namespace_available else "no usable mount+network namespace or sandbox runtime",
    }
    temporary = RUN_DIR / ".run.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, RUN_DIR / "run.json")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
