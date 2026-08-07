#!/usr/bin/env python3
"""Run and persist the synthetic E0 firewall, process, IsoARC, and scorer tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.resources import ResourceMonitor


TEST_MODULES = (
    "tests.test_execution",
    "tests.test_process_firewall",
    "tests.test_firewall",
    "tests.test_isoarc",
    "tests.test_reference_scoring",
)
SOURCE_FILES = (
    "arc_agi_eval/execution.py",
    "arc_agi_eval/firewall.py",
    "arc_agi_eval/isoarc.py",
    "arc_agi_eval/reference_scoring.py",
    "arc_agi_eval/scoring.py",
    "tests/test_execution.py",
    "tests/test_process_firewall.py",
    "tests/test_firewall.py",
    "tests/test_isoarc.py",
    "tests/test_reference_scoring.py",
)
SUMMARY_PATTERN = re.compile(r"Ran (\d+) tests? in ([0-9.]+)s")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: dict[str, object]) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-contracts"
        / "20260806-firewall-isoarc-process-terminal",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "unittest", *TEST_MODULES, "-v"]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    monitor = ResourceMonitor(include_nvidia=False).start()
    started_at = utc_now()
    started = time.perf_counter()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-contracts",
        "run_id": output_directory.name,
        "runner": "scripts.audit_e0_contracts",
        "status": "failed",
        "scope": "synthetic-firewall-process-isoarc-reference-scorer-contracts",
        "started_at_utc": started_at,
        "command": command,
        "working_directory": str(ROOT),
        "model_or_solver_executed": False,
        "network_requested": False,
        "gpu_requested": False,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        stdout_path = output_directory / "stdout.log"
        stderr_path = output_directory / "stderr.log"
        atomic_text(stdout_path, completed.stdout)
        atomic_text(stderr_path, completed.stderr)
        match = SUMMARY_PATTERN.search(completed.stderr + "\n" + completed.stdout)
        observed_tests = int(match.group(1)) if match else None
        tests_passed = completed.returncode == 0 and match is not None
        record.update(
            {
                "status": "passed" if tests_passed else "failed",
                "test_summary": {
                    "return_code": completed.returncode,
                    "observed_test_count": observed_tests,
                    "unittest_reported_seconds": (
                        float(match.group(2)) if match else None
                    ),
                    "wall_time_seconds": time.perf_counter() - started,
                    "modules": list(TEST_MODULES),
                },
                "contract_checks": {
                    "challenge_test_outputs_removed": tests_passed,
                    "challenge_tree_manifest_verified": tests_passed,
                    "label_mutation_prediction_bytes_stable": tests_passed,
                    "timeout_kills_process_group": tests_passed,
                    "d4_and_color_round_trip": tests_passed,
                    "independent_reference_scorers_agree": tests_passed,
                },
                "implementation_files": [
                    {
                        "path": declared,
                        "sha256": sha256_file(ROOT / declared),
                        "bytes": (ROOT / declared).stat().st_size,
                    }
                    for declared in SOURCE_FILES
                ],
                "files": {
                    "stdout": {
                        "path": "stdout.log",
                        "sha256": sha256_file(stdout_path),
                        "bytes": stdout_path.stat().st_size,
                    },
                    "stderr": {
                        "path": "stderr.log",
                        "sha256": sha256_file(stderr_path),
                        "bytes": stderr_path.stat().st_size,
                    },
                },
                "claim_boundary": (
                    "This terminalizes synthetic and fixture-based contract tests. "
                    "It does not materialize a locked-public challenge tree, prove "
                    "host namespace isolation, execute a solver, or freeze protocol v1."
                ),
                "limitations": [
                    "ResourceMonitor covers the parent audit process only; child-test resource use is not included.",
                    "Generated or otherwise untrusted solver code remains blocked because strict host isolation is unavailable.",
                ],
            }
        )
    except subprocess.TimeoutExpired as error:
        record["error"] = {
            "type": type(error).__name__,
            "message": f"contract test command exceeded {args.timeout_seconds:g}s",
        }
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        usage = monitor.stop()
        record["ended_at_utc"] = usage.ended_at_utc
        record["resources"] = usage.to_dict()
        atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
