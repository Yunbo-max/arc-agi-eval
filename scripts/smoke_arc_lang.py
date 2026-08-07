#!/usr/bin/env python3
"""Import ARC Lang and parse challenge-only data without allowing network I/O."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.execution import run_process


EXPECTED_REVISION = "3d820fc07a9d904e7bbda325a11d7bfd4fa486f5"
NETWORK_GUARD = '''\
import socket

def _blocked(*args, **kwargs):
    raise RuntimeError("network disabled by ARC Lang zero-dollar smoke")

socket.create_connection = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
'''
PROBE = '''\
import json
from pathlib import Path
from src.models import Challenge
import src.run
from src.configs.gpt52_configs import gpt52_config_prod

path = Path("data/arc-prize-2025/arc-agi_training_challenges.json")
raw_challenges = json.loads(path.read_text())
challenges = {
    task_id: Challenge.model_validate({**value, "task_id": task_id})
    for task_id, value in raw_challenges.items()
}
first_id = sorted(challenges)[0]
first = challenges[first_id]
result = {
    "challenge_count": len(challenges),
    "first_task_id": first_id,
    "train_examples": len(first.train),
    "test_inputs": len(first.test),
    "test_model_fields": sorted(type(first.test[0]).model_fields),
    "default_config": "gpt52_config_prod",
    "default_step_count": len(gpt52_config_prod.steps),
    "default_final_follow_times": gpt52_config_prod.final_follow_times,
    "src_run_imported": True,
}
print("ARC_LANG_PROBE=" + json.dumps(result, sort_keys=True))
'''
PACKAGE_LIST_CODE = '''\
import importlib.metadata
for distribution in sorted(
    importlib.metadata.distributions(),
    key=lambda item: (item.metadata.get("Name") or "").lower(),
):
    print(f'{distribution.metadata.get("Name") or "unknown"}=={distribution.version}')
'''


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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    source = args.source.resolve()
    python = args.python.absolute()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if revision != EXPECTED_REVISION:
        parser.error(f"expected {EXPECTED_REVISION}, found {revision}")

    output_directory.mkdir(parents=True, exist_ok=True)
    guard_directory = output_directory / "network_guard"
    guard_directory.mkdir()
    (guard_directory / "sitecustomize.py").write_text(
        NETWORK_GUARD, encoding="utf-8"
    )
    environment = dict(os.environ)
    for key in (
        "LOGFIRE_API_KEY",
        "LOGFIRE_TOKEN",
        "NEON_DSN",
        "XAI_API_KEY",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "OPENAI_API_KEY": "audit-dummy-not-a-secret",
            "ANTHROPIC_API_KEY": "audit-dummy-not-a-secret",
            "DEEPSEEK_API_KEY": "audit-dummy-not-a-secret",
            "OPENROUTER_API_KEY": "audit-dummy-not-a-secret",
            "GEMINI_API_KEY": "audit-dummy-not-a-secret",
            "MAX_CONCURRENCY": "1",
            "LOCAL_LOGS_ONLY": "1",
            "LOG_FILE": str(output_directory / "arc-lang.log"),
            "VIZ": "0",
            "LOG_GRIDS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join([str(guard_directory), str(source)]),
        }
    )
    command = [str(python), "-c", PROBE]
    started_at = utc_now()
    process = run_process(
        command,
        cwd=source,
        timeout_seconds=args.timeout_seconds,
        environment=environment,
    )
    (output_directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output_directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
    result = None
    for line in process.stdout.splitlines():
        if line.startswith("ARC_LANG_PROBE="):
            result = json.loads(line.removeprefix("ARC_LANG_PROBE="))
    packages = subprocess.run(
        [str(python), "-c", PACKAGE_LIST_CODE],
        text=True,
        capture_output=True,
        check=False,
    )
    passed = (
        process.status == "passed"
        and isinstance(result, dict)
        and result.get("src_run_imported") is True
        and result.get("test_model_fields") == ["input"]
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "arc-lang-public",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_arc_lang",
        "status": "passed" if passed else (
            "timeout" if process.status == "timeout" else "failed"
        ),
        "scope": "zero-dollar-import-config-challenge-parser",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "source": {"path": str(source), "revision": revision},
        "command": command,
        "process": process.as_dict(),
        "result": result,
        "environment": {
            "interpreter": str(python),
            "installed_packages": packages.stdout.splitlines(),
            "package_collection_error": (
                None if packages.returncode == 0 else packages.stderr
            ),
            "network_guard": "socket connect/create_connection fail closed",
            "provider_credentials": "dummy only",
            "max_concurrency": 1,
            "local_logs_only": True,
        },
        "fairness": {
            "api_requests": 0,
            "evaluation_solutions_loaded": False,
            "score_eligible_for_fair_main_board": False,
            "reason": (
                "Only imports, the production config, and a training challenge-only "
                "parser were exercised. No instruction generation or ARC prediction ran."
            ),
        },
        "claim_boundary": (
            "Passing proves dependency/import/config/parser compatibility only; the "
            "upstream default still needs a label-free runner and hard cost fuse."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
