#!/usr/bin/env python3
"""Run ArcMemo's native fixed-memory dry-run with API/network fail-closed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.execution import run_process


EXPECTED_REVISION = "e2598f1c093a4ddd8c871bd949e6a425d70f0fa7"
PACKAGE_LIST_CODE = """\
import importlib.metadata
for distribution in sorted(
    importlib.metadata.distributions(),
    key=lambda item: (item.metadata.get('Name') or '').lower(),
):
    name = distribution.metadata.get('Name') or 'unknown'
    print(f'{name}=={distribution.version}')
"""
NETWORK_GUARD = '''\
import socket

def _blocked(*args, **kwargs):
    raise RuntimeError("network disabled by ArcMemo zero-dollar smoke")

socket.create_connection = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifacts(output_directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(output_directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "run.json"
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    source = args.source.resolve()
    python = args.python.absolute()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    if not python.is_file():
        parser.error(f"missing Python interpreter: {python}")
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
    upstream_output = output_directory / "upstream"
    command = [
        str(python),
        "-m",
        "concept_mem.evaluation.driver",
        "data=val2",
        "model=o4_mini",
        "generation=gen_default",
        "puzzle_retry.max_passes=1",
        "dry_run=true",
        f"hydra.run.dir={upstream_output}",
        "hydra.job.chdir=false",
    ]
    environment = dict(os.environ)
    for key in (
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "OPENAI_API_KEY": "audit-dummy-not-a-secret",
            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(guard_directory), str(source)]
            ),
        }
    )
    started_at = utc_now()
    process = run_process(
        command,
        cwd=source,
        timeout_seconds=args.timeout_seconds,
        environment=environment,
    )
    (output_directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output_directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
    package_process = subprocess.run(
        [str(python), "-c", PACKAGE_LIST_CODE],
        text=True,
        capture_output=True,
        check=False,
    )
    packages = package_process.stdout.splitlines()
    target_python = subprocess.run(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    produced = artifacts(output_directory)
    relative_paths = {item["path"] for item in produced}
    prompts_written = any(path.endswith("prompts.json") for path in relative_paths)
    model_outputs_written = any(
        path.endswith("model_outputs.json") for path in relative_paths
    )
    token_usage_written = any(
        path.endswith("token_usage.json") for path in relative_paths
    )
    passed = (
        process.status == "passed"
        and prompts_written
        and not model_outputs_written
        and not token_usage_written
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "arcmemo",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_arcmemo",
        "status": "passed" if passed else (
            "timeout" if process.status == "timeout" else "failed"
        ),
        "scope": "native-fixed-memory-zero-dollar-dry-run",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "source": {"path": str(source), "revision": revision},
        "command": command,
        "environment": {
            "wrapper_python": platform.python_version(),
            "target_python": target_python,
            "interpreter": str(python),
            "installed_packages": packages,
            "package_collection_error": (
                None if package_process.returncode == 0 else package_process.stderr
            ),
            "network_guard": "socket connect/create_connection fail closed",
            "provider_credentials": "dummy OpenAI key; all other known provider keys removed",
        },
        "process": process.as_dict(),
        "checks": {
            "prompts_written": prompts_written,
            "model_outputs_written": model_outputs_written,
            "token_usage_written": token_usage_written,
            "api_requests_expected": 0,
        },
        "artifacts": produced,
        "fairness": {
            "test_labels_accessible_to_upstream_process": True,
            "score_eligible_for_fair_main_board": False,
            "continual_mode_exercised": False,
            "reason": (
                "This validates the upstream dry-run pipeline only. The ordinary "
                "driver scores public test outputs in-process; a later benchmark "
                "requires dummy labels and independent scoring."
            ),
        },
        "claim_boundary": (
            "No provider generation was requested. Dummy completions and any reported "
            "scores are protocol diagnostics, not ARC predictions or benchmark results."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
