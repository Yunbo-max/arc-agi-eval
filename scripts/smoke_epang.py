#!/usr/bin/env python3
"""Run epang's data model and trusted executor without labels, pickle, or API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


EXPECTED_REVISION = "2b1a3c12a38b8a62627b75d78fd0cd55eacff4a1"
EXPECTED_LIBRARY_SHA256 = (
    "8b3eb5e50fcfa766c495e9b9ee40ae9718a9d6907ce82fdb27676642157c6650"
)
NETWORK_GUARD = '''\
import socket

def _blocked(*args, **kwargs):
    raise RuntimeError("network disabled by epang zero-dollar smoke")

socket.create_connection = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
'''
PROBE = '''\
import json
from src.models import Challenge, Example
from src.run_python import run_python_transform_sync

challenge = Challenge(
    id="synthetic-no-label",
    train=[Example(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]])],
    test=[Example(input=[[5, 6], [7, 8]], output=[[0], [0]])],
)
trusted_code = """
def transform(grid_list):
    return [list(row) for row in zip(*grid_list[::-1])]
"""
execution = run_python_transform_sync(
    code=trusted_code,
    grid_lists=[challenge.test[0].input],
    timeout=10,
    raise_exception=False,
)
result = {
    "challenge_id": challenge.id,
    "test_model_has_output_field": "output" in type(challenge.test[0]).model_fields,
    "executor_return_code": execution.return_code,
    "executor_timed_out": execution.timed_out,
    "transform_results": execution.transform_results,
}
print("EPANG_PROBE=" + json.dumps(result, sort_keys=True))
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
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
    library = source / "saved_library_1000.pkl"
    library_sha256 = sha256(library)

    output_directory.mkdir(parents=True, exist_ok=True)
    guard_directory = output_directory / "network_guard"
    guard_directory.mkdir()
    (guard_directory / "sitecustomize.py").write_text(
        NETWORK_GUARD, encoding="utf-8"
    )
    environment = dict(os.environ)
    for key in (
        "LOGFIRE_TOKEN",
        "NEON_DB_DSN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PRINT_LOGS": "0",
            "PLOT": "0",
            "USE_GRID_URL": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join([str(guard_directory), str(source)]),
        }
    )
    command = [str(python), "-c", PROBE]
    started_at = utc_now()
    process = run_process(
        command,
        cwd=output_directory,
        timeout_seconds=args.timeout_seconds,
        environment=environment,
    )
    (output_directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output_directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
    result = None
    for line in process.stdout.splitlines():
        if line.startswith("EPANG_PROBE="):
            result = json.loads(line.removeprefix("EPANG_PROBE="))
    packages = subprocess.run(
        [str(python), "-c", PACKAGE_LIST_CODE],
        text=True,
        capture_output=True,
        check=False,
    )
    passed = (
        process.status == "passed"
        and library_sha256 == EXPECTED_LIBRARY_SHA256
        and isinstance(result, dict)
        and result.get("executor_return_code") == 0
        and result.get("transform_results") == [[[7, 5], [8, 6]]]
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "epang-arc-agi",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_epang",
        "status": "passed" if passed else (
            "timeout" if process.status == "timeout" else "failed"
        ),
        "scope": "zero-dollar-data-model-and-trusted-executor",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "source": {"path": str(source), "revision": revision},
        "command": command,
        "process": process.as_dict(),
        "result": result,
        "bundled_library": {
            "path": library.name,
            "size_bytes": library.stat().st_size,
            "sha256": library_sha256,
            "deserialized": False,
        },
        "environment": {
            "interpreter": str(python),
            "installed_packages": packages.stdout.splitlines(),
            "network_guard": "socket connect/create_connection fail closed",
            "provider_credentials_present": False,
        },
        "fairness": {
            "src_data_imported": False,
            "evaluation_labels_loaded": False,
            "api_requests": 0,
            "score_eligible_for_fair_main_board": False,
            "upstream_test_output_field_required": True,
        },
        "security": {
            "generated_code_executed": False,
            "trusted_fixed_probe_executed": True,
            "namespace_sandbox_available": False,
            "claim_boundary": (
                "The upstream executor has no filesystem/network sandbox. Only the "
                "fixed audit transform was allowed; model-generated code remains blocked."
            ),
        },
        "claim_boundary": (
            "This is an import/data-model/trusted-executor component smoke. It does "
            "not load the pickle, LPN, an API model, or solve an ARC task."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
