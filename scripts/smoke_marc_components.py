#!/usr/bin/env python3
"""Exercise MARC's ARC task, submission, and voting components without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi_eval.resources import ResourceMonitor


EXPECTED_REVISION = "95b334872d435d5639135b32039577e2853a706b"


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


def git_output(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    revision = git_output(source, "rev-parse", "HEAD")
    dirty_paths = git_output(source, "status", "--porcelain", "--untracked-files=all")
    license_path = source / "LICENSE"
    if revision != EXPECTED_REVISION:
        parser.error(f"expected {EXPECTED_REVISION}, found {revision}")
    if dirty_paths:
        parser.error("source checkout is dirty")
    if not license_path.is_file():
        parser.error(f"missing license: {license_path}")

    submodule_status = git_output(source, "submodule", "status")
    challenge = {
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]],
            }
        ],
        # Deliberately no test output: this smoke does not load a test label.
        "test": [{"input": [[2, 0], [0, 2]]}],
    }
    majority = ((0, 2), (2, 0))
    alternate = ((2, 0), (0, 2))

    monitor = ResourceMonitor(include_nvidia=False).start()
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from arclib.arc import Task, make_submission
        from arclib.voting import get_all_type_of_votingsv2

        monitor.sample()
        tasks = Task.read_tasks_from_dict(challenge, test=True)
        task = tasks[0]
        task.name = "synthetic-0"
        roundtrip = Task.deserialize(task.serialize(), test=True)
        top_three = get_all_type_of_votingsv2(
            [majority, majority, majority, alternate]
        )
        submission = make_submission(
            [task], [[np.asarray(top_three[0]), np.asarray(alternate)]]
        )
        monitor.sample()
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_dont_write_bytecode
        resources = monitor.stop().to_dict()

    expected_submission = {
        "synthetic": [
            {
                "attempt_1": [[0, 2], [2, 0]],
                "attempt_2": [[2, 0], [0, 2]],
            }
        ]
    }
    checks = {
        "one_task_deserialized": len(tasks) == 1,
        "training_pair_preserved": bool(
            np.array_equal(task.train_examples[0].output, [[0, 1], [1, 0]])
        ),
        "test_output_is_input_placeholder": bool(
            np.array_equal(task.test_example.output, task.test_example.input)
        ),
        "roundtrip_preserved": bool(
            roundtrip.name == task.name
            and roundtrip.train_examples == task.train_examples
            and roundtrip.test_example == task.test_example
        ),
        "majority_vote_selected": top_three[0] == majority,
        "submission_schema_exact": submission == expected_submission,
    }
    passed = all(checks.values())
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "marc",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_marc_components",
        "status": "passed" if passed else "failed",
        "scope": "arc-native-task-submission-and-voting-component",
        "started_at_utc": resources["started_at_utc"],
        "ended_at_utc": resources["ended_at_utc"],
        "source": {
            "path": str(source),
            "expected_revision": EXPECTED_REVISION,
            "observed_revision": revision,
            "dirty_paths": [],
            "files_exercised": ["arclib/arc.py", "arclib/voting.py"],
            "torchtune_submodule_status": submodule_status,
        },
        "license": {
            "spdx": "MIT",
            "path": "LICENSE",
            "sha256": sha256_file(license_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "result": {
            "checks": checks,
            "synthetic_training_pairs": 1,
            "synthetic_test_inputs": 1,
            "test_labels_loaded": 0,
            "vote_input_count": 4,
            "top_vote": [list(row) for row in top_three[0]],
            "submission_sha256": sha256_json(submission),
        },
        "resources": resources,
        "execution_policy": {
            "model_checkpoint_loaded": False,
            "api_called": False,
            "gpu_requested": False,
            "generated_code_executed": False,
            "benchmark_data_loaded": False,
            "test_labels_loaded": False,
        },
        "limitations": [
            "Uses one fixed synthetic task, not an ARC benchmark split.",
            "Does not run MARC training-time test-time training or inference.",
            "Does not load the 8B model, torchtune, vLLM, or any checkpoint.",
            "The pinned third_party/torchtune submodule is not initialized in the source-only checkout.",
        ],
        "claim_boundary": (
            "This validates MARC's ARC-native task representation, two-attempt "
            "submission formatting, and fixed-candidate voting only; it is not a "
            "solver run or ARC-AGI score."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
