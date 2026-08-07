#!/usr/bin/env python3
"""Run a zero-dollar SOAR source/data/helper smoke without executing programs."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.resources import ResourceMonitor


LOCKED_REVISION = "8ed0890b60b647f4ca8582b30f6dbc2c709ff443"
CHALLENGE_RELATIVE_PATH = Path(
    "arc-prize-2024/arc-agi_evaluation_challenges.json"
)


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
    completed = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_grid(value: object) -> None:
    if not isinstance(value, list) or not value or len(value) > 30:
        raise ValueError("grid must have between 1 and 30 rows")
    if not all(isinstance(row, list) for row in value):
        raise ValueError("grid rows must be lists")
    widths = {len(row) for row in value}
    if len(widths) != 1:
        raise ValueError("grid rows must be lists with equal widths")
    width = next(iter(widths))
    if not 1 <= width <= 30:
        raise ValueError("grid must have between 1 and 30 columns")
    for row in value:
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise ValueError("grid cells must be integer colors 0 through 9")


def challenge_summary(challenges: object) -> dict[str, int]:
    if not isinstance(challenges, dict):
        raise ValueError("challenge JSON root is not an object")
    train_pairs = 0
    test_pairs = 0
    train_output_keys = 0
    test_output_keys = 0
    grids = 0
    for task_id, task in challenges.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise ValueError("malformed task entry")
        train = task.get("train")
        test = task.get("test")
        if not isinstance(train, list) or not isinstance(test, list):
            raise ValueError(f"task {task_id} lacks train/test lists")
        for split_name, pairs in (("train", train), ("test", test)):
            for pair in pairs:
                if not isinstance(pair, dict) or "input" not in pair:
                    raise ValueError(f"task {task_id} has malformed {split_name} pair")
                validate_grid(pair["input"])
                grids += 1
                if "output" in pair:
                    validate_grid(pair["output"])
                    grids += 1
                    if split_name == "train":
                        train_output_keys += 1
                    else:
                        test_output_keys += 1
        train_pairs += len(train)
        test_pairs += len(test)
    return {
        "task_count": len(challenges),
        "train_pair_count": train_pairs,
        "test_pair_count": test_pairs,
        "train_output_key_count": train_output_keys,
        "test_output_key_count": test_output_keys,
        "validated_grid_count": grids,
    }


def raw_exec_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exec"
    )


def import_fixed_preprocess(path: Path) -> tuple[ModuleType, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef)
    unexpected = [type(node).__name__ for node in tree.body if not isinstance(node, allowed)]
    if unexpected:
        raise RuntimeError(f"unexpected preprocess top-level statements: {unexpected}")
    imported_names = sorted(
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    if imported_names != ["copy", "json"]:
        raise RuntimeError(f"unexpected preprocess imports: {imported_names}")
    specification = importlib.util.spec_from_file_location(
        "_locked_soar_preprocess", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not create preprocess import specification")
    module = importlib.util.module_from_spec(specification)
    sys.dont_write_bytecode = True
    specification.loader.exec_module(module)
    return module, imported_names


class TrustedArray:
    """Minimal deterministic stand-in for the helper's array protocol."""

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def astype(self, target: object) -> TrustedArray:
        if target is not int:
            raise TypeError("the smoke only permits conversion to int")
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            os.environ.get("ARC_SOAR_SOURCE", "/root/arc-paper-assets/sources/soar")
        ),
    )
    parser.add_argument("--expected-revision", default=LOCKED_REVISION)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "soar"
        / "20260806-zero-dollar-source-data-smoke",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    monitor = ResourceMonitor()
    monitor.start()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "soar",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_soar",
        "status": "failed",
        "scope": "locked-source, bundled-label-free-challenge, and fixed-helper only",
        "started_at_utc": utc_now(),
        "source": {
            "path": str(source),
            "expected_revision": args.expected_revision,
            "observed_revision": None,
            "dirty_paths": None,
        },
        "controls": {
            "estimated_external_cost_usd": 0.0,
            "network_or_api_calls": False,
            "downloads": False,
            "gpu_requested": False,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "model_or_checkpoint_loaded": False,
            "training_executed": False,
            "benchmark_executed": False,
            "predictions_generated": False,
            "test_labels_loaded": False,
            "generated_or_untrusted_code_executed": False,
        },
        "limitations": [
            "This is not a solver, model, checkpoint, accuracy, or paper-result reproduction.",
            "Only the bundled ARC-AGI-1 evaluation challenge JSON was opened; no solution JSON was opened.",
            "Only one fixed, locked preprocessing helper was called on synthetic trusted values.",
            "The upstream program-execution modules were inspected as text/AST and were never imported or called.",
            "Resource CPU and RSS measurements cover this Python process only; git subprocesses are excluded.",
        ],
    }

    try:
        if not (source / ".git").is_dir():
            raise FileNotFoundError(f"source checkout not found: {source}")
        revision = git_output(source, "rev-parse", "HEAD")
        dirty_paths = git_output(source, "status", "--porcelain").splitlines()
        record["source"] = {
            "path": str(source),
            "expected_revision": args.expected_revision,
            "observed_revision": revision,
            "dirty_paths": dirty_paths,
        }
        if revision != args.expected_revision:
            raise RuntimeError(
                f"source revision {revision} does not match {args.expected_revision}"
            )
        if dirty_paths:
            raise RuntimeError("source checkout is dirty")
        monitor.sample()

        license_path = source / "LICENSE.md"
        license_text = license_path.read_text(encoding="utf-8")
        if not license_text.startswith("The MIT License (MIT)\n"):
            raise RuntimeError("locked root license no longer declares MIT")
        record["license"] = {
            "declared_identifier": "MIT",
            "path": license_path.relative_to(source).as_posix(),
            "sha256": sha256_file(license_path),
            "scope": "root license text identifier; not legal advice",
        }

        challenge_path = source / CHALLENGE_RELATIVE_PATH
        with challenge_path.open(encoding="utf-8") as handle:
            challenges = json.load(handle)
        summary = challenge_summary(challenges)
        if summary["task_count"] != 400:
            raise RuntimeError(f"unexpected ARC-AGI-1 task count: {summary['task_count']}")
        if summary["test_output_key_count"] != 0:
            raise RuntimeError("challenge file contains test outputs; label firewall failed")
        record["data_firewall"] = {
            "opened_data_paths": [CHALLENGE_RELATIVE_PATH.as_posix()],
            "challenge_sha256": sha256_file(challenge_path),
            "solution_paths_opened": [],
            "test_labels_loaded": False,
            **summary,
        }
        monitor.sample()

        preprocess_path = source / "soar" / "preprocess.py"
        preprocess, imports = import_fixed_preprocess(preprocess_path)
        trusted_input = [
            [TrustedArray([0, 1, 2]), TrustedArray([9, 8, 7])]
        ]
        converted = preprocess.convert_to_list(trusted_input)
        expected = [[[0, 1, 2], [9, 8, 7]]]
        if converted != expected:
            raise RuntimeError(f"unexpected convert_to_list output: {converted!r}")
        record["fixed_trusted_code_probe"] = {
            "module_path": preprocess_path.relative_to(source).as_posix(),
            "module_sha256": sha256_file(preprocess_path),
            "verified_top_level_imports": imports,
            "function_called": "convert_to_list",
            "input_kind": "synthetic in-memory trusted array-protocol stubs",
            "output": converted,
            "generated_code_executed": False,
        }

        dangerous_paths = [
            source / "soar" / "sandbox" / "execute_code.py",
            source / "soar" / "sandbox" / "execute_code_less_safe.py",
        ]
        static_exec_sites = {
            path.relative_to(source).as_posix(): raw_exec_lines(path)
            for path in dangerous_paths
        }
        if not all(static_exec_sites.values()):
            raise RuntimeError("expected upstream raw exec sites were not found")
        record["security_boundary"] = {
            "static_raw_exec_sites": static_exec_sites,
            "files_imported": [],
            "functions_called": [],
            "program_execution_enabled": False,
            "blocker": (
                "Official SOAR candidate checking reaches raw exec; full execution remains disabled "
                "until independently verified OS/container isolation is available."
            ),
        }
        record["fairness_blockers"] = [
            {
                "kind": "public-evaluation-label-access",
                "evidence": "soar/preprocess.py get_dataset names and opens arc-agi_evaluation_solutions.json",
                "consequence": (
                    "The official dataset loader is not eligible for a label-firewalled heldout ARC-AGI-1 "
                    "evaluation without a replacement loader and protocol audit."
                ),
            },
            {
                "kind": "unisolated-generated-code-execution",
                "evidence": static_exec_sites,
                "consequence": "Official program checking was not run on this host.",
            },
        ]
        record["status"] = "passed"
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        usage = monitor.stop()
        record["started_at_utc"] = usage.started_at_utc
        record["ended_at_utc"] = usage.ended_at_utc
        record["resources"] = usage.to_dict()
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "run_json": str(output_directory / "run.json"),
                "resources": record["resources"],
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
