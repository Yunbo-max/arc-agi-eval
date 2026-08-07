#!/usr/bin/env python3
"""Statically validate AgentPrimitives config and organizer schema without models."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi_eval.resources import ResourceMonitor


EXPECTED_REVISION = "b9906548f8e6b79416b43450847a7352aec6e1b9"


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


def static_dataset_calls(tree: ast.AST) -> list[list[str]]:
    calls: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "load_dataset":
            continue
        values: list[str] = []
        for argument in node.args[:2]:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                values.append(argument.value)
        if values:
            calls.append(values)
    return sorted(calls)


def organizer_schema(tree: ast.AST) -> tuple[list[str], list[str], list[str]]:
    methods: list[str] = []
    lower_case_imports: list[str] = []
    primitive_literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Organizer":
            methods = sorted(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("primitives."):
                lower_case_imports.append(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in {"review", "vote", "plan_execute"}:
                primitive_literals.add(node.value)
    return methods, sorted(lower_case_imports), sorted(primitive_literals)


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
    config_path = source / "configs" / "Qwen3.yaml"
    organizer_path = source / "Primitives" / "Organizer.py"
    data_path = source / "data.py"
    readme_path = source / "README.md"
    if revision != EXPECTED_REVISION:
        parser.error(f"expected {EXPECTED_REVISION}, found {revision}")
    if dirty_paths:
        parser.error("source checkout is dirty")
    for required_path in (config_path, organizer_path, data_path, readme_path):
        if not required_path.is_file():
            parser.error(f"missing required source file: {required_path}")

    root_license_files = sorted(
        path.name
        for path in source.iterdir()
        if path.is_file() and "license" in path.name.lower()
    )

    monitor = ResourceMonitor(include_nvidia=False).start()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    organizer_tree = ast.parse(
        organizer_path.read_text(encoding="utf-8"), filename=str(organizer_path)
    )
    data_tree = ast.parse(data_path.read_text(encoding="utf-8"), filename=str(data_path))
    readme = readme_path.read_text(encoding="utf-8")
    monitor.sample()
    methods, lower_case_imports, primitive_literals = organizer_schema(organizer_tree)
    dataset_calls = static_dataset_calls(data_tree)
    required_methods = [
        "_fallback_plan",
        "_organizer_prompt",
        "_parse_plan_json",
        "_validate_or_fallback",
        "plan_only",
        "solve",
    ]
    target_dataset_call = ["allenai/ai2_arc", "ARC-Challenge"]
    case_sensitive_import_mismatch = bool(
        lower_case_imports
        and (source / "Primitives").is_dir()
        and not (source / "primitives").exists()
    )
    run_demo_exists = (source / "run_demo.py").is_file()
    readme_demo_is_future = "Coming Soon" in readme and "run_demo.py" in readme
    monitor.sample()
    resources = monitor.stop().to_dict()

    checks = {
        "config_sections_exact": sorted(config) == [
            "inference",
            "kv_cache",
            "logging",
            "model",
            "token_limits",
        ],
        "qwen3_model_config": (
            config["model"]["name"] == "Qwen3-8B"
            and config["model"]["family"] == "qwen"
            and config["model"]["max_context_length"] == 32768
            and config["model"]["seed"] == 42
        ),
        "kv_cache_config": (
            config["kv_cache"]["enable"] is True
            and config["kv_cache"]["cache_strategy"] == "rolling"
            and config["kv_cache"]["cache_window"] == 8192
        ),
        "organizer_methods_present": set(required_methods).issubset(methods),
        "primitive_types_present": primitive_literals
        == ["plan_execute", "review", "vote"],
        "native_dataset_is_ai2_arc_challenge": target_dataset_call in dataset_calls,
        "case_sensitive_import_mismatch_detected": case_sensitive_import_mismatch,
        "missing_root_license_detected": not root_license_files,
        "future_demo_status_detected": readme_demo_is_future and not run_demo_exists,
    }
    passed = all(checks.values())
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "agent-primitives",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_agent_primitives_components",
        "status": "passed" if passed else "failed",
        "scope": "generic-mas-config-and-organizer-schema-only",
        "started_at_utc": resources["started_at_utc"],
        "ended_at_utc": resources["ended_at_utc"],
        "source": {
            "path": str(source),
            "expected_revision": EXPECTED_REVISION,
            "observed_revision": revision,
            "dirty_paths": [],
            "files_parsed": ["configs/Qwen3.yaml", "Primitives/Organizer.py", "data.py"],
            "file_sha256": {
                "configs/Qwen3.yaml": sha256_file(config_path),
                "Primitives/Organizer.py": sha256_file(organizer_path),
                "data.py": sha256_file(data_path),
            },
        },
        "license": {
            "status": "unspecified-no-root-license-file",
            "spdx": None,
            "root_license_files": root_license_files,
        },
        "environment": {
            "python": platform.python_version(),
            "pyyaml": yaml.__version__,
        },
        "result": {
            "checks": checks,
            "config_model": config["model"],
            "config_token_limits": config["token_limits"],
            "organizer_methods": methods,
            "organizer_primitive_types": primitive_literals,
            "lower_case_primitive_imports": lower_case_imports,
            "source_directory_spelling": "Primitives",
            "static_dataset_calls": dataset_calls,
            "native_arc_reference": "AI2 ARC-Challenge multiple-choice QA",
            "arc_agi_native": False,
            "arc_agi_score_eligible": False,
            "run_demo_exists": run_demo_exists,
            "runnable_pipeline_validated": False,
        },
        "resources": resources,
        "execution_policy": {
            "upstream_python_imported": False,
            "model_checkpoint_loaded": False,
            "api_called": False,
            "gpu_requested": False,
            "generated_code_executed": False,
            "benchmark_data_loaded": False,
            "test_labels_loaded": False,
        },
        "blockers": [
            "No root LICENSE file specifies redistribution or execution terms.",
            "Organizer.py imports lowercase primitives.* while the checkout contains uppercase Primitives/; this fails on case-sensitive filesystems.",
            "The README marks the end-to-end run_demo.py as Coming Soon and that file is absent.",
            "Organizer construction immediately requests Qwen/Qwen3-8B and requires the heavyweight model stack.",
        ],
        "limitations": [
            "This is static AST/YAML validation; no upstream Python module was imported.",
            "AI2 ARC-Challenge is multiple-choice science QA, not ARC-AGI grids.",
            "No dataset examples, answers, model, tokenizer, primitive execution, or inference was run.",
            "An ARC-AGI adapter would be a new transfer experiment, not a native reproduction.",
        ],
        "claim_boundary": (
            "This validates only the generic MAS config and Organizer source schema "
            "and persistently records current runnable-pipeline blockers. It is not "
            "an ARC-AGI component, solver run, or score."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
