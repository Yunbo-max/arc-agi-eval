#!/usr/bin/env python3
"""Validate LatentMAS's native AI2 ARC prompt and agent schema without inference."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi_eval.resources import ResourceMonitor


EXPECTED_REVISION = "9a9e4d331eb11430bd9e64754c6b252b06d73031"


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


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not construct module specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def static_dataset_calls(path: Path) -> list[list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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


def valid_messages(messages: object, question: str) -> bool:
    return bool(
        isinstance(messages, list)
        and len(messages) == 2
        and all(isinstance(message, dict) for message in messages)
        and [message.get("role") for message in messages] == ["system", "user"]
        and all(isinstance(message.get("content"), str) for message in messages)
        and question in messages[1]["content"]
    )


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
    prompts_path = source / "prompts.py"
    agents_path = source / "methods" / "__init__.py"
    data_path = source / "data.py"
    license_path = source / "LICENSE"
    if revision != EXPECTED_REVISION:
        parser.error(f"expected {EXPECTED_REVISION}, found {revision}")
    if dirty_paths:
        parser.error("source checkout is dirty")
    for required_path in (prompts_path, agents_path, data_path, license_path):
        if not required_path.is_file():
            parser.error(f"missing required source file: {required_path}")

    question = "Which fixed option is the synthetic answer? A: red B: blue C: green D: gray"
    prompt_args = SimpleNamespace(model_name="Qwen/Qwen3-4B", task="arc_challenge")

    monitor = ResourceMonitor(include_nvidia=False).start()
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        prompts = load_module("_latentmas_prompts_smoke", prompts_path)
        agents = load_module("_latentmas_agents_smoke", agents_path)
        monitor.sample()
        roles = [agent.role for agent in agents.default_agents()]
        sequential = {
            role: prompts.build_agent_message_sequential_latent_mas(
                role,
                question,
                method="latent_mas",
                args=prompt_args,
            )
            for role in roles
        }
        hierarchical = {
            role: prompts.build_agent_message_hierarchical_latent_mas(
                role,
                question,
                method="latent_mas",
                args=prompt_args,
            )
            for role in roles
        }
        dataset_calls = static_dataset_calls(data_path)
        monitor.sample()
    finally:
        sys.modules.pop("_latentmas_prompts_smoke", None)
        sys.modules.pop("_latentmas_agents_smoke", None)
        sys.dont_write_bytecode = previous_dont_write_bytecode
        resources = monitor.stop().to_dict()

    target_dataset_call = ["allenai/ai2_arc", "ARC-Challenge"]
    checks = {
        "default_roles_exact": roles == ["planner", "critic", "refiner", "judger"],
        "sequential_message_schema": all(
            valid_messages(messages, question) for messages in sequential.values()
        ),
        "hierarchical_message_schema": all(
            valid_messages(messages, question) for messages in hierarchical.values()
        ),
        "sequential_judger_is_multiple_choice": (
            "A,B,C,D" in sequential["judger"][1]["content"]
        ),
        "hierarchical_judger_is_multiple_choice": (
            "A,B,C,D" in hierarchical["judger"][1]["content"]
        ),
        "native_dataset_is_ai2_arc_challenge": target_dataset_call in dataset_calls,
    }
    passed = all(checks.values())
    message_summary = {
        "roles": roles,
        "sequential_user_lengths": {
            role: len(messages[1]["content"])
            for role, messages in sequential.items()
        },
        "hierarchical_user_lengths": {
            role: len(messages[1]["content"])
            for role, messages in hierarchical.items()
        },
    }
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "latentmas",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_latentmas_components",
        "status": "passed" if passed else "failed",
        "scope": "native-ai2-arc-prompt-and-agent-schema-only",
        "started_at_utc": resources["started_at_utc"],
        "ended_at_utc": resources["ended_at_utc"],
        "source": {
            "path": str(source),
            "expected_revision": EXPECTED_REVISION,
            "observed_revision": revision,
            "dirty_paths": [],
            "files_exercised": ["prompts.py", "methods/__init__.py", "data.py"],
            "file_sha256": {
                "prompts.py": sha256_file(prompts_path),
                "methods/__init__.py": sha256_file(agents_path),
                "data.py": sha256_file(data_path),
            },
        },
        "license": {
            "spdx": "Apache-2.0",
            "path": "LICENSE",
            "sha256": sha256_file(license_path),
        },
        "environment": {"python": platform.python_version()},
        "result": {
            "checks": checks,
            "message_summary": message_summary,
            "static_dataset_calls": dataset_calls,
            "native_benchmark_identity": "AI2 ARC-Challenge multiple-choice QA",
            "arc_agi_native": False,
            "arc_agi_score_eligible": False,
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
            "ARC-Challenge here means AllenAI AI2 ARC multiple-choice science QA, not ARC-AGI grids.",
            "No dataset examples, answers, or ARC-AGI tasks were loaded.",
            "No Qwen checkpoint, tokenizer, latent KV state, vLLM, or inference path was run.",
            "Adapting LatentMAS to ARC-AGI would be a new transfer experiment, not a native reproduction.",
        ],
        "claim_boundary": (
            "This validates native agent roles and prompt construction for the "
            "repository's AI2 ARC-Challenge configuration. It is not an ARC-AGI "
            "component, solver run, or score."
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
