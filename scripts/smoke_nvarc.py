#!/usr/bin/env python3
"""Run a zero-dollar NVARC source/config/helper smoke without executing code."""

from __future__ import annotations

import argparse
import ast
import configparser
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.resources import ResourceMonitor


LOCKED_REVISION = "846d0198efa752534594e321fc3289fc0a06c657"
EXPECTED_GITLINKS = {
    "ARC-AGI-2": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
    "BARC": "a7b51a6b1ff969da3a78a71c533b6d79a93966e7",
    "ConceptARC": "b22ef526b4656679816b7811e78f55cc24d736d7",
    "MINI-ARC": "792d082c40d496f2f106f63fa7125bb115c8230b",
    "TinyRecursiveModels": "e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9",
    "h-arc": "2983eb8672097cd555685a8d140e2f66e1a3a91a",
    "re-arc": "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce",
}
EXPECTED_TRAIN_PATHS = [
    "data/grids_v15/arc2_training",
    "data/grids_v15/mini",
    "data/grids_v15/concept",
    "data/grids_v15/rearc",
    "data/grids_v15/nvarc_training",
    "data/grids_v15/nvarc_full",
]


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


def root_license_files(source: Path) -> list[str]:
    names = {
        "copying",
        "copying.md",
        "copying.txt",
        "license",
        "license.md",
        "license.txt",
    }
    return sorted(
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name.lower() in names
    )


def parse_gitmodules(source: Path) -> list[dict[str, str]]:
    configuration = configparser.ConfigParser()
    configuration.read(source / ".gitmodules", encoding="utf-8")
    result = []
    for section in configuration.sections():
        if not section.startswith('submodule "'):
            raise RuntimeError(f"unexpected .gitmodules section: {section}")
        result.append(
            {
                "name": section.removeprefix('submodule "').removesuffix('"'),
                "path": configuration[section]["path"],
                "url": configuration[section]["url"],
            }
        )
    return sorted(result, key=lambda item: item["path"])


def gitlinks(source: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in git_output(source, "ls-tree", "HEAD:external").splitlines():
        metadata, name = line.split("\t", 1)
        mode, kind, revision = metadata.split()
        if mode != "160000" or kind != "commit":
            raise RuntimeError(f"unexpected external tree entry: {line}")
        result[name] = revision
    return result


def scalar(text: str, key: str) -> str:
    matches = re.findall(rf"^\s*{re.escape(key)}:\s*([^#\n]+?)\s*(?:#.*)?$", text, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {key} scalar, found {len(matches)}")
    return matches[0].strip().strip('"')


def training_paths(text: str) -> list[str]:
    match = re.search(
        r"^\s{2}train_dataset_path:\s*$\n(?P<body>(?:\s{4}-[^\n]+\n)+)",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("could not locate train_dataset_path list")
    return [line.split("-", 1)[1].strip() for line in match.group("body").splitlines()]


def raw_exec_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exec"
    )


def import_fixed_puzzle(path: Path) -> tuple[ModuleType, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)
    unexpected = [type(node).__name__ for node in tree.body if not isinstance(node, allowed)]
    if unexpected:
        raise RuntimeError(f"unexpected puzzle top-level statements: {unexpected}")
    top_level_calls = sorted(
        node.lineno
        for statement in tree.body
        if not isinstance(statement, ast.FunctionDef)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    )
    if top_level_calls:
        raise RuntimeError(f"puzzle module has top-level calls: {top_level_calls}")
    imported_names = sorted(
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    expected_imports = ["io", "json", "numpy", "os", "signal"]
    if imported_names != expected_imports:
        raise RuntimeError(f"unexpected puzzle imports: {imported_names}")
    specification = importlib.util.spec_from_file_location("_locked_nvarc_puzzle", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not create puzzle import specification")
    module = importlib.util.module_from_spec(specification)
    sys.dont_write_bytecode = True
    specification.loader.exec_module(module)
    return module, imported_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            os.environ.get("ARC_NVARC_SOURCE", "/root/arc-paper-assets/sources/nvarc")
        ),
    )
    parser.add_argument("--expected-revision", default=LOCKED_REVISION)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "nvarc"
        / "20260806-zero-dollar-component-source-smoke",
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
        "method_id": "nvarc",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_nvarc",
        "status": "failed",
        "scope": "locked source, component wiring/config, and fixed trusted helper only",
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
            "submodules_initialized": False,
            "gpu_requested": False,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "model_or_checkpoint_loaded": False,
            "training_executed": False,
            "benchmark_executed": False,
            "arc_data_loaded": False,
            "test_labels_loaded": False,
            "predictions_generated": False,
            "generated_or_untrusted_code_executed": False,
        },
        "limitations": [
            "This is not the NVARC ensemble, a solver, a model/checkpoint load, or an accuracy reproduction.",
            "The seven external gitlinks were verified but their repositories and datasets were not initialized or downloaded.",
            "YAML was inspected as locked text; NeMo/Megatron training dependencies were not installed or imported.",
            "Only fixed locked helper functions were called on synthetic trusted values; returned code-like text was not executed.",
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

        licenses = root_license_files(source)
        record["license"] = {
            "root_license_files": licenses,
            "status": "not-identified-at-repository-root" if not licenses else "identified",
            "scope": "repository-root filename audit only; not legal advice",
        }

        modules = parse_gitmodules(source)
        observed_gitlinks = gitlinks(source)
        if len(modules) != 7:
            raise RuntimeError(f"unexpected submodule count: {len(modules)}")
        if observed_gitlinks != EXPECTED_GITLINKS:
            raise RuntimeError(f"unexpected gitlinks: {observed_gitlinks}")
        module_paths = {Path(item["path"]).name for item in modules}
        if module_paths != set(EXPECTED_GITLINKS):
            raise RuntimeError(".gitmodules paths do not match locked gitlinks")
        record["component_wiring"] = {
            "gitmodules_sha256": sha256_file(source / ".gitmodules"),
            "submodule_count": len(modules),
            "submodules": modules,
            "locked_gitlinks": observed_gitlinks,
            "initialized_or_downloaded_by_smoke": False,
        }

        config_path = source / "ARChitects" / "sft_mg.yaml"
        config_text = config_path.read_text(encoding="utf-8")
        train_paths = training_paths(config_text)
        config_summary = {
            "model_name": scalar(config_text, "model_name"),
            "max_num_steps": int(scalar(config_text, "max_num_steps")),
            "tensor_model_parallel_size": int(
                scalar(config_text, "tensor_model_parallel_size")
            ),
            "gpus_per_node": int(scalar(config_text, "gpus_per_node")),
            "num_nodes": int(scalar(config_text, "num_nodes")),
            "train_dataset_paths": train_paths,
            "val_dataset_path": scalar(config_text, "val_dataset_path"),
        }
        expected_summary = {
            "model_name": "/models/Qwen3-4B-Thinking-2507",
            "max_num_steps": 10000,
            "tensor_model_parallel_size": 8,
            "gpus_per_node": 8,
            "num_nodes": 1,
            "train_dataset_paths": EXPECTED_TRAIN_PATHS,
            "val_dataset_path": "data/grids_v15/arc2_evaluation6",
        }
        if config_summary != expected_summary:
            raise RuntimeError(f"locked SFT configuration changed: {config_summary}")
        record["configuration_probe"] = {
            "path": config_path.relative_to(source).as_posix(),
            "sha256": sha256_file(config_path),
            **config_summary,
            "configuration_executed": False,
        }
        monitor.sample()

        build_path = source / "SDG" / "scripts" / "build_datasets.py"
        build_text = build_path.read_text(encoding="utf-8")
        evaluation_source = 'convert_arc_to_messages("external/ARC-AGI-2/data/evaluation/*.json", num_samples=6)'
        evaluation_target = 'save_to_disk(f"{output_path}/arc2_evaluation6")'
        if evaluation_source not in build_text or evaluation_target not in build_text:
            raise RuntimeError("ARC-AGI-2 evaluation validation provenance changed")
        record["public_evaluation_provenance"] = {
            "path": build_path.relative_to(source).as_posix(),
            "sha256": sha256_file(build_path),
            "source_glob": "external/ARC-AGI-2/data/evaluation/*.json",
            "derived_dataset": "data/grids_v15/arc2_evaluation6",
            "configured_role": "validation",
            "data_opened_by_smoke": False,
        }

        puzzle_path = source / "SDG" / "scripts" / "puzzle.py"
        exec_lines = raw_exec_lines(puzzle_path)
        if not exec_lines:
            raise RuntimeError("expected upstream raw exec site was not found")
        puzzle, imports = import_fixed_puzzle(puzzle_path)
        import numpy as np

        valid_grid = puzzle.validate_and_convert_grid(
            np.asarray([[0, 1, 9], [3, 4, 5]], dtype=np.int64)
        )
        invalid_grid = puzzle.validate_and_convert_grid(
            np.zeros((31, 1), dtype=np.int64)
        )
        if valid_grid != [[0, 1, 9], [3, 4, 5]] or invalid_grid is not None:
            raise RuntimeError("unexpected validate_and_convert_grid behavior")
        filtered = puzzle.filter_input_tests(
            {
                "test_identity": "def test_identity(input_grid):\n    return input_grid",
                "test_empty": "def test_empty():\n    return True",
                "helper": "def helper(input_grid):\n    return input_grid",
            }
        )
        expected_filtered = "\ntest_identity(input_grid)\ntest_empty()"
        if filtered != expected_filtered:
            raise RuntimeError(f"unexpected filter_input_tests result: {filtered!r}")
        record["fixed_trusted_code_probe"] = {
            "module_path": puzzle_path.relative_to(source).as_posix(),
            "module_sha256": sha256_file(puzzle_path),
            "verified_top_level_imports": imports,
            "numpy_version": np.__version__,
            "functions_called": ["validate_and_convert_grid", "filter_input_tests"],
            "valid_grid_output": valid_grid,
            "invalid_oversize_grid_rejected": invalid_grid is None,
            "filtered_test_calls": filtered,
            "filtered_test_calls_executed": False,
            "raw_exec_function_called": False,
        }
        record["security_boundary"] = {
            "static_raw_exec_sites": {
                puzzle_path.relative_to(source).as_posix(): exec_lines
            },
            "execute_code_called": False,
            "blocker": (
                "SDG execute_code uses raw exec with a signal timeout but no filesystem/network "
                "isolation; generated program execution remains disabled."
            ),
        }
        record["fairness_blockers"] = [
            {
                "kind": "missing-top-level-license",
                "evidence": {"root_license_files": licenses},
                "consequence": "Redistribution and executable reuse require a separate license review.",
            },
            {
                "kind": "public-evaluation-used-for-validation",
                "evidence": {
                    "build_source": "external/ARC-AGI-2/data/evaluation/*.json",
                    "validation_path": "data/grids_v15/arc2_evaluation6",
                },
                "consequence": (
                    "A score on the same public ARC-AGI-2 evaluation tasks is not a clean heldout "
                    "post-selection metric; hidden Kaggle evaluation must be kept separate."
                ),
            },
            {
                "kind": "unavailable-provenance-assets",
                "evidence": "nvarc_training/nvarc_full pair assets and model checkpoints are not in the source checkout",
                "consequence": "Training overlap and checkpoint provenance cannot be independently audited here.",
            },
            {
                "kind": "unisolated-generated-code-execution",
                "evidence": {"raw_exec_lines": exec_lines},
                "consequence": "The SDG program execution stage was not run on this host.",
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
