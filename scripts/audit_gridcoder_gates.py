#!/usr/bin/env python3
"""Audit GridCoder2024 source, dependency, label, artifact, and CPU gates.

The audit is deliberately static: it never imports upstream code, opens ARC
task data, loads a checkpoint, initializes CUDA, or accesses the network.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
import tokenize
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LICENSE_NAMES = {
    "copying",
    "copying.md",
    "copying.txt",
    "license",
    "license.md",
    "license.txt",
    "notice",
    "notice.md",
    "notice.txt",
}
EXPECTED_BLOCKER_IDS = [
    "upstream-provenance",
    "source-license",
    "dependency-api-and-data",
    "checkpoint-provenance",
    "label-firewall",
    "cpu-entrypoint",
    "benchmark-coverage",
]
EXPECTED_CONTROL_KEYS = {
    "network_allowed",
    "gpu_allowed",
    "arc_data_allowed",
    "checkpoint_byte_read_allowed",
    "checkpoint_load_allowed",
    "upstream_import_allowed",
    "upstream_execution_allowed",
    "solver_execution_allowed",
}
EXPECTED_CHECKPOINT_PATHS = [
    "external/GridCoder2024/model_full.pth",
    "/usr/paper-assets/arc/sources/gridcoder2024/model_full.pth",
    "/usr/paper-assets/arc/checkpoints/gridcoder2024/model_full.pth",
    "/model/gridcoder2024/model_full.pth",
]
EXPECTED_SOURCE_TRACKED_PATHS = [
    "Hodel_primitives_atomic.py",
    "Hodel_primitives_atomicV2.py",
    "Hodel_primitives_atomicV3.py",
    "Hodel_primitives_full_trainingV2.py",
    "README.md",
    "datasets/__init__.py",
    "datasets/generators/merge_split_generator_atomic.py",
    "datasets/generators/object_recombiner_generator.py",
    "datasets/generators/object_selector_generator.py",
    "datasets/generators/tiling_generator_atomic.py",
    "datasets/generators/trivial_objectness_generator.py",
    "datasets/generators/windowing_generator.py",
    "datasets/similarity_dataset_p_star_atomic.py",
    "datasets/task_generator.py",
    "generate_training_data_full.py",
    "model/LVM.py",
    "model/heuristic.py",
    "search/p_star.py",
    "search/p_star_muzero.py",
    "search/p_star_superposition.py",
    "search/program_interpreter.py",
    "search/program_interpreter_V3.py",
    "test_gridcoder.py",
    "train_full.py",
    "utils/grid_utils.py",
    "utils/heuristics.py",
    "utils/program_path_utils.py",
    "utils/sequence_utils.py",
    "utils/validation_utils.py",
]
EXPECTED_ARC_GYM_TRACKED_PATHS = [
    "arc_evaluation_dataset.py",
    "dataset.py",
    "grid_sampling/grid_sampler.py",
    "grid_sampling/object_grid_generation.py",
    "utils/object_detector.py",
    "utils/tokenization.py",
    "utils/visualization.py",
]
EXPECTED_SOURCE_CRITICAL_PATHS = [
    "README.md",
    "test_gridcoder.py",
    "model/LVM.py",
    "search/p_star_superposition.py",
    "search/program_interpreter_V3.py",
    "Hodel_primitives_atomicV3.py",
    "train_full.py",
    "generate_training_data_full.py",
]
EXPECTED_ARC_GYM_CRITICAL_PATHS = [
    "ARC_gym/arc_evaluation_dataset.py",
    "ARC_gym/dataset.py",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def run_git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def smoke_python_tree_digest(root: Path, paths: list[Path]) -> str:
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def canonical_tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def syntax_failures(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for path in paths:
        try:
            with tokenize.open(path) as handle:
                ast.parse(handle.read(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            failures.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return failures


def root_license_files(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name.lower() in LICENSE_NAMES
    )


def critical_file_checks(
    root: Path, declarations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for declaration in declarations:
        try:
            path = contained_regular_file(root, declaration["path"])
        except FileNotFoundError:
            path = root / declaration["path"]
            regular_file = False
        else:
            regular_file = True
        observed = sha256(path) if regular_file else None
        checks.append(
            {
                "path": declaration["path"],
                "expected_sha256": declaration["sha256"],
                "observed_sha256": observed,
                "regular_non_symlink_file": regular_file,
                "matched": observed == declaration["sha256"],
            }
        )
    return checks


def imported_modules(path: Path) -> list[dict[str, Any]]:
    with tokenize.open(path) as handle:
        tree = ast.parse(handle.read(), filename=str(path))
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            imports.append({"module": node.module, "line": node.lineno})
    return sorted(imports, key=lambda item: (item["line"], str(item["module"])))


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def test_output_flow(path: Path) -> dict[str, Any]:
    with tokenize.open(path) as handle:
        tree = ast.parse(handle.read(), filename=str(path))
    output_subscripts_in_test_loop: list[int] = []
    test_output_return_lines: list[int] = []
    all_tasks_append_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Subscript):
            continue
        if not isinstance(node.iter.value, ast.Name) or node.iter.value.id != "content":
            continue
        if _constant_string(node.iter.slice) != "test":
            continue
        for descendant in ast.walk(node):
            if (
                isinstance(descendant, ast.Subscript)
                and _constant_string(descendant.slice) == "output"
            ):
                output_subscripts_in_test_loop.append(descendant.lineno)

    yq_assignment_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            if any(
                isinstance(descendant, ast.Name)
                and descendant.id == "test_output_grids"
                for descendant in ast.walk(node.value)
            ):
                test_output_return_lines.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
            and node.func.value.attr == "all_tasks"
            and any(
                isinstance(descendant, ast.Name)
                and descendant.id == "test_output"
                for argument in node.args
                for descendant in ast.walk(argument)
            )
        ):
            all_tasks_append_lines.append(node.lineno)
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            for descendant in ast.walk(target):
                if (
                    isinstance(descendant, ast.Subscript)
                    and _constant_string(descendant.slice) == "yq"
                ):
                    yq_assignment_lines.append(descendant.lineno)

    lines = path.read_text(encoding="utf-8").splitlines()
    append_lines = [
        index
        for index, line in enumerate(lines, start=1)
        if "test_output_grids.append" in line
    ]
    return {
        "test_output_subscript_lines": sorted(set(output_subscripts_in_test_loop)),
        "test_output_append_lines": append_lines,
        "test_output_return_lines": sorted(set(test_output_return_lines)),
        "all_tasks_append_lines": sorted(set(all_tasks_append_lines)),
        "yq_assignment_lines": sorted(set(yq_assignment_lines)),
        "flow_detected": bool(
            output_subscripts_in_test_loop
            and append_lines
            and test_output_return_lines
            and all_tasks_append_lines
            and yq_assignment_lines
        ),
    }


def runner_static_controls(path: Path) -> dict[str, Any]:
    with tokenize.open(path) as handle:
        tree = ast.parse(handle.read(), filename=str(path))
    torch_loads: list[dict[str, Any]] = []
    cuda_literal_lines: list[int] = []
    model_train_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "cuda":
            cuda_literal_lines.append(node.lineno)
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        ):
            torch_loads.append(
                {
                    "line": node.lineno,
                    "weights_only_declared": any(
                        keyword.arg == "weights_only" for keyword in node.keywords
                    ),
                }
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
            model_train_lines.append(node.lineno)
    text = path.read_text(encoding="utf-8")
    return {
        "torch_load_calls": sorted(torch_loads, key=lambda item: item["line"]),
        "cuda_literal_lines": sorted(set(cuda_literal_lines)),
        "model_train_call_lines": sorted(set(model_train_lines)),
        "hardcoded_cuda_detected": bool(cuda_literal_lines),
        "checkpoint_load_detected": bool(torch_loads),
        "all_torch_load_calls_declare_weights_only": bool(torch_loads)
        and all(item["weights_only_declared"] for item in torch_loads),
        "challenge_only_candidate_branch_detected": (
            "arc-agi_test_challenges.json" in text
            and "if args.task == 'Kaggle'" in text
            and "output_grid = input_grid" in text
        ),
    }


def readme_coverage(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    entries = re.findall(
        r"^- ([0-9a-f]{8})\.json:\s*(SUCCESS|FAILURE)(?:\s|$)",
        text,
        re.MULTILINE,
    )
    task_ids = [task_id for task_id, _ in entries]
    return {
        "task_count": len(entries),
        "unique_task_count": len(set(task_ids)),
        "success_count": sum(status == "SUCCESS" for _, status in entries),
        "failure_count": sum(status == "FAILURE" for _, status in entries),
        "task_ids_sha256": hashlib.sha256(
            "\n".join(task_ids).encode("utf-8")
        ).hexdigest(),
    }


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{field} keys mismatch: expected {sorted(expected)}, found {sorted(observed)}"
        )
    return value


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def require_git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def require_safe_relative_file(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError(f"{field} must be a contained relative file path")
    return path.as_posix()


def contained_regular_file(root: Path, relative: str) -> Path:
    relative_path = Path(require_safe_relative_file(relative, "contained file"))
    current = root.absolute()
    root_mode = os.lstat(current).st_mode
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise ValueError(f"content root is not a regular directory: {root}")
    for index, part in enumerate(relative_path.parts):
        current = current / part
        mode = os.lstat(current).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"contained path has a symlink component: {relative}")
        if index < len(relative_path.parts) - 1:
            if not stat.S_ISDIR(mode):
                raise ValueError(f"contained path parent is not a directory: {relative}")
        elif not stat.S_ISREG(mode):
            raise ValueError(f"contained path is not a regular file: {relative}")
    return current


def contained_directory(root: Path, relative: str) -> Path:
    relative_path = Path(require_safe_relative_file(relative, "contained directory"))
    current = root.absolute()
    for part in relative_path.parts:
        current = current / part
        mode = os.lstat(current).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"directory path has a symlink/non-directory component: {relative}")
    return current


def validate_critical_files(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    paths: list[str] = []
    for index, item in enumerate(value):
        declaration = require_exact_keys(
            item, {"path", "sha256"}, f"{field}[{index}]"
        )
        paths.append(
            require_safe_relative_file(declaration["path"], f"{field}[{index}].path")
        )
        require_sha256(declaration["sha256"], f"{field}[{index}].sha256")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field} contains duplicate paths")
    return value


def validate_config(config: Any) -> dict[str, Any]:
    config = require_exact_keys(
        config,
        {
            "schema_version",
            "config_id",
            "method_id",
            "scope",
            "counted_toward_smoke",
            "expected_blocker_ids",
            "source",
            "arc_gym",
            "checkpoint",
            "coverage",
            "data",
            "prior_architecture_evidence",
            "controls",
        },
        "config",
    )
    if config["schema_version"] != 3:
        raise ValueError("config.schema_version must equal 3")
    if config["config_id"] != "gridcoder2024-source-dependency-label-artifact-gate-v3":
        raise ValueError("unexpected config.config_id")
    if config["method_id"] != "gridcoder2024":
        raise ValueError("config.method_id must equal gridcoder2024")
    if config["scope"] != "source-dependency-label-artifact-gate-audit-only":
        raise ValueError("unexpected config.scope")
    if config["counted_toward_smoke"] is not False:
        raise ValueError("config.counted_toward_smoke must be false")
    if config["expected_blocker_ids"] != EXPECTED_BLOCKER_IDS:
        raise ValueError("config.expected_blocker_ids mismatch")

    source = require_exact_keys(
        config["source"],
        {
            "repository_url",
            "expected_revision",
            "source_lock_path",
            "source_lock_sha256",
            "local_snapshot",
            "prepared_checkout",
            "expected_python_file_count",
            "expected_python_bytes",
            "expected_python_tree_sha256",
            "expected_clean_file_count",
            "expected_clean_tree_sha256",
            "expected_root_tracked_file_count",
            "expected_tracked_paths",
            "expected_root_license_files",
            "expected_independent_git_checkout",
            "expected_prepared_checkout_present",
            "expected_revision_object_present_in_root_git",
            "critical_files",
        },
        "config.source",
    )
    if source["repository_url"] != "https://github.com/SimonOuellette35/GridCoder2024":
        raise ValueError("unexpected config.source.repository_url")
    require_git_sha(source["expected_revision"], "config.source.expected_revision")
    if source["source_lock_path"] != "configs/source_locks.json":
        raise ValueError("unexpected config.source.source_lock_path")
    require_sha256(source["source_lock_sha256"], "config.source.source_lock_sha256")
    if source["local_snapshot"] != "external/GridCoder2024":
        raise ValueError("unexpected config.source.local_snapshot")
    if source["prepared_checkout"] != "/usr/paper-assets/arc/sources/gridcoder2024":
        raise ValueError("unexpected config.source.prepared_checkout")
    for key in (
        "expected_python_file_count",
        "expected_python_bytes",
        "expected_clean_file_count",
        "expected_root_tracked_file_count",
    ):
        require_nonnegative_int(source[key], f"config.source.{key}")
    require_sha256(
        source["expected_python_tree_sha256"],
        "config.source.expected_python_tree_sha256",
    )
    require_sha256(
        source["expected_clean_tree_sha256"],
        "config.source.expected_clean_tree_sha256",
    )
    if source["expected_tracked_paths"] != EXPECTED_SOURCE_TRACKED_PATHS:
        raise ValueError("config.source.expected_tracked_paths mismatch")
    if source["expected_root_license_files"] != []:
        raise ValueError("config.source.expected_root_license_files must be empty")
    for key in (
        "expected_independent_git_checkout",
        "expected_prepared_checkout_present",
        "expected_revision_object_present_in_root_git",
    ):
        if source[key] is not False:
            raise ValueError(f"config.source.{key} must be false")
    validate_critical_files(source["critical_files"], "config.source.critical_files")
    if [item["path"] for item in source["critical_files"]] != (
        EXPECTED_SOURCE_CRITICAL_PATHS
    ):
        raise ValueError("config.source.critical_files path set mismatch")

    arc_gym = require_exact_keys(
        config["arc_gym"],
        {
            "repository_url",
            "local_checkout",
            "expected_revision",
            "expected_python_file_count",
            "expected_python_bytes",
            "expected_python_tree_sha256",
            "expected_tracked_paths",
            "expected_root_license_files",
            "critical_files",
            "required_runner_modules",
            "expected_missing_runner_modules",
            "expected_source_lock_entry_present",
        },
        "config.arc_gym",
    )
    if arc_gym["repository_url"] != "https://github.com/SimonOuellette35/ARC_gym":
        raise ValueError("unexpected config.arc_gym.repository_url")
    if arc_gym["local_checkout"] != "external/ARC_gym":
        raise ValueError("unexpected config.arc_gym.local_checkout")
    require_git_sha(arc_gym["expected_revision"], "config.arc_gym.expected_revision")
    for key in ("expected_python_file_count", "expected_python_bytes"):
        require_nonnegative_int(arc_gym[key], f"config.arc_gym.{key}")
    require_sha256(
        arc_gym["expected_python_tree_sha256"],
        "config.arc_gym.expected_python_tree_sha256",
    )
    if arc_gym["expected_tracked_paths"] != EXPECTED_ARC_GYM_TRACKED_PATHS:
        raise ValueError("config.arc_gym.expected_tracked_paths mismatch")
    if arc_gym["expected_root_license_files"] != []:
        raise ValueError("config.arc_gym.expected_root_license_files must be empty")
    validate_critical_files(arc_gym["critical_files"], "config.arc_gym.critical_files")
    if [item["path"] for item in arc_gym["critical_files"]] != (
        EXPECTED_ARC_GYM_CRITICAL_PATHS
    ):
        raise ValueError("config.arc_gym.critical_files path set mismatch")
    expected_modules = ["ARC_gym/utils/graphs.py", "ARC_gym/utils/batching.py"]
    if arc_gym["required_runner_modules"] != expected_modules:
        raise ValueError("config.arc_gym.required_runner_modules mismatch")
    if arc_gym["expected_missing_runner_modules"] != expected_modules:
        raise ValueError("config.arc_gym.expected_missing_runner_modules mismatch")
    if arc_gym["expected_source_lock_entry_present"] is not False:
        raise ValueError("config.arc_gym.expected_source_lock_entry_present must be false")

    checkpoint = require_exact_keys(
        config["checkpoint"],
        {
            "filename",
            "unverified_prior_remote_metadata",
            "known_local_paths",
            "expected_present_count",
        },
        "config.checkpoint",
    )
    if checkpoint["filename"] != "model_full.pth":
        raise ValueError("unexpected config.checkpoint.filename")
    metadata = require_exact_keys(
        checkpoint["unverified_prior_remote_metadata"],
        {"reported_uncompressed_bytes", "reported_license", "reverified_by_this_audit"},
        "config.checkpoint.unverified_prior_remote_metadata",
    )
    require_nonnegative_int(
        metadata["reported_uncompressed_bytes"],
        "config.checkpoint.unverified_prior_remote_metadata.reported_uncompressed_bytes",
    )
    if not isinstance(metadata["reported_license"], str) or not metadata[
        "reported_license"
    ]:
        raise ValueError("reported checkpoint license note must be non-empty")
    if metadata["reverified_by_this_audit"] is not False:
        raise ValueError("remote metadata must remain explicitly unverified")
    if checkpoint["known_local_paths"] != EXPECTED_CHECKPOINT_PATHS:
        raise ValueError("config.checkpoint.known_local_paths mismatch")
    require_nonnegative_int(
        checkpoint["expected_present_count"],
        "config.checkpoint.expected_present_count",
    )
    if checkpoint["expected_present_count"] != 0:
        raise ValueError("config.checkpoint.expected_present_count must equal zero")

    coverage = require_exact_keys(
        config["coverage"],
        {
            "expected_readme_task_count",
            "expected_readme_success_count",
            "expected_readme_failure_count",
            "native_benchmark",
            "scope",
        },
        "config.coverage",
    )
    for key in (
        "expected_readme_task_count",
        "expected_readme_success_count",
        "expected_readme_failure_count",
    ):
        require_nonnegative_int(coverage[key], f"config.coverage.{key}")
    if coverage["native_benchmark"] != "arc_agi_1" or coverage[
        "scope"
    ] != "declared-dsl-solvable-subset":
        raise ValueError("config.coverage taxonomy mismatch")

    data = require_exact_keys(
        config["data"],
        {"default_evaluation_directory", "expected_default_evaluation_directory_present"},
        "config.data",
    )
    if data["default_evaluation_directory"] != (
        "external/GridCoder2024/ARC/data/evaluation"
    ):
        raise ValueError("unexpected config.data.default_evaluation_directory")
    if data["expected_default_evaluation_directory_present"] is not False:
        raise ValueError("default evaluation directory expectation must be false")

    prior = require_exact_keys(
        config["prior_architecture_evidence"],
        {"run_path", "run_sha256", "runner_path", "runner_sha256"},
        "config.prior_architecture_evidence",
    )
    if prior["run_path"] != (
        "reports/gridcoder2024/20260806-architecture-forward-smoke/run.json"
    ) or prior["runner_path"] != "scripts/smoke_gridcoder.py":
        raise ValueError("unexpected prior architecture evidence path")
    require_sha256(prior["run_sha256"], "config.prior_architecture_evidence.run_sha256")
    require_sha256(
        prior["runner_sha256"], "config.prior_architecture_evidence.runner_sha256"
    )

    controls = require_exact_keys(config["controls"], EXPECTED_CONTROL_KEYS, "config.controls")
    if not all(controls[key] is False for key in EXPECTED_CONTROL_KEYS):
        raise ValueError("all config.controls values must be false")
    return config


def evidence_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    unique_paths = sorted({path.resolve() for path in paths}, key=lambda path: str(path))
    return [
        {
            "path": display_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in unique_paths
    ]


def tracked_allowlist(
    repository: Path,
    pathspec: str,
    content_root: Path,
    expected_relative_paths: list[str],
) -> tuple[list[str], list[Path]]:
    result = run_git(repository, "ls-files", "--", pathspec)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    repository = repository.resolve()
    content_root = content_root.absolute()
    content_prefix = content_root.relative_to(repository)
    relative_paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        repository_relative = Path(require_safe_relative_file(line, "git tracked path"))
        try:
            relative = repository_relative.relative_to(content_prefix).as_posix()
        except ValueError as error:
            raise ValueError(f"tracked path escapes content root: {line}")
        relative_paths.append(relative)
    relative_paths = sorted(relative_paths)
    if relative_paths != expected_relative_paths:
        raise ValueError("Git tracked path allowlist mismatch; aborting before file reads")

    absolute_paths: list[Path] = []
    for relative in relative_paths:
        try:
            candidate = contained_regular_file(content_root, relative)
        except ValueError as error:
            raise ValueError(
                f"tracked allowlist path is not a regular non-symlink file: {relative}: {error}"
            ) from error
        absolute_paths.append(candidate)
    return relative_paths, absolute_paths


def file_inventory(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def assert_forbidden_material_absent(
    checkpoint_observations: list[dict[str, Any]],
    default_evaluation_directory_present: bool,
) -> None:
    if any(item.get("present") is True for item in checkpoint_observations) or (
        default_evaluation_directory_present
    ):
        raise RuntimeError(
            "forbidden checkpoint or ARC evaluation material is present; "
            "aborting before retained-source reads"
        )


def audit(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    config = validate_config(config)
    source_config = config["source"]
    arc_gym_config = config["arc_gym"]
    source = ROOT / "external" / "GridCoder2024"
    arc_gym = ROOT / "external" / "ARC_gym"
    source_lock_path = ROOT / "configs" / "source_locks.json"
    prepared_checkout = Path(source_config["prepared_checkout"])
    default_evaluation_directory = (
        ROOT / "external" / "GridCoder2024" / "ARC" / "data" / "evaluation"
    ).absolute()

    # Fail closed before reading any retained source: forbidden checkpoint/data
    # material is observed only with exists/stat and is never opened.
    checkpoint_paths = [resolve_from_root(value).resolve() for value in EXPECTED_CHECKPOINT_PATHS]
    checkpoint_observations = [
        {
            "path": str(path),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": None,
            "bytes_read": False,
            "loaded": False,
        }
        for path in checkpoint_paths
    ]
    present_checkpoint_count = sum(item["present"] for item in checkpoint_observations)
    default_evaluation_directory_present = default_evaluation_directory.is_dir()
    assert_forbidden_material_absent(
        checkpoint_observations, default_evaluation_directory_present
    )
    source = contained_directory(ROOT, "external/GridCoder2024")
    arc_gym = contained_directory(ROOT, "external/ARC_gym")
    source_lock_path = contained_regular_file(ROOT, "configs/source_locks.json")

    tracked_relative, source_tracked_paths = tracked_allowlist(
        ROOT,
        "external/GridCoder2024",
        source,
        source_config["expected_tracked_paths"],
    )
    source_python_paths = [path for path in source_tracked_paths if path.suffix == ".py"]
    source_python_digest = smoke_python_tree_digest(source, source_python_paths)
    source_clean_digest = canonical_tree_digest(source, source_tracked_paths)
    source_syntax_failures = syntax_failures(source, source_python_paths)
    source_critical = critical_file_checks(source, source_config["critical_files"])
    source_critical_tracked = all(
        item["path"] in tracked_relative for item in source_config["critical_files"]
    )
    source_licenses = root_license_files(source)

    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_lock_entry = source_lock["sources"][config["method_id"]]
    arc_gym_source_lock_present = "arc-gym" in source_lock["sources"]
    source_lock_matches = (
        sha256(source_lock_path) == source_config["source_lock_sha256"]
        and source_lock_entry["url"] == source_config["repository_url"]
        and source_lock_entry["revision"] == source_config["expected_revision"]
    )
    revision_object = run_git(
        ROOT, "cat-file", "-e", f"{source_config['expected_revision']}^{{commit}}"
    )

    arc_relative, arc_tracked_paths = tracked_allowlist(
        arc_gym,
        "ARC_gym",
        arc_gym / "ARC_gym",
        arc_gym_config["expected_tracked_paths"],
    )
    arc_python_paths = [path for path in arc_tracked_paths if path.suffix == ".py"]
    arc_python_digest = smoke_python_tree_digest(arc_gym / "ARC_gym", arc_python_paths)
    arc_syntax_failures = syntax_failures(arc_gym / "ARC_gym", arc_python_paths)
    arc_revision = run_git(arc_gym, "rev-parse", "HEAD")
    arc_tracked_status = run_git(arc_gym, "status", "--porcelain", "--untracked-files=no")
    arc_untracked_status = run_git(arc_gym, "status", "--porcelain")
    arc_critical = critical_file_checks(arc_gym, arc_gym_config["critical_files"])
    arc_licenses = root_license_files(arc_gym)

    required_modules = arc_gym_config["required_runner_modules"]
    missing_modules = sorted(
        module for module in required_modules if not (arc_gym / module).is_file()
    )
    runner_path = source / "test_gridcoder.py"
    runner_imports = imported_modules(runner_path)
    imported_names = {item["module"] for item in runner_imports}
    required_import_names = {"ARC_gym.utils.batching", "ARC_gym.arc_evaluation_dataset"}
    arc_loader_path = arc_gym / "ARC_gym" / "arc_evaluation_dataset.py"
    loader_imports = imported_modules(arc_loader_path)
    loader_import_names = {item["module"] for item in loader_imports}
    label_flow = test_output_flow(arc_loader_path)
    runner_controls = runner_static_controls(runner_path)
    coverage = readme_coverage(source / "README.md")

    prior_config = config["prior_architecture_evidence"]
    prior_run_path = contained_regular_file(ROOT, prior_config["run_path"])
    prior_runner_path = contained_regular_file(ROOT, prior_config["runner_path"])
    prior_evidence_matches = (
        prior_run_path.is_file()
        and sha256(prior_run_path) == prior_config["run_sha256"]
        and prior_runner_path.is_file()
        and sha256(prior_runner_path) == prior_config["runner_sha256"]
    )

    source_snapshot_matches = (
        len(source_python_paths) == source_config["expected_python_file_count"]
        and sum(path.stat().st_size for path in source_python_paths)
        == source_config["expected_python_bytes"]
        and source_python_digest == source_config["expected_python_tree_sha256"]
        and len(source_tracked_paths) == source_config["expected_clean_file_count"]
        and source_clean_digest == source_config["expected_clean_tree_sha256"]
        and len(tracked_relative) == source_config["expected_root_tracked_file_count"]
        and not source_syntax_failures
        and source_critical_tracked
        and all(item["matched"] for item in source_critical)
    )
    source_provenance_observation_matches = (
        (source / ".git").is_dir() == source_config["expected_independent_git_checkout"]
        and prepared_checkout.is_dir() == source_config["expected_prepared_checkout_present"]
        and (revision_object.returncode == 0)
        == source_config["expected_revision_object_present_in_root_git"]
    )
    arc_gym_matches = (
        arc_revision.returncode == 0
        and arc_revision.stdout.strip() == arc_gym_config["expected_revision"]
        and arc_tracked_status.returncode == 0
        and not arc_tracked_status.stdout.strip()
        and len(arc_python_paths) == arc_gym_config["expected_python_file_count"]
        and sum(path.stat().st_size for path in arc_python_paths)
        == arc_gym_config["expected_python_bytes"]
        and arc_python_digest == arc_gym_config["expected_python_tree_sha256"]
        and not arc_syntax_failures
        and all(item["matched"] for item in arc_critical)
    )
    license_observation_matches = (
        source_licenses == source_config["expected_root_license_files"]
        and arc_licenses == arc_gym_config["expected_root_license_files"]
    )
    dependency_observation_matches = (
        missing_modules == sorted(arc_gym_config["expected_missing_runner_modules"])
        and required_import_names.issubset(imported_names)
        and "ARC_gym.utils.graphs" in loader_import_names
        and arc_gym_source_lock_present
        == arc_gym_config["expected_source_lock_entry_present"]
    )
    checkpoint_observation_matches = (
        present_checkpoint_count == config["checkpoint"]["expected_present_count"]
    )
    data_observation_matches = (
        default_evaluation_directory_present
        == config["data"]["expected_default_evaluation_directory_present"]
    )
    coverage_matches = (
        coverage["task_count"] == config["coverage"]["expected_readme_task_count"]
        and coverage["success_count"] == config["coverage"]["expected_readme_success_count"]
        and coverage["failure_count"] == config["coverage"]["expected_readme_failure_count"]
        and coverage["unique_task_count"] == coverage["task_count"]
    )

    evidence_paths = [
        config_path,
        Path(__file__).resolve(),
        source_lock_path,
        prior_run_path,
        prior_runner_path,
        *source_tracked_paths,
        *arc_tracked_paths,
        *[arc_gym / item["path"] for item in arc_gym_config["critical_files"]],
    ]
    manifest = evidence_manifest(evidence_paths)
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    validation = {
        "config_contract_validated": True,
        "source_lock_matches": source_lock_matches,
        "source_snapshot_matches": source_snapshot_matches,
        "source_provenance_observation_matches": source_provenance_observation_matches,
        "arc_gym_snapshot_matches": arc_gym_matches,
        "license_observation_matches": license_observation_matches,
        "dependency_observation_matches": dependency_observation_matches,
        "label_flow_detected": label_flow["flow_detected"],
        "challenge_only_candidate_detected": runner_controls[
            "challenge_only_candidate_branch_detected"
        ],
        "runner_gpu_and_checkpoint_controls_detected": runner_controls[
            "hardcoded_cuda_detected"
        ]
        and runner_controls["checkpoint_load_detected"],
        "checkpoint_observation_matches": checkpoint_observation_matches,
        "data_observation_matches": data_observation_matches,
        "coverage_matches": coverage_matches,
        "prior_architecture_evidence_matches": prior_evidence_matches,
        "static_controls_are_fail_closed": set(config["controls"])
        == EXPECTED_CONTROL_KEYS
        and all(config["controls"][key] is False for key in EXPECTED_CONTROL_KEYS),
    }
    blocker_ids = list(config["expected_blocker_ids"])
    deterministic_observations = {
        "config_sha256": sha256(config_path),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "evidence_manifest_sha256": manifest_digest,
        "source": {
            "source_lock_matches": source_lock_matches,
            "independent_git_checkout": (source / ".git").is_dir(),
            "prepared_checkout_present": prepared_checkout.is_dir(),
            "revision_object_present_in_root_git": revision_object.returncode == 0,
            "tracked_paths": tracked_relative,
            "python_tree_sha256": source_python_digest,
            "clean_tree_sha256": source_clean_digest,
            "root_license_files": source_licenses,
        },
        "arc_gym": {
            "observed_revision": arc_revision.stdout.strip()
            if arc_revision.returncode == 0
            else None,
            "tracked_python_paths": arc_relative,
            "python_tree_sha256": arc_python_digest,
            "missing_runner_modules": missing_modules,
            "source_lock_entry_present": arc_gym_source_lock_present,
            "root_license_files": arc_licenses,
        },
        "label_flow": label_flow,
        "runner_controls": runner_controls,
        "checkpoint_observations": checkpoint_observations,
        "default_evaluation_directory": str(default_evaluation_directory),
        "default_evaluation_directory_present": default_evaluation_directory_present,
        "coverage": coverage,
        "blocker_ids": blocker_ids,
        "validation": validation,
    }
    observation_digest = hashlib.sha256(
        json.dumps(
            deterministic_observations,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": 1,
        "method_id": "gridcoder2024",
        "run_id": "",
        "runner": "scripts.audit_gridcoder_gates",
        "status": "passed" if all(validation.values()) else "failed",
        "status_semantics": "Frozen static blocker observations matched the prospective config; method execution remains blocked.",
        "scope": config["scope"],
        "method_gate_status": "blocked",
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "blocker_ids": blocker_ids,
        "gate_summary": {"blocked": len(blocker_ids), "passed": 0},
        "observation_digest_sha256": observation_digest,
        "config": {
            "path": display_path(config_path),
            "sha256": sha256(config_path),
            "config_id": config["config_id"],
            "contract_validated_before_path_resolution": True,
        },
        "controls": {
            "network_used": False,
            "gpu_requested": False,
            "gpu_api_initialized": False,
            "arc_data_loaded": False,
            "test_labels_loaded": False,
            "checkpoint_bytes_read": False,
            "checkpoint_loaded": False,
            "upstream_code_imported": False,
            "upstream_code_executed": False,
            "solver_executed": False,
            "predictions_generated": False,
        },
        "source": {
            "path": str(source),
            "repository_url": source_config["repository_url"],
            "declared_revision": source_config["expected_revision"],
            "source_lock_path": source_config["source_lock_path"],
            "source_lock_sha256": sha256(source_lock_path),
            "source_lock_matches": source_lock_matches,
            "independent_git_checkout": (source / ".git").is_dir(),
            "prepared_checkout": str(prepared_checkout),
            "prepared_checkout_present": prepared_checkout.is_dir(),
            "revision_object_present_in_root_git": revision_object.returncode == 0,
            "root_tracked_file_count": len(tracked_relative),
            "root_tracked_paths_match_snapshot": source_snapshot_matches,
            "tracked_file_inventory": file_inventory(source, source_tracked_paths),
            "python_file_count": len(source_python_paths),
            "python_bytes": sum(path.stat().st_size for path in source_python_paths),
            "python_tree_sha256": source_python_digest,
            "clean_file_count": len(source_tracked_paths),
            "clean_tree_sha256": source_clean_digest,
            "critical_files": source_critical,
            "syntax_failures": source_syntax_failures,
            "root_license_files": source_licenses,
        },
        "source_provenance_gate": {
            "status": "blocked",
            "reason": "The retained allowlisted snapshot is byte-locked, but it is not an independent Git checkout, the prepared checkout is absent, and the declared upstream commit object is unavailable locally.",
        },
        "license_gate": {
            "status": "blocked",
            "gridcoder_root_license_files": source_licenses,
            "arc_gym_root_license_files": arc_licenses,
            "reason": "No repository-root license file is present in either locked source tree; this is a filename audit, not legal advice.",
        },
        "arc_gym": {
            "path": str(arc_gym),
            "repository_url": arc_gym_config["repository_url"],
            "expected_revision": arc_gym_config["expected_revision"],
            "observed_revision": arc_revision.stdout.strip()
            if arc_revision.returncode == 0
            else None,
            "tracked_dirty_paths": arc_tracked_status.stdout.splitlines(),
            "untracked_paths": arc_untracked_status.stdout.splitlines(),
            "source_lock_entry_present": arc_gym_source_lock_present,
            "tracked_python_inventory": file_inventory(arc_gym / "ARC_gym", arc_python_paths),
            "python_file_count": len(arc_python_paths),
            "python_bytes": sum(path.stat().st_size for path in arc_python_paths),
            "python_tree_sha256": arc_python_digest,
            "critical_files": arc_critical,
            "syntax_failures": arc_syntax_failures,
            "root_license_files": arc_licenses,
        },
        "dependency_gate": {
            "status": "blocked",
            "required_runner_modules": required_modules,
            "missing_runner_modules": missing_modules,
            "default_evaluation_directory": str(default_evaluation_directory),
            "default_evaluation_directory_present": default_evaluation_directory_present,
            "runner_imports": [
                item for item in runner_imports if str(item["module"]).startswith("ARC_gym")
            ],
            "evaluation_loader_imports": [
                item for item in loader_imports if str(item["module"]).startswith("ARC_gym")
            ],
            "reason": "The pinned ARC_gym checkout lacks modules imported unconditionally by the official evaluator, and the default evaluation overlay is absent.",
        },
        "label_firewall_gate": {
            "status": "blocked",
            "loader_path": arc_loader_path.relative_to(ROOT).as_posix(),
            **label_flow,
            "challenge_only_branch_present_in_runner": runner_controls[
                "challenge_only_candidate_branch_detected"
            ],
            "reason": "The default evaluation loader reads test outputs and carries them through the returned task tuple into yq. The separate challenge-only candidate branch has not passed a strict A/B adapter audit.",
        },
        "checkpoint_gate": {
            "status": "blocked",
            "filename": config["checkpoint"]["filename"],
            "unverified_prior_remote_metadata": config["checkpoint"][
                "unverified_prior_remote_metadata"
            ],
            "present_count": present_checkpoint_count,
            "paths": checkpoint_observations,
            "reason": "No known local checkpoint is present and no local SHA-256 lock exists.",
        },
        "runtime_portability_gate": {
            "status": "blocked",
            **runner_controls,
            "reason": "The official runner statically hard-codes CUDA and direct torch.load calls; no CPU solver path was executed or validated.",
        },
        "coverage_gate": {
            "status": "blocked",
            "classification": "limited-to-preselected-subset",
            **coverage,
            "native_benchmark": config["coverage"]["native_benchmark"],
            "scope": config["coverage"]["scope"],
            "reason": "The README reports only a preselected 49-task DSL-solvable subset, not the complete ARC-AGI-1 evaluation split.",
        },
        "solver_gate_passed": False,
        "prior_architecture_evidence": {
            "run_path": display_path(prior_run_path),
            "run_sha256": sha256(prior_run_path),
            "runner_path": display_path(prior_runner_path),
            "runner_sha256": sha256(prior_runner_path),
            "matched": prior_evidence_matches,
            "scope": "synthetic-weight-gpu-architecture-forward-only",
        },
        "evidence_manifest": manifest,
        "evidence_manifest_sha256": manifest_digest,
        "fairness": {
            "performance_table_eligible": False,
            "strict_runtime_promotion_eligible": False,
            "evidence_scope": "blocker_audit",
            "reason": "No solver prediction was produced; provenance, license, dependency, checkpoint, label-firewall, and runtime gates remain unresolved.",
        },
        "validation": validation,
        "limitations": [
            "Only the root-Git-tracked 29-file GridCoder allowlist and the pinned ARC_gym tracked package plus declared critical metadata files were opened.",
            "Checkpoint candidates and the ARC evaluation directory were checked only with exists/stat before source hashing; no checkpoint or ARC task byte was read.",
            "This audit imported and executed no upstream code and initialized no GPU or network client.",
            "Remote checkpoint size and license are explicitly unverified prior metadata; this offline audit did not re-query Kaggle.",
            "Checkpoint paths outside the exact prospective list were not searched.",
            "Root-license filename absence is not a legal conclusion.",
            "CPU/RSS accounting covers this process only; read-only Git subprocesses are excluded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "gridcoder2024_gate_v3.json",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    started_at = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record: dict[str, Any] = {
        "schema_version": 1,
        "method_id": "gridcoder2024",
        "run_id": output_directory.name,
        "runner": "scripts.audit_gridcoder_gates",
        "status": "failed",
        "scope": "source-dependency-label-artifact-gate-audit-only",
        "started_at_utc": started_at,
        "controls": {
            "network_used": False,
            "gpu_requested": False,
            "gpu_api_initialized": False,
            "arc_data_loaded": False,
            "test_labels_loaded": False,
            "checkpoint_bytes_read": False,
            "checkpoint_loaded": False,
            "upstream_code_imported": False,
            "upstream_code_executed": False,
            "solver_executed": False,
            "predictions_generated": False,
        },
    }
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record = audit(config, config_path)
        record["run_id"] = output_directory.name
        record["started_at_utc"] = started_at
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
            "ru_maxrss_before": rss_start,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "KiB on Linux",
            "accounting_scope": "current_process",
            "children_included": False,
        }
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "solver_gate_passed": record.get("solver_gate_passed"),
                "run_json": str(output_directory / "run.json"),
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
