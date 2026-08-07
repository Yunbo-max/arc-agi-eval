#!/usr/bin/env python3
"""Statically audit the retained 2D nGPT reproduction gates.

This auditor never imports or executes upstream code, initializes an
accelerator or network client, or opens checkpoint, re-ARC, fixed-size, or ARC
solution artifacts.  Forbidden artifacts and ignored bytecode caches are
observed with filesystem metadata only.
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
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_SOURCE_LOCK_SHA256 = (
    "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
)
EXPECTED_PRIOR_RUN_SHA256 = (
    "8d5ac324769781a6a580907ddd9f2795786a7e6464ba4c77ee9b8b16ffc1132a"
)
EXPECTED_PRIOR_RUNNER_SHA256 = (
    "8673d35ea09a343350523d264739ef0fff70197453ae37e818c3bbd641ea5568"
)

EXPECTED_BLOCKER_IDS = [
    "upstream-provenance",
    "checkpoint-provenance",
    "re-arc-data",
    "fixed-size-and-solution-data",
    "label-firewall",
    "runtime-portability",
    "reproduction-contract",
]
EXPECTED_CONTROL_KEYS = {
    "network_allowed",
    "gpu_allowed",
    "arc_solution_byte_read_allowed",
    "generated_re_arc_byte_read_allowed",
    "fixed_size_byte_read_allowed",
    "checkpoint_byte_read_allowed",
    "checkpoint_load_allowed",
    "upstream_import_allowed",
    "upstream_execution_allowed",
    "solver_execution_allowed",
}
EXPECTED_SOURCE_FILES = [
    {
        "path": "LICENSE",
        "bytes": 11357,
        "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
    {
        "path": "README.md",
        "bytes": 2844,
        "sha256": "d899b77e4b28ffe26716cb54bc42dadcddf9e534729fd0b1770e08e820cca2cf",
    },
    {
        "path": "cfg/cfg_053.py",
        "bytes": 1882,
        "sha256": "6566326afe6aca1e9b60076e5f24dc3dbc3a33631220641ed8082e62fd896328",
    },
    {
        "path": "cfg/cfg_064.py",
        "bytes": 2188,
        "sha256": "d675369de387aa70bccb996f86d9524df273b4534e852965a64cee0a8c91e537",
    },
    {
        "path": "cfg/readme.md",
        "bytes": 1,
        "sha256": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    },
    {
        "path": "code/053.py",
        "bytes": 48158,
        "sha256": "0828c40ff6c6700431de9367754de0e2c73e204114b9a15b63bfe511cd689d74",
    },
    {
        "path": "code/064.py",
        "bytes": 64114,
        "sha256": "2d9ef3455b0550149312dcaafdf2116fc16008d2770d28563e6ac230de2704c7",
    },
    {
        "path": "notebooks/gen10000.ipynb",
        "bytes": 923242,
        "sha256": "66af17202133b7191edc9df3328cb2bf9c3f62c04af597f2db3cdb7f678dce4d",
    },
    {
        "path": "notebooks/train.ipynb",
        "bytes": 25502,
        "sha256": "f41ba8a462baa2642d485f2b38d3c5b58098c34c20b7eedf1fd1f8c2623886d5",
    },
    {
        "path": "notebooks/ttt.ipynb",
        "bytes": 12129,
        "sha256": "a443a7638bdf5807103ad575832997ca764fdc11ed6869acb158046f718a4cd1",
    },
]
EXPECTED_REQUIRED_DIRECTORIES = ["cfg", "code", "notebooks"]
EXPECTED_OPTIONAL_CACHE_DIRECTORIES = ["cfg/__pycache__", "code/__pycache__"]
EXPECTED_METADATA_ONLY_CACHE_FILES = [
    {"path": "cfg/__pycache__/cfg_064.cpython-310.pyc", "bytes": 2044},
    {"path": "code/__pycache__/064.cpython-310.pyc", "bytes": 44484},
]
EXPECTED_CHECKPOINT_PATHS = [
    "external/ARC-AGI-Challenge-2024/checkpoints/ngc/exp_50.pt",
    "external/ARC-AGI-Challenge-2024/checkpoints/ngc/exp_54.pt",
    "external/checkpoints/ngc/exp_50.pt",
    "external/checkpoints/ngc/exp_54.pt",
    "/usr/paper-assets/arc/sources/2d-ngpt/checkpoints/ngc/exp_50.pt",
    "/usr/paper-assets/arc/sources/2d-ngpt/checkpoints/ngc/exp_54.pt",
    "/usr/paper-assets/arc/checkpoints/2d-ngpt/exp_50.pt",
    "/usr/paper-assets/arc/checkpoints/2d-ngpt/exp_54.pt",
    "/model/2d-ngpt/exp_50.pt",
    "/model/2d-ngpt/exp_54.pt",
]
EXPECTED_RE_ARC_PATHS = [
    "external/ARC-AGI-Challenge-2024/re-arc/gen10000/tasks",
    "external/re-arc/gen10000/tasks",
    "/usr/paper-assets/arc/sources/2d-ngpt/re-arc/gen10000/tasks",
    "/usr/paper-assets/arc/data/re-arc/gen10000/tasks",
]
EXPECTED_FIXED_SIZE_PATHS = [
    "external/ARC-AGI-Challenge-2024/input/arc-prize-2024/fixed_size.pkl",
    "external/input/arc-prize-2024/fixed_size.pkl",
    "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/fixed_size.pkl",
    "/usr/paper-assets/arc/data/arc-prize-2024/fixed_size.pkl",
]
EXPECTED_SOLUTION_PATHS = [
    "external/ARC-AGI-Challenge-2024/input/arc-prize-2024/arc-agi_training_solutions.json",
    "external/ARC-AGI-Challenge-2024/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    "external/input/arc-prize-2024/arc-agi_training_solutions.json",
    "external/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/arc-agi_training_solutions.json",
    "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    "/usr/paper-assets/arc/data/arc-prize-2024/arc-agi_training_solutions.json",
    "/usr/paper-assets/arc/data/arc-prize-2024/arc-agi_evaluation_solutions.json",
]
EXPECTED_TUNE_FLOW_CALLS = [
    "load_checkpoint",
    "load_train",
    "add_validation_data",
    "get_tune_datasets",
    "train_model",
    "get_data_loader",
    "valid_epoch",
]
EXPECTED_GLOBAL_CFG_SCOPES = ["ARCModel.forward", "add_validation_data"]
EXPECTED_LABEL_FLOW_LINES = {
    "test_solution_to_test_sample_output_lines": [113, 122],
    "get_color_perm_sample_output_lines": [132],
    "evaluation_solution_to_eval_train_lines": [201],
    "evaluation_solution_to_eval_test_lines": [202],
    "eval_train_and_eval_test_samples_lines": [223],
    "eval_solution_color_perm_lines": [225],
    "eval_aug_color_dataset_write_lines": [230],
    "tune_train_dataset_constructor_lines": [268],
    "tune_valid_dataset_constructor_lines": [269],
    "arc_eval_dataset_eval_aug_color_lines": [416],
    "arc_eval_dataset_eval_aug_color_use_lines": [465],
    "arc_valid_dataset_eval_aug_color_lines": [538],
    "arc_valid_dataset_eval_test_lines": [504],
    "run_model_update_lines": [1793],
    "run_train_model_lines": [1801],
    "arc_valid_sample_output_read_lines": [564],
    "arc_valid_output_batch_value_lines": [599, 624, 654, 673],
    "arc_model_prepare_batch_output_lines": [1114],
    "arc_model_forward_output_loss_lines": [1173],
    "valid_epoch_return_loss_model_lines": [1293],
    "valid_epoch_output_metric_helper_lines": [1299, 1303, 1306, 1309, 1312],
    "metric_helper_output_read_lines": [1351, 1440, 1486],
    "metric_helper_output_loss_lines": [1420, 1471, 1526],
    "metric_helper_output_accuracy_lines": [1426, 1476, 1531],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def open_verified_regular(path: Path) -> tuple[int, os.stat_result]:
    """Open a regular file without following any path-component symlink."""
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"refusing to read non-regular or symlink file: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    absolute = path.absolute()
    directory_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parent.parts[1:]:
            next_descriptor = os.open(
                part, directory_flags, dir_fd=directory_descriptor
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        parent_view = os.stat(
            absolute.name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if stat.S_ISLNK(parent_view.st_mode) or not stat.S_ISREG(parent_view.st_mode):
            raise ValueError(f"refusing to read non-regular or symlink file: {path}")
        descriptor = os.open(
            absolute.name, file_flags, dir_fd=directory_descriptor
        )
    finally:
        os.close(directory_descriptor)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino)
        != (before.st_dev, before.st_ino)
        or (opened.st_dev, opened.st_ino)
        != (parent_view.st_dev, parent_view.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"file changed identity between lstat and open: {path}")
    return descriptor, before


def sha256(path: Path) -> str:
    descriptor, before = open_verified_regular(path)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError(f"file changed while being read: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_regular_bytes(path: Path) -> bytes:
    descriptor, before = open_verified_regular(path)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError(f"file changed while being read: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def read_regular_text(path: Path) -> str:
    return read_regular_bytes(path).decode("utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def prepare_fresh_output_directory(path: Path) -> int:
    """Create a fresh output directory through non-symlink directory FDs."""
    absolute = path.absolute()
    if absolute == Path(absolute.anchor) or not absolute.name:
        raise ValueError("output directory must be a fresh non-root path")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parent.parts[1:]:
            try:
                metadata = os.stat(
                    part, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError as error:
                raise ValueError(
                    f"output parent component does not exist: {part}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"output parent has a symlink/non-directory component: {part}"
                )
            next_descriptor = os.open(
                part, directory_flags, dir_fd=parent_descriptor
            )
            opened = os.fstat(next_descriptor)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(next_descriptor)
                raise ValueError("output parent changed identity during validation")
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor

        try:
            leaf = os.stat(
                absolute.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(leaf.st_mode):
                raise ValueError(f"output directory leaf must not be a symlink: {absolute}")
            raise ValueError(f"output directory must not already exist: {absolute}")
        os.mkdir(absolute.name, mode=0o700, dir_fd=parent_descriptor)
        created = os.stat(
            absolute.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if stat.S_ISLNK(created.st_mode) or not stat.S_ISDIR(created.st_mode):
            raise ValueError("fresh output path is not a regular directory")
        output_descriptor = os.open(
            absolute.name, directory_flags, dir_fd=parent_descriptor
        )
        opened = os.fstat(output_descriptor)
        if (opened.st_dev, opened.st_ino) != (created.st_dev, created.st_ino):
            os.close(output_descriptor)
            raise ValueError("fresh output directory changed identity during creation")
        return output_descriptor
    finally:
        os.close(parent_descriptor)


def atomic_json_at(directory_descriptor: int, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary_name = f".run.json.{os.getpid()}.{time.time_ns()}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(
        temporary_name, flags, 0o600, dir_fd=directory_descriptor
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary_name,
            "run.json",
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        raise


def require_exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if set(value) != expected:
        raise ValueError(
            f"{field} keys mismatch: expected {sorted(expected)}, found {sorted(value)}"
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
            "forbidden_artifacts",
            "static_contract",
            "prior_architecture_evidence",
            "controls",
        },
        "config",
    )
    if config["schema_version"] != 1:
        raise ValueError("config.schema_version must equal 1")
    if config["config_id"] != "2d-ngpt-source-artifact-label-runtime-gate-v1":
        raise ValueError("unexpected config.config_id")
    if config["method_id"] != "2d-ngpt":
        raise ValueError("config.method_id must equal 2d-ngpt")
    if config["scope"] != "source-artifact-label-runtime-gate-audit-only":
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
            "expected_independent_git_checkout",
            "expected_prepared_checkout_present",
            "expected_revision_object_present_in_root_git",
            "expected_file_count",
            "expected_total_bytes",
            "expected_tree_sha256",
            "expected_tree_digest_algorithm",
            "expected_external_sources_prefixed_tree_sha256",
            "expected_files",
            "required_directories",
            "optional_cache_directories",
            "metadata_only_cache_files",
            "expected_root_license_files",
        },
        "config.source",
    )
    exact_source_scalars = {
        "repository_url": "https://github.com/jfpuget/ARC-AGI-Challenge-2024",
        "source_lock_path": "configs/source_locks.json",
        "local_snapshot": "external/ARC-AGI-Challenge-2024",
        "prepared_checkout": "/usr/paper-assets/arc/sources/2d-ngpt",
        "expected_file_count": 10,
        "expected_total_bytes": 1091417,
        "expected_tree_sha256": "05bb50747e8cf0b98c149f75f65dee8a37bde3b7fe5c1926ad661bf1513e7c21",
        "expected_tree_digest_algorithm": "sha256(utf8(concat(file_sha256 + two_spaces + source_root_relative_path + newline)))_in_lexicographic_path_order",
        "expected_external_sources_prefixed_tree_sha256": "f7d595edbc89619d83f1570532ff5ed58f155accac1dd53a6d11ea268a04d1dd",
    }
    for key, expected in exact_source_scalars.items():
        if source[key] != expected:
            raise ValueError(f"config.source.{key} mismatch")
    require_git_sha(source["expected_revision"], "config.source.expected_revision")
    if source["expected_revision"] != "e5420b10b9470b3b5c6548572768d2d4c15130f6":
        raise ValueError("config.source.expected_revision mismatch")
    require_sha256(source["source_lock_sha256"], "config.source.source_lock_sha256")
    if source["source_lock_sha256"] != EXPECTED_SOURCE_LOCK_SHA256:
        raise ValueError("config.source.source_lock_sha256 mismatch")
    require_sha256(source["expected_tree_sha256"], "config.source.expected_tree_sha256")
    for key in (
        "expected_independent_git_checkout",
        "expected_prepared_checkout_present",
        "expected_revision_object_present_in_root_git",
    ):
        if source[key] is not False:
            raise ValueError(f"config.source.{key} must be false")
    if source["expected_files"] != EXPECTED_SOURCE_FILES:
        raise ValueError("config.source.expected_files mismatch")
    for index, item in enumerate(source["expected_files"]):
        require_exact_keys(item, {"path", "bytes", "sha256"}, f"expected_files[{index}]")
        require_safe_relative_file(item["path"], f"expected_files[{index}].path")
        require_nonnegative_int(item["bytes"], f"expected_files[{index}].bytes")
        require_sha256(item["sha256"], f"expected_files[{index}].sha256")
    if source["required_directories"] != EXPECTED_REQUIRED_DIRECTORIES:
        raise ValueError("config.source.required_directories mismatch")
    if source["optional_cache_directories"] != EXPECTED_OPTIONAL_CACHE_DIRECTORIES:
        raise ValueError("config.source.optional_cache_directories mismatch")
    if source["metadata_only_cache_files"] != EXPECTED_METADATA_ONLY_CACHE_FILES:
        raise ValueError("config.source.metadata_only_cache_files mismatch")
    for index, item in enumerate(source["metadata_only_cache_files"]):
        require_exact_keys(item, {"path", "bytes"}, f"metadata_only_cache_files[{index}]")
        require_safe_relative_file(item["path"], f"metadata_only_cache_files[{index}].path")
        require_nonnegative_int(item["bytes"], f"metadata_only_cache_files[{index}].bytes")
    if source["expected_root_license_files"] != ["LICENSE"]:
        raise ValueError("config.source.expected_root_license_files mismatch")

    artifacts = require_exact_keys(
        config["forbidden_artifacts"],
        {
            "checkpoint_paths",
            "re_arc_paths",
            "fixed_size_paths",
            "solution_paths",
            "expected_present_count",
        },
        "config.forbidden_artifacts",
    )
    for key, expected in (
        ("checkpoint_paths", EXPECTED_CHECKPOINT_PATHS),
        ("re_arc_paths", EXPECTED_RE_ARC_PATHS),
        ("fixed_size_paths", EXPECTED_FIXED_SIZE_PATHS),
        ("solution_paths", EXPECTED_SOLUTION_PATHS),
    ):
        if artifacts[key] != expected:
            raise ValueError(f"config.forbidden_artifacts.{key} mismatch")
    if artifacts["expected_present_count"] != 0:
        raise ValueError("forbidden artifact expected_present_count must equal zero")

    contract = require_exact_keys(
        config["static_contract"],
        {
            "runner_path",
            "config_path",
            "expected_solution_literals",
            "expected_tune_flow_calls",
            "expected_global_cfg_scopes",
            "expected_pretrained_literal",
            "expected_re_arc_literal",
            "expected_input_literal",
            "expected_device_literal",
            "expected_tune_model_default",
        },
        "config.static_contract",
    )
    expected_contract = {
        "runner_path": "code/064.py",
        "config_path": "cfg/cfg_064.py",
        "expected_solution_literals": [
            "arc-agi_training_solutions.json",
            "arc-agi_evaluation_solutions.json",
        ],
        "expected_tune_flow_calls": EXPECTED_TUNE_FLOW_CALLS,
        "expected_global_cfg_scopes": EXPECTED_GLOBAL_CFG_SCOPES,
        "expected_pretrained_literal": "../checkpoints/ngc/exp_54.pt",
        "expected_re_arc_literal": "../re-arc/%s/tasks",
        "expected_input_literal": "../input/arc-prize-2024/",
        "expected_device_literal": "cuda",
        "expected_tune_model_default": True,
    }
    if contract != expected_contract:
        raise ValueError("config.static_contract mismatch")

    prior = require_exact_keys(
        config["prior_architecture_evidence"],
        {
            "run_path",
            "run_sha256",
            "runner_path",
            "runner_sha256",
            "classification",
            "component_smoke_counted",
            "solver_prediction_smoke",
        },
        "config.prior_architecture_evidence",
    )
    if prior["run_path"] != "reports/2d-ngpt/20260806-large-architecture-forward-smoke/run.json":
        raise ValueError("unexpected prior architecture run path")
    if prior["runner_path"] != "scripts/smoke_2d_ngpt.py":
        raise ValueError("unexpected prior architecture runner path")
    require_sha256(prior["run_sha256"], "prior run sha256")
    require_sha256(prior["runner_sha256"], "prior runner sha256")
    if prior["run_sha256"] != EXPECTED_PRIOR_RUN_SHA256:
        raise ValueError("config.prior_architecture_evidence.run_sha256 mismatch")
    if prior["runner_sha256"] != EXPECTED_PRIOR_RUNNER_SHA256:
        raise ValueError("config.prior_architecture_evidence.runner_sha256 mismatch")
    if prior["classification"] != "component-evidence-only":
        raise ValueError("prior evidence must remain component-evidence-only")
    if prior["component_smoke_counted"] is not True:
        raise ValueError("prior architecture evidence must remain a counted component smoke")
    if prior["solver_prediction_smoke"] is not False:
        raise ValueError("prior architecture evidence cannot be solver prediction smoke")

    controls = require_exact_keys(config["controls"], EXPECTED_CONTROL_KEYS, "config.controls")
    if not all(controls[key] is False for key in EXPECTED_CONTROL_KEYS):
        raise ValueError("all config.controls values must be false")
    return config


def lexical_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return Path(os.path.abspath(path))


def lexically_within(path: Path, parent: Path) -> bool:
    try:
        path.absolute().relative_to(parent.absolute())
    except ValueError:
        return False
    return True


def sensitive_input_roots() -> list[Path]:
    roots = [lexical_path("external/ARC-AGI-Challenge-2024")]
    roots.extend(lexical_path(value) for value in EXPECTED_RE_ARC_PATHS)
    return roots


def sensitive_input_files() -> list[Path]:
    return [
        lexical_path(value)
        for value in (
            EXPECTED_CHECKPOINT_PATHS
            + EXPECTED_FIXED_SIZE_PATHS
            + EXPECTED_SOLUTION_PATHS
        )
    ]


def reject_sensitive_control_path(path: Path, field: str) -> None:
    absolute = path.absolute()
    if absolute in sensitive_input_files() or any(
        lexically_within(absolute, root) for root in sensitive_input_roots()
    ):
        raise ValueError(f"{field} must remain outside source and forbidden artifacts")


def mode_kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def observe_forbidden_path(path: Path) -> dict[str, Any]:
    """Walk with lstat only; never follow a symlinked artifact parent."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return {
                "present": False,
                "file_type": None,
                "bytes": None,
                "missing_component": str(current),
                "forbidden_parent_component": None,
            }
        final = index == len(absolute.parts[1:]) - 1
        if not final and (
            stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode)
        ):
            return {
                "present": True,
                "file_type": "forbidden_parent_component",
                "bytes": metadata.st_size,
                "missing_component": None,
                "forbidden_parent_component": {
                    "path": str(current),
                    "file_type": mode_kind(metadata.st_mode),
                },
            }
        if final:
            return {
                "present": True,
                "file_type": mode_kind(metadata.st_mode),
                "bytes": metadata.st_size,
                "missing_component": None,
                "forbidden_parent_component": None,
            }
    raise ValueError(f"invalid forbidden artifact path: {path}")


def observe_forbidden_paths(groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Use lstat only; forbidden artifact bytes are never opened or hashed."""
    observations: list[dict[str, Any]] = []
    for kind in ("checkpoint", "re_arc", "fixed_size", "solution"):
        for declared in groups[kind]:
            path = lexical_path(declared)
            observed = observe_forbidden_path(path)
            observations.append(
                {
                    "kind": kind,
                    "declared_path": declared,
                    "path": str(path),
                    **observed,
                    "sha256": None,
                    "bytes_read": False,
                    "loaded": False,
                }
            )
    return observations


def assert_forbidden_material_absent(observations: list[dict[str, Any]]) -> None:
    if any(item.get("present") is True for item in observations):
        raise RuntimeError(
            "forbidden checkpoint/re-ARC/fixed-size/solution material is present; "
            "aborting before retained-source reads"
        )


def lstat_directory(path: Path, label: str) -> Path:
    current = Path(path.anchor)
    for part in path.absolute().parts[1:]:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} has a symlink/non-directory component: {current}")
    return path.absolute()


def contained_regular_file(root: Path, relative: str) -> Path:
    relative_path = Path(require_safe_relative_file(relative, "contained file"))
    current = lstat_directory(root.absolute(), "content root")
    for index, part in enumerate(relative_path.parts):
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"contained path has a symlink component: {relative}")
        if index < len(relative_path.parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"contained path parent is not a directory: {relative}")
        elif not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"contained path is not a regular file: {relative}")
    return current


def metadata_only_source_inventory(
    root: Path,
    expected_files: list[str],
    required_directories: list[str],
    optional_cache_directories: list[str],
    metadata_only_cache_files: list[dict[str, Any]],
) -> dict[str, Any]:
    """Close the filesystem tree with lstat/scandir before any source read."""
    root = lstat_directory(root.absolute(), "source root")
    expected = set(expected_files)
    required = set(required_directories)
    optional_dirs = set(optional_cache_directories)
    allowed_dirs = required | optional_dirs
    caches = {item["path"]: item["bytes"] for item in metadata_only_cache_files}
    found_files: list[str] = []
    found_directories: list[str] = []
    cache_inventory: list[dict[str, Any]] = []
    stack: list[tuple[str, Path]] = [("", root)]
    while stack:
        prefix, directory = stack.pop()
        with os.scandir(directory) as entries_handle:
            entries = sorted(entries_handle, key=lambda item: item.name)
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            metadata = entry.stat(follow_symlinks=False)
            kind = mode_kind(metadata.st_mode)
            if kind == "symlink":
                raise ValueError(
                    f"source filesystem contains symlink before retained-source reads: {relative}"
                )
            if kind == "directory":
                if relative not in allowed_dirs:
                    raise ValueError(
                        "source filesystem contains unknown directory before "
                        f"retained-source reads: {relative}"
                    )
                found_directories.append(relative)
                stack.append((relative, Path(entry.path)))
                continue
            if kind != "regular_file":
                raise ValueError(
                    "source filesystem contains non-regular path before "
                    f"retained-source reads: {relative}"
                )
            if relative in expected:
                found_files.append(relative)
            elif relative in caches:
                if metadata.st_size != caches[relative]:
                    raise ValueError(
                        "metadata-only cache size mismatch before retained-source "
                        f"reads: {relative}"
                    )
                cache_inventory.append(
                    {
                        "path": relative,
                        "bytes": metadata.st_size,
                        "expected_bytes": caches[relative],
                        "file_type": "regular_file",
                        "sha256": None,
                        "bytes_read": False,
                        "manifest_included": False,
                    }
                )
            else:
                raise ValueError(
                    "source filesystem contains unknown file before "
                    f"retained-source reads: {relative}"
                )
    if set(found_files) != expected:
        missing = sorted(expected - set(found_files))
        raise ValueError(f"source filesystem is missing retained files: {missing}")
    if not required.issubset(found_directories):
        missing = sorted(required - set(found_directories))
        raise ValueError(f"source filesystem is missing required directories: {missing}")
    return {
        "retained_paths": sorted(found_files),
        "directories": sorted(found_directories),
        "ignored_cache_inventory": sorted(
            cache_inventory, key=lambda item: item["path"]
        ),
        "closed_world": True,
        "retained_source_bytes_read": False,
        "cache_bytes_read": False,
        "ignored_cache_bytes_read": False,
        "ignored_cache_executable_content_trusted": False,
        "filesystem_has_ignored_cache": bool(cache_inventory),
        "runtime_source_tree_approved": False,
    }


def run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def tracked_allowlist(source_root: Path, expected_paths: list[str]) -> list[str]:
    pathspec = source_root.relative_to(ROOT).as_posix()
    result = run_git("ls-files", "--", pathspec)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    prefix = f"{pathspec}/"
    tracked: list[str] = []
    for line in result.stdout.splitlines():
        if not line.startswith(prefix):
            raise ValueError(f"tracked path escapes source root: {line}")
        tracked.append(require_safe_relative_file(line[len(prefix) :], "tracked path"))
    tracked = sorted(tracked)
    if tracked != sorted(expected_paths):
        raise ValueError(
            "Git tracked source allowlist mismatch; aborting before retained-source reads"
        )
    return tracked


def source_inventory(
    root: Path, declarations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, str, str]:
    inventory: list[dict[str, Any]] = []
    for declaration in declarations:
        path = contained_regular_file(root, declaration["path"])
        observed_size = os.lstat(path).st_size
        observed_sha = sha256(path)
        inventory.append(
            {
                "path": declaration["path"],
                "bytes": observed_size,
                "sha256": observed_sha,
                "expected_bytes": declaration["bytes"],
                "expected_sha256": declaration["sha256"],
                "matched": observed_size == declaration["bytes"]
                and observed_sha == declaration["sha256"],
            }
        )
    lines = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in inventory
    ).encode("utf-8")
    prefixed_lines = "".join(
        f"{item['sha256']}  ARC-AGI-Challenge-2024/{item['path']}\n"
        for item in inventory
    ).encode("utf-8")
    python_lines = "".join(
        f"{item['sha256']}  {item['path']}\n"
        for item in inventory
        if item["path"].endswith(".py")
    ).encode("utf-8")
    return (
        inventory,
        hashlib.sha256(lines).hexdigest(),
        hashlib.sha256(prefixed_lines).hexdigest(),
        hashlib.sha256(python_lines).hexdigest(),
    )


def parse_python(path: Path, expected_sha256: str) -> ast.Module:
    source_bytes = read_regular_bytes(path)
    observed = hashlib.sha256(source_bytes).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"AST source bytes do not match the prospectively locked hash: {path}"
        )
    return ast.parse(source_bytes.decode("utf-8"), filename=str(path))


def call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def string_constants(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def named_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ValueError(f"expected exactly one function {name}")
    return matches[0]


def named_method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one class {class_name}")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(methods) != 1:
        raise ValueError(f"expected exactly one method {class_name}.{name}")
    return methods[0]


def solution_and_tune_flow(tree: ast.Module) -> dict[str, Any]:
    solution_sites: list[dict[str, Any]] = []
    for function_name in ("load_data", "add_validation_data"):
        function = named_function(tree, function_name)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or call_name(node) != "load_solutions":
                continue
            literals = [
                value for value in string_constants(node) if value.endswith("_solutions.json")
            ]
            targets: list[str] = []
            for candidate in ast.walk(function):
                if isinstance(candidate, ast.Assign) and candidate.value is node:
                    targets.extend(
                        target.id
                        for target in candidate.targets
                        if isinstance(target, ast.Name)
                    )
            solution_sites.append(
                {
                    "function": function_name,
                    "line": node.lineno,
                    "solution_literals": literals,
                    "assigned_names": sorted(targets),
                }
            )

    validation = named_function(tree, "add_validation_data")
    original_tasks = named_function(tree, "get_original_tasks")
    test_solution_output_injection_lines = [
        node.lineno
        for node in ast.walk(original_tasks)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and node.value.id == "test_solution"
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "test_sample"
            and "output" in string_constants(target.slice)
            for target in node.targets
        )
    ]
    color_perm = named_function(tree, "get_color_perm")
    get_color_perm_sample_output_lines = [
        node.lineno
        for node in ast.walk(color_perm)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sample"
        and "output" in string_constants(node.slice)
    ]
    eval_test_lines: list[int] = []
    eval_train_lines: list[int] = []
    for node in ast.walk(validation):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if call_name(node.value) != "get_original_tasks":
            continue
        uses_solutions = any(
            isinstance(child, ast.Name) and child.id == "evaluation_solutions"
            for child in ast.walk(node.value)
        )
        if not uses_solutions:
            continue
        target_keys = [
            child.value
            for target in node.targets
            for child in ast.walk(target)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        ]
        if "eval_test" in target_keys:
            eval_test_lines.append(node.lineno)
        if "eval_train" in target_keys:
            eval_train_lines.append(node.lineno)

    eval_color_perm_lines: list[int] = []
    eval_solution_color_perm_lines: list[int] = []
    samples_from_eval_train_and_test_lines: list[int] = []
    for node in ast.walk(validation):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "samples"
            for target in node.targets
        ):
            literals = set(string_constants(node.value))
            if {"eval_train", "eval_test"}.issubset(literals):
                samples_from_eval_train_and_test_lines.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and call_name(node) == "get_color_perm"
            and any(
                isinstance(child, ast.Name) and child.id == "samples"
                for argument in node.args
                for child in ast.walk(argument)
            )
        ):
            eval_color_perm_lines.append(node.lineno)
    for node in ast.walk(validation):
        if not isinstance(node, ast.For) or "eval_keys" not in string_constants(node.iter):
            continue
        loop_has_eval_samples = any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "samples"
                for target in child.targets
            )
            and {"eval_train", "eval_test"}.issubset(
                set(string_constants(child.value))
            )
            for statement in node.body
            for child in ast.walk(statement)
        )
        if loop_has_eval_samples:
            eval_solution_color_perm_lines.extend(
                child.lineno
                for statement in node.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Call)
                and call_name(child) == "get_color_perm"
            )
    eval_aug_color_dataset_write_lines = [
        node.lineno
        for node in ast.walk(validation)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and node.value.id == "perms"
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "dataset"
            and "eval_aug_color" in string_constants(target.slice)
            for target in node.targets
        )
    ]

    tune_datasets = named_function(tree, "get_tune_datasets")
    tune_train_dataset_lines = [
        node.lineno
        for node in ast.walk(tune_datasets)
        if isinstance(node, ast.Call) and call_name(node) == "ARCEvalDataset"
    ]
    tune_valid_dataset_lines = [
        node.lineno
        for node in ast.walk(tune_datasets)
        if isinstance(node, ast.Call)
        and call_name(node) == "ARCValidDataset"
        and any(
            keyword.arg == "evaluation"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    valid_init = named_method(tree, "ARCValidDataset", "__init__")
    valid_eval_test_lines = [
        node.lineno
        for node in ast.walk(valid_init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "data"
            for target in node.targets
        )
        and "eval_test" in string_constants(node.value)
    ]
    arc_eval_init = named_method(tree, "ARCEvalDataset", "__init__")
    arc_eval_aug_color_lines = [
        node.lineno
        for node in ast.walk(arc_eval_init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "eval_aug_color"
            for target in node.targets
        )
        and "eval_aug_color" in string_constants(node.value)
    ]
    arc_eval_getitem = named_method(tree, "ARCEvalDataset", "__getitem__")
    arc_eval_aug_color_use_lines = [
        node.lineno
        for node in ast.walk(arc_eval_getitem)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "perms"
            for target in node.targets
        )
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Attribute)
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == "self"
        and node.value.value.attr == "eval_aug_color"
    ]
    arc_valid_aug_color_lines = [
        node.lineno
        for node in ast.walk(valid_init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "aug_color"
            for target in node.targets
        )
        and "eval_aug_color" in string_constants(node.value)
    ]

    run = named_function(tree, "run")
    tune_candidates: list[ast.If] = []
    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "cfg"
            and child.attr == "tune_model"
            for child in ast.walk(node.test)
        ):
            tune_candidates.append(node)
    if not tune_candidates:
        raise ValueError("cfg.tune_model branch was not found")
    tune_if = max(
        tune_candidates,
        key=lambda candidate: sum(
            1
            for child in ast.walk(candidate)
            if isinstance(child, ast.Call) and call_name(child) in EXPECTED_TUNE_FLOW_CALLS
        ),
    )
    calls = sorted(
        (
            {"name": call_name(node), "line": node.lineno}
            for statement in tune_if.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and call_name(node) is not None
        ),
        key=lambda item: (item["line"], str(item["name"])),
    )
    first_call_lines: dict[str, int] = {}
    for item in calls:
        first_call_lines.setdefault(str(item["name"]), int(item["line"]))
    expected_lines = [first_call_lines.get(name) for name in EXPECTED_TUNE_FLOW_CALLS]
    run_model_update_lines = [
        node.lineno
        for statement in tune_if.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
        and node.func.attr == "update"
        and any(
            isinstance(child, ast.Name) and child.id == "train_dataset"
            for argument in node.args
            for child in ast.walk(argument)
        )
    ]
    run_train_model_lines = [
        node.lineno
        for statement in tune_if.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and call_name(node) == "train_model"
    ]

    valid_getitem = named_method(tree, "ARCValidDataset", "__getitem__")
    arc_valid_sample_output_read_lines = [
        node.lineno
        for node in ast.walk(valid_getitem)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "sample_output"
            for target in node.targets
        )
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sample"
        and "output" in string_constants(node.value.slice)
    ]
    arc_valid_output_batch_value_lines: list[int] = []
    for node in ast.walk(valid_getitem):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "output"
                and any(
                    isinstance(child, ast.Name)
                    and child.id.startswith("sample_output")
                    for child in ast.walk(value)
                )
            ):
                arc_valid_output_batch_value_lines.append(value.lineno)

    prepare_data = named_method(tree, "ARCModel", "prepare_data")
    arc_model_prepare_batch_output_lines = [
        node.lineno
        for node in ast.walk(prepare_data)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "output"
            for target in node.targets
        )
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "batch"
        and "output" in string_constants(node.value.slice)
    ]
    forward = named_method(tree, "ARCModel", "forward")
    arc_model_forward_output_loss_lines: list[int] = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.If) or not any(
            isinstance(child, ast.Name) and child.id == "return_loss"
            for child in ast.walk(node.test)
        ):
            continue
        arc_model_forward_output_loss_lines.extend(
            child.lineno
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and call_name(child) == "loss_fn"
            and any(
                isinstance(argument_child, ast.Name)
                and argument_child.id == "output"
                for argument in child.args
                for argument_child in ast.walk(argument)
            )
        )

    valid_epoch = named_function(tree, "valid_epoch")
    valid_epoch_return_loss_model_lines = [
        node.lineno
        for node in ast.walk(valid_epoch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "model"
        and any(
            keyword.arg == "return_loss"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    metric_helper_names = {
        "get_avg_loss_color_transpose",
        "get_avg_loss",
        "get_avg_color_loss",
    }
    valid_epoch_output_metric_helper_lines = [
        node.lineno
        for node in ast.walk(valid_epoch)
        if isinstance(node, ast.Call) and call_name(node) in metric_helper_names
    ]
    metric_helper_output_read_lines: list[int] = []
    metric_helper_output_loss_lines: list[int] = []
    metric_helper_output_accuracy_lines: list[int] = []
    for helper_name in sorted(metric_helper_names):
        helper = named_function(tree, helper_name)
        metric_helper_output_read_lines.extend(
            node.lineno
            for node in ast.walk(helper)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "outputs"
                for target in node.targets
            )
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "batch"
            and "output" in string_constants(node.value.slice)
        )
        metric_helper_output_loss_lines.extend(
            node.lineno
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and call_name(node) == "loss_fn"
            and any(
                isinstance(child, ast.Name) and child.id == "output"
                for argument in node.args
                for child in ast.walk(argument)
            )
        )
        metric_helper_output_accuracy_lines.extend(
            node.lineno
            for node in ast.walk(helper)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "pred"
            and any(
                isinstance(comparator, ast.Name) and comparator.id == "output"
                for comparator in node.comparators
            )
        )

    flow_lines = {
        "test_solution_to_test_sample_output_lines": sorted(
            test_solution_output_injection_lines
        ),
        "get_color_perm_sample_output_lines": sorted(
            get_color_perm_sample_output_lines
        ),
        "evaluation_solution_to_eval_train_lines": sorted(eval_train_lines),
        "evaluation_solution_to_eval_test_lines": sorted(eval_test_lines),
        "eval_train_and_eval_test_samples_lines": sorted(
            samples_from_eval_train_and_test_lines
        ),
        "eval_solution_color_perm_lines": sorted(eval_solution_color_perm_lines),
        "eval_aug_color_dataset_write_lines": sorted(
            eval_aug_color_dataset_write_lines
        ),
        "tune_train_dataset_constructor_lines": sorted(tune_train_dataset_lines),
        "tune_valid_dataset_constructor_lines": sorted(tune_valid_dataset_lines),
        "arc_eval_dataset_eval_aug_color_lines": sorted(arc_eval_aug_color_lines),
        "arc_eval_dataset_eval_aug_color_use_lines": sorted(
            arc_eval_aug_color_use_lines
        ),
        "arc_valid_dataset_eval_aug_color_lines": sorted(arc_valid_aug_color_lines),
        "arc_valid_dataset_eval_test_lines": sorted(valid_eval_test_lines),
        "run_model_update_lines": sorted(run_model_update_lines),
        "run_train_model_lines": sorted(run_train_model_lines),
        "arc_valid_sample_output_read_lines": sorted(
            arc_valid_sample_output_read_lines
        ),
        "arc_valid_output_batch_value_lines": sorted(
            arc_valid_output_batch_value_lines
        ),
        "arc_model_prepare_batch_output_lines": sorted(
            arc_model_prepare_batch_output_lines
        ),
        "arc_model_forward_output_loss_lines": sorted(
            arc_model_forward_output_loss_lines
        ),
        "valid_epoch_return_loss_model_lines": sorted(
            valid_epoch_return_loss_model_lines
        ),
        "valid_epoch_output_metric_helper_lines": sorted(
            valid_epoch_output_metric_helper_lines
        ),
        "metric_helper_output_read_lines": sorted(metric_helper_output_read_lines),
        "metric_helper_output_loss_lines": sorted(metric_helper_output_loss_lines),
        "metric_helper_output_accuracy_lines": sorted(
            metric_helper_output_accuracy_lines
        ),
    }
    exact_full_chain_lines_match = flow_lines == EXPECTED_LABEL_FLOW_LINES
    return {
        "solution_load_sites": sorted(solution_sites, key=lambda item: item["line"]),
        "solution_literals": sorted(
            {
                literal
                for item in solution_sites
                for literal in item["solution_literals"]
            }
        ),
        **flow_lines,
        "eval_train_and_eval_test_to_color_perm_lines": flow_lines[
            "eval_train_and_eval_test_samples_lines"
        ],
        "get_color_perm_from_eval_samples_lines": sorted(eval_color_perm_lines),
        "tune_valid_dataset_evaluation_lines": sorted(tune_valid_dataset_lines),
        "tune_branch_line": tune_if.lineno,
        "tune_branch_calls": calls,
        "expected_tune_flow_first_call_lines": dict(
            zip(EXPECTED_TUNE_FLOW_CALLS, expected_lines)
        ),
        "exact_full_chain_lines_match": exact_full_chain_lines_match,
        "flow_detected": (
            len(solution_sites) == 3
            and len(test_solution_output_injection_lines) == 2
            and bool(eval_train_lines)
            and bool(eval_test_lines)
            and bool(samples_from_eval_train_and_test_lines)
            and bool(eval_color_perm_lines)
            and bool(tune_valid_dataset_lines)
            and bool(valid_eval_test_lines)
            and all(line is not None for line in expected_lines)
            and expected_lines == sorted(expected_lines)
            and exact_full_chain_lines_match
        ),
    }


class _CfgLoadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.lines: list[int] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "cfg" and isinstance(node.ctx, ast.Load):
            self.lines.append(node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def global_cfg_scopes(tree: ast.Module) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    candidates: list[tuple[str, ast.FunctionDef]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            candidates.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            candidates.extend(
                (f"{node.name}.{member.name}", member)
                for member in node.body
                if isinstance(member, ast.FunctionDef)
            )
    for qualified_name, function in candidates:
        arguments = {
            argument.arg
            for argument in (
                list(function.args.posonlyargs)
                + list(function.args.args)
                + list(function.args.kwonlyargs)
            )
        }
        if function.args.vararg is not None:
            arguments.add(function.args.vararg.arg)
        if function.args.kwarg is not None:
            arguments.add(function.args.kwarg.arg)
        if "cfg" in arguments:
            continue
        visitor = _CfgLoadVisitor()
        for statement in function.body:
            visitor.visit(statement)
        if visitor.lines:
            scopes.append(
                {"scope": qualified_name, "lines": sorted(set(visitor.lines))}
            )
    return sorted(scopes, key=lambda item: item["scope"])


def runtime_static_controls(tree: ast.Module, config_tree: ast.Module) -> dict[str, Any]:
    torch_load_calls: list[dict[str, Any]] = []
    cuda_literal_lines: list[int] = []
    torch_cuda_call_lines: list[int] = []
    torch_device_cuda_lines: list[int] = []
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
            torch_load_calls.append(
                {
                    "line": node.lineno,
                    "map_location_literals": [
                        value
                        for keyword in node.keywords
                        if keyword.arg == "map_location"
                        for value in string_constants(keyword.value)
                    ],
                    "weights_only_declared": any(
                        keyword.arg == "weights_only" for keyword in node.keywords
                    ),
                }
            )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "torch"
            and node.func.value.attr == "cuda"
        ):
            torch_cuda_call_lines.append(node.lineno)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "device"
            and "cuda" in string_constants(node)
        ):
            torch_device_cuda_lines.append(node.lineno)

    defaults: dict[str, dict[str, Any]] = {}
    for node in ast.walk(config_tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "cfg"
            ):
                defaults[target.attr] = {
                    "line": node.lineno,
                    "string_literals": string_constants(node.value),
                    "constant": node.value.value
                    if isinstance(node.value, ast.Constant)
                    else None,
                }
    globals_found = global_cfg_scopes(tree)
    return {
        "torch_load_calls": sorted(torch_load_calls, key=lambda item: item["line"]),
        "cuda_literal_lines": sorted(set(cuda_literal_lines)),
        "torch_cuda_call_lines": sorted(set(torch_cuda_call_lines)),
        "torch_device_cuda_lines": sorted(set(torch_device_cuda_lines)),
        "global_cfg_scopes": globals_found,
        "config_defaults": defaults,
        "checkpoint_load_detected": len(torch_load_calls) == 1,
        "hardcoded_cuda_detected": bool(torch_device_cuda_lines and torch_cuda_call_lines),
        "global_cfg_blocker_detected": set(EXPECTED_GLOBAL_CFG_SCOPES).issubset(
            {item["scope"] for item in globals_found}
        ),
    }


def config_defaults_match(controls: dict[str, Any]) -> bool:
    defaults = controls["config_defaults"]
    expected_strings = {
        "pretrained_path": "../checkpoints/ngc/exp_54.pt",
        "data_path": "../re-arc/%s/tasks",
        "input_path": "../input/arc-prize-2024/",
        "device": "cuda",
    }
    if not all(
        key in defaults and value in defaults[key]["string_literals"]
        for key, value in expected_strings.items()
    ):
        return False
    return (
        "tune_model" in defaults and defaults["tune_model"]["constant"] is True
    )


def evidence_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    unique = sorted({path.absolute() for path in paths}, key=lambda item: str(item))
    manifest: list[dict[str, Any]] = []
    for path in unique:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"evidence is not a regular non-symlink file: {path}")
        manifest.append(
            {
                "path": display_path(path),
                "bytes": metadata.st_size,
                "sha256": sha256(path),
            }
        )
    return manifest


def audit(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    config = validate_config(config)
    source_config = config["source"]

    # This is the security boundary: only lstat metadata is collected for all
    # forbidden artifacts, and any presence terminates before source hashing,
    # parsing, source-lock reads, or prior-evidence reads.
    forbidden_config = config["forbidden_artifacts"]
    forbidden = observe_forbidden_paths(
        {
            "checkpoint": forbidden_config["checkpoint_paths"],
            "re_arc": forbidden_config["re_arc_paths"],
            "fixed_size": forbidden_config["fixed_size_paths"],
            "solution": forbidden_config["solution_paths"],
        }
    )
    assert_forbidden_material_absent(forbidden)

    source_root = lexical_path(source_config["local_snapshot"])
    metadata_inventory = metadata_only_source_inventory(
        source_root,
        [item["path"] for item in source_config["expected_files"]],
        source_config["required_directories"],
        source_config["optional_cache_directories"],
        source_config["metadata_only_cache_files"],
    )
    tracked = tracked_allowlist(
        source_root, [item["path"] for item in source_config["expected_files"]]
    )

    inventory, tree_digest, prefixed_tree_digest, python_tree_digest = source_inventory(
        source_root, source_config["expected_files"]
    )
    runner_path = contained_regular_file(
        source_root, config["static_contract"]["runner_path"]
    )
    upstream_config_path = contained_regular_file(
        source_root, config["static_contract"]["config_path"]
    )
    expected_hashes = {
        item["path"]: item["sha256"] for item in source_config["expected_files"]
    }
    runner_tree = parse_python(
        runner_path, expected_hashes[config["static_contract"]["runner_path"]]
    )
    upstream_config_tree = parse_python(
        upstream_config_path,
        expected_hashes[config["static_contract"]["config_path"]],
    )
    flow = solution_and_tune_flow(runner_tree)
    runtime_controls = runtime_static_controls(runner_tree, upstream_config_tree)

    source_lock_path = contained_regular_file(ROOT, source_config["source_lock_path"])
    source_lock = json.loads(read_regular_text(source_lock_path))
    source_lock_entry = source_lock["sources"]["2d-ngpt"]
    source_lock_matches = (
        sha256(source_lock_path) == source_config["source_lock_sha256"]
        and source_lock_entry["url"] == source_config["repository_url"]
        and source_lock_entry["revision"] == source_config["expected_revision"]
    )
    revision_object = run_git(
        "cat-file", "-e", f"{source_config['expected_revision']}^{{commit}}"
    )
    prepared_checkout = lexical_path(source_config["prepared_checkout"])
    prepared_present = os.path.isdir(prepared_checkout)
    independent_checkout = os.path.isdir(source_root / ".git")
    license_files = sorted(
        path.name
        for path in (contained_regular_file(source_root, "LICENSE"),)
    )

    prior_config = config["prior_architecture_evidence"]
    prior_run_path = contained_regular_file(ROOT, prior_config["run_path"])
    prior_runner_path = contained_regular_file(ROOT, prior_config["runner_path"])
    prior_record = json.loads(read_regular_text(prior_run_path))
    prior_matches = (
        sha256(prior_run_path) == prior_config["run_sha256"]
        and sha256(prior_runner_path) == prior_config["runner_sha256"]
        and prior_record.get("status") == "passed"
        and prior_record.get("configuration", {}).get("synthetic_untrained_weights")
        is True
        and prior_record.get("source", {}).get("revision")
        == source_config["expected_revision"]
    )

    artifact_present_count = sum(item["present"] for item in forbidden)
    source_matches = (
        len(inventory) == source_config["expected_file_count"]
        and sum(item["bytes"] for item in inventory)
        == source_config["expected_total_bytes"]
        and tree_digest == source_config["expected_tree_sha256"]
        and prefixed_tree_digest
        == source_config["expected_external_sources_prefixed_tree_sha256"]
        and all(item["matched"] for item in inventory)
    )
    provenance_observation_matches = (
        independent_checkout == source_config["expected_independent_git_checkout"]
        and prepared_present == source_config["expected_prepared_checkout_present"]
        and (revision_object.returncode == 0)
        == source_config["expected_revision_object_present_in_root_git"]
    )
    expected_solution_literals = sorted(
        config["static_contract"]["expected_solution_literals"]
    )
    solution_flow_matches = (
        flow["flow_detected"]
        and flow["solution_literals"] == expected_solution_literals
    )
    torch_load_matches = (
        runtime_controls["checkpoint_load_detected"]
        and runtime_controls["torch_load_calls"][0]["map_location_literals"] == ["cpu"]
    )

    evidence_paths = [
        config_path,
        Path(__file__).resolve(),
        source_lock_path,
        prior_run_path,
        prior_runner_path,
        *[contained_regular_file(source_root, item["path"]) for item in inventory],
    ]
    manifest = evidence_manifest(evidence_paths)
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_by_path = {item["path"]: item for item in manifest}
    manifest_source_matches_inventory = all(
        manifest_by_path[
            (source_root / item["path"]).relative_to(ROOT).as_posix()
        ]["sha256"]
        == item["sha256"]
        for item in inventory
    )

    validation = {
        "config_contract_validated": True,
        "forbidden_artifacts_absent_before_source_reads": artifact_present_count == 0,
        "filesystem_closed_before_source_reads": metadata_inventory["closed_world"],
        "metadata_only_caches_never_read_or_manifested": (
            metadata_inventory["cache_bytes_read"] is False
            and metadata_inventory["ignored_cache_executable_content_trusted"] is False
            and metadata_inventory["runtime_source_tree_approved"] is False
            and all(
                item["bytes_read"] is False and item["manifest_included"] is False
                for item in metadata_inventory["ignored_cache_inventory"]
            )
        ),
        "git_tracked_allowlist_matches": tracked
        == [item["path"] for item in source_config["expected_files"]],
        "source_snapshot_matches": source_matches,
        "evidence_manifest_source_bytes_match_inventory": manifest_source_matches_inventory,
        "source_lock_matches": source_lock_matches,
        "source_provenance_observation_matches": provenance_observation_matches,
        "license_observation_matches": license_files
        == source_config["expected_root_license_files"],
        "solution_to_tune_validation_flow_detected": solution_flow_matches,
        "torch_load_detected": torch_load_matches,
        "hardcoded_cuda_detected": runtime_controls["hardcoded_cuda_detected"],
        "global_cfg_blocker_detected": runtime_controls[
            "global_cfg_blocker_detected"
        ],
        "upstream_config_defaults_match": config_defaults_match(runtime_controls),
        "prior_architecture_component_evidence_matches": prior_matches,
        "static_controls_are_fail_closed": set(config["controls"])
        == EXPECTED_CONTROL_KEYS
        and all(config["controls"][key] is False for key in EXPECTED_CONTROL_KEYS),
    }

    deterministic_observations = {
        "config_sha256": sha256(config_path),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "evidence_manifest_sha256": manifest_digest,
        "source_inventory": inventory,
        "metadata_only_source_inventory": metadata_inventory,
        "source_tree_sha256": tree_digest,
        "external_sources_prefixed_tree_sha256": prefixed_tree_digest,
        "python_tree_sha256": python_tree_digest,
        "source_lock_matches": source_lock_matches,
        "independent_git_checkout": independent_checkout,
        "prepared_checkout_present": prepared_present,
        "revision_object_present_in_root_git": revision_object.returncode == 0,
        "forbidden_artifacts": forbidden,
        "solution_and_tune_flow": flow,
        "runtime_static_controls": runtime_controls,
        "prior_component_evidence_matches": prior_matches,
        "blocker_ids": EXPECTED_BLOCKER_IDS,
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
        "method_id": "2d-ngpt",
        "run_id": "",
        "runner": "scripts.audit_2d_ngpt_gates",
        "status": "passed" if all(validation.values()) else "failed",
        "status_semantics": "Frozen static blocker observations matched the prospective config; method execution remains blocked.",
        "scope": config["scope"],
        "method_gate_status": "blocked",
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "solver_gate_passed": False,
        "ignored_cache_bytes_read": False,
        "ignored_cache_executable_content_trusted": False,
        "filesystem_has_ignored_cache": metadata_inventory[
            "filesystem_has_ignored_cache"
        ],
        "runtime_source_tree_approved": False,
        "blocker_ids": list(EXPECTED_BLOCKER_IDS),
        "gate_summary": {"blocked": len(EXPECTED_BLOCKER_IDS), "passed": 1},
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
            "arc_solution_bytes_read": False,
            "generated_re_arc_bytes_read": False,
            "fixed_size_bytes_read": False,
            "checkpoint_bytes_read": False,
            "checkpoint_loaded": False,
            "metadata_only_cache_bytes_read": False,
            "ignored_cache_bytes_read": False,
            "upstream_code_imported": False,
            "upstream_code_executed": False,
            "solver_executed": False,
            "predictions_generated": False,
        },
        "source": {
            "path": str(source_root),
            "repository_url": source_config["repository_url"],
            "declared_revision": source_config["expected_revision"],
            "source_lock_path": source_config["source_lock_path"],
            "source_lock_sha256": sha256(source_lock_path),
            "source_lock_matches": source_lock_matches,
            "independent_git_checkout": independent_checkout,
            "prepared_checkout": str(prepared_checkout),
            "prepared_checkout_present": prepared_present,
            "revision_object_present_in_root_git": revision_object.returncode == 0,
            "tracked_file_count": len(tracked),
            "tracked_paths": tracked,
            "filesystem_inventory": metadata_inventory,
            "tracked_file_inventory": inventory,
            "total_bytes": sum(item["bytes"] for item in inventory),
            "tree_sha256": tree_digest,
            "tree_digest_algorithm": source_config["expected_tree_digest_algorithm"],
            "external_sources_prefixed_tree_sha256": prefixed_tree_digest,
            "external_sources_prefixed_tree_digest_algorithm": "same_line_manifest_with_ARC-AGI-Challenge-2024_path_prefix",
            "historical_four_python_file_tree_sha256": python_tree_digest,
            "root_license_files": license_files,
        },
        "source_provenance_gate": {
            "status": "blocked",
            "reason": "The ten-file retained snapshot is byte-locked, but it is not an independent Git checkout, the prepared checkout is absent, and the declared commit object is unavailable locally.",
        },
        "license_gate": {
            "status": "passed",
            "root_license_files": license_files,
            "license_sha256": inventory[0]["sha256"],
            "reason": "The retained root LICENSE matches the prospectively locked Apache-2.0 text; this is a file-integrity observation, not legal advice.",
        },
        "forbidden_artifact_observations": forbidden,
        "checkpoint_gate": {
            "status": "blocked",
            "present_count": sum(
                item["present"] for item in forbidden if item["kind"] == "checkpoint"
            ),
            "reason": "No declared exp_50/exp_54 checkpoint is present and no model checksum or provenance lock is available.",
        },
        "re_arc_data_gate": {
            "status": "blocked",
            "present_count": sum(
                item["present"] for item in forbidden if item["kind"] == "re_arc"
            ),
            "reason": "The generated 10,000-example-per-task re-ARC corpus is absent and its exact revision and artifact hash are unspecified.",
        },
        "fixed_size_and_solution_data_gate": {
            "status": "blocked",
            "fixed_size_present_count": sum(
                item["present"] for item in forbidden if item["kind"] == "fixed_size"
            ),
            "solution_present_count": sum(
                item["present"] for item in forbidden if item["kind"] == "solution"
            ),
            "reason": "The fixed_size index and official training/evaluation solution aggregates are absent from the runner's declared layouts; none were opened by this audit.",
        },
        "label_firewall_gate": {
            "status": "blocked",
            **flow,
            "reason": "code/064.py statically injects evaluation solutions into test outputs; those outputs affect eval color permutations, TTT model.update/train_model batches, forward loss, and validation loss/accuracy helpers. No strict challenge-only adapter has been audited.",
        },
        "runtime_portability_gate": {
            "status": "blocked",
            **runtime_controls,
            "reason": "The runner statically performs direct torch.load, hard-codes CUDA in run(), and relies on module-global cfg in add_validation_data and ARCModel.forward.",
        },
        "reproduction_contract_gate": {
            "status": "blocked",
            "reason": "The retained workflow lacks code/063.py, cfg/cfg_063.py, an exact dependency lock, checkpoint checksums, and a complete single-device reproduction contract.",
        },
        "prior_architecture_evidence": {
            "run_path": display_path(prior_run_path),
            "run_sha256": sha256(prior_run_path),
            "runner_path": display_path(prior_runner_path),
            "runner_sha256": sha256(prior_runner_path),
            "matched": prior_matches,
            "classification": "component-evidence-only",
            "component_smoke_counted": True,
            "solver_prediction_smoke": False,
            "solver_evidence": False,
            "synthetic_untrained_weights": True,
            "parameters": prior_record.get("parameters"),
            "output_shape": prior_record.get("output_shape"),
        },
        "evidence_manifest": manifest,
        "evidence_manifest_sha256": manifest_digest,
        "fairness": {
            "performance_table_eligible": False,
            "strict_runtime_promotion_eligible": False,
            "evidence_scope": "blocker_audit",
            "reason": "No solver prediction was produced; artifact, label-firewall, runtime, and reproduction gates remain blocked.",
        },
        "validation": validation,
        "limitations": [
            "Only the exact ten root-Git-tracked retained files were opened after all forbidden-artifact and filesystem-closure checks passed.",
            "The two explicitly allowed __pycache__ files were observed with lstat metadata only; their bytes were never opened, hashed, or included in the evidence manifest.",
            "Checkpoint, re-ARC, fixed_size, and solution paths were checked only with lstat; no artifact byte was read.",
            "This audit imported and executed no upstream module and initialized no GPU or network client.",
            "The historical synthetic-weight architecture forward is component evidence only, not a solver smoke or prediction result.",
            "Artifact paths outside the exact prospective lists were not searched outside the closed retained source tree.",
        ],
    }


def failed_record(run_id: str, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method_id": "2d-ngpt",
        "run_id": run_id,
        "runner": "scripts.audit_2d_ngpt_gates",
        "status": "failed",
        "scope": "source-artifact-label-runtime-gate-audit-only",
        "method_gate_status": "blocked",
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "solver_gate_passed": False,
        "ignored_cache_bytes_read": False,
        "ignored_cache_executable_content_trusted": False,
        "filesystem_has_ignored_cache": False,
        "runtime_source_tree_approved": False,
        "started_at_utc": started_at,
        "controls": {
            "network_used": False,
            "gpu_requested": False,
            "gpu_api_initialized": False,
            "arc_solution_bytes_read": False,
            "generated_re_arc_bytes_read": False,
            "fixed_size_bytes_read": False,
            "checkpoint_bytes_read": False,
            "checkpoint_loaded": False,
            "metadata_only_cache_bytes_read": False,
            "ignored_cache_bytes_read": False,
            "upstream_code_imported": False,
            "upstream_code_executed": False,
            "solver_executed": False,
            "predictions_generated": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "ngpt2d_gate_v1.json"
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.absolute()
    output_directory = args.output_directory.absolute()
    try:
        reject_sensitive_control_path(config_path, "config path")
        reject_sensitive_control_path(output_directory, "output directory")
    except ValueError as error:
        parser.error(str(error))
    try:
        output_descriptor = prepare_fresh_output_directory(output_directory)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    started_at = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = failed_record(output_directory.name, started_at)
    try:
        config = json.loads(read_regular_text(config_path))
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
            "scope": "auditor-process-only; read-only Git subprocess excluded",
        }
        try:
            atomic_json_at(output_descriptor, record)
        finally:
            os.close(output_descriptor)

    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
