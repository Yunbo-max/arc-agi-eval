#!/usr/bin/env python3
"""Run BARC's static source, artifact, label-firewall, and resource gate.

The auditor is deliberately metadata-first.  It never imports or executes
BARC, opens ARC/answer JSON or JSONL bundles, opens pickle/bytecode/model
weights, initializes a GPU, or accesses the network.  Passing this audit
records blockers; it is not a solver smoke and produces no prediction.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tokenize
import io
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
CONFIG_ID = "barc-source-artifact-label-resource-gate-v1"
METHOD_ID = "barc"
SCOPE = "source-artifact-label-resource-gate-audit-only"
CANONICAL_CONFIG_RELATIVE = "configs/barc_gate_v1.json"
SOURCE_LOCK_RELATIVE = "configs/source_locks.json"
SOURCE_PATH = Path("/root/arc-paper-assets/sources/barc")
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "e2b66aecb4996526093490c046615c6f71e2e42fc199710b6dc5122e45fc2c56"
)
EXPECTED_SOURCE_LOCK_SHA256 = (
    "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
)
EXPECTED_SOURCE_LOCK_ENTRY = {
    "url": "https://github.com/xu3kev/BARC",
    "branch": "master",
    "revision": "a7b51a6b1ff969da3a78a71c533b6d79a93966e7",
    "asset_subpath": "sources/barc",
}
EXPECTED_REVISION = "a7b51a6b1ff969da3a78a71c533b6d79a93966e7"
EXPECTED_COMMIT_TREE = "55ea72e3290ef7d3ec0ebed3554a9a60b83110ad"
EXPECTED_GIT_TREE_LISTING_SHA256 = (
    "8f075066aad62340d1eea64a8987509776a2da96d267e703659bb7bd9e45447e"
)
EXPECTED_CLOSED_INVENTORY_METADATA_SHA256 = (
    "9a11fdd5c1b59c921e779b9e19938faaac365ef49d45184c79e75b666fad5428"
)
EXPECTED_BLOCKER_IDS = [
    "root-license",
    "base-artifact-provenance",
    "lora-artifact-provenance",
    "safe-offline-model-load",
    "label-firewall",
    "dependency-lock",
    "single-gpu-capacity",
    "solver-prediction-and-parity-contract",
]
EXPECTED_BASE_MODELS = [
    "barc0/Llama-3.1-ARC-Heavy-Induction-8B",
    "barc0/Llama-3.1-ARC-Heavy-Transduction-8B",
    "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    "barc0/Llama-3.1-ARC-Potpourri-Transduction-8B",
]
EXPECTED_LORAS = [
    "barc0/engineer1-heavy-barc-llama3.1-8b-instruct-lora64-testtime-finetuning",
    "barc0/heavy-barc-llama3.1-8b-instruct-lora64-testtime-finetuning",
]
EXPECTED_NEVER_READ_SUFFIXES = {
    ".json",
    ".jsonl",
    ".pkl",
    ".pickle",
    ".pyc",
    ".png",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
}
MODEL_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
}
PYTHON_PATHS = {
    "data_processing/test-time-finetune/gen_transduction_only_formatted.py",
    "data_processing/test-time-finetune/get_pseudo_eval_task.py",
    "eval_code_samples.py",
    "evaluation.py",
    "execution.py",
    "finetune/inference/vllm_inference.py",
    "finetune/inference/vllm_inference_transduction_concept_arc.py",
    "finetune/inference/vllm_inference_transduction_evaluation.py",
    "finetune/inference/vllm_transduction_reranking.py",
}
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_SOURCE_TEXT_BYTES = 512 * 1024
GIT_CONFIG_PATH = ".git/config"
GIT_COMMAND_TIMEOUT_SECONDS = 10.0
GIT_STDERR_MAX_BYTES = 16 * 1024
GIT_COMMAND_STDOUT_LIMITS = {
    ("rev-parse", "--verify", EXPECTED_REVISION): 128,
    ("rev-parse", "--verify", f"{EXPECTED_REVISION}^{{tree}}"): 128,
    ("ls-tree", "-rz", "--full-tree", EXPECTED_REVISION): 512 * 1024,
}
RENAME_NOREPLACE = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
GIT_TREE_RECORD_RE = re.compile(
    rb"^(?P<mode>[0-7]{6}) (?P<type>blob|commit) "
    rb"(?P<oid>[0-9a-f]{40})\t(?P<path>.+)$"
)


class OutputPathError(ValueError):
    """The requested output is not a fresh, pinned directory leaf."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(payload: bytes, field: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON: {error}") from error


def safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a contained POSIX relative path")
    if path.as_posix() != value:
        raise ValueError(f"{field} must use canonical POSIX syntax")
    return value


def _absolute_parts(path: Path) -> tuple[str, ...]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if not absolute.is_absolute() or any(part == ".." for part in absolute.parts):
        raise ValueError(f"unsafe absolute path: {path}")
    return absolute.parts


def lexical_path(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def lexical_within(path: Path, parent: Path) -> bool:
    path_parts = _absolute_parts(path)
    parent_parts = _absolute_parts(parent)
    return path_parts[: len(parent_parts)] == parent_parts


def stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def require_secure_open_flags() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC"):
        if not hasattr(os, name) or not isinstance(getattr(os, name), int):
            raise RuntimeError(f"required secure-open flag unavailable: {name}")


def directory_flags() -> int:
    require_secure_open_flags()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def regular_file_flags() -> int:
    require_secure_open_flags()
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def open_absolute_directory(path: Path) -> int:
    parts = _absolute_parts(path)
    descriptor = os.open("/", directory_flags())
    try:
        for part in parts[1:]:
            next_descriptor = os.open(part, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError(f"not a non-symlink directory: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_absolute_parent(path: Path) -> tuple[int, str]:
    parts = _absolute_parts(path)
    if len(parts) < 2 or parts[-1] in {"", ".", ".."}:
        raise ValueError(f"unsafe leaf path: {path}")
    return open_absolute_directory(Path(*parts[:-1])), parts[-1]


def open_relative_parent(root_fd: int, path: str) -> tuple[int, str]:
    parts = PurePosixPath(safe_relative_path(path, "source read path")).parts
    descriptor = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def verify_directory_path_identity(path: Path, pinned_fd: int) -> None:
    reopened = open_absolute_directory(path)
    try:
        if stat_signature(os.fstat(reopened)) != stat_signature(os.fstat(pinned_fd)):
            raise RuntimeError(f"directory path identity changed during audit: {path}")
    finally:
        os.close(reopened)


def _read_fd_stable(
    descriptor: int,
    before: os.stat_result,
    max_bytes: int,
    on_chunk: Any | None = None,
) -> tuple[bytes, os.stat_result]:
    if before.st_size > max_bytes:
        raise ValueError(f"verified file exceeds maximum size: {before.st_size}")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if on_chunk is not None:
            on_chunk(chunk)
        if total > max_bytes:
            raise ValueError("verified file grew beyond its size limit")
    after = os.fstat(descriptor)
    if stat_signature(before) != stat_signature(after):
        raise RuntimeError("file changed while verified bytes were read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise RuntimeError("verified read length does not match fstat size")
    return payload, after


class ReadAttempt:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.hasher = hashlib.sha256()


class ReadLedger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.git_subprocesses: list[dict[str, Any]] = []
        self._config_binding: tuple[str, ...] | None = None
        self._source_policy: dict[str, str] = {}

    def bind_config(self, path: Path) -> None:
        binding = _absolute_parts(path)
        expected = _absolute_parts(ROOT / CANONICAL_CONFIG_RELATIVE)
        if binding != expected:
            raise ValueError("config reader is bound only to the canonical BARC config")
        if self._config_binding is not None and self._config_binding != binding:
            raise ValueError("config reader binding cannot be changed")
        self._config_binding = binding

    def bind_source_policy(self, declarations: list[dict[str, Any]]) -> None:
        policy: dict[str, str] = {}
        for item in declarations:
            path = safe_relative_path(item["path"], "retained source path")
            role = item["role"]
            if path in policy or role not in {
                "source_python",
                "source_documentation",
                "dependency_manifest",
                "artifact_recipe",
                "vendored_license",
            }:
                raise ValueError("invalid retained source read policy")
            if PurePosixPath(path).suffix.lower() in EXPECTED_NEVER_READ_SUFFIXES:
                raise ValueError("retained source policy intersects a never-read suffix")
            policy[path] = role
        if self._source_policy and self._source_policy != policy:
            raise ValueError("source reader policy cannot be changed")
        self._source_policy = policy

    def authorize_absolute(self, path: Path, role: str) -> str:
        parts = _absolute_parts(path)
        if role == "canonical_config":
            if self._config_binding is None or parts != self._config_binding:
                raise ValueError("canonical config role/path binding mismatch")
            return "gate_config"
        if role == "source_lock":
            if parts != _absolute_parts(ROOT / SOURCE_LOCK_RELATIVE):
                raise ValueError("source-lock reader is not bound to its canonical path")
            return "source_lock"
        raise ValueError(f"untrusted absolute reader role: {role}")

    def authorize_relative(self, path: str, role: str) -> str:
        safe_relative_path(path, "verified source reader path")
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in EXPECTED_NEVER_READ_SUFFIXES:
            raise ValueError(f"metadata-only suffix may never enter a reader: {path}")
        if self._source_policy.get(path) != role:
            raise ValueError(f"source reader role/path binding mismatch: {role}:{path}")
        return role

    def authorize_git_metadata(self, path: str, role: str) -> str:
        safe_relative_path(path, "Git metadata reader path")
        allowed = {
            (GIT_CONFIG_PATH, "git_local_config"),
            (".git/HEAD", "git_head"),
        }
        if (path, role) not in allowed:
            raise ValueError(f"Git metadata reader role/path binding mismatch: {role}:{path}")
        return role

    def _begin(self, path: str, role: str, category: str) -> ReadAttempt:
        record: dict[str, Any] = {
            "path": path,
            "role": role,
            "category": category,
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "read_status": "authorized-attempt",
            "validation_status": "pending",
        }
        self.records.append(record)
        return ReadAttempt(record)

    def begin_absolute(
        self, path: Path, role: str, display: str
    ) -> ReadAttempt:
        return self._begin(display, role, self.authorize_absolute(path, role))

    def begin_relative(self, path: str, role: str) -> ReadAttempt:
        return self._begin(path, role, self.authorize_relative(path, role))

    def begin_git_metadata(self, path: str, role: str) -> ReadAttempt:
        return self._begin(path, role, self.authorize_git_metadata(path, role))

    @staticmethod
    def note_chunk(attempt: ReadAttempt, chunk: bytes) -> None:
        attempt.hasher.update(chunk)
        attempt.record["bytes"] += len(chunk)
        attempt.record["sha256"] = attempt.hasher.copy().hexdigest()
        attempt.record["read_status"] = "reading"

    @staticmethod
    def finish_read(attempt: ReadAttempt) -> None:
        attempt.record["sha256"] = attempt.hasher.hexdigest()
        attempt.record["read_status"] = "completed"

    @staticmethod
    def finish_validation(attempt: ReadAttempt, passed: bool) -> None:
        attempt.record["validation_status"] = "passed" if passed else "failed"

    @staticmethod
    def fail(attempt: ReadAttempt, error: BaseException) -> None:
        if attempt.record["read_status"] != "completed":
            attempt.record["read_status"] = "failed"
        if attempt.record["validation_status"] == "pending":
            attempt.record["validation_status"] = "failed"
        attempt.record["failure_type"] = type(error).__name__

    def begin_git_subprocess(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "argv": ["/usr/bin/git", *arguments],
            "status": "authorized-attempt",
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "worktree_content_requested": False,
            "potential_repository_reads": (
                "auditor-owned isolated Git-dir/object view backed only by pinned "
                "object-file descriptors; kernel-level byte volume is not measured"
            ),
            "untrusted_repository_local_config_available": False,
        }
        self.git_subprocesses.append(record)
        return record

    @staticmethod
    def finish_git_subprocess(
        record: dict[str, Any], *, returncode: int, stdout_bytes: int, stderr_bytes: int
    ) -> None:
        record.update(
            {
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "stdout_bytes": stdout_bytes,
                "stderr_bytes": stderr_bytes,
            }
        )

    @staticmethod
    def fail_git_subprocess(record: dict[str, Any], error: BaseException) -> None:
        record["status"] = "failed"
        record["failure_type"] = type(error).__name__

    def snapshot(self) -> dict[str, Any]:
        return {
            "file_read_attempts": [dict(item) for item in self.records],
            "git_subprocesses": [dict(item) for item in self.git_subprocesses],
        }

    def count(self, category: str) -> int:
        return sum(item["category"] == category for item in self.records)


def secure_read_absolute(
    path: Path,
    *,
    max_bytes: int,
    role: str,
    ledger: ReadLedger,
    display: str | None = None,
) -> tuple[bytes, os.stat_result]:
    attempt = ledger.begin_absolute(path, role, display or str(path))
    try:
        parent_fd, leaf = open_absolute_parent(path)
        descriptor: int | None = None
        try:
            before_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(before_path.st_mode) or stat.S_ISLNK(
                before_path.st_mode
            ):
                raise ValueError(f"verified path is not a regular non-symlink: {path}")
            if before_path.st_nlink != 1:
                raise ValueError(f"verified path must have one hard link: {path}")
            descriptor = os.open(leaf, regular_file_flags(), dir_fd=parent_fd)
            before_fd = os.fstat(descriptor)
            if stat_signature(before_path) != stat_signature(before_fd):
                raise RuntimeError(f"file identity changed before read: {path}")
            payload, after_fd = _read_fd_stable(
                descriptor,
                before_fd,
                max_bytes,
                lambda chunk: ledger.note_chunk(attempt, chunk),
            )
            ledger.finish_read(attempt)
            after_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if stat_signature(after_fd) != stat_signature(after_path):
                raise RuntimeError(f"file identity changed after read: {path}")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_fd)
        ledger.finish_validation(attempt, True)
        return payload, after_fd
    except BaseException as error:
        ledger.fail(attempt, error)
        raise


def secure_read_relative(
    root_fd: int,
    declaration: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
    ledger: ReadLedger,
) -> bytes:
    path = declaration["path"]
    role = declaration["role"]
    attempt = ledger.begin_relative(path, role)
    try:
        if path not in snapshot or snapshot[path]["kind"] != "tracked":
            raise ValueError(f"retained source path missing from inventory: {path}")
        parent_fd, leaf = open_relative_parent(root_fd, path)
        descriptor: int | None = None
        try:
            before_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if stat_signature(before_path) != snapshot[path]["signature"]:
                raise RuntimeError(f"retained source changed after inventory: {path}")
            if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
                raise ValueError(
                    f"retained source is not a single-link regular file: {path}"
                )
            descriptor = os.open(leaf, regular_file_flags(), dir_fd=parent_fd)
            before_fd = os.fstat(descriptor)
            if stat_signature(before_path) != stat_signature(before_fd):
                raise RuntimeError(f"retained source raced before read: {path}")
            payload, after_fd = _read_fd_stable(
                descriptor,
                before_fd,
                MAX_SOURCE_TEXT_BYTES,
                lambda chunk: ledger.note_chunk(attempt, chunk),
            )
            ledger.finish_read(attempt)
            after_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if stat_signature(after_fd) != stat_signature(after_path):
                raise RuntimeError(f"retained source raced after read: {path}")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_fd)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != declaration["bytes"]:
            ledger.finish_validation(attempt, False)
            raise ValueError(f"retained source byte count mismatch: {path}")
        if digest != declaration["sha256"]:
            ledger.finish_validation(attempt, False)
            raise ValueError(f"retained source SHA-256 mismatch: {path}")
        ledger.finish_validation(attempt, True)
        return payload
    except BaseException as error:
        ledger.fail(attempt, error)
        raise


def validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("BARC gate config must be an object")
    if canonical_sha256(value) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise ValueError("BARC gate config does not match the hardcoded v1 contract")
    if value.get("config_id") != CONFIG_ID or value.get("method_id") != METHOD_ID:
        raise ValueError("BARC gate config identity mismatch")
    if value.get("scope") != SCOPE or value.get("counted_toward_smoke") is not False:
        raise ValueError("BARC gate config claim boundary mismatch")
    return value


def validate_config_location(config_path: Path, output_path: Path) -> None:
    del output_path
    if _absolute_parts(config_path) != _absolute_parts(
        ROOT / CANONICAL_CONFIG_RELATIVE
    ):
        raise ValueError("production config path must equal configs/barc_gate_v1.json")


def _git_config_entries(payload: bytes) -> dict[tuple[str, str], str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("BARC local Git config is not UTF-8") from error
    section: str | None = None
    entries: dict[tuple[str, str], str] = {}
    allowed_sections = {
        "core",
        'remote "origin"',
        'branch "master"',
    }
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section not in allowed_sections:
                raise ValueError(
                    f"forbidden or unknown section in BARC local Git config: {section}"
                )
            continue
        if section is None or "=" not in line:
            raise ValueError(f"malformed BARC local Git config line: {line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        normalized = (section, key.lower())
        if not key or normalized in entries:
            raise ValueError("duplicate or empty BARC local Git config key")
        entries[normalized] = value
    expected = {
        ("core", "repositoryformatversion"): "1",
        ("core", "filemode"): "true",
        ("core", "bare"): "false",
        ("core", "logallrefupdates"): "true",
        ('remote "origin"', "url"): "https://github.com/xu3kev/BARC",
        ('remote "origin"', "fetch"): "+refs/heads/*:refs/remotes/origin/*",
        ('remote "origin"', "promisor"): "true",
        ('remote "origin"', "partialclonefilter"): "blob:none",
        ('branch "master"', "remote"): "origin",
        ('branch "master"', "merge"): "refs/heads/master",
    }
    if entries != expected:
        raise ValueError("BARC local Git config entries differ from the safe allowlist")
    return entries


def secure_read_git_metadata_file(
    git_dir_fd: int,
    declaration: dict[str, Any],
    ledger: ReadLedger,
    role: str,
) -> tuple[bytes, tuple[int, ...]]:
    path = safe_relative_path(declaration["path"], "Git config contract path")
    prefix = ".git/"
    if not path.startswith(prefix) or "/" in path[len(prefix) :]:
        raise ValueError("BARC Git metadata file contract is not a direct .git leaf")
    leaf = path[len(prefix) :]
    attempt = ledger.begin_git_metadata(path, role)
    descriptor: int | None = None
    try:
        before_path = os.stat(leaf, dir_fd=git_dir_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before_path.st_mode)
            or stat.S_ISLNK(before_path.st_mode)
            or before_path.st_nlink != 1
        ):
            raise ValueError(f"BARC {path} must be a single-link regular file")
        if format(stat.S_IMODE(before_path.st_mode), "04o") != declaration["mode"]:
            raise ValueError(f"BARC {path} mode mismatch")
        if before_path.st_size != declaration["bytes"]:
            raise ValueError(f"BARC {path} byte-count contract mismatch")
        descriptor = os.open(leaf, regular_file_flags(), dir_fd=git_dir_fd)
        before_fd = os.fstat(descriptor)
        if stat_signature(before_path) != stat_signature(before_fd):
            raise RuntimeError(f"BARC {path} raced before read")
        payload, after_fd = _read_fd_stable(
            descriptor,
            before_fd,
            MAX_CONFIG_BYTES,
            lambda chunk: ledger.note_chunk(attempt, chunk),
        )
        ledger.finish_read(attempt)
        after_path = os.stat(leaf, dir_fd=git_dir_fd, follow_symlinks=False)
        if stat_signature(after_fd) != stat_signature(after_path):
            raise RuntimeError(f"BARC {path} raced after read")
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != declaration["bytes"] or digest != declaration["sha256"]:
            ledger.finish_validation(attempt, False)
            raise ValueError(f"BARC {path} byte/SHA-256 contract mismatch")
        if role == "git_local_config":
            _git_config_entries(payload)
        ledger.finish_validation(attempt, True)
        return payload, stat_signature(after_fd)
    except BaseException as error:
        ledger.fail(attempt, error)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def secure_read_git_config(
    git_dir_fd: int,
    declaration: dict[str, Any],
    ledger: ReadLedger,
) -> tuple[bytes, tuple[int, ...]]:
    return secure_read_git_metadata_file(
        git_dir_fd, declaration, ledger, "git_local_config"
    )


def require_git_metadata_path_absent(git_dir_fd: int, repository_path: str) -> None:
    path = safe_relative_path(repository_path, "forbidden Git metadata path")
    prefix = ".git/"
    if not path.startswith(prefix):
        raise ValueError("forbidden Git metadata path must be inside .git")
    relative = path[len(prefix) :]
    parent_fd: int | None = None
    try:
        try:
            parent_fd, leaf = open_relative_parent(git_dir_fd, relative)
        except FileNotFoundError:
            return
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ValueError(f"forbidden auxiliary Git metadata path exists: {path}")
    except OSError as error:
        if error.errno == errno.ENOENT:
            return
        raise ValueError(f"unsafe parent in forbidden Git metadata path: {path}") from error
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _read_capped_git_process(
    process: subprocess.Popen[bytes],
    *,
    stdout_limit: int,
    stderr_limit: int,
    timeout_seconds: float,
    process_record: dict[str, Any],
) -> tuple[bytes, bytes, int]:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("fixed Git subprocess pipes were not created")
    streams = {
        process.stdout.fileno(): ("stdout", stdout_limit, bytearray()),
        process.stderr.fileno(): ("stderr", stderr_limit, bytearray()),
    }
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout_seconds
    try:
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("fixed Git metadata command exceeded its timeout")
            events = selector.select(remaining)
            if not events:
                raise TimeoutError("fixed Git metadata command exceeded its timeout")
            for key, _ in events:
                stream_name, limit, output = streams[key.fd]
                chunk = os.read(key.fd, min(64 * 1024, limit + 1 - len(output)))
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                output.extend(chunk)
                process_record[f"{stream_name}_bytes"] = len(output)
                if len(output) > limit:
                    raise RuntimeError(
                        f"fixed Git metadata command exceeded {stream_name} limit"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fixed Git metadata command exceeded its timeout")
        returncode = process.wait(timeout=remaining)
        return bytes(streams[process.stdout.fileno()][2]), bytes(
            streams[process.stderr.fileno()][2]
        ), returncode
    finally:
        selector.close()


def _git(
    isolated_git_fd: int,
    isolated_object_view_fd: int,
    pinned_object_fds: tuple[int, ...],
    ledger: ReadLedger,
    *arguments: str,
) -> bytes:
    command = tuple(arguments)
    if command not in GIT_COMMAND_STDOUT_LIMITS:
        raise ValueError(f"Git command is outside the BARC metadata allowlist: {command}")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_DIR": f"/proc/self/fd/{isolated_git_fd}",
        "GIT_OBJECT_DIRECTORY": f"/proc/self/fd/{isolated_object_view_fd}",
    }
    process_record = ledger.begin_git_subprocess(command)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            ["/usr/bin/git", *arguments],
            cwd=f"/proc/self/fd/{isolated_git_fd}",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            pass_fds=(
                isolated_git_fd,
                isolated_object_view_fd,
                *pinned_object_fds,
            ),
            close_fds=True,
            start_new_session=True,
        )
        stdout, stderr, returncode = _read_capped_git_process(
            process,
            stdout_limit=GIT_COMMAND_STDOUT_LIMITS[command],
            stderr_limit=GIT_STDERR_MAX_BYTES,
            timeout_seconds=GIT_COMMAND_TIMEOUT_SECONDS,
            process_record=process_record,
        )
        ledger.finish_git_subprocess(
            process_record,
            returncode=returncode,
            stdout_bytes=len(stdout),
            stderr_bytes=len(stderr),
        )
        if returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"fixed Git metadata command failed: {detail}")
        return stdout
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1.0)
        ledger.fail_git_subprocess(process_record, error)
        raise
    finally:
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


def parse_git_tree(payload: bytes) -> dict[str, dict[str, Any]]:
    if not payload or not payload.endswith(b"\0"):
        raise ValueError("Git tree listing is not NUL-terminated")
    result: dict[str, dict[str, Any]] = {}
    for raw_record in payload[:-1].split(b"\0"):
        match = GIT_TREE_RECORD_RE.fullmatch(raw_record)
        if match is None or match.group("type") != b"blob":
            raise ValueError("unsupported entry in locked Git tree")
        if match.group("mode") != b"100644":
            raise ValueError("unsupported mode in locked Git tree")
        try:
            path = match.group("path").decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Git tree path is not UTF-8") from error
        safe_relative_path(path, "Git tree path")
        if path in result:
            raise ValueError(f"duplicate Git tree path: {path}")
        result[path] = {
            "mode": match.group("mode").decode("ascii"),
            "blob_oid": match.group("oid").decode("ascii"),
        }
    return result


def extension_counts(paths: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        pure = PurePosixPath(path)
        name = pure.name.lower()
        suffix = (
            name
            if name.startswith(".") and name.count(".") == 1
            else pure.suffix.lower() or "<none>"
        )
        counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def initialize_isolated_git_directory(path: Path) -> int:
    """Create a minimal auditor-owned Git-dir with no local config."""

    descriptor = open_absolute_directory(path)
    head_fd: int | None = None
    try:
        os.mkdir("objects", mode=0o700, dir_fd=descriptor)
        os.mkdir("refs", mode=0o500, dir_fd=descriptor)
        head_fd = os.open(
            "HEAD",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=descriptor,
        )
        payload = (EXPECTED_REVISION + "\n").encode("ascii")
        if os.write(head_fd, payload) != len(payload):
            raise OSError("short write while creating isolated Git HEAD")
        os.fsync(head_fd)
        os.close(head_fd)
        head_fd = None
        os.fsync(descriptor)
        try:
            os.stat("config", dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("isolated Git-dir unexpectedly contains local config")
        return descriptor
    except BaseException:
        if head_fd is not None:
            os.close(head_fd)
        os.close(descriptor)
        raise


def git_object_store_metadata_inventory(root_fd: int) -> dict[str, tuple[int, ...]]:
    """Inventory object-store path/type/stat metadata without reading file bytes."""

    result: dict[str, tuple[int, ...]] = {}

    def visit(directory_fd: int, prefix: str) -> None:
        for entry in sorted(os.scandir(directory_fd), key=lambda item: item.name):
            name = entry.name
            if name in {"", ".", ".."} or "/" in name:
                raise ValueError("unsafe Git object-store entry")
            path = f"{prefix}/{name}" if prefix else name
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                result[path + "/"] = stat_signature(info)
                child_fd = os.open(name, directory_flags(), dir_fd=directory_fd)
                try:
                    if stat_signature(os.fstat(child_fd)) != stat_signature(info):
                        raise RuntimeError(
                            f"Git object-store directory raced: {path}"
                        )
                    visit(child_fd, path)
                finally:
                    os.close(child_fd)
                continue
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
            ):
                raise ValueError(f"unsafe Git object-store leaf: {path}")
            result[path] = stat_signature(info)

    visit(root_fd, "")
    return result


def populate_isolated_object_view(
    isolated_git_fd: int,
    source_objects_fd: int,
    inventory: dict[str, tuple[int, ...]],
) -> tuple[int, tuple[int, ...]]:
    """Expose pinned object-file descriptors through an auditor-owned view."""

    view_fd = os.open("objects", directory_flags(), dir_fd=isolated_git_fd)
    object_fds: list[int] = []
    try:
        directories = sorted(
            (path[:-1] for path in inventory if path.endswith("/")),
            key=lambda path: (path.count("/"), path),
        )
        for path in directories:
            parent_fd, leaf = open_relative_parent(view_fd, path)
            try:
                os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
        for path, expected_signature in sorted(inventory.items()):
            if path.endswith("/"):
                continue
            source_parent_fd, source_leaf = open_relative_parent(
                source_objects_fd, path
            )
            source_fd: int | None = None
            try:
                before = os.stat(
                    source_leaf,
                    dir_fd=source_parent_fd,
                    follow_symlinks=False,
                )
                if stat_signature(before) != expected_signature:
                    raise RuntimeError(f"Git object leaf changed before pinning: {path}")
                source_fd = os.open(
                    source_leaf, regular_file_flags(), dir_fd=source_parent_fd
                )
                if stat_signature(os.fstat(source_fd)) != expected_signature:
                    raise RuntimeError(f"Git object leaf raced while pinning: {path}")
                object_fds.append(source_fd)
                source_fd = None
            finally:
                if source_fd is not None:
                    os.close(source_fd)
                os.close(source_parent_fd)
            view_parent_fd, view_leaf = open_relative_parent(view_fd, path)
            try:
                os.symlink(
                    f"/proc/self/fd/{object_fds[-1]}",
                    view_leaf,
                    dir_fd=view_parent_fd,
                )
            finally:
                os.close(view_parent_fd)
        return view_fd, tuple(object_fds)
    except BaseException:
        for descriptor in object_fds:
            os.close(descriptor)
        os.close(view_fd)
        raise


def validate_git_metadata_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["source"]["git_metadata_contract"]
    expected_commands = [list(command) for command in GIT_COMMAND_STDOUT_LIMITS]
    if contract != {
        "local_config": {
            "path": GIT_CONFIG_PATH,
            "bytes": 304,
            "sha256": "b99b39f2e5e40142bc7030ddf55ad6db560d560360a7879a7ab3e04b521d8f6f",
            "mode": "0644",
        },
        "head": {
            "path": ".git/HEAD",
            "bytes": 41,
            "sha256": "5f2f32d25df3807c3450a68f58322cd74b01ddc8aa9b88004418ba30fd95384c",
            "mode": "0644",
        },
        "required_absent_paths": [
            ".git/commondir",
            ".git/gitdir",
            ".git/config.worktree",
            ".git/info/attributes",
            ".git/objects/info/alternates",
            ".git/objects/info/http-alternates",
        ],
        "allowed_commands": expected_commands,
        "timeout_seconds": GIT_COMMAND_TIMEOUT_SECONDS,
        "stdout_max_bytes": max(GIT_COMMAND_STDOUT_LIMITS.values()),
        "stderr_max_bytes": GIT_STDERR_MAX_BYTES,
        "lazy_fetch_disabled": True,
        "terminal_prompt_disabled": True,
        "isolated_git_dir_required": True,
    }:
        raise ValueError("BARC Git metadata subprocess contract mismatch")
    return contract


def verify_git_contract(
    root_fd: int, config: dict[str, Any], ledger: ReadLedger
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    source = config["source"]
    git_contract = validate_git_metadata_contract(config)
    git_path_before = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
    if not stat.S_ISDIR(git_path_before.st_mode) or stat.S_ISLNK(
        git_path_before.st_mode
    ):
        raise ValueError("BARC .git must be a real directory")
    git_dir_fd = os.open(".git", directory_flags(), dir_fd=root_fd)
    try:
        if stat_signature(git_path_before) != stat_signature(os.fstat(git_dir_fd)):
            raise RuntimeError("BARC .git directory raced before metadata verification")
        config_before, config_signature_before = secure_read_git_config(
            git_dir_fd, git_contract["local_config"], ledger
        )
        head_before, head_signature_before = secure_read_git_metadata_file(
            git_dir_fd, git_contract["head"], ledger, "git_head"
        )
        if head_before != (EXPECTED_REVISION + "\n").encode("ascii"):
            raise ValueError("BARC .git/HEAD is not the locked detached revision")
        entries = _git_config_entries(config_before)
        remote = entries[('remote "origin"', "url")]
        for path in git_contract["required_absent_paths"]:
            require_git_metadata_path_absent(git_dir_fd, path)

        objects_path_before = os.stat(
            "objects", dir_fd=git_dir_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(objects_path_before.st_mode) or stat.S_ISLNK(
            objects_path_before.st_mode
        ):
            raise ValueError("BARC .git/objects must be a real directory")
        objects_fd = os.open("objects", directory_flags(), dir_fd=git_dir_fd)
        try:
            objects_signature = stat_signature(os.fstat(objects_fd))
            if objects_signature != stat_signature(objects_path_before):
                raise RuntimeError("BARC .git/objects raced before Git verification")
            objects_inventory = git_object_store_metadata_inventory(objects_fd)
            with tempfile.TemporaryDirectory(
                prefix="barc-isolated-git-", dir="/tmp"
            ) as safe_path:
                isolated_git_fd = initialize_isolated_git_directory(Path(safe_path))
                isolated_object_view_fd: int | None = None
                pinned_object_fds: tuple[int, ...] = ()
                try:
                    (
                        isolated_object_view_fd,
                        pinned_object_fds,
                    ) = populate_isolated_object_view(
                        isolated_git_fd, objects_fd, objects_inventory
                    )
                    revision = _git(
                        isolated_git_fd,
                        isolated_object_view_fd,
                        pinned_object_fds,
                        ledger,
                        "rev-parse",
                        "--verify",
                        EXPECTED_REVISION,
                    ).strip().decode("ascii")
                    tree = _git(
                        isolated_git_fd,
                        isolated_object_view_fd,
                        pinned_object_fds,
                        ledger,
                        "rev-parse",
                        "--verify",
                        f"{EXPECTED_REVISION}^{{tree}}",
                    ).strip().decode("ascii")
                    listing = _git(
                        isolated_git_fd,
                        isolated_object_view_fd,
                        pinned_object_fds,
                        ledger,
                        "ls-tree",
                        "-rz",
                        "--full-tree",
                        EXPECTED_REVISION,
                    )
                    try:
                        os.stat(
                            "config",
                            dir_fd=isolated_git_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        raise RuntimeError(
                            "isolated Git-dir local config appeared during verification"
                        )
                    expected_file_signatures = [
                        signature
                        for path, signature in sorted(objects_inventory.items())
                        if not path.endswith("/")
                    ]
                    if [
                        stat_signature(os.fstat(descriptor))
                        for descriptor in pinned_object_fds
                    ] != expected_file_signatures:
                        raise RuntimeError(
                            "pinned Git object file changed during verification"
                        )
                finally:
                    for descriptor in pinned_object_fds:
                        os.close(descriptor)
                    if isolated_object_view_fd is not None:
                        os.close(isolated_object_view_fd)
                    os.close(isolated_git_fd)
            objects_path_after = os.stat(
                "objects", dir_fd=git_dir_fd, follow_symlinks=False
            )
            if (
                stat_signature(objects_path_after) != objects_signature
                or stat_signature(os.fstat(objects_fd)) != objects_signature
                or git_object_store_metadata_inventory(objects_fd)
                != objects_inventory
            ):
                raise RuntimeError("BARC .git/objects changed during Git verification")
        finally:
            os.close(objects_fd)

        config_after, config_signature_after = secure_read_git_config(
            git_dir_fd, git_contract["local_config"], ledger
        )
        head_after, head_signature_after = secure_read_git_metadata_file(
            git_dir_fd, git_contract["head"], ledger, "git_head"
        )
        for path in git_contract["required_absent_paths"]:
            require_git_metadata_path_absent(git_dir_fd, path)
        if (
            config_after != config_before
            or config_signature_after != config_signature_before
            or head_after != head_before
            or head_signature_after != head_signature_before
        ):
            raise RuntimeError("BARC .git/config or HEAD changed during Git metadata commands")
        git_path_after = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        if stat_signature(git_path_after) != stat_signature(git_path_before):
            raise RuntimeError("BARC .git directory changed during metadata verification")
    finally:
        os.close(git_dir_fd)

    listing_sha256 = hashlib.sha256(listing).hexdigest()
    tracked = parse_git_tree(listing)
    if revision != EXPECTED_REVISION or revision != source["expected_revision"]:
        raise ValueError("BARC Git revision mismatch")
    if tree != EXPECTED_COMMIT_TREE or tree != source["expected_commit_tree"]:
        raise ValueError("BARC Git commit tree mismatch")
    if remote != source["repository_url"]:
        raise ValueError("BARC Git remote URL mismatch")
    if listing_sha256 != EXPECTED_GIT_TREE_LISTING_SHA256 or listing_sha256 != source[
        "git_tree_listing_sha256"
    ]:
        raise ValueError("BARC Git tree listing digest mismatch")
    if len(tracked) != source["expected_tracked_file_count"]:
        raise ValueError("BARC tracked file count mismatch")
    if extension_counts(set(tracked)) != source["expected_extension_counts"]:
        raise ValueError("BARC tracked extension inventory mismatch")
    if any(item["mode"] != "100644" for item in tracked.values()):
        raise ValueError("BARC locked tree contains a non-100644 blob")
    for declaration in source["retained_text"]:
        observed = tracked.get(declaration["path"])
        if observed is None or observed != {
            "mode": "100644",
            "blob_oid": declaration["blob_oid"],
        }:
            raise ValueError(
                f"retained source does not match Git tree: {declaration['path']}"
            )
    observation = {
        "observed_revision": revision,
        "observed_commit_tree": tree,
        "observed_remote_url": remote,
        "local_git_config_path": GIT_CONFIG_PATH,
        "local_git_config_bytes": len(config_before),
        "local_git_config_sha256": hashlib.sha256(config_before).hexdigest(),
        "git_head_sha256": hashlib.sha256(head_before).hexdigest(),
        "git_subprocess_uses_isolated_config_free_git_dir": True,
        "forbidden_auxiliary_git_paths_absent": list(
            git_contract["required_absent_paths"]
        ),
        "git_lazy_fetch_disabled": True,
        "git_terminal_prompt_disabled": True,
        "git_command_timeout_seconds": GIT_COMMAND_TIMEOUT_SECONDS,
        "git_command_output_capped": True,
        "tracked_file_count": len(tracked),
        "git_tree_listing_sha256": listing_sha256,
    }
    metadata_state = {
        "git_directory_signature": stat_signature(git_path_before),
        "local_config_signature": config_signature_before,
        "local_config_sha256": observation["local_git_config_sha256"],
        "head_signature": head_signature_before,
        "head_sha256": observation["git_head_sha256"],
        "objects_directory_signature": objects_signature,
        "objects_metadata_inventory": objects_inventory,
        "required_absent_paths": tuple(git_contract["required_absent_paths"]),
    }
    return observation, tracked, metadata_state


def _expected_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            directories.add(PurePosixPath(*parts[:index]).as_posix())
    return directories


def closed_world_inventory(
    root_fd: int,
    tracked: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ignored = {
        item["path"]: item for item in config["source"]["ignored_metadata_only"]
    }
    retained = {
        item["path"]: item for item in config["source"]["retained_text"]
    }
    expected_files = set(tracked) | set(ignored)
    expected_directories = _expected_directories(expected_files)
    internal: dict[str, dict[str, Any]] = {}
    public: list[dict[str, Any]] = []

    def visit(directory_fd: int, prefix: str) -> None:
        entries = sorted(os.scandir(directory_fd), key=lambda item: item.name)
        for entry in entries:
            name = entry.name
            if name in {"", ".", ".."} or "/" in name:
                raise ValueError("unsafe directory entry in BARC source")
            path = f"{prefix}/{name}" if prefix else name
            info = entry.stat(follow_symlinks=False)
            if path == ".git":
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    raise ValueError("BARC .git must be an opaque real directory")
                internal[path] = {"kind": "opaque_directory", "signature": stat_signature(info)}
                public.append({"path": path, "kind": "opaque_directory"})
                continue
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                if path not in expected_directories:
                    raise ValueError(f"unknown directory in BARC source: {path}")
                internal[path] = {"kind": "directory", "signature": stat_signature(info)}
                public.append(
                    {
                        "path": path,
                        "kind": "directory",
                        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                    }
                )
                child_fd = os.open(name, directory_flags(), dir_fd=directory_fd)
                try:
                    if stat_signature(os.fstat(child_fd)) != stat_signature(info):
                        raise RuntimeError(f"directory raced during inventory: {path}")
                    visit(child_fd, path)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError(f"nonregular entry in BARC source: {path}")
            if info.st_nlink != 1:
                raise ValueError(f"hard-linked file in BARC source: {path}")
            mode = format(stat.S_IMODE(info.st_mode), "04o")
            if path in tracked:
                if mode != "0644":
                    raise ValueError(f"tracked file metadata mismatch: {path}")
                if path in retained and info.st_size != retained[path]["bytes"]:
                    raise ValueError(f"retained tracked file size mismatch: {path}")
                kind = "tracked"
            elif path in ignored:
                declaration = ignored[path]
                if mode != declaration["mode"] or info.st_size != declaration["bytes"]:
                    raise ValueError(f"ignored metadata-only file mismatch: {path}")
                if PurePosixPath(path).suffix.lower() != ".pyc":
                    raise ValueError("ignored allowance is not bytecode")
                kind = "ignored_metadata_only"
            else:
                raise ValueError(f"unknown file in BARC source: {path}")
            internal[path] = {"kind": kind, "signature": stat_signature(info)}
            public.append(
                {
                    "path": path,
                    "kind": kind,
                    "mode": mode,
                    "bytes": info.st_size,
                }
            )

    visit(root_fd, "")
    found_files = {
        path
        for path, item in internal.items()
        if item["kind"] in {"tracked", "ignored_metadata_only"}
    }
    if found_files != expected_files:
        missing = sorted(expected_files - found_files)
        extra = sorted(found_files - expected_files)
        raise ValueError(f"BARC closed inventory mismatch: missing={missing}, extra={extra}")
    if ".git" not in internal:
        raise ValueError("BARC opaque .git directory is missing")
    return internal, sorted(public, key=lambda item: item["path"])


def verify_terminal_state(
    root_fd: int,
    source_path: Path,
    tracked: dict[str, dict[str, Any]],
    config: dict[str, Any],
    initial_git: dict[str, Any],
    initial_git_metadata: dict[str, Any],
    initial_inventory: dict[str, dict[str, Any]],
    initial_public_inventory: list[dict[str, Any]],
    ledger: ReadLedger,
) -> None:
    # Git metadata commands run before the last worktree observation.  This
    # ordering ensures a leaf change during the second Git observation cannot
    # occur after the terminal closed-world signature check.
    final_git, final_tracked, final_git_metadata = verify_git_contract(
        root_fd, config, ledger
    )
    if (
        final_git != initial_git
        or final_tracked != tracked
        or final_git_metadata != initial_git_metadata
    ):
        raise RuntimeError(
            "BARC Git revision/tree/listing/config metadata changed during audit"
        )
    final_inventory, final_public_inventory = closed_world_inventory(
        root_fd, tracked, config
    )
    if (
        final_inventory != initial_inventory
        or final_public_inventory != initial_public_inventory
    ):
        raise RuntimeError("BARC worktree metadata changed during static audit")
    verify_directory_path_identity(source_path, root_fd)


def parse_python(payload: bytes, path: str) -> ast.Module:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
        text = payload.decode(encoding)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot decode locked Python source: {path}") from error
    try:
        return ast.parse(text, filename=path)
    except SyntaxError as error:
        raise ValueError(f"cannot parse locked Python source: {path}") from error


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def literal_subscript_key(node: ast.Subscript) -> str | int | None:
    value = node.slice
    if isinstance(value, ast.Constant) and isinstance(value.value, (str, int)):
        return value.value
    return None


def subscript_chain(node: ast.AST) -> tuple[str, tuple[str | int, ...]] | None:
    keys: list[str | int] = []
    current = node
    while isinstance(current, ast.Subscript):
        key = literal_subscript_key(current)
        if key is None:
            return None
        keys.append(key)
        current = current.value
    root = dotted_name(current)
    if root is None:
        return None
    return root, tuple(reversed(keys))


def keyword_literal(call: ast.Call, name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


NESTED_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def executable_scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """Return nodes in one execution scope, excluding nested definitions.

    The traversal is deliberately conservative but removes branches that are
    provably unreachable from literals, short-circuit evaluation, or a
    statement that definitely transfers control.  That prevents dead source
    text from satisfying a label-flow contract.
    """

    result: list[ast.AST] = [scope]
    loop_break_stack: list[dict[str, bool]] = []

    def literal_truth(node: ast.AST) -> bool | None:
        if isinstance(node, ast.Constant):
            return bool(node.value)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            if not node.elts:
                return False
            if any(not isinstance(element, ast.Starred) for element in node.elts):
                return True
            return None
        if isinstance(node, ast.Dict):
            if not node.keys:
                return False
            if any(key is not None for key in node.keys):
                return True
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = literal_truth(node.operand)
            return None if value is None else not value
        return None

    def pattern_is_irrefutable(pattern: ast.pattern) -> bool:
        if isinstance(pattern, ast.MatchAs):
            return pattern.pattern is None or pattern_is_irrefutable(pattern.pattern)
        if isinstance(pattern, ast.MatchOr):
            return any(pattern_is_irrefutable(item) for item in pattern.patterns)
        return False

    def with_body_has_non_suppressible_transfer(statements: list[ast.stmt]) -> bool:
        """Prove only transfers that __exit__/__aexit__ cannot suppress.

        Earlier evaluated statements could raise and be suppressed, so this
        intentionally accepts only pass statements followed by a bare return,
        break, or continue.  Everything else remains conservatively reachable.
        """

        for statement in statements:
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Return) and statement.value is None:
                return True
            if isinstance(statement, (ast.Break, ast.Continue)):
                return True
            return False
        return False

    def visit_block(statements: list[ast.stmt]) -> bool:
        for statement in statements:
            if visit(statement):
                return True
        return False

    def visit(node: ast.AST) -> bool:
        if isinstance(node, NESTED_SCOPE_NODES):
            return False
        result.append(node)
        if isinstance(node, (ast.Return, ast.Raise)):
            for child in ast.iter_child_nodes(node):
                visit(child)
            return True
        if isinstance(node, ast.Break):
            if loop_break_stack:
                loop_break_stack[-1]["reachable_break"] = True
            return True
        if isinstance(node, ast.Continue):
            return True
        if isinstance(node, ast.IfExp):
            visit(node.test)
            truth = literal_truth(node.test)
            if truth is True:
                visit(node.body)
            elif truth is False:
                visit(node.orelse)
            else:
                visit(node.body)
                visit(node.orelse)
            return False
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                visit(value)
                truth = literal_truth(value)
                if isinstance(node.op, ast.And) and truth is False:
                    break
                if isinstance(node.op, ast.Or) and truth is True:
                    break
            return False
        if isinstance(node, ast.If):
            visit(node.test)
            truth = literal_truth(node.test)
            if truth is not None:
                branch = node.body if truth else node.orelse
                return visit_block(branch)
            body_stops = visit_block(node.body)
            else_stops = visit_block(node.orelse)
            return body_stops and bool(node.orelse) and else_stops
        if isinstance(node, ast.While):
            visit(node.test)
            truth = literal_truth(node.test)
            if truth is False:
                return visit_block(node.orelse)
            loop_state = {"reachable_break": False}
            loop_break_stack.append(loop_state)
            try:
                visit_block(node.body)
            finally:
                loop_break_stack.pop()
            if truth is True:
                return not loop_state["reachable_break"]
            visit_block(node.orelse)
            return False
        if isinstance(node, (ast.For, ast.AsyncFor)):
            visit(node.target)
            visit(node.iter)
            loop_state = {"reachable_break": False}
            loop_break_stack.append(loop_state)
            try:
                visit_block(node.body)
            finally:
                loop_break_stack.pop()
            visit_block(node.orelse)
            return False
        if isinstance(node, ast.Match):
            visit(node.subject)
            every_selected_body_stops = True
            for case in node.cases:
                result.append(case)
                visit(case.pattern)
                if case.guard is not None:
                    visit(case.guard)
                body_stops = visit_block(case.body)
                every_selected_body_stops = (
                    every_selected_body_stops and body_stops
                )
                if case.guard is None and pattern_is_irrefutable(case.pattern):
                    return every_selected_body_stops
            return False
        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            enclosing_loop = loop_break_stack[-1] if loop_break_stack else None
            break_before_try = (
                enclosing_loop["reachable_break"]
                if enclosing_loop is not None
                else False
            )
            body_stops = visit_block(node.body)
            handler_stops: list[bool] = []
            for handler in node.handlers:
                result.append(handler)
                if handler.type is not None:
                    visit(handler.type)
                handler_stops.append(visit_block(handler.body))
            if body_stops:
                normal_stops = True
            elif node.orelse:
                normal_stops = visit_block(node.orelse)
            else:
                normal_stops = False
            pre_final_stops = normal_stops and all(handler_stops)
            if node.finalbody:
                pre_final_break = (
                    enclosing_loop["reachable_break"]
                    if enclosing_loop is not None
                    else False
                )
                if enclosing_loop is not None:
                    enclosing_loop["reachable_break"] = break_before_try
                final_stops = visit_block(node.finalbody)
                if enclosing_loop is not None and not final_stops:
                    enclosing_loop["reachable_break"] = (
                        enclosing_loop["reachable_break"] or pre_final_break
                    )
            else:
                final_stops = False
            return final_stops or pre_final_stops
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                visit(item)
            visit_block(node.body)
            # A context manager may suppress body exceptions, but it cannot
            # cancel a bare return/break/continue after normal body execution.
            return with_body_has_non_suppressible_transfer(node.body)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            return visit_block(node.body)
        for field, value in ast.iter_fields(node):
            if field in {"body", "orelse", "finalbody"} and isinstance(value, list):
                visit_block([item for item in value if isinstance(item, ast.stmt)])
                continue
            if isinstance(value, ast.AST):
                visit(value)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        visit(child)
        return False

    if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
        visit_block(scope.body)
    else:
        result.clear()
        visit(scope)
    return result


def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def reachable_scopes(tree: ast.Module) -> list[ast.AST]:
    functions = top_level_functions(tree)
    scopes: list[ast.AST] = [tree]
    queued: list[ast.AST] = [tree]
    seen = {id(tree)}
    while queued:
        scope = queued.pop(0)
        nodes = executable_scope_nodes(scope)
        alias_sources: dict[str, set[str]] = {}
        for node in nodes:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
                value = node.value
            if not isinstance(value, ast.Name):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    alias_sources.setdefault(target.id, set()).add(value.id)

        def resolve_functions(name: str) -> set[str]:
            resolved: set[str] = set()
            pending = [name]
            examined: set[str] = set()
            while pending:
                candidate = pending.pop()
                if candidate in examined:
                    continue
                examined.add(candidate)
                if candidate in functions:
                    resolved.add(candidate)
                pending.extend(sorted(alias_sources.get(candidate, set())))
            return resolved

        referenced: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Call):
                name = dotted_name(node.func)
                if name is not None:
                    referenced.update(resolve_functions(name))
        for name in sorted(referenced):
            function = functions[name]
            if id(function) not in seen:
                seen.add(id(function))
                scopes.append(function)
                queued.append(function)
    return scopes


def scope_calls(scope: ast.AST, terminal_name: str) -> list[ast.Call]:
    return [
        node
        for node in executable_scope_nodes(scope)
        if isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1] == terminal_name
    ]


def scope_constants(scope: ast.AST) -> set[str]:
    return {
        node.value
        for node in executable_scope_nodes(scope)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def assignment_target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        current: ast.AST = node
        while isinstance(current, (ast.Subscript, ast.Attribute)):
            current = current.value
        return {current.id} if isinstance(current, ast.Name) else set()
    if isinstance(node, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for element in node.elts:
            result.update(assignment_target_names(element))
        return result
    return set()


def referenced_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in executable_scope_nodes(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def expression_message_indices(node: ast.AST) -> set[int]:
    indices: set[int] = set()
    for child in executable_scope_nodes(node):
        if not isinstance(child, ast.Subscript):
            continue
        key = literal_subscript_key(child)
        if not isinstance(key, int):
            continue
        current: ast.AST = child.value
        if isinstance(current, ast.Name) and current.id == "messages":
            indices.add(key)
            continue
        while isinstance(current, ast.Subscript):
            if literal_subscript_key(current) == "messages":
                indices.add(key)
                break
            current = current.value
    return indices


def expression_has_keyed_subscript(node: ast.AST, key: str) -> bool:
    return any(
        isinstance(child, ast.Subscript) and literal_subscript_key(child) == key
        for child in executable_scope_nodes(node)
    )


def assigned_call_targets(scope: ast.AST, terminal_name: str) -> dict[str, ast.Call]:
    result: dict[str, ast.Call] = {}
    for node in executable_scope_nodes(scope):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if (
            isinstance(value, ast.Call)
            and (dotted_name(value.func) or "").split(".")[-1] == terminal_name
        ):
            for target in targets:
                for name in assignment_target_names(target):
                    result[name] = value
    return result


def tainted_names_in_scope(scope: ast.AST, seeds: set[str]) -> set[str]:
    tainted = set(seeds)
    nodes = executable_scope_nodes(scope)
    changed = True
    while changed:
        changed = False
        for node in nodes:
            targets: set[str] = set()
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    targets.update(assignment_target_names(target))
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets.update(assignment_target_names(node.target))
                value = node.value
            elif isinstance(node, ast.NamedExpr):
                targets.update(assignment_target_names(node.target))
                value = node.value
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                targets.update(assignment_target_names(node.target))
                value = node.iter
            if value is not None and referenced_names(value) & tainted:
                before = len(tainted)
                tainted.update(targets)
                changed = changed or len(tainted) != before
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "extend", "update", "add"}
                and any(referenced_names(argument) & tainted for argument in node.args)
            ):
                before = len(tainted)
                tainted.update(assignment_target_names(node.func.value))
                changed = changed or len(tainted) != before
    return tainted


def message_tainted_names(scope: ast.AST, index: int) -> set[str]:
    seeds: set[str] = set()
    for node in executable_scope_nodes(scope):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is not None and index in expression_message_indices(value):
            for target in targets:
                seeds.update(assignment_target_names(target))
    return tainted_names_in_scope(scope, seeds)


def runner_scope_flow(scope: ast.AST) -> dict[str, bool]:
    template_targets = assigned_call_targets(scope, "apply_chat_template")
    message_taints = {
        index: message_tainted_names(scope, index) for index in (0, 1, 2)
    }
    prompt_contracts: dict[str, dict[str, Any]] = {}
    for name, call in template_targets.items():
        indices = expression_message_indices(call)
        call_names = referenced_names(call)
        indices.update(
            index
            for index, tainted in message_taints.items()
            if call_names & tainted
        )
        prompt_contracts[name] = {
            "message_indices": indices,
            "generation_prompt": keyword_literal(call, "add_generation_prompt") is True,
        }
    generate_assignments = assigned_call_targets(scope, "generate")
    generate_calls = scope_calls(scope, "generate")

    def prompt_names_for(call: ast.Call) -> set[str]:
        if not call.args:
            return set()
        return referenced_names(call.args[0]) & set(prompt_contracts)

    message2_to_generate = any(
        any(2 in prompt_contracts[name]["message_indices"] for name in prompt_names_for(call))
        for call in generate_calls
    )
    challenge_prompt_to_generate = any(
        any(
            {0, 1}.issubset(prompt_contracts[name]["message_indices"])
            and 2 not in prompt_contracts[name]["message_indices"]
            and prompt_contracts[name]["generation_prompt"]
            for name in prompt_names_for(call)
        )
        for call in generate_calls
    )
    generated = tainted_names_in_scope(scope, set(generate_assignments))
    generated_answer_comparison = False
    for node in executable_scope_nodes(scope):
        if not isinstance(node, ast.Compare):
            continue
        expressions = [node.left, *node.comparators]
        has_generated = any(referenced_names(value) & generated for value in expressions)
        has_answer = any(expression_has_keyed_subscript(value, "answer") for value in expressions)
        if has_generated and has_answer:
            generated_answer_comparison = True
            break
    return {
        "message2_prompt_to_generate": message2_to_generate,
        "challenge_prompt_to_generate": challenge_prompt_to_generate,
        "generated_output_to_answer_comparison": generated_answer_comparison,
    }


def analyze_pseudo_eval(tree: ast.Module) -> dict[str, Any]:
    scope = tree
    open_calls = scope_calls(scope, "open")
    opened_literals = {
        call.args[0].value
        for call in open_calls
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    data_seeds: set[str] = set()
    json_load_detected = False
    for node in executable_scope_nodes(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if isinstance(value, ast.Call) and dotted_name(value.func) == "json.load":
            json_load_detected = True
            for target in targets:
                data_seeds.update(assignment_target_names(target))
    tainted = tainted_names_in_scope(scope, data_seeds)
    pseudo_split = False
    promoted_targets: set[str] = set()
    for node in executable_scope_nodes(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (
            any(isinstance(target, ast.Name) and target.id == "new_test_dataset" for target in targets)
            and isinstance(value, ast.List)
            and len(value.elts) == 1
            and isinstance(value.elts[0], ast.Name)
            and value.elts[0].id == "train"
            and value.elts[0].id in tainted
        ):
            pseudo_split = True
            for target in targets:
                promoted_targets.update(assignment_target_names(target))
    promoted_flow = tainted_names_in_scope(scope, promoted_targets)
    derived_dump = any(
        isinstance(node, ast.Call)
        and dotted_name(node.func) == "json.dump"
        and node.args
        and bool(referenced_names(node.args[0]) & promoted_flow)
        for node in executable_scope_nodes(scope)
    )
    observation = {
        "evaluation_bundle_open_detected": (
            "arc_all_evaluation.json" in opened_literals
        ),
        "evaluation_json_load_detected": json_load_detected,
        "training_example_promoted_to_test_detected": pseudo_split,
        "derived_evaluation_bundle_write_detected": (
            "dataset/arc_all_evaluation_new_seperate.json" in opened_literals
            and derived_dump
            and bool(promoted_targets)
        ),
    }
    observation["pseudo_evaluation_flow_detected"] = all(observation.values())
    return observation


def analyze_transduction_formatter(tree: ast.Module) -> dict[str, Any]:
    functions = top_level_functions(tree)
    main_scope: ast.AST = functions.get("main") or tree
    helper_scope = functions.get("convert_chat_format_transduction")
    reachable = {id(scope) for scope in reachable_scopes(tree)}
    if (
        main_scope is None
        or helper_scope is None
        or id(main_scope) not in reachable
        or id(helper_scope) not in reachable
    ):
        return {
            "jsonl_input_detected": False,
            "test_output_label_access_detected": False,
            "assistant_answer_materialization_detected": False,
            "messages_index_2_access_detected": False,
            "generated_jsonl_write_detected": False,
            "transduction_label_materialization_detected": False,
        }
    main_nodes = executable_scope_nodes(main_scope)
    helper_nodes = executable_scope_nodes(helper_scope)
    test_label_attribute = False
    nested_messages_index_2 = False
    assistant_answer_dict = False
    answer_assigned_from_label = False
    for node in main_nodes:
        if isinstance(node, ast.Attribute) and node.attr == "y":
            indexed = node.value
            if (
                isinstance(indexed, ast.Subscript)
                and literal_subscript_key(indexed) == 0
                and isinstance(indexed.value, ast.Attribute)
                and indexed.value.attr == "test_pairs"
            ):
                test_label_attribute = True
        if isinstance(node, ast.Subscript) and literal_subscript_key(node) == 2:
            current: ast.AST = node.value
            while isinstance(current, ast.Subscript):
                if literal_subscript_key(current) == "messages":
                    nested_messages_index_2 = True
                    break
                current = current.value
    for node in helper_nodes:
        if isinstance(node, ast.Dict):
            pairs = {
                key.value: value
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if (
                isinstance(pairs.get("role"), ast.Constant)
                and pairs["role"].value == "assistant"
                and isinstance(pairs.get("content"), ast.Name)
                and pairs["content"].id == "answer"
            ):
                assistant_answer_dict = True
    for node in main_nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any("answer" in assignment_target_names(target) for target in targets):
                value = node.value
                if value is not None and any(
                    isinstance(child, ast.Attribute)
                    and child.attr == "y"
                    and isinstance(child.value, ast.Subscript)
                    and isinstance(child.value.value, ast.Attribute)
                    and child.value.value.attr == "test_pairs"
                    for child in executable_scope_nodes(value)
                ):
                    answer_assigned_from_label = True
    helper_answer_call = any(
        isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1]
        == "convert_chat_format_transduction"
        and any("answer" in referenced_names(argument) for argument in node.args)
        for node in main_nodes
    )
    helper_result_appended = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and (dotted_name(node.args[0].func) or "").split(".")[-1]
        == "convert_chat_format_transduction"
        and "answer" in referenced_names(node.args[0])
        for node in main_nodes
    )
    open_calls = [
        node
        for node in main_nodes
        if isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1] == "open"
    ]
    jsonl_input = any(
        call.args
        and dotted_name(call.args[0]) == "args.load_file"
        for call in open_calls
    )
    output_jsonl = any(
        call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "dataset/transduction_formatted_test-time_finetune.jsonl"
        for call in open_calls
    )
    output_handles: set[str] = set()
    for node in main_nodes:
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            context = item.context_expr
            if (
                isinstance(context, ast.Call)
                and (dotted_name(context.func) or "").split(".")[-1] == "open"
                and context.args
                and isinstance(context.args[0], ast.Constant)
                and context.args[0].value
                == "dataset/transduction_formatted_test-time_finetune.jsonl"
                and item.optional_vars is not None
            ):
                output_handles.update(assignment_target_names(item.optional_vars))
    label_flow = tainted_names_in_scope(
        main_scope, {"answer"} if answer_assigned_from_label else set()
    )
    label_derived_write = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in output_handles
        and node.args
        and bool(referenced_names(node.args[0]) & label_flow)
        for node in main_nodes
    )
    observation = {
        "jsonl_input_detected": jsonl_input,
        "test_output_label_access_detected": (
            test_label_attribute and answer_assigned_from_label
        ),
        "assistant_answer_materialization_detected": (
            assistant_answer_dict and helper_answer_call and helper_result_appended
        ),
        "messages_index_2_access_detected": nested_messages_index_2,
        "generated_jsonl_write_detected": output_jsonl and label_derived_write,
    }
    observation["transduction_label_materialization_detected"] = all(
        observation.values()
    )
    return observation


def analyze_vllm_runner(tree: ast.Module) -> dict[str, Any]:
    scopes = reachable_scopes(tree)
    constants = set().union(*(scope_constants(scope) for scope in scopes))
    tokenizer_loads = [
        call for scope in scopes for call in scope_calls(scope, "from_pretrained")
    ]
    llm_loads = [call for scope in scopes for call in scope_calls(scope, "LLM")]
    generate_calls = [
        call for scope in scopes for call in scope_calls(scope, "generate")
    ]
    flows = [runner_scope_flow(scope) for scope in scopes]
    indices = set().union(
        *(
            expression_message_indices(node)
            for scope in scopes
            for node in executable_scope_nodes(scope)
        )
    )
    answer_access = any(
        expression_has_keyed_subscript(node, "answer")
        for scope in scopes
        for node in executable_scope_nodes(scope)
    )
    explicit_local_only = bool(tokenizer_loads) and all(
        keyword_literal(call, "local_files_only") is True for call in tokenizer_loads
    )
    explicit_no_remote_code = bool(tokenizer_loads) and all(
        keyword_literal(call, "trust_remote_code") is False for call in tokenizer_loads
    )
    pinned_revision = bool(tokenizer_loads) and all(
        isinstance(keyword_literal(call, "revision"), str)
        and bool(keyword_literal(call, "revision"))
        for call in tokenizer_loads
    )
    jsonl_references = sorted(value for value in constants if ".jsonl" in value)
    observation = {
        "tokenizer_from_pretrained_call_count": len(tokenizer_loads),
        "vllm_model_constructor_call_count": len(llm_loads),
        "generate_call_count": len(generate_calls),
        "message_indices": sorted(indices),
        "messages_index_2_access_detected": 2 in indices,
        "answer_access_detected": answer_access,
        "jsonl_references": jsonl_references,
        "local_files_only_explicit": explicit_local_only,
        "trust_remote_code_false_explicit": explicit_no_remote_code,
        "revision_pin_explicit": pinned_revision,
        "safe_offline_tokenizer_load_detected": (
            explicit_local_only and explicit_no_remote_code and pinned_revision
        ),
        "challenge_only_prompt_pattern_detected": (
            any(flow["challenge_prompt_to_generate"] for flow in flows)
            and not any(flow["generated_output_to_answer_comparison"] for flow in flows)
        ),
        "direct_transduction_structure_detected": (
            any(
                flow["challenge_prompt_to_generate"]
                or flow["message2_prompt_to_generate"]
                for flow in flows
            )
            and any("transduction" in value.lower() for value in constants)
        ),
        "published_runner_label_flow_detected": (
            any(
                flow["message2_prompt_to_generate"]
                and flow["generated_output_to_answer_comparison"]
                for flow in flows
            )
        ),
    }
    return observation


def analyze_reranking(tree: ast.Module) -> dict[str, Any]:
    functions = top_level_functions(tree)
    main_scope: ast.AST = functions.get("main") or tree
    build_scope = functions.get("build_transformed_prompt")
    generate_scope = functions.get("generate_candidates")
    frequency_scope = functions.get("frequency_ranking")
    if None in {build_scope, generate_scope, frequency_scope}:
        return {
            "frequency_ranking_detected": False,
            "messages_index_2_access_detected": False,
            "answer_access_detected": False,
            "correctness_fields_detected": False,
            "score_flow_detected": False,
            "jsonl_input_detected": False,
            "label_aware_reranking_detected": False,
        }
    assert build_scope is not None
    assert generate_scope is not None
    assert frequency_scope is not None
    main_nodes = executable_scope_nodes(main_scope)
    build_nodes = executable_scope_nodes(build_scope)
    frequency_nodes = executable_scope_nodes(frequency_scope)

    def returned_from_flow(scope: ast.AST, flow: set[str]) -> bool:
        return any(
            isinstance(node, ast.Return)
            and node.value is not None
            and bool(referenced_names(node.value) & flow)
            for node in executable_scope_nodes(scope)
        )

    build_message2_taint = message_tainted_names(build_scope, 2)
    build_flow = tainted_names_in_scope(build_scope, build_message2_taint)
    build_message2_template = any(
        isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1]
        == "apply_chat_template"
        and (
            2 in expression_message_indices(node)
            or bool(referenced_names(node) & build_flow)
        )
        for node in build_nodes
    )
    build_returns_label_prompt = build_message2_template and any(
        isinstance(node, ast.Return)
        and node.value is not None
        and (
            bool(referenced_names(node.value) & build_flow)
            or 2 in expression_message_indices(node.value)
        )
        for node in build_nodes
    )

    main_build_targets = assigned_call_targets(main_scope, "build_transformed_prompt")
    main_generate_targets = assigned_call_targets(main_scope, "generate_candidates")
    generate_parameters = [
        argument.arg
        for argument in [*generate_scope.args.posonlyargs, *generate_scope.args.args]
    ]
    tainted_generate_parameters: set[str] = set()
    build_to_generate = False
    for call in main_generate_targets.values():
        for index, argument in enumerate(call.args):
            if referenced_names(argument) & set(main_build_targets):
                build_to_generate = True
                if index < len(generate_parameters):
                    tainted_generate_parameters.add(generate_parameters[index])
        for keyword in call.keywords:
            if (
                keyword.arg in generate_parameters
                and referenced_names(keyword.value) & set(main_build_targets)
            ):
                build_to_generate = True
                tainted_generate_parameters.add(keyword.arg)
    generate_flow = tainted_names_in_scope(
        generate_scope, tainted_generate_parameters
    )
    helper_generate_flow = any(
        call.args and bool(referenced_names(call.args[0]) & generate_flow)
        for call in scope_calls(generate_scope, "generate")
    ) and returned_from_flow(generate_scope, generate_flow)

    main_generated_flow = tainted_names_in_scope(
        main_scope, set(main_generate_targets)
    )
    frequency_targets = assigned_call_targets(main_scope, "frequency_ranking")
    frequency_parameters = [
        argument.arg
        for argument in [*frequency_scope.args.posonlyargs, *frequency_scope.args.args]
    ]
    tainted_frequency_parameters: set[str] = set()
    frequency_call_from_generated = False
    for call in frequency_targets.values():
        for index, argument in enumerate(call.args):
            if referenced_names(argument) & main_generated_flow:
                frequency_call_from_generated = True
                if index < len(frequency_parameters):
                    tainted_frequency_parameters.add(frequency_parameters[index])
        for keyword in call.keywords:
            if (
                keyword.arg in frequency_parameters
                and referenced_names(keyword.value) & main_generated_flow
            ):
                frequency_call_from_generated = True
                tainted_frequency_parameters.add(keyword.arg)
    frequency_flow = tainted_names_in_scope(
        frequency_scope, tainted_frequency_parameters
    )
    frequency_returns_generated = returned_from_flow(
        frequency_scope, frequency_flow
    )
    main_frequency_flow = tainted_names_in_scope(
        main_scope, set(frequency_targets)
    )
    answer_assignments = {
        name
        for name, call in assigned_call_targets(main_scope, "strip").items()
        if expression_has_keyed_subscript(call, "answer")
    }
    answer_comparisons: set[str] = set()
    correctness_names: dict[str, set[str]] = {"original": set(), "frequency": set()}
    for node in main_nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if (
            not isinstance(value, ast.Call)
            or (dotted_name(value.func) or "") != "any"
            or not value.args
            or not isinstance(value.args[0], ast.GeneratorExp)
        ):
            continue
        generator = value.args[0]
        if not (referenced_names(generator.elt) & answer_assignments):
            continue
        for comprehension in generator.generators:
            references = referenced_names(comprehension.iter)
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            target_names = set().union(
                *(assignment_target_names(target) for target in targets)
            )
            if "original_candidates" in references and (
                references & main_generated_flow
            ):
                answer_comparisons.add("original")
                correctness_names["original"].update(target_names)
            if references & set(frequency_targets) and (
                references & main_frequency_flow
            ):
                answer_comparisons.add("frequency")
                correctness_names["frequency"].update(target_names)
    response_fields: set[str] = set()
    for node in main_nodes:
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(
                    key.value, str
                ):
                    continue
                if key.value == "is_correct_original" and (
                    referenced_names(value) & correctness_names["original"]
                ):
                    response_fields.add(key.value)
                if key.value == "is_correct_frequency" and (
                    referenced_names(value) & correctness_names["frequency"]
                ):
                    response_fields.add(key.value)
    frequency_names = {
        node.id for node in frequency_nodes if isinstance(node, ast.Name)
    }
    open_calls = [
        node
        for node in main_nodes
        if isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1] == "open"
    ]
    observation = {
        "frequency_ranking_detected": (
            frequency_call_from_generated
            and frequency_returns_generated
            and {"candidate_stats", "scores"}.issubset(frequency_names)
        ),
        "messages_index_2_access_detected": (
            build_returns_label_prompt
            and build_to_generate
            and helper_generate_flow
        ),
        "answer_access_detected": (
            bool(answer_assignments) and answer_comparisons == {"original", "frequency"}
        ),
        "correctness_fields_detected": {
            "is_correct_original",
            "is_correct_frequency",
        }.issubset(response_fields),
        "score_flow_detected": (
            bool(tainted_frequency_parameters)
            and frequency_call_from_generated
            and frequency_returns_generated
        ),
        "jsonl_input_detected": any(
            call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "validation_data.jsonl"
            for call in open_calls
        ),
    }
    observation["label_aware_reranking_detected"] = all(observation.values())
    return observation


def analyze_generated_code_execution(
    eval_code_tree: ast.Module,
    execution_tree: ast.Module,
    evaluation_tree: ast.Module,
) -> dict[str, Any]:
    eval_scopes = reachable_scopes(eval_code_tree)
    eval_nodes = [node for scope in eval_scopes for node in executable_scope_nodes(scope)]
    eval_constants = set().union(*(scope_constants(scope) for scope in eval_scopes))

    def call_present(node: ast.AST, terminal: str) -> bool:
        return any(
            isinstance(child, ast.Call)
            and (dotted_name(child.func) or "").split(".")[-1] == terminal
            for child in executable_scope_nodes(node)
        )

    eval_seed_names: set[str] = set()
    for node in eval_nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not call_present(value, "loads"):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            eval_seed_names.update(assignment_target_names(target))
    eval_scope = top_level_functions(eval_code_tree).get("main") or eval_code_tree
    eval_flow = tainted_names_in_scope(eval_scope, eval_seed_names)
    parse_calls = [
        node
        for node in executable_scope_nodes(eval_scope)
        if isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1] == "parse_code"
        and node.args
        and bool(referenced_names(node.args[0]) & eval_flow)
    ]
    parsed_targets = {
        name
        for name, call in assigned_call_targets(eval_scope, "parse_code").items()
        if call in parse_calls
    }
    parsed_flow = tainted_names_in_scope(eval_scope, eval_flow | parsed_targets)

    eval_functions = top_level_functions(eval_code_tree)
    dispatch_flow = False
    for node in executable_scope_nodes(eval_scope):
        if not isinstance(node, ast.Call) or not any(
            referenced_names(argument) & parsed_flow for argument in node.args
        ):
            continue
        terminal = (dotted_name(node.func) or "").split(".")[-1]
        if terminal == "multi_execute_transformation":
            dispatch_flow = True
            break
        helper = eval_functions.get(terminal)
        if helper is None:
            continue
        parameters = [
            argument.arg
            for argument in [*helper.args.posonlyargs, *helper.args.args]
        ]
        seeds = {
            parameters[index]
            for index, argument in enumerate(node.args)
            if index < len(parameters) and referenced_names(argument) & parsed_flow
        }
        helper_flow = tainted_names_in_scope(helper, seeds)
        if any(
            call.args
            and any(referenced_names(argument) & helper_flow for argument in call.args)
            for call in scope_calls(helper, "multi_execute_transformation")
        ):
            dispatch_flow = True
            break

    execution_functions = top_level_functions(execution_tree)
    execution_entry = execution_functions.get("multi_execute_transformation")
    process_pool_flow = False
    exec_flow = False
    execution_exec_calls: list[ast.Call] = []
    if execution_entry is not None:
        entry_parameters = [
            argument.arg
            for argument in [
                *execution_entry.args.posonlyargs,
                *execution_entry.args.args,
            ]
        ]
        entry_seeds = {entry_parameters[0]} if entry_parameters else set()
        entry_flow = tainted_names_in_scope(execution_entry, entry_seeds)
        process_helper = execution_functions.get("multi_process_execute")
        worker = execution_functions.get("_worker_with_id")
        if process_helper is not None and worker is not None:
            process_parameters = [
                argument.arg
                for argument in [
                    *process_helper.args.posonlyargs,
                    *process_helper.args.args,
                ]
            ]
            process_seeds: set[str] = set()
            for call in scope_calls(execution_entry, "multi_process_execute"):
                for index, argument in enumerate(call.args):
                    if index < len(process_parameters) and (
                        referenced_names(argument) & entry_flow
                    ):
                        process_seeds.add(process_parameters[index])
            process_flow = tainted_names_in_scope(process_helper, process_seeds)
            pool_created = bool(scope_calls(process_helper, "ProcessPool"))
            worker_dispatched = any(
                len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "_worker_with_id"
                and bool(referenced_names(call.args[1]) & process_flow)
                for call in scope_calls(process_helper, "map")
            )
            worker_parameters = [
                argument.arg
                for argument in [*worker.args.posonlyargs, *worker.args.args]
            ]
            worker_flow = tainted_names_in_scope(
                worker, {worker_parameters[0]} if worker_parameters else set()
            )
            execution_exec_calls = [
                call
                for call in scope_calls(worker, "exec")
                if call.args and bool(referenced_names(call.args[0]) & worker_flow)
            ]
            process_pool_flow = pool_created and worker_dispatched
            exec_flow = process_pool_flow and bool(execution_exec_calls)

    evaluation_scopes = reachable_scopes(evaluation_tree)
    evaluation_nodes = [
        node for scope in evaluation_scopes for node in executable_scope_nodes(scope)
    ]
    evaluation_constants = set().union(
        *(scope_constants(scope) for scope in evaluation_scopes)
    )
    evaluation_scope = top_level_functions(evaluation_tree).get("main") or evaluation_tree
    evaluation_seed_names: set[str] = set()
    for node in evaluation_nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and node.args
            and call_present(node.args[0], "load")
        ):
            evaluation_seed_names.update(assignment_target_names(node.func.value))
    evaluation_flow = tainted_names_in_scope(
        evaluation_scope, evaluation_seed_names
    )
    answer_file_open = any(
        isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1] == "open"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "answer_file"
        for node in eval_nodes
    )
    answer_json_decode = any(
        isinstance(node, ast.Call)
        and dotted_name(node.func) == "json.loads"
        and any(
            isinstance(parent, (ast.Assign, ast.AnnAssign))
            and parent.value is not None
            and node in executable_scope_nodes(parent.value)
            for parent in eval_nodes
        )
        for node in eval_nodes
    )
    train_verdict_flow = any(
        isinstance(node, ast.Call)
        and (dotted_name(node.func) or "").split(".")[-1] == "zip"
        and expression_has_keyed_subscript(node, "train_verdicts")
        and bool(referenced_names(node) & evaluation_flow)
        for node in evaluation_nodes
    )
    ground_truth_label = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and "ground_truth" in assignment_target_names(
            node.targets[0] if isinstance(node, ast.Assign) else node.target
        )
        and node.value is not None
        and any(
            isinstance(child, ast.Attribute)
            and child.attr == "y"
            and "test_pair" in (dotted_name(child.value) or "")
            for child in executable_scope_nodes(node.value)
        )
        for node in evaluation_nodes
    )
    ground_truth_scoring = ground_truth_label and any(
        isinstance(node, ast.GeneratorExp)
        and "ground_truth" in referenced_names(node.elt)
        and any(
            bool(referenced_names(comprehension.iter) & evaluation_flow)
            for comprehension in node.generators
        )
        for node in evaluation_nodes
    )
    observation = {
        "induction_answer_jsonl_reader_detected": (
            answer_file_open
            and answer_json_decode
            and any("answer file" in value.lower() for value in eval_constants)
        ),
        "generated_response_code_parser_detected": bool(parse_calls),
        "generated_code_execution_dispatch_detected": dispatch_flow,
        "raw_exec_call_count": len(execution_exec_calls) if exec_flow else 0,
        "process_pool_detected": process_pool_flow,
        "induction_jsonl_result_load_detected": any(
            "induction_samples_with_execution_results" in value
            for value in evaluation_constants
        )
        and bool(evaluation_seed_names),
        "train_verdict_ranking_detected": train_verdict_flow,
        "ground_truth_scoring_detected": ground_truth_scoring,
    }
    observation["induction_generated_exec_label_flow_detected"] = all(
        bool(value) for value in observation.values()
    )
    return observation


def analyze_dependencies(requirements_text: str, readme_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in requirements_text.splitlines() if line.strip()]
    observation = {
        "requirements_lines": lines,
        "requirements_fully_pinned": all("==" in line for line in lines),
        "git_dependency_revision_pinned": any(
            line.startswith("git+") and "@" in line.rsplit("/", 1)[-1]
            for line in lines
        ),
        "induction_vllm_version": "0.6.0" if "vllm==0.6.0" in readme_text else None,
        "transduction_vllm_version": (
            "0.5.4" if "vllm==0.5.4" in readme_text else None
        ),
        "eight_process_training_detected": "--num_processes=8" in readme_text,
    }
    observation["conflicting_vllm_paths_detected"] = (
        observation["induction_vllm_version"] == "0.6.0"
        and observation["transduction_vllm_version"] == "0.5.4"
    )
    observation["reproducible_dependency_lock_detected"] = (
        observation["requirements_fully_pinned"]
        and observation["git_dependency_revision_pinned"]
        and not observation["conflicting_vllm_paths_detected"]
    )
    return observation


def controls_record(ledger: ReadLedger) -> dict[str, Any]:
    source_records = [
        item
        for item in ledger.records
        if item["category"]
        in {
            "source_python",
            "source_documentation",
            "dependency_manifest",
            "artifact_recipe",
            "vendored_license",
        }
    ]
    restricted = [
        item
        for item in source_records
        if PurePosixPath(item["path"]).suffix.lower() in EXPECTED_NEVER_READ_SUFFIXES
    ]
    git_config_records = [
        item for item in ledger.records if item["category"] == "git_local_config"
    ]
    git_head_records = [
        item for item in ledger.records if item["category"] == "git_head"
    ]
    return {
        "network_used": False,
        "gpu_used": False,
        "upstream_imported": False,
        "upstream_executed": False,
        "generated_code_executed": False,
        "solver_prediction_produced": False,
        "retained_source_read_attempts": len(source_records),
        "retained_source_files_read": sum(
            item["read_status"] == "completed" or item["bytes"] > 0
            for item in source_records
        ),
        "git_local_config_read_attempts": len(git_config_records),
        "git_local_config_bytes_read": sum(
            item["bytes"] for item in git_config_records
        ),
        "git_head_read_attempts": len(git_head_records),
        "git_head_bytes_read": sum(item["bytes"] for item in git_head_records),
        "git_subprocesses_started": len(ledger.git_subprocesses),
        "git_subprocess_object_database_reads_possible": bool(
            ledger.git_subprocesses
        ),
        "git_subprocess_object_database_bytes_measured": False,
        "git_subprocess_worktree_content_requested": any(
            item["worktree_content_requested"] for item in ledger.git_subprocesses
        ),
        "git_subprocess_timeout_and_output_caps_enforced": bool(
            ledger.git_subprocesses
        ),
        "git_subprocess_untrusted_local_config_available": any(
            item["untrusted_repository_local_config_available"]
            for item in ledger.git_subprocesses
        ),
        "auditor_process_arc_or_label_worktree_leaf_bytes_read": sum(
            item["bytes"]
            for item in restricted
            if PurePosixPath(item["path"]).suffix.lower() in {".json", ".jsonl"}
        ),
        "auditor_process_pickle_worktree_leaf_bytes_read": sum(
            item["bytes"]
            for item in restricted
            if PurePosixPath(item["path"]).suffix.lower() in {".pkl", ".pickle"}
        ),
        "auditor_process_model_weight_worktree_leaf_bytes_read": sum(
            item["bytes"]
            for item in restricted
            if PurePosixPath(item["path"]).suffix.lower() in MODEL_WEIGHT_SUFFIXES
        ),
        "auditor_process_pyc_worktree_leaf_bytes_read": sum(
            item["bytes"]
            for item in restricted
            if PurePosixPath(item["path"]).suffix.lower() == ".pyc"
        ),
    }


def base_record(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "run_id": run_id,
        "method_id": METHOD_ID,
        "runner": "scripts.audit_barc_gates",
        "scope": SCOPE,
        "method_gate_status": "blocked",
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "solver_gate_passed": False,
        "fairness": {
            "evidence_scope": "blocker_audit",
            "performance_table_eligible": False,
            "promotion_effect": "none",
        },
    }


def failure_record(
    run_id: str, stage: str, error: BaseException, ledger: ReadLedger
) -> dict[str, Any]:
    record = base_record(run_id)
    record.update(
        {
            "status": "failed",
            "controls": controls_record(ledger),
            "read_ledger": ledger.snapshot(),
            "error": {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
            "claim_boundary": (
                "The BARC audit failed closed and grants no execution or performance claim."
            ),
        }
    )
    return record


def run_static_audit(
    config_path: Path, run_id: str, ledger: ReadLedger
) -> dict[str, Any]:
    ledger.bind_config(config_path)
    config_payload, _ = secure_read_absolute(
        config_path,
        max_bytes=MAX_CONFIG_BYTES,
        role="canonical_config",
        ledger=ledger,
        display=CANONICAL_CONFIG_RELATIVE,
    )
    config_raw_sha256 = hashlib.sha256(config_payload).hexdigest()
    config = validate_config(strict_json(config_payload, "BARC gate config"))
    ledger.bind_source_policy(config["source"]["retained_text"])

    source_lock_payload, _ = secure_read_absolute(
        ROOT / SOURCE_LOCK_RELATIVE,
        max_bytes=MAX_CONFIG_BYTES,
        role="source_lock",
        ledger=ledger,
        display=SOURCE_LOCK_RELATIVE,
    )
    if hashlib.sha256(source_lock_payload).hexdigest() != EXPECTED_SOURCE_LOCK_SHA256:
        raise ValueError("source_locks.json SHA-256 mismatch")
    source_locks = strict_json(source_lock_payload, "source_locks.json")
    if (
        not isinstance(source_locks, dict)
        or not isinstance(source_locks.get("sources"), dict)
        or source_locks["sources"].get("barc") != EXPECTED_SOURCE_LOCK_ENTRY
    ):
        raise ValueError("source_locks.json BARC entry mismatch")

    root_fd = open_absolute_directory(SOURCE_PATH)
    try:
        initial_git, tracked, initial_git_metadata = verify_git_contract(
            root_fd, config, ledger
        )
        initial_inventory, public_inventory = closed_world_inventory(
            root_fd, tracked, config
        )
        initial_inventory_digest = canonical_sha256(public_inventory)
        if (
            initial_inventory_digest
            != EXPECTED_CLOSED_INVENTORY_METADATA_SHA256
            or initial_inventory_digest
            != config["source"]["closed_inventory_metadata_sha256"]
        ):
            raise ValueError("BARC closed worktree metadata digest mismatch")
        tracked_worktree_bytes = sum(
            item["bytes"]
            for item in public_inventory
            if item["kind"] == "tracked"
        )
        if tracked_worktree_bytes != config["source"]["expected_tracked_bytes"]:
            raise ValueError("BARC tracked worktree byte metadata mismatch")

        payloads: dict[str, bytes] = {}
        trees: dict[str, ast.Module] = {}
        retained_observations: list[dict[str, Any]] = []
        for declaration in config["source"]["retained_text"]:
            payload = secure_read_relative(
                root_fd, declaration, initial_inventory, ledger
            )
            path = declaration["path"]
            payloads[path] = payload
            if path in PYTHON_PATHS:
                trees[path] = parse_python(payload, path)
            retained_observations.append(
                {
                    "path": path,
                    "role": declaration["role"],
                    "bytes": declaration["bytes"],
                    "sha256": declaration["sha256"],
                    "blob_oid": declaration["blob_oid"],
                    "ast_parsed": path in PYTHON_PATHS,
                }
            )

        readme = payloads["README.md"].decode("utf-8")
        requirements = payloads["requirements.txt"].decode("utf-8")
        recipe = payloads[
            "finetune/alignment-handbook/recipes/barc/"
            "transduction_config_testtime-finetune_engineer_heavy_model.yaml"
        ].decode("utf-8")
        vendored_license = payloads["finetune/alignment-handbook/LICENSE"].decode(
            "utf-8"
        )

        pseudo = analyze_pseudo_eval(
            trees["data_processing/test-time-finetune/get_pseudo_eval_task.py"]
        )
        formatter = analyze_transduction_formatter(
            trees[
                "data_processing/test-time-finetune/"
                "gen_transduction_only_formatted.py"
            ]
        )
        induction_runner = analyze_vllm_runner(
            trees["finetune/inference/vllm_inference.py"]
        )
        transduction_evaluation = analyze_vllm_runner(
            trees[
                "finetune/inference/vllm_inference_transduction_evaluation.py"
            ]
        )
        transduction_concept = analyze_vllm_runner(
            trees[
                "finetune/inference/vllm_inference_transduction_concept_arc.py"
            ]
        )
        rerank_runner = analyze_vllm_runner(
            trees["finetune/inference/vllm_transduction_reranking.py"]
        )
        reranking = analyze_reranking(
            trees["finetune/inference/vllm_transduction_reranking.py"]
        )
        generated_execution = analyze_generated_code_execution(
            trees["eval_code_samples.py"],
            trees["execution.py"],
            trees["evaluation.py"],
        )
        dependencies = analyze_dependencies(requirements, readme)

        if not pseudo["pseudo_evaluation_flow_detected"]:
            raise ValueError("BARC pseudo-evaluation AST contract mismatch")
        if not formatter["transduction_label_materialization_detected"]:
            raise ValueError("BARC transduction formatter AST contract mismatch")
        if not induction_runner["challenge_only_prompt_pattern_detected"]:
            raise ValueError("BARC label-free induction prompt pattern mismatch")
        if not transduction_evaluation["published_runner_label_flow_detected"]:
            raise ValueError("BARC evaluation transduction label-flow mismatch")
        if not transduction_concept["published_runner_label_flow_detected"]:
            raise ValueError("BARC ConceptARC transduction label-flow mismatch")
        if not reranking["label_aware_reranking_detected"]:
            raise ValueError("BARC reranking label-flow mismatch")
        if not generated_execution[
            "induction_generated_exec_label_flow_detected"
        ]:
            raise ValueError("BARC induction generated-exec flow mismatch")
        runners = {
            "induction": induction_runner,
            "transduction_evaluation": transduction_evaluation,
            "transduction_concept_arc": transduction_concept,
            "transduction_reranking": rerank_runner,
        }
        if any(
            observation["safe_offline_tokenizer_load_detected"]
            for observation in runners.values()
        ):
            raise ValueError("unexpected safe-offline load observation changed")
        if dependencies["requirements_lines"] != config["dependency_contract"][
            "expected_requirements_lines"
        ]:
            raise ValueError("BARC requirements manifest changed")
        if not dependencies["conflicting_vllm_paths_detected"]:
            raise ValueError("BARC vLLM version split observation changed")
        if dependencies["reproducible_dependency_lock_detected"]:
            raise ValueError("unexpected reproducible dependency lock detected")

        tracked_paths = set(tracked)
        root_license_paths = sorted(
            tracked_paths & set(config["license"]["root_candidates"])
        )
        if len(root_license_paths) != config["license"]["expected_root_license_count"]:
            raise ValueError("BARC root-license inventory changed")
        if "Apache License" not in vendored_license or "Version 2.0" not in vendored_license:
            raise ValueError("vendored alignment-handbook license content changed")

        if EXPECTED_BASE_MODELS[1] not in payloads[
            "finetune/inference/vllm_transduction_reranking.py"
        ].decode("utf-8"):
            raise ValueError("published Heavy-Transduction base string changed")
        if EXPECTED_LORAS[1] not in payloads[
            "finetune/inference/vllm_inference_transduction_evaluation.py"
        ].decode("utf-8"):
            raise ValueError("published Heavy test-time LoRA string changed")
        if EXPECTED_LORAS[0] not in recipe:
            raise ValueError("published Engineer test-time LoRA recipe changed")
        if "Test-Time-Finetune Adapters" not in readme or "Models" not in readme:
            raise ValueError("BARC artifact documentation anchors changed")

        weight_paths = sorted(
            path
            for path in tracked
            if PurePosixPath(path).suffix.lower() in MODEL_WEIGHT_SUFFIXES
        )
        if len(weight_paths) != config["artifacts"]["expected_local_weight_file_count"]:
            raise ValueError("BARC local model-weight count changed")

        verify_terminal_state(
            root_fd,
            SOURCE_PATH,
            tracked,
            config,
            initial_git,
            initial_git_metadata,
            initial_inventory,
            public_inventory,
            ledger,
        )
    finally:
        os.close(root_fd)

    controls = controls_record(ledger)
    if controls["retained_source_files_read"] != len(
        config["source"]["retained_text"]
    ):
        raise RuntimeError("BARC retained-source read ledger count mismatch")
    for key in (
        "auditor_process_arc_or_label_worktree_leaf_bytes_read",
        "auditor_process_pickle_worktree_leaf_bytes_read",
        "auditor_process_model_weight_worktree_leaf_bytes_read",
        "auditor_process_pyc_worktree_leaf_bytes_read",
    ):
        if controls[key] != 0:
            raise RuntimeError(f"restricted-byte control violated: {key}")
    if controls["git_local_config_read_attempts"] != 4 or controls[
        "git_local_config_bytes_read"
    ] != 4 * config["source"]["git_metadata_contract"]["local_config"]["bytes"]:
        raise RuntimeError("BARC Git local-config read ledger mismatch")
    if controls["git_head_read_attempts"] != 4 or controls[
        "git_head_bytes_read"
    ] != 4 * config["source"]["git_metadata_contract"]["head"]["bytes"]:
        raise RuntimeError("BARC Git HEAD read ledger mismatch")
    if controls["git_subprocesses_started"] != 6 or any(
        item["status"] != "completed" for item in ledger.git_subprocesses
    ):
        raise RuntimeError("BARC fixed Git subprocess ledger mismatch")

    file_entries = [
        item
        for item in public_inventory
        if item["kind"] in {"tracked", "ignored_metadata_only"}
    ]
    directory_entries = [
        item
        for item in public_inventory
        if item["kind"] in {"directory", "opaque_directory"}
    ]
    retained_paths = {item["path"] for item in config["source"]["retained_text"]}
    metadata_only_tracked = sorted(set(tracked) - retained_paths)
    metadata_digest = initial_inventory_digest

    safe_load_summary = {
        name: {
            "tokenizer_from_pretrained_call_count": value[
                "tokenizer_from_pretrained_call_count"
            ],
            "vllm_model_constructor_call_count": value[
                "vllm_model_constructor_call_count"
            ],
            "local_files_only_explicit": value["local_files_only_explicit"],
            "trust_remote_code_false_explicit": value[
                "trust_remote_code_false_explicit"
            ],
            "revision_pin_explicit": value["revision_pin_explicit"],
        }
        for name, value in runners.items()
    }
    stable_observation = {
        "source_lock_sha256": EXPECTED_SOURCE_LOCK_SHA256,
        "git": initial_git,
        "closed_inventory_metadata_sha256": metadata_digest,
        "retained_source": retained_observations,
        "pseudo_eval": pseudo,
        "transduction_formatter": formatter,
        "runners": runners,
        "reranking": reranking,
        "generated_execution": generated_execution,
        "dependencies": dependencies,
        "root_license_paths": root_license_paths,
        "weight_paths": weight_paths,
    }
    record = base_record(run_id)
    record.update(
        {
            "status": "passed",
            "config": {
                "path": CANONICAL_CONFIG_RELATIVE,
                "sha256": config_raw_sha256,
                "canonical_contract_sha256": EXPECTED_CONFIG_CANONICAL_SHA256,
            },
            "source_lock": {
                "path": SOURCE_LOCK_RELATIVE,
                "sha256": EXPECTED_SOURCE_LOCK_SHA256,
                "entry_matches": True,
            },
            "source": {
                **initial_git,
                "repository_path": str(SOURCE_PATH),
                "opaque_directories": [".git"],
                "closed_world_path_type_size_mode_passed": True,
                "closed_inventory_metadata_sha256": metadata_digest,
                "tracked_file_count": len(tracked),
                "tracked_bytes_from_closed_worktree_metadata": tracked_worktree_bytes,
                "directory_entry_count": len(directory_entries),
                "retained_byte_exact_file_count": len(retained_observations),
                "retained_byte_exact_files": retained_observations,
                "metadata_only_tracked_file_count": len(metadata_only_tracked),
                "ignored_pyc_metadata_only_count": len(
                    config["source"]["ignored_metadata_only"]
                ),
                "working_tree_all_files_byte_exact_verified": False,
                "unknown_entry_count": 0,
                "file_entry_count_including_ignored": len(file_entries),
            },
            "license_gate": {
                "status": "blocked",
                "root_license_count": 0,
                "root_license_paths": [],
                "root_source_license_verified": False,
                "vendored_license": {
                    "path": config["license"]["vendored_path"],
                    "identifier": "Apache-2.0",
                    "sha256": next(
                        item["sha256"]
                        for item in retained_observations
                        if item["path"] == config["license"]["vendored_path"]
                    ),
                    "scope": "finetune/alignment-handbook vendored component only",
                    "applied_to_repository_root": False,
                },
            },
            "artifact_gate": {
                "status": "blocked",
                "local_weight_file_count_in_closed_source_tree": len(weight_paths),
                "external_cache_roots_inspected": False,
                "base_models": [
                    {
                        "repo_id": item["repo_id"],
                        "planning_size_gib": item["planning_size_gib"],
                        "source_status": "expected-official-matrix-unverified-remote",
                        "immutable_revision": None,
                        "manifest_sha256": None,
                        "artifact_license": None,
                        "provenance_verified": False,
                    }
                    for item in config["artifacts"]["base_models"]
                ],
                "lora_adapters": [
                    {
                        "repo_id": repo_id,
                        "source_string_or_recipe_detected": True,
                        "immutable_revision": None,
                        "manifest_sha256": None,
                        "base_binding_verified": False,
                        "artifact_license": None,
                        "provenance_verified": False,
                    }
                    for repo_id in config["artifacts"]["lora_adapters"]
                ],
            },
            "safe_offline_model_load_gate": {
                "status": "blocked",
                "runner_observations": safe_load_summary,
                "network_disabled_audit_only": True,
                "local_files_only_contract_validated": False,
                "trust_remote_code_false_contract_validated": False,
                "immutable_revision_binding_validated": False,
                "safe_weight_manifest_validated": False,
            },
            "label_firewall_gate": {
                "status": "blocked",
                "pseudo_evaluation": pseudo,
                "transduction_formatter": formatter,
                "published_transduction_evaluation": transduction_evaluation,
                "published_transduction_concept_arc": transduction_concept,
                "published_transduction_reranking": reranking,
                "induction_jsonl_and_generated_execution": generated_execution,
                "challenge_only_direct_transduction_candidate": {
                    "status": "design-candidate-not-implemented",
                    "source_pattern": "finetune/inference/vllm_inference.py",
                    "label_free_system_user_prompt_pattern_detected": True,
                    "required_change": (
                        "Reuse only messages[0:2] with add_generation_prompt=true, "
                        "bind one verified direct-transduction base, stage challenge-only "
                        "input, and move answer scoring to a separate process."
                    ),
                    "solver_prediction_validated": False,
                },
            },
            "dependency_gate": {
                "status": "blocked",
                **dependencies,
                "lockfile_present_and_validated": False,
            },
            "resource_gate": {
                "status": "blocked",
                **config["resource_contract"],
                "four_base_plus_lora_disk_floor_gib": ">59.88",
                "single_selected_base_capacity_validated": False,
                "runtime_vram_measurement_performed": False,
                "runtime_disk_reserve_measurement_performed": False,
                "paper_eight_process_training_supported_on_single_gpu": False,
            },
            "solver_prediction_gate": {
                "status": "blocked",
                "prediction_produced": False,
                "challenge_only_adapter_executed": False,
                "strict_parity_contract_validated": False,
                "prior_seed_program_smoke_promoted": False,
            },
            "blockers": [
                {"id": blocker, "status": "blocked"}
                for blocker in EXPECTED_BLOCKER_IDS
            ],
            "gate_summary": {
                "static_audit_status": "passed",
                "blocked_gate_count": len(EXPECTED_BLOCKER_IDS),
                "method_status": "blocked",
            },
            "controls": controls,
            "read_ledger": ledger.snapshot(),
            "validation": {
                "canonical_config_exact": True,
                "source_lock_exact": True,
                "git_local_config_exact_and_safe": True,
                "git_subprocess_config_aba_isolated": True,
                "git_revision_tree_and_listing_exact": True,
                "git_metadata_stable_after_reads": True,
                "source_root_path_identity_stable": True,
                "closed_world_path_type_size_mode_exact": True,
                "closed_world_metadata_stable_after_reads": True,
                "retained_13_files_byte_sha_and_signature_exact": True,
                "restricted_worktree_leaves_not_opened_by_auditor_process": True,
                "static_label_flow_contracts_detected": True,
                "root_vs_vendored_license_scope_separated": True,
            },
            "observation_digest_sha256": canonical_sha256(stable_observation),
            "claim_boundary": (
                "This is a metadata-first blocker audit only. Git revision, commit-tree "
                "and ls-tree values describe committed Git metadata. The closed working "
                "tree check proves exact paths, entry types, modes and sizes; it does not "
                "claim byte-exact content for 1464 metadata-only tracked files. Only the "
                "13 retained source/small-metadata files are byte- and SHA-256-exact. No "
                "BARC model was loaded and no ARC solver prediction was produced. Fixed "
                "Git commands run against an auditor-owned config-free Git-dir and the "
                "pinned source object directory; they may read object-database "
                "indexes/packs. That subprocess byte volume is unmeasured and no "
                "worktree content command is allowed."
            ),
            "limitations": [
                "The .git directory is opaque except for exact safe reads of .git/config and detached HEAD, a pinned objects-directory descriptor, and terminal absence checks for six dangerous auxiliary paths; it is not recursively inventoried.",
                "No git status or diff command is used because it could read restricted working-tree bytes.",
                "The auditor process did not open or hash ARC/answer JSON and JSONL, PNG, pickle, bytecode, or model-weight leaves; fixed Git subprocess object-database byte reads are separately disclosed as possible and unmeasured.",
                "Git blob ids pin committed-tree objects; except for the 13 retained files, correspondence to working-tree bytes is not asserted beyond path/type/mode/size metadata.",
                "External Hugging Face/model cache roots were not inspected and no remote artifact provenance was verified.",
                "The 14.97 GiB/base and host-capacity values are planning inputs, not measurements made by this audit.",
            ],
        }
    )
    return record


class FreshOutput:
    def __init__(
        self,
        descriptor: int,
        parent_descriptor: int,
        parent_path: Path,
        leaf: str,
        created: os.stat_result,
    ) -> None:
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.parent_path = parent_path
        self.leaf = leaf
        self.created_identity = (created.st_dev, created.st_ino, created.st_mode)

    def verify_leaf(self) -> None:
        try:
            verify_directory_path_identity(
                self.parent_path, self.parent_descriptor
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise OutputPathError(
                "fresh output parent path was replaced after creation"
            ) from error
        path_info = os.stat(
            self.leaf,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        fd_info = os.fstat(self.descriptor)
        if not stat.S_ISDIR(path_info.st_mode) or stat.S_ISLNK(path_info.st_mode):
            raise OutputPathError("fresh output leaf is no longer a directory")
        if stat_signature(path_info) != stat_signature(fd_info):
            raise OutputPathError("fresh output leaf was replaced after creation")
        if (fd_info.st_dev, fd_info.st_ino, fd_info.st_mode) != self.created_identity:
            raise OutputPathError("fresh output descriptor identity changed")

    def close(self, *, record_committed: bool = False) -> None:
        errors: list[OSError] = []
        for descriptor in (self.descriptor, self.parent_descriptor):
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        if errors and not record_committed:
            raise errors[0]


def create_fresh_output(path: Path) -> FreshOutput:
    if lexical_within(path, SOURCE_PATH) or lexical_within(path, ROOT / "configs"):
        raise OutputPathError("output directory may not be inside BARC source or configs")
    path_parts = _absolute_parts(path)
    parent_path = Path(*path_parts[:-1])
    parent_fd, leaf = open_absolute_parent(path)
    output_fd: int | None = None
    try:
        try:
            os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise OutputPathError("output path must not exist") from error
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise OutputPathError(
                    "output path has a symlink/non-directory component"
                ) from error
            raise
        path_info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(path_info.st_mode) or stat.S_ISLNK(path_info.st_mode):
            raise OutputPathError("created output leaf is not a directory")
        output_fd = os.open(leaf, directory_flags(), dir_fd=parent_fd)
        fd_info = os.fstat(output_fd)
        if stat_signature(path_info) != stat_signature(fd_info):
            raise OutputPathError("created output leaf raced before pinning")
        output = FreshOutput(output_fd, parent_fd, parent_path, leaf, fd_info)
        output.verify_leaf()
        # Persist the newly created directory entry before any report can be
        # published inside it.  The output-directory sync in
        # write_json_no_clobber commits run.json itself.
        os.fsync(parent_fd)
        output.verify_leaf()
        return output
    except BaseException:
        if output_fd is not None:
            os.close(output_fd)
        os.close(parent_fd)
        raise


def rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    """Atomically move one directory entry without replacing the destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("renameat2 is required for no-clobber report publication")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def write_json_no_clobber(output: FreshOutput, value: dict[str, Any]) -> None:
    """Durably publish run.json without overwriting or deleting any path."""

    output.verify_leaf()
    output_fd = output.descriptor
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = f".run.json.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor: int | None = None
    owned_identity: tuple[int, int, int, int] | None = None
    record_committed = False

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_size,
        )

    def descriptor_payload_matches() -> bool:
        if descriptor is None or owned_identity is None:
            return False
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            before = os.fstat(descriptor)
            if before.st_nlink != 1 or identity(before) != owned_identity:
                return False
            observed, after = _read_fd_stable(descriptor, before, len(payload))
        except (OSError, ValueError, RuntimeError):
            return False
        return (
            after.st_nlink == 1
            and identity(after) == owned_identity
            and observed == payload
        )

    def final_path_matches() -> bool:
        if owned_identity is None:
            return False
        try:
            observed = os.stat(
                "run.json", dir_fd=output_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return False
        return (
            stat.S_ISREG(observed.st_mode)
            and observed.st_nlink == 1
            and identity(observed) == owned_identity
        )

    def directory_entries_match() -> bool:
        try:
            return os.listdir(output_fd) == ["run.json"]
        except OSError:
            return False

    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=output_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing BARC run record")
            view = view[written:]
        os.fsync(descriptor)
        written_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(written_info.st_mode)
            or written_info.st_nlink != 1
            or written_info.st_size != len(payload)
        ):
            raise OutputPathError("temporary report file identity is invalid")
        owned_identity = identity(written_info)
        if not descriptor_payload_matches():
            raise OutputPathError("temporary report descriptor changed before publish")
        temporary_info = os.stat(
            temporary, dir_fd=output_fd, follow_symlinks=False
        )
        if identity(temporary_info) != owned_identity:
            raise OutputPathError("temporary report path changed before publish")
        output.verify_leaf()
        rename_noreplace(output_fd, temporary, "run.json")
        if (
            not final_path_matches()
            or not descriptor_payload_matches()
            or not directory_entries_match()
        ):
            raise OutputPathError("published report identity or payload mismatch")
        # The directory sync is the commit point.  A valid path alone must
        # never recover a failed sync into success.
        os.fsync(output_fd)
        output.verify_leaf()
        if (
            not final_path_matches()
            or not descriptor_payload_matches()
            or not directory_entries_match()
        ):
            raise OutputPathError("report path identity changed after commit sync")
        record_committed = True
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                if not record_committed:
                    raise


def execute_audit(
    config_path: Path, output_path: Path
) -> tuple[int, dict[str, Any]]:
    validate_config_location(config_path, output_path)
    output = create_fresh_output(output_path)
    ledger = ReadLedger()
    started = time.perf_counter()
    started_at = utc_now()
    record_committed = False
    try:
        try:
            record = run_static_audit(config_path, output_path.name, ledger)
        except BaseException as error:
            record = failure_record(output_path.name, "static-audit", error, ledger)
            exit_code = 1
        else:
            exit_code = 0
        record["started_at_utc"] = started_at
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_time_seconds": round(time.perf_counter() - started, 6),
            "current_process_max_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "scope": "auditor-process-only; fixed Git subprocesses excluded",
        }
        write_json_no_clobber(output, record)
        record_committed = True
        return exit_code, record
    finally:
        output.close(record_committed=record_committed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config_path = lexical_path(str(args.config))
    output_path = lexical_path(str(args.output_directory))
    try:
        exit_code, record = execute_audit(config_path, output_path)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    stream = sys.stdout if exit_code == 0 else sys.stderr
    try:
        print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False), file=stream)
    except BrokenPipeError:
        pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
