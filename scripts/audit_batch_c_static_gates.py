#!/usr/bin/env python3
"""Run the metadata-first static blocker gate for one Batch C method.

The auditor never imports or executes upstream code, opens ARC JSON/solution
leaves, deserializes artifacts, initializes a provider/GPU, or intentionally
uses the network.  A passing record means that the pinned static observations
and blockers were reproduced.  It is not a solver smoke or a prediction.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tokenize
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
AUDITOR_PATH = ROOT / "scripts" / "audit_batch_c_static_gates.py"
CANONICAL_CONFIG_RELATIVE = "configs/batch_c_static_gate_v1.json"
SOURCE_LOCK_RELATIVE = "configs/source_locks.json"
CONFIG_ID = "batch-c-static-source-label-api-artifact-gate-v1"
SCOPE = "static-source-label-api-artifact-blocker-audit-only"
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "57bcf9023a8f895fb7287037ae9f93f83ff87d3b0f4c8301725034f17ec41753"
)
EXPECTED_SOURCE_LOCK_SHA256 = (
    "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
)
EXPECTED_METHOD_IDS = ("arcmemo", "arc-lang-public", "epang-arc-agi")
TEST_OUTPUT_ROOT = Path("/tmp/arc-agi-eval-batch-c-tests")
MAX_CONFIG_BYTES = 512 * 1024
MAX_PRIOR_REPORT_BYTES = 512 * 1024
MAX_RETAINED_BYTES = 4 * 1024 * 1024
MAX_GIT_STDOUT_BYTES = 512 * 1024
MAX_GIT_STDERR_BYTES = 16 * 1024
GIT_TIMEOUT_SECONDS = 10.0
RENAME_NOREPLACE = 1
FORBIDDEN_RETAINED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".pkl",
    ".pickle",
    ".pdf",
    ".ipynb",
    ".pyc",
    ".png",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
}
ROOT_LICENSE_NAMES = {
    "license",
    "license.md",
    "license.txt",
    "copying",
    "copying.md",
    "copying.txt",
    "notice",
    "notice.md",
    "notice.txt",
}
FORBIDDEN_GIT_PATHS = (
    ".git/commondir",
    ".git/gitdir",
    ".git/config.worktree",
    ".git/info/attributes",
    ".git/objects/info/alternates",
    ".git/objects/info/http-alternates",
)
GIT_TREE_RECORD_RE = re.compile(
    rb"^(?P<mode>[0-7]{6}) (?P<type>blob|commit) "
    rb"(?P<oid>[0-9a-f]{40})\t(?P<path>.+)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class OutputPathError(ValueError):
    """The output path is not a fresh, pinned directory leaf."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def usage_snapshot() -> dict[str, resource.struct_rusage]:
    return {
        "self": resource.getrusage(resource.RUSAGE_SELF),
        "children": resource.getrusage(resource.RUSAGE_CHILDREN),
    }


def usage_record(
    started: dict[str, resource.struct_rusage],
    ended: dict[str, resource.struct_rusage],
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "wall_seconds": wall_seconds,
        "self_user_cpu_seconds": ended["self"].ru_utime - started["self"].ru_utime,
        "self_system_cpu_seconds": ended["self"].ru_stime - started["self"].ru_stime,
        "child_user_cpu_seconds": ended["children"].ru_utime
        - started["children"].ru_utime,
        "child_system_cpu_seconds": ended["children"].ru_stime
        - started["children"].ru_stime,
        "max_rss_kib": ended["self"].ru_maxrss,
        "provider_requests": 0,
        "currency_spend_usd": 0.0,
        "gpu_used": False,
        "network_used": False,
    }


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
        raise ValueError(f"non-finite JSON constant forbidden in {field}: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {field}: {key}")
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
        raise ValueError(f"{field} is not canonical POSIX syntax")
    return value


def _absolute_parts(path: Path) -> tuple[str, ...]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if not absolute.is_absolute() or any(part == ".." for part in absolute.parts):
        raise ValueError(f"unsafe absolute path: {path}")
    return absolute.parts


def lexical_equal(left: Path, right: Path) -> bool:
    return _absolute_parts(left) == _absolute_parts(right)


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
        if not hasattr(os, name):
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
            before_path = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(before_path.st_mode):
                raise ValueError(f"non-directory component in absolute path: {path}")
            next_descriptor = os.open(part, directory_flags(), dir_fd=descriptor)
            if stat_signature(before_path) != stat_signature(os.fstat(next_descriptor)):
                os.close(next_descriptor)
                raise RuntimeError(f"absolute directory component raced: {path}")
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"not a directory: {path}")
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
    parts = PurePosixPath(safe_relative_path(path, "relative file path")).parts
    descriptor = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            before_path = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(before_path.st_mode):
                raise ValueError(f"non-directory component in relative path: {path}")
            next_descriptor = os.open(part, directory_flags(), dir_fd=descriptor)
            if stat_signature(before_path) != stat_signature(os.fstat(next_descriptor)):
                os.close(next_descriptor)
                raise RuntimeError(f"relative directory component raced: {path}")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def verify_directory_identity(path: Path, descriptor: int) -> None:
    reopened = open_absolute_directory(path)
    try:
        if stat_signature(os.fstat(reopened)) != stat_signature(os.fstat(descriptor)):
            raise RuntimeError(f"directory identity changed during audit: {path}")
    finally:
        os.close(reopened)


class ReadLedger:
    def __init__(self, retained_policy: dict[str, str]) -> None:
        self.retained_policy = dict(retained_policy)
        self.file_reads: list[dict[str, Any]] = []
        self.git_processes: list[dict[str, Any]] = []

    def begin(self, path: str, role: str, category: str) -> dict[str, Any]:
        record = {
            "path": path,
            "role": role,
            "category": category,
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "status": "authorized-attempt",
        }
        self.file_reads.append(record)
        return record

    def authorize_retained(self, path: str, role: str) -> None:
        safe_relative_path(path, "retained path")
        if self.retained_policy.get(path) != role:
            raise ValueError(f"retained read not in closed policy: {role}:{path}")
        if PurePosixPath(path).suffix.lower() in FORBIDDEN_RETAINED_SUFFIXES:
            raise ValueError(f"restricted suffix entered retained reader: {path}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "file_read_attempts": [dict(item) for item in self.file_reads],
            "git_subprocesses": [dict(item) for item in self.git_processes],
        }


def _read_stable_fd(
    descriptor: int,
    before: os.stat_result,
    max_bytes: int,
    record: dict[str, Any],
) -> bytes:
    if before.st_size > max_bytes:
        raise ValueError(f"file exceeds byte limit: {before.st_size} > {max_bytes}")
    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        hasher.update(chunk)
        total += len(chunk)
        record.update(bytes=total, sha256=hasher.hexdigest(), status="reading")
        if total > max_bytes:
            raise ValueError("file grew beyond byte limit")
    after = os.fstat(descriptor)
    if stat_signature(before) != stat_signature(after):
        raise RuntimeError("file changed during verified read")
    if total != before.st_size:
        raise RuntimeError("read byte count differs from fstat size")
    record.update(bytes=total, sha256=hasher.hexdigest(), status="completed")
    return b"".join(chunks)


def secure_read_absolute(
    path: Path,
    *,
    role: str,
    category: str,
    max_bytes: int,
    ledger: ReadLedger,
) -> bytes:
    record = ledger.begin(str(path), role, category)
    descriptor: int | None = None
    parent_fd: int | None = None
    try:
        parent_fd, leaf = open_absolute_parent(path)
        before_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise ValueError(f"absolute input is not a single-link regular file: {path}")
        descriptor = os.open(leaf, regular_file_flags(), dir_fd=parent_fd)
        before_fd = os.fstat(descriptor)
        if stat_signature(before_path) != stat_signature(before_fd):
            raise RuntimeError(f"absolute input raced before read: {path}")
        payload = _read_stable_fd(descriptor, before_fd, max_bytes, record)
        after_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat_signature(os.fstat(descriptor)) != stat_signature(after_path):
            raise RuntimeError(f"absolute input raced after read: {path}")
        return payload
    except BaseException as error:
        record["status"] = "failed"
        record["failure_type"] = type(error).__name__
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def secure_read_retained(
    root_fd: int,
    path: str,
    role: str,
    expected_signature: tuple[int, ...],
    ledger: ReadLedger,
) -> bytes:
    ledger.authorize_retained(path, role)
    record = ledger.begin(path, role, "retained_source_text")
    descriptor: int | None = None
    parent_fd: int | None = None
    try:
        parent_fd, leaf = open_relative_parent(root_fd, path)
        before_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat_signature(before_path) != expected_signature:
            raise RuntimeError(f"retained source changed after inventory: {path}")
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise ValueError(f"retained source is not a single-link regular file: {path}")
        descriptor = os.open(leaf, regular_file_flags(), dir_fd=parent_fd)
        before_fd = os.fstat(descriptor)
        if stat_signature(before_path) != stat_signature(before_fd):
            raise RuntimeError(f"retained source raced before read: {path}")
        payload = _read_stable_fd(descriptor, before_fd, MAX_RETAINED_BYTES, record)
        after_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat_signature(os.fstat(descriptor)) != stat_signature(after_path):
            raise RuntimeError(f"retained source raced after read: {path}")
        return payload
    except BaseException as error:
        record["status"] = "failed"
        record["failure_type"] = type(error).__name__
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def parse_git_tree(payload: bytes) -> dict[str, dict[str, str]]:
    if not payload or not payload.endswith(b"\0"):
        raise ValueError("Git tree listing is not NUL-terminated")
    result: dict[str, dict[str, str]] = {}
    for raw in payload[:-1].split(b"\0"):
        match = GIT_TREE_RECORD_RE.fullmatch(raw)
        if match is None or match.group("type") != b"blob":
            raise ValueError("unsupported entry in locked Git tree")
        if match.group("mode") not in {b"100644", b"100755"}:
            raise ValueError("locked Batch C tree contains an unsupported leaf mode")
        try:
            path = match.group("path").decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Git tree path is not UTF-8") from error
        safe_relative_path(path, "Git tree path")
        if path in result:
            raise ValueError(f"duplicate Git tree path: {path}")
        result[path] = {
            "path": path,
            "mode": match.group("mode").decode("ascii"),
            "blob_oid": match.group("oid").decode("ascii"),
        }
    return result


def git_tree_manifest(tree: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [tree[path] for path in sorted(tree)]


def extension_counts(paths: set[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        pure = PurePosixPath(path)
        name = pure.name.lower()
        suffix = name if name.startswith(".") and name.count(".") == 1 else pure.suffix.lower() or "<none>"
        result[suffix] = result.get(suffix, 0) + 1
    return dict(sorted(result.items()))


def git_object_metadata(git_fd: int) -> dict[str, tuple[int, ...]]:
    objects_fd = os.open("objects", directory_flags(), dir_fd=git_fd)
    result: dict[str, tuple[int, ...]] = {}

    def visit(descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."}:
                raise ValueError("invalid Git object entry")
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            path = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, directory_flags(), dir_fd=descriptor)
                try:
                    if stat_signature(info) != stat_signature(os.fstat(child)):
                        raise RuntimeError(f"Git object directory raced before visit: {path}")
                    result[path + "/"] = stat_signature(info)
                    visit(child, path)
                    after_path = os.stat(
                        name, dir_fd=descriptor, follow_symlinks=False
                    )
                    if stat_signature(after_path) != stat_signature(os.fstat(child)):
                        raise RuntimeError(f"Git object directory raced after visit: {path}")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ValueError(f"Git object leaf has a hardlink alias: {path}")
                result[path] = stat_signature(info)
            else:
                raise ValueError(f"unsafe Git object entry type: {path}")

    try:
        visit(objects_fd, "")
    finally:
        os.close(objects_fd)
    return result


def require_git_path_absent(git_fd: int, repository_relative: str) -> None:
    prefix = ".git/"
    if not repository_relative.startswith(prefix):
        raise ValueError("forbidden Git path must be inside .git")
    relative = repository_relative[len(prefix) :]
    parent_fd: int | None = None
    try:
        try:
            parent_fd, leaf = open_relative_parent(git_fd, relative)
        except FileNotFoundError:
            return
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ValueError(f"forbidden auxiliary Git path exists: {repository_relative}")
    except OSError as error:
        if error.errno == errno.ENOENT:
            return
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def parse_git_config_entries(payload: bytes) -> dict[tuple[str, str], str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Git local config is not UTF-8") from error
    section: str | None = None
    entries: dict[tuple[str, str], str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section not in {"core", 'remote "origin"', 'branch "main"'}:
                raise ValueError(f"unknown Git local config section: {section}")
            continue
        if section is None or "=" not in line:
            raise ValueError(f"malformed Git local config line: {line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        normalized = (section, key.lower())
        if not key or normalized in entries:
            raise ValueError("duplicate or empty Git local config key")
        entries[normalized] = value
    return entries


def validate_git_config_entries(
    payload: bytes, repository_url: str, branch: str
) -> None:
    if branch != "main":
        raise ValueError("Batch C Git config only permits the pinned main branch")
    expected = {
        ("core", "repositoryformatversion"): "1",
        ("core", "filemode"): "true",
        ("core", "bare"): "false",
        ("core", "logallrefupdates"): "true",
        ('remote "origin"', "url"): repository_url,
        ('remote "origin"', "fetch"): "+refs/heads/*:refs/remotes/origin/*",
        ('remote "origin"', "promisor"): "true",
        ('remote "origin"', "partialclonefilter"): "blob:none",
        ('branch "main"', "remote"): "origin",
        ('branch "main"', "merge"): "refs/heads/main",
    }
    if parse_git_config_entries(payload) != expected:
        raise ValueError("Git local config differs from the exact safe allowlist")


def initialize_isolated_git_directory(path: Path) -> int:
    os.chmod(path, 0o700)
    root_fd = open_absolute_directory(path)
    try:
        for name in ("objects", "refs"):
            os.mkdir(name, mode=0o700, dir_fd=root_fd)
        descriptor = os.open(
            "HEAD",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=root_fd,
        )
        try:
            payload = b"ref: refs/heads/unused\n"
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(root_fd)
        return root_fd
    except BaseException:
        os.close(root_fd)
        raise


def run_git(
    isolated_git_fd: int,
    object_directory_fd: int,
    arguments: tuple[str, ...],
    allowed: set[tuple[str, ...]],
    ledger: ReadLedger,
) -> bytes:
    if arguments not in allowed:
        raise ValueError(f"Git command outside exact allowlist: {arguments}")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_DIR": f"/proc/self/fd/{isolated_git_fd}",
        "GIT_OBJECT_DIRECTORY": f"/proc/self/fd/{object_directory_fd}",
    }
    process_record: dict[str, Any] = {
        "argv": ["/usr/bin/git", *arguments],
        "status": "authorized-attempt",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "worktree_content_requested": False,
        "untrusted_system_or_global_config_available": False,
        "source_repository_local_config_available": False,
        "isolated_git_directory_used": True,
    }
    ledger.git_processes.append(process_record)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            ["/usr/bin/git", *arguments],
            cwd="/",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            pass_fds=(isolated_git_fd, object_directory_fd),
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=GIT_TIMEOUT_SECONDS)
        process_record.update(
            stdout_bytes=len(stdout), stderr_bytes=len(stderr), returncode=process.returncode
        )
        if len(stdout) > MAX_GIT_STDOUT_BYTES or len(stderr) > MAX_GIT_STDERR_BYTES:
            raise RuntimeError("Git metadata output exceeded its fixed byte cap")
        if process.returncode != 0:
            raise RuntimeError(
                "fixed Git metadata command failed: "
                + stderr.decode("utf-8", "replace").strip()
            )
        process_record["status"] = "completed"
        return stdout
    except BaseException as error:
        process_record.update(status="failed", failure_type=type(error).__name__)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1.0)
        raise
    finally:
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


def walk_worktree_metadata(
    root_fd: int,
    tree: dict[str, dict[str, str]],
    opaque_paths: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[int, ...]], list[dict[str, Any]]]:
    expected_files = set(tree)
    expected_directories: set[str] = set()
    for path in expected_files:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            expected_directories.add("/".join(parts[:index]))
    files: dict[str, dict[str, Any]] = {}
    signatures: dict[str, tuple[int, ...]] = {}
    opaque_observed: list[dict[str, Any]] = []

    def visit(descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(descriptor)):
            if not prefix and name == ".git":
                continue
            path = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            signatures[path] = stat_signature(info)
            if path in opaque_paths:
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"declared opaque path is not a directory: {path}")
                opaque_observed.append(
                    {"path": path, "kind": "directory", "contents_inspected": False}
                )
                continue
            if stat.S_ISDIR(info.st_mode):
                if path not in expected_directories:
                    raise ValueError(f"unknown worktree directory: {path}")
                child = os.open(name, directory_flags(), dir_fd=descriptor)
                try:
                    if stat_signature(info) != stat_signature(os.fstat(child)):
                        raise RuntimeError(f"worktree directory raced before visit: {path}")
                    visit(child, path)
                    after_path = os.stat(
                        name, dir_fd=descriptor, follow_symlinks=False
                    )
                    if stat_signature(after_path) != stat_signature(os.fstat(child)):
                        raise RuntimeError(f"worktree directory raced after visit: {path}")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if path not in expected_files:
                    raise ValueError(f"unknown worktree file: {path}")
                if info.st_nlink != 1:
                    raise ValueError(f"worktree leaf has a hardlink alias: {path}")
                files[path] = {
                    "path": path,
                    "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                    "bytes": info.st_size,
                }
            else:
                raise ValueError(f"unsafe worktree entry type: {path}")

    visit(root_fd, "")
    if set(files) != expected_files:
        missing = sorted(expected_files - set(files))
        raise ValueError(f"tracked worktree leaves missing: {missing[:5]}")
    if set(item["path"] for item in opaque_observed) != opaque_paths:
        raise ValueError("declared opaque worktree paths do not match observed paths")
    return files, signatures, opaque_observed


def parse_python(payload: bytes, path: str) -> ast.Module:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
        text = payload.decode(encoding)
        return ast.parse(text, filename=path)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError(f"retained Python source cannot be parsed: {path}: {error}") from error


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def function_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one function named {name}, found {len(matches)}")
    return matches[0]


def calls(scope: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(scope) if isinstance(node, ast.Call)]


def call_names(scope: ast.AST) -> list[str]:
    return [name for node in calls(scope) if (name := dotted_name(node.func)) is not None]


def string_constants(scope: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(scope)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def keyword_names(call: ast.Call) -> set[str]:
    return {item.arg for item in call.keywords if item.arg is not None}


def call_has_keyword(call: ast.Call, keyword: str) -> bool:
    return keyword in keyword_names(call)


def top_level_call_names(tree: ast.Module) -> list[str]:
    result: list[str] = []
    for statement in tree.body:
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and node is not statement:
                continue
            if isinstance(node, ast.Call):
                name = dotted_name(node.func)
                if name:
                    result.append(name)
    return result


def source_text(payload: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
    return payload.decode(encoding)


class _ModuleScopeCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if name:
            self.names.append(name)
        self.generic_visit(node)


def module_scope_call_names(tree: ast.Module) -> list[str]:
    visitor = _ModuleScopeCallVisitor()
    visitor.visit(tree)
    return visitor.names


def class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one top-level class named {name}")
    return matches[0]


def assigned_names(scope: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(scope):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                targets = [node.target]
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name):
                    result.add(child.id)
    return result


def names_and_attributes(scope: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Name):
            result.add(node.id)
        elif isinstance(node, ast.Attribute):
            result.add(node.attr)
            if (name := dotted_name(node)) is not None:
                result.add(name)
    return result


def direct_call_assignments(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Call) or (call_name := dotted_name(value.func)) is None:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                result[target.id] = call_name
    return result


def argument_default(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> ast.AST | None:
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults: dict[str, ast.AST] = {}
    offset = len(positional) - len(function.args.defaults)
    for index, default in enumerate(function.args.defaults):
        positional_defaults[positional[offset + index].arg] = default
    keyword_defaults = {
        argument.arg: default
        for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if default is not None
    }
    return {**positional_defaults, **keyword_defaults}.get(name)


def call_line_numbers(scope: ast.AST, terminal: str) -> list[int]:
    return [
        node.lineno
        for node in calls(scope)
        if (name := dotted_name(node.func)) is not None
        and (name == terminal or name.endswith("." + terminal))
    ]


def analyze_arc_lang(
    trees: dict[str, ast.Module], payloads: dict[str, bytes], tree_paths: set[str]
) -> dict[str, Any]:
    del tree_paths
    run_tree = trees["src/run.py"]
    models_tree = trees["src/models.py"]
    structured_tree = trees["src/llms/structured.py"]
    init_tree = trees["src/__init__.py"]
    logging_tree = trees["src/logging_config.py"]
    run_from_json = function_def(run_tree, "run_from_json")
    default_run = function_def(run_tree, "run")

    truth_read_lines: list[int] = []
    for node in calls(run_from_json):
        if dotted_name(node.func) == "truth_solutions_path.read_text":
            truth_read_lines.append(node.lineno)
    solver_lines = call_line_numbers(run_from_json, "solve_challenges")
    default_run_calls = calls(default_run)
    default_truth_keyword = any(
        (name := dotted_name(call.func)) is not None
        and name.endswith("run_from_json")
        and any(
            keyword.arg == "truth_solutions_path"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "solutions_path"
            for keyword in call.keywords
        )
        for call in default_run_calls
    )
    truth_if_has_else = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "truth_solutions_path"
        and bool(node.orelse)
        for node in ast.walk(run_from_json)
    )
    input_class = class_def(models_tree, "Input")
    input_fields = sorted(
        node.target.id
        for node in input_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )
    module_clients = direct_call_assignments(structured_tree)
    expected_clients = {
        "openai_client": "AsyncOpenAI",
        "anthropic_client": "AsyncAnthropic",
        "deepseek_client": "AsyncOpenAI",
        "openrouter_client": "AsyncOpenAI",
        "gemini_client": "genai.Client",
        "API_SEMAPHORE": "MonitoredSemaphore",
    }
    source_names = {
        path: names_and_attributes(parsed) for path, parsed in trees.items()
    }
    forbidden_exec_terms = {"exec", "eval", "compile", "os.system", "subprocess.run", "subprocess.Popen"}
    observed_exec_terms = sorted(
        term
        for term in forbidden_exec_terms
        if any(term in values for values in source_names.values())
    )
    runtime_timeout_refs = sum(
        "timeout_secs" in source_names[path]
        for path in ("src/run.py", "src/llms/messages.py", "src/llms/openai_responses.py", "src/llms/structured.py")
    )
    provider_sink_suffixes = (
        "messages.create",
        "responses.create",
        "responses.retrieve",
        "chat.completions.create",
        "chat.create",
        "generate_content",
    )
    provider_sinks: list[dict[str, Any]] = []
    for path in (
        "src/llms/messages.py",
        "src/llms/openai_responses.py",
        "src/llms/structured.py",
    ):
        for node in calls(trees[path]):
            name = dotted_name(node.func)
            if name is not None and any(
                name.endswith(suffix) for suffix in provider_sink_suffixes
            ):
                provider_sinks.append(
                    {"path": path, "line": node.lineno, "call": name}
                )
    provider_sinks.sort(key=lambda item: (item["path"], item["line"], item["call"]))
    all_text = "\n".join(source_text(payloads[path]) for path in sorted(payloads))
    return {
        "default_runner_passes_solution_path": default_truth_keyword,
        "truth_solution_read_detected": bool(truth_read_lines),
        "truth_read_precedes_solver_call": bool(truth_read_lines and solver_lines)
        and max(truth_read_lines) < min(solver_lines),
        "challenge_only_none_branch_detected": truth_if_has_else,
        "truth_parameter_has_none_default": isinstance(
            argument_default(run_from_json, "truth_solutions_path"), ast.Constant
        )
        and argument_default(run_from_json, "truth_solutions_path").value is None,
        "raw_input_model_fields": input_fields,
        "raw_input_extra_forbid_declared": "model_config" in assigned_names(input_class),
        "eager_provider_clients": module_clients,
        "expected_eager_provider_clients_bound": all(
            module_clients.get(name) == target for name, target in expected_clients.items()
        ),
        "dotenv_called_at_import": "load_dotenv" in module_scope_call_names(init_tree)
        and "load_dotenv" in module_scope_call_names(logging_tree),
        "remote_logfire_default_expression_detected": "send_to_logfire=not LOCAL_LOGS_ONLY" in source_text(payloads["src/logging_config.py"]),
        "scrubbing_returns_original_value": "return m.value" in source_text(payloads["src/logging_config.py"]),
        "neon_egress_path_detected": "NEON_DSN" in source_text(payloads["src/run.py"]),
        "provider_request_sinks_across_retained_llm_sources": provider_sinks,
        "provider_request_sink_callsite_count": len(provider_sinks),
        "runtime_timeout_secs_reference_file_count": runtime_timeout_refs,
        "pre_request_budget_reservation_detected": any(
            term in all_text.lower()
            for term in ("pre_debit", "predebit", "reserve_budget", "request_budget.reserve")
        ),
        "max_retry_20_detected": "max_retries=20" in source_text(payloads["src/llms/structured.py"]),
        "mutable_model_aliases_detected": all(
            alias in source_text(payloads["src/llms/models.py"])
            for alias in ('"gpt-5.2"', '"grok-4"')
        ),
        "generated_code_execution_terms": observed_exec_terms,
        "generated_code_execution_detected": bool(observed_exec_terms),
    }


def analyze_arcmemo(
    trees: dict[str, ast.Module], payloads: dict[str, bytes], tree_paths: set[str]
) -> dict[str, Any]:
    driver = trees["concept_mem/evaluation/driver.py"]
    continual = trees["concept_mem/evaluation/continual_driver.py"]
    solution_tree = trees["concept_mem/evaluation/solution_tree.py"]
    score_tree = trees["concept_mem/evaluation/score_tree.py"]
    retry_tree = trees["concept_mem/evaluation/retry_policy.py"]
    executor = trees["concept_mem/utils/code_execution/exec.py"]
    lesson_memory = trees["concept_mem/lesson_memory.py"]
    llm_job = trees["concept_mem/utils/llm_job.py"]
    default_text = source_text(payloads["configs/default.yaml"])
    prompt_text = source_text(payloads["concept_mem/evaluation/prompts.py"])
    readme_text = source_text(payloads["readme.md"])
    requirements_text = source_text(payloads["requirements.txt"])

    async_main = function_def(driver, "async_main")
    initial_solve = function_def(driver, "initial_solve_step")
    strict_method = function_def(solution_tree, "is_strictly_correct")
    worker = function_def(executor, "_worker")
    terminate = function_def(executor, "terminate_all_processes")
    official_pair = function_def(score_tree, "_official_per_pair")
    official = function_def(score_tree, "official_score")
    select_concepts = function_def(lesson_memory, "select_concepts")
    continual_initial = function_def(continual, "_process_batch") if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_process_batch"
        for node in ast.walk(continual)
    ) else function_def(continual, "process_batch")
    update_memory = function_def(continual, "update_memory")

    continual_calls = calls(continual_initial)
    strict_lines = [
        node.lineno
        for node in continual_calls
        if (name := dotted_name(node.func)) is not None and name.endswith("is_strictly_correct")
    ]
    update_lines = [
        node.lineno
        for node in continual_calls
        if (name := dotted_name(node.func)) is not None and name.endswith("update_memory")
    ]
    update_calls = calls(update_memory)
    select_calls = [
        node
        for node in calls(continual)
        if (name := dotted_name(node.func)) is not None and name.endswith("select_concepts")
    ]
    update_llm_calls = [
        node
        for node in update_calls
        if (name := dotted_name(node.func)) is not None
        and any(name.endswith(suffix) for suffix in ("thought_process", "extract_lessons"))
    ]
    disallowed_assignment = next(
        (
            node
            for node in executor.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DISALLOWED_IMPORTS" for target in node.targets)
        ),
        None,
    )
    disallowed_values: list[str] = []
    if isinstance(disallowed_assignment, ast.Assign) and isinstance(disallowed_assignment.value, (ast.Tuple, ast.List)):
        disallowed_values = [
            item.value
            for item in disallowed_assignment.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
    requirement_lines = [
        line.strip()
        for line in requirements_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    unpinned_requirements = [
        line
        for line in requirement_lines
        if "==" not in line and not re.search(r"@[0-9a-f]{40}(?:#|$)", line)
    ]
    readme_paths = {
        "selection_typo": "validation_n100_uids.json.json" in readme_text,
        "placeholder_prompt_info": ".../prompt_info.json" in readme_text,
    }
    missing_declared_artifacts = sorted(
        path
        for path in (
            "data/descriptions/gpt41_vlm.json",
            "data/lessons/from_trace_fs/gpt41_lessons.json",
        )
        if path not in tree_paths
    )
    return {
        "default_problem_data_is_null": bool(re.search(r"(?m)^\s*problem_data:\s*null\s*$", default_text)),
        "default_long_cot_selection_disabled": bool(re.search(r"(?m)^\s*use_lcs:\s*false\s*$", default_text)),
        "default_retry_criterion_train": 'criterion: "train"' in default_text,
        "dry_run_dummy_completion_detected": "if dry_run:" in source_text(payloads["concept_mem/utils/llm_job.py"])
        and '[["dummy"]' in source_text(payloads["concept_mem/utils/llm_job.py"]),
        "driver_loads_problems_and_initializes_llm": all(
            any(name.endswith(required) for name in call_names(async_main))
            for required in ("load_problems_from_config", "LLMClient")
        ),
        "driver_scores_attempts_inline": any(
            name.endswith("score_problem_attempt") for name in call_names(initial_solve)
        ),
        "prompt_test_input_only_signature": "[format_grid(pair.x) for pair in problem.test_pairs]" in prompt_text,
        "test_expected_output_scoring_detected": "io_pairs[i].y" in source_text(payloads["concept_mem/evaluation/score_tree.py"])
        and "problem.test_pairs" in source_text(payloads["concept_mem/evaluation/score_tree.py"]),
        "retry_test_criterion_available": "TEST" in assigned_names(retry_tree)
        or "TEST" in source_text(payloads["concept_mem/evaluation/retry_policy.py"]),
        "strict_correctness_combines_train_and_test": "self.train_results + self.test_results" in source_text(payloads["concept_mem/evaluation/solution_tree.py"])
        and "correct" in names_and_attributes(strict_method),
        "continual_strict_filter_precedes_memory_update": bool(strict_lines and update_lines)
        and min(strict_lines) < min(update_lines),
        "continual_selection_dry_run_propagated": bool(select_calls)
        and all(call_has_keyword(call, "dry_run") for call in select_calls),
        "continual_update_llm_dry_run_propagated": bool(update_llm_calls)
        and all(call_has_keyword(call, "dry_run") for call in update_llm_calls),
        "lesson_memory_select_accepts_dry_run": any(
            argument.arg == "dry_run"
            for argument in [*select_concepts.args.args, *select_concepts.args.kwonlyargs]
        ),
        "executor_exec_call_count": sum(
            1 for name in call_names(worker) if name == "exec"
        ),
        "executor_disallowed_imports": sorted(disallowed_values),
        "executor_dangerous_builtins_retained": not any(
            value in source_text(payloads["concept_mem/utils/code_execution/exec.py"])
            for value in ('safe_builtins.pop("open"', 'safe_builtins.pop("eval"', 'safe_builtins.pop("exec"', 'safe_builtins.pop("compile"')
        ),
        "executor_kills_all_recursive_descendants": "children(recursive=True)" in source_text(payloads["concept_mem/utils/code_execution/exec.py"])
        and any(name.endswith("kill_process") for name in call_names(terminate)),
        "official_attempt_limit_default_is_none": isinstance(argument_default(official, "attempts_allowed"), ast.Constant)
        and argument_default(official, "attempts_allowed").value is None,
        "official_per_case_any_detected": any(name.endswith("any") for name in call_names(official_pair)),
        "readme_artifact_path_issues": readme_paths,
        "missing_declared_method_artifacts": missing_declared_artifacts,
        "unpinned_requirement_count": len(unpinned_requirements),
        "unpinned_llmplus_git_dependency": any(line.startswith("llmplus @ git+") and "@" in line for line in unpinned_requirements),
    }


def analyze_epang(
    trees: dict[str, ast.Module], payloads: dict[str, bytes], tree_paths: set[str]
) -> dict[str, Any]:
    submission = trees["src/submission.py"]
    data_tree = trees["src/data.py"]
    models = trees["src/models.py"]
    logic = trees["src/logic.py"]
    executor = trees["src/run_python.py"]
    init_tree = trees["src/__init__.py"]
    llms = trees["src/llms/__init__.py"]
    experiments = trees["src/trees/experiments.py"]
    readme_text = source_text(payloads["README.md"])

    eager_data_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "src.data"
        and {alias.name for alias in node.names}
        >= {"eval_challenges", "training_challenges", "v2_training_challenges", "v2_eval_challenges"}
        for node in submission.body
    )
    module_assignments = direct_call_assignments(data_tree)
    data_text = source_text(payloads["src/data.py"])
    submission_text = source_text(payloads["src/submission.py"])
    executor_text = source_text(payloads["src/run_python.py"])
    llm_text = source_text(payloads["src/llms/__init__.py"])
    experiments_text = source_text(payloads["src/trees/experiments.py"])
    test_accuracy = function_def(models, "test_accuracy")
    eval_attempts = function_def(logic, "eval_attempts")
    llm_to_grids = function_def(models, "llm_response_to_result_grids")
    run_python = function_def(executor, "run_python_transform_sync")
    pickle_calls = [name for name in call_names(submission) if name in {"pickle.load", "pickle.loads"}]
    isolation_terms = {
        term
        for term in ("unshare", "seccomp", "setuid", "setgid", "setrlimit", "cgroup", "prctl")
        if term in executor_text.lower()
    }
    root_license_paths = sorted(
        path for path in tree_paths if "/" not in path and PurePosixPath(path).name.lower() in ROOT_LICENSE_NAMES
    )
    vendored_license_paths = sorted(
        path for path in tree_paths if "/" in path and PurePosixPath(path).name.lower() in ROOT_LICENSE_NAMES
    )
    return {
        "submission_eager_imports_all_data": eager_data_import,
        "data_module_global_challenge_assignments": sorted(
            name
            for name in ("training_challenges", "eval_challenges", "v2_training_challenges", "v2_eval_challenges")
            if name in module_assignments
        ),
        "arc1_solution_paths_at_module_import": all(
            path in data_text
            for path in ("arc-agi_training_solutions.json", "arc-agi_evaluation_solutions.json")
        ),
        "arc2_evaluation_loaded_at_module_import": "v2_eval_challenges = build_challenges_v2" in data_text,
        "path_argument_cannot_precede_eager_import": eager_data_import,
        "test_output_metric_detected": "challenge.test[0].output" in source_text(payloads["src/models.py"])
        and "test_attempt" in names_and_attributes(test_accuracy),
        "inline_test_aggregate_metrics_detected": all(
            term in names_and_attributes(eval_attempts)
            for term in ("test_accuracy",)
        )
        and all(term in assigned_names(eval_attempts) for term in ("avg_test_accuracy", "total_correct")),
        "llm_completion_to_generated_python_executor": all(
            any(name.endswith(required) for name in call_names(llm_to_grids))
            for required in ("parse_python_backticks", "run_python_transform_async")
        ),
        "executor_uses_host_tempfile_and_python": all(
            any(name.endswith(required) for name in call_names(run_python))
            for required in ("NamedTemporaryFile", "subprocess.Popen")
        ),
        "executor_isolation_terms": sorted(isolation_terms),
        "executor_only_kills_direct_child_on_timeout": "process.kill()" in executor_text
        and "start_new_session" not in executor_text
        and "killpg" not in executor_text,
        "executor_accepts_child_control_prefix": "TRANSFORM_RESULT:" in executor_text,
        "pickle_load_call_count": len(pickle_calls),
        "saved_library_path_detected": "saved_library_1000.pkl" in submission_text,
        "challenge_score_pickle_loader_detected": "challenge_primitive_accuracy_scores.pkl" in submission_text,
        "dotenv_called_at_import": "load_dotenv" in module_scope_call_names(init_tree),
        "logfire_remote_configuration_detected": "logfire.configure" in call_names(init_tree),
        "max_tokens_120000_detected": "max_tokens=120000" in llm_text,
        "pre_request_budget_reservation_detected": any(
            term in llm_text.lower() for term in ("pre_debit", "predebit", "reserve_budget")
        ),
        "grok_tree_five_attempts_detected": "grok_dreamcoder_tree" in experiments_text
        and "attempts=5" in experiments_text
        and "Model.grok_4" in experiments_text,
        "arc2_training_library_claim_detected": "ARC-AGI-2 public training set of 1,000 tasks" in readme_text
        and "538 programs" in readme_text,
        "root_license_paths": root_license_paths,
        "vendored_license_paths": vendored_license_paths,
    }


ANALYZERS = {
    "arc-lang-public": analyze_arc_lang,
    "arcmemo": analyze_arcmemo,
    "epang-arc-agi": analyze_epang,
}


def secure_read_pinned_relative(
    root_fd: int,
    path: str,
    *,
    role: str,
    category: str,
    max_bytes: int,
    ledger: ReadLedger,
) -> tuple[bytes, tuple[int, ...]]:
    safe_relative_path(path, "pinned relative path")
    record = ledger.begin(path, role, category)
    descriptor: int | None = None
    parent_fd: int | None = None
    try:
        parent_fd, leaf = open_relative_parent(root_fd, path)
        before_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise ValueError(f"pinned input is not a single-link regular file: {path}")
        descriptor = os.open(leaf, regular_file_flags(), dir_fd=parent_fd)
        before_fd = os.fstat(descriptor)
        if stat_signature(before_path) != stat_signature(before_fd):
            raise RuntimeError(f"pinned relative input raced before read: {path}")
        payload = _read_stable_fd(descriptor, before_fd, max_bytes, record)
        after_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat_signature(os.fstat(descriptor)) != stat_signature(after_path):
            raise RuntimeError(f"pinned relative input raced after read: {path}")
        return payload, stat_signature(after_path)
    except BaseException as error:
        record["status"] = "failed"
        record["failure_type"] = type(error).__name__
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def validate_hash(value: Any, field: str, pattern: re.Pattern[str] = SHA256_RE) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} is not a valid lowercase digest")
    return value


def validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Batch C gate config must be an object")
    if EXPECTED_CONFIG_CANONICAL_SHA256 == "TO_BE_PINNED":
        raise ValueError("auditor config digest has not been pinned")
    if canonical_sha256(value) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise ValueError("Batch C gate config differs from the hardcoded v1 contract")
    if value.get("schema_version") != 1 or value.get("config_id") != CONFIG_ID:
        raise ValueError("Batch C gate config identity mismatch")
    if value.get("scope") != SCOPE or value.get("counted_toward_smoke") is not False:
        raise ValueError("Batch C gate config claim boundary mismatch")
    policy = value.get("config_read_policy")
    if policy != {
        "canonical_path": CANONICAL_CONFIG_RELATIVE,
        "alternate_paths_allowed": False,
    }:
        raise ValueError("Batch C canonical config policy mismatch")
    source_lock = value.get("source_lock")
    if not isinstance(source_lock, dict) or source_lock.get("path") != SOURCE_LOCK_RELATIVE:
        raise ValueError("Batch C source-lock declaration mismatch")
    if source_lock.get("sha256") != EXPECTED_SOURCE_LOCK_SHA256:
        raise ValueError("Batch C source-lock digest declaration mismatch")
    methods = value.get("methods")
    if not isinstance(methods, list) or [item.get("method_id") for item in methods] != list(EXPECTED_METHOD_IDS):
        raise ValueError("Batch C method order/identity mismatch")
    for method in methods:
        validate_method_config(method)
    controls = value.get("controls")
    expected_controls = {
        "network_allowed": False,
        "gpu_allowed": False,
        "upstream_import_allowed": False,
        "upstream_execution_allowed": False,
        "arc_or_solution_byte_read_allowed": False,
        "pickle_byte_read_allowed": False,
        "generated_code_execution_allowed": False,
        "provider_call_allowed": False,
        "prediction_allowed": False,
    }
    if controls != expected_controls:
        raise ValueError("Batch C control policy mismatch")
    return value


def validate_method_config(method: Any) -> None:
    if not isinstance(method, dict) or method.get("method_id") not in EXPECTED_METHOD_IDS:
        raise ValueError("invalid Batch C method declaration")
    source = method.get("source")
    if not isinstance(source, dict):
        raise ValueError("Batch C source declaration must be an object")
    repository_path = Path(source.get("repository_path", ""))
    if not repository_path.is_absolute() or not str(repository_path).startswith("/root/arc-paper-assets/sources/"):
        raise ValueError("Batch C source root is outside its exact asset namespace")
    validate_hash(source.get("expected_revision"), "source revision", GIT_SHA_RE)
    validate_hash(source.get("expected_commit_tree"), "source tree", GIT_SHA_RE)
    for field in (
        "git_tree_manifest_sha256",
        "worktree_metadata_sha256",
        "retained_manifest_sha256",
    ):
        validate_hash(source.get(field), field)
    for field in ("expected_tracked_file_count", "expected_tracked_bytes", "expected_retained_bytes"):
        if not isinstance(source.get(field), int) or source[field] < 0:
            raise ValueError(f"invalid nonnegative source count: {field}")
    retained = source.get("retained_text")
    if not isinstance(retained, list) or not retained:
        raise ValueError("Batch C retained source policy must be non-empty")
    seen: set[str] = set()
    for item in retained:
        if not isinstance(item, dict) or set(item) != {"path", "role"}:
            raise ValueError("retained declarations contain unexpected keys")
        path = safe_relative_path(item["path"], "retained path")
        if path in seen or not isinstance(item["role"], str) or not item["role"]:
            raise ValueError("duplicate or role-less retained declaration")
        if PurePosixPath(path).suffix.lower() in FORBIDDEN_RETAINED_SUFFIXES:
            raise ValueError(f"restricted content was declared retained: {path}")
        seen.add(path)
    if source.get("expected_retained_file_count") != len(retained):
        raise ValueError("retained count declaration mismatch")
    opaque = source.get("opaque_worktree_paths", [])
    if not isinstance(opaque, list) or len(set(opaque)) != len(opaque):
        raise ValueError("invalid opaque worktree paths")
    for path in opaque:
        safe_relative_path(path, "opaque worktree path")
        if path == ".git" or path in seen:
            raise ValueError("invalid opaque worktree path overlap")
    git_contract = source.get("git_metadata_contract")
    if not isinstance(git_contract, dict):
        raise ValueError("missing Git metadata contract")
    for leaf in ("local_config", "head"):
        declaration = git_contract.get(leaf)
        if not isinstance(declaration, dict):
            raise ValueError(f"missing Git {leaf} contract")
        safe_relative_path(declaration.get("path"), f"Git {leaf} path")
        if not isinstance(declaration.get("bytes"), int) or declaration["bytes"] <= 0:
            raise ValueError(f"invalid Git {leaf} bytes")
        validate_hash(declaration.get("sha256"), f"Git {leaf} SHA-256")
        if declaration.get("mode") != "0644":
            raise ValueError(f"unsupported Git {leaf} mode")
    blockers = method.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ValueError("Batch C blocker list must be non-empty")
    blocker_ids = [item.get("id") for item in blockers if isinstance(item, dict)]
    if len(blocker_ids) != len(blockers) or len(blocker_ids) != len(set(blocker_ids)):
        raise ValueError("Batch C blocker IDs must be unique")
    if not all(set(item) == {"id", "gate", "detail"} for item in blockers):
        raise ValueError("Batch C blocker declaration keys mismatch")
    if not isinstance(method.get("expected_analysis"), dict) or not method["expected_analysis"]:
        raise ValueError("Batch C expected analysis must be non-empty")
    prior_reports = method.get("prior_reports")
    if not isinstance(prior_reports, list) or not prior_reports:
        raise ValueError("Batch C prior-report bindings must be non-empty")
    for report in prior_reports:
        if not isinstance(report, dict) or set(report) != {"path", "sha256", "assertions", "role"}:
            raise ValueError("prior-report binding keys mismatch")
        safe_relative_path(report["path"], "prior report path")
        validate_hash(report["sha256"], "prior report SHA-256")
        if not isinstance(report["assertions"], dict) or not isinstance(report["role"], str):
            raise ValueError("invalid prior-report binding")


def assert_subset(observed: Any, expected: Any, field: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f"{field} expected an object")
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(f"{field} missing asserted key: {key}")
            assert_subset(observed[key], value, f"{field}.{key}")
    elif observed != expected:
        raise ValueError(f"{field} assertion mismatch: expected {expected!r}, got {observed!r}")


def source_lock_entry(value: dict[str, Any], method_id: str) -> dict[str, Any]:
    entries = value.get("sources")
    if not isinstance(entries, dict) or method_id not in entries:
        raise ValueError(f"source lock missing Batch C method: {method_id}")
    entry = entries[method_id]
    if not isinstance(entry, dict):
        raise ValueError("source lock entry must be an object")
    return entry


def verify_exact_file_contract(
    payload: bytes, info: tuple[int, ...], declaration: dict[str, Any], field: str
) -> None:
    mode = format(stat.S_IMODE(info[2]), "04o")
    if len(payload) != declaration["bytes"] or hashlib.sha256(payload).hexdigest() != declaration["sha256"]:
        raise ValueError(f"{field} byte/SHA-256 contract mismatch")
    if mode != declaration["mode"]:
        raise ValueError(f"{field} mode contract mismatch")


def retained_manifest(
    declarations: list[dict[str, str]],
    payloads: dict[str, bytes],
    tree: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for declaration in sorted(declarations, key=lambda item: item["path"]):
        path = declaration["path"]
        payload = payloads[path]
        if path not in tree:
            raise ValueError(f"retained path absent from Git tree: {path}")
        oid = git_blob_oid(payload)
        if oid != tree[path]["blob_oid"]:
            raise ValueError(f"retained bytes differ from locked Git blob: {path}")
        result.append(
            {
                "path": path,
                "role": declaration["role"],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "blob_oid": oid,
            }
        )
    return result


def method_config(config: dict[str, Any], method_id: str) -> dict[str, Any]:
    matches = [item for item in config["methods"] if item["method_id"] == method_id]
    if len(matches) != 1:
        raise ValueError(f"method is not uniquely declared: {method_id}")
    return matches[0]


def run_static_audit(
    config: dict[str, Any],
    method: dict[str, Any],
    config_payload: bytes,
    source_lock_payload: bytes,
    ledger: ReadLedger,
    run_id: str,
) -> dict[str, Any]:
    method_id = method["method_id"]
    source = method["source"]
    source_path = Path(source["repository_path"])
    source_lock = strict_json(source_lock_payload, "source lock")
    lock_entry = source_lock_entry(source_lock, method_id)
    if lock_entry != method["source_lock_entry"]:
        raise ValueError(f"source-lock entry differs from the gate contract: {method_id}")
    if lock_entry.get("revision") != source["expected_revision"]:
        raise ValueError("source-lock revision differs from source contract")

    prior_observations: list[dict[str, Any]] = []
    for declaration in method["prior_reports"]:
        report_path = ROOT / declaration["path"]
        payload = secure_read_absolute(
            report_path,
            role=declaration["role"],
            category="prior_report",
            max_bytes=MAX_PRIOR_REPORT_BYTES,
            ledger=ledger,
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != declaration["sha256"]:
            raise ValueError(f"prior report SHA-256 mismatch: {declaration['path']}")
        report = strict_json(payload, declaration["path"])
        assert_subset(report, declaration["assertions"], declaration["path"])
        prior_observations.append(
            {
                "path": declaration["path"],
                "role": declaration["role"],
                "sha256": digest,
                "assertions_passed": True,
            }
        )

    root_fd = open_absolute_directory(source_path)
    git_fd: int | None = None
    object_directory_fd: int | None = None
    isolated_git_fd: int | None = None
    isolated_git_temp: tempfile.TemporaryDirectory[str] | None = None
    git_directory_signature: tuple[int, ...] | None = None
    try:
        verify_directory_identity(source_path, root_fd)
        git_path_info = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(git_path_info.st_mode):
            raise ValueError("source .git is not a non-symlink directory")
        git_fd = os.open(".git", directory_flags(), dir_fd=root_fd)
        git_directory_signature = stat_signature(git_path_info)
        if git_directory_signature != stat_signature(os.fstat(git_fd)):
            raise RuntimeError("source .git raced before pinning")
        for path in FORBIDDEN_GIT_PATHS:
            require_git_path_absent(git_fd, path)
        git_contract = source["git_metadata_contract"]
        git_config_payload, git_config_signature = secure_read_pinned_relative(
            git_fd,
            "config",
            role="git_local_config",
            category="git_metadata",
            max_bytes=MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        git_head_payload, git_head_signature = secure_read_pinned_relative(
            git_fd,
            "HEAD",
            role="git_head",
            category="git_metadata",
            max_bytes=MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        verify_exact_file_contract(
            git_config_payload,
            git_config_signature,
            git_contract["local_config"],
            "Git local config",
        )
        verify_exact_file_contract(
            git_head_payload,
            git_head_signature,
            git_contract["head"],
            "Git HEAD",
        )
        validate_git_config_entries(
            git_config_payload,
            source["repository_url"],
            method["source_lock_entry"]["branch"],
        )
        allowed_heads = {
            f"{source['expected_revision']}\n".encode("ascii"),
            f"ref: refs/heads/{method['source_lock_entry']['branch']}\n".encode("ascii"),
        }
        if git_head_payload not in allowed_heads:
            raise ValueError("Git HEAD is neither detached at the pin nor on the pinned branch")
        for forbidden_token in (b"command =", b"insteadOf", b"include.path", b"fsmonitor", b"sshCommand"):
            if forbidden_token.lower() in git_config_payload.lower():
                raise ValueError("Git local config contains a forbidden execution/redirection key")

        object_path_info = os.stat(
            "objects", dir_fd=git_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(object_path_info.st_mode):
            raise ValueError("source Git object store is not a directory")
        object_directory_fd = os.open("objects", directory_flags(), dir_fd=git_fd)
        object_directory_signature = stat_signature(object_path_info)
        if object_directory_signature != stat_signature(
            os.fstat(object_directory_fd)
        ):
            raise RuntimeError("source Git object directory raced before pinning")
        isolated_git_temp = tempfile.TemporaryDirectory(
            prefix="arc-agi-eval-batch-c-git-", dir="/tmp"
        )
        isolated_git_fd = initialize_isolated_git_directory(
            Path(isolated_git_temp.name)
        )

        revision = source["expected_revision"]
        allowed_git = {
            ("rev-parse", "--verify", revision),
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            ("ls-tree", "-rz", "--full-tree", revision),
        }
        objects_before = git_object_metadata(git_fd)
        observed_revision = run_git(
            isolated_git_fd,
            object_directory_fd,
            ("rev-parse", "--verify", revision),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        observed_tree = run_git(
            isolated_git_fd,
            object_directory_fd,
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        listing_payload = run_git(
            isolated_git_fd,
            object_directory_fd,
            ("ls-tree", "-rz", "--full-tree", revision),
            allowed_git,
            ledger,
        )
        if observed_revision != revision or observed_tree != source["expected_commit_tree"]:
            raise ValueError("locked Git revision/tree mismatch")
        tree = parse_git_tree(listing_payload)
        tree_manifest = git_tree_manifest(tree)
        if canonical_sha256(tree_manifest) != source["git_tree_manifest_sha256"]:
            raise ValueError("Git tree manifest digest mismatch")
        if len(tree) != source["expected_tracked_file_count"]:
            raise ValueError("tracked file count mismatch")
        if extension_counts(set(tree)) != source["expected_extension_counts"]:
            raise ValueError("tracked extension counts mismatch")

        worktree, initial_signatures, opaque_observed = walk_worktree_metadata(
            root_fd, tree, set(source.get("opaque_worktree_paths", []))
        )
        worktree_manifest = [worktree[path] for path in sorted(worktree)]
        if canonical_sha256(worktree_manifest) != source["worktree_metadata_sha256"]:
            raise ValueError("worktree metadata manifest digest mismatch")
        if sum(item["bytes"] for item in worktree_manifest) != source["expected_tracked_bytes"]:
            raise ValueError("tracked worktree byte total mismatch")
        for path, item in tree.items():
            expected_mode = "0755" if item["mode"] == "100755" else "0644"
            if worktree[path]["mode"] != expected_mode:
                raise ValueError(f"worktree mode differs from Git mode: {path}")

        retained_policy = {item["path"]: item["role"] for item in source["retained_text"]}
        payloads_first: dict[str, bytes] = {}
        for path, role in retained_policy.items():
            payloads_first[path] = secure_read_retained(
                root_fd, path, role, initial_signatures[path], ledger
            )
        manifest_first = retained_manifest(source["retained_text"], payloads_first, tree)
        if len(manifest_first) != source["expected_retained_file_count"]:
            raise ValueError("retained manifest count mismatch")
        if sum(item["bytes"] for item in manifest_first) != source["expected_retained_bytes"]:
            raise ValueError("retained manifest byte total mismatch")
        if canonical_sha256(manifest_first) != source["retained_manifest_sha256"]:
            raise ValueError("retained manifest digest mismatch")

        parsed_trees = {
            path: parse_python(payload, path)
            for path, payload in payloads_first.items()
            if PurePosixPath(path).suffix.lower() == ".py"
        }
        analysis = ANALYZERS[method_id](parsed_trees, payloads_first, set(tree))
        if analysis != method["expected_analysis"]:
            raise ValueError(
                "method static analysis differs from pinned expectation: "
                + json.dumps(analysis, sort_keys=True)
            )

        root_license_paths = sorted(
            path for path in tree if "/" not in path and PurePosixPath(path).name.lower() in ROOT_LICENSE_NAMES
        )
        if root_license_paths != source["expected_root_license_paths"]:
            raise ValueError("root license path observation mismatch")

        terminal_worktree, terminal_signatures, terminal_opaque = walk_worktree_metadata(
            root_fd, tree, set(source.get("opaque_worktree_paths", []))
        )
        if terminal_worktree != worktree or terminal_signatures != initial_signatures or terminal_opaque != opaque_observed:
            raise RuntimeError("source worktree metadata changed during static audit")
        payloads_terminal: dict[str, bytes] = {}
        for path, role in retained_policy.items():
            payloads_terminal[path] = secure_read_retained(
                root_fd, path, role, terminal_signatures[path], ledger
            )
        manifest_terminal = retained_manifest(
            source["retained_text"], payloads_terminal, tree
        )
        if manifest_terminal != manifest_first or payloads_terminal != payloads_first:
            raise RuntimeError("retained source bytes changed during static audit")

        terminal_config, terminal_config_signature = secure_read_pinned_relative(
            git_fd,
            "config",
            role="git_local_config_terminal",
            category="git_metadata",
            max_bytes=MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        terminal_head, terminal_head_signature = secure_read_pinned_relative(
            git_fd,
            "HEAD",
            role="git_head_terminal",
            category="git_metadata",
            max_bytes=MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        if (
            terminal_config != git_config_payload
            or terminal_head != git_head_payload
            or terminal_config_signature != git_config_signature
            or terminal_head_signature != git_head_signature
        ):
            raise RuntimeError("Git metadata files changed during static audit")
        terminal_revision = run_git(
            isolated_git_fd,
            object_directory_fd,
            ("rev-parse", "--verify", revision),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        terminal_tree = run_git(
            isolated_git_fd,
            object_directory_fd,
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        terminal_listing = run_git(
            isolated_git_fd,
            object_directory_fd,
            ("ls-tree", "-rz", "--full-tree", revision),
            allowed_git,
            ledger,
        )
        if (
            terminal_revision != observed_revision
            or terminal_tree != observed_tree
            or terminal_listing != listing_payload
            or git_object_metadata(git_fd) != objects_before
        ):
            raise RuntimeError("Git object/tree state changed during static audit")
        terminal_git_path = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        terminal_object_path = os.stat(
            "objects", dir_fd=git_fd, follow_symlinks=False
        )
        if (
            git_directory_signature != stat_signature(terminal_git_path)
            or git_directory_signature != stat_signature(os.fstat(git_fd))
        ):
            raise RuntimeError("source .git directory identity changed during audit")
        if (
            object_directory_signature != stat_signature(terminal_object_path)
            or object_directory_signature
            != stat_signature(os.fstat(object_directory_fd))
        ):
            raise RuntimeError("source Git object directory identity changed during audit")
        if set(os.listdir(isolated_git_fd)) != {"HEAD", "objects", "refs"}:
            raise RuntimeError("isolated Git directory gained an unexpected entry")
        for directory in ("objects", "refs"):
            child = os.open(directory, directory_flags(), dir_fd=isolated_git_fd)
            try:
                if os.listdir(child):
                    raise RuntimeError(
                        f"isolated Git {directory} directory was unexpectedly modified"
                    )
            finally:
                os.close(child)
        verify_directory_identity(source_path, root_fd)
    finally:
        if isolated_git_fd is not None:
            os.close(isolated_git_fd)
        if isolated_git_temp is not None:
            isolated_git_temp.cleanup()
        if object_directory_fd is not None:
            os.close(object_directory_fd)
        if git_fd is not None:
            os.close(git_fd)
        os.close(root_fd)

    blockers = [
        {**item, "status": "blocked"} for item in method["blockers"]
    ]
    controls = {
        "network_used": False,
        "gpu_used": False,
        "upstream_imported": False,
        "upstream_executed": False,
        "provider_called": False,
        "generated_code_executed": False,
        "solver_prediction_produced": False,
        "auditor_process_arc_or_solution_worktree_leaf_bytes_read": 0,
        "auditor_process_pickle_worktree_leaf_bytes_read": 0,
        "metadata_only_worktree_leaf_content_reads": 0,
        "retained_source_read_passes": 2,
        "retained_source_read_attempts": 2 * len(source["retained_text"]),
        "git_subprocesses_started": len(ledger.git_processes),
        "git_subprocess_worktree_content_requested": False,
        "git_subprocess_object_database_reads_possible": True,
        "git_subprocess_object_database_bytes_measured": False,
        "git_local_config_prevalidated": True,
        "git_subprocess_source_local_config_available": False,
        "git_subprocess_isolated_git_directory_used": True,
        "git_object_leaf_hardlink_aliases_allowed": False,
        "git_system_and_global_config_disabled": True,
        "git_lazy_fetch_disabled": True,
    }
    source_observation = {
        "repository_path": str(source_path),
        "expected_revision": source["expected_revision"],
        "observed_revision": observed_revision,
        "observed_commit_tree": observed_tree,
        "tracked_file_count": len(tree),
        "tracked_worktree_bytes": sum(item["bytes"] for item in worktree_manifest),
        "git_tree_manifest_sha256": canonical_sha256(tree_manifest),
        "worktree_metadata_sha256": canonical_sha256(worktree_manifest),
        "extension_counts": extension_counts(set(tree)),
        "retained_byte_exact_file_count": len(manifest_first),
        "retained_byte_exact_bytes": sum(item["bytes"] for item in manifest_first),
        "retained_manifest_sha256": canonical_sha256(manifest_first),
        "metadata_only_tracked_file_count": len(tree) - len(manifest_first),
        "metadata_only_tracked_bytes": sum(item["bytes"] for item in worktree_manifest)
        - sum(item["bytes"] for item in manifest_first),
        "opaque_worktree_paths": opaque_observed,
        "root_license_paths": root_license_paths,
        "working_tree_all_files_byte_exact_verified": False,
    }
    observation = {
        "method_id": method_id,
        "scope": SCOPE,
        "source": source_observation,
        "static_analysis": analysis,
        "prior_reports": prior_observations,
        "blockers": blockers,
        "controls": controls,
        "benchmark_policy": method["benchmark_policy"],
        "prior_evidence_interpretation": method["prior_evidence_interpretation"],
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "method_id": method_id,
        "run_id": run_id,
        "runner": "scripts.audit_batch_c_static_gates",
        "status": "passed",
        "method_gate_status": "blocked",
        "scope": SCOPE,
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "solver_gate_passed": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "config": {
            "path": CANONICAL_CONFIG_RELATIVE,
            "file_sha256": hashlib.sha256(config_payload).hexdigest(),
            "canonical_sha256": canonical_sha256(config),
        },
        **observation,
        "observation_digest_sha256": canonical_sha256(observation),
        "read_ledger": ledger.snapshot(),
        "validation": {
            "canonical_config_bound": True,
            "source_lock_bound": True,
            "revision_and_tree_bound": True,
            "isolated_git_metadata_view_bound": True,
            "closed_worktree_metadata_bound": True,
            "retained_bytes_bound_twice": True,
            "restricted_content_not_opened": True,
            "static_analysis_matches": True,
            "prior_reports_bound": True,
            "terminal_state_unchanged": True,
            "all_method_gates_remain_blocked": True,
        },
        "claim_boundary": (
            f"This metadata-first audit makes byte-exact claims only for the {len(manifest_first)} retained text leaves. "
            f"It makes path/mode/OID/size metadata claims, not worktree-content claims, for the remaining {len(tree) - len(manifest_first)} tracked leaves. "
            "Git subprocess object-database byte volume is unmeasured. Passing reproduces blockers; it is not a solver smoke, prediction, score, or benchmark."
        ),
        "limitations": [
            "Metadata-only tracked leaves were not opened by the auditor and their worktree content was not byte-verified.",
            "Git metadata subprocesses may read pinned local object files; kernel-level object byte volume is not measured.",
            "Static AST/source signatures characterize the exact pinned tree and are not a general proof for modified upstream code.",
            "Opaque ignored directories, when declared, are not traversed and make no content claim.",
        ],
    }
    return record


class FreshOutput:
    def __init__(
        self,
        descriptor: int,
        parent_descriptor: int,
        parent_path: Path,
        leaf: str,
        identity: tuple[int, int, int],
    ) -> None:
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.parent_path = parent_path
        self.leaf = leaf
        self.identity = identity

    def verify(self) -> None:
        verify_directory_identity(self.parent_path, self.parent_descriptor)
        by_fd = os.fstat(self.descriptor)
        by_path = os.stat(self.leaf, dir_fd=self.parent_descriptor, follow_symlinks=False)
        if stat_signature(by_fd) != stat_signature(by_path):
            raise OutputPathError("fresh output leaf identity changed")
        if (by_fd.st_dev, by_fd.st_ino, by_fd.st_mode) != self.identity:
            raise OutputPathError("fresh output descriptor metadata changed")

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_descriptor)


def validate_output_location(path: Path, method_id: str) -> None:
    parts = _absolute_parts(path)
    if len(parts) < 2 or parts[-1] in {"", ".", ".."} or parts[-1].startswith("."):
        raise OutputPathError("output must use a visible fresh leaf name")
    absolute = Path(*parts)
    production_parent = ROOT / "reports" / method_id
    if Path(*parts[:-1]) == Path(*_absolute_parts(production_parent)):
        return
    if lexical_within(absolute, TEST_OUTPUT_ROOT) and not lexical_equal(
        absolute, TEST_OUTPUT_ROOT
    ):
        return
    raise OutputPathError(
        "output must be one fresh leaf under the method report directory or the dedicated Batch C test root"
    )


def create_fresh_output(path: Path) -> FreshOutput:
    parent_fd, leaf = open_absolute_parent(path)
    if leaf in {"", ".", ".."}:
        os.close(parent_fd)
        raise OutputPathError("unsafe output leaf")
    try:
        try:
            os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise OutputPathError("output path must not exist") from error
        descriptor = os.open(leaf, directory_flags(), dir_fd=parent_fd)
        info = os.fstat(descriptor)
        return FreshOutput(
            descriptor,
            parent_fd,
            Path(*_absolute_parts(path)[:-1]),
            leaf,
            (info.st_dev, info.st_ino, info.st_mode),
        )
    except BaseException:
        os.close(parent_fd)
        raise


def rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(directory_fd, os.fsencode(source), directory_fd, os.fsencode(destination), RENAME_NOREPLACE) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def write_json_no_clobber(output: FreshOutput, value: dict[str, Any]) -> None:
    output.verify()
    temporary = f".run.json.tmp-{os.getpid()}-{time.time_ns()}"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(temporary, flags, 0o600, dir_fd=output.descriptor)
    owned_identity: tuple[int, int, int, int] | None = None

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size)

    def descriptor_payload_matches(payload: bytes) -> bool:
        if owned_identity is None:
            return False
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or identity(before) != owned_identity
            ):
                return False
            readback = bytearray()
            while len(readback) < len(payload) + 1:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, len(payload) + 1 - len(readback)),
                )
                if not chunk:
                    break
                readback.extend(chunk)
            after = os.fstat(descriptor)
        except (OSError, RuntimeError, ValueError):
            return False
        return (
            after.st_nlink == 1
            and identity(after) == owned_identity
            and stat_signature(after) == stat_signature(before)
            and bytes(readback) == payload
        )

    def final_path_matches() -> bool:
        if owned_identity is None:
            return False
        try:
            observed = os.stat(
                "run.json", dir_fd=output.descriptor, follow_symlinks=False
            )
        except OSError:
            return False
        return (
            stat.S_ISREG(observed.st_mode)
            and observed.st_nlink == 1
            and identity(observed) == owned_identity
        )

    def directory_entries_match() -> bool:
        try:
            return os.listdir(output.descriptor) == ["run.json"]
        except OSError:
            return False

    try:
        payload = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode("utf-8") + b"\n"
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise RuntimeError("temporary report write made no progress")
            written += count
        os.fsync(descriptor)
        before_fd = os.fstat(descriptor)
        before_path = os.stat(
            temporary, dir_fd=output.descriptor, follow_symlinks=False
        )
        if (
            before_fd.st_nlink != 1
            or stat_signature(before_fd) != stat_signature(before_path)
        ):
            raise RuntimeError("temporary report leaf changed before publication")
        owned_identity = identity(before_fd)
        if not descriptor_payload_matches(payload):
            raise RuntimeError("temporary report descriptor payload is unstable")
        output.verify()
        rename_noreplace(output.descriptor, temporary, "run.json")
        if (
            not final_path_matches()
            or not descriptor_payload_matches(payload)
            or not directory_entries_match()
        ):
            raise RuntimeError("published report leaf differs from the pinned temporary inode")
        os.fsync(descriptor)
        os.fsync(output.descriptor)
        os.fsync(output.parent_descriptor)
        output.verify()
        if (
            not final_path_matches()
            or not descriptor_payload_matches(payload)
            or not directory_entries_match()
        ):
            raise RuntimeError("report identity or payload changed after commit sync")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # Never unlink a failed temporary path here. Another same-directory
        # writer can replace that path between any identity check and unlink;
        # retaining the leaf is fail-closed and preserves forensic evidence.


def failure_record(method_id: str, run_id: str, error: BaseException, ledger: ReadLedger) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "method_id": method_id,
        "run_id": run_id,
        "runner": "scripts.audit_batch_c_static_gates",
        "status": "failed",
        "method_gate_status": "blocked",
        "scope": SCOPE,
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "performance_table_eligible": False,
        "failure": {"type": type(error).__name__, "message": str(error)},
        "read_ledger": ledger.snapshot(),
        "controls": {
            "network_used": False,
            "gpu_used": False,
            "upstream_imported": False,
            "upstream_executed": False,
            "provider_called": False,
            "generated_code_executed": False,
            "solver_prediction_produced": False,
        },
        "claim_boundary": "The static audit failed; no solver or prediction was run.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=EXPECTED_METHOD_IDS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    started_at = utc_now()
    started_monotonic = time.monotonic()
    started_usage = usage_snapshot()

    canonical_config = ROOT / CANONICAL_CONFIG_RELATIVE
    if not lexical_equal(args.config, canonical_config):
        parser.error(f"production config path must equal {CANONICAL_CONFIG_RELATIVE}")
    output_path = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
    try:
        validate_output_location(output_path, args.method_id)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if lexical_equal(output_path, canonical_config) or lexical_equal(output_path, ROOT / SOURCE_LOCK_RELATIVE):
        parser.error("output path overlaps a protected input")

    bootstrap_ledger = ReadLedger({})
    try:
        config_payload = secure_read_absolute(
            canonical_config,
            role="canonical_config",
            category="gate_config",
            max_bytes=MAX_CONFIG_BYTES,
            ledger=bootstrap_ledger,
        )
        auditor_payload = secure_read_absolute(
            AUDITOR_PATH,
            role="auditor_source",
            category="auditor_source",
            max_bytes=4 * 1024 * 1024,
            ledger=bootstrap_ledger,
        )
        config = validate_config(strict_json(config_payload, "Batch C gate config"))
        method = method_config(config, args.method_id)
        retained_policy = {
            item["path"]: item["role"] for item in method["source"]["retained_text"]
        }
        ledger = ReadLedger(retained_policy)
        ledger.file_reads.extend(bootstrap_ledger.file_reads)
        source_lock_payload = secure_read_absolute(
            ROOT / SOURCE_LOCK_RELATIVE,
            role="source_lock",
            category="source_lock",
            max_bytes=MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        if hashlib.sha256(source_lock_payload).hexdigest() != EXPECTED_SOURCE_LOCK_SHA256:
            raise ValueError("source-lock file SHA-256 mismatch")
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    try:
        output = create_fresh_output(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    try:
        try:
            record = run_static_audit(
                config,
                method,
                config_payload,
                source_lock_payload,
                ledger,
                output_path.name,
            )
            terminal_config_payload = secure_read_absolute(
                canonical_config,
                role="canonical_config_terminal",
                category="gate_config",
                max_bytes=MAX_CONFIG_BYTES,
                ledger=ledger,
            )
            terminal_source_lock_payload = secure_read_absolute(
                ROOT / SOURCE_LOCK_RELATIVE,
                role="source_lock_terminal",
                category="source_lock",
                max_bytes=MAX_CONFIG_BYTES,
                ledger=ledger,
            )
            terminal_auditor_payload = secure_read_absolute(
                AUDITOR_PATH,
                role="auditor_source_terminal",
                category="auditor_source",
                max_bytes=4 * 1024 * 1024,
                ledger=ledger,
            )
            if (
                terminal_config_payload != config_payload
                or terminal_source_lock_payload != source_lock_payload
                or terminal_auditor_payload != auditor_payload
            ):
                raise RuntimeError(
                    "canonical config, source lock, or auditor source changed during audit"
                )
            record["runner_provenance"] = {
                "path": "scripts/audit_batch_c_static_gates.py",
                "bytes": len(auditor_payload),
                "sha256": hashlib.sha256(auditor_payload).hexdigest(),
                "terminal_bytes_equal": True,
            }
            returncode = 0
        except BaseException as error:
            record = failure_record(args.method_id, output_path.name, error, ledger)
            returncode = 1
        record["read_ledger"] = ledger.snapshot()
        record["started_at_utc"] = started_at
        record["ended_at_utc"] = utc_now()
        record["resources"] = usage_record(
            started_usage,
            usage_snapshot(),
            time.monotonic() - started_monotonic,
        )
        write_json_no_clobber(output, record)
        print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
        return returncode
    finally:
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())
