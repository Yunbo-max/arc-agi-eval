#!/usr/bin/env python3
"""Hardened, source-only gate audit for the ARChitects 2024 method.

This program deliberately does not import or execute upstream Python, open ARC
datasets, inspect checkpoint files, deserialize caches, contact a network, or
initialize a GPU.  It binds a locked source tree and two already-existing,
small JSON reports, then performs syntax/AST analysis over an explicit source
allowlist.  Its output is blocker evidence, never a solver smoke or score.
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
import secrets
import stat
import sys
import time
from typing import Any, Callable, Iterable


ROOT = Path(__file__).absolute().parents[1]
CANONICAL_CONFIG = ROOT / "configs" / "architects_2024_gate_v1.json"
SOURCE_LOCK = ROOT / "configs" / "source_locks.json"
CHECKPOINT_REPORT = (
    ROOT
    / "reports"
    / "architects-2024"
    / "20260806-4bit-checkpoint-integrity"
    / "run.json"
)
PREFLIGHT_REPORT = (
    ROOT
    / "reports"
    / "architects-2024"
    / "20260806-forward-preflight-gpu-occupied"
    / "run.json"
)
EXPECTED_SOURCE_ROOT = Path("/root/arc-paper-assets/sources/architects-2024")
EXPECTED_REVISION = "d3ac3f6ebf6fb609bfdc782561ee99977ca35d95"
EXPECTED_TREE = "46eedc2d304d60bcb3a393063528d2a694223199"
EXPECTED_BLOCKERS = (
    "model-license-review",
    "arc1-training-contamination",
    "local-runner-label-firewall",
    "safe-offline-model-load",
    "pickle-cache-isolation",
    "dependency-environment-parity",
    "resource-capacity",
    "solver-prediction-and-parity",
)

ABSOLUTE_ROLES: dict[str, tuple[Path, str]] = {
    "canonical_config": (CANONICAL_CONFIG, "gate_config"),
    "source_lock": (SOURCE_LOCK, "source_lock"),
    "checkpoint_integrity_report": (CHECKPOINT_REPORT, "prior_report"),
    "resource_preflight_report": (PREFLIGHT_REPORT, "prior_report"),
}

ABSOLUTE_MAX_BYTES = {
    "canonical_config": 131072,
    "source_lock": 131072,
    "checkpoint_integrity_report": 16384,
    "resource_preflight_report": 16384,
}

SOURCE_ROLES: dict[str, str] = {
    "LICENSE.txt": "code_license",
    "NOTICE": "source_notice",
    "README.md": "source_readme",
    "kaggle_notebooks/arc-prize-2024_kaggle.ipynb": "official_notebook",
    "kaggle_notebooks/arc-prize-2024_updated.ipynb": "updated_notebook",
    "training_code/arc_downloader.py": "python_source",
    "training_code/arc_loader.py": "python_source",
    "training_code/inference_tools.py": "python_source",
    "training_code/model_tools.py": "model_loader_source",
    "training_code/run_evaluation_Llama-rearc_with_ttt.py": "label_bearing_runner",
    "training_code/run_evaluation_Llama-rearc_without_ttt.py": "label_bearing_runner",
    "training_code/run_finetuning_Llama-rearc.py": "python_source",
    "training_code/run_finetuning_Nemo-full.py": "contamination_source",
    "training_code/selection.py": "python_source",
}

SOURCE_CATEGORY_BY_ROLE = {
    "code_license": "source_text",
    "source_notice": "source_text",
    "source_readme": "source_text",
    "official_notebook": "source_notebook",
    "updated_notebook": "source_notebook",
    "python_source": "source_python",
    "model_loader_source": "source_python",
    "label_bearing_runner": "source_python",
    "contamination_source": "source_python",
}

FORBIDDEN_DATA_SUFFIXES = (
    "_solutions.json",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".bz2",
)

FORBIDDEN_GIT_AUXILIARY_PATHS = (
    ".git/commondir",
    ".git/config.worktree",
    ".git/info/attributes",
    ".git/objects/info/alternates",
    ".git/objects/info/http-alternates",
)


class OutputPathError(ValueError):
    """The requested output is not a fresh, safely pinned directory."""


class FreshOutput:
    def __init__(
        self,
        descriptor: int,
        parent_descriptor: int,
        parent_path: Path,
        leaf: str,
        created_identity: tuple[int, int, int],
    ) -> None:
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.parent_path = parent_path
        self.leaf = leaf
        self.created_identity = created_identity

    def verify_leaf(self) -> None:
        verify_directory_path_identity(self.parent_path, self.parent_descriptor)
        fd_info = os.fstat(self.descriptor)
        leaf_info = os.stat(
            self.leaf,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(fd_info.st_mode) or not stat.S_ISDIR(leaf_info.st_mode):
            raise OutputPathError("fresh output leaf is no longer a directory")
        if stat_signature(leaf_info) != stat_signature(fd_info):
            raise OutputPathError("fresh output leaf was replaced after creation")
        if (
            int(fd_info.st_dev),
            int(fd_info.st_ino),
            int(fd_info.st_mode),
        ) != self.created_identity:
            raise OutputPathError("fresh output descriptor identity changed")

    def close(self, *, record_committed: bool) -> None:
        errors: list[OSError] = []
        for descriptor in (self.descriptor, self.parent_descriptor):
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        if errors and not record_committed:
            raise errors[0]


class ReadLedger:
    """Closed role/path authorization and an auditable byte-read ledger."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.git_query_count = 0
        self.git_worktree_verification_used = False
        self._pending_reads: dict[int, tuple[object, dict[str, Any]]] = {}
        self._tracked_policy: dict[str, dict[str, Any]] | None = None

    def authorize_absolute(self, path: Path, role: str) -> str:
        binding = ABSOLUTE_ROLES.get(role)
        if binding is None:
            raise ValueError(f"untrusted absolute reader role: {role}")
        expected, category = binding
        if lexical_absolute(path) != lexical_absolute(expected):
            raise ValueError(f"absolute reader role/path mismatch: {role}")
        return category

    def authorize_relative(self, path: str, role: str) -> str:
        safe_relative_path(path, "source read path")
        expected_role = SOURCE_ROLES.get(path)
        if expected_role is None or expected_role != role:
            raise ValueError(f"untrusted relative reader role/path: {role}:{path}")
        return SOURCE_CATEGORY_BY_ROLE[role]

    def _issue_read(self, declaration: dict[str, Any]) -> object:
        token = object()
        self._pending_reads[id(token)] = (token, declaration)
        return token

    def authorize_retained_read(
        self,
        path: str,
        role: str,
        expected_bytes: int,
        expected_sha256: str,
    ) -> object:
        category = self.authorize_relative(path, role)
        return self._issue_read(
            {
                "path": path,
                "role": role,
                "category": category,
                "expected_bytes": expected_bytes,
                "expected_sha256": expected_sha256,
                "expected_blob_oid": None,
                "expected_mode": None,
            }
        )

    def authorize_git_config_read(
        self,
        config: dict[str, Any],
        *,
        phase: str,
        point: str,
    ) -> object:
        if phase not in {"initial", "terminal"} or point not in {"before", "after"}:
            raise ValueError("Git config verification phase is not authorized")
        source = config["source"]
        return self._issue_read(
            {
                "path": ".git/config",
                "role": f"git_local_config_{phase}_{point}",
                "category": "git_local_config",
                "expected_bytes": source["expected_git_config_bytes"],
                "expected_sha256": source["expected_git_config_sha256"],
                "expected_blob_oid": None,
                "expected_mode": "100644",
            }
        )

    def authorize_git_head_read(
        self,
        config: dict[str, Any],
        *,
        phase: str,
        point: str,
    ) -> object:
        if phase not in {"initial", "terminal"} or point not in {"before", "after"}:
            raise ValueError("Git HEAD verification phase is not authorized")
        source = config["source"]
        return self._issue_read(
            {
                "path": ".git/HEAD",
                "role": f"git_head_{phase}_{point}",
                "category": "git_head",
                "expected_bytes": source["expected_git_head_bytes"],
                "expected_sha256": source["expected_git_head_sha256"],
                "expected_blob_oid": None,
                "expected_mode": "100644",
            }
        )

    def bind_tracked_policy(self, entries: list[dict[str, Any]]) -> None:
        path_digest = hashlib.sha256(
            b"".join(entry["path"].encode("utf-8") + b"\0" for entry in entries)
        ).hexdigest()
        if len(entries) != 18 or path_digest != (
            "3791e2090d230f4d23d4795b304188f51f5c29e5ce27c7ad859cea1d6223dbe6"
        ):
            raise ValueError("tracked read policy does not match the hard-coded source lock")
        policy = {entry["path"]: dict(entry) for entry in entries}
        if len(policy) != len(entries):
            raise ValueError("tracked read policy contains duplicate paths")
        if self._tracked_policy is not None and self._tracked_policy != policy:
            raise RuntimeError("tracked read policy drifted")
        self._tracked_policy = policy

    def authorize_tracked_binding(self, path: str, *, phase: str) -> object:
        if phase not in {"initial", "terminal"}:
            raise ValueError("tracked byte-binding phase is not authorized")
        path = safe_relative_path(path, "tracked byte-binding path")
        if self._tracked_policy is None or path not in self._tracked_policy:
            raise ValueError("tracked byte-binding path is outside the bound Git policy")
        expected = self._tracked_policy[path]
        return self._issue_read(
            {
                "path": path,
                "role": f"tracked_blob_binding_{phase}",
                "category": "source_worktree_binding",
                "expected_bytes": expected["bytes"],
                "expected_sha256": None,
                "expected_blob_oid": expected["blob_oid"],
                "expected_mode": expected["mode"],
            }
        )

    def consume_read(self, token: object) -> tuple[int, dict[str, Any]]:
        pending = self._pending_reads.pop(id(token), None)
        if pending is None or pending[0] is not token:
            raise ValueError("read capability is absent, forged, or already consumed")
        declaration = pending[1]
        ledger_token = self.begin(
            path=declaration["path"],
            role=declaration["role"],
            category=declaration["category"],
        )
        return ledger_token, declaration

    def begin(self, *, path: str, role: str, category: str) -> int:
        """Record authorization before any path traversal or byte-read attempt."""
        self.entries.append(
            {
                "path": path,
                "role": role,
                "category": category,
                "outcome": "authorized",
                "read_started": False,
                "bytes": 0,
                "sha256": None,
                "error_type": None,
            }
        )
        return len(self.entries) - 1

    def mark_read_started(self, token: int) -> None:
        entry = self.entries[token]
        entry["read_started"] = True
        entry["outcome"] = "reading"

    def progress(self, token: int, size: int, digest: str | None) -> None:
        entry = self.entries[token]
        entry["bytes"] = size
        entry["sha256"] = digest

    def complete(self, token: int, size: int, digest: str) -> None:
        self.progress(token, size, digest)
        self.entries[token]["outcome"] = "verified"

    def fail(self, token: int, error: BaseException) -> None:
        entry = self.entries[token]
        entry["outcome"] = "failed"
        entry["error_type"] = type(error).__name__

    def count(self, category: str) -> int:
        return sum(
            entry["category"] == category and bool(entry["read_started"])
            for entry in self.entries
        )

    def bytes_for(self, category: str) -> int:
        return sum(
            int(entry["bytes"])
            for entry in self.entries
            if entry["category"] == category
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        return (text + "\n").encode("utf-8")
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_secure_open_flags() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC"):
        if not hasattr(os, name):
            raise RuntimeError(f"required secure-open flag is unavailable: {name}")


def directory_flags() -> int:
    require_secure_open_flags()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def regular_read_flags() -> int:
    require_secure_open_flags()
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ValueError(f"{field} must be normalized POSIX relative path")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{field} contains an unsafe component")
    return value


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{field} keys mismatch; missing={missing}, extra={extra}")
    return value


def require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def require_git_sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase full Git object id")
    return value


def require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, field: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {field}: {error}") from error


def stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def lexical_absolute(path: Path) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise ValueError("NUL byte in path")
    if any(part == ".." for part in Path(raw).parts):
        raise ValueError(f"path traversal is forbidden: {raw}")
    if not os.path.isabs(raw):
        raw = os.path.join(os.fspath(ROOT), raw)
    return Path(os.path.normpath(raw))


def lexical_within(path: Path, parent: Path) -> bool:
    child_s = os.fspath(lexical_absolute(path))
    parent_s = os.fspath(lexical_absolute(parent))
    try:
        return os.path.commonpath((child_s, parent_s)) == parent_s
    except ValueError:
        return False


def _absolute_parts(path: Path) -> tuple[str, ...]:
    absolute = lexical_absolute(path)
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise ValueError(f"absolute path required: {path}")
    return tuple(parts[1:])


def open_absolute_directory(path: Path) -> int:
    parts = _absolute_parts(path)
    descriptor = os.open(os.sep, directory_flags())
    try:
        for component in parts:
            next_descriptor = os.open(component, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_absolute_parent(path: Path) -> tuple[int, str, Path]:
    absolute = lexical_absolute(path)
    if absolute == Path(os.sep) or not absolute.name:
        raise ValueError("root path cannot be used as a file or output leaf")
    parent = absolute.parent
    return open_absolute_directory(parent), absolute.name, parent


def verify_directory_path_identity(path: Path, pinned_fd: int) -> None:
    check_fd = open_absolute_directory(path)
    try:
        if stat_signature(os.fstat(check_fd)) != stat_signature(os.fstat(pinned_fd)):
            raise OutputPathError(f"directory path identity changed: {path}")
    finally:
        os.close(check_fd)


def _read_fd_stable(
    fd: int,
    before: os.stat_result,
    progress: Callable[[int, str | None], None] | None = None,
) -> bytes:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        size += len(chunk)
        if progress is not None:
            progress(size, digest.hexdigest())
    payload = b"".join(chunks)
    after = os.fstat(fd)
    if stat_signature(after) != stat_signature(before):
        raise RuntimeError("file changed while being read")
    if len(payload) != before.st_size:
        raise RuntimeError("stable read byte count mismatch")
    return payload


def secure_read_absolute(path: Path, role: str, ledger: ReadLedger) -> bytes:
    category = ledger.authorize_absolute(path, role)
    absolute_path = os.fspath(lexical_absolute(path))
    token = ledger.begin(path=absolute_path, role=role, category=category)
    parent_fd: int | None = None
    file_fd: int | None = None
    try:
        parent_fd, leaf, _ = open_absolute_parent(path)
        entry_before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry_before.st_mode) or entry_before.st_nlink != 1:
            raise ValueError(f"absolute input must be one regular, non-hardlinked file: {path}")
        maximum = ABSOLUTE_MAX_BYTES[role]
        if entry_before.st_size > maximum:
            raise ValueError(
                f"absolute input exceeds role-specific byte cap ({maximum}): {role}"
            )
        file_fd = os.open(leaf, regular_read_flags(), dir_fd=parent_fd)
        opened = os.fstat(file_fd)
        if stat_signature(opened) != stat_signature(entry_before):
            raise RuntimeError(f"absolute input raced before pinning: {path}")
        ledger.mark_read_started(token)
        payload = _read_fd_stable(
            file_fd,
            opened,
            lambda size, digest: ledger.progress(token, size, digest),
        )
        entry_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat_signature(entry_after) != stat_signature(opened):
            raise RuntimeError(f"absolute input path changed while reading: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        ledger.complete(token, len(payload), digest)
        return payload
    except BaseException as error:
        ledger.fail(token, error)
        raise
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def validate_invocation_paths(config_path: Path, output_path: Path) -> None:
    config_absolute = lexical_absolute(config_path)
    output_absolute = lexical_absolute(output_path)
    if config_absolute != lexical_absolute(CANONICAL_CONFIG):
        raise ValueError("--config must be the canonical role-bound gate config")
    if output_absolute == Path(os.sep) or not output_absolute.name:
        raise OutputPathError("output directory must be a non-root leaf")
    for protected in (
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "tests",
        EXPECTED_SOURCE_ROOT,
    ):
        if lexical_within(output_absolute, protected):
            raise OutputPathError(f"output directory is inside protected input: {protected}")
    if output_absolute == config_absolute:
        raise OutputPathError("output directory collides with gate config")


def create_fresh_output(path: Path) -> FreshOutput:
    absolute = lexical_absolute(path)
    parent_fd, leaf, parent_path = open_absolute_parent(absolute)
    output_fd: int | None = None
    try:
        try:
            os.mkdir(leaf, mode=0o755, dir_fd=parent_fd)
        except FileExistsError as error:
            raise OutputPathError("output path must not exist") from error
        created_info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(created_info.st_mode):
            raise OutputPathError("created output leaf is not a directory")
        created_signature = stat_signature(created_info)
        created_identity = (
            int(created_info.st_dev),
            int(created_info.st_ino),
            int(created_info.st_mode),
        )
        output_fd = os.open(leaf, directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(output_fd)
        if stat_signature(opened) != created_signature:
            raise OutputPathError("created output leaf raced before pinning")
        result = FreshOutput(
            output_fd,
            parent_fd,
            parent_path,
            leaf,
            created_identity,
        )
        result.verify_leaf()
        os.fsync(parent_fd)
        result.verify_leaf()
        return result
    except BaseException:
        if output_fd is not None:
            os.close(output_fd)
        os.close(parent_fd)
        raise


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write while publishing report")
        offset += written


def rename_no_replace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one relative leaf without replacing an existing leaf."""
    if any("\x00" in name or "/" in name for name in (source_name, destination_name)):
        raise ValueError("rename leaves must be simple non-NUL names")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination_name)
        raise OSError(error_number, os.strerror(error_number), destination_name)


def write_json_no_clobber(output: FreshOutput, value: dict[str, Any]) -> None:
    output.verify_leaf()
    payload = canonical_bytes(value, pretty=True)
    temporary = f"._run-{os.getpid()}-{secrets.token_hex(12)}.tmp"
    output_fd = output.descriptor
    descriptor: int | None = None
    owned_identity: tuple[int, int, int, int] | None = None
    record_committed = False

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(stat.S_IFMT(value.st_mode)),
            int(value.st_size),
        )

    def owned_path(name: str) -> bool:
        if owned_identity is None:
            return False
        try:
            observed = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return stat.S_ISREG(observed.st_mode) and identity(observed) == owned_identity

    def leaf_matches() -> bool:
        try:
            output.verify_leaf()
        except (OSError, ValueError):
            return False
        return True

    def descriptor_payload_matches() -> bool:
        if descriptor is None or owned_identity is None:
            return False
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            before = os.fstat(descriptor)
            if identity(before) != owned_identity:
                return False
            observed = _read_fd_stable(descriptor, before)
            after = os.fstat(descriptor)
        except (OSError, ValueError, RuntimeError):
            return False
        return identity(after) == owned_identity and observed == payload

    def committed_record_is_valid() -> bool:
        return (
            leaf_matches()
            and owned_path("run.json")
            and descriptor_payload_matches()
            and owned_path("run.json")
        )

    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o644,
            dir_fd=output_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        written_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(written_info.st_mode)
            or written_info.st_nlink != 1
            or written_info.st_size != len(payload)
        ):
            raise OutputPathError("temporary report file identity is invalid")
        owned_identity = identity(written_info)
        if not owned_path(temporary) or not descriptor_payload_matches():
            raise OutputPathError("temporary report bytes do not match serialized record")
        output.verify_leaf()
        rename_no_replace(
            output_fd,
            temporary,
            output_fd,
            "run.json",
        )
        # The no-replace rename is the publication point.  Never unlink a
        # pathname during recovery: POSIX has no inode-conditional unlink,
        # and a stat-then-unlink sequence could delete a racer's replacement.
        if not committed_record_is_valid():
            raise OutputPathError("renamed report identity or bytes are invalid")
        os.fsync(output_fd)
        if not committed_record_is_valid():
            raise OutputPathError("report path identity changed before commit")
        record_committed = True
        if not committed_record_is_valid():
            record_committed = False
            raise OutputPathError("report path identity changed after commit")
    except FileExistsError as error:
        raise OutputPathError("run.json already exists; refusing to clobber") from error
    finally:
        close_error: OSError | None = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                close_error = error
        if close_error is not None and not record_committed:
            raise close_error


def validate_config(value: Any) -> dict[str, Any]:
    config = exact_keys(
        value,
        {
            "schema_version",
            "config_id",
            "method_id",
            "scope",
            "counted_toward_smoke",
            "config_read_policy",
            "expected_blocker_ids",
            "source_lock",
            "source",
            "notebooks",
            "ast_contract",
            "prior_reports",
            "licenses",
            "runtime_policy",
            "benchmark_policy",
            "controls",
        },
        "config",
    )
    expected_scalars = {
        "schema_version": 1,
        "config_id": "architects-2024-source-artifact-label-runtime-gate-v1",
        "method_id": "architects-2024",
        "scope": "source-artifact-label-runtime-gate-audit-only",
        "counted_toward_smoke": False,
    }
    for key, expected in expected_scalars.items():
        if config[key] != expected or type(config[key]) is not type(expected):
            raise ValueError(f"config.{key} mismatch")
    if config["config_read_policy"] != {
        "canonical_path": "configs/architects_2024_gate_v1.json",
        "alternate_paths_allowed": False,
    }:
        raise ValueError("config.config_read_policy mismatch")
    if config["expected_blocker_ids"] != list(EXPECTED_BLOCKERS):
        raise ValueError("config.expected_blocker_ids mismatch")

    source_lock = exact_keys(
        config["source_lock"], {"path", "sha256", "entry"}, "config.source_lock"
    )
    if source_lock["path"] != "configs/source_locks.json":
        raise ValueError("config.source_lock.path mismatch")
    if require_sha256(source_lock["sha256"], "config.source_lock.sha256") != (
        "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
    ):
        raise ValueError("config.source_lock.sha256 mismatch")
    expected_entry = {
        "url": "https://github.com/da-fr/arc-prize-2024",
        "branch": "main",
        "revision": EXPECTED_REVISION,
        "asset_subpath": "sources/architects-2024",
    }
    if source_lock["entry"] != expected_entry:
        raise ValueError("config.source_lock.entry mismatch")

    source = exact_keys(
        config["source"],
        {
            "repository_path",
            "repository_url",
            "expected_revision",
            "expected_commit_tree",
            "expected_git_config_bytes",
            "expected_git_config_sha256",
            "expected_git_head_bytes",
            "expected_git_head_sha256",
            "expected_tracked_file_count",
            "tracked_path_list_sha256",
            "tracked_metadata_sha256",
            "tracked_files",
            "read_files",
            "expected_python_file_count",
            "expected_python_bytes",
            "opaque_directories",
            "expected_unknown_entry_count",
        },
        "config.source",
    )
    if source["repository_path"] != os.fspath(EXPECTED_SOURCE_ROOT):
        raise ValueError("config.source.repository_path mismatch")
    if source["repository_url"] != expected_entry["url"]:
        raise ValueError("config.source.repository_url mismatch")
    if require_git_sha(source["expected_revision"], "source revision") != EXPECTED_REVISION:
        raise ValueError("config.source.expected_revision mismatch")
    if require_git_sha(source["expected_commit_tree"], "source tree") != EXPECTED_TREE:
        raise ValueError("config.source.expected_commit_tree mismatch")
    if require_nonnegative_int(
        source["expected_git_config_bytes"], "Git config byte count"
    ) != 309:
        raise ValueError("config.source.expected_git_config_bytes mismatch")
    if require_sha256(
        source["expected_git_config_sha256"], "Git config SHA-256"
    ) != "1737a734def03c3a14f4af03bf84e452d3a4fd431e28073ba808670873e296fc":
        raise ValueError("config.source.expected_git_config_sha256 mismatch")
    if require_nonnegative_int(
        source["expected_git_head_bytes"], "Git HEAD byte count"
    ) != 41:
        raise ValueError("config.source.expected_git_head_bytes mismatch")
    if require_sha256(
        source["expected_git_head_sha256"], "Git HEAD SHA-256"
    ) != "721b0545affeb16fe2d3fdf5079cbc5cb9ce2d9b713a94556187a998ad7f3f8a":
        raise ValueError("config.source.expected_git_head_sha256 mismatch")
    if require_nonnegative_int(source["expected_tracked_file_count"], "tracked count") != 18:
        raise ValueError("config.source.expected_tracked_file_count mismatch")
    if require_sha256(source["tracked_path_list_sha256"], "tracked path digest") != (
        "3791e2090d230f4d23d4795b304188f51f5c29e5ce27c7ad859cea1d6223dbe6"
    ):
        raise ValueError("config.source.tracked_path_list_sha256 mismatch")
    if require_sha256(
        source["tracked_metadata_sha256"], "tracked metadata digest"
    ) != "b501cc77005eaf78d097da247a5ad147be6cf5d7512a9447e8a5fefa915a3efb":
        raise ValueError("config.source.tracked_metadata_sha256 mismatch")
    if source["opaque_directories"] != [".git"]:
        raise ValueError("config.source.opaque_directories mismatch")
    if source["expected_unknown_entry_count"] != 0:
        raise ValueError("config.source.expected_unknown_entry_count mismatch")
    if source["expected_python_file_count"] != 9 or source["expected_python_bytes"] != 60690:
        raise ValueError("config.source Python inventory mismatch")

    tracked = source["tracked_files"]
    if not isinstance(tracked, list) or len(tracked) != 18:
        raise ValueError("config.source.tracked_files must contain 18 entries")
    tracked_paths: list[str] = []
    for index, entry in enumerate(tracked):
        item = exact_keys(
            entry, {"path", "mode", "blob_oid", "bytes"}, f"tracked_files[{index}]"
        )
        path = safe_relative_path(item["path"], f"tracked_files[{index}].path")
        if item["mode"] != "100644":
            raise ValueError("only locked regular non-executable source files are permitted")
        require_git_sha(item["blob_oid"], f"tracked_files[{index}].blob_oid")
        require_nonnegative_int(item["bytes"], f"tracked_files[{index}].bytes")
        tracked_paths.append(path)
    if tracked_paths != sorted(tracked_paths) or len(set(tracked_paths)) != len(tracked_paths):
        raise ValueError("tracked_files paths must be unique and sorted")
    path_digest = hashlib.sha256(
        b"".join(path.encode("utf-8") + b"\0" for path in tracked_paths)
    ).hexdigest()
    if path_digest != source["tracked_path_list_sha256"]:
        raise ValueError("tracked file path digest mismatch")
    if canonical_sha256(tracked) != source["tracked_metadata_sha256"]:
        raise ValueError("tracked file metadata digest mismatch")
    if git_tree_oid_from_manifest(tracked) != source["expected_commit_tree"]:
        raise ValueError("tracked manifest does not reconstruct the locked Git tree")

    reads = source["read_files"]
    if not isinstance(reads, list) or len(reads) != len(SOURCE_ROLES):
        raise ValueError("config.source.read_files count mismatch")
    seen_reads: set[str] = set()
    for index, entry in enumerate(reads):
        item = exact_keys(
            entry, {"path", "role", "bytes", "sha256"}, f"read_files[{index}]"
        )
        path = safe_relative_path(item["path"], f"read_files[{index}].path")
        if path in seen_reads or path not in tracked_paths:
            raise ValueError("read_files contains duplicate or untracked path")
        if SOURCE_ROLES.get(path) != item["role"]:
            raise ValueError(f"read_files role mismatch for {path}")
        require_nonnegative_int(item["bytes"], f"read_files[{index}].bytes")
        require_sha256(item["sha256"], f"read_files[{index}].sha256")
        seen_reads.add(path)
    if seen_reads != set(SOURCE_ROLES):
        raise ValueError("read_files does not match hard-coded role allowlist")

    notebooks = exact_keys(config["notebooks"], {"official", "updated"}, "notebooks")
    official = exact_keys(
        notebooks["official"],
        {"path", "role", "sha256", "challenge_file_suffix", "fake_only_reply_file_suffix"},
        "notebooks.official",
    )
    updated = exact_keys(
        notebooks["updated"], {"path", "role", "sha256"}, "notebooks.updated"
    )
    expected_official = {
        "path": "kaggle_notebooks/arc-prize-2024_kaggle.ipynb",
        "role": "official-kaggle-53.5-submission",
        "sha256": "4ecb52b2811226711b656f40bf3ecff62509c2a189a222c1394cb14180b2bfe9",
        "challenge_file_suffix": "arc-agi_test_challenges.json",
        "fake_only_reply_file_suffix": "arc-agi_training_solutions.json",
    }
    expected_updated = {
        "path": "kaggle_notebooks/arc-prize-2024_updated.ipynb",
        "role": "updated-local-candidate-not-official-submission",
        "sha256": "28e76d2c888ab98a0169d9e6889d1b4ed4d03dc54ebd877401e055658d41d6dd",
    }
    if official != expected_official or updated != expected_updated:
        raise ValueError("official/updated notebook role binding mismatch")
    if official["sha256"] == updated["sha256"]:
        raise ValueError("official and updated notebooks must remain distinct")

    expected_ast = {
        "local_label_bearing_runners": [
            "training_code/run_evaluation_Llama-rearc_with_ttt.py",
            "training_code/run_evaluation_Llama-rearc_without_ttt.py",
        ],
        "contamination_training_path": "training_code/run_finetuning_Nemo-full.py",
        "model_loader_path": "training_code/model_tools.py",
        "required_local_challenge_suffix": "arc-agi_evaluation_challenges.json",
        "required_local_solution_suffix": "arc-agi_evaluation_solutions.json",
    }
    if config["ast_contract"] != expected_ast:
        raise ValueError("config.ast_contract mismatch")

    prior = exact_keys(
        config["prior_reports"], {"checkpoint_integrity", "resource_preflight"}, "prior_reports"
    )
    checkpoint = exact_keys(
        prior["checkpoint_integrity"],
        {
            "path",
            "sha256",
            "scope",
            "status",
            "repo_id",
            "revision",
            "file_count",
            "total_bytes",
            "model_file",
            "notice_file",
            "model_card_file",
        },
        "prior_reports.checkpoint_integrity",
    )
    if checkpoint["path"] != CHECKPOINT_REPORT.relative_to(ROOT).as_posix():
        raise ValueError("checkpoint report path mismatch")
    if require_sha256(checkpoint["sha256"], "checkpoint report hash") != (
        "6cf23296baf789502b990be884e0420898b067ba49e041dfade1396c0a5ef8f3"
    ):
        raise ValueError("checkpoint report hash mismatch")
    if checkpoint["scope"] != "checkpoint-download-integrity-only" or checkpoint["status"] != "passed":
        raise ValueError("checkpoint report claim mismatch")
    if checkpoint["repo_id"] != "da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit":
        raise ValueError("checkpoint repo mismatch")
    require_git_sha(checkpoint["revision"], "checkpoint revision")
    if checkpoint["revision"] != "6de719999a213e717fe339fb5a29177ddc4310d9":
        raise ValueError("checkpoint revision mismatch")
    if checkpoint["file_count"] != 9 or checkpoint["total_bytes"] != 3790920477:
        raise ValueError("checkpoint inventory summary mismatch")
    expected_artifact_files = {
        "model_file": {
            "path": "model.safetensors",
            "bytes": 3790909677,
            "sha256": "96ae74f8955c5bf3e84d5732494525d13076410710a060641893359db13300c5",
        },
        "notice_file": {
            "path": "NOTICE",
            "bytes": 588,
            "sha256": "13167b3e366cf2cf314c8d6b934791d34496183cd1ab594ebb23d128c9bb7f62",
        },
        "model_card_file": {
            "path": "README.md",
            "bytes": 2357,
            "sha256": "9c1f7dbd70693d145f0324b1d4a4ec671b40f220eaf36aaff62776a350f992fb",
        },
    }
    for key, expected in expected_artifact_files.items():
        if checkpoint[key] != expected:
            raise ValueError(f"checkpoint {key} mismatch")

    preflight = exact_keys(
        prior["resource_preflight"],
        {"path", "sha256", "status", "minimum_free_vram_bytes", "observed_free_vram_bytes"},
        "prior_reports.resource_preflight",
    )
    expected_preflight = {
        "path": PREFLIGHT_REPORT.relative_to(ROOT).as_posix(),
        "sha256": "3c1ca104f4f5317404196308e515f2e42f7161be63f45395e7df2be93aff74c6",
        "status": "blocked",
        "minimum_free_vram_bytes": 10737418240,
        "observed_free_vram_bytes": 5335154688,
    }
    if preflight != expected_preflight:
        raise ValueError("resource preflight binding mismatch")

    expected_licenses = {
        "code": {
            "path": "LICENSE.txt",
            "identifier": "Apache-2.0",
            "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        },
        "model": {
            "identifier": "NVIDIA Open Model License",
            "notice_bound_from_integrity_report": True,
            "redistribution_review_status": "blocked-separate-review-required",
        },
    }
    if config["licenses"] != expected_licenses:
        raise ValueError("config.licenses mismatch")
    expected_runtime = {
        "checkpoint_local_only_required": True,
        "trust_remote_code_required": False,
        "pickle_cache_allowed": False,
        "historical_dependency_stack": "Unsloth/PyTorch-2.4-era",
        "local_dependency_stack": "Transformers-4.44.2/bitsandbytes-0.44.1/accelerate-0.34.2/PyTorch-2.13.0+cu126",
        "minimum_free_vram_bytes": 10737418240,
    }
    if config["runtime_policy"] != expected_runtime:
        raise ValueError("config.runtime_policy mismatch")
    expected_benchmark = {
        "arc_agi_1": "training-contaminated-ineligible-for-clean-main-board",
        "arc_agi_2": "potential-new-transfer-after-challenge-only-adapter-and-all-runtime-gates",
    }
    if config["benchmark_policy"] != expected_benchmark:
        raise ValueError("config.benchmark_policy mismatch")
    expected_control_keys = {
        "network_allowed",
        "gpu_allowed",
        "upstream_import_allowed",
        "upstream_execution_allowed",
        "arc_solution_byte_read_allowed",
        "checkpoint_byte_read_allowed",
        "pickle_cache_byte_read_allowed",
        "solver_execution_allowed",
        "prediction_allowed",
    }
    controls = exact_keys(config["controls"], expected_control_keys, "config.controls")
    if any(value is not False for value in controls.values()):
        raise ValueError("all static audit controls must be false")
    return config


def expected_directories(config: dict[str, Any]) -> set[str]:
    directories = {".git"}
    for entry in config["source"]["tracked_files"]:
        path = PurePosixPath(entry["path"])
        for index in range(1, len(path.parts)):
            directories.add(PurePosixPath(*path.parts[:index]).as_posix())
    return directories


def _mode_string(mode: int) -> str:
    executable_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    return "100755" if mode & executable_bits else "100644"


def _scan_directory(
    fd: int,
    prefix: str,
    *,
    opaque: set[str],
    files: dict[str, dict[str, Any]],
    directories: set[str],
) -> None:
    with os.scandir(fd) as iterator:
        entries = sorted(iterator, key=lambda item: item.name)
    for entry in entries:
        if entry.name in (".", "..") or "/" in entry.name or "\x00" in entry.name:
            raise ValueError("unsafe source directory entry")
        relative = f"{prefix}/{entry.name}" if prefix else entry.name
        info = entry.stat(follow_symlinks=False)
        if relative in opaque:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"opaque entry is not a directory: {relative}")
            directories.add(relative)
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"source symlink is forbidden: {relative}")
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
            child_fd = os.open(entry.name, directory_flags(), dir_fd=fd)
            try:
                if stat_signature(os.fstat(child_fd)) != stat_signature(info):
                    raise RuntimeError(f"source directory raced: {relative}")
                _scan_directory(
                    child_fd,
                    relative,
                    opaque=opaque,
                    files=files,
                    directories=directories,
                )
                after = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                if stat_signature(after) != stat_signature(info):
                    raise RuntimeError(f"source directory changed: {relative}")
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"source entry is not a private regular file: {relative}")
        files[relative] = {
            "path": relative,
            "mode": _mode_string(info.st_mode),
            "bytes": int(info.st_size),
            "_stat_signature": stat_signature(info),
        }


def closed_world_inventory(
    root_fd: int,
    config: dict[str, Any],
    ledger: ReadLedger,
    *,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"initial", "terminal"}:
        raise ValueError("closed-world inventory phase is not authorized")
    files: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    opaque = set(config["source"]["opaque_directories"])
    _scan_directory(
        root_fd,
        "",
        opaque=opaque,
        files=files,
        directories=directories,
    )
    expected_files = {
        entry["path"]: entry for entry in config["source"]["tracked_files"]
    }
    if set(files) != set(expected_files):
        raise ValueError(
            "closed-world source file mismatch; "
            f"missing={sorted(set(expected_files) - set(files))}, "
            f"extra={sorted(set(files) - set(expected_files))}"
        )
    for path, observed in files.items():
        expected = expected_files[path]
        if observed["mode"] != expected["mode"] or observed["bytes"] != expected["bytes"]:
            raise ValueError(f"closed-world metadata mismatch: {path}")
    expected_dirs = expected_directories(config)
    if directories != expected_dirs:
        raise ValueError(
            "closed-world source directory mismatch; "
            f"missing={sorted(expected_dirs - directories)}, "
            f"extra={sorted(directories - expected_dirs)}"
        )
    byte_bindings: list[dict[str, Any]] = []
    for path in sorted(expected_files):
        expected = expected_files[path]
        capability = ledger.authorize_tracked_binding(path, phase=phase)
        payload = _secure_read_locked_relative(
            root_fd,
            ledger,
            capability,
        )
        byte_bindings.append(
            {
                "path": path,
                "mode": expected["mode"],
                "bytes": len(payload),
                "blob_oid": git_blob_oid(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    final_files: dict[str, dict[str, Any]] = {}
    final_directories: set[str] = set()
    _scan_directory(
        root_fd,
        "",
        opaque=opaque,
        files=final_files,
        directories=final_directories,
    )
    if final_files != files or final_directories != directories:
        raise RuntimeError("closed-world source identities changed during blob binding")
    public_metadata = [
        {
            "path": files[path]["path"],
            "mode": files[path]["mode"],
            "bytes": files[path]["bytes"],
        }
        for path in sorted(files)
    ]
    return {
        "file_count": len(files),
        "directory_count": len(directories),
        "unknown_entry_count": 0,
        "opaque_directories": sorted(opaque),
        "metadata_sha256": canonical_sha256(public_metadata),
        "tracked_bytes": sum(item["bytes"] for item in byte_bindings),
        "tracked_blob_binding_sha256": canonical_sha256(byte_bindings),
        "tracked_blob_bindings": byte_bindings,
    }


def git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def git_tree_oid_from_manifest(entries: list[dict[str, Any]]) -> str:
    def new_directory() -> dict[str, dict[str, Any]]:
        return {"files": {}, "directories": {}}

    root = new_directory()
    for entry in entries:
        parts = PurePosixPath(entry["path"]).parts
        directory = root
        for component in parts[:-1]:
            if component in directory["files"]:
                raise ValueError("tracked manifest has a file/directory collision")
            directory = directory["directories"].setdefault(
                component, new_directory()
            )
        leaf = parts[-1]
        if leaf in directory["directories"] or leaf in directory["files"]:
            raise ValueError("tracked manifest has a duplicate or path collision")
        directory["files"][leaf] = entry

    def tree_oid(directory: dict[str, dict[str, Any]]) -> str:
        items: list[tuple[bytes, str, str, str]] = []
        for name, entry in directory["files"].items():
            items.append((name.encode("utf-8"), entry["mode"], name, entry["blob_oid"]))
        for name, child in directory["directories"].items():
            items.append(
                ((name + "/").encode("utf-8"), "40000", name, tree_oid(child))
            )
        body = b"".join(
            f"{mode} {name}\0".encode("utf-8") + bytes.fromhex(object_id)
            for _sort_key, mode, name, object_id in sorted(items, key=lambda item: item[0])
        )
        header = f"tree {len(body)}\0".encode("ascii")
        return hashlib.sha1(header + body, usedforsecurity=False).hexdigest()

    return tree_oid(root)


def _secure_read_locked_relative(
    root_fd: int,
    ledger: ReadLedger,
    capability: object,
) -> bytes:
    token, declaration = ledger.consume_read(capability)
    relative = safe_relative_path(declaration["path"], "relative source path")
    expected_bytes = declaration["expected_bytes"]
    expected_sha256 = declaration["expected_sha256"]
    expected_blob_oid = declaration["expected_blob_oid"]
    expected_mode = declaration["expected_mode"]
    parts = PurePosixPath(relative).parts
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.dup(root_fd)
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        leaf = parts[-1]
        entry = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise ValueError(f"source read target is not a private regular file: {relative}")
        if int(entry.st_size) != expected_bytes:
            raise ValueError(f"source byte count lock mismatch: {relative}")
        if expected_mode is not None and _mode_string(entry.st_mode) != expected_mode:
            raise ValueError(f"source mode lock mismatch: {relative}")
        file_fd = os.open(leaf, regular_read_flags(), dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        if stat_signature(opened) != stat_signature(entry):
            raise RuntimeError(f"source file raced before pinning: {relative}")
        ledger.mark_read_started(token)
        payload = _read_fd_stable(
            file_fd,
            opened,
            lambda size, digest: ledger.progress(token, size, digest),
        )
        after = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat_signature(after) != stat_signature(opened):
            raise RuntimeError(f"source file changed while reading: {relative}")
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes:
            raise ValueError(f"source byte count lock mismatch: {relative}")
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"source byte lock mismatch: {relative}")
        if expected_blob_oid is not None and git_blob_oid(payload) != expected_blob_oid:
            raise ValueError(f"source Git blob lock mismatch: {relative}")
        ledger.complete(token, len(payload), digest)
        return payload
    except BaseException as error:
        ledger.fail(token, error)
        raise
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def secure_read_relative(
    root_fd: int,
    relative: str,
    role: str,
    expected_bytes: int,
    expected_sha256: str,
    ledger: ReadLedger,
) -> bytes:
    capability = ledger.authorize_retained_read(
        relative,
        role,
        expected_bytes,
        expected_sha256,
    )
    return _secure_read_locked_relative(
        root_fd,
        ledger,
        capability,
    )


def verify_local_git_config(
    root_fd: int,
    config: dict[str, Any],
    ledger: ReadLedger,
    *,
    phase: str,
    point: str,
) -> dict[str, Any]:
    if phase not in {"initial", "terminal"} or point not in {"before", "after"}:
        raise ValueError("Git config verification phase is not authorized")
    source = config["source"]
    capability = ledger.authorize_git_config_read(
        config, phase=phase, point=point
    )
    payload = _secure_read_locked_relative(
        root_fd,
        ledger,
        capability,
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("local Git config is not UTF-8") from error
    section = ""
    remote_urls: list[str] = []
    dangerous_sections = {"include", "includeif", "filter", "credential", "url"}
    dangerous_keys = {
        "askpass",
        "clean",
        "command",
        "editor",
        "fsmonitor",
        "helper",
        "hookspath",
        "pager",
        "process",
        "proxy",
        "smudge",
        "sshcommand",
    }
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            family = section.split(None, 1)[0].lower()
            if family in dangerous_sections:
                raise ValueError(
                    f"dangerous local Git config section at line {line_number}"
                )
            continue
        if not section or "=" not in line:
            raise ValueError(f"malformed local Git config at line {line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if key.lower() in dangerous_keys:
            raise ValueError(
                f"dangerous local Git config directive at line {line_number}"
            )
        if section.lower() == 'remote "origin"' and key.lower() == "url":
            remote_urls.append(value)
    if remote_urls != [source["repository_url"]]:
        raise ValueError("local Git config origin URL mismatch")
    return {
        "path": ".git/config",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "origin_url": remote_urls[0],
        "dangerous_include_filter_helper_directives": False,
    }


def verify_git_head(
    root_fd: int,
    config: dict[str, Any],
    ledger: ReadLedger,
    *,
    phase: str,
    point: str,
) -> dict[str, Any]:
    capability = ledger.authorize_git_head_read(
        config, phase=phase, point=point
    )
    payload = _secure_read_locked_relative(root_fd, ledger, capability)
    expected = (config["source"]["expected_revision"] + "\n").encode("ascii")
    if payload != expected:
        raise ValueError("Git HEAD is not the exact detached locked revision")
    return {
        "path": ".git/HEAD",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "detached_revision": payload.decode("ascii").strip(),
    }


def verify_git_auxiliary_paths_absent(root_fd: int) -> dict[str, Any]:
    for relative in FORBIDDEN_GIT_AUXILIARY_PATHS:
        parts = PurePosixPath(relative).parts
        directory_fd = os.dup(root_fd)
        try:
            missing_parent = False
            for component in parts[:-1]:
                try:
                    next_fd = os.open(component, directory_flags(), dir_fd=directory_fd)
                except FileNotFoundError:
                    missing_parent = True
                    break
                os.close(directory_fd)
                directory_fd = next_fd
            if missing_parent:
                continue
            try:
                os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError(f"forbidden Git auxiliary path exists: {relative}")
        finally:
            os.close(directory_fd)
    return {
        "required_absent_paths": list(FORBIDDEN_GIT_AUXILIARY_PATHS),
        "all_absent": True,
    }


def verify_git_contract(
    root_fd: int,
    config: dict[str, Any],
    ledger: ReadLedger,
    *,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"initial", "terminal"}:
        raise ValueError("Git verification phase is not authorized")
    auxiliary_before = verify_git_auxiliary_paths_absent(root_fd)
    config_before = verify_local_git_config(
        root_fd, config, ledger, phase=phase, point="before"
    )
    head_before = verify_git_head(
        root_fd, config, ledger, phase=phase, point="before"
    )
    entries = [dict(entry) for entry in config["source"]["tracked_files"]]
    path_digest = hashlib.sha256(
        b"".join(entry["path"].encode("utf-8") + b"\0" for entry in entries)
    ).hexdigest()
    if path_digest != config["source"]["tracked_path_list_sha256"]:
        raise ValueError("locked tracked path digest mismatch")
    metadata_digest = canonical_sha256(entries)
    if metadata_digest != config["source"]["tracked_metadata_sha256"]:
        raise ValueError("locked tracked metadata digest mismatch")
    reconstructed_tree = git_tree_oid_from_manifest(entries)
    if reconstructed_tree != EXPECTED_TREE or reconstructed_tree != config["source"]["expected_commit_tree"]:
        raise ValueError("locked manifest does not reconstruct the expected Git tree")
    ledger.bind_tracked_policy(entries)
    head_after = verify_git_head(
        root_fd, config, ledger, phase=phase, point="after"
    )
    config_after = verify_local_git_config(
        root_fd, config, ledger, phase=phase, point="after"
    )
    auxiliary_after = verify_git_auxiliary_paths_absent(root_fd)
    if config_after != config_before:
        raise RuntimeError("local Git config drifted across metadata verification")
    if head_after != head_before:
        raise RuntimeError("Git HEAD drifted across metadata verification")
    if auxiliary_after != auxiliary_before:
        raise RuntimeError("Git auxiliary-path policy drifted across metadata verification")
    return {
        "revision": head_before["detached_revision"],
        "commit_tree": config["source"]["expected_commit_tree"],
        "remote_url": config_before["origin_url"],
        "local_config": config_before,
        "detached_head": head_before,
        "forbidden_auxiliary_paths": auxiliary_before,
        "subprocess_used": False,
        "object_database_read": False,
        "tracked_metadata_source": "canonical-hard-coded-manifest",
        "worktree_status_command_used": False,
        "git_may_read_worktree_bytes": False,
        "object_database_reads_possible": False,
        "object_database_bytes_measured": True,
        "tracked_file_count": len(entries),
        "tracked_path_list_sha256": path_digest,
        "tracked_metadata_sha256": metadata_digest,
        "reconstructed_tree_oid": reconstructed_tree,
    }


def parse_python(payload: bytes, path: str) -> ast.Module:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Python source is not UTF-8: {path}") from error
    try:
        return ast.parse(text, filename=path)
    except SyntaxError as error:
        raise ValueError(f"Python syntax failure in {path}: {error}") from error


def _call_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _string_constants(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _has_control_ancestor(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    controls = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    while current is not None:
        if isinstance(current, controls):
            return True
        current = parents.get(current)
    return False


def analyze_local_runner(tree: ast.Module, path: str) -> dict[str, Any]:
    parents = _parent_map(tree)
    strings = _string_constants(tree)
    challenge_suffix = "arc-agi_evaluation_challenges.json"
    solution_suffix = "arc-agi_evaluation_solutions.json"
    challenge_calls: list[int] = []
    solution_calls: list[int] = []
    validation_calls: list[int] = []
    unconditional_solution_calls: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_path(node.func) or ""
        if name.endswith("load_from_json"):
            challenge_calls.append(node.lineno)
        if name.endswith("load_solutions"):
            solution_calls.append(node.lineno)
            if not _has_control_ancestor(node, parents):
                unconditional_solution_calls.append(node.lineno)
        if name.endswith("validate_submission"):
            validation_calls.append(node.lineno)
    observation = {
        "path": path,
        "challenge_suffix_present": any(value.endswith(challenge_suffix) for value in strings),
        "solution_suffix_present": any(value.endswith(solution_suffix) for value in strings),
        "challenge_load_call_lines": sorted(challenge_calls),
        "solution_load_call_lines": sorted(solution_calls),
        "unconditional_solution_load_call_lines": sorted(unconditional_solution_calls),
        "same_process_validation_call_lines": sorted(validation_calls),
    }
    observation["unconditional_solution_read"] = bool(
        observation["solution_suffix_present"] and unconditional_solution_calls
    )
    observation["contract_passed"] = bool(
        observation["challenge_suffix_present"]
        and observation["solution_suffix_present"]
        and challenge_calls
        and unconditional_solution_calls
        and validation_calls
    )
    return observation


def _is_action_train(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    values = _string_constants(test)
    names = {node.id for node in ast.walk(test) if isinstance(node, ast.Name)}
    return "action" in names and "train" in values and isinstance(test.ops[0], ast.Eq)


def analyze_contamination(tree: ast.Module) -> dict[str, Any]:
    train_if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If) and _is_action_train(node.test)]
    load_lines: list[int] = []
    move_lines: list[int] = []
    repeat_lines: list[int] = []
    train_strings: list[str] = []
    for train_if in train_if_nodes:
        body_nodes = [child for statement in train_if.body for child in ast.walk(statement)]
        train_strings.extend(
            child.value
            for child in body_nodes
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
        for child in body_nodes:
            if not isinstance(child, ast.Call):
                continue
            name = _call_path(child.func) or ""
            if name.endswith("load_solutions"):
                load_lines.append(child.lineno)
            if name.endswith("move_test_to_train"):
                move_lines.append(child.lineno)
            if name.endswith("repeat"):
                repeat_lines.append(child.lineno)
    arceval_key_lines = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "arceval"
    )
    result = {
        "train_branch_lines": sorted(node.lineno for node in train_if_nodes),
        "evaluation_solution_suffix_in_train_branch": any(
            value.endswith("arc-agi_evaluation_solutions.json") for value in train_strings
        ),
        "load_solutions_lines": sorted(load_lines),
        "move_test_to_train_lines": sorted(move_lines),
        "repeat_lines": sorted(repeat_lines),
        "arceval_mix_key_lines": arceval_key_lines,
    }
    result["arc1_training_contamination_confirmed"] = bool(
        result["train_branch_lines"]
        and result["evaluation_solution_suffix_in_train_branch"]
        and load_lines
        and move_lines
        and repeat_lines
        and arceval_key_lines
    )
    return result


def analyze_model_loader(tree: ast.Module) -> dict[str, Any]:
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_unsloth_4bit"
    ]
    if len(functions) != 1:
        raise ValueError("expected exactly one load_unsloth_4bit function")
    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and (_call_path(node.func) or "").endswith("from_pretrained")
    ]
    if len(calls) != 1:
        raise ValueError("expected exactly one model from_pretrained call")
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords if keyword.arg}
    local_only = keywords.get("local_files_only")
    trust_remote = keywords.get("trust_remote_code")
    return {
        "function_line": functions[0].lineno,
        "from_pretrained_line": calls[0].lineno,
        "keyword_names": sorted(keywords),
        "local_files_only_explicit_true": isinstance(local_only, ast.Constant) and local_only.value is True,
        "trust_remote_code_explicit_false": isinstance(trust_remote, ast.Constant) and trust_remote.value is False,
        "safe_offline_contract_passed": bool(
            isinstance(local_only, ast.Constant)
            and local_only.value is True
            and isinstance(trust_remote, ast.Constant)
            and trust_remote.value is False
        ),
    }


def _sanitize_ipython_source(source: str) -> str:
    output: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith(("%%", "%", "!")):
            output.append(indent + "pass")
        else:
            output.append(line)
    return "\n".join(output) + "\n"


def _notebook_code_cells(payload: bytes, path: str) -> list[tuple[int, str, ast.Module]]:
    notebook = strict_json(payload, path)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"invalid notebook cell structure: {path}")
    result: list[tuple[int, str, ast.Module]] = []
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source")
        if isinstance(source_value, list) and all(isinstance(item, str) for item in source_value):
            source = "".join(source_value)
        elif isinstance(source_value, str):
            source = source_value
        else:
            raise ValueError(f"invalid code-cell source in {path} cell {index}")
        sanitized = _sanitize_ipython_source(source)
        try:
            tree = ast.parse(sanitized, filename=f"{path}:cell-{index}")
        except SyntaxError as error:
            raise ValueError(f"notebook code syntax failure in {path} cell {index}: {error}") from error
        result.append((index, source, tree))
    return result


def _test_is_fake(test: ast.AST) -> bool:
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "is_fake"
        and isinstance(test.value, ast.Name)
        and test.value.id == "arc_test_set"
    )


def _node_in_statements(node: ast.AST, statements: Iterable[ast.stmt]) -> bool:
    return any(node is descendant for statement in statements for descendant in ast.walk(statement))


def _positive_fake_ancestor(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.If) and _test_is_fake(current.test):
            if _node_in_statements(node, current.body):
                return True
        current = parents.get(current)
    return False


def _constant_dead_ancestor(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.If) and isinstance(current.test, ast.Constant):
            active = current.body if bool(current.test.value) else current.orelse
            inactive = current.orelse if bool(current.test.value) else current.body
            if _node_in_statements(node, inactive):
                return True
            if _node_in_statements(node, active):
                current = parents.get(current)
                continue
        current = parents.get(current)
    return False


def _assigned_to_name(
    node: ast.AST, parents: dict[ast.AST, ast.AST], expected: str
) -> bool:
    current = parents.get(node)
    while current is not None and isinstance(
        current, (ast.Attribute, ast.Call, ast.Subscript, ast.BinOp)
    ):
        current = parents.get(current)
    if not isinstance(current, (ast.Assign, ast.AnnAssign)):
        return False
    targets = current.targets if isinstance(current, ast.Assign) else [current.target]
    return any(isinstance(target, ast.Name) and target.id == expected for target in targets)


def _name_argument(call: ast.Call, index: int, expected: str) -> bool:
    return (
        len(call.args) > index
        and isinstance(call.args[index], ast.Name)
        and call.args[index].id == expected
    )


def _assignment_value(statement: ast.stmt, expected_target: str) -> ast.AST | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
        value = statement.value
    else:
        return None
    if not isinstance(target, ast.Name) or target.id != expected_target:
        return None
    return value


def _direct_assignment_call(
    statement: ast.stmt, expected_target: str, expected_call_path: str
) -> ast.Call | None:
    value = _assignment_value(statement, expected_target)
    if not isinstance(value, ast.Call) or _call_path(value.func) != expected_call_path:
        return None
    return value


def _constant_bool_keyword(call: ast.Call, keyword_name: str, expected: bool) -> bool:
    values = [keyword.value for keyword in call.keywords if keyword.arg == keyword_name]
    return len(values) == 1 and isinstance(values[0], ast.Constant) and values[0].value is expected


def _effective_scope_nodes(statements: Iterable[ast.stmt]) -> Iterable[ast.AST]:
    """Yield executable nodes, excluding nested scopes and constant-false decoys."""

    def visit(node: ast.AST) -> Iterable[ast.AST]:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return
        yield node
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            branch = node.body if node.test.value else node.orelse
            for statement in branch:
                yield from visit(statement)
            return
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    for statement in statements:
        yield from visit(statement)


def _prepare_dataset_contract(
    cell_index: int, function: ast.FunctionDef
) -> dict[str, Any]:
    argument_names = [argument.arg for argument in function.args.args]
    train_ifs = [
        (index, statement)
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "train"
    ]
    seed_indices = [
        index
        for index, statement in enumerate(function.body)
        if isinstance(_assignment_value(statement, "ds"), ast.Name)
        and _assignment_value(statement, "ds").id == "arc_test_set"  # type: ignore[union-attr]
    ]
    return_indices = [
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Name)
        and statement.value.id == "ds"
    ]
    removal_lines: list[int] = []
    augment_lines: list[int] = []
    cut_lines: list[int] = []
    consumer_positions: list[tuple[int, int]] = []
    removal_positions: list[int] = []
    train_if_line: int | None = None
    if len(train_ifs) == 1:
        train_index, train_if = train_ifs[0]
        train_if_line = train_if.lineno
        for statement_index, statement in enumerate(train_if.body):
            removal = _direct_assignment_call(statement, "ds", "ds.remove_replies")
            if removal is not None and not removal.args and not removal.keywords:
                removal_positions.append(statement_index)
                removal_lines.append(removal.lineno)
            augment_assignment = _direct_assignment_call(
                statement, "ds", "ds.augment"
            )
            if augment_assignment is not None:
                augment_lines.append(augment_assignment.lineno)
            cut_assignment = _direct_assignment_call(
                statement, "ds", "ds.cut_to_len"
            )
            if cut_assignment is not None:
                cut_lines.append(cut_assignment.lineno)
            for node in _effective_scope_nodes([statement]):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_path(node.func) or ""
                if name == "ds.augment":
                    consumer_positions.append((statement_index, node.lineno))
                elif name == "ds.cut_to_len":
                    consumer_positions.append((statement_index, node.lineno))
                elif name.endswith("training_run"):
                    consumer_positions.append((statement_index, node.lineno))
        seed_before_train = bool(seed_indices and max(seed_indices) < train_index)
        return_after_train = bool(return_indices and min(return_indices) > train_index)
    else:
        seed_before_train = False
        return_after_train = False
    removal_before_consumers = bool(
        len(removal_positions) == 1
        and consumer_positions
        and removal_positions[0] < min(index for index, _line in consumer_positions)
    )
    contract_passed = bool(
        argument_names[:2] == ["formatter", "train"]
        and len(train_ifs) == 1
        and seed_before_train
        and return_after_train
        and len(removal_positions) == 1
        and augment_lines
        and cut_lines
        and removal_before_consumers
    )
    return {
        "cell": cell_index,
        "function_line": function.lineno,
        "train_branch_line": train_if_line,
        "remove_replies_lines": sorted(removal_lines),
        "augment_consumer_lines": sorted(set(augment_lines)),
        "cut_consumer_lines": sorted(set(cut_lines)),
        "seeded_from_arc_test_set_before_train": seed_before_train,
        "returns_dataset_after_train_branch": return_after_train,
        "remove_replies_before_train_consumers": removal_before_consumers,
        "contract_passed": contract_passed,
    }


def _training_consumer_contract(
    cell_index: int, function: ast.FunctionDef
) -> dict[str, Any]:
    parents = _parent_map(function)
    prepare_lines: list[int] = []
    consumer_lines: list[int] = []
    for node in _effective_scope_nodes(function.body):
        if not isinstance(node, ast.Call):
            continue
        name = _call_path(node.func) or ""
        if (
            name == "prepare_dataset"
            and _constant_bool_keyword(node, "train", True)
            and _assigned_to_name(node, parents, "dataset")
        ):
            prepare_lines.append(node.lineno)
        if name.endswith("training_run") and any(
            isinstance(argument, ast.Name) and argument.id == "dataset"
            for argument in node.args
        ):
            consumer_lines.append(node.lineno)
    flow_passed = bool(
        prepare_lines
        and consumer_lines
        and min(prepare_lines) < min(consumer_lines)
    )
    return {
        "cell": cell_index,
        "function_line": function.lineno,
        "prepare_train_dataset_lines": sorted(prepare_lines),
        "training_consumer_lines": sorted(consumer_lines),
        "contract_passed": flow_passed,
    }


def analyze_official_notebook(payload: bytes, path: str) -> dict[str, Any]:
    cells = _notebook_code_cells(payload, path)
    assignments: dict[str, list[tuple[str, int, int]]] = {
        "arc_challenge_file": [],
        "arc_solutions_file": [],
    }
    reply_calls: list[tuple[int, int, bool, bool, bool]] = []
    validation_calls: list[tuple[int, int, bool, bool]] = []
    pickle_load_lines: list[tuple[int, int]] = []
    local_only_true_lines: list[tuple[int, int]] = []
    trust_remote_false_lines: list[tuple[int, int]] = []
    from_file_lines: list[tuple[int, int]] = []
    challenge_load_lines: list[tuple[int, int]] = []
    prepare_definitions: list[tuple[int, ast.FunctionDef]] = []
    training_definitions: list[tuple[int, ast.FunctionDef]] = []
    for cell_index, _source, tree in cells:
        parents = _parent_map(tree)
        for statement in tree.body:
            for assignment_name in assignments:
                value = _assignment_value(statement, assignment_name)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    assignments[assignment_name].append(
                        (value.value, cell_index, statement.lineno)
                    )
            challenge_call = _direct_assignment_call(
                statement, "arc_test_set", "ArcDataset.from_file"
            )
            if (
                challenge_call is not None
                and len(challenge_call.args) == 1
                and _name_argument(challenge_call, 0, "arc_challenge_file")
            ):
                challenge_load_lines.append((cell_index, challenge_call.lineno))
            if isinstance(statement, ast.FunctionDef):
                if statement.name == "prepare_dataset":
                    prepare_definitions.append((cell_index, statement))
                elif statement.name == "start_training":
                    training_definitions.append((cell_index, statement))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_path(node.func) or ""
            if name.endswith("from_file"):
                from_file_lines.append((cell_index, node.lineno))
            if name.endswith("load_replies"):
                fake_only = _positive_fake_ancestor(node, parents)
                bound = bool(
                    name == "arc_test_set.load_replies"
                    and len(node.args) == 1
                    and _name_argument(node, 0, "arc_solutions_file")
                )
                reply_calls.append(
                    (
                        cell_index,
                        node.lineno,
                        fake_only,
                        bound,
                        not _constant_dead_ancestor(node, parents),
                    )
                )
            if name.endswith("validate_submission"):
                fake_only = _positive_fake_ancestor(node, parents)
                validation_calls.append(
                    (
                        cell_index,
                        node.lineno,
                        fake_only,
                        not _constant_dead_ancestor(node, parents),
                    )
                )
            if name == "pickle.load":
                pickle_load_lines.append((cell_index, node.lineno))
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            local_only = keywords.get("local_files_only")
            if isinstance(local_only, ast.Constant) and local_only.value is True:
                local_only_true_lines.append((cell_index, node.lineno))
            trust_remote = keywords.get("trust_remote_code")
            if isinstance(trust_remote, ast.Constant) and trust_remote.value is False:
                trust_remote_false_lines.append((cell_index, node.lineno))

    challenge_values = assignments["arc_challenge_file"]
    solution_values = assignments["arc_solutions_file"]
    challenge = challenge_values[0] if len(challenge_values) == 1 else None
    solutions = solution_values[0] if len(solution_values) == 1 else None
    prepare_contracts = [
        _prepare_dataset_contract(cell_index, function)
        for cell_index, function in prepare_definitions
    ]
    training_contracts = [
        _training_consumer_contract(cell_index, function)
        for cell_index, function in training_definitions
    ]
    prepare_contract = prepare_contracts[0] if len(prepare_contracts) == 1 else None
    training_contract = training_contracts[0] if len(training_contracts) == 1 else None
    challenge_suffix = "arc-agi_test_challenges.json"
    fake_solution_suffix = "arc-agi_training_solutions.json"
    challenge_only = bool(
        challenge
        and challenge[0].endswith(challenge_suffix)
        and solutions
        and solutions[0].endswith(fake_solution_suffix)
        and len(challenge_load_lines) == 1
        and reply_calls
        and all(item[2] and item[3] and item[4] for item in reply_calls)
        and prepare_contract
        and prepare_contract["contract_passed"]
        and training_contract
        and training_contract["contract_passed"]
        and validation_calls
        and all(item[2] and item[3] for item in validation_calls)
    )
    return {
        "path": path,
        "code_cell_count": len(cells),
        "challenge_assignment": list(challenge) if challenge else None,
        "fake_reply_assignment": list(solutions) if solutions else None,
        "from_file_lines": [list(item) for item in sorted(from_file_lines)],
        "challenge_load_lines": [list(item) for item in sorted(challenge_load_lines)],
        "load_replies_lines": [list(item) for item in sorted(reply_calls)],
        "remove_replies_lines": (
            [[prepare_contract["cell"], line] for line in prepare_contract["remove_replies_lines"]]
            if prepare_contract
            else []
        ),
        "prepare_dataset": prepare_contract,
        "training_consumer": training_contract,
        "validation_lines": [list(item) for item in sorted(validation_calls)],
        "pickle_load_lines": [list(item) for item in sorted(pickle_load_lines)],
        "local_files_only_true_lines": [list(item) for item in sorted(local_only_true_lines)],
        "trust_remote_code_false_lines": [list(item) for item in sorted(trust_remote_false_lines)],
        "true_test_challenge_only_candidate": challenge_only,
        "pickle_cache_read_present": bool(pickle_load_lines),
        "explicit_local_files_only_present": bool(local_only_true_lines),
        "explicit_trust_remote_code_false_present": bool(trust_remote_false_lines),
    }


def analyze_updated_notebook(payload: bytes, path: str) -> dict[str, Any]:
    cells = _notebook_code_cells(payload, path)
    all_text = "\n".join(source for _, source, _ in cells)
    return {
        "path": path,
        "code_cell_count": len(cells),
        "contains_test_challenge_reference": "arc-agi_test_challenges.json" in all_text,
        "contains_fake_only_reply_guard": (
            "if arc_test_set.is_fake: arc_test_set.load_replies(arc_solutions_file)" in all_text
        ),
        "role": "updated-local-candidate-not-official-submission",
    }


def verify_source_lock(payload: bytes, config: dict[str, Any]) -> dict[str, Any]:
    if hashlib.sha256(payload).hexdigest() != config["source_lock"]["sha256"]:
        raise ValueError("source lock file hash mismatch")
    document = strict_json(payload, "source lock")
    if not isinstance(document, dict) or not isinstance(document.get("sources"), dict):
        raise ValueError("source lock document shape mismatch")
    entry = document["sources"].get("architects-2024")
    if not isinstance(entry, dict):
        raise ValueError("source lock must contain one architects-2024 entry")
    observed = {key: entry.get(key) for key in ("url", "branch", "revision", "asset_subpath")}
    if observed != config["source_lock"]["entry"]:
        raise ValueError("source lock architects-2024 entry mismatch")
    return observed


def verify_checkpoint_report(payload: bytes, config: dict[str, Any]) -> dict[str, Any]:
    expected = config["prior_reports"]["checkpoint_integrity"]
    if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError("checkpoint integrity report hash mismatch")
    report = strict_json(payload, "checkpoint integrity report")
    if not isinstance(report, dict):
        raise ValueError("checkpoint integrity report must be an object")
    if report.get("method_id") != "architects-2024" or report.get("status") != expected["status"] or report.get("scope") != expected["scope"]:
        raise ValueError("checkpoint integrity report identity/status mismatch")
    snapshot = report.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("checkpoint integrity snapshot missing")
    if (
        snapshot.get("revision") != expected["revision"]
        or snapshot.get("file_count") != expected["file_count"]
        or snapshot.get("expected_total_bytes") != expected["total_bytes"]
        or snapshot.get("observed_total_bytes") != expected["total_bytes"]
    ):
        raise ValueError("checkpoint integrity snapshot summary mismatch")
    files = snapshot.get("files")
    if not isinstance(files, list) or len(files) != expected["file_count"]:
        raise ValueError("checkpoint integrity file inventory mismatch")
    by_path = {item.get("path"): item for item in files if isinstance(item, dict)}
    if len(by_path) != len(files):
        raise ValueError("checkpoint integrity file paths are invalid or duplicate")
    for key in ("model_file", "notice_file", "model_card_file"):
        wanted = expected[key]
        observed = by_path.get(wanted["path"])
        if observed != {
            "path": wanted["path"],
            "sha256": wanted["sha256"],
            "size_bytes": wanted["bytes"],
        }:
            raise ValueError(f"checkpoint integrity {key} mismatch")
    snapshot_path = snapshot.get("path")
    if not isinstance(snapshot_path, str) or expected["revision"] not in snapshot_path:
        raise ValueError("checkpoint snapshot path metadata mismatch")
    return {
        "report_sha256": expected["sha256"],
        "revision": snapshot["revision"],
        "file_count": snapshot["file_count"],
        "total_bytes": snapshot["observed_total_bytes"],
        "model_file": expected["model_file"],
        "notice_file": expected["notice_file"],
        "model_card_file": expected["model_card_file"],
        "checkpoint_bytes_reopened": False,
    }


def verify_preflight_report(payload: bytes, config: dict[str, Any]) -> dict[str, Any]:
    expected = config["prior_reports"]["resource_preflight"]
    if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError("resource preflight report hash mismatch")
    report = strict_json(payload, "resource preflight report")
    if not isinstance(report, dict) or report.get("method_id") != "architects-2024" or report.get("status") != "blocked":
        raise ValueError("resource preflight report identity/status mismatch")
    preflight = report.get("preflight")
    if not isinstance(preflight, dict) or not isinstance(preflight.get("gpu"), dict):
        raise ValueError("resource preflight fields missing")
    gpu = preflight["gpu"]
    if (
        preflight.get("minimum_free_vram_bytes") != expected["minimum_free_vram_bytes"]
        or gpu.get("free_memory_bytes") != expected["observed_free_vram_bytes"]
    ):
        raise ValueError("resource preflight memory binding mismatch")
    model = report.get("model")
    if not isinstance(model, dict) or model.get("revision") != config["prior_reports"]["checkpoint_integrity"]["revision"]:
        raise ValueError("resource preflight checkpoint revision mismatch")
    return {
        "report_sha256": expected["sha256"],
        "status": "blocked",
        "minimum_free_vram_bytes": preflight["minimum_free_vram_bytes"],
        "observed_free_vram_bytes": gpu["free_memory_bytes"],
        "gpu_query_repeated": False,
    }


def controls_record(ledger: ReadLedger) -> dict[str, Any]:
    source_byte_events = [
        entry
        for entry in ledger.entries
        if entry["read_started"]
        and entry["category"]
        in {"source_notebook", "source_python", "source_text", "source_worktree_binding"}
    ]
    return {
        "network_used": False,
        "gpu_used": False,
        "gpu_initialized": False,
        "upstream_imported": False,
        "upstream_executed": False,
        "solver_executed": False,
        "prediction_produced": False,
        "arc_solution_bytes_read": False,
        "checkpoint_bytes_read": False,
        "pickle_cache_bytes_read": False,
        "source_notebook_files_read": sum(
            str(entry["path"]).endswith(".ipynb") for entry in source_byte_events
        ),
        "source_python_files_read": sum(
            str(entry["path"]).endswith(".py") for entry in source_byte_events
        ),
        "retained_source_notebook_files_read": ledger.count("source_notebook"),
        "retained_source_python_files_read": ledger.count("source_python"),
        "prior_report_files_read": ledger.count("prior_report"),
        "source_worktree_binding_files_read": ledger.count(
            "source_worktree_binding"
        ),
        "source_worktree_binding_bytes_read": ledger.bytes_for(
            "source_worktree_binding"
        ),
        "git_local_config_files_read": ledger.count("git_local_config"),
        "git_head_files_read": ledger.count("git_head"),
        "explicit_read_file_count": len(ledger.entries),
        "explicit_read_bytes": sum(int(entry["bytes"]) for entry in ledger.entries),
        "explicit_read_bytes_scope": "all audit input byte reads; no Git subprocess or object-database read",
        "explicit_read_verified_count": sum(
            entry["outcome"] == "verified" for entry in ledger.entries
        ),
        "explicit_read_failed_count": sum(
            entry["outcome"] == "failed" for entry in ledger.entries
        ),
        "git_metadata_queries_used": ledger.git_query_count > 0,
        "git_query_count": ledger.git_query_count,
        "git_subprocess_used": False,
        "git_subprocess_object_database_reads_possible": False,
        "git_subprocess_object_database_bytes_measured": True,
        "git_worktree_content_command_used": False,
        "git_worktree_verification_used": ledger.git_worktree_verification_used,
        "git_may_read_locked_source_worktree_bytes": ledger.git_worktree_verification_used,
        "full_tracked_worktree_blob_binding_used": ledger.count(
            "source_worktree_binding"
        ) > 0,
    }


def base_record(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": "architects-2024",
        "runner": "scripts.audit_architects_2024_gates",
        "scope": "source-artifact-label-runtime-gate-audit-only",
        "counted_toward_smoke": False,
        "strict_runtime_promotion": False,
        "strict_runtime_promoted": False,
        "performance_claim": False,
        "performance_table_eligible": False,
        "prediction_produced": False,
        "solver_prediction_produced": False,
        "solver_gate_passed": False,
        "method_gate_status": "blocked",
        "evidence_scope": "blocker_audit",
        "evidence_scope_detail": "Hardened static source/artifact/label/runtime gate audit; no solver execution or prediction.",
        "fairness": {
            "arc_agi_1": "training-contaminated-ineligible-for-clean-main-board",
            "arc_agi_2": "potential-new-transfer-after-challenge-only-adapter-and-all-runtime-gates",
        },
    }


def failure_record(run_id: str, stage: str, error: BaseException, ledger: ReadLedger) -> dict[str, Any]:
    record = base_record(run_id)
    record.update(
        {
            "status": "failed",
            "stage": stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "method_gate": {"status": "blocked", "reason": "audit-failed"},
            "controls": controls_record(ledger),
            "read_ledger": ledger.entries,
            "claim_boundary": "The static audit failed; no method, smoke, prediction, or performance claim is permitted. No Git subprocess or Git object-database read was used.",
        }
    )
    return record


def _gate(identifier: str, status_value: str, detail: str, evidence: list[str], *, blocking: bool) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": status_value,
        "blocking": blocking,
        "detail": detail,
        "evidence": evidence,
    }


def run_static_audit(config_path: Path, run_id: str, ledger: ReadLedger) -> dict[str, Any]:
    config_payload = secure_read_absolute(config_path, "canonical_config", ledger)
    config = validate_config(strict_json(config_payload, "gate config"))

    source_lock_payload = secure_read_absolute(SOURCE_LOCK, "source_lock", ledger)
    source_lock_observation = verify_source_lock(source_lock_payload, config)
    checkpoint_payload = secure_read_absolute(
        CHECKPOINT_REPORT, "checkpoint_integrity_report", ledger
    )
    checkpoint_observation = verify_checkpoint_report(checkpoint_payload, config)
    preflight_payload = secure_read_absolute(
        PREFLIGHT_REPORT, "resource_preflight_report", ledger
    )
    preflight_observation = verify_preflight_report(preflight_payload, config)

    source_fd = open_absolute_directory(EXPECTED_SOURCE_ROOT)
    try:
        verify_directory_path_identity(EXPECTED_SOURCE_ROOT, source_fd)
        initial_root_signature = stat_signature(os.fstat(source_fd))
        initial_git_observation = verify_git_contract(
            source_fd, config, ledger, phase="initial"
        )
        initial_filesystem_inventory = closed_world_inventory(
            source_fd, config, ledger, phase="initial"
        )
        verify_directory_path_identity(EXPECTED_SOURCE_ROOT, source_fd)

        payloads: dict[str, bytes] = {}
        for declaration in config["source"]["read_files"]:
            payloads[declaration["path"]] = secure_read_relative(
                source_fd,
                declaration["path"],
                declaration["role"],
                declaration["bytes"],
                declaration["sha256"],
                ledger,
            )

        # Terminal source verification deliberately follows every retained
        # content read.  Fixed Git metadata is checked first, then all 18
        # worktree leaves are securely re-read and bound to their commit blob
        # OIDs, and finally the lexical source-root identity is revalidated.
        terminal_git_observation = verify_git_contract(
            source_fd, config, ledger, phase="terminal"
        )
        terminal_filesystem_inventory = closed_world_inventory(
            source_fd, config, ledger, phase="terminal"
        )
        verify_directory_path_identity(EXPECTED_SOURCE_ROOT, source_fd)
        terminal_root_signature = stat_signature(os.fstat(source_fd))
        if initial_git_observation != terminal_git_observation:
            raise RuntimeError("source Git metadata drifted during retained reads")
        if initial_filesystem_inventory != terminal_filesystem_inventory:
            raise RuntimeError("source closed-world inventory drifted during retained reads")
        if initial_root_signature != terminal_root_signature:
            raise RuntimeError("source root identity drifted during retained reads")
    finally:
        os.close(source_fd)

    git_observation = terminal_git_observation
    filesystem_inventory = terminal_filesystem_inventory

    python_trees: dict[str, ast.Module] = {}
    for path, payload in payloads.items():
        if path.endswith(".py"):
            python_trees[path] = parse_python(payload, path)
    if len(python_trees) != config["source"]["expected_python_file_count"]:
        raise RuntimeError("Python syntax audit count mismatch")
    python_bytes = sum(len(payloads[path]) for path in python_trees)
    if python_bytes != config["source"]["expected_python_bytes"]:
        raise RuntimeError("Python syntax audit byte count mismatch")

    official_path = config["notebooks"]["official"]["path"]
    updated_path = config["notebooks"]["updated"]["path"]
    official = analyze_official_notebook(payloads[official_path], official_path)
    updated = analyze_updated_notebook(payloads[updated_path], updated_path)
    if not official["true_test_challenge_only_candidate"]:
        raise RuntimeError("official notebook challenge-only candidate contract drifted")

    runner_observations = [
        analyze_local_runner(python_trees[path], path)
        for path in config["ast_contract"]["local_label_bearing_runners"]
    ]
    if not all(item["contract_passed"] for item in runner_observations):
        raise RuntimeError("local label-bearing runner contract drifted")
    contamination = analyze_contamination(
        python_trees[config["ast_contract"]["contamination_training_path"]]
    )
    if not contamination["arc1_training_contamination_confirmed"]:
        raise RuntimeError("ARC-AGI-1 training contamination contract drifted")
    model_loader = analyze_model_loader(
        python_trees[config["ast_contract"]["model_loader_path"]]
    )

    license_text = payloads[config["licenses"]["code"]["path"]].decode("utf-8")
    readme_text = payloads["README.md"].decode("utf-8")
    code_license_passed = (
        "Apache License" in license_text
        and "Version 2.0" in license_text
        and "Apache 2.0 license" in readme_text
    )
    if not code_license_passed:
        raise RuntimeError("source code license contract drifted")

    gates = [
        _gate(
            "source-lock",
            "passed",
            "Exact detached HEAD/local config, the canonical hard-coded tree manifest, and two full-byte closed-world worktree inventories matched without a Git subprocess.",
            ["configs/source_locks.json", ".git/HEAD", ".git/config", "18 tracked Git blob bindings"],
            blocking=False,
        ),
        _gate(
            "code-license",
            "passed",
            "The locked source repository carries an Apache-2.0 root license and matching README statement.",
            ["LICENSE.txt", "README.md"],
            blocking=False,
        ),
        _gate(
            "checkpoint-integrity-record",
            "passed",
            "The immutable prior integrity report binds the exact nine-file 4-bit snapshot; this audit did not reopen any snapshot file.",
            [config["prior_reports"]["checkpoint_integrity"]["path"]],
            blocking=False,
        ),
        _gate(
            "official-challenge-only-candidate",
            "passed",
            "The official Kaggle notebook statically loads test challenges, gates reply loading and validation behind is_fake, and removes replies before adaptation.",
            [official_path],
            blocking=False,
        ),
        _gate(
            "model-license-review",
            "blocked",
            "The model NOTICE is hash-bound through the prior integrity report, but NVIDIA Open Model License/NOTICE terms are separate from the Apache-2.0 code license and need redistribution review.",
            [config["prior_reports"]["checkpoint_integrity"]["path"]],
            blocking=True,
        ),
        _gate(
            "arc1-training-contamination",
            "blocked",
            "The locked full-model training runner loads ARC-AGI-1 evaluation solutions, moves test pairs into training, repeats them, and mixes them as arceval.",
            [config["ast_contract"]["contamination_training_path"]],
            blocking=True,
        ),
        _gate(
            "local-runner-label-firewall",
            "blocked",
            "Both local run_evaluation entry points unconditionally load ARC evaluation solutions and validate in the prediction process; neither is a strict challenge-only runtime.",
            config["ast_contract"]["local_label_bearing_runners"],
            blocking=True,
        ),
        _gate(
            "safe-offline-model-load",
            "blocked",
            "The reusable local model loader does not explicitly require local_files_only=True and trust_remote_code=False.",
            [config["ast_contract"]["model_loader_path"]],
            blocking=True,
        ),
        _gate(
            "pickle-cache-isolation",
            "blocked",
            "The official notebook contains pickle.load cache paths; a fresh, trusted, no-reuse cache stage has not been implemented.",
            [official_path],
            blocking=True,
        ),
        _gate(
            "dependency-environment-parity",
            "blocked",
            "The historical notebook stack and current local compatibility stack are not an exact, immutable dependency match.",
            [official_path, "configs/architects_2024_gate_v1.json"],
            blocking=True,
        ),
        _gate(
            "resource-capacity",
            "blocked",
            "The bound prior preflight observed free VRAM below the 10-GiB allocation gate; this audit did not repeat a GPU query.",
            [config["prior_reports"]["resource_preflight"]["path"]],
            blocking=True,
        ),
        _gate(
            "solver-prediction-and-parity",
            "blocked",
            "No checkpoint was opened, no solver ran, no prediction was produced, and no official/local parity run completed.",
            ["this static audit"],
            blocking=True,
        ),
    ]
    blocking_ids = [gate["id"] for gate in gates if gate["blocking"] and gate["status"] == "blocked"]
    if blocking_ids != list(EXPECTED_BLOCKERS):
        raise RuntimeError("blocking gate ordering/content mismatch")
    controls = controls_record(ledger)
    if controls["arc_solution_bytes_read"] or controls["checkpoint_bytes_read"] or controls["pickle_cache_bytes_read"]:
        raise RuntimeError("forbidden bytes were read")
    if (
        controls["retained_source_notebook_files_read"] != 2
        or controls["retained_source_python_files_read"] != 9
    ):
        raise RuntimeError("source read ledger count mismatch")
    if controls["prior_report_files_read"] != 2:
        raise RuntimeError("prior report read ledger count mismatch")

    record = base_record(run_id)
    record.update(
        {
            "status": "passed",
            "audit_status": "passed",
            "config_id": config["config_id"],
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
            "method_gate": {
                "status": "blocked",
                "blocking_gate_ids": blocking_ids,
                "passed_gate_ids": [gate["id"] for gate in gates if gate["status"] == "passed"],
            },
            "gate_summary": {
                "passed": sum(gate["status"] == "passed" for gate in gates),
                "blocked": sum(gate["status"] == "blocked" for gate in gates),
                "failed": sum(gate["status"] == "failed" for gate in gates),
            },
            "gates": gates,
            "source": {
                "path": os.fspath(EXPECTED_SOURCE_ROOT),
                "lock": source_lock_observation,
                "git": git_observation,
                "filesystem": filesystem_inventory,
                "initial_terminal_match": True,
                "terminal_verification_order": [
                    "detached-head-local-config-and-locked-manifest",
                    "closed-world-metadata-and-18-blob-binding",
                    "source-root-path-identity",
                ],
                "python_syntax": {
                    "file_count": len(python_trees),
                    "bytes": python_bytes,
                    "failures": [],
                },
            },
            "notebook_roles": {
                "official": {
                    **official,
                    "sha256": config["notebooks"]["official"]["sha256"],
                    "role": config["notebooks"]["official"]["role"],
                },
                "updated": {
                    **updated,
                    "sha256": config["notebooks"]["updated"]["sha256"],
                },
                "role_hashes_distinct": config["notebooks"]["official"]["sha256"] != config["notebooks"]["updated"]["sha256"],
            },
            "ast_observations": {
                "local_label_bearing_runners": runner_observations,
                "arc1_training_contamination": contamination,
                "model_loader": model_loader,
            },
            "artifact": checkpoint_observation,
            "resource_preflight": preflight_observation,
            "licenses": {
                "code": {**config["licenses"]["code"], "status": "passed"},
                "model": {**config["licenses"]["model"], "status": "blocked"},
            },
            "benchmark_policy": config["benchmark_policy"],
            "controls": controls,
            "read_ledger": ledger.entries,
            "validation": {
                "canonical_config_role_bound": True,
                "source_lock_exact": True,
                "source_closed_world": True,
                "source_clean": True,
                "source_all_tracked_bytes_match_commit_blobs": True,
                "source_initial_terminal_match": True,
                "git_status_not_invoked": True,
                "official_updated_roles_distinct": True,
                "official_true_test_challenge_only_candidate": True,
                "both_local_runners_unconditionally_read_solutions": True,
                "arc1_training_contamination_confirmed": True,
                "checkpoint_report_bound_without_checkpoint_read": True,
                "resource_report_bound_without_gpu_query": True,
                "method_remains_blocked": True,
                "counted_toward_smoke_false": True,
                "prediction_false": True,
            },
            "claim_boundary": "This is a static source/artifact/report gate audit. It is not a dependency, component, solver, strict-runtime, benchmark, or paper-parity run and reports no score. No Git subprocess or Git object-database read was used.",
        }
    )
    record["observation_sha256"] = canonical_sha256(
        {
            "gates": gates,
            "source": record["source"],
            "notebook_roles": record["notebook_roles"],
            "ast_observations": record["ast_observations"],
            "artifact": record["artifact"],
            "resource_preflight": record["resource_preflight"],
            "licenses": record["licenses"],
            "benchmark_policy": record["benchmark_policy"],
            "controls": record["controls"],
            "read_ledger": record["read_ledger"],
        }
    )
    return record


def execute_audit(config_path: Path, output_path: Path) -> tuple[int, dict[str, Any]]:
    validate_invocation_paths(config_path, output_path)
    output = create_fresh_output(output_path)
    ledger = ReadLedger()
    committed = False
    started = time.monotonic()
    started_at = utc_now()
    try:
        try:
            record = run_static_audit(config_path, output_path.name, ledger)
            exit_code = 0
        except BaseException as error:
            record = failure_record(output_path.name, "static-audit", error, ledger)
            exit_code = 1
        record["started_at_utc"] = started_at
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_time_seconds": max(0.0, time.monotonic() - started),
            "gpu_count": 0,
            "network_requests": 0,
            "checkpoint_bytes_read": 0,
            "arc_solution_bytes_read": 0,
            "predictions": 0,
        }
        write_json_no_clobber(output, record)
        committed = True
        return exit_code, record
    finally:
        output.close(record_committed=committed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        exit_code, _record = execute_audit(arguments.config, arguments.output_directory)
        return exit_code
    except BaseException as error:
        print(f"architects-2024 gate audit refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
