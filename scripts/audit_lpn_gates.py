#!/usr/bin/env python3
"""Run the static, metadata-first LPN source/artifact/label gate.

This auditor never imports or executes LPN, reads bundled ARC JSON, reads
Hydra YAML or notebooks, reads bytecode/checkpoints, initializes an
accelerator, or accesses the network.  A passing audit only confirms the
locked blocker observations; it cannot promote a solver smoke.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import stat
import subprocess
import sys
import time
import tokenize
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
CONFIG_ID = "lpn-source-artifact-data-label-gate-v1"
METHOD_ID = "lpn"
SCOPE = "source-artifact-data-label-gate-audit-only"
SOURCE_RELATIVE = "external/LPN"
SOURCE_LOCK_RELATIVE = "configs/source_locks.json"
CANONICAL_CONFIG_RELATIVE = "configs/lpn_gate_v1.json"
EXPECTED_SOURCE_LOCK_SHA256 = (
    "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
)
EXPECTED_REVISION = "0adfe56b86d2cba5ae5794edb02da6399a96d98a"
EXPECTED_COMMIT_TREE = "5793cc33c7a1166b5d9e0e61b5774c0ebe534c58"
EXPECTED_TRACKED_ALLOWLIST_SHA256 = (
    "b7c70e775d6122ad64ae356f011d751577bc91eb32492a56da388859d6fe0a0a"
)
EXPECTED_PYTHON_ALLOWLIST_SHA256 = (
    "6feb7b140351fdac33ac2c50a23760e19836fe814a6d7c5584c8e72b59dc511c"
)
EXPECTED_PYC_ALLOWLIST_SHA256 = (
    "6e0158fabfce7babe02018758dce888315d6bd8df5f5ea42783024cea980b5ca"
)
EXPECTED_SOURCE_LOCK_ENTRY = {
    "url": "https://github.com/clement-bonnet/lpn",
    "branch": "main",
    "revision": EXPECTED_REVISION,
    "asset_subpath": "sources/lpn",
}
EXPECTED_BLOCKER_IDS = [
    "checkpoint-artifact-provenance",
    "artifact-license",
    "artifact-config-binding",
    "label-process-firewall",
    "data-mount-isolation",
    "resource-capacity",
    "runtime-clean-staging",
]
EXPECTED_WANDB_ARTIFACT_IDS = [
    "TheThinker/ARC/faithful-dawn-316--checkpoint:v76",
    "TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5",
    "TheThinker/ARC/ominous-monster-839--checkpoint:v2",
    "TheThinker/ARC/playful-monkey-758--checkpoint:v1",
    "TheThinker/ARC/playful-sun-1060--checkpoint:v1",
    "TheThinker/ARC/solar-salad-1050--checkpoint:v18",
    "TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9",
]
EXPECTED_NEVER_READ_SUFFIXES = [
    ".json",
    ".yaml",
    ".yml",
    ".ipynb",
    ".pyc",
    ".msgpack",
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".bin",
]
EXPECTED_CONTROL_KEYS = {
    "network_allowed",
    "gpu_allowed",
    "upstream_import_allowed",
    "upstream_execution_allowed",
    "arc_data_byte_read_allowed",
    "yaml_byte_read_allowed",
    "notebook_byte_read_allowed",
    "checkpoint_byte_read_allowed",
    "pyc_byte_read_allowed",
    "solver_execution_allowed",
    "prediction_allowed",
}
CHECKPOINT_SUFFIXES = {
    ".msgpack",
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".bin",
}
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_RE = re.compile(
    r"TheThinker/ARC/[A-Za-z0-9_-]+--checkpoint:v[0-9]+"
)
MAX_CONFIG_BYTES = 2 * 1024 * 1024


class OutputPathError(ValueError):
    """The caller did not supply a fresh, safe output leaf."""


class FreshOutput:
    def __init__(
        self,
        descriptor: int,
        parent_descriptor: int,
        leaf: str,
        created: os.stat_result,
    ) -> None:
        self.descriptor = descriptor
        self.parent_descriptor = parent_descriptor
        self.leaf = leaf
        self.created_identity = (created.st_dev, created.st_ino, created.st_mode)

    def verify_leaf(self) -> None:
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


class ReadLedger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._config_binding: tuple[str, ...] | None = None
        self._retained_python_paths: set[str] = set()

    def bind_config(self, path: Path) -> None:
        binding = _absolute_parts(path)
        if binding != _absolute_parts(ROOT / CANONICAL_CONFIG_RELATIVE):
            raise ValueError("config reader is bound only to the canonical config path")
        if self._config_binding is not None and self._config_binding != binding:
            raise ValueError("read ledger config binding cannot be changed")
        self._config_binding = binding

    def bind_source_policy(self, retained_python_paths: set[str]) -> None:
        for path in retained_python_paths:
            safe_relative_path(path, "retained Python policy path")
            if not path.endswith(".py"):
                raise ValueError("retained Python policy contains a non-Python path")
            if PurePosixPath(path).suffix.lower() in EXPECTED_NEVER_READ_SUFFIXES:
                raise ValueError("retained Python policy intersects never-read suffixes")
        if self._retained_python_paths and self._retained_python_paths != retained_python_paths:
            raise ValueError("read ledger source policy cannot be changed")
        self._retained_python_paths = set(retained_python_paths)

    def authorize_absolute(self, path: Path, role: str) -> str:
        parts = _absolute_parts(path)
        if role == "canonical_config":
            if self._config_binding is None:
                raise ValueError("config reader role is not bound")
            if parts != self._config_binding:
                raise ValueError("config reader role/path binding mismatch")
            return "gate_config"
        if role == "source_lock":
            if parts != _absolute_parts(ROOT / SOURCE_LOCK_RELATIVE):
                raise ValueError("source-lock reader is not bound to the canonical path")
            return "source_lock"
        raise ValueError(f"untrusted absolute reader role: {role}")

    def authorize_relative(self, path: str, role: str) -> str:
        safe_relative_path(path, "verified relative reader path")
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in EXPECTED_NEVER_READ_SUFFIXES:
            raise ValueError(f"metadata-only suffix may never enter a reader: {path}")
        if role == "retained_python":
            if path not in self._retained_python_paths:
                raise ValueError(f"retained Python reader path is not allowlisted: {path}")
            return "retained_python"
        if role == "code_license" and path == "LICENSE":
            return "code_license"
        raise ValueError(f"untrusted relative reader role/path: {role}:{path}")

    def add_absolute(
        self, path: Path, role: str, size: int, digest: str, display: str
    ) -> None:
        category = self.authorize_absolute(path, role)
        self._append(display, role, category, size, digest)

    def add_relative(self, path: str, role: str, size: int, digest: str) -> None:
        category = self.authorize_relative(path, role)
        self._append(path, role, category, size, digest)

    def _append(
        self,
        path: str,
        role: str,
        category: str,
        size: int,
        digest: str,
    ) -> None:
        self.records.append(
            {
                "path": path,
                "role": role,
                "category": category,
                "bytes": size,
                "sha256": digest,
            }
        )

    def count(self, category: str) -> int:
        return sum(item["category"] == category for item in self.records)


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


def require_secure_open_flags() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC"):
        if not hasattr(os, name) or not isinstance(getattr(os, name), int):
            raise RuntimeError(f"required secure-open flag is unavailable: {name}")


def directory_flags() -> int:
    require_secure_open_flags()
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def regular_file_flags() -> int:
    require_secure_open_flags()
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a contained POSIX relative path")
    if path.as_posix() != value:
        raise ValueError(f"{field} must be canonical POSIX syntax")
    return value


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if set(value) != expected:
        raise ValueError(
            f"{field} keys mismatch: expected {sorted(expected)}, found {sorted(value)}"
        )
    return value


def nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def lowercase_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase Git object id")
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
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
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON: {error}") from error


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


def public_metadata(value: os.stat_result, kind: str) -> dict[str, Any]:
    return {
        "type": kind,
        "mode": format(stat.S_IMODE(value.st_mode), "04o"),
        "bytes": value.st_size if kind == "file" else None,
    }


def _absolute_parts(path: Path) -> tuple[str, ...]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if any(part == ".." for part in absolute.parts):
        raise ValueError(f"path traversal is forbidden: {path}")
    if not absolute.is_absolute():
        raise ValueError(f"absolute path required: {path}")
    return absolute.parts


def open_absolute_directory(path: Path) -> int:
    parts = _absolute_parts(path)
    descriptor = os.open("/", directory_flags())
    try:
        for part in parts[1:]:
            next_descriptor = os.open(
                part, directory_flags(), dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ValueError(f"directory is not a non-symlink directory: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_absolute_parent(path: Path) -> tuple[int, str]:
    parts = _absolute_parts(path)
    if len(parts) < 2 or parts[-1] in {"", ".", ".."}:
        raise ValueError(f"unsafe leaf path: {path}")
    parent = Path(*parts[:-1])
    return open_absolute_directory(parent), parts[-1]


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
        if total > max_bytes:
            raise ValueError("verified file grew beyond maximum size")
    after = os.fstat(descriptor)
    if stat_signature(before) != stat_signature(after):
        raise RuntimeError("file changed while verified bytes were read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise RuntimeError("verified read length does not match fstat size")
    return payload, after


def secure_read_absolute(
    path: Path,
    *,
    max_bytes: int,
    role: str,
    ledger: ReadLedger,
    display: str | None = None,
) -> tuple[bytes, os.stat_result]:
    ledger.authorize_absolute(path, role)
    parent_fd, leaf = open_absolute_parent(path)
    descriptor: int | None = None
    try:
        before_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode) or stat.S_ISLNK(before_path.st_mode):
            raise ValueError(f"verified path is not a regular non-symlink file: {path}")
        if before_path.st_nlink != 1:
            raise ValueError(f"verified path must have exactly one hard link: {path}")
        descriptor = os.open(leaf, regular_file_flags(), dir_fd=parent_fd)
        before_fd = os.fstat(descriptor)
        if stat_signature(before_path) != stat_signature(before_fd):
            raise RuntimeError(f"file identity changed before verified read: {path}")
        payload, after_fd = _read_fd_stable(descriptor, before_fd, max_bytes)
        after_path = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat_signature(after_fd) != stat_signature(after_path):
            raise RuntimeError(f"file identity changed after verified read: {path}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    digest = hashlib.sha256(payload).hexdigest()
    ledger.add_absolute(path, role, len(payload), digest, display or str(path))
    return payload, after_fd


def lexical_path(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def lexical_within(path: Path, parent: Path) -> bool:
    path_parts = _absolute_parts(path)
    parent_parts = _absolute_parts(parent)
    return path_parts[: len(parent_parts)] == parent_parts


def validate_config_location(
    config_path: Path,
    output_path: Path,
) -> str:
    source = ROOT / SOURCE_RELATIVE
    if lexical_within(config_path, source):
        raise ValueError("config path may not be inside the LPN source tree")
    if lexical_within(config_path, output_path):
        raise ValueError("config path may not be inside the output directory")
    if config_path.suffix.lower() in CHECKPOINT_SUFFIXES:
        raise ValueError("config path may not be a checkpoint-like path")
    if _absolute_parts(config_path) != _absolute_parts(
        ROOT / CANONICAL_CONFIG_RELATIVE
    ):
        raise ValueError(
            "production config path must equal configs/lpn_gate_v1.json"
        )
    return "canonical_config"


def create_fresh_output(path: Path) -> FreshOutput:
    source = ROOT / SOURCE_RELATIVE
    configs = ROOT / "configs"
    if lexical_within(path, source) or lexical_within(path, configs):
        raise OutputPathError("output directory may not be inside source or configs")
    parent_fd, leaf = open_absolute_parent(path)
    output_fd: int | None = None
    try:
        try:
            os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise OutputPathError("output path must not exist") from error
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise OutputPathError("output path has a symlink/non-directory component") from error
            raise
        path_info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(path_info.st_mode) or stat.S_ISLNK(path_info.st_mode):
            raise OutputPathError("created output leaf is not a directory")
        output_fd = os.open(leaf, directory_flags(), dir_fd=parent_fd)
        fd_info = os.fstat(output_fd)
        if stat_signature(path_info) != stat_signature(fd_info):
            raise OutputPathError("created output leaf raced before pinning")
        result = FreshOutput(output_fd, parent_fd, leaf, fd_info)
        result.verify_leaf()
        return result
    except BaseException:
        if output_fd is not None:
            os.close(output_fd)
        os.close(parent_fd)
        raise


def write_json_no_clobber(output: FreshOutput, value: dict[str, Any]) -> None:
    output.verify_leaf()
    output_fd = output.descriptor
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = f".run.json.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor: int | None = None
    owned_identity: tuple[int, int, int, int] | None = None
    linked = False
    record_committed = False

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_size,
        )

    def owned_path(name: str) -> bool:
        if owned_identity is None:
            return False
        try:
            observed = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return stat.S_ISREG(observed.st_mode) and identity(observed) == owned_identity

    def unlink_owned(name: str) -> bool:
        if not owned_path(name):
            return False
        os.unlink(name, dir_fd=output_fd)
        return True

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
            observed, after = _read_fd_stable(descriptor, before, len(payload))
        except (OSError, ValueError, RuntimeError):
            return False
        return identity(after) == owned_identity and observed == payload

    def committed_record_is_valid() -> bool:
        return (
            leaf_matches()
            and owned_path("run.json")
            and descriptor_payload_matches()
        )

    def rollback_owned_links() -> None:
        # The pinned directory was created fresh and empty by this process.
        # Unlinking these two exact reserved names never follows a symlink or
        # removes an external inode; it also clears a raced replacement link.
        for name in ("run.json", temporary):
            try:
                os.unlink(name, dir_fd=output_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            os.fsync(output_fd)
        except OSError:
            pass

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
                raise OSError("short write while publishing run record")
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
        if not owned_path(temporary) or not descriptor_payload_matches():
            raise OutputPathError("temporary report file changed before linking")
        output.verify_leaf()
        os.link(
            temporary,
            "run.json",
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
            follow_symlinks=False,
        )
        linked = True
        # The link is not considered committed until the directory is synced,
        # the requested leaf still names the pinned directory, and run.json is
        # still the exact descriptor-backed payload.
        try:
            if not owned_path(temporary) or not owned_path("run.json"):
                raise OutputPathError(
                    "linked report identity does not match its descriptor"
                )
            if not descriptor_payload_matches():
                raise OutputPathError(
                    "linked report bytes do not match the serialized record"
                )
            os.fsync(output_fd)
            if not committed_record_is_valid():
                raise OutputPathError("report path identity changed before commit")
        except BaseException:
            rollback_owned_links()
            if committed_record_is_valid():
                # Rollback was refused but the requested path still exposes the
                # exact record.  Returning preserves result/file consistency.
                record_committed = True
                return
            linked = False
            raise
        record_committed = True

        # run.json is now the commit point.  Temporary-name cleanup and its
        # follow-up sync are best effort and may not reinterpret that record.
        for _ in range(2):
            try:
                if not unlink_owned(temporary):
                    break
                break
            except OSError:
                continue
        try:
            os.fsync(output_fd)
        except OSError:
            pass
        try:
            if not committed_record_is_valid():
                raise OutputPathError("report path identity changed after commit")
        except BaseException:
            rollback_owned_links()
            if committed_record_is_valid():
                record_committed = True
                return
            record_committed = False
            linked = False
            raise
    finally:
        close_error: OSError | None = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                close_error = error
        if not linked:
            try:
                unlink_owned(temporary)
            except OSError:
                pass
        if close_error is not None and not record_committed:
            raise close_error


def validate_file_declarations(
    value: Any,
    *,
    field: str,
    keys: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    paths: list[str] = []
    for index, item in enumerate(value):
        item = exact_keys(item, keys, f"{field}[{index}]")
        paths.append(safe_relative_path(item["path"], f"{field}[{index}].path"))
        nonnegative_int(item["bytes"], f"{field}[{index}].bytes")
        if "sha256" in keys:
            lowercase_sha256(item["sha256"], f"{field}[{index}].sha256")
        if "blob_oid" in keys:
            git_sha(item["blob_oid"], f"{field}[{index}].blob_oid")
        if "mode" in keys and item["mode"] != "100644":
            raise ValueError(f"{field}[{index}].mode must equal 100644")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{field} paths must be unique and sorted")
    return value


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
            "metadata_only_policy",
            "ast_contract",
            "artifact",
            "license",
            "controls",
        },
        "config",
    )
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("config.schema_version must equal 1")
    if config["config_id"] != CONFIG_ID:
        raise ValueError("unexpected config.config_id")
    if config["method_id"] != METHOD_ID or config["scope"] != SCOPE:
        raise ValueError("unexpected method or scope")
    if config["counted_toward_smoke"] is not False:
        raise ValueError("config.counted_toward_smoke must be false")
    config_read_policy = exact_keys(
        config["config_read_policy"],
        {"canonical_path", "alternate_paths_allowed"},
        "config.config_read_policy",
    )
    if config_read_policy != {
        "canonical_path": CANONICAL_CONFIG_RELATIVE,
        "alternate_paths_allowed": False,
    }:
        raise ValueError("config.config_read_policy mismatch")
    if config["expected_blocker_ids"] != EXPECTED_BLOCKER_IDS:
        raise ValueError("config.expected_blocker_ids mismatch")

    source_lock = exact_keys(
        config["source_lock"], {"path", "sha256", "entry"}, "config.source_lock"
    )
    if source_lock != {
        "path": SOURCE_LOCK_RELATIVE,
        "sha256": EXPECTED_SOURCE_LOCK_SHA256,
        "entry": EXPECTED_SOURCE_LOCK_ENTRY,
    }:
        raise ValueError("config.source_lock is not the hardcoded contract")

    source = exact_keys(
        config["source"],
        {
            "repository_path",
            "repository_url",
            "expected_revision",
            "expected_commit_tree",
            "expected_tracked_file_count",
            "tracked_allowlist_sha256",
            "tracked_files",
            "expected_python_file_count",
            "expected_python_bytes",
            "python_allowlist_sha256",
            "retained_python",
            "ignored_pyc_allowlist_sha256",
            "ignored_pyc",
            "opaque_directories",
            "expected_unknown_entry_count",
        },
        "config.source",
    )
    fixed_source = {
        "repository_path": SOURCE_RELATIVE,
        "repository_url": EXPECTED_SOURCE_LOCK_ENTRY["url"],
        "expected_revision": EXPECTED_REVISION,
        "expected_commit_tree": EXPECTED_COMMIT_TREE,
        "expected_tracked_file_count": 79,
        "tracked_allowlist_sha256": EXPECTED_TRACKED_ALLOWLIST_SHA256,
        "expected_python_file_count": 21,
        "expected_python_bytes": 785886,
        "python_allowlist_sha256": EXPECTED_PYTHON_ALLOWLIST_SHA256,
        "ignored_pyc_allowlist_sha256": EXPECTED_PYC_ALLOWLIST_SHA256,
        "opaque_directories": [".git"],
        "expected_unknown_entry_count": 0,
    }
    for key in (
        "expected_tracked_file_count",
        "expected_python_file_count",
        "expected_python_bytes",
        "expected_unknown_entry_count",
    ):
        nonnegative_int(source[key], f"config.source.{key}")
    for key, expected in fixed_source.items():
        if source[key] != expected:
            raise ValueError(f"config.source.{key} is not the hardcoded contract")
    tracked = validate_file_declarations(
        source["tracked_files"],
        field="config.source.tracked_files",
        keys={"path", "mode", "blob_oid", "bytes"},
    )
    retained = validate_file_declarations(
        source["retained_python"],
        field="config.source.retained_python",
        keys={"path", "bytes", "sha256"},
    )
    pyc = validate_file_declarations(
        source["ignored_pyc"],
        field="config.source.ignored_pyc",
        keys={"path", "bytes", "mode"},
    )
    if len(tracked) != 79 or canonical_sha256(tracked) != EXPECTED_TRACKED_ALLOWLIST_SHA256:
        raise ValueError("config.source.tracked_files allowlist mismatch")
    if len(retained) != 21 or canonical_sha256(retained) != EXPECTED_PYTHON_ALLOWLIST_SHA256:
        raise ValueError("config.source.retained_python allowlist mismatch")
    if len(pyc) != 14 or canonical_sha256(pyc) != EXPECTED_PYC_ALLOWLIST_SHA256:
        raise ValueError("config.source.ignored_pyc allowlist mismatch")
    tracked_map = {item["path"]: item for item in tracked}
    retained_paths = {item["path"] for item in retained}
    if retained_paths != {path for path in tracked_map if path.endswith(".py")}:
        raise ValueError("retained Python paths do not equal tracked Python paths")
    if sum(item["bytes"] for item in retained) != 785886:
        raise ValueError("retained Python byte total mismatch")
    for item in retained:
        if tracked_map[item["path"]]["bytes"] != item["bytes"]:
            raise ValueError("retained Python metadata disagrees with tracked tree")
    if {item["path"] for item in pyc} & set(tracked_map):
        raise ValueError("ignored pyc and tracked paths overlap")

    metadata = exact_keys(
        config["metadata_only_policy"],
        {"upstream_never_read_suffixes", "retained_non_python_paths"},
        "config.metadata_only_policy",
    )
    if metadata != {
        "upstream_never_read_suffixes": EXPECTED_NEVER_READ_SUFFIXES,
        "retained_non_python_paths": ["LICENSE"],
    }:
        raise ValueError("config.metadata_only_policy mismatch")

    ast_contract = exact_keys(
        config["ast_contract"],
        {
            "evaluator_path",
            "train_path",
            "evaluate_checkpoint_path",
            "challenge_only_candidate",
            "label_bearing_runner",
        },
        "config.ast_contract",
    )
    if ast_contract != {
        "evaluator_path": "src/evaluator.py",
        "train_path": "src/train.py",
        "evaluate_checkpoint_path": "src/evaluate_checkpoint.py",
        "challenge_only_candidate": "Evaluator.json_submission",
        "label_bearing_runner": "Trainer.test_json_submission",
    }:
        raise ValueError("config.ast_contract mismatch")

    artifact = exact_keys(
        config["artifact"],
        {
            "expected_locked_source_tree_checkpoint_count",
            "source_declared_wandb_artifact_ids",
            "remote_verification",
            "verification_status",
        },
        "config.artifact",
    )
    nonnegative_int(
        artifact["expected_locked_source_tree_checkpoint_count"],
        "config.artifact.expected_locked_source_tree_checkpoint_count",
    )
    if artifact != {
        "expected_locked_source_tree_checkpoint_count": 0,
        "source_declared_wandb_artifact_ids": EXPECTED_WANDB_ARTIFACT_IDS,
        "remote_verification": "forbidden",
        "verification_status": "source_declared_unverified",
    }:
        raise ValueError("config.artifact mismatch")

    license_config = exact_keys(
        config["license"],
        {"path", "bytes", "sha256", "code_identifier", "artifact_license_status"},
        "config.license",
    )
    if license_config != {
        "path": "LICENSE",
        "bytes": 11345,
        "sha256": "291a774b261f46090a7d98b923f13debc37cf9c0cbbc38987e3fe4b4d3168c80",
        "code_identifier": "Apache-2.0",
        "artifact_license_status": "blocked-unverified",
    }:
        raise ValueError("config.license mismatch")
    controls = exact_keys(config["controls"], EXPECTED_CONTROL_KEYS, "config.controls")
    if any(value is not False for value in controls.values()):
        raise ValueError("all config.controls values must be false")
    return config


def expected_directories(config: dict[str, Any]) -> set[str]:
    directories: set[str] = {".git"}
    paths = [item["path"] for item in config["source"]["tracked_files"]]
    paths.extend(item["path"] for item in config["source"]["ignored_pyc"])
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent.as_posix() not in {"", "."}:
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _scan_directory(
    descriptor: int,
    relative: str,
    *,
    expected_files: dict[str, dict[str, Any]],
    expected_dirs: set[str],
    opaque_dirs: set[str],
    snapshot: dict[str, dict[str, Any]],
) -> None:
    with os.scandir(descriptor) as iterator:
        entries = sorted(iterator, key=lambda item: item.name)
    for entry in entries:
        if entry.name in {"", ".", ".."} or "/" in entry.name:
            raise ValueError("unsafe directory entry name")
        path = f"{relative}/{entry.name}" if relative else entry.name
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"closed-world inventory rejects symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            if path not in expected_dirs:
                raise ValueError(f"closed-world inventory rejects unknown directory: {path}")
            child_fd = os.open(entry.name, directory_flags(), dir_fd=descriptor)
            try:
                opened = os.fstat(child_fd)
                if stat_signature(info) != stat_signature(opened):
                    raise RuntimeError(f"directory raced during inventory: {path}")
                snapshot[path] = {
                    "kind": "directory",
                    "signature": stat_signature(opened),
                    "metadata": public_metadata(opened, "directory"),
                }
                if path not in opaque_dirs:
                    _scan_directory(
                        child_fd,
                        path,
                        expected_files=expected_files,
                        expected_dirs=expected_dirs,
                        opaque_dirs=opaque_dirs,
                        snapshot=snapshot,
                    )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"closed-world inventory rejects nonregular entry: {path}")
        if path not in expected_files:
            suffix = PurePosixPath(path).suffix.lower()
            kind = "checkpoint-like" if suffix in CHECKPOINT_SUFFIXES else "file"
            raise ValueError(f"closed-world inventory rejects unknown {kind}: {path}")
        expected = expected_files[path]
        if info.st_nlink != 1:
            raise ValueError(f"closed-world inventory rejects hard-linked file: {path}")
        if info.st_size != expected["bytes"]:
            raise ValueError(f"closed-world inventory size mismatch: {path}")
        if stat.S_IMODE(info.st_mode) != 0o644:
            raise ValueError(f"closed-world inventory mode mismatch: {path}")
        snapshot[path] = {
            "kind": "file",
            "signature": stat_signature(info),
            "metadata": public_metadata(info, "file"),
        }


def closed_world_inventory(root_fd: int, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracked = {item["path"]: item for item in config["source"]["tracked_files"]}
    pyc = {item["path"]: item for item in config["source"]["ignored_pyc"]}
    expected_files = {**tracked, **pyc}
    expected_dirs = expected_directories(config)
    root_info = os.fstat(root_fd)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("LPN root is not a non-symlink directory")
    snapshot: dict[str, dict[str, Any]] = {
        "": {
            "kind": "directory",
            "signature": stat_signature(root_info),
            "metadata": public_metadata(root_info, "directory"),
        }
    }
    _scan_directory(
        root_fd,
        "",
        expected_files=expected_files,
        expected_dirs=expected_dirs,
        opaque_dirs=set(config["source"]["opaque_directories"]),
        snapshot=snapshot,
    )
    observed_files = {path for path, value in snapshot.items() if value["kind"] == "file"}
    observed_dirs = {path for path, value in snapshot.items() if value["kind"] == "directory" and path}
    if observed_files != set(expected_files):
        raise ValueError("closed-world file set mismatch")
    if observed_dirs != expected_dirs:
        raise ValueError("closed-world directory set mismatch")
    readable = {item["path"] for item in config["source"]["retained_python"]}
    if "LICENSE" in observed_files:
        readable.add("LICENSE")
    unread = observed_files - readable
    readable_inodes = {
        snapshot[path]["signature"][:2] for path in readable
    }
    unread_inodes = {snapshot[path]["signature"][:2] for path in unread}
    if readable_inodes & unread_inodes:
        raise ValueError("readable and metadata-only files share an inode")
    return snapshot


def secure_read_relative(
    root_fd: int,
    relative: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    snapshot: dict[str, dict[str, Any]],
    role: str,
    ledger: ReadLedger,
) -> bytes:
    ledger.authorize_relative(relative, role)
    parts = PurePosixPath(relative).parts
    directory_fd = os.dup(root_fd)
    descriptor: int | None = None
    try:
        current_parts: list[str] = []
        for part in parts[:-1]:
            current_parts.append(part)
            path = "/".join(current_parts)
            next_fd = os.open(part, directory_flags(), dir_fd=directory_fd)
            opened = os.fstat(next_fd)
            if stat_signature(opened) != snapshot[path]["signature"]:
                os.close(next_fd)
                raise RuntimeError(f"parent directory changed before read: {relative}")
            os.close(directory_fd)
            directory_fd = next_fd
        leaf = parts[-1]
        before_path = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat_signature(before_path) != snapshot[relative]["signature"]:
            raise RuntimeError(f"retained file changed after inventory: {relative}")
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise ValueError(f"retained path is not a single-link regular file: {relative}")
        descriptor = os.open(leaf, regular_file_flags(), dir_fd=directory_fd)
        before_fd = os.fstat(descriptor)
        if stat_signature(before_fd) != stat_signature(before_path):
            raise RuntimeError(f"retained file raced before read: {relative}")
        payload, after_fd = _read_fd_stable(descriptor, before_fd, expected_bytes)
        after_path = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat_signature(after_fd) != stat_signature(after_path):
            raise RuntimeError(f"retained file raced after read: {relative}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
    if len(payload) != expected_bytes:
        raise ValueError(f"retained file byte count mismatch: {relative}")
    digest = hashlib.sha256(payload).hexdigest()
    ledger.add_relative(relative, role, len(payload), digest)
    if digest != expected_sha256:
        raise ValueError(f"retained file SHA-256 mismatch: {relative}")
    return payload


def parse_python(payload: bytes, path: str) -> ast.Module:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
        text = payload.decode(encoding)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError(f"Python encoding failure for {path}: {error}") from error
    try:
        return ast.parse(text, filename=path)
    except SyntaxError as error:
        raise ValueError(f"Python syntax failure for {path}: {error}") from error


def _git(root_fd: int, *arguments: str) -> bytes:
    allowed = {
        ("rev-parse", "--verify", "HEAD^{commit}"),
        ("rev-parse", "--verify", "HEAD^{tree}"),
        ("ls-tree", "-r", "-l", "-z", "--full-tree", "HEAD"),
    }
    if tuple(arguments) not in allowed:
        raise ValueError("unapproved Git command")
    executable = "/usr/bin/git"
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    completed = subprocess.run(
        [
            executable,
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            f"/proc/self/fd/{root_fd}",
            *arguments,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        env=environment,
        pass_fds=(root_fd,),
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "local Git metadata command failed")
    return completed.stdout


def parse_ls_tree(payload: bytes) -> list[dict[str, Any]]:
    if not payload.endswith(b"\0"):
        raise ValueError("Git ls-tree output is not NUL terminated")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(payload[:-1].split(b"\0")):
        try:
            metadata, raw_path = item.split(b"\t", 1)
            mode, kind, oid, size = metadata.split(b" ", 3)
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"malformed Git ls-tree entry {index}") from error
        if kind != b"blob":
            raise ValueError(f"non-blob tracked entry: {path}")
        result.append(
            {
                "path": safe_relative_path(path, f"Git tree path {index}"),
                "mode": mode.decode("ascii"),
                "blob_oid": oid.decode("ascii"),
                "bytes": int(size),
            }
        )
    return result


def verify_git_contract(root_fd: int, config: dict[str, Any]) -> dict[str, Any]:
    revision = _git(root_fd, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    tree = _git(root_fd, "rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    inventory = parse_ls_tree(
        _git(root_fd, "ls-tree", "-r", "-l", "-z", "--full-tree", "HEAD")
    )
    if revision != EXPECTED_REVISION:
        raise ValueError("observed LPN revision mismatch")
    if tree != EXPECTED_COMMIT_TREE:
        raise ValueError("observed LPN commit tree mismatch")
    if inventory != config["source"]["tracked_files"]:
        raise ValueError("observed Git tree allowlist mismatch")
    return {
        "observed_revision": revision,
        "observed_commit_tree": tree,
        "tracked_file_count": len(inventory),
        "tracked_allowlist_sha256": canonical_sha256(inventory),
    }


def _string_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one class {name}")
    return matches[0]


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one method {class_node.name}.{name}")
    return matches[0]


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one function {name}")
    return matches[0]


def analyze_evaluator(tree: ast.Module) -> dict[str, Any]:
    method = _find_method(_find_class(tree, "Evaluator"), "json_submission")
    arguments = [argument.arg for argument in method.args.args]
    expected_arguments = [
        "self",
        "challenges",
        "params",
        "only_n_tasks",
        "overfit_task",
        "progress_bar",
        "key",
        "train",
    ]
    loops: dict[str, ast.For] = {}
    for node in ast.walk(method):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Subscript):
            continue
        if isinstance(node.iter.value, ast.Name) and node.iter.value.id == "task":
            key = _string_key(node.iter.slice)
            if key in {"train", "test"}:
                if key in loops:
                    raise ValueError(f"duplicate task[{key!r}] loop")
                loops[key] = node
    if set(loops) != {"train", "test"}:
        raise ValueError("Evaluator.json_submission train/test loops not found")

    def example_keys(loop: ast.For) -> set[str]:
        return {
            key
            for node in ast.walk(loop)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "example"
            and (key := _string_key(node.slice)) is not None
        }

    train_keys = example_keys(loops["train"])
    test_keys = example_keys(loops["test"])
    function_forbidden_calls: list[dict[str, Any]] = []
    test_forbidden_calls: list[dict[str, Any]] = []
    forbidden_names = {
        "open",
        "eval",
        "exec",
        "compile",
        "json.load",
        "json.loads",
        "evaluate_generations",
        "score",
        "wandb.init",
        "use_artifact",
        "download",
        "read",
        "read_bytes",
        "read_text",
    }
    test_call_ids = {
        id(node) for node in ast.walk(loops["test"]) if isinstance(node, ast.Call)
    }
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        name = _call_path(node.func) or ""
        if name in forbidden_names or name.rsplit(".", 1)[-1] in forbidden_names or name.startswith("wandb."):
            observation = {"line": node.lineno, "call": name}
            function_forbidden_calls.append(observation)
            if id(node) in test_call_ids:
                test_forbidden_calls.append(observation)
    test_output_string_lines = sorted(
        node.lineno
        for node in ast.walk(loops["test"])
        if isinstance(node, ast.Subscript) and _string_key(node.slice) == "output"
    )
    attempt_keys: set[str] = set()
    for node in ast.walk(loops["test"]):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "attempts" for target in node.targets):
            continue
        if isinstance(node.value, ast.Dict):
            attempt_keys.update(
                key for item in node.value.keys if (key := _string_key(item)) is not None
            )
    return_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "results"
    ]
    forbidden_reference_lines = sorted(
        {node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Name)
        and node.id
        in {
            "solutions",
            "solution",
            "json_solutions_file",
            "answers",
            "answer_file",
            "wandb",
        }}
    )
    passed = all(
        [
            arguments == expected_arguments,
            train_keys == {"input", "output"},
            test_keys == {"input"},
            not test_output_string_lines,
            not function_forbidden_calls,
            attempt_keys == {"attempt_1", "attempt_2"},
            bool(return_lines),
            not forbidden_reference_lines,
        ]
    )
    return {
        "candidate": "Evaluator.json_submission",
        "arguments": arguments,
        "train_example_keys": sorted(train_keys),
        "test_example_keys": sorted(test_keys),
        "test_output_subscript_lines": test_output_string_lines,
        "function_forbidden_calls": function_forbidden_calls,
        "test_forbidden_calls": test_forbidden_calls,
        "attempt_keys": sorted(attempt_keys),
        "return_results_lines": return_lines,
        "forbidden_reference_lines": forbidden_reference_lines,
        "challenge_only_candidate_detected": passed,
    }


def _open_load_lines(method: ast.FunctionDef, file_name: str, target_name: str) -> tuple[int, int] | None:
    for node in ast.walk(method):
        if not isinstance(node, ast.With):
            continue
        open_lines = []
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and _call_path(call.func) == "open"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == file_name
            ):
                open_lines.append(call.lineno)
        if not open_lines:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == target_name for target in child.targets):
                continue
            if isinstance(child.value, ast.Call) and _call_path(child.value.func) == "json.load":
                return min(open_lines), child.lineno
    return None


def _assigned_call_line(method: ast.FunctionDef, target_name: str, call_suffix: str) -> int | None:
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == target_name for target in node.targets):
            continue
        if isinstance(node.value, ast.Call) and (_call_path(node.value.func) or "").endswith(call_suffix):
            return node.lineno
    return None


def analyze_train(tree: ast.Module) -> dict[str, Any]:
    trainer = _find_class(tree, "Trainer")
    method = _find_method(trainer, "test_json_submission")
    challenge = _open_load_lines(method, "json_challenges_file", "challenges")
    generation = _assigned_call_line(method, "generations", "json_submission")
    solution = _open_load_lines(method, "json_solutions_file", "solutions")
    scoring = _assigned_call_line(method, "metrics", "evaluate_generations")
    visualization_lines = sorted(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (_call_path(node.func) or "").endswith("visualize_json_submission")
        and any(isinstance(desc, ast.Name) and desc.id == "solutions" for arg in node.args for desc in ast.walk(arg))
    )
    init = _find_method(trainer, "__init__")
    paired_config_lines = sorted(
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, (ast.List, ast.Tuple))
        and [_string_key(item) for item in node.elts] == ["challenges", "solutions"]
    )
    train_epoch = _find_method(trainer, "train_epoch")
    train_epoch_call_lines = sorted(
        node.lineno
        for node in ast.walk(train_epoch)
        if isinstance(node, ast.Call)
        and (_call_path(node.func) or "").endswith("test_json_submission")
    )
    order = (
        [challenge[1], generation, solution[1], scoring]
        if challenge and generation and solution and scoring
        else []
    )
    detected = bool(order) and order == sorted(order) and bool(paired_config_lines) and bool(train_epoch_call_lines)
    return {
        "runner": "Trainer.test_json_submission",
        "challenge_open_line": challenge[0] if challenge else None,
        "challenge_load_line": challenge[1] if challenge else None,
        "generation_line": generation,
        "solution_open_line": solution[0] if solution else None,
        "solution_load_line": solution[1] if solution else None,
        "scoring_line": scoring,
        "solution_visualization_lines": visualization_lines,
        "paired_json_config_lines": paired_config_lines,
        "train_epoch_call_lines": train_epoch_call_lines,
        "generation_before_solution_read": detected,
        "same_function_and_process_label_flow": detected,
    }


def analyze_evaluate_checkpoint(tree: ast.Module) -> dict[str, Any]:
    evaluate_json = _find_function(tree, "evaluate_json")
    forwarding_calls = []
    for node in ast.walk(evaluate_json):
        if not isinstance(node, ast.Call) or not (_call_path(node.func) or "").endswith("test_json_submission"):
            continue
        forwarding_calls.append(
            {
                "line": node.lineno,
                "keywords": sorted(keyword.arg for keyword in node.keywords if keyword.arg),
            }
        )
    add_arguments: dict[str, list[int]] = {"challenge": [], "solution": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not (_call_path(node.func) or "").endswith("add_argument"):
            continue
        values = [_string_key(argument) for argument in node.args]
        if "-jc" in values and "--json-challenges-file" in values:
            add_arguments["challenge"].append(node.lineno)
        if "-js" in values and "--json-solutions-file" in values:
            add_arguments["solution"].append(node.lineno)
    pairing_error_lines = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "Must provide both the json challenges (-jc) and solutions (-js) files" in node.value
    )
    main = _find_function(tree, "main")
    call_lines: dict[str, list[int]] = {name: [] for name in ("wandb.init", "run.use_artifact", "artifact.download")}
    wandb_mode_lines: list[int] = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            name = _call_path(node.func)
            if name in call_lines:
                call_lines[name].append(node.lineno)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == "run":
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and _string_key(target.slice) == "WANDB_MODE"
                    and _call_path(target.value) == "os.environ"
                ):
                    wandb_mode_lines.append(node.lineno)
    artifacts = sorted(
        {
            match.group(0)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            for match in ARTIFACT_RE.finditer(node.value)
        }
    )
    detected = all(
        [
            bool(forwarding_calls),
            bool(add_arguments["challenge"]),
            bool(add_arguments["solution"]),
            bool(pairing_error_lines),
            bool(wandb_mode_lines),
            all(call_lines.values()),
            artifacts == EXPECTED_WANDB_ARTIFACT_IDS,
        ]
    )
    return {
        "evaluate_json_forwarding_calls": forwarding_calls,
        "challenge_cli_lines": add_arguments["challenge"],
        "solution_cli_lines": add_arguments["solution"],
        "paired_cli_error_lines": pairing_error_lines,
        "wandb_mode_run_lines": sorted(wandb_mode_lines),
        "wandb_calls": call_lines,
        "source_declared_wandb_artifact_ids": artifacts,
        "artifact_identifier_status": "source_declared_unverified",
        "official_network_and_paired_label_flow_detected": detected,
    }


def controls_record(ledger: ReadLedger) -> dict[str, Any]:
    return {
        "network_accessed": False,
        "gpu_initialized": False,
        "upstream_imported": False,
        "upstream_executed": False,
        "arc_data_bytes_read": ledger.count("arc_data") > 0,
        "yaml_bytes_read": ledger.count("yaml") > 0,
        "notebook_bytes_read": ledger.count("notebook") > 0,
        "checkpoint_bytes_read": ledger.count("checkpoint") > 0,
        "pyc_bytes_read": ledger.count("pyc") > 0,
        "solver_executed": False,
        "prediction_produced": False,
        "retained_python_files_read": ledger.count("retained_python"),
    }


def base_record(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "runner": "scripts.audit_lpn_gates",
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


def failure_record(run_id: str, stage: str, error: BaseException, ledger: ReadLedger) -> dict[str, Any]:
    record = base_record(run_id)
    record.update(
        {
            "status": "failed",
            "method_gate_status": "blocked",
            "controls": controls_record(ledger),
            "error": {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
            "claim_boundary": "The audit failed closed and grants no execution or performance claim.",
        }
    )
    return record


def run_static_audit(
    config_path: Path,
    run_id: str,
    ledger: ReadLedger,
) -> dict[str, Any]:
    ledger.bind_config(config_path)
    config_payload, _ = secure_read_absolute(
        config_path,
        max_bytes=MAX_CONFIG_BYTES,
        role="canonical_config",
        ledger=ledger,
        display=str(config_path),
    )
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    config = validate_config(strict_json(config_payload, "LPN gate config"))
    ledger.bind_source_policy(
        {item["path"] for item in config["source"]["retained_python"]}
    )

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
    if not isinstance(source_locks, dict) or source_locks.get("sources", {}).get("lpn") != EXPECTED_SOURCE_LOCK_ENTRY:
        raise ValueError("source_locks.json LPN entry mismatch")

    root_fd = open_absolute_directory(ROOT / SOURCE_RELATIVE)
    try:
        initial = closed_world_inventory(root_fd, config)
        git_observation = verify_git_contract(root_fd, config)

        python_payloads: dict[str, bytes] = {}
        python_trees: dict[str, ast.Module] = {}
        python_inventory: list[dict[str, Any]] = []
        for declaration in config["source"]["retained_python"]:
            payload = secure_read_relative(
                root_fd,
                declaration["path"],
                expected_bytes=declaration["bytes"],
                expected_sha256=declaration["sha256"],
                snapshot=initial,
                role="retained_python",
                ledger=ledger,
            )
            tree = parse_python(payload, declaration["path"])
            python_payloads[declaration["path"]] = payload
            python_trees[declaration["path"]] = tree
            python_inventory.append(
                {
                    **declaration,
                    "ast_parsed": True,
                    "bytes_read": True,
                }
            )

        license_config = config["license"]
        license_payload = secure_read_relative(
            root_fd,
            license_config["path"],
            expected_bytes=license_config["bytes"],
            expected_sha256=license_config["sha256"],
            snapshot=initial,
            role="code_license",
            ledger=ledger,
        )
        if not license_payload.startswith(b"                                 Apache License\n"):
            raise ValueError("locked code license header mismatch")

        evaluator = analyze_evaluator(python_trees[config["ast_contract"]["evaluator_path"]])
        train = analyze_train(python_trees[config["ast_contract"]["train_path"]])
        evaluate_checkpoint = analyze_evaluate_checkpoint(
            python_trees[config["ast_contract"]["evaluate_checkpoint_path"]]
        )
        if not evaluator["challenge_only_candidate_detected"]:
            raise ValueError("Evaluator.json_submission AST contract mismatch")
        if not train["same_function_and_process_label_flow"]:
            raise ValueError("Trainer.test_json_submission label-flow contract mismatch")
        if not evaluate_checkpoint["official_network_and_paired_label_flow_detected"]:
            raise ValueError("evaluate_checkpoint official-flow contract mismatch")

        final = closed_world_inventory(root_fd, config)
        if initial != final:
            raise RuntimeError("LPN worktree metadata changed during the audit")
        verify_directory_path_identity(ROOT / SOURCE_RELATIVE, root_fd)
        final_git_observation = verify_git_contract(root_fd, config)
        if final_git_observation != git_observation:
            raise RuntimeError("LPN Git revision/tree metadata changed during the audit")
    finally:
        os.close(root_fd)

    retained_paths = {item["path"] for item in config["source"]["retained_python"]} | {"LICENSE"}
    tracked_paths = {item["path"] for item in config["source"]["tracked_files"]}
    pyc_paths = {item["path"] for item in config["source"]["ignored_pyc"]}
    metadata_only_paths = sorted((tracked_paths - retained_paths) | pyc_paths)
    metadata_only_inventory = [
        {
            "path": path,
            **initial[path]["metadata"],
            "sha256": None,
            "bytes_read": False,
            "manifest_included": False,
            "executable_content_trusted": False,
        }
        for path in metadata_only_paths
    ]
    checkpoint_paths = [
        item for item in metadata_only_inventory if PurePosixPath(item["path"]).suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    if checkpoint_paths or len(checkpoint_paths) != config["artifact"][
        "expected_locked_source_tree_checkpoint_count"
    ]:
        raise ValueError("local checkpoint count mismatch")

    controls = controls_record(ledger)
    for key in (
        "arc_data_bytes_read",
        "yaml_bytes_read",
        "notebook_bytes_read",
        "checkpoint_bytes_read",
        "pyc_bytes_read",
    ):
        if controls[key]:
            raise RuntimeError(f"restricted-byte control violated: {key}")
    if controls["retained_python_files_read"] != 21:
        raise RuntimeError("retained Python read ledger count mismatch")
    stable_observation = {
        "source_lock_sha256": EXPECTED_SOURCE_LOCK_SHA256,
        "git": git_observation,
        "python": python_inventory,
        "metadata_only": metadata_only_inventory,
        "evaluator": evaluator,
        "train": train,
        "evaluate_checkpoint": evaluate_checkpoint,
        "license_sha256": license_config["sha256"],
        "checkpoint_count": len(checkpoint_paths),
        "blockers": EXPECTED_BLOCKER_IDS,
    }
    record = base_record(run_id)
    record.update(
        {
            "status": "passed",
            "config": {
                "path": str(config_path),
                "sha256": config_sha256,
                "config_id": CONFIG_ID,
            },
            "source_lock": {
                "path": SOURCE_LOCK_RELATIVE,
                "sha256": EXPECTED_SOURCE_LOCK_SHA256,
                "entry_matches": True,
            },
            "source": {
                **git_observation,
                "repository_path": SOURCE_RELATIVE,
                "closed_world_inventory_passed": True,
                "unknown_entry_count": 0,
                "python_file_count": len(python_inventory),
                "python_bytes": sum(item["bytes"] for item in python_inventory),
                "retained_python_inventory": python_inventory,
                "metadata_only_inventory": metadata_only_inventory,
                "ignored_pyc_count": len(pyc_paths),
                "ignored_pyc_bytes_read": False,
            },
            "ast_gate": {
                "status": "passed",
                "evaluator_challenge_only_candidate": evaluator,
                "official_train_same_process_label_flow": train,
                "official_evaluate_checkpoint_flow": evaluate_checkpoint,
            },
            "checkpoint_gate": {
                "status": "blocked",
                "present_count": 0,
                "presence_scope": (
                    "exact closed LPN worktree inventory excluding opaque .git; "
                    "no external artifact/cache roots inspected"
                ),
                "checkpoint_bytes_read": False,
                "reason": (
                    "No checkpoint-like file is present in the exact closed LPN "
                    "worktree inventory; external artifact/cache roots were not inspected, "
                    "and declared W&B artifacts remain unverified source strings."
                ),
            },
            "artifact_gate": {
                "status": "blocked",
                "source_declared_wandb_artifact_ids": EXPECTED_WANDB_ARTIFACT_IDS,
                "identifier_status": "source_declared_unverified",
                "remote_verification_performed": False,
                "provenance_verified": False,
                "config_binding_verified": False,
                "artifact_license_verified": False,
            },
            "label_firewall_gate": {
                "status": "blocked",
                "challenge_only_candidate_detected": True,
                "official_runner_same_process_solution_read_detected": True,
                "strict_adapter_validated": False,
            },
            "data_mount_gate": {
                "status": "blocked",
                "bundled_arc_json_metadata_only": True,
                "bundled_arc_json_bytes_read": False,
                "challenge_only_staging_validated": False,
            },
            "license_gate": {
                "code_status": "passed",
                "code_identifier": "Apache-2.0",
                "code_license_sha256": license_config["sha256"],
                "artifact_status": "blocked-unverified",
            },
            "resource_gate": {
                "status": "blocked",
                "checkpoint_resource_profile_known": False,
                "child_inclusive_accounting_validated": False,
            },
            "runtime_staging_gate": {
                "status": "blocked",
                "ignored_pyc_present": True,
                "ignored_pyc_count": len(pyc_paths),
                "clean_code_only_staging_validated": False,
            },
            "blockers": [
                {"id": blocker, "status": "blocked"}
                for blocker in EXPECTED_BLOCKER_IDS
            ],
            "gate_summary": {"passed": 3, "blocked": len(EXPECTED_BLOCKER_IDS)},
            "controls": controls,
            "validation": {
                "config_exact_contract": True,
                "source_lock_matches": True,
                "git_revision_and_tree_match": True,
                "git_metadata_stable_after_reads": True,
                "closed_world_inventory_before_reads": True,
                "closed_world_inventory_stable_after_reads": True,
                "retained_python_hashes_and_syntax_match": True,
                "ast_contract_matches": True,
                "metadata_only_files_unread": True,
                "code_license_matches": True,
                "checkpoint_absent_in_closed_source_tree": True,
            },
            "observation_digest_sha256": canonical_sha256(stable_observation),
            "claim_boundary": (
                "This static audit matches the locked source/artifact/data/label blockers only. "
                "It produces no solver prediction and cannot promote the prior component smoke, "
                "strict runtime, benchmark admission, or performance eligibility."
            ),
            "limitations": [
                "The .git directory is pinned as opaque local VCS metadata; its internal files are not a closed-world worktree claim.",
                "Git object ids are commit-tree metadata; bundled JSON, YAML, notebooks, bytecode, and checkpoint-like files were not opened or hashed.",
                "W&B identifiers were extracted from locked Python AST constants and were not resolved over the network.",
                "Process RSS below covers the auditor only; Git subprocess resource use is not child-inclusive evidence for LPN.",
            ],
        }
    )
    return record


def execute_audit(
    config_path: Path,
    output_path: Path,
) -> tuple[int, dict[str, Any]]:
    validate_config_location(config_path, output_path)
    output = create_fresh_output(output_path)
    ledger = ReadLedger()
    started = time.perf_counter()
    started_at = utc_now()
    record_committed = False
    try:
        try:
            record = run_static_audit(
                config_path,
                output_path.name,
                ledger,
            )
        except BaseException as error:
            record = failure_record(output_path.name, "static-audit", error, ledger)
            exit_code = 1
        else:
            exit_code = 0
        record["started_at_utc"] = started_at
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_time_seconds": round(time.perf_counter() - started, 6),
            "current_process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "auditor-process-only; Git subprocesses excluded",
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
    except (OSError, ValueError) as error:
        parser.error(str(error))
    stream = sys.stdout if exit_code == 0 else sys.stderr
    try:
        print(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False),
            file=stream,
        )
    except BrokenPipeError:
        pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
