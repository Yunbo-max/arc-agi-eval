#!/usr/bin/python3
"""Verify the closed TRM runner manifest, then execute its auditor bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import types
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
MANIFEST_ID = "trm-gate-runner-manifest-v1"
METHOD_ID = "tiny-recursive-models"
MAX_MEMBER_BYTES = 4 * 1024 * 1024
EXPECTED_MEMBERS = {
    "launcher": "scripts/launch_trm_gate.py",
    "auditor": "scripts/audit_trm_gates.py",
    "support": "scripts/audit_batch_c_static_gates.py",
    "config": "configs/trm_gate_v1.json",
    "source_lock": "configs/source_locks.json",
}
_DIRECT_SOURCE_ENTRY_TOKEN: object | None = None
REQUIRED_PYTHON_EXECUTABLE = "/usr/bin/python3"


def require_linux_file_safety_flags() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    if missing:
        raise RuntimeError(
            "TRM launcher requires non-degrading file-safety flags: "
            + ", ".join(missing)
        )


def isolated_python_context() -> dict[str, Any]:
    """Describe the minimum bootstrap controls required for a production run."""

    sys_path_excludes_cwd = all(
        isinstance(item, str) and item and os.path.isabs(item) for item in sys.path
    )
    return {
        "python_executable": os.path.abspath(sys.executable),
        "required_python_executable": REQUIRED_PYTHON_EXECUTABLE,
        "isolated": sys.flags.isolated == 1,
        "ignore_environment": sys.flags.ignore_environment == 1,
        "no_user_site": sys.flags.no_user_site == 1,
        "no_site": sys.flags.no_site == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "sys_path_excludes_cwd_and_relative_entries": sys_path_excludes_cwd,
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from error


def safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe {field}: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"noncanonical {field}: {value!r}")
    return value


def stat_signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def secure_read_relative(root_fd: int, relative: str, *, max_bytes: int) -> bytes:
    path = PurePosixPath(safe_relative_path(relative, "runner member path"))
    directory_fd = os.dup(root_fd)
    try:
        for component in path.parts[:-1]:
            child_fd = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
        descriptor = os.open(
            path.parts[-1],
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(f"runner member is not a single-link file: {relative}")
            if before.st_size > max_bytes:
                raise ValueError(f"runner member exceeds byte limit: {relative}")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"runner member truncated while reading: {relative}")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError(f"runner member grew while reading: {relative}")
            after = os.fstat(descriptor)
            if stat_signature(before) != stat_signature(after):
                raise ValueError(f"runner member changed while reading: {relative}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)


def load_verified_manifest(
    manifest_relative: str,
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    require_linux_file_safety_flags()
    root_fd = os.open(
        ROOT,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
    )
    try:
        manifest_payload = secure_read_relative(
            root_fd, manifest_relative, max_bytes=512 * 1024
        )
        manifest = strict_json(manifest_payload, "TRM runner manifest")
        if not isinstance(manifest, dict):
            raise ValueError("TRM runner manifest must be an object")
        members = manifest.get("members")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("manifest_id") != MANIFEST_ID
            or manifest.get("method_id") != METHOD_ID
            or manifest.get("member_count") != len(EXPECTED_MEMBERS)
            or not isinstance(members, list)
            or len(members) != len(EXPECTED_MEMBERS)
        ):
            raise ValueError("TRM runner manifest identity/count mismatch")
        roles = [item.get("role") for item in members if isinstance(item, dict)]
        paths = [item.get("path") for item in members if isinstance(item, dict)]
        if (
            len(roles) != len(EXPECTED_MEMBERS)
            or len(set(roles)) != len(roles)
            or dict(zip(roles, paths, strict=True)) != EXPECTED_MEMBERS
        ):
            raise ValueError("TRM runner manifest member closure mismatch")
        if manifest.get("members_sha256") != canonical_sha256(members):
            raise ValueError("TRM runner manifest member digest mismatch")

        payloads: dict[str, bytes] = {}
        for member in members:
            allowed = {"role", "path", "bytes", "sha256"}
            if member.get("role") == "config":
                allowed.add("canonical_sha256")
            if set(member) != allowed:
                raise ValueError("TRM runner manifest member fields mismatch")
            relative = safe_relative_path(member["path"], "runner member path")
            payload = secure_read_relative(
                root_fd, relative, max_bytes=MAX_MEMBER_BYTES
            )
            if (
                member.get("bytes") != len(payload)
                or member.get("sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError(f"TRM runner member hash/size mismatch: {relative}")
            if member["role"] == "config":
                config = strict_json(payload, "TRM gate config")
                if member.get("canonical_sha256") != canonical_sha256(config):
                    raise ValueError("TRM config canonical digest mismatch")
            payloads[relative] = payload
        return manifest, manifest_payload, payloads
    finally:
        os.close(root_fd)


def main(
    argv: list[str] | None = None, *, _entry_token: object | None = None
) -> int:
    python_context = isolated_python_context()
    python_context_ok = python_context == {
        "python_executable": REQUIRED_PYTHON_EXECUTABLE,
        "required_python_executable": REQUIRED_PYTHON_EXECUTABLE,
        "isolated": True,
        "ignore_environment": True,
        "no_user_site": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "sys_path_excludes_cwd_and_relative_entries": True,
    }
    direct_source_entry = (
        _entry_token is not None
        and _entry_token is _DIRECT_SOURCE_ENTRY_TOKEN
        and __name__ == "__main__"
        and __spec__ is None
        and globals().get("__cached__") is None
        and os.path.abspath(sys.argv[0]) == os.path.abspath(__file__)
        and python_context_ok
    )
    if not direct_source_entry:
        print(
            "launch_trm_gate: production reports require canonical direct-source "
            "script entry",
            file=sys.stderr,
        )
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_relative = safe_relative_path(
        args.manifest.as_posix(), "runner manifest path"
    )
    if manifest_relative != "configs/trm_gate_runner_manifest_v1.json":
        parser.error("production TRM runner manifest path is not canonical")
    expected_manifest_sha256 = args.expected_manifest_sha256
    if (
        len(expected_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
    ):
        parser.error("expected runner-manifest SHA-256 must be 64 lowercase hex characters")
    try:
        manifest, manifest_payload, payloads = load_verified_manifest(
            manifest_relative
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    observed_manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if observed_manifest_sha256 != expected_manifest_sha256:
        parser.error("operator-supplied runner-manifest SHA-256 mismatch")

    auditor_path = EXPECTED_MEMBERS["auditor"]
    auditor_payload = payloads[auditor_path]
    module = types.ModuleType("_manifest_verified_trm_gate_auditor")
    module.__file__ = str(ROOT / auditor_path)
    module.__package__ = "scripts"
    module.__dict__["__verified_runner_manifest_context__"] = {
        "manifest_path": manifest_relative,
        "manifest_sha256": observed_manifest_sha256,
        "operator_supplied_manifest_sha256": expected_manifest_sha256,
        "manifest_payload": manifest_payload,
        "member_payloads": payloads,
        "executed_auditor_sha256": hashlib.sha256(auditor_payload).hexdigest(),
        "launcher_source_execution": {
            "mode": "canonical-direct-script-source",
            "name_is_main": True,
            "spec_is_none": True,
            "cached_is_none": True,
            "argv0_is_canonical_script": True,
            **python_context,
        },
    }
    sys.dont_write_bytecode = True
    try:
        exec(
            compile(auditor_payload, str(ROOT / auditor_path), "exec", dont_inherit=True),
            module.__dict__,
        )
        return module.main(
            [
                "--config",
                args.config.as_posix(),
                "--runner-manifest",
                manifest_relative,
                "--output-directory",
                args.output_directory.as_posix(),
            ]
        )
    except SystemExit:
        raise
    except BaseException as error:
        print(
            f"launch_trm_gate: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    _DIRECT_SOURCE_ENTRY_TOKEN = object()
    raise SystemExit(main(_entry_token=_DIRECT_SOURCE_ENTRY_TOKEN))
