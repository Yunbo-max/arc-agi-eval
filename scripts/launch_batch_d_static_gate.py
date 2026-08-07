#!/usr/bin/python3
"""Verify one closed Batch D runner manifest, then execute its auditor bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import types
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
LAUNCHER_RELATIVE = "scripts/launch_batch_d_static_gate.py"
AUDITOR_RELATIVE = "scripts/audit_batch_d_static_gates.py"
SUPPORT_RELATIVE = "scripts/audit_batch_c_static_gates.py"
SOURCE_LOCK_RELATIVE = "configs/source_locks.json"
MAX_MANIFEST_BYTES = 512 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024
REQUIRED_PYTHON_EXECUTABLE = "/usr/bin/python3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PROFILES: dict[str, dict[str, str]] = {
    "configs/soar_gate_runner_manifest_v1.json": {
        "manifest_path": "configs/soar_gate_runner_manifest_v1.json",
        "manifest_id": "soar-gate-runner-manifest-v1",
        "method_id": "soar",
        "config_path": "configs/soar_gate_v1.json",
        "report_namespace": "reports/soar",
    },
    "configs/nvarc_gate_runner_manifest_v1.json": {
        "manifest_path": "configs/nvarc_gate_runner_manifest_v1.json",
        "manifest_id": "nvarc-gate-runner-manifest-v1",
        "method_id": "nvarc",
        "config_path": "configs/nvarc_gate_v1.json",
        "report_namespace": "reports/nvarc",
    },
}

_DIRECT_SOURCE_ENTRY_TOKEN: object | None = None


def require_linux_file_safety_flags() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    if missing:
        raise RuntimeError(
            "Batch D launcher requires non-degrading file-safety flags: "
            + ", ".join(missing)
        )


def isolated_python_context() -> dict[str, Any]:
    """Describe the canonical interpreter controls required for publication."""

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
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(
                    f"runner member is not a single-link file: {relative}"
                )
            if before.st_size > max_bytes:
                raise ValueError(f"runner member exceeds byte limit: {relative}")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(
                        f"runner member truncated while reading: {relative}"
                    )
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


def expected_members(profile: dict[str, str]) -> dict[str, str]:
    return {
        "launcher": LAUNCHER_RELATIVE,
        "auditor": AUDITOR_RELATIVE,
        "support": SUPPORT_RELATIVE,
        "config": profile["config_path"],
        "source_lock": SOURCE_LOCK_RELATIVE,
    }


def validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def validate_canonical_config_argument(value: Path, profile: dict[str, str]) -> None:
    supplied = value.as_posix()
    canonical_relative = profile["config_path"]
    canonical_absolute = (ROOT / canonical_relative).as_posix()
    if supplied not in {canonical_relative, canonical_absolute}:
        raise ValueError(
            f"production config path must equal {canonical_relative}"
        )


def load_verified_manifest(
    manifest_relative: str,
    profile: dict[str, str],
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    require_linux_file_safety_flags()
    expected = expected_members(profile)
    root_fd = os.open(
        ROOT,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        manifest_payload = secure_read_relative(
            root_fd, manifest_relative, max_bytes=MAX_MANIFEST_BYTES
        )
        manifest = strict_json(manifest_payload, "Batch D runner manifest")
        if not isinstance(manifest, dict):
            raise ValueError("Batch D runner manifest must be an object")
        if set(manifest) != {
            "schema_version",
            "manifest_id",
            "method_id",
            "member_count",
            "members",
            "members_sha256",
        }:
            raise ValueError("Batch D runner manifest fields mismatch")
        members = manifest.get("members")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("manifest_id") != profile["manifest_id"]
            or manifest.get("method_id") != profile["method_id"]
            or manifest.get("member_count") != len(expected)
            or not isinstance(members, list)
            or len(members) != len(expected)
            or any(not isinstance(item, dict) for item in members)
        ):
            raise ValueError("Batch D runner manifest identity/count mismatch")

        observed: dict[str, str] = {}
        for member in members:
            role = member.get("role")
            if not isinstance(role, str) or role not in expected or role in observed:
                raise ValueError("Batch D runner manifest member role mismatch")
            allowed = {"role", "path", "bytes", "sha256"}
            if role == "config":
                allowed.add("canonical_sha256")
            if set(member) != allowed:
                raise ValueError("Batch D runner manifest member fields mismatch")
            observed[role] = safe_relative_path(
                member.get("path"), "runner member path"
            )
            if (
                not isinstance(member.get("bytes"), int)
                or isinstance(member.get("bytes"), bool)
                or member["bytes"] < 0
                or member["bytes"] > MAX_MEMBER_BYTES
            ):
                raise ValueError("Batch D runner member byte declaration is invalid")
            validate_sha256(member.get("sha256"), "runner member SHA-256")
            if role == "config":
                validate_sha256(
                    member.get("canonical_sha256"),
                    "runner config canonical SHA-256",
                )
        if observed != expected:
            raise ValueError("Batch D runner manifest member closure mismatch")
        if (
            validate_sha256(
                manifest.get("members_sha256"), "runner members SHA-256"
            )
            != canonical_sha256(members)
        ):
            raise ValueError("Batch D runner manifest member digest mismatch")

        payloads: dict[str, bytes] = {}
        for member in members:
            relative = member["path"]
            payload = secure_read_relative(
                root_fd, relative, max_bytes=MAX_MEMBER_BYTES
            )
            if (
                member["bytes"] != len(payload)
                or member["sha256"] != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError(
                    f"Batch D runner member hash/size mismatch: {relative}"
                )
            if member["role"] == "config":
                config = strict_json(payload, "Batch D gate config")
                if not isinstance(config, dict):
                    raise ValueError("Batch D gate config must be an object")
                if member["canonical_sha256"] != canonical_sha256(config):
                    raise ValueError("Batch D config canonical digest mismatch")
            payloads[relative] = payload
        if set(payloads) != set(expected.values()):
            raise ValueError("Batch D verified member payload closure mismatch")
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
            "launch_batch_d_static_gate: production reports require canonical "
            "direct-source script entry",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        manifest_relative = safe_relative_path(
            args.manifest.as_posix(), "runner manifest path"
        )
    except ValueError as error:
        parser.error(str(error))
    profile = PROFILES.get(manifest_relative)
    if profile is None:
        parser.error("production Batch D runner manifest path is not canonical")
    try:
        validate_canonical_config_argument(args.config, profile)
        expected_manifest_sha256 = validate_sha256(
            args.expected_manifest_sha256,
            "expected runner-manifest SHA-256",
        )
        manifest, manifest_payload, payloads = load_verified_manifest(
            manifest_relative, profile
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    observed_manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if observed_manifest_sha256 != expected_manifest_sha256:
        parser.error("operator-supplied runner-manifest SHA-256 mismatch")

    auditor_payload = payloads[AUDITOR_RELATIVE]
    module = types.ModuleType(
        f"_manifest_verified_batch_d_static_gate_auditor_{profile['method_id']}"
    )
    module.__file__ = str(ROOT / AUDITOR_RELATIVE)
    module.__package__ = "scripts"
    module.__dict__["__verified_runner_manifest_context__"] = {
        "selected_profile": dict(profile),
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
            compile(
                auditor_payload,
                str(ROOT / AUDITOR_RELATIVE),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
        auditor_main = getattr(module, "main", None)
        if not callable(auditor_main):
            raise RuntimeError("verified Batch D auditor has no callable main")
        return auditor_main(
            [
                "--config",
                profile["config_path"],
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
            f"launch_batch_d_static_gate: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    _DIRECT_SOURCE_ENTRY_TOKEN = object()
    raise SystemExit(main(_entry_token=_DIRECT_SOURCE_ENTRY_TOKEN))
