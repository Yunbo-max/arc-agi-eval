#!/usr/bin/env python3
"""Audit host prerequisites for process-tree resource accounting without enabling it.

The probe is deliberately non-destructive.  It does not create or modify a
cgroup, send a signal, initialize a GPU workload, or claim child-inclusive
measurements.  A successful probe therefore terminalizes the audit while
leaving the process-tree resource gate blocked.
"""

from __future__ import annotations

import argparse
import csv
import ctypes.util
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    ROOT
    / "reports"
    / "e0-resources"
    / "20260806-process-tree-resource-gate-probe"
)
MIB = 1024**2
REQUIRED_V2_CONTROLLERS = ("cpu", "memory", "pids")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path = ROOT) -> dict[str, object]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve())
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _unescape_mount_field(value: str) -> str:
    for escaped, plain in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(escaped, plain)
    return value


def parse_mountinfo(text: str) -> list[dict[str, object]]:
    """Return the cgroup mounts declared by Linux ``/proc/self/mountinfo``."""

    mounts: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            left, right = line.split(" - ", 1)
        except ValueError:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            continue
        filesystem_type = right_fields[0]
        if filesystem_type not in {"cgroup", "cgroup2"}:
            continue
        mount_options = sorted(set(left_fields[5].split(",")))
        super_options = sorted(set(right_fields[2].split(",")))
        generic_options = {
            "async",
            "diratime",
            "exec",
            "lazytime",
            "noatime",
            "nodev",
            "nodiratime",
            "noexec",
            "nosuid",
            "relatime",
            "ro",
            "rw",
            "strictatime",
            "suid",
            "sync",
        }
        controllers = (
            sorted(set(super_options) - generic_options)
            if filesystem_type == "cgroup"
            else []
        )
        mounts.append(
            {
                "line_number": line_number,
                "mount_id": int(left_fields[0]),
                "root": _unescape_mount_field(left_fields[3]),
                "mount_point": _unescape_mount_field(left_fields[4]),
                "mount_options": mount_options,
                "filesystem_type": filesystem_type,
                "source": _unescape_mount_field(right_fields[1]),
                "super_options": super_options,
                "controllers": controllers,
            }
        )
    return mounts


def parse_proc_cgroup(text: str) -> list[dict[str, object]]:
    """Parse Linux ``/proc/self/cgroup`` without interpreting controller state."""

    memberships: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy_id, controller_text, membership_path = fields
        try:
            numeric_hierarchy_id = int(hierarchy_id)
        except ValueError:
            continue
        memberships.append(
            {
                "line_number": line_number,
                "hierarchy_id": numeric_hierarchy_id,
                "controllers": sorted(
                    controller
                    for controller in controller_text.split(",")
                    if controller
                ),
                "path": membership_path,
            }
        )
    return memberships


def _membership_for_mount(
    mount: dict[str, object], memberships: list[dict[str, object]]
) -> dict[str, object] | None:
    if mount["filesystem_type"] == "cgroup2":
        return next(
            (
                item
                for item in memberships
                if item["hierarchy_id"] == 0 and not item["controllers"]
            ),
            None,
        )
    mount_controllers = set(mount["controllers"])
    return next(
        (
            item
            for item in memberships
            if mount_controllers.intersection(item["controllers"])
        ),
        None,
    )


def resolve_membership_directory(
    mount: dict[str, object], membership: dict[str, object] | None
) -> Path | None:
    """Resolve a namespaced cgroup membership beneath its visible mount."""

    if membership is None:
        return None
    mount_root = PurePosixPath(str(mount["root"]))
    membership_path = PurePosixPath(str(membership["path"]))
    try:
        relative = membership_path.relative_to(mount_root)
    except ValueError:
        return None
    return Path(str(mount["mount_point"])).joinpath(*relative.parts)


def _access_snapshot(path: Path) -> dict[str, object]:
    try:
        metadata = path.stat()
    except OSError as error:
        return {
            "path": str(path),
            "exists": False,
            "error": f"{type(error).__name__}: {error}",
            "readable": False,
            "writable": False,
            "executable": False,
        }
    return {
        "path": str(path),
        "exists": True,
        "kind": "directory" if stat.S_ISDIR(metadata.st_mode) else "file",
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
        "executable": os.access(path, os.X_OK),
    }


def _read_words(path: Path) -> list[str] | None:
    try:
        return sorted(set(path.read_text(encoding="utf-8").split()))
    except OSError:
        return None


def probe_cgroup_environment(
    *, mountinfo_text: str | None = None, membership_text: str | None = None
) -> dict[str, object]:
    if mountinfo_text is None:
        mountinfo_text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    if membership_text is None:
        membership_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")

    mounts = parse_mountinfo(mountinfo_text)
    memberships = parse_proc_cgroup(membership_text)
    v1_mounts = [item for item in mounts if item["filesystem_type"] == "cgroup"]
    v2_mounts = [item for item in mounts if item["filesystem_type"] == "cgroup2"]
    if v1_mounts and v2_mounts:
        hierarchy = "hybrid"
    elif v2_mounts:
        hierarchy = "v2"
    elif v1_mounts:
        hierarchy = "v1"
    else:
        hierarchy = "none"

    mount_records: list[dict[str, object]] = []
    for mount in mounts:
        membership = _membership_for_mount(mount, memberships)
        directory = resolve_membership_directory(mount, membership)
        record = dict(mount)
        record["membership"] = membership
        record["membership_directory"] = (
            None if directory is None else _access_snapshot(directory)
        )
        if directory is not None:
            record["cgroup_procs"] = _access_snapshot(directory / "cgroup.procs")
        mount_records.append(record)

    unified_mount = next(iter(v2_mounts), None)
    unified_membership = (
        None
        if unified_mount is None
        else _membership_for_mount(unified_mount, memberships)
    )
    unified_directory = (
        None
        if unified_mount is None
        else resolve_membership_directory(unified_mount, unified_membership)
    )
    controllers = (
        None
        if unified_directory is None
        else _read_words(unified_directory / "cgroup.controllers")
    )
    required_controller_set = set(REQUIRED_V2_CONTROLLERS)
    controller_set = set(controllers or [])
    v2_files: dict[str, dict[str, object]] = {}
    for name in (
        "cgroup.controllers",
        "cgroup.events",
        "cgroup.procs",
        "cgroup.subtree_control",
        "cpu.stat",
        "memory.current",
        "memory.events",
        "memory.peak",
        "pids.current",
    ):
        if unified_directory is not None:
            v2_files[name] = _access_snapshot(unified_directory / name)

    unified_access = (
        None if unified_directory is None else _access_snapshot(unified_directory)
    )
    writable_by_access_checks = bool(
        unified_mount is not None
        and "rw" in unified_mount["mount_options"]
        and unified_access is not None
        and unified_access["writable"]
        and unified_access["executable"]
        and v2_files.get("cgroup.procs", {}).get("writable")
        and v2_files.get("cgroup.subtree_control", {}).get("writable")
        and required_controller_set.issubset(controller_set)
    )
    delegation = {
        "required_hierarchy": "cgroup_v2_unified",
        "required_controllers": list(REQUIRED_V2_CONTROLLERS),
        "controllers_visible": controllers,
        "required_controllers_visible": required_controller_set.issubset(
            controller_set
        ),
        "membership_directory": (
            None if unified_directory is None else str(unified_directory)
        ),
        "membership_directory_access": unified_access,
        "control_file_access": v2_files,
        "writable_by_non_mutating_access_checks": writable_by_access_checks,
        "mutation_test_performed": False,
        "delegation_verified": False,
        "usable_for_gate": False,
        "claim_boundary": (
            "Mount flags, ownership, mode bits, and os.access are read-only hints. "
            "No child cgroup was created and no PID was migrated, so delegation is "
            "not verified by this audit."
        ),
    }
    return {
        "status": "ok",
        "hierarchy": hierarchy,
        "cgroup_v1_mounted": bool(v1_mounts),
        "cgroup_v2_mounted": bool(v2_mounts),
        "mounts": mount_records,
        "memberships": memberships,
        "delegation": delegation,
    }


def _identity() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "cgroup": parse_proc_cgroup(
            Path("/proc/self/cgroup").read_text(encoding="utf-8")
        ),
    }


SETSID_CHILD_SOURCE = """\
import json
import os
from pathlib import Path

def state():
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "cgroup_text": Path("/proc/self/cgroup").read_text(encoding="utf-8"),
    }

before = state()
os.setsid()
after = state()
print(json.dumps({"before_setsid": before, "after_setsid": after}, sort_keys=True))
"""


SETSID_ROOT_SOURCE = f"""\
import json
import os
from pathlib import Path
import subprocess
import sys

def state():
    return {{
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "cgroup_text": Path("/proc/self/cgroup").read_text(encoding="utf-8"),
    }}

child = subprocess.Popen(
    [sys.executable, "-c", {SETSID_CHILD_SOURCE!r}],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
stdout, stderr = child.communicate()
payload = {{
    "root": state(),
    "child_return_code": child.returncode,
    "child_stderr": stderr,
    "child": json.loads(stdout),
    "child_reaped_by_normal_wait": True,
}}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if child.returncode == 0 else 1)
"""


def _normalize_fixture_identity(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    cgroup_text = normalized.pop("cgroup_text", None)
    if isinstance(cgroup_text, str):
        normalized["cgroup"] = parse_proc_cgroup(cgroup_text)
    return normalized


def run_setsid_escape_fixture() -> dict[str, object]:
    """Observe a cooperative setsid escape and reap it normally without signals."""

    parent = _identity()
    process = subprocess.Popen(
        [sys.executable, "-c", SETSID_ROOT_SOURCE],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = process.communicate()
    parse_error: str | None = None
    payload: dict[str, Any] | None = None
    try:
        loaded = json.loads(stdout)
        if not isinstance(loaded, dict):
            raise TypeError("fixture output is not an object")
        payload = loaded
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = f"{type(error).__name__}: {error}"

    assertions = {
        "fixture_root_started_new_session": False,
        "child_inherited_root_process_group": False,
        "child_setsid_changed_process_group": False,
        "child_escaped_root_process_group": False,
        "setsid_preserved_cgroup_membership": False,
        "child_reaped_by_normal_wait": False,
    }
    normalized_payload: dict[str, Any] | None = None
    if payload is not None:
        root = _normalize_fixture_identity(payload["root"])
        child = payload["child"]
        before = _normalize_fixture_identity(child["before_setsid"])
        after = _normalize_fixture_identity(child["after_setsid"])
        normalized_payload = {
            "root": root,
            "child_return_code": payload["child_return_code"],
            "child_stderr": payload["child_stderr"],
            "child_reaped_by_normal_wait": payload["child_reaped_by_normal_wait"],
            "child": {"before_setsid": before, "after_setsid": after},
        }
        assertions = {
            "fixture_root_started_new_session": (
                root["pid"] == root["pgid"] == root["sid"]
            ),
            "child_inherited_root_process_group": before["pgid"] == root["pgid"],
            "child_setsid_changed_process_group": (
                before["pgid"] != after["pgid"]
                and after["pid"] == after["pgid"] == after["sid"]
            ),
            "child_escaped_root_process_group": after["pgid"] != root["pgid"],
            "setsid_preserved_cgroup_membership": (
                root["cgroup"] == before["cgroup"] == after["cgroup"]
            ),
            "child_reaped_by_normal_wait": bool(
                payload["child_reaped_by_normal_wait"]
            ),
        }
    passed = process.returncode == 0 and all(assertions.values())
    return {
        "status": "passed" if passed else "failed",
        "executable": sys.executable,
        "fixture_root_source_sha256": sha256_bytes(
            SETSID_ROOT_SOURCE.encode("utf-8")
        ),
        "fixture_child_source_sha256": sha256_bytes(
            SETSID_CHILD_SOURCE.encode("utf-8")
        ),
        "root_return_code": process.returncode,
        "parent": parent,
        "payload": normalized_payload,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr": stderr,
        "parse_error": parse_error,
        "assertions": assertions,
        "signals_sent": [],
        "timeout_used": False,
        "gpu_used": False,
        "claim_boundary": (
            "The fixture only demonstrates that a descendant can leave its root "
            "process group with setsid while retaining cgroup membership. It is not "
            "a resource measurement or a containment test."
        ),
    }


CompletedRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_command_runner(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )


def _command_record(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return {
        "command": list(completed.args),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _optional_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    return int(stripped)


def _csv_rows(text: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(text.splitlines())
        if row
    ]


def _python_nvml_binding() -> str | None:
    for module_name in ("pynvml", "nvidia_ml_py"):
        try:
            if importlib.util.find_spec(module_name) is not None:
                return module_name
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return None


def probe_nvidia(
    *,
    nvidia_smi: str | None = None,
    nvml_library: str | None = None,
    python_binding: str | None = None,
    runner: CompletedRunner | None = None,
) -> dict[str, object]:
    """Capture only read-only NVIDIA/NVML state; no device context is created."""

    executable = shutil.which("nvidia-smi") if nvidia_smi is None else nvidia_smi
    library = ctypes.util.find_library("nvidia-ml") if nvml_library is None else nvml_library
    binding = _python_nvml_binding() if python_binding is None else python_binding
    if not executable:
        return {
            "status": "unavailable",
            "nvidia_smi": None,
            "nvml_shared_library": library,
            "python_nvml_binding": binding,
            "nvml_initialized_by_audit": False,
            "read_only_queries": True,
            "gpu_workload_launched": False,
            "devices": [],
            "compute_processes": [],
            "accounted_processes": [],
            "accounting_enabled_on_all_devices": False,
            "gpu_occupied": False,
            "commands": [],
        }
    command_runner = _default_command_runner if runner is None else runner
    device_fields = [
        "index",
        "name",
        "uuid",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "compute_mode",
        "accounting.mode",
    ]
    device_result = command_runner(
        [
            executable,
            f"--query-gpu={','.join(device_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    process_fields = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
    process_result = command_runner(
        [
            executable,
            f"--query-compute-apps={','.join(process_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    accounted_fields = [
        "gpu_uuid",
        "pid",
        "gpu_utilization",
        "max_memory_usage",
        "time",
    ]
    accounted_result = command_runner(
        [
            executable,
            f"--query-accounted-apps={','.join(accounted_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )

    devices: list[dict[str, object]] = []
    parse_errors: list[str] = []
    if device_result.returncode == 0:
        for row in _csv_rows(device_result.stdout):
            if len(row) != len(device_fields):
                parse_errors.append(f"unexpected device row with {len(row)} fields")
                continue
            try:
                total_mib = _optional_int(row[3])
                used_mib = _optional_int(row[4])
                free_mib = _optional_int(row[5])
                devices.append(
                    {
                        "index": int(row[0]),
                        "name": row[1],
                        "uuid": row[2],
                        "memory_total_bytes": (
                            None if total_mib is None else total_mib * MIB
                        ),
                        "memory_used_bytes": (
                            None if used_mib is None else used_mib * MIB
                        ),
                        "memory_free_bytes": (
                            None if free_mib is None else free_mib * MIB
                        ),
                        "utilization_percent": _optional_int(row[6]),
                        "compute_mode": row[7],
                        "accounting_mode": row[8],
                    }
                )
            except (ValueError, TypeError) as error:
                parse_errors.append(f"device row parse error: {error}")

    compute_processes: list[dict[str, object]] = []
    if process_result.returncode == 0:
        for row in _csv_rows(process_result.stdout):
            if len(row) != len(process_fields):
                parse_errors.append(f"unexpected process row with {len(row)} fields")
                continue
            try:
                memory_mib = _optional_int(row[3])
                compute_processes.append(
                    {
                        "gpu_uuid": row[0],
                        "pid": int(row[1]),
                        "process_name": row[2],
                        "used_gpu_memory_bytes": (
                            None if memory_mib is None else memory_mib * MIB
                        ),
                    }
                )
            except (ValueError, TypeError) as error:
                parse_errors.append(f"process row parse error: {error}")

    accounted_processes: list[dict[str, object]] = []
    if accounted_result.returncode == 0:
        for row in _csv_rows(accounted_result.stdout):
            if len(row) != len(accounted_fields):
                parse_errors.append(
                    f"unexpected accounted-process row with {len(row)} fields"
                )
                continue
            try:
                max_memory_mib = _optional_int(row[3])
                accounted_processes.append(
                    {
                        "gpu_uuid": row[0],
                        "pid": int(row[1]),
                        "gpu_utilization_percent": _optional_int(row[2]),
                        "max_memory_bytes": (
                            None
                            if max_memory_mib is None
                            else max_memory_mib * MIB
                        ),
                        "accounting_time_ms": _optional_int(row[4]),
                    }
                )
            except (ValueError, TypeError) as error:
                parse_errors.append(f"accounted-process row parse error: {error}")

    accounting_enabled = bool(devices) and all(
        str(device["accounting_mode"]).lower() == "enabled" for device in devices
    )
    gpu_occupied = bool(compute_processes) or any(
        (device["utilization_percent"] or 0) > 0 for device in devices
    )
    commands = [
        _command_record(device_result),
        _command_record(process_result),
        _command_record(accounted_result),
    ]
    query_success = all(item["return_code"] == 0 for item in commands)
    return {
        "status": "ok" if query_success and not parse_errors else "partial",
        "nvidia_smi": executable,
        "nvml_shared_library": library,
        "python_nvml_binding": binding,
        "nvml_initialized_by_audit": False,
        "read_only_queries": True,
        "gpu_workload_launched": False,
        "devices": devices,
        "compute_processes": compute_processes,
        "accounted_processes": accounted_processes,
        "accounting_enabled_on_all_devices": accounting_enabled,
        "gpu_occupied": gpu_occupied,
        "parse_errors": parse_errors,
        "commands": commands,
        "claim_boundary": (
            "nvidia-smi queries are an instantaneous read-only occupancy and "
            "accounting-mode snapshot. They do not attribute GPU use to a process "
            "tree, reserve a device, or continuously sample it."
        ),
    }


def evaluate_gate(
    cgroup: dict[str, object],
    nvidia: dict[str, object],
    setsid_fixture: dict[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    if cgroup["hierarchy"] != "v2":
        blockers.append(
            {
                "code": "cgroup_v2_unified_unavailable",
                "detail": (
                    f"Observed cgroup hierarchy is {cgroup['hierarchy']}; the audited "
                    "backend requires a unified cgroup v2 hierarchy."
                ),
            }
        )
    delegation = cgroup["delegation"]
    if not delegation["writable_by_non_mutating_access_checks"]:
        blockers.append(
            {
                "code": "cgroup_delegation_not_writable",
                "detail": (
                    "The current cgroup v2 membership does not expose all required "
                    "controller and write-access prerequisites."
                ),
            }
        )
    blockers.append(
        {
            "code": "cgroup_delegation_not_mutation_verified",
            "detail": (
                "This non-destructive audit did not create a child cgroup or migrate "
                "a PID, so apparent delegation cannot be treated as verified."
            ),
        }
    )
    if nvidia["status"] == "unavailable":
        blockers.append(
            {
                "code": "nvml_accounting_unavailable",
                "detail": "No nvidia-smi/NVML accounting observation is available.",
            }
        )
    elif not nvidia["accounting_enabled_on_all_devices"]:
        blockers.append(
            {
                "code": "nvidia_accounting_not_enabled",
                "detail": (
                    "NVIDIA accounting mode is not enabled on every visible device; "
                    "the read-only snapshot cannot supply process-tree GPU totals."
                ),
            }
        )
    if nvidia.get("gpu_occupied"):
        blockers.append(
            {
                "code": "gpu_currently_occupied",
                "detail": (
                    "The read-only snapshot observed a live GPU workload, so an "
                    "exclusive calibration was neither available nor attempted."
                ),
            }
        )
    if setsid_fixture["status"] != "passed":
        blockers.append(
            {
                "code": "setsid_fixture_failed",
                "detail": "The cooperative PGID-escape fixture did not complete.",
            }
        )
    blockers.append(
        {
            "code": "process_tree_backend_not_implemented_or_calibrated",
            "detail": (
                "This prerequisite probe implements no child-inclusive CPU, RSS, or "
                "GPU accounting backend and performs no accuracy/overhead calibration."
            ),
        }
    )
    return {
        "gate_id": "lp.process-tree-resources",
        "status": "blocked",
        "passed": False,
        "blockers": blockers,
        "required_next_evidence": [
            "A delegated per-run cgroup backend that survives setsid/double-fork membership changes.",
            "Terminal CPU, memory peak, OOM, and descendant-membership records from that cgroup.",
            "A separately calibrated GPU process-tree attribution policy, including PID reuse and MPS limitations.",
            "Synthetic parent/child/setsid calibration with known resource loads and error bounds.",
        ],
        "claim_boundary": (
            "The audit completed successfully, but the capability gate remains "
            "blocked. No child-inclusive resource result was produced."
        ),
    }


def build_record(*, output_directory: Path) -> dict[str, object]:
    started_at = utc_now()
    cgroup = probe_cgroup_environment()
    setsid_fixture = run_setsid_escape_fixture()
    nvidia = probe_nvidia()
    gate = evaluate_gate(cgroup, nvidia, setsid_fixture)
    implementation_paths = [
        ROOT / "scripts" / "audit_process_tree_resource_gate.py",
        ROOT / "tests" / "test_process_tree_resource_gate.py",
    ]
    return {
        "schema_version": 1,
        "method_id": "e0-resources",
        "run_id": output_directory.name,
        "runner": "scripts.audit_process_tree_resource_gate",
        "status": "passed",
        "scope": "read_only_process_tree_resource_gate_prerequisite_audit",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
        "implementation_files": [file_record(path) for path in implementation_paths],
        "cgroup": cgroup,
        "setsid_escape_fixture": setsid_fixture,
        "nvidia_read_only_snapshot": nvidia,
        "resource_claim": {
            "accounting_scope": "none",
            "children_included": False,
            "cpu_accounted": False,
            "rss_accounted": False,
            "gpu_accounted": False,
            "process_tree_measurement_performed": False,
        },
        "safety": {
            "cgroup_mutations_attempted": False,
            "signals_sent": [],
            "gpu_workload_launched": False,
            "nvidia_queries_read_only": True,
            "fixture_children_reaped_by_normal_wait": bool(
                setsid_fixture["assertions"]["child_reaped_by_normal_wait"]
            ),
        },
        "gate": gate,
        "limitations": [
            "Delegation writability is only a non-mutating access observation, not proof that child cgroups or PID migration are permitted.",
            "The setsid fixture demonstrates process-group escape but does not measure or contain descendants.",
            "GPU occupancy and accounting mode are instantaneous read-only nvidia-smi observations, not process-tree attribution.",
            "No child-inclusive CPU, RSS, peak-memory, OOM, or GPU measurement is claimed.",
            "A passed audit status means the blocker probe completed; it does not mean the process-tree resource gate passed.",
        ],
    }


def write_immutable_report(output_directory: Path, record: dict[str, object]) -> Path:
    """Atomically create one report directory and refuse every overwrite."""

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_directory.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"immutable output directory already exists: {output_directory}"
        ) from error
    destination = output_directory / "run.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".run.json.", dir=output_directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_descriptor = os.open(output_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        try:
            output_directory.rmdir()
        except OSError:
            pass
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    args = parser.parse_args(argv)
    output_directory = args.output_directory.resolve()
    record = build_record(output_directory=output_directory)
    destination = write_immutable_report(output_directory, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "gate": record["gate"],
                "report": str(destination),
                "report_sha256": sha256_file(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
