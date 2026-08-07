#!/usr/bin/env python3
"""Reproduce the metadata-first SOAR or NVARC blocker gate.

The auditor opens only an explicit retained-text allowlist.  It never imports
or executes upstream code, opens restricted ARC/JSON/notebook/pickle/PDF/image
or archive leaves, enters descendants of NVARC gitlink roots, initializes a
model/provider/GPU, or uses the network.  A passing report reproduces static
blockers; it is not a solver smoke, prediction, score, benchmark, or
paper-result reproduction.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time
import types
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
AUDITOR_RELATIVE = "scripts/audit_batch_d_static_gates.py"
LAUNCHER_RELATIVE = "scripts/launch_batch_d_static_gate.py"
SUPPORT_RELATIVE = "scripts/audit_batch_c_static_gates.py"
SOURCE_LOCK_RELATIVE = "configs/source_locks.json"
AUDITOR_PATH = ROOT / AUDITOR_RELATIVE
LAUNCHER_PATH = ROOT / LAUNCHER_RELATIVE
SUPPORT_PATH = ROOT / SUPPORT_RELATIVE
EXPECTED_SUPPORT_SHA256 = (
    "8860877257cf2864ddf8304fdef407d76de72339b6aba9d47391db5a57c7626e"
)
EXPECTED_SOURCE_LOCK_SHA256 = (
    "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
)
MAX_BOOTSTRAP_BYTES = 4 * 1024 * 1024
MAX_BOUND_METADATA_BYTES = 4 * 1024 * 1024
TEST_OUTPUT_ROOT = Path("/tmp/arc-agi-eval-batch-d-tests")

PROFILES: dict[str, dict[str, Any]] = {
    "configs/soar_gate_runner_manifest_v1.json": {
        "manifest_path": "configs/soar_gate_runner_manifest_v1.json",
        "manifest_id": "soar-gate-runner-manifest-v1",
        "method_id": "soar",
        "config_path": "configs/soar_gate_v1.json",
        "config_id": "soar-source-artifact-dataset-label-api-code-resource-gate-v1",
        "scope": "source-artifact-dataset-label-api-code-resource-gate-audit-only",
        "report_namespace": "reports/soar",
        "expected_config_canonical_sha256": "9867f1c85644eb2a9a2a981191742bac248144124143991902fa728e7735889b",
        "retained_count": 28,
        "metadata_only_count": 18,
        "gitlink_count": 0,
        "positive_retained_suffixes": {".py", ".md", ".sh", ".txt"},
    },
    "configs/nvarc_gate_runner_manifest_v1.json": {
        "manifest_path": "configs/nvarc_gate_runner_manifest_v1.json",
        "manifest_id": "nvarc-gate-runner-manifest-v1",
        "method_id": "nvarc",
        "config_path": "configs/nvarc_gate_v1.json",
        "config_id": "nvarc-source-gitlink-artifact-dataset-label-code-resource-gate-v1",
        "scope": "source-gitlink-artifact-dataset-label-code-resource-gate-audit-only",
        "report_namespace": "reports/nvarc",
        "expected_config_canonical_sha256": "f2766a0fdd049bb951d19951c46751ceb3979be17c4c94ad3871503a5a11365f",
        "retained_count": 24,
        "metadata_only_count": 15,
        "gitlink_count": 7,
        "positive_retained_suffixes": {".py", ".md", ".sh", ".yaml", ".j2"},
    },
}

core: Any = None


def _require_file_safety_flags() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    if missing:
        raise RuntimeError(
            "Batch D auditor requires non-degrading file-safety flags: "
            + ", ".join(missing)
        )


def _stat_signature(info: os.stat_result) -> tuple[int, ...]:
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


def _bootstrap_secure_read(path: Path, *, max_bytes: int, field: str) -> bytes:
    if not path.is_absolute():
        raise ValueError(f"{field} path must be absolute")
    _require_file_safety_flags()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{field} must be a single-link regular file")
        if before.st_size > max_bytes:
            raise ValueError(f"{field} exceeds the byte limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{field} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{field} grew while reading")
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise ValueError(f"{field} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json(payload: bytes, field: str) -> Any:
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


def _execute_verified_support(payload: bytes) -> types.ModuleType:
    digest = hashlib.sha256(payload).hexdigest()
    module = types.ModuleType(f"_verified_batch_d_support_{digest}")
    module.__file__ = str(SUPPORT_PATH)
    module.__package__ = "scripts"
    module.__dict__["__verified_source_sha256__"] = digest
    exec(
        compile(payload, str(SUPPORT_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    return module


def _profile_from_context() -> dict[str, Any]:
    context = globals().get("__verified_runner_manifest_context__")
    if not isinstance(context, dict):
        raise ValueError("Batch D production audit must enter through the verified launcher")
    selected = context.get("selected_profile")
    if not isinstance(selected, dict):
        raise ValueError("Batch D launcher profile context is missing")
    manifest_path = selected.get("manifest_path")
    profile = PROFILES.get(manifest_path)
    public_profile = {
        key: profile[key]
        for key in (
            "manifest_path",
            "manifest_id",
            "method_id",
            "config_path",
            "report_namespace",
        )
    } if profile is not None else None
    if selected != public_profile:
        raise ValueError("Batch D launcher profile differs from the auditor profile")
    return profile


def bootstrap_verified_runtime(
    config_path: Path,
    runner_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    bytes,
    bytes,
    bytes,
    bytes,
    dict[str, Any],
]:
    global core
    profile = _profile_from_context()
    canonical_config = ROOT / profile["config_path"]
    canonical_manifest = ROOT / profile["manifest_path"]
    if os.path.abspath(config_path) != os.path.abspath(canonical_config):
        raise ValueError(f"production config path must equal {profile['config_path']}")
    if os.path.abspath(runner_manifest_path) != os.path.abspath(canonical_manifest):
        raise ValueError(
            f"production runner manifest must equal {profile['manifest_path']}"
        )

    context = globals()["__verified_runner_manifest_context__"]
    manifest_payload = context.get("manifest_payload")
    payloads = context.get("member_payloads")
    if not isinstance(manifest_payload, bytes) or not isinstance(payloads, dict):
        raise ValueError("Batch D verified-launcher context is incomplete")
    manifest = _strict_json(manifest_payload, "Batch D runner manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Batch D runner manifest must be an object")
    expected_paths = {
        LAUNCHER_RELATIVE,
        AUDITOR_RELATIVE,
        SUPPORT_RELATIVE,
        profile["config_path"],
        SOURCE_LOCK_RELATIVE,
    }
    members = manifest.get("members")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("manifest_id") != profile["manifest_id"]
        or manifest.get("method_id") != profile["method_id"]
        or manifest.get("member_count") != 5
        or not isinstance(members, list)
        or len(members) != 5
        or any(not isinstance(item, dict) for item in members)
        or {item.get("path") for item in members} != expected_paths
        or set(payloads) != expected_paths
    ):
        raise ValueError("Batch D runner manifest closure mismatch")
    by_path = {item["path"]: item for item in members}
    if len(by_path) != 5:
        raise ValueError("Batch D runner manifest paths are not unique")
    for path, payload in payloads.items():
        member = by_path[path]
        if (
            not isinstance(payload, bytes)
            or member.get("bytes") != len(payload)
            or member.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"Batch D runner member mismatch: {path}")
    if context.get("manifest_path") != profile["manifest_path"]:
        raise ValueError("Batch D launcher used a noncanonical manifest")
    manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
    if (
        context.get("manifest_sha256") != manifest_digest
        or context.get("operator_supplied_manifest_sha256") != manifest_digest
    ):
        raise ValueError("Batch D runner-manifest digest context mismatch")
    expected_execution = {
        "mode": "canonical-direct-script-source",
        "name_is_main": True,
        "spec_is_none": True,
        "cached_is_none": True,
        "argv0_is_canonical_script": True,
        "python_executable": "/usr/bin/python3",
        "required_python_executable": "/usr/bin/python3",
        "isolated": True,
        "ignore_environment": True,
        "no_user_site": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "sys_path_excludes_cwd_and_relative_entries": True,
    }
    if context.get("launcher_source_execution") != expected_execution:
        raise ValueError("Batch D launcher was not entered as canonical direct source")

    config_payload = payloads[profile["config_path"]]
    auditor_payload = payloads[AUDITOR_RELATIVE]
    support_payload = payloads[SUPPORT_RELATIVE]
    source_lock_payload = payloads[SOURCE_LOCK_RELATIVE]
    auditor_digest = hashlib.sha256(auditor_payload).hexdigest()
    support_digest = hashlib.sha256(support_payload).hexdigest()
    if context.get("executed_auditor_sha256") != auditor_digest:
        raise ValueError("executed Batch D auditor differs from the runner manifest")
    if support_digest != EXPECTED_SUPPORT_SHA256:
        raise ValueError("Batch D support SHA-256 mismatch")
    verified_core = _execute_verified_support(support_payload)
    if verified_core.__dict__.get("__verified_source_sha256__") != support_digest:
        raise RuntimeError("verified Batch D support provenance was lost")
    core = verified_core
    config = _strict_json(config_payload, "Batch D gate config")
    config = validate_config(config, profile)
    if by_path[profile["config_path"]].get("canonical_sha256") != core.canonical_sha256(config):
        raise ValueError("Batch D config canonical digest differs from runner manifest")
    return (
        profile,
        config,
        config_payload,
        auditor_payload,
        support_payload,
        source_lock_payload,
        manifest,
    )


def _text(payloads: dict[str, bytes], path: str) -> str:
    return core.source_text(payloads[path])


def _dotted_name(node: ast.AST) -> str | None:
    return core.dotted_name(node)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == name
    ]


def _expression(node: ast.AST) -> str:
    return ast.unparse(node)


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return None


def _argument_defaults(tree: ast.AST) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "add_argument":
            continue
        names = [_literal(arg) for arg in call.args]
        options = [name for name in names if isinstance(name, str) and name.startswith("-")]
        if not options:
            continue
        option = next((name for name in options if name.startswith("--")), options[-1])
        key = option.lstrip("-").replace("-", "_")
        default_node = next((kw.value for kw in call.keywords if kw.arg == "default"), None)
        if default_node is not None:
            result[key] = _literal(default_node)
    return result


def _require_marker(text: str, marker: str, field: str) -> bool:
    if marker not in text:
        raise ValueError(f"static marker missing for {field}")
    return True


def _root_license_paths(paths: set[str]) -> list[str]:
    return sorted(
        path
        for path in paths
        if "/" not in path
        and PurePosixPath(path).name.lower() in core.ROOT_LICENSE_NAMES
    )


def _parse_batch_d_tree(
    payload: bytes,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[dict[str, str]]]:
    if not payload or not payload.endswith(b"\0"):
        raise ValueError("Batch D Git tree listing is not NUL-terminated")
    blobs: dict[str, dict[str, str]] = {}
    gitlinks: dict[str, dict[str, str]] = {}
    entries: list[dict[str, str]] = []
    for raw in payload[:-1].split(b"\0"):
        match = core.GIT_TREE_RECORD_RE.fullmatch(raw)
        if match is None:
            raise ValueError("malformed Batch D Git tree entry")
        mode = match.group("mode").decode("ascii")
        object_type = match.group("type").decode("ascii")
        try:
            path = match.group("path").decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Batch D Git tree path is not UTF-8") from error
        core.safe_relative_path(path, "Batch D Git tree path")
        if path in blobs or path in gitlinks:
            raise ValueError(f"duplicate Batch D Git tree path: {path}")
        oid = match.group("oid").decode("ascii")
        entry = {
            "path": path,
            "mode": mode,
            "object_type": object_type,
            "object_oid": oid,
        }
        if object_type == "blob" and mode in {"100644", "100755"}:
            blobs[path] = {"path": path, "mode": mode, "blob_oid": oid}
        elif object_type == "commit" and mode == "160000":
            gitlinks[path] = {
                "path": path,
                "mode": mode,
                "object_type": object_type,
                "object_oid": oid,
            }
        else:
            raise ValueError("unsupported Batch D Git tree type/mode")
        entries.append(entry)
    entries.sort(key=lambda item: item["path"])
    return blobs, gitlinks, entries


def _walk_batch_d_worktree(
    root_fd: int,
    blobs: dict[str, dict[str, str]],
    gitlinks: dict[str, dict[str, str]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, tuple[int, ...]],
]:
    expected_files = set(blobs)
    expected_gitlinks = set(gitlinks)
    expected_directories: set[str] = set()
    for path in expected_files | expected_gitlinks:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            expected_directories.add("/".join(parts[:index]))
    files: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    signatures: dict[str, tuple[int, ...]] = {}

    def visit(descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(descriptor)):
            if not prefix and name == ".git":
                continue
            path = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            signatures[path] = core.stat_signature(info)
            if path in expected_gitlinks:
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"NVARC gitlink worktree leaf is not a directory: {path}")
                child = os.open(name, core.directory_flags(), dir_fd=descriptor)
                try:
                    if core.stat_signature(info) != core.stat_signature(os.fstat(child)):
                        raise RuntimeError(f"NVARC gitlink raced before inspection: {path}")
                    names = sorted(os.listdir(child))
                    if names:
                        raise ValueError(f"NVARC gitlink is initialized or nonempty: {path}")
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if core.stat_signature(after) != core.stat_signature(os.fstat(child)):
                        raise RuntimeError(f"NVARC gitlink raced after inspection: {path}")
                finally:
                    os.close(child)
                links.append(
                    {
                        "path": path,
                        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                        "kind": "empty-directory",
                        "entry_count": 0,
                        "gitlink_oid": gitlinks[path]["object_oid"],
                    }
                )
            elif stat.S_ISDIR(info.st_mode):
                if path not in expected_directories:
                    raise ValueError(f"unknown Batch D worktree directory: {path}")
                child = os.open(name, core.directory_flags(), dir_fd=descriptor)
                try:
                    if core.stat_signature(info) != core.stat_signature(os.fstat(child)):
                        raise RuntimeError(f"Batch D directory raced before visit: {path}")
                    visit(child, path)
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if core.stat_signature(after) != core.stat_signature(os.fstat(child)):
                        raise RuntimeError(f"Batch D directory raced after visit: {path}")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if path not in expected_files:
                    raise ValueError(f"unknown Batch D worktree file: {path}")
                if info.st_nlink != 1:
                    raise ValueError(f"Batch D worktree leaf has a hardlink alias: {path}")
                files[path] = {
                    "path": path,
                    "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                    "bytes": info.st_size,
                }
            else:
                raise ValueError(f"unsafe Batch D worktree entry type: {path}")

    visit(root_fd, "")
    if set(files) != expected_files:
        raise ValueError("Batch D tracked worktree files do not close over the Git blobs")
    if {item["path"] for item in links} != expected_gitlinks:
        raise ValueError("Batch D gitlink worktree directories do not match the Git tree")
    return files, sorted(links, key=lambda item: item["path"]), signatures


def _extension_counts(
    blob_paths: set[str], gitlink_count: int,
) -> dict[str, int]:
    result = core.extension_counts(blob_paths)
    if gitlink_count:
        result["<gitlink>"] = gitlink_count
        result = dict(sorted(result.items()))
    return result


def _verify_git_file(
    payload: bytes,
    signature: tuple[int, ...],
    contract: dict[str, Any],
    field: str,
) -> None:
    core.verify_exact_file_contract(payload, signature, contract, field)


def _read_bound_evidence(
    declarations: list[dict[str, Any]], ledger: Any
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    objects: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        declared = declaration["path"]
        path = Path(declared)
        if not path.is_absolute():
            path = ROOT / path
        payload = core.secure_read_absolute(
            path,
            role=declaration["role"],
            category="bound_metadata",
            max_bytes=MAX_BOUND_METADATA_BYTES,
            ledger=ledger,
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != declaration["sha256"]:
            raise ValueError(f"bound evidence SHA-256 mismatch: {declared}")
        parsed = core.strict_json(payload, declared)
        if not isinstance(parsed, dict):
            raise ValueError(f"bound evidence is not an object: {declared}")
        core.assert_subset(parsed, declaration["assertions"], declared)
        observations.append(
            {
                "path": declared,
                "role": declaration["role"],
                "bytes": len(payload),
                "sha256": digest,
                "assertions_matched": True,
            }
        )
        payloads[declared] = payload
        objects[declared] = parsed
    return observations, payloads, objects


def _parse_submodules(text: str) -> list[dict[str, str]]:
    current: str | None = None
    records: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        match = re.fullmatch(r'\[submodule "([^"]+)"\]', line)
        if match:
            current = match.group(1)
            if current in records:
                raise ValueError("duplicate .gitmodules section")
            records[current] = {"name": current}
            continue
        if not line:
            continue
        if current is None or "=" not in line:
            raise ValueError("malformed .gitmodules content")
        key, value = [part.strip() for part in line.split("=", 1)]
        if key not in {"path", "url"} or key in records[current]:
            raise ValueError("unsupported or duplicate .gitmodules key")
        records[current][key] = value
    result = []
    for name in sorted(records):
        item = records[name]
        if set(item) != {"name", "path", "url"} or item["name"] != item["path"]:
            raise ValueError("incomplete or path-mismatched .gitmodules entry")
        result.append(item)
    return result


def analyze_soar(
    parsed: dict[str, ast.Module],
    payloads: dict[str, bytes],
    tree: dict[str, dict[str, str]],
    asset_status: dict[str, Any],
) -> dict[str, Any]:
    readme = _text(payloads, "README.md")
    license_text = _text(payloads, "LICENSE.md")
    requirements_text = _text(payloads, "requirements.txt")
    setup_text = _text(payloads, "setup.py")
    preprocess_text = _text(payloads, "soar/preprocess.py")
    sample_text = _text(payloads, "soar/inference/sample_phase.py")
    llm_text = _text(payloads, "soar/llm_utils.py")
    active_exec_text = _text(payloads, "soar/sandbox/execute_code_less_safe.py")
    alternate_exec_text = _text(payloads, "soar/sandbox/execute_code.py")
    api_text = _text(payloads, "soar/api.py")
    shell_text = _text(payloads, "experience/qwen.sh")
    tutorial_text = _text(payloads, "experience/tuto_expe.md")
    training_utils_text = _text(payloads, "soar/training/utils_process_data.py")
    training_runner_text = _text(payloads, "soar/training/train_unsloth.py")

    requirements = [
        line.strip()
        for line in requirements_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    preprocess_load_lines = sorted(
        call.lineno for call in _calls(parsed["soar/preprocess.py"], "json.load")
    )
    pickle_load_count = sum(len(_calls(tree_node, "pickle.load")) for tree_node in parsed.values())
    pickle_dump_count = sum(len(_calls(tree_node, "pickle.dump")) for tree_node in parsed.values())
    raw_exec_sites = []
    for path, active in (
        ("soar/sandbox/execute_code.py", False),
        ("soar/sandbox/execute_code_less_safe.py", True),
    ):
        calls = _calls(parsed[path], "exec")
        if len(calls) != 1 or len(calls[0].args) < 2:
            raise ValueError(f"unexpected SOAR exec shape: {path}")
        raw_exec_sites.append(
            {
                "path": path,
                "line": calls[0].lineno,
                "active": active,
                "globals_expression": _expression(calls[0].args[1]),
            }
        )
    api_sinks: list[dict[str, Any]] = []
    for call_name in (
        "execute_shell_command",
        "requests.get",
        "client.chat.completions.create",
        "subprocess.Popen",
    ):
        for call in _calls(parsed["soar/api.py"], call_name):
            api_sinks.append(
                {"path": "soar/api.py", "line": call.lineno, "call": call_name}
            )
    api_sinks.sort(key=lambda item: item["line"])
    sample_defaults_all = _argument_defaults(parsed["soar/inference/sample_phase.py"])
    rex_defaults_all = _argument_defaults(parsed["soar/repair/rex_inference.py"])
    sample_keys = (
        "path_model", "n_gpu", "model_len", "fp8", "k", "bs_inference",
        "temperature", "top_p", "min_p", "top_k", "max_tokens", "split",
        "use_fewshot_example", "smart_inference", "seed", "gpu_mem",
    )
    rex_keys = (
        "path_model", "n_gpu", "model_len", "fp8", "temperature", "top_p",
        "min_p", "top_k", "max_tokens", "split", "sampling_method",
        "correctness", "total_budget", "n_completion", "use_prev_gen",
        "smart_inference", "seed", "gpu_mem",
    )
    sample_defaults = {key: sample_defaults_all[key] for key in sample_keys}
    rex_defaults = {key: rex_defaults_all[key] for key in rex_keys}
    split_values = sorted(set(re.findall(r'"(train|val|train_eval)"', sample_text)))
    imported_less_safe = []
    for path in ("soar/inference/sample_phase.py", "soar/repair/rex.py"):
        for node in ast.walk(parsed[path]):
            if isinstance(node, ast.ImportFrom) and node.module == "soar.sandbox.execute_code_less_safe":
                imported_less_safe.append({"path": path, "line": node.lineno})
    model_checkpoint_suffixes = {
        ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
    }
    import_time_risky = [
        "soar/post_process/dedup.py",
        "soar/post_process/merge_filter.py",
        "soar/post_process/process_repair_for_training.py",
        "soar/post_process/process_sample_for_training.py",
        "soar/repair/rex_inference.py",
        "soar/training/train_unsloth.py",
    ]
    for path in import_time_risky:
        if not core.module_scope_call_names(parsed[path]):
            raise ValueError(f"expected import-time SOAR calls disappeared: {path}")
    created_at = asset_status.get("created_at")
    if not isinstance(created_at, str) or not re.match(r"^\d{4}-", created_at):
        raise ValueError("SOAR asset status creation time is malformed")
    pickle_path = "soar/inference/train_solutions.pkl"
    return {
        "readme_icml_2025_link": _require_marker(readme, "icml.cc/virtual/2025", "SOAR ICML year"),
        "readme_arc_agi_1_claim": _require_marker(readme, "ARC-AGI-1", "SOAR benchmark"),
        "official_arc_prize_entry_evidence_detected": bool(
            re.search(r"official ARC Prize (entry|submission)", readme, re.I)
        ),
        "readme_numeric_score_or_percent_detected": bool(
            re.search(r"(?:\bscore\b.{0,20}\d|\d+(?:\.\d+)?\s*%)", readme, re.I)
        ),
        "readme_score_claim_is_qualitative_unverified": _require_marker(
            readme, "outperforming much larger closed source LLMs", "SOAR qualitative claim"
        ),
        "asset_status_created_year": int(created_at[:4]),
        "asset_status_total_asset_count": len(asset_status.get("assets", {})),
        "asset_status_weight_download_deferred": _require_marker(
            asset_status.get("ready_definition", ""), "weight download is deferred", "SOAR deferred weights"
        ),
        "tracked_model_or_checkpoint_file_count": sum(
            PurePosixPath(path).suffix.lower() in model_checkpoint_suffixes for path in tree
        ),
        "tracked_pickle_file_count": int(pickle_path in tree),
        "tracked_pickle_path": pickle_path,
        "tracked_pickle_bytes": 32874958 if pickle_path in tree else None,
        "tracked_pickle_blob_oid": tree[pickle_path]["blob_oid"],
        "root_license_mit": license_text.startswith("The MIT License (MIT)\n"),
        "requirements_entries": requirements,
        "requirements_entry_count": len(requirements),
        "unpinned_requirement_count": sum("==" not in line for line in requirements),
        "dependency_hash_count": sum("--hash=" in line for line in requirements),
        "dependency_lock_has_transitive_hash_closure": bool(requirements) and all(
            "==" in line and "--hash=" in line for line in requirements
        ),
        "setup_install_requires_empty": _require_marker(setup_text, "install_requires=[", "SOAR setup requirements")
        and not re.search(r"install_requires=\[\s*[^\]\s#]", setup_text),
        "native_loader_challenge_paths": sorted(set(re.findall(r'arc-agi_(?:training|evaluation|test)_challenges\.json', preprocess_text))),
        "native_loader_solution_paths": sorted(set(re.findall(r'arc-agi_(?:training|evaluation)_solutions\.json', preprocess_text))),
        "preprocess_json_load_call_lines": preprocess_load_lines,
        "preprocess_merges_training_ground_truth": _require_marker(
            preprocess_text, "merge_GT(data_train,data_train_GT)", "SOAR train GT merge"
        ),
        "preprocess_merges_evaluation_ground_truth": _require_marker(
            preprocess_text, "merge_GT(data_val,data_val_GT)", "SOAR eval GT merge"
        ),
        "merge_gt_writes_test_output": _require_marker(
            preprocess_text,
            "data_merged[puzzle]['test'][i]['output']=data_GT[puzzle][i]",
            "SOAR GT output injection",
        ),
        "native_checker_reads_train_output": '"train"' in active_exec_text and '"output"' in active_exec_text,
        "native_checker_reads_test_output": '"test"' in active_exec_text and '"output"' in active_exec_text,
        "test_oracle_metric_detected": "correct_test_input" in llm_text,
        "sample_runner_split_values": split_values,
        "sample_runner_heldout_test_split_detected": "test" in split_values,
        "training_combines_arc1_and_arc2_train_eval_labels": all(
            marker in training_runner_text
            for marker in (
                "train_data_arc2",
                "val_data_arc2",
                "train_val_data_arc_1",
                "train_val_data_all.update(train_val_data_arc_1)",
            )
        ),
        "active_less_safe_executor_imports": imported_less_safe,
        "raw_exec_sites": raw_exec_sites,
        "active_executor_constructs_self_globals": _require_marker(active_exec_text, "self.globals", "SOAR constructed globals"),
        "active_executor_exec_uses_self_globals": "exec(code, self.globals" in active_exec_text,
        "active_executor_banned_modules": sorted(
            set(re.findall(r"['\"](os|sys|subprocess|shutil|socket)['\"]", active_exec_text))
        ),
        "active_executor_isolation_terms": {
            term: term in active_exec_text
            for term in ("ProcessPoolExecutor", "Pool", "RLIMIT_AS", "seccomp", "unshare", "setns", "chroot", "setsid", "killpg")
        },
        "api_egress_and_process_sinks": api_sinks,
        "sglang_binds_all_interfaces": "--host 0.0.0.0" in api_text,
        "vllm_binds_all_interfaces": (
            'host="0.0.0.0"' in api_text or "'--host', '0.0.0.0'" in api_text
        ),
        "completion_retry_min_seconds": 30 if "min=30" in api_text else None,
        "completion_retry_max_seconds": 600 if "max=600" in api_text else None,
        "pre_request_budget_reservation_detected": bool(re.search(r"reserve.*(?:token|request|usd)|budget.*before", api_text, re.I)),
        "llm_server_default_max_timeout_seconds": 10800 if "60*60*3" in api_text else None,
        "llm_server_default_max_workers": 64 if "max_workers = 64" in api_text else None,
        "llm_server_constructor_generates_hello": 'self.generate(["Hello"])' in api_text,
        "pickle_load_call_count": pickle_load_count,
        "pickle_dump_call_count": pickle_dump_count,
        "import_time_risky_module_count": len(import_time_risky),
        "import_time_risky_modules": import_time_risky,
        "main_guard_modules": ["soar/inference/sample_phase.py"] if 'if __name__ == "__main__"' in sample_text else [],
        "sample_native_output_is_pickle": (
            "save_pkl_secure(path_pkl_save" in sample_text and ".pkl" in sample_text
        ),
        "sample_json_dump_call_count": len(_calls(parsed["soar/inference/sample_phase.py"], "json.dump")),
        "official_submission_json_writer_detected": "submission.json" in sample_text and bool(_calls(parsed["soar/inference/sample_phase.py"], "json.dump")),
        "sample_selection_reads_test_correctness": "correct_test_input" in sample_text,
        "sample_selects_n_best_codes": 2 if "n_best_codes=2" in sample_text else None,
        "sample_defaults": sample_defaults,
        "rex_defaults": rex_defaults,
        "readme_model_gpu_guidance": {
            "up_to_7b_gpu_count": 1,
            "up_to_14b_gpu_count": 2,
            "above_14b_gpu_count": 4,
        } if all(marker in tutorial_text for marker in ("7B", "14B", "4 GPU")) else None,
        "shell_recipe_huggingface_download_call_count": shell_text.count("huggingface-cli download"),
    }


def _yaml_scalar(text: str, key: str) -> Any:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", text)
    if match is None:
        raise ValueError(f"YAML scalar missing: {key}")
    value = match.group(1).strip().strip('"')
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _torch_load_records(parsed: dict[str, ast.Module], paths: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        for call in _calls(parsed[path], "torch.load"):
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
            result.append(
                {
                    "path": path,
                    "lineno": call.lineno,
                    "map_location": _literal(keywords.get("map_location")),
                    "weights_only": "absent" if "weights_only" not in keywords else _literal(keywords["weights_only"]),
                }
            )
    return result


def analyze_nvarc(
    parsed: dict[str, ast.Module],
    payloads: dict[str, bytes],
    tree: dict[str, dict[str, str]],
    gitlinks: dict[str, dict[str, str]],
    gitlink_worktree: list[dict[str, Any]],
    asset_status: dict[str, Any],
) -> dict[str, Any]:
    readme = _text(payloads, "README.md")
    arc1_readme = _text(payloads, "ARC-AGI1/README.md")
    trm_readme = _text(payloads, "TRM/README.md")
    builder_text = _text(payloads, "SDG/scripts/build_datasets.py")
    puzzle_text = _text(payloads, "SDG/scripts/puzzle.py")
    input_text = _text(payloads, "SDG/scripts/generate_input_grids.py")
    output_text = _text(payloads, "SDG/scripts/generate_output_grids.py")
    pairs_text = _text(payloads, "SDG/scripts/make_pairs.py")
    sft_text = _text(payloads, "ARChitects/sft_mg.yaml")
    sft_shell = _text(payloads, "ARChitects/run_sft_4b.sh")
    sft_runner = _text(payloads, "ARChitects/run_sft.py")
    trm_paths = ("TRM/eval-arc-k-10.py", "TRM/pretrain-no-eval.py")
    trm_texts = "\n".join(_text(payloads, path) for path in trm_paths)
    loads = _torch_load_records(parsed, trm_paths)
    raw_calls = _calls(parsed["SDG/scripts/puzzle.py"], "exec")
    if len(raw_calls) != 1:
        raise ValueError("NVARC raw exec count changed")
    raw_call = raw_calls[0]
    defaults_input = _argument_defaults(parsed["SDG/scripts/generate_input_grids.py"])
    defaults_output = _argument_defaults(parsed["SDG/scripts/generate_output_grids.py"])
    defaults_pairs = _argument_defaults(parsed["SDG/scripts/make_pairs.py"])
    dataset_paths_match = re.search(
        r"train_dataset_path:\s*\n(?P<body>(?:\s+- [^\n]+\n)+)", sft_text
    )
    if dataset_paths_match is None:
        raise ValueError("NVARC SFT training dataset list is missing")
    training_paths = [
        line.strip()[2:].strip()
        for line in dataset_paths_match.group("body").splitlines()
    ]
    shell_config_match = re.search(r"--config\s+(\S+)", sft_shell)
    if shell_config_match is None:
        raise ValueError("NVARC SFT launcher config path is missing")
    shell_config = shell_config_match.group(1)
    def shell_int(key: str) -> int:
        match = re.search(rf"{re.escape(key)}=(\d+)", sft_shell)
        if match is None:
            raise ValueError(f"NVARC shell override missing: {key}")
        return int(match.group(1))
    model_match = re.search(r'policy\.model_name="([^"]+)"', sft_shell)
    if model_match is None:
        raise ValueError("NVARC effective model path is missing")
    pass_values: dict[int, float] = {}
    for k, value in re.findall(r"'ARC/pass@(\d+)':\s*([0-9.]+)", trm_readme):
        pass_values[int(k)] = float(value)
    leaderboard_match = re.search(r"scored\s+([0-9.]+)\s+on Kaggle public leaderboard", trm_readme)
    if leaderboard_match is None:
        raise ValueError("NVARC TRM README leaderboard self-report is missing")
    blob_names = set(tree)
    dependency_names = {
        "requirements.txt", "requirements-dev.txt", "pyproject.toml", "poetry.lock",
        "Pipfile.lock", "environment.yml", "environment.yaml", "uv.lock",
    }
    dependency_manifests = [path for path in blob_names if PurePosixPath(path).name in dependency_names]
    torch_save_count = sum(len(_calls(parsed[path], "torch.save")) for path in trm_paths)
    gitmodule_records = _parse_submodules(_text(payloads, ".gitmodules"))
    if {item["path"] for item in gitmodule_records} != set(gitlinks):
        raise ValueError("NVARC .gitmodules paths differ from gitlinks")
    return {
        "root_license_file_count": len(_root_license_paths(blob_names)),
        "dependency_manifest_file_count": len(dependency_manifests),
        "dependency_hash_count": 0,
        "dependency_lock_has_transitive_hash_closure": False,
        "gitlink_count": len(gitlinks),
        "gitlink_worktree_empty_directory_count": sum(item["entry_count"] == 0 for item in gitlink_worktree),
        "asset_status_total_asset_count": len(asset_status.get("assets", {})),
        "readme_arc_prize_2025_submission_claim": _require_marker(readme, "NVARC submissions to the", "NVARC submission self-claim"),
        "official_arc_prize_entry_evidence_detected": bool(re.search(r"official (?:competition )?(?:entry|result)", readme, re.I)),
        "readme_scores_are_unverified_component_self_reports": bool(pass_values) and leaderboard_match is not None,
        "reported_trm_arc_pass_at_1": pass_values[1],
        "reported_trm_arc_pass_at_2": pass_values[2],
        "reported_trm_arc_pass_at_5": pass_values[5],
        "reported_trm_arc_pass_at_10": pass_values[10],
        "reported_trm_arc_pass_at_100": pass_values[100],
        "reported_trm_arc_pass_at_1000": pass_values[1000],
        "reported_trm_kaggle_public_leaderboard_value": float(leaderboard_match.group(1)),
        "nvarc_ensemble_score_reported_in_retained_text": bool(re.search(r"NVARC.{0,80}(?:score|leaderboard).{0,20}\d", readme, re.I | re.S)),
        "readme_arc1_public_evaluation_test_time_finetuning": _require_marker(arc1_readme, "test time fine tuning with the public evaluation data", "NVARC ARC1 public TTT"),
        "readme_synthetic_puzzle_count": 103000 if "103k synthetic puzzles" in readme else None,
        "readme_augmented_puzzle_count": 3200000 if "3.2M augmented puzzles" in readme else None,
        "arc_builder_arc2_evaluation_glob": "external/ARC-AGI-2/data/evaluation/*.json" if "external/ARC-AGI-2/data/evaluation/*.json" in builder_text else None,
        "arc_builder_arc2_training_glob": "external/ARC-AGI-2/data/training/*.json" if "external/ARC-AGI-2/data/training/*.json" in builder_text else None,
        "arc_builder_combines_train_and_test_pairs": _require_marker(builder_text, 'pairs = data["train"] + data["test"]', "NVARC label combination"),
        "arc_builder_evaluation_num_samples": 6 if 'num_samples=6' in builder_text else None,
        "arc_builder_evaluation_output_path": "data/grids_v15/arc2_evaluation6" if 'f"{output_path}/arc2_evaluation6"' in builder_text else None,
        "arc_builder_training_output_path": "data/grids_v15/arc2_training" if 'f"{output_path}/arc2_training"' in builder_text else None,
        "sft_validation_dataset_path": _yaml_scalar(sft_text, "val_dataset_path"),
        "sft_training_dataset_paths": training_paths,
        "sft_checkpoint_metric_name": _yaml_scalar(sft_text, "metric_name"),
        "sft_checkpoint_keep_top_k": _yaml_scalar(sft_text, "keep_top_k"),
        "sft_base_save_period": _yaml_scalar(sft_text, "save_period"),
        "sft_base_validation_period": _yaml_scalar(sft_text, "val_period"),
        "public_arc2_evaluation_used_for_validation": _yaml_scalar(sft_text, "val_dataset_path") == "data/grids_v15/arc2_evaluation6",
        "public_arc2_evaluation_labels_used_for_checkpoint_selection": (
            'pairs = data["train"] + data["test"]' in builder_text
            and _yaml_scalar(sft_text, "val_dataset_path") == "data/grids_v15/arc2_evaluation6"
            and _yaml_scalar(sft_text, "metric_name") == "val_loss"
            and _yaml_scalar(sft_text, "keep_top_k") == 3
        ),
        "comprehensive_dataset_overlap_manifest_detected": any("overlap" in PurePosixPath(path).name.lower() for path in blob_names),
        "raw_exec_call_count": len(raw_calls),
        "raw_exec_calls": [{"path": "SDG/scripts/puzzle.py", "lineno": raw_call.lineno, "expression": _expression(raw_call)}],
        "raw_exec_signal_timeout_only": "signal.alarm(timeout)" in puzzle_text and "exec(code, result)" in puzzle_text,
        "raw_exec_builtins_restricted": "__builtins__" in puzzle_text,
        "generated_code_filesystem_isolation_detected": any(term in puzzle_text for term in ("chroot", "pivot_root", "mount namespace")),
        "generated_code_network_isolation_detected": any(term in puzzle_text for term in ("network namespace", "seccomp", "unshare")),
        "generated_code_process_isolation_detected": any(term in puzzle_text for term in ("pid namespace", "setsid", "killpg")),
        "generated_code_memory_limit_detected": any(term in puzzle_text for term in ("RLIMIT_AS", "RLIMIT_DATA", "setrlimit")),
        "sdg_default_initial_seed_is_random": "random.randint(0, 10000)" in input_text,
        "sdg_input_completion_glob_sorted": "sorted(glob.glob(inputs_mask))" in input_text,
        "sdg_recorded_seed_is_post_incremented": "seed += 1" in input_text and "unique_grids.append((seed, grid))" in input_text,
        "sdg_default_input_grid_count": defaults_input["num_grids"],
        "sdg_output_code_candidate_limit": 20 if "for i in range(20):" in output_text else None,
        "sdg_min_solutions_per_puzzle": defaults_output["min_solutions_per_puzzle"],
        "sdg_min_majority_per_grid": defaults_pairs["min_majority_per_grid"],
        "sdg_min_pairs_per_puzzle": defaults_pairs["min_pairs_per_puzzle"],
        "sdg_min_correct_solutions": defaults_pairs["min_correct_solutions"],
        "sft_base_max_num_steps": _yaml_scalar(sft_text, "max_num_steps"),
        "sft_base_num_nodes": _yaml_scalar(sft_text, "num_nodes"),
        "sft_gpus_per_node": _yaml_scalar(sft_text, "gpus_per_node"),
        "sft_launcher_declared_config_path": shell_config,
        "sft_launcher_declared_config_path_exists": f"ARChitects/{shell_config.removeprefix('./')}" in blob_names,
        "sft_launcher_num_nodes_override": shell_int("cluster.num_nodes"),
        "sft_effective_gpu_count": shell_int("cluster.num_nodes") * int(_yaml_scalar(sft_text, "gpus_per_node")),
        "sft_launcher_max_num_steps": shell_int("sft.max_num_steps"),
        "sft_launcher_validation_period": shell_int("sft.val_period"),
        "sft_launcher_checkpoint_save_period": shell_int("checkpointing.save_period"),
        "sft_effective_model_path": model_match.group(1),
        "sft_launcher_global_batch_size": shell_int("policy.train_global_batch_size"),
        "sft_launcher_train_mb_tokens": shell_int("policy.sequence_packing.train_mb_tokens"),
        "sft_tracked_config_path": "ARChitects/sft_mg.yaml",
        "sft_wandb_enabled": _yaml_scalar(sft_text, "wandb_enabled"),
        "sft_offline_or_disabled_wandb_guard_detected": any(
            marker in sft_text + sft_shell
            for marker in ("WANDB_MODE", 'mode="offline"', 'mode="disabled"')
        ),
        "sft_preprocessor_prints_message_content_and_token_ids": all(marker in sft_runner for marker in ("content start", "token ids start", "token ids end")),
        "trm_torch_load_call_count": len(loads),
        "trm_torch_load_map_location_cuda_count": sum(item["map_location"] == "cuda" for item in loads),
        "trm_torch_load_weights_only_absent_count": sum(item["weights_only"] == "absent" for item in loads),
        "trm_torch_save_call_count": torch_save_count,
        "trm_checkpoint_save_model_state_only": trm_texts.count("torch.save(train_state.model.state_dict()") == 2,
        "trm_checkpoint_save_excludes_optimizer_state": trm_texts.count("torch.save(train_state.model.state_dict()") == 2 and "optimizers:" in trm_texts,
        "trm_checkpoint_save_excludes_train_step": trm_texts.count("torch.save(train_state.model.state_dict()") == 2 and "step:" in trm_texts,
        "trm_checkpoint_save_excludes_rng_state": trm_texts.count("torch.save(train_state.model.state_dict()") == 2 and "rng" not in "torch.save(train_state.model.state_dict()",
        "trm_rng_state_resume_contract_complete": False,
        "trm_eval_save_outputs_default": [],
        "trm_configured_evaluation_outputs_can_persist_batch_values": "for collection in (batch, preds):" in trm_texts and "torch.save(" in trm_texts,
        "trm_submission_k_override": 10 if "submission_K=10" in _text(payloads, "TRM/eval-arc-k-10.py") else None,
        "trm_hardcoded_cuda_runtime": all(item["map_location"] == "cuda" for item in loads) and ".cuda()" in trm_texts,
        "readme_trm_training_gpu_count": 8 if "8 H100 GPUs" in trm_readme else None,
        "readme_trm_training_gpu_model": "H100" if "8 H100 GPUs" in trm_readme else None,
        "readme_trm_source_puzzle_count": 4073 if "4073 puzzles" in trm_readme else None,
        "readme_trm_training_augmentation_count": 256 if "256 augmentations" in trm_readme else None,
        "readme_trm_augmented_puzzle_count": 1041207 if "1041207 puzzles" in trm_readme else None,
        "readme_trm_evaluation_augmentation_count": 128 if "128 augmentations per puzzle" in trm_readme else None,
        "readme_trm_training_global_batch_size": 3072 if "global_batch_size=3072" in trm_readme else None,
        "readme_trm_training_epochs": 10000 if "epochs=10000" in trm_readme else None,
        "readme_trm_expected_checkpoint_count": 10 if "save 10 checkpoints" in trm_readme else None,
        "readme_trm_evaluation_gpu_count": 4 if "We use 4 GPUs only" in trm_readme else None,
        "readme_trm_evaluation_epochs": 2000 if "epochs=2000" in trm_readme else None,
        "readme_trm_evaluation_global_batch_size": 128 if "global_batch_size=128" in trm_readme else None,
        "readme_trm_evaluation_halt_max_steps": 10 if "halt_max_steps=10" in trm_readme else None,
        "readme_trm_pinned_pip_requirement_count": len(re.findall(r"\b[a-zA-Z0-9_.-]+==[0-9]", trm_readme)),
        "architects_inference_runner_present_in_retained_text": any("inference" in path.lower() and path.startswith("ARChitects/") for path in blob_names if path.endswith(".py")),
        "full_ensemble_runner_present": any("ensemble" in path.lower() and path.endswith(".py") for path in blob_names),
        "fixed_two_attempt_submission_contract_documented": bool(re.search(r"two[- ]attempt|attempt_1.*attempt_2", readme, re.I | re.S)),
        "fixed_candidate_voting_and_tie_break_contract_documented": bool(re.search(r"tie[- ]break|fixed voting", readme, re.I)),
        "fixed_checkpoint_selection_policy_without_public_labels_documented": bool(re.search(r"checkpoint selection.*without.*public", readme, re.I | re.S)),
        "fixed_seed_repetition_policy_documented": bool(re.search(r"fixed seeds?|seed repetitions?", readme, re.I)),
        "static_storage_lower_bound_available": bool(re.search(r"storage lower bound|disk requirement", readme, re.I)),
    }


def validate_config(value: Any, profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Batch D gate config must be an object")
    expected_digest = profile["expected_config_canonical_sha256"]
    if expected_digest.startswith("TO_BE_PINNED"):
        raise ValueError("Batch D auditor config digest has not been pinned")
    if core.canonical_sha256(value) != expected_digest:
        raise ValueError("Batch D gate config differs from the hardcoded v1 contract")
    method_id = profile["method_id"]
    if (
        value.get("schema_version") != 1
        or value.get("config_id") != profile["config_id"]
        or value.get("method_id") != method_id
        or value.get("scope") != profile["scope"]
        or value.get("counted_toward_smoke") is not False
    ):
        raise ValueError("Batch D config identity or claim scope mismatch")
    if value.get("config_read_policy") != {
        "canonical_path": profile["config_path"],
        "alternate_paths_allowed": False,
    }:
        raise ValueError("Batch D canonical config policy mismatch")
    if value.get("static_audit_support") != {
        "path": SUPPORT_RELATIVE,
        "sha256": EXPECTED_SUPPORT_SHA256,
    }:
        raise ValueError("Batch D support binding mismatch")
    if value.get("source_lock") != {
        "path": SOURCE_LOCK_RELATIVE,
        "sha256": EXPECTED_SOURCE_LOCK_SHA256,
    }:
        raise ValueError("Batch D source-lock binding mismatch")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("Batch D source contract is missing")
    if source.get("expected_shallow_repository") is not True:
        raise ValueError("Batch D shallow-repository contract must be explicit")
    core.validate_hash(source.get("expected_revision"), "Batch D revision", core.GIT_SHA_RE)
    core.validate_hash(source.get("expected_commit_tree"), "Batch D tree", core.GIT_SHA_RE)
    for field in (
        "git_tree_manifest_sha256",
        "retained_manifest_sha256",
    ):
        core.validate_hash(source.get(field), f"Batch D {field}")
    worktree_field = "worktree_metadata_sha256" if method_id == "soar" else "blob_worktree_metadata_sha256"
    core.validate_hash(source.get(worktree_field), f"Batch D {worktree_field}")
    if method_id == "nvarc":
        core.validate_hash(source.get("gitlink_worktree_metadata_sha256"), "NVARC gitlink worktree")
    git_contract = source.get("git_metadata_contract")
    if not isinstance(git_contract, dict):
        raise ValueError("Batch D Git metadata contract is missing")
    shallow_contract = git_contract.get("shallow")
    if not isinstance(shallow_contract, dict) or set(shallow_contract) != {
        "path", "bytes", "sha256", "mode"
    }:
        raise ValueError("Batch D shallow-file contract is malformed")
    if (
        shallow_contract["path"] != ".git/shallow"
        or shallow_contract["bytes"] != 41
        or shallow_contract["mode"] != "0644"
    ):
        raise ValueError("Batch D shallow-file shape mismatch")
    core.validate_hash(shallow_contract["sha256"], "Batch D shallow-file hash")
    retained = source.get("retained_text")
    metadata_only = source.get("metadata_only_paths")
    gitlinks = source.get("gitlinks")
    if not isinstance(retained, list) or len(retained) != profile["retained_count"]:
        raise ValueError("Batch D retained-text count mismatch")
    if not isinstance(metadata_only, list) or len(metadata_only) != profile["metadata_only_count"]:
        raise ValueError("Batch D metadata-only count mismatch")
    if not isinstance(gitlinks, list) or len(gitlinks) != profile["gitlink_count"]:
        raise ValueError("Batch D gitlink count mismatch")
    retained_paths: set[str] = set()
    for item in retained:
        if not isinstance(item, dict) or set(item) != {"path", "role"}:
            raise ValueError("Batch D retained declaration malformed")
        path = core.safe_relative_path(item["path"], "Batch D retained path")
        suffix = PurePosixPath(path).suffix.lower()
        if path in retained_paths or not item["role"]:
            raise ValueError("Batch D retained path duplicated or role-less")
        if path != ".gitmodules" and suffix not in profile["positive_retained_suffixes"]:
            raise ValueError("Batch D retained path is outside the positive allowlist")
        if method_id == "nvarc" and path.startswith("SDG/prompts/"):
            raise ValueError("NVARC restricted prompt entered retained text")
        retained_paths.add(path)
    if len(set(metadata_only)) != len(metadata_only):
        raise ValueError("Batch D metadata-only paths are not unique")
    for path in metadata_only:
        core.safe_relative_path(path, "Batch D metadata-only path")
        if path in retained_paths:
            raise ValueError("Batch D retained/metadata partition overlaps")
    gitlink_paths: list[str] = []
    for item in gitlinks:
        if not isinstance(item, dict) or set(item) != {
            "path", "mode", "object_type", "object_oid"
        }:
            raise ValueError("Batch D gitlink declaration malformed")
        path = core.safe_relative_path(item["path"], "Batch D gitlink path")
        if (
            path in gitlink_paths
            or item["mode"] != "160000"
            or item["object_type"] != "commit"
        ):
            raise ValueError("Batch D gitlink declaration duplicated or unsupported")
        core.validate_hash(item["object_oid"], "Batch D gitlink OID", core.GIT_SHA_RE)
        gitlink_paths.append(path)
    opaque_paths = source.get("opaque_worktree_paths")
    if opaque_paths != gitlink_paths:
        raise ValueError("Batch D opaque worktree paths must exactly match gitlinks")
    if method_id == "soar":
        expected_blob_count = source.get("expected_tracked_file_count")
        expected_entry_count = expected_blob_count
        expected_blob_bytes = source.get("expected_tracked_bytes")
    else:
        expected_blob_count = source.get("expected_blob_file_count")
        expected_entry_count = source.get("expected_tracked_entry_count")
        expected_blob_bytes = source.get("expected_tracked_blob_bytes")
        if source.get("expected_gitlink_count") != len(gitlinks):
            raise ValueError("NVARC expected gitlink count mismatch")
    integer_contracts = {
        "expected blob count": expected_blob_count,
        "expected entry count": expected_entry_count,
        "expected blob bytes": expected_blob_bytes,
        "expected retained count": source.get("expected_retained_file_count"),
        "expected retained bytes": source.get("expected_retained_bytes"),
        "expected metadata-only count": source.get("expected_metadata_only_file_count"),
        "expected metadata-only bytes": source.get("expected_metadata_only_bytes"),
    }
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in integer_contracts.values()):
        raise ValueError("Batch D declared count/byte contract is malformed")
    if expected_blob_count != len(retained) + len(metadata_only):
        raise ValueError("Batch D declared blob count does not close its partition")
    if expected_entry_count != expected_blob_count + len(gitlinks):
        raise ValueError("Batch D declared entry count does not close its partition")
    if expected_blob_bytes != source["expected_retained_bytes"] + source["expected_metadata_only_bytes"]:
        raise ValueError("Batch D declared blob bytes do not close their partition")
    blockers = value.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ValueError("Batch D blockers are missing")
    blocker_ids = [item.get("id") for item in blockers if isinstance(item, dict)]
    if len(blocker_ids) != len(blockers) or len(set(blocker_ids)) != len(blocker_ids):
        raise ValueError("Batch D blocker IDs are malformed or duplicated")
    if not all(set(item) == {"id", "gate", "detail"} for item in blockers):
        raise ValueError("Batch D blocker fields mismatch")
    if value.get("expected_blocker_ids") != blocker_ids:
        raise ValueError("Batch D expected blocker IDs differ from the blocker list")
    expected_analysis = value.get("expected_analysis", value.get("expected_static_analysis"))
    if not isinstance(expected_analysis, dict) or not expected_analysis:
        raise ValueError("Batch D expected analysis is missing")
    for group in (value.get("external_metadata"), value.get("prior_reports")):
        if not isinstance(group, list) or not group:
            raise ValueError("Batch D bound evidence group is missing")
        for item in group:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "role", "assertions"}:
                raise ValueError("Batch D bound evidence declaration malformed")
            core.validate_hash(item["sha256"], "Batch D bound evidence hash")
            if not item["role"] or not isinstance(item["assertions"], dict):
                raise ValueError("Batch D bound evidence assertion is missing")
    return value


def _source_lock_entry(value: dict[str, Any], method_id: str) -> dict[str, Any]:
    entries = value.get("sources")
    if not isinstance(entries, dict) or method_id not in entries:
        raise ValueError(f"source lock has no {method_id} entry")
    entry = entries[method_id]
    if not isinstance(entry, dict):
        raise ValueError(f"{method_id} source-lock entry is malformed")
    return entry


def run_static_audit(
    profile: dict[str, Any],
    config: dict[str, Any],
    config_payload: bytes,
    source_lock_payload: bytes,
    ledger: Any,
    run_id: str,
) -> dict[str, Any]:
    method_id = profile["method_id"]
    source = config["source"]
    source_path = Path(source["repository_path"])
    source_lock = core.strict_json(source_lock_payload, "Batch D source lock")
    lock_entry = _source_lock_entry(source_lock, method_id)
    if lock_entry != config["source_lock_entry"]:
        raise ValueError("Batch D source-lock entry differs from the gate contract")
    if lock_entry.get("revision") != source["expected_revision"]:
        raise ValueError("Batch D source-lock revision mismatch")

    declarations = config["external_metadata"] + config["prior_reports"]
    bound_observations, bound_payloads, bound_objects = _read_bound_evidence(declarations, ledger)
    asset_path = config["external_metadata"][0]["path"]
    root_fd = core.open_absolute_directory(source_path)
    git_fd: int | None = None
    object_fd: int | None = None
    isolated_fd: int | None = None
    isolated_temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        core.verify_directory_identity(source_path, root_fd)
        git_info = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(git_info.st_mode):
            raise ValueError("Batch D .git is not a real directory")
        git_fd = os.open(".git", core.directory_flags(), dir_fd=root_fd)
        if core.stat_signature(git_info) != core.stat_signature(os.fstat(git_fd)):
            raise RuntimeError("Batch D .git raced before pinning")
        for forbidden in core.FORBIDDEN_GIT_PATHS:
            core.require_git_path_absent(git_fd, forbidden)
        git_contract = source["git_metadata_contract"]
        git_config, git_config_signature = core.secure_read_pinned_relative(
            git_fd, "config", role="git_local_config", category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES, ledger=ledger,
        )
        git_head, git_head_signature = core.secure_read_pinned_relative(
            git_fd, "HEAD", role="git_head", category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES, ledger=ledger,
        )
        git_shallow, git_shallow_signature = core.secure_read_pinned_relative(
            git_fd, "shallow", role="git_shallow", category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES, ledger=ledger,
        )
        _verify_git_file(git_config, git_config_signature, git_contract["local_config"], "Batch D Git config")
        _verify_git_file(git_head, git_head_signature, git_contract["head"], "Batch D Git HEAD")
        _verify_git_file(
            git_shallow,
            git_shallow_signature,
            git_contract["shallow"],
            "Batch D Git shallow boundary",
        )
        core.validate_git_config_entries(git_config, source["repository_url"], lock_entry["branch"])
        revision = source["expected_revision"]
        if git_shallow != f"{revision}\n".encode("ascii"):
            raise ValueError("Batch D shallow boundary is not the locked revision")
        allowed_heads = {
            f"{revision}\n".encode("ascii"),
            f"ref: refs/heads/{lock_entry['branch']}\n".encode("ascii"),
        }
        if git_head not in allowed_heads:
            raise ValueError("Batch D Git HEAD is not at the locked revision/branch")
        object_info = os.stat("objects", dir_fd=git_fd, follow_symlinks=False)
        if not stat.S_ISDIR(object_info.st_mode):
            raise ValueError("Batch D Git object store is not a directory")
        object_fd = os.open("objects", core.directory_flags(), dir_fd=git_fd)
        if core.stat_signature(object_info) != core.stat_signature(os.fstat(object_fd)):
            raise RuntimeError("Batch D Git object directory raced before pinning")
        isolated_temp = tempfile.TemporaryDirectory(prefix=f"arc-agi-eval-{method_id}-git-", dir="/tmp")
        isolated_fd = core.initialize_isolated_git_directory(Path(isolated_temp.name))
        allowed_git = {
            ("rev-parse", "--verify", revision),
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            ("ls-tree", "-rz", "--full-tree", revision),
        }
        objects_before = core.git_object_metadata(git_fd)
        observed_revision = core.run_git(
            isolated_fd, object_fd, ("rev-parse", "--verify", revision), allowed_git, ledger
        ).decode("ascii").strip()
        observed_tree = core.run_git(
            isolated_fd, object_fd, ("rev-parse", "--verify", f"{revision}^{{tree}}"), allowed_git, ledger
        ).decode("ascii").strip()
        tree_payload = core.run_git(
            isolated_fd, object_fd, ("ls-tree", "-rz", "--full-tree", revision), allowed_git, ledger
        )
        if observed_revision != revision or observed_tree != source["expected_commit_tree"]:
            raise ValueError("Batch D Git revision/tree mismatch")
        blobs, gitlinks, tree_manifest = _parse_batch_d_tree(tree_payload)
        if core.canonical_sha256(tree_manifest) != source["git_tree_manifest_sha256"]:
            raise ValueError("Batch D Git tree manifest mismatch")
        if _extension_counts(set(blobs), len(gitlinks)) != source["expected_extension_counts"]:
            raise ValueError("Batch D extension count mismatch")
        objects_after = core.git_object_metadata(git_fd)
        if objects_after != objects_before:
            raise RuntimeError("Batch D Git object store changed during metadata commands")
        worktree, gitlink_worktree, initial_signatures = _walk_batch_d_worktree(root_fd, blobs, gitlinks)
        worktree_manifest = [worktree[path] for path in sorted(worktree)]
        observed_blob_count = len(blobs)
        observed_entry_count = observed_blob_count + len(gitlinks)
        observed_blob_bytes = sum(item["bytes"] for item in worktree_manifest)
        if method_id == "soar":
            expected_blob_count = source["expected_tracked_file_count"]
            expected_entry_count = expected_blob_count
            expected_blob_bytes = source["expected_tracked_bytes"]
        else:
            expected_blob_count = source["expected_blob_file_count"]
            expected_entry_count = source["expected_tracked_entry_count"]
            expected_blob_bytes = source["expected_tracked_blob_bytes"]
            if len(gitlinks) != source["expected_gitlink_count"]:
                raise ValueError("NVARC observed gitlink count mismatch")
        if (
            observed_blob_count != expected_blob_count
            or observed_entry_count != expected_entry_count
            or observed_blob_bytes != expected_blob_bytes
        ):
            raise ValueError("Batch D observed tracked count/byte contract mismatch")
        worktree_digest_field = "worktree_metadata_sha256" if method_id == "soar" else "blob_worktree_metadata_sha256"
        if core.canonical_sha256(worktree_manifest) != source[worktree_digest_field]:
            raise ValueError("Batch D blob worktree metadata mismatch")
        if method_id == "nvarc":
            if gitlink_worktree != source["gitlink_worktree"]:
                raise ValueError("NVARC gitlink worktree declarations mismatch")
            if core.canonical_sha256(gitlink_worktree) != source["gitlink_worktree_metadata_sha256"]:
                raise ValueError("NVARC gitlink worktree digest mismatch")
        elif gitlinks or gitlink_worktree or source.get("gitlinks"):
            raise ValueError("SOAR unexpectedly contains gitlinks")
        for path, item in blobs.items():
            expected_mode = "0755" if item["mode"] == "100755" else "0644"
            if worktree[path]["mode"] != expected_mode:
                raise ValueError(f"Batch D worktree/Git mode mismatch: {path}")

        retained_policy = {item["path"]: item["role"] for item in source["retained_text"]}
        metadata_paths = set(source["metadata_only_paths"])
        if set(blobs) != set(retained_policy) | metadata_paths or set(retained_policy) & metadata_paths:
            raise ValueError("Batch D retained/metadata-only partition is not closed")
        declared_gitlinks = {item["path"]: item for item in source["gitlinks"]}
        if declared_gitlinks != gitlinks:
            raise ValueError("Batch D configured gitlinks differ from the Git tree")
        if source["opaque_worktree_paths"] != sorted(gitlinks):
            raise ValueError("Batch D opaque worktree paths differ from observed gitlinks")
        metadata_inventory = [
            {
                "path": path,
                "mode": worktree[path]["mode"],
                "bytes": worktree[path]["bytes"],
                "git_blob_oid": blobs[path]["blob_oid"],
                "worktree_bytes_read": False,
                "worktree_sha256": None,
            }
            for path in sorted(metadata_paths)
        ]
        expected_metadata_count = source["expected_metadata_only_file_count"]
        expected_metadata_bytes = source["expected_metadata_only_bytes"]
        if len(metadata_inventory) != expected_metadata_count or sum(item["bytes"] for item in metadata_inventory) != expected_metadata_bytes:
            raise ValueError("Batch D metadata-only count/byte contract mismatch")
        first_payloads = {
            path: core.secure_read_retained(root_fd, path, role, initial_signatures[path], ledger)
            for path, role in retained_policy.items()
        }
        first_manifest = core.retained_manifest(source["retained_text"], first_payloads, blobs)
        if len(first_manifest) != source["expected_retained_file_count"] or sum(item["bytes"] for item in first_manifest) != source["expected_retained_bytes"]:
            raise ValueError("Batch D retained count/byte contract mismatch")
        if core.canonical_sha256(first_manifest) != source["retained_manifest_sha256"]:
            raise ValueError("Batch D retained manifest digest mismatch")
        parsed = {
            path: core.parse_python(payload, path)
            for path, payload in first_payloads.items()
            if PurePosixPath(path).suffix.lower() == ".py"
        }
        if method_id == "soar":
            analysis = analyze_soar(parsed, first_payloads, blobs, bound_objects[asset_path])
            expected_analysis = config["expected_analysis"]
        else:
            analysis = analyze_nvarc(
                parsed, first_payloads, blobs, gitlinks, gitlink_worktree,
                bound_objects[asset_path],
            )
            expected_analysis = config["expected_static_analysis"]
        if analysis != expected_analysis:
            raise ValueError(
                f"{method_id} static analysis differs from the pinned expectation: "
                + json.dumps(analysis, sort_keys=True)
            )
        license_paths = _root_license_paths(set(blobs))
        if license_paths != source["expected_root_license_paths"]:
            raise ValueError("Batch D root-license path observation mismatch")

        terminal_worktree, terminal_gitlinks, terminal_signatures = _walk_batch_d_worktree(root_fd, blobs, gitlinks)
        if terminal_worktree != worktree or terminal_gitlinks != gitlink_worktree or terminal_signatures != initial_signatures:
            raise RuntimeError("Batch D worktree metadata changed during audit")
        terminal_payloads = {
            path: core.secure_read_retained(root_fd, path, role, terminal_signatures[path], ledger)
            for path, role in retained_policy.items()
        }
        terminal_manifest = core.retained_manifest(source["retained_text"], terminal_payloads, blobs)
        if terminal_payloads != first_payloads or terminal_manifest != first_manifest:
            raise RuntimeError("Batch D retained text changed during audit")
        terminal_git_config, terminal_config_signature = core.secure_read_pinned_relative(
            git_fd, "config", role="git_local_config_terminal", category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES, ledger=ledger,
        )
        terminal_git_head, terminal_head_signature = core.secure_read_pinned_relative(
            git_fd, "HEAD", role="git_head_terminal", category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES, ledger=ledger,
        )
        terminal_git_shallow, terminal_shallow_signature = core.secure_read_pinned_relative(
            git_fd, "shallow", role="git_shallow_terminal", category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES, ledger=ledger,
        )
        if (
            terminal_git_config != git_config
            or terminal_git_head != git_head
            or terminal_git_shallow != git_shallow
            or terminal_config_signature != git_config_signature
            or terminal_head_signature != git_head_signature
            or terminal_shallow_signature != git_shallow_signature
            or core.git_object_metadata(git_fd) != objects_before
        ):
            raise RuntimeError("Batch D Git metadata changed during audit")
        terminal_bound, terminal_bound_payloads, _ = _read_bound_evidence(declarations, ledger)
        if terminal_bound != bound_observations or terminal_bound_payloads != bound_payloads:
            raise RuntimeError("Batch D bound metadata changed during audit")
    finally:
        if isolated_fd is not None:
            os.close(isolated_fd)
        if isolated_temp is not None:
            isolated_temp.cleanup()
        if object_fd is not None:
            os.close(object_fd)
        if git_fd is not None:
            os.close(git_fd)
        os.close(root_fd)

    blockers = [
        {**item, "status": "blocked", "blocking": True}
        for item in config["blockers"]
    ]
    tracked_count = len(blobs) + len(gitlinks)
    source_observation = {
        "repository_path": source["repository_path"],
        "repository_url": source["repository_url"],
        "expected_revision": source["expected_revision"],
        "observed_revision": observed_revision,
        "expected_commit_tree": source["expected_commit_tree"],
        "observed_commit_tree": observed_tree,
        "repository_shallow": True,
        "shallow_boundary_sha256": hashlib.sha256(git_shallow).hexdigest(),
        "tracked_entry_count": tracked_count,
        "tracked_blob_file_count": len(blobs),
        "tracked_blob_bytes": sum(item["bytes"] for item in worktree_manifest),
        "tracked_gitlink_count": len(gitlinks),
        "git_tree_manifest_sha256": core.canonical_sha256(tree_manifest),
        "blob_worktree_metadata_sha256": core.canonical_sha256(worktree_manifest),
        "gitlink_worktree_metadata_sha256": core.canonical_sha256(gitlink_worktree),
        "extension_counts": _extension_counts(set(blobs), len(gitlinks)),
        "retained_text_file_count": len(first_manifest),
        "retained_text_bytes": sum(item["bytes"] for item in first_manifest),
        "retained_manifest_sha256": core.canonical_sha256(first_manifest),
        "metadata_only_file_count": len(metadata_inventory),
        "metadata_only_bytes": sum(item["bytes"] for item in metadata_inventory),
        "metadata_only_inventory": metadata_inventory,
        "gitlinks": [gitlinks[path] for path in sorted(gitlinks)],
        "gitlink_worktree": gitlink_worktree,
        "gitlink_root_directory_enumerated_count": len(gitlinks),
        "gitlink_descendant_entries_opened_count": 0,
        "gitlink_descendant_leaf_bytes_read": False,
        "root_license_paths": license_paths,
        "working_tree_all_files_byte_exact_verified": False,
    }
    integrity_detail = (
        "The exact shallow revision/tree, retained text, and restricted metadata inventory reproduced."
        if method_id == "soar"
        else "The exact shallow revision/tree, retained text, restricted metadata inventory, and gitlink root-empty metadata reproduced without entering gitlink descendants."
    )
    passed_gates = [
        {
            "id": "locked-source-integrity",
            "gate": "source_provenance",
            "status": "passed",
            "blocking": False,
            "detail": integrity_detail,
        }
    ]
    if method_id == "soar":
        passed_gates.append(
            {
                "id": "root-code-license-present",
                "gate": "source_license_presence",
                "status": "passed",
                "blocking": False,
                "detail": "The locked root LICENSE.md begins with MIT License; this does not clear bundled data, pickle, model, or dependency terms.",
            }
        )
    observation = {
        "method_id": method_id,
        "scope": profile["scope"],
        "source": source_observation,
        "static_analysis": analysis,
        "bound_evidence": bound_observations,
        "passed_gates": passed_gates,
        "blockers": blockers,
        "gate_summary": {"passed": len(passed_gates), "blocked": len(blockers)},
        "controls": config["controls"],
        "classification": config["classification_contract"],
        "benchmark_policy": config["benchmark_policy"],
        "prior_evidence_interpretation": config["prior_evidence_interpretation"],
    }
    return {
        "schema_version": 1,
        "config_id": profile["config_id"],
        "method_id": method_id,
        "run_id": run_id,
        "runner": "scripts.audit_batch_d_static_gates",
        "status": "passed",
        "method_gate_status": "blocked",
        "scope": profile["scope"],
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "solver_gate_passed": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "config": {
            "path": profile["config_path"],
            "file_sha256": hashlib.sha256(config_payload).hexdigest(),
            "canonical_sha256": core.canonical_sha256(config),
        },
        **observation,
        "observation_digest_sha256": core.canonical_sha256(observation),
        "read_ledger": ledger.snapshot(),
        "validation": {
            "canonical_config_bound": True,
            "support_source_bound": True,
            "source_lock_bound": True,
            "revision_and_tree_bound": True,
            "isolated_git_metadata_view_bound": True,
            "closed_worktree_metadata_bound": True,
            "retained_bytes_bound_twice": True,
            "restricted_worktree_content_not_opened": True,
            "gitlink_root_handling_matches_declared_scope": True,
            "gitlink_descendant_entries_not_opened": True,
            "gitlink_descendant_leaf_bytes_not_read": True,
            "bound_metadata_verified_twice": True,
            "pinned_ast_and_source_marker_analysis_matches": True,
            "terminal_state_matched_at_last_observation": True,
            "all_method_blockers_preserved": True,
        },
        "claim_boundary": (
            "This metadata-first audit makes byte-exact claims only for the explicit retained-text allowlist. "
            "Restricted ARC/JSON/notebook/pickle/PDF/image/archive leaves receive path/mode/Git-OID/size metadata claims only; where present, NVARC gitlink roots are enumerated only to prove they are empty, with no descendant entry opened or leaf bytes read. "
            "The method is classified in the 2025 paper/method bucket and the asset snapshot in 2026 within the bound workspace evidence; external official classification is out of scope for this gate. "
            "Network/GPU syscalls and Git object-store byte volume are unmeasured. Passing reproduces blockers and is not a solver smoke, prediction, score, benchmark, checkpoint run, or paper reproduction."
        ),
        "limitations": [
            "Restricted leaves were not opened and their worktree bytes were not checked against Git blobs.",
            "For NVARC, each gitlink root directory was opened and enumerated only to establish emptiness; no descendant entry was opened and no gitlink leaf bytes were read, initialized, fetched, imported, or executed. SOAR has no gitlinks.",
            "Static analysis equality covers pinned AST facts and source markers; it is not a proof of whole-program semantics.",
            "Git metadata subprocesses may read pinned object files; kernel-level object byte volume is not measured.",
            "Network and GPU use are not instrumented or prevented by an OS namespace; actual use remains unknown and only intentional audited requests/spend are zero.",
            "Two complete observations plus terminal rechecks narrow but do not eliminate filesystem TOCTOU risk.",
            "The launcher requires canonical /usr/bin/python3 -I -B -S entry, but launcher/interpreter/Git binaries are not independently preauthenticated.",
            "The operator manifest digest requires a subsequent input-freeze anchor; no external signature or transparency log is verified here.",
        ],
    }


def failure_record(
    profile: dict[str, Any], run_id: str, error: BaseException, ledger: Any
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_id": profile["config_id"],
        "method_id": profile["method_id"],
        "run_id": run_id,
        "runner": "scripts.audit_batch_d_static_gates",
        "status": "failed",
        "method_gate_status": "blocked",
        "scope": profile["scope"],
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "solver_gate_passed": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "failure": {"type": type(error).__name__, "message": str(error)},
        "read_ledger": ledger.snapshot(),
        "controls": {
            "network_used": None,
            "gpu_used": None,
            "upstream_imported": None,
            "upstream_executed": None,
            "restricted_content_opened": None,
            "gitlink_descendant_entries_opened": None,
            "gitlink_descendant_leaf_bytes_read": None,
            "provider_or_wandb_called": None,
            "solver_executed": None,
            "solver_prediction_produced": False,
            "measurement_status": "unknown-after-audit-failure",
        },
        "claim_boundary": "The static audit failed; no solver or prediction was run.",
    }


def validate_output_location(path: Path, profile: dict[str, Any]) -> None:
    parts = core._absolute_parts(path)
    if len(parts) < 2 or parts[-1] in {"", ".", ".."} or parts[-1].startswith("."):
        raise core.OutputPathError("output must use a visible fresh leaf name")
    absolute = Path(*parts)
    production_parent = ROOT / profile["report_namespace"]
    if Path(*parts[:-1]) == Path(*core._absolute_parts(production_parent)):
        return
    test_parent = TEST_OUTPUT_ROOT / profile["method_id"]
    if core.lexical_within(absolute, test_parent) and not core.lexical_equal(absolute, test_parent):
        return
    raise core.OutputPathError(
        f"output must be a fresh {profile['method_id']} report leaf or enter its dedicated Batch D test root"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        (
            profile,
            config,
            config_payload,
            auditor_payload,
            support_payload,
            source_lock_payload,
            runner_manifest,
        ) = bootstrap_verified_runtime(args.config, args.runner_manifest)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    started_at = core.utc_now()
    started_monotonic = time.monotonic()
    started_usage = core.usage_snapshot()
    output_path = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
    try:
        validate_output_location(output_path, profile)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    protected = {
        ROOT / profile["config_path"], ROOT / profile["manifest_path"],
        ROOT / SOURCE_LOCK_RELATIVE, AUDITOR_PATH, SUPPORT_PATH, LAUNCHER_PATH,
    }
    if any(core.lexical_equal(output_path, path) for path in protected):
        parser.error("output path overlaps a protected input")
    retained_policy = {item["path"]: item["role"] for item in config["source"]["retained_text"]}
    if hashlib.sha256(source_lock_payload).hexdigest() != EXPECTED_SOURCE_LOCK_SHA256:
        parser.error("Batch D source-lock SHA-256 mismatch")
    try:
        output = core.create_fresh_output(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    phase_ledgers: list[Any] = []
    ledger = core.ReadLedger(retained_policy)
    try:
        try:
            first_ledger = core.ReadLedger(retained_policy)
            phase_ledgers.append(first_ledger)
            first = run_static_audit(
                profile, config, config_payload, source_lock_payload, first_ledger,
                output_path.name,
            )
            first_digest = first["observation_digest_sha256"]
            replay_config = _bootstrap_secure_read(
                ROOT / profile["config_path"], max_bytes=core.MAX_CONFIG_BYTES,
                field="Batch D gate config replay",
            )
            replay_lock = _bootstrap_secure_read(
                ROOT / SOURCE_LOCK_RELATIVE, max_bytes=core.MAX_CONFIG_BYTES,
                field="Batch D source lock replay",
            )
            if replay_config != config_payload or replay_lock != source_lock_payload:
                raise RuntimeError("Batch D config or source lock changed between audits")
            ledger = core.ReadLedger(retained_policy)
            phase_ledgers.append(ledger)
            record = run_static_audit(
                profile, config, config_payload, source_lock_payload, ledger,
                output_path.name,
            )
            second_digest = record["observation_digest_sha256"]
            if first_digest != second_digest:
                raise RuntimeError("Batch D complete replay observation changed before commit")
            context = globals()["__verified_runner_manifest_context__"]
            member_payloads = context["member_payloads"]
            terminal_payloads = {
                profile["manifest_path"]: _bootstrap_secure_read(
                    ROOT / profile["manifest_path"], max_bytes=core.MAX_CONFIG_BYTES,
                    field="Batch D manifest terminal",
                ),
                profile["config_path"]: _bootstrap_secure_read(
                    ROOT / profile["config_path"], max_bytes=core.MAX_CONFIG_BYTES,
                    field="Batch D config terminal",
                ),
                SOURCE_LOCK_RELATIVE: _bootstrap_secure_read(
                    ROOT / SOURCE_LOCK_RELATIVE, max_bytes=core.MAX_CONFIG_BYTES,
                    field="Batch D source lock terminal",
                ),
                AUDITOR_RELATIVE: _bootstrap_secure_read(
                    AUDITOR_PATH, max_bytes=MAX_BOOTSTRAP_BYTES, field="Batch D auditor terminal",
                ),
                SUPPORT_RELATIVE: _bootstrap_secure_read(
                    SUPPORT_PATH, max_bytes=MAX_BOOTSTRAP_BYTES, field="Batch D support terminal",
                ),
                LAUNCHER_RELATIVE: _bootstrap_secure_read(
                    LAUNCHER_PATH, max_bytes=MAX_BOOTSTRAP_BYTES, field="Batch D launcher terminal",
                ),
            }
            if terminal_payloads.pop(profile["manifest_path"]) != context["manifest_payload"]:
                raise RuntimeError("Batch D runner manifest changed during audit")
            if any(terminal_payloads[path] != member_payloads[path] for path in terminal_payloads):
                raise RuntimeError("A Batch D runner member changed during audit")
            runner_members = {item["role"]: item for item in runner_manifest["members"]}
            record["runner_provenance"] = {
                "path": AUDITOR_RELATIVE,
                "bytes": len(auditor_payload),
                "sha256": hashlib.sha256(auditor_payload).hexdigest(),
                "expected_sha256": runner_members["auditor"]["sha256"],
                "manifest_bound": True,
                "executed_from_verified_source_bytes": True,
                "terminal_bytes_equal": True,
            }
            record["support_provenance"] = {
                "path": SUPPORT_RELATIVE,
                "bytes": len(support_payload),
                "sha256": hashlib.sha256(support_payload).hexdigest(),
                "expected_sha256": runner_members["support"]["sha256"],
                "loaded_via": "preverified-source-bytes-compile-exec",
                "normal_import_used": False,
                "pyc_used": False,
                "terminal_bytes_equal": True,
            }
            record["launcher_provenance"] = {
                "path": LAUNCHER_RELATIVE,
                "sha256": runner_members["launcher"]["sha256"],
                "canonical_direct_script_source": True,
                "isolated_python_flags_required": True,
                "terminal_file_matches_manifest": True,
                "executed_launcher_bytes_preauthenticated": False,
                "interpreter_binary_digest_verified": False,
                "source_execution_context": context["launcher_source_execution"],
            }
            record["runner_manifest_provenance"] = {
                "path": profile["manifest_path"],
                "sha256": context["manifest_sha256"],
                "operator_supplied_expected_sha256": context["operator_supplied_manifest_sha256"],
                "operator_supplied_digest_matched": True,
                "repository_external_signature_verified": False,
                "next_input_freeze_anchor_required": True,
                "release_evidence_status": "provisional-until-input-freeze-anchor",
                "members_sha256": runner_manifest["members_sha256"],
                "member_count": runner_manifest["member_count"],
                "launcher_sha256": runner_members["launcher"]["sha256"],
                "terminal_bytes_equal": True,
            }
            first_ledger_snapshot = first_ledger.snapshot()
            second_ledger_snapshot = ledger.snapshot()
            record["phase_read_ledgers"] = {
                "scope": "run_static_audit content and Git subprocess attempts only; launcher/bootstrap, replay bootstrap, and outer terminal manifest/member reads are not ledger-instrumented",
                "first_complete_observation": first_ledger_snapshot,
                "second_complete_observation": second_ledger_snapshot,
            }
            record["read_ledger_scope"] = "second_complete_observation; both complete observations are preserved under phase_read_ledgers"
            record["replay_consistency"] = {
                "complete_observation_count": 2,
                "first_observation_digest_sha256": first_digest,
                "second_observation_digest_sha256": second_digest,
                "equal": True,
                "first_read_ledger_sha256": core.canonical_sha256(first_ledger_snapshot),
                "second_read_ledger_sha256": core.canonical_sha256(second_ledger_snapshot),
            }
            record["commit_consistency"] = {
                "strategy": "two-complete-source-observations-plus-terminal-byte-recheck",
                "immutable_mount": False,
                "toctou_eliminated": False,
                "claim": "Inputs matched at the final observation before commit.",
            }
            record["validation"].update(
                {
                    "runner_sha256_manifest_bound": True,
                    "runner_manifest_members_verified_by_launcher": True,
                    "operator_supplied_manifest_digest_matched": True,
                    "launcher_direct_source_and_isolated_flags_required": True,
                    "launcher_terminal_file_matches_manifest": True,
                    "support_loaded_only_after_sha256_verification": True,
                    "double_complete_observation_equal": True,
                    "runner_manifest_terminally_stable_at_last_observation": True,
                }
            )
            returncode = 0
        except BaseException as error:
            aggregate = core.ReadLedger(retained_policy)
            for phase in phase_ledgers:
                aggregate.file_reads.extend(phase.file_reads)
                aggregate.git_processes.extend(phase.git_processes)
            ledger = aggregate
            record = failure_record(profile, output_path.name, error, ledger)
            returncode = 1
        record["read_ledger"] = ledger.snapshot()
        record["started_at_utc"] = started_at
        record["ended_at_utc"] = core.utc_now()
        resources = core.usage_record(
            started_usage, core.usage_snapshot(), time.monotonic() - started_monotonic
        )
        resources.update(
            {
                "provider_requests": None,
                "currency_spend_usd": None,
                "gpu_used": None,
                "network_used": None,
                "network_usage_measurement": "not-instrumented",
                "gpu_usage_measurement": "not-instrumented",
                "intentional_provider_requests": 0,
                "intentional_currency_spend_usd": 0.0,
                "wall_and_cpu_deltas_include_bootstrap": False,
                "max_rss_scope": "process-lifetime-high-water-mark-may-include-bootstrap",
                "max_rss_includes_premeasurement_process_lifetime": True,
            }
        )
        record["resources"] = resources
        core.write_json_no_clobber(output, record)
        print(
            json.dumps(record, indent=2, sort_keys=True),
            file=sys.stdout if returncode == 0 else sys.stderr,
        )
        return returncode
    finally:
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())
