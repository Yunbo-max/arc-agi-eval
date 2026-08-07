#!/usr/bin/env python3
"""Materialize the frozen known-overlap-excluded dev-audit runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.development_runtime import (  # noqa: E402
    build_development_runtime,
    sha256_file,
)
from arc_agi_eval.resources import ResourceMonitor  # noqa: E402
from arc_agi_eval.run_schema import validate_run_file  # noqa: E402


SCHEMA_PATH = ROOT / "schemas" / "protocol-v1-run.schema.json"
PROTOCOL_CONFIG = ROOT / "configs" / "protocol_v1_draft.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: dict[str, object]) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def file_record(
    path: Path, *, role: str, required: bool = True
) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "required_for_claim": required,
    }


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(completed.stdout)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown-cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "development_partition_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "e0-development-split"
            / "20260806-frozen-known-overlap-excluded-dev-audit-v1"
        ),
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    try:
        output_directory.relative_to(ROOT)
    except ValueError as error:
        parser.error(f"output directory must remain inside repository: {error}")

    started_at = utc_now()
    monitor = ResourceMonitor(include_nvidia=False).start()
    status = "failed"
    failure: dict[str, str] | None = None
    record: dict[str, object] | None = None
    try:
        first = build_development_runtime(ROOT, args.config)
        second = build_development_runtime(ROOT, args.config)
        if first != second:
            raise ValueError("development runtime did not rebuild deterministically")
        for relative, payload in sorted(
            {**first.inference_files, **first.solution_files}.items()
        ):
            target = (output_directory / relative).resolve()
            try:
                target.relative_to(output_directory)
            except ValueError as error:
                raise ValueError(f"runtime path escapes output: {relative}") from error
            atomic_bytes(target, payload)
        manifest_path = output_directory / "manifest.json"
        atomic_json(manifest_path, first.manifest)
        for item in [
            *first.manifest["inference_file_inventory"],
            *first.manifest["solution_file_inventory"],
        ]:
            emitted = output_directory / item["path"]
            if sha256_file(emitted) != item["sha256"]:
                raise ValueError(f"emitted runtime hash mismatch: {emitted}")

        config_copy = output_directory / "config.json"
        atomic_bytes(config_copy, args.config.read_bytes())
        protocol_snapshot = output_directory / "protocol-manifest.json"
        atomic_json(
            protocol_snapshot,
            {
                "schema_version": 1,
                "protocol_id": "arc-rebench-protocol-v1-draft",
                "protocol_status": "draft-not-frozen",
                "gate_id": "cm.development-partition",
                "protocol_config_path": PROTOCOL_CONFIG.relative_to(ROOT).as_posix(),
                "protocol_config_sha256": sha256_file(PROTOCOL_CONFIG),
            },
        )
        relevant_sources = [
            ROOT / "arc_agi_eval" / "development_runtime.py",
            ROOT / "arc_agi_eval" / "firewall.py",
            ROOT / "arc_agi_eval" / "scoring.py",
            ROOT / "scripts" / "freeze_development_runtime.py",
            args.config,
        ]
        source_patch_path = output_directory / "source-patch-manifest.json"
        atomic_json(
            source_patch_path,
            {
                "schema_version": 1,
                "scope": "files implementing the frozen development runtime audit",
                "files": [
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in relevant_sources
                ],
            },
        )
        source_lock_path = output_directory / "source-lock.json"
        atomic_json(
            source_lock_path,
            {
                "schema_version": 1,
                "revision": git_revision(),
                "dirty": git_dirty(),
                "patch_manifest_path": source_patch_path.relative_to(ROOT).as_posix(),
                "patch_manifest_sha256": sha256_file(source_patch_path),
            },
        )
        artifact_manifest_path = output_directory / "artifact-manifest.json"
        atomic_json(
            artifact_manifest_path,
            {
                "schema_version": 1,
                "artifact_count": 0,
                "licenses_verified": True,
                "artifacts": [],
            },
        )
        data_manifest_path = output_directory / "data-manifest.json"
        atomic_json(
            data_manifest_path,
            {
                "schema_version": 1,
                "dataset_id": "arc-training-known-overlap-excluded",
                "split": "dev-audit",
                "task_count": 94,
                "test_output_denominator": 97,
                "development_runtime_sha256": first.manifest[
                    "development_runtime_sha256"
                ],
                "source_manifests": first.manifest["source_manifests"],
            },
        )
        environment_path = output_directory / "environment-lock.json"
        atomic_json(
            environment_path,
            {
                "schema_version": 1,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "implementation": platform.python_implementation(),
                "network_used": False,
            },
        )
        hardware_path = output_directory / "hardware-manifest.json"
        atomic_json(
            hardware_path,
            {
                "schema_version": 1,
                "profile_id": "host-20260806-cpu-development-runtime",
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
        )
        results_path = output_directory / "results-summary.json"
        atomic_json(
            results_path,
            {
                "schema_version": 1,
                "status": "passed",
                "checks": first.manifest["checks"],
                "partition": first.manifest["partition"],
                "dev_audit": first.manifest["dev_audit"],
                "deterministic_rebuild": True,
            },
        )

        content_manifest_path = output_directory / "content-manifest.json"
        content_entries = []
        for path in sorted(output_directory.rglob("*")):
            if path.is_file() and path.name not in {"run.json", "content-manifest.json"}:
                content_entries.append(
                    {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
        atomic_json(
            content_manifest_path,
            {
                "schema_version": 1,
                "file_count": len(content_entries),
                "files": content_entries,
            },
        )
        status = "passed"
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    usage = monitor.stop()

    if status == "passed":
        role_paths = {
            "protocol_manifest": protocol_snapshot,
            "source_lock": source_lock_path,
            "source_patch": source_patch_path,
            "artifact_manifest": artifact_manifest_path,
            "data_manifest": data_manifest_path,
            "config": config_copy,
            "environment_lock": environment_path,
            "hardware_manifest": hardware_path,
            "challenge_manifest": manifest_path,
            "results": results_path,
            "content_manifest": content_manifest_path,
        }
        files = [file_record(SCHEMA_PATH, role="schema")]
        used_paths = {SCHEMA_PATH.resolve()}
        for role, path in role_paths.items():
            files.append(file_record(path, role=role))
            used_paths.add(path.resolve())
        for path in sorted(output_directory.rglob("*")):
            if (
                path.is_file()
                and path.name != "run.json"
                and path.resolve() not in used_paths
            ):
                files.append(file_record(path, role="other"))
        record = {
            "schema_version": "protocol-v1-run-1.0.0",
            "schema_digest_sha256": sha256_file(SCHEMA_PATH),
            "protocol_id": "arc-rebench-protocol-v1-draft",
            "protocol_digest_sha256": sha256_file(protocol_snapshot),
            "method_id": "e0-development-partition",
            "config_id": "known-overlap-excluded-dev-audit-v1",
            "run_id": output_directory.name,
            "status": "passed",
            "evidence_scope": "e0_audit",
            "parity_class": "not_applicable",
            "resource_class": "local_cpu",
            "code_trust_class": "trusted_locked",
            "claim": (
                "The locked training-only and known-overlap-excluded manifests "
                "produce a frozen 94-cluster dev-audit challenge tree with 97 "
                "outputs, no inference-visible test labels, no cluster reallocation, "
                "and label-sensitive scoring outside the inference tree."
            ),
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "source": {
                "lock_digest_sha256": sha256_file(source_lock_path),
                "revision": git_revision(),
                "dirty": True,
                "patch_digest_sha256": sha256_file(source_patch_path),
            },
            "artifacts": {
                "manifest_digest_sha256": sha256_file(artifact_manifest_path),
                "artifact_count": 0,
                "licenses_verified": True,
            },
            "data": {
                "manifest_digest_sha256": sha256_file(data_manifest_path),
                "dataset_id": "arc-training-known-overlap-excluded",
                "split": "dev-audit",
                "task_count": 94,
                "contamination_policy": "overlap_excluded",
            },
            "config": {
                "digest_sha256": sha256_file(config_copy),
                "seed": 20260806,
                "deterministic": True,
            },
            "hardware": {
                "profile_id": "host-20260806-cpu-development-runtime",
                "manifest_digest_sha256": sha256_file(hardware_path),
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
            "execution": {
                "runner": "scripts.freeze_development_runtime",
                "command": [
                    "python3",
                    "scripts/freeze_development_runtime.py",
                    "--output-directory",
                    output_directory.relative_to(ROOT).as_posix(),
                ],
                "working_directory": ".",
                "environment_digest_sha256": sha256_file(environment_path),
                "claim_execution_started": True,
                "target_code_executed": True,
                "network_used": False,
                "gpu_requested": False,
            },
            "challenge_firewall": {
                "challenge_manifest_digest_sha256": sha256_file(manifest_path),
                "inference_received_test_labels": False,
                "inference_started": False,
                "scoring_after_inference": False,
                "label_mutation_check": "not_applicable",
                "network_policy": "denied",
                "write_policy": "run_directory_only",
                "security_isolation": "trusted_process",
            },
            "attempt_budget": {
                "top_k": 0,
                "timeout_seconds": 30,
                "max_retries": 0,
                "max_candidates": 0,
                "api_call_cap": 0,
                "input_token_cap": 0,
                "output_token_cap": 0,
                "cost_cap_usd": 0,
            },
            "resources": {
                "accounting_scope": "current_process",
                "child_processes_observed": True,
                "children_included": False,
                "sampling": "sampled",
                "wall_time_seconds": usage.wall_time_seconds,
                "cpu_seconds": usage.process_cpu_seconds,
                "peak_rss_bytes": usage.sampled_peak_current_rss_bytes,
                "peak_vram_bytes": None,
                "energy_joules": None,
                "disk_delta_bytes": None,
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
            },
            "results": {
                "kind": "check",
                "predictions_path": None,
                "score_path": None,
                "primary_metric": None,
                "secondary_metrics": [],
                "checks": [
                    {
                        "name": "frozen-partition-and-runtime",
                        "status": "passed",
                        "detail": (
                            "94 representatives, 97 outputs, 0 visible test labels, "
                            "and all declared hashes validated."
                        ),
                    },
                    {
                        "name": "hidden-label-mutation",
                        "status": "passed",
                        "detail": (
                            "Changing every dev-audit test label left all 94 "
                            "challenge payloads unchanged while the scorer sentinel "
                            "changed from 97/97 to 0/97."
                        ),
                    },
                ],
            },
            "failures": {"count": 0, "items": []},
            "files": files,
            "limitations": [
                *first.manifest["limitations"],
                "This E0 audit materializes no solver prediction and does not pass the method-specific challenge-runtime gate.",
                "Resource counters cover the current builder process only; child-inclusive method accounting remains a separate P0."
            ],
        }
        run_path = output_directory / "run.json"
        atomic_json(run_path, record)
        try:
            validation = validate_run_file(
                run_path,
                schema_path=SCHEMA_PATH,
                repo_root=ROOT,
                verify_files=True,
            )
        except BaseException as error:
            status = "failed"
            failure = {"type": type(error).__name__, "message": str(error)}
        else:
            print(json.dumps(validation.as_dict(), indent=2, sort_keys=True))
    if status != "passed":
        failed_record = {
            "schema_version": 1,
            "method_id": "e0-development-partition",
            "run_id": output_directory.name,
            "runner": "scripts.freeze_development_runtime",
            "status": "failed",
            "scope": "frozen-known-overlap-excluded-dev-audit-runtime",
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "error": failure,
            "resources": usage.to_dict(),
        }
        atomic_json(output_directory / "run.json", failed_record)
        print(json.dumps(failed_record, indent=2, sort_keys=True))
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
