#!/usr/bin/env python3
"""Materialize and independently audit the frozen 64-base-task IsoARC design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.isoarc_manifest import (  # noqa: E402
    audit_labeled_references,
    build_fixed64_design,
    canonical_sha256,
    sha256_file,
)
from arc_agi_eval.resources import ResourceMonitor  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
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
        (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "isoarc_fixed64.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "reports" / "e0-isoarc" / "20260806-fixed64-design-v1",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor(include_nvidia=False).start()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-isoarc",
        "run_id": output_directory.name,
        "runner": "scripts.build_fixed64_isoarc",
        "status": "failed",
        "scope": "frozen-fixed64-isoarc-input-and-round-trip-audit",
        "protocol_status": "draft-not-frozen",
        "started_at_utc": utc_now(),
        "solver_executed": False,
        "network_used": False,
        "gpu_requested": False,
    }
    try:
        first = build_fixed64_design(ROOT, args.config)
        frozen_root_before_audit = first.manifest["fixed64_manifest_sha256"]
        second = build_fixed64_design(ROOT, args.config)
        deterministic_rebuild = (
            first.manifest == second.manifest
            and first.variant_files == second.variant_files
        )
        if not deterministic_rebuild:
            raise ValueError("fixed64 design did not rebuild byte-identically")
        labeled_audit = audit_labeled_references(ROOT, first.manifest)
        if first.manifest["fixed64_manifest_sha256"] != frozen_root_before_audit:
            raise ValueError("labeled auditor modified the frozen design")

        for relative, payload in sorted(first.variant_files.items()):
            target = (output_directory / relative).resolve()
            try:
                target.relative_to(output_directory)
            except ValueError as error:
                raise ValueError(f"variant path escapes output: {relative}") from error
            atomic_bytes(target, payload)
        for item in first.manifest["variant_file_inventory"]:
            emitted = output_directory / item["path"]
            if sha256_file(emitted) != item["sha256"]:
                raise ValueError(f"emitted variant hash mismatch: {emitted}")
            if emitted.stat().st_size != item["bytes"]:
                raise ValueError(f"emitted variant size mismatch: {emitted}")

        manifest_path = output_directory / "manifest.json"
        atomic_json(manifest_path, first.manifest)
        summary = dict(first.manifest["summary"])
        contract_checks = {
            **first.manifest["contract_checks"],
            "deterministic_rebuild": deterministic_rebuild,
            "hidden_label_mutation_stable": labeled_audit[
                "hidden_label_mutation_stable"
            ],
            "all_labeled_task_round_trips": labeled_audit[
                "all_labeled_task_round_trips"
            ],
            "all_labeled_prediction_round_trips": labeled_audit[
                "all_labeled_prediction_round_trips"
            ],
            "scorer_label_sensitivity_passed": labeled_audit[
                "scorer_label_sensitivity_passed"
            ],
            "selection_or_assignment_modified_by_auditor": labeled_audit[
                "selection_or_assignment_modified_by_auditor"
            ],
        }
        if not all(
            value is True
            for key, value in contract_checks.items()
            if key != "selector_opened_labeled_files"
            and key != "selection_or_assignment_modified_by_auditor"
        ):
            raise ValueError("one or more fixed64 contract checks failed")
        if contract_checks["selector_opened_labeled_files"] != 0:
            raise ValueError("selector opened a labeled file")
        if contract_checks["selection_or_assignment_modified_by_auditor"] is not False:
            raise ValueError("labeled auditor modified selection or assignment")
        record.update(
            {
                "status": "passed",
                "manifest": {
                    "path": "manifest.json",
                    "sha256": sha256_file(manifest_path),
                    "fixed64_manifest_sha256": first.manifest[
                        "fixed64_manifest_sha256"
                    ],
                    **first.manifest["digests"],
                },
                "summary": summary,
                "contract_checks": contract_checks,
                "labeled_reference_audit": labeled_audit,
                "validation": {"error_count": 0, "errors": []},
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "claim_boundary": (
                    "This freezes 64 globally unique public base tasks, 1,135 "
                    "logical IsoARC variants, and their input-visible selection and "
                    "round-trip contracts. It is not a solver run, an accuracy claim, "
                    "a private benchmark, or evidence that a method obeys the runtime "
                    "label firewall."
                ),
                "limitations": first.manifest["limitations"],
            }
        )
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        usage = monitor.stop()
        record["ended_at_utc"] = usage.ended_at_utc
        record["resources"] = usage.to_dict()
        atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
