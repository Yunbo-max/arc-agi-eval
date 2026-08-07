#!/usr/bin/env python3
"""Materialize the draft ARC-1 known-overlap-excluded development view."""

from __future__ import annotations

import argparse
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

from arc_agi_eval.development_clean_view import (
    build_arc1_clean_development_view,
    load_locked_source_manifest,
    validate_arc1_clean_development_view,
)
from arc_agi_eval.development_split import sha256_file
from arc_agi_eval.resources import ResourceMonitor


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "b21f60c7fc381977b3ac7ba2655270f037c5ddb6172cbdcf155b8e7c62ff7313"
)
SOURCE_MANIFEST_REFERENCE = (
    "reports/e0-development-split/"
    "20260806-training-only-deterministic-split-retry1/manifest.json"
)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / SOURCE_MANIFEST_REFERENCE,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "e0-development-split"
            / "20260806-arc1-clean-overlap-excluded-draft-view"
        ),
    )
    args = parser.parse_args()

    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    source_manifest_path = args.source_manifest.resolve()

    monitor = ResourceMonitor(include_nvidia=False).start()
    source_manifest = load_locked_source_manifest(
        source_manifest_path,
        expected_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    view = build_arc1_clean_development_view(
        source_manifest,
        source_manifest_reference=SOURCE_MANIFEST_REFERENCE,
        source_manifest_file_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    validate_arc1_clean_development_view(
        view,
        source_manifest=source_manifest,
        observed_source_manifest_file_sha256=sha256_file(source_manifest_path),
    )
    monitor.sample()
    resources = monitor.stop().to_dict()

    manifest_path = output_directory / "manifest.json"
    atomic_json(manifest_path, view)
    manifest_file_sha256 = sha256_file(manifest_path)
    run: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-development-split",
        "run_id": output_directory.name,
        "runner": "scripts.build_arc1_clean_development_view",
        "status": "passed",
        "scope": "arc1-known-overlap-excluded-draft-development-view",
        "protocol_status": view["protocol_status"],
        "started_at_utc": resources["started_at_utc"],
        "ended_at_utc": resources["ended_at_utc"],
        "source_manifest": {
            **view["source_manifest"],
            "resolved_path": str(source_manifest_path),
        },
        "manifest": {
            "path": manifest_path.name,
            "file_sha256": manifest_file_sha256,
            "canonical_payload_sha256": view["digests"][
                "audit_payload_sha256"
            ],
        },
        "summary": view["summary"],
        "exclusion": {
            "known_overlap_task_id_count": view["exclusion"][
                "known_overlap_task_id_count"
            ],
            "excluded_cluster_count": view["exclusion"][
                "excluded_cluster_count"
            ],
            "excluded_source_record_count": view["exclusion"][
                "excluded_source_record_count"
            ],
            "excluded_source_record_counts": view["exclusion"][
                "excluded_source_record_counts"
            ],
            "exclusion_sha256": view["digests"]["exclusion_sha256"],
        },
        "splits": {
            name: {
                "original_cluster_count": split["original_cluster_count"],
                "excluded_cluster_count": split["excluded_cluster_count"],
                "cluster_count": split["cluster_count"],
                "original_source_record_count": split[
                    "original_source_record_count"
                ],
                "excluded_source_record_count": split[
                    "excluded_source_record_count"
                ],
                "source_record_count": split["source_record_count"],
                "source_record_counts": split["source_record_counts"],
                "split_payload_sha256": split["digests"][
                    "split_payload_sha256"
                ],
            }
            for name, split in view["splits"].items()
        },
        "digests": view["digests"],
        "resources": resources,
        "environment": {"python": platform.python_version()},
        "execution_policy": {
            "source_manifest_files_read": 1,
            "training_task_files_read": 0,
            "evaluation_task_files_read": 0,
            "evaluation_label_files_read": 0,
            "solution_files_read": 0,
            "network_used": False,
            "gpu_requested": False,
            "cluster_reallocation_performed": False,
        },
        "implementation": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(
                ROOT / "arc_agi_eval" / "development_clean_view.py"
            ),
            "source_builder_module_sha256": sha256_file(
                ROOT / "arc_agi_eval" / "development_split.py"
            ),
            "empty_json_sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "limitations": view["limitations"],
        "claim_boundary": view["claim_boundary"],
    }
    atomic_json(output_directory / "run.json", run)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
