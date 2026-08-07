#!/usr/bin/env python3
"""Build a deterministic draft development split from ARC training tasks only."""

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

from arc_agi_eval.development_split import (
    DEFAULT_MAX_SEARCH_STATES,
    DEFAULT_PUBLIC_SEED,
    SOURCE_NAMES,
    build_development_manifest,
    canonical_json_bytes,
    load_overlap_id_ledger,
    validate_development_manifest,
)
from arc_agi_eval.resources import ResourceMonitor


EXPECTED_OVERLAP_AUDIT_SHA256 = (
    "ce5920b1bb43546f0decdd29db2540b2a1537a9149d905b12d0b40fdeea31159"
)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
            )
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc-agi-1-training",
        type=Path,
        default=ROOT / "third_party" / "arc-agi-1" / "data" / "training",
    )
    parser.add_argument(
        "--arc-agi-2-training",
        type=Path,
        default=ROOT / "third_party" / "arc-agi-2" / "data" / "training",
    )
    parser.add_argument(
        "--arc1-overlap-audit",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "e0-overlap"
            / "20260806-arc1-eval-vs-arc2-train-retry1"
            / "run.json"
        ),
    )
    parser.add_argument(
        "--max-search-states",
        type=int,
        default=DEFAULT_MAX_SEARCH_STATES,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "e0-development-split"
            / "20260806-training-only-deterministic-split"
        ),
    )
    args = parser.parse_args()

    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor(include_nvidia=False).start()
    overlap_ledger = load_overlap_id_ledger(
        args.arc1_overlap_audit,
        expected_sha256=EXPECTED_OVERLAP_AUDIT_SHA256,
        report_reference=(
            "reports/e0-overlap/"
            "20260806-arc1-eval-vs-arc2-train-retry1/run.json"
        ),
    )
    manifest = build_development_manifest(
        {
            SOURCE_NAMES[0]: args.arc_agi_1_training,
            SOURCE_NAMES[1]: args.arc_agi_2_training,
        },
        expected_counts={SOURCE_NAMES[0]: 400, SOURCE_NAMES[1]: 1000},
        public_seed=DEFAULT_PUBLIC_SEED,
        max_search_states=args.max_search_states,
        arc1_overlap_ledger=overlap_ledger,
    )
    validate_development_manifest(manifest)
    monitor.sample()
    resources = monitor.stop().to_dict()

    manifest_path = output_directory / "manifest.json"
    atomic_json(manifest_path, manifest)
    manifest_file_sha256 = sha256_file(manifest_path)
    summary = manifest["summary"]
    inconclusive_count = int(summary["inconclusive_pair_count"])
    run: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-development-split",
        "run_id": output_directory.name,
        "runner": "scripts.build_development_split",
        "status": "passed",
        "scope": "training-only-draft-development-cluster-manifest",
        "started_at_utc": resources["started_at_utc"],
        "ended_at_utc": resources["ended_at_utc"],
        "protocol_status": manifest["protocol_status"],
        "manifest": {
            "path": manifest_path.name,
            "file_sha256": manifest_file_sha256,
            "canonical_payload_sha256": manifest["digests"][
                "audit_payload_sha256"
            ],
        },
        "sources": {
            SOURCE_NAMES[0]: {
                **manifest["sources"][SOURCE_NAMES[0]],
                "resolved_path": str(args.arc_agi_1_training.resolve()),
            },
            SOURCE_NAMES[1]: {
                **manifest["sources"][SOURCE_NAMES[1]],
                "resolved_path": str(args.arc_agi_2_training.resolve()),
            },
        },
        "allocation": manifest["allocation"],
        "summary": summary,
        "cluster_digest": manifest["digests"]["cluster_membership_sha256"],
        "assignment_digest": manifest["digests"]["assignment_sha256"],
        "splits": {
            name: {
                "cluster_count": split["cluster_count"],
                "deduplicated_task_count": split["deduplicated_task_count"],
                "source_record_count": split["source_record_count"],
                "source_record_counts": split["source_record_counts"],
                "representative_source_counts": split[
                    "representative_source_counts"
                ],
            }
            for name, split in manifest["splits"].items()
        },
        "matching_completeness": (
            "all invariant-candidate pairs decided"
            if inconclusive_count == 0
            else "conservative verified matches only; capped pairs left separate"
        ),
        "contamination": {
            "status": manifest["contamination"]["status"],
            "assignment_influence": manifest["contamination"][
                "assignment_influence"
            ],
            "reference_audit": manifest["contamination"].get(
                "reference_audit"
            ),
            "overlap_task_id_count": manifest["contamination"].get(
                "overlap_task_id_count"
            ),
            "flagged_cluster_count": manifest["contamination"].get(
                "flagged_cluster_count"
            ),
            "by_split": manifest["contamination"].get("by_split"),
            "arc1_public_evaluation_claim": manifest["contamination"].get(
                "arc1_public_evaluation_claim"
            ),
            "arc1_clean_view": manifest["contamination"].get(
                "arc1_clean_view"
            ),
        },
        "resources": resources,
        "environment": {"python": platform.python_version()},
        "execution_policy": {
            "evaluation_tasks_read": 0,
            "evaluation_labels_read": 0,
            "solution_files_read": 0,
            "prior_overlap_audit_id_ledgers_read": 1,
            "overlap_ledger_used_for_assignment": False,
            "network_used": False,
            "gpu_requested": False,
            "model_loaded": False,
        },
        "implementation": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(
                ROOT / "arc_agi_eval" / "development_split.py"
            ),
            "canonical_empty_object_sha256": hashlib.sha256(
                canonical_json_bytes({})
            ).hexdigest(),
        },
        "limitations": manifest["limitations"],
        "claim_boundary": (
            "This is a deterministic draft split of complete ARC-AGI-1 and "
            "ARC-AGI-2 training tasks only. It is not protocol v1, a protocol "
            "freeze, an ARC-AGI-1-clean development view, a benchmark run, or "
            "a score. The prior overlap audit contributes IDs for annotation "
            "only and no evaluation task file is opened."
        ),
    }
    atomic_json(output_directory / "run.json", run)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
