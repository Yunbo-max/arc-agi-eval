#!/usr/bin/env python3
"""Materialize immutable label-free ARC-AGI-1/2 public challenge trees."""

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

from arc_agi_eval.firewall import challenge_only, generate_challenge_tree
from arc_agi_eval.resources import ResourceMonitor
from arc_agi_eval.validation import load_task


SOURCES = {
    "arc_agi_1": ROOT / "third_party" / "arc-agi-1" / "data" / "evaluation",
    "arc_agi_2": ROOT / "third_party" / "arc-agi-2" / "data" / "evaluation",
}
SOURCE_AUDIT_MANIFEST = (
    ROOT
    / "reports"
    / "e0-benchmark-data"
    / "20260806-public-snapshot-integrity-v1"
    / "manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
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
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-challenge-data"
        / "20260806-locked-public-challenge-trees-draft",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    if not SOURCE_AUDIT_MANIFEST.is_file():
        parser.error(f"source audit manifest is missing: {SOURCE_AUDIT_MANIFEST}")

    monitor = ResourceMonitor(include_nvidia=False).start()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-challenge-data",
        "run_id": output_directory.name,
        "runner": "scripts.materialize_locked_public_challenges",
        "status": "failed",
        "scope": "draft-locked-public-label-free-challenge-trees",
        "protocol_status": "draft-not-frozen",
        "started_at_utc": utc_now(),
        "public_labeled_sources_read": True,
        "solver_executed": False,
        "network_used": False,
        "gpu_requested": False,
    }
    try:
        source_audit = json.loads(SOURCE_AUDIT_MANIFEST.read_text(encoding="utf-8"))
        views: dict[str, object] = {}
        total_tasks = 0
        total_test_inputs = 0
        for benchmark_id, source in SOURCES.items():
            destination = output_directory / "data" / benchmark_id / "evaluation"
            generated = generate_challenge_tree(
                source,
                destination,
                source_id=f"{benchmark_id}:evaluation",
            )
            file_records: list[dict[str, object]] = []
            test_input_count = 0
            expected_source_split = source_audit["benchmarks"][benchmark_id]["splits"][
                "evaluation"
            ]
            expected_sources = {
                item["task_id"]: item for item in expected_source_split["tasks"]
            }
            generated_ids = {Path(item["path"]).stem for item in generated["files"]}
            if generated_ids != set(expected_sources):
                raise ValueError(f"challenge/source task-ID set mismatch: {benchmark_id}")
            for generated_record in generated["files"]:
                path = destination / generated_record["path"]
                task = load_task(path, require_test_outputs=False)
                if any("output" in pair for pair in task["test"]):
                    raise ValueError(f"test output leaked into challenge view: {path}")
                test_input_count += len(task["test"])
                observed_sha256 = sha256_file(path)
                if observed_sha256 != generated_record["sha256"]:
                    raise ValueError(f"challenge hash mismatch: {path}")
                task_id = path.stem
                source_record = expected_sources[task_id]
                source_path = (ROOT / source_record["path"]).resolve()
                try:
                    source_path.relative_to(ROOT)
                except ValueError as error:
                    raise ValueError(f"source audit path escapes repository: {source_path}") from error
                if sha256_file(source_path) != source_record["sha256"]:
                    raise ValueError(f"source audit hash mismatch: {source_path}")
                expected_challenge = challenge_only(load_task(source_path))
                if task != expected_challenge:
                    raise ValueError(f"source-to-challenge payload mismatch: {task_id}")
                file_records.append(
                    {
                        "task_id": task_id,
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": observed_sha256,
                        "bytes": path.stat().st_size,
                        "source_path": source_path.relative_to(ROOT).as_posix(),
                        "source_sha256": source_record["sha256"],
                        "train_example_count": len(task["train"]),
                        "test_input_count": len(task["test"]),
                        "test_output_count": 0,
                    }
                )
            if len(file_records) != expected_source_split["task_count"]:
                raise ValueError(f"challenge task count mismatch: {benchmark_id}")
            if test_input_count != expected_source_split["test_output_count"]:
                raise ValueError(f"challenge test-input denominator mismatch: {benchmark_id}")
            visible_manifest_path = destination / "MANIFEST"
            visible_manifest = json.loads(visible_manifest_path.read_text(encoding="utf-8"))
            visible_serialized = json.dumps(visible_manifest, sort_keys=True)
            forbidden_locators = [str(source.resolve()), str(ROOT.resolve()), "third_party/"]
            if any(locator in visible_serialized for locator in forbidden_locators):
                raise ValueError(f"labeled source locator leaked into visible manifest: {benchmark_id}")
            challenge_file_records = [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"source_path", "source_sha256"}
                }
                for record in file_records
            ]
            visible_inventory = [
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in file_records
            ]
            visible_inventory.append(
                {
                    "path": visible_manifest_path.relative_to(output_directory).as_posix(),
                    "sha256": sha256_file(visible_manifest_path),
                    "bytes": visible_manifest_path.stat().st_size,
                }
            )
            views[benchmark_id] = {
                "split": "evaluation",
                "task_count": len(file_records),
                "test_input_count": test_input_count,
                "test_output_fields_present": 0,
                # Keep the original digest domain restricted to emitted task
                # records; provenance fields live outside the visible tree.
                "challenge_tree_sha256": canonical_sha256(challenge_file_records),
                "visible_tree_sha256": canonical_sha256(visible_inventory),
                "visible_manifest": {
                    "path": visible_manifest_path.relative_to(output_directory).as_posix(),
                    "sha256": sha256_file(visible_manifest_path),
                    "source_id": visible_manifest["source_id"],
                    "contains_labeled_source_locator": False,
                },
                "source_task_inventory_sha256": expected_source_split[
                    "task_inventory_sha256"
                ],
                "source_hashes_verified": True,
                "files": file_records,
            }
            total_tasks += len(file_records)
            total_test_inputs += test_input_count

        manifest: dict[str, object] = {
            "schema_version": 1,
            "manifest_id": "arc-public-evaluation-challenge-only-draft-20260806",
            "protocol_status": "draft-not-frozen",
            "format": "arc-agi-challenge-only-v1",
            "source_audit_manifest": {
                "path": SOURCE_AUDIT_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(SOURCE_AUDIT_MANIFEST),
                "payload_sha256": source_audit["manifest_payload_sha256"],
            },
            "views": views,
            "summary": {
                "benchmark_count": len(views),
                "task_count": total_tasks,
                "test_input_count": total_test_inputs,
                "test_output_fields_present": 0,
                "source_hash_mismatch_count": 0,
                "source_to_challenge_mismatch_count": 0,
                "visible_label_locator_count": 0,
            },
            "limitations": [
                "The sources and labels are public and were already locally accessible; this view is an execution-interface control, not a private or unseen benchmark.",
                "The host has no strict filesystem namespace, so generated or untrusted code must not run merely because this directory is label-free.",
                "Per-method label-mutation checks and independent post-inference scoring remain required."
            ],
        }
        manifest["challenge_manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = output_directory / "manifest.json"
        atomic_json(manifest_path, manifest)
        record.update(
            {
                "status": "passed",
                "manifest": {
                    "path": "manifest.json",
                    "sha256": sha256_file(manifest_path),
                    "challenge_manifest_sha256": manifest[
                        "challenge_manifest_sha256"
                    ],
                    "source_audit_manifest_sha256": manifest[
                        "source_audit_manifest"
                    ]["sha256"],
                },
                "summary": manifest["summary"],
                "views": {
                    benchmark_id: {
                        key: value
                        for key, value in view.items()
                        if key != "files"
                    }
                    for benchmark_id, view in views.items()
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "claim_boundary": (
                    "Every public evaluation test output field was removed from "
                    "the 520-task execution view; every labeled source hash, stripped "
                    "payload, emitted file hash, and inference-visible manifest was "
                    "verified without exposing a label locator. This does not establish "
                    "strict process isolation or protocol freeze."
                ),
                "limitations": manifest["limitations"],
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
