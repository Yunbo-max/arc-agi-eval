#!/usr/bin/env python3
"""Persist per-file hashes and schema validation for vendored ARC benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.benchmark_manifest import build_benchmark_manifest, sha256_file
from arc_agi_eval.resources import ResourceMonitor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


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
        "--config", type=Path, default=ROOT / "configs" / "benchmark_sources.json"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-benchmark-data"
        / "20260806-public-snapshot-integrity-v1",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor(include_nvidia=False).start()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-benchmark-data",
        "run_id": output_directory.name,
        "runner": "scripts.audit_benchmark_datasets",
        "status": "failed",
        "scope": "public-benchmark-file-integrity-and-schema-audit",
        "started_at_utc": utc_now(),
        "solver_executed": False,
        "public_test_outputs_parsed": True,
    }
    try:
        manifest = build_benchmark_manifest(ROOT, args.config)
        manifest_path = output_directory / "manifest.json"
        atomic_json(manifest_path, manifest)
        record.update(
            {
                "status": "passed",
                "manifest": {
                    "path": "manifest.json",
                    "sha256": sha256_file(manifest_path),
                    "payload_sha256": manifest["manifest_payload_sha256"],
                    "config_sha256": manifest["config"]["sha256"],
                },
                "summary": manifest["summary"],
                "benchmark_snapshot_digests": {
                    benchmark_id: declaration["snapshot_inventory_sha256"]
                    for benchmark_id, declaration in manifest["benchmarks"].items()
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "claim_boundary": (
                    "All 1,920 declared public task files were parsed with test outputs, "
                    "schema-validated, and hashed. This is data integrity evidence, not "
                    "a solver benchmark, private-label claim, or upstream Git-object proof."
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
