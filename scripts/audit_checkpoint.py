#!/usr/bin/env python3
"""Hash every file in a locally prepared immutable checkpoint snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-total-bytes", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    output_directory = args.output_directory.resolve()
    if snapshot.name != args.expected_revision:
        parser.error(
            f"snapshot revision mismatch: expected {args.expected_revision}, found {snapshot.name}"
        )
    if not snapshot.is_dir():
        parser.error(f"missing snapshot: {snapshot}")
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    started_at = utc_now()
    wall_start = time.perf_counter()
    paths = sorted(path for path in snapshot.rglob("*") if path.is_file())
    files = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    total_bytes = sum(int(item["size_bytes"]) for item in files)
    passed = (
        total_bytes == args.expected_total_bytes
        and any(item["path"] == "config.json" for item in files)
        and any(item["path"] == "model.safetensors" for item in files)
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": args.method_id,
        "run_id": output_directory.name,
        "runner": "scripts.audit_checkpoint",
        "status": "passed" if passed else "failed",
        "scope": "checkpoint-download-integrity-only",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "snapshot": {
            "path": str(snapshot),
            "revision": args.expected_revision,
            "file_count": len(files),
            "expected_total_bytes": args.expected_total_bytes,
            "observed_total_bytes": total_bytes,
            "files": files,
        },
        "resources": {"wall_time_seconds": time.perf_counter() - wall_start},
        "claim_boundary": (
            "The downloaded files and revision were verified; no model class was "
            "instantiated and no forward or solver inference was performed"
        ),
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
