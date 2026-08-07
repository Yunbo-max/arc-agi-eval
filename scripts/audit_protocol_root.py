#!/usr/bin/env python3
"""Validate and persist the single machine-readable protocol readiness root."""

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

from arc_agi_eval.protocol import build_protocol_manifest, sha256_file
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
        "--config", type=Path, default=ROOT / "configs" / "protocol_v1_draft.json"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "reports" / "e0-protocol" / "20260806-protocol-v1-draft-root",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor(include_nvidia=False).start()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-protocol",
        "run_id": output_directory.name,
        "runner": "scripts.audit_protocol_root",
        "status": "failed",
        "scope": "protocol-v1-draft-readiness-root",
        "started_at_utc": utc_now(),
    }
    try:
        manifest = build_protocol_manifest(ROOT, args.config)
        manifest_path = output_directory / "manifest.json"
        atomic_json(manifest_path, manifest)
        record.update(
            {
                "status": "passed",
                "protocol_status": manifest["protocol_status"],
                "manifest": {
                    "path": "manifest.json",
                    "sha256": sha256_file(manifest_path),
                    "protocol_root_sha256": manifest["protocol_root_sha256"],
                    "config_sha256": manifest["config"]["sha256"],
                },
                "readiness": manifest["readiness"],
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "claim_boundary": (
                    "A passed audit means the draft is internally consistent and "
                    "all declared evidence assertions hold. It is frozen only when "
                    "protocol_status=frozen and readiness.freeze_ready=true."
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
