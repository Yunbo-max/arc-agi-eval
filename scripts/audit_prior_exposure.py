#!/usr/bin/env python3
"""Persist the workspace-bounded pre-protocol exposure disclosure inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.prior_exposure import build_prior_exposure_manifest
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "prior_exposure.json"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "e0-prior-exposure"
        / "20260806-workspace-disclosure-draft",
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor(include_nvidia=False).start()
    started_at = utc_now()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "e0-prior-exposure",
        "run_id": output_directory.name,
        "runner": "scripts.audit_prior_exposure",
        "status": "failed",
        "scope": "workspace-bounded-prior-exposure-disclosure-draft",
        "started_at_utc": started_at,
        "protocol_status": "draft-not-frozen",
    }
    try:
        manifest = build_prior_exposure_manifest(
            ROOT, args.config, inventory_cutoff_utc=started_at
        )
        manifest_path = output_directory / "manifest.json"
        atomic_json(manifest_path, manifest)
        record.update(
            {
                "status": "passed",
                "manifest": {
                    "path": "manifest.json",
                    "sha256": sha256_file(manifest_path),
                    "inventory_sha256": manifest["inventory_sha256"],
                    "config_sha256": manifest["config"]["sha256"],
                },
                "summary": manifest["summary"],
                "limitations": manifest["limitations"],
                "claim_boundary": (
                    "Complete for in-scope parseable reports/**/run.json and "
                    "results/* present at the pre-attestation cutoff. Declared "
                    "derived control-plane attestations are excluded to avoid a "
                    "circular hash; this is not proof about unrecorded or external "
                    "human/account activity."
                ),
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
