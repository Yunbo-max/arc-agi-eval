#!/usr/bin/env python3
"""Validate one new protocol-v1 terminal run without touching legacy evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.run_schema import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    RunSchemaValidationError,
    validate_run_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate a new protocol-v1 terminal run record. "
            "Legacy run.json files are not migrated or grandfathered."
        )
    )
    parser.add_argument("run_json", help="new protocol-v1 run.json to audit")
    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA_PATH),
        help="pinned JSON Schema path",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="repository root used for path containment and file integrity",
    )
    parser.add_argument(
        "--no-verify-files",
        action="store_true",
        help="validate declarations only; do not require referenced files",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = validate_run_file(
            args.run_json,
            schema_path=args.schema,
            repo_root=args.repo_root,
            verify_files=not args.no_verify_files,
        )
    except RunSchemaValidationError as exc:
        payload = {
            "error_count": len(exc.issues),
            "errors": list(exc.issues),
            "status": "failed",
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for issue in exc.issues:
                print(f"error: {issue}", file=sys.stderr)
            print(
                f"protocol-v1 run validation failed: {len(exc.issues)} issue(s)",
                file=sys.stderr,
            )
        return 1
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "error_count": 1,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "status": "error",
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"validated protocol-v1 terminal run: {result.run_id}")
        print(f"record_sha256: {result.record_sha256}")
        print(f"schema_sha256: {result.schema_sha256}")
        print(
            "files: "
            f"{result.verified_file_count}/{result.declared_file_count} verified"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
