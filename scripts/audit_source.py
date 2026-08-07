#!/usr/bin/env python3
"""Create a non-executing, immutable syntax audit for a locked source tree."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tokenize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.resources import ResourceMonitor


IGNORED_PARTS = {".git", ".hg", ".svn", ".tox", ".venv", "node_modules"}
LICENSE_NAMES = {
    "copying",
    "copying.md",
    "copying.txt",
    "license",
    "license.md",
    "license.txt",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def source_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*.py")
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
    )


def parse_python(path: Path) -> str | None:
    try:
        with tokenize.open(path) as handle:
            ast.parse(handle.read(), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def tree_digest(source: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if not (source / ".git").is_dir():
        parser.error(f"not a Git checkout: {source}")
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    monitor = ResourceMonitor()
    monitor.start()
    revision = git(source, "rev-parse", "HEAD")
    dirty_paths = git(source, "status", "--porcelain").splitlines()
    monitor.sample()
    paths = source_files(source)
    failures: list[dict[str, str]] = []
    for path in paths:
        error = parse_python(path)
        if error is not None:
            failures.append(
                {"path": path.relative_to(source).as_posix(), "error": error}
            )
    license_files = sorted(
        path.relative_to(source).as_posix()
        for path in source.iterdir()
        if path.is_file() and path.name.lower() in LICENSE_NAMES
    )
    passed = (
        revision == args.expected_revision and not dirty_paths and not failures
    )
    usage = monitor.stop()
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": args.method_id,
        "run_id": output_directory.name,
        "runner": "scripts.audit_source",
        "status": "passed" if passed else "failed",
        "scope": "source-lock-and-python-syntax-only",
        "started_at_utc": usage.started_at_utc,
        "ended_at_utc": usage.ended_at_utc,
        "source": {
            "path": str(source),
            "expected_revision": args.expected_revision,
            "observed_revision": revision,
            "dirty_paths": dirty_paths,
            "python_file_count": len(paths),
            "python_bytes": sum(path.stat().st_size for path in paths),
            "python_tree_sha256": tree_digest(source, paths),
            "root_license_files": license_files,
        },
        "license_audit": {
            "scope": "repository-root filenames only; license text was not interpreted",
            "root_license_files": license_files,
            "status": "identified" if license_files else "not-identified-at-repository-root",
        },
        "syntax_failures": failures,
        "resources": usage.to_dict(),
        "claim_boundary": (
            "Files were parsed to an AST without importing or executing upstream code; "
            "this is not a dependency, component, solver, or benchmark smoke"
        ),
        "limitations": [
            "Only Python files were parsed; notebooks, compiled code, shell scripts, and other languages were not validated.",
            "Resource CPU and RSS measurements cover this Python process only; git subprocesses are excluded.",
            "A root license filename is evidence of a candidate license file, not a legal interpretation of its terms.",
        ],
    }
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
