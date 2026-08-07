#!/usr/bin/env python3
"""Persist ARC-VSA's dependency and label-firewall blocker audit."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import tokenize


ROOT = Path(__file__).resolve().parents[1]
LOCKED_REVISION = "c031a9c6b4885ab03b28fbfdcd97b6b3693df564"


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


def hash_tree(source: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
        total += len(payload)
    return digest.hexdigest(), count, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "external" / "ARC-VSA-2025"
    )
    parser.add_argument(
        "--preparation-status",
        type=Path,
        default=Path("/usr/paper-assets/arc/status/arc-vsa-2025.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "arc-vsa-2025"
        / "20260806-dependency-label-gate-audit",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "arc-vsa-2025",
        "run_id": output_directory.name,
        "runner": "scripts.probe_arc_vsa",
        "status": "failed",
        "scope": "source-dependency-and-label-gate-audit-only",
        "started_at_utc": utc_now(),
        "solver_executed": False,
        "arc_data_loaded": False,
        "labels_loaded": False,
        "network_used": False,
        "gpu_used": False,
    }
    try:
        status = json.loads(args.preparation_status.read_text(encoding="utf-8"))
        observed_revision = status["source"]["revision"]
        if observed_revision != LOCKED_REVISION:
            raise RuntimeError(
                f"prepared revision {observed_revision} != lock {LOCKED_REVISION}"
            )
        if Path(status["source"]["path"]).resolve() != source:
            raise RuntimeError("preparation status source path does not match probe source")

        python_paths = sorted(source.rglob("*.py"))
        syntax_failures: list[dict[str, str]] = []
        for path in python_paths:
            try:
                with tokenize.open(path) as handle:
                    ast.parse(handle.read(), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as error:
                syntax_failures.append(
                    {
                        "path": path.relative_to(source).as_posix(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if syntax_failures:
            raise RuntimeError(f"source syntax failures: {syntax_failures}")

        requirements_path = source / "requirements.txt"
        requirements = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        vsa_path = source / "src" / "vsa.py"
        vsa_text = vsa_path.read_text(encoding="utf-8")
        solver_path = source / "src" / "solver.py"
        solver_lines = solver_path.read_text(encoding="utf-8").splitlines()
        label_lines = [
            index
            for index, line in enumerate(solver_lines, start=1)
            if 'pair["output"]' in line and 'task["test"]' in line
        ]
        imports_sspspace = any(
            isinstance(node, ast.Import)
            and any(alias.name == "sspspace" for alias in node.names)
            for node in ast.walk(ast.parse(vsa_text, filename=str(vsa_path)))
        )
        requirement_names = {
            requirement.split("==", 1)[0].split(">=", 1)[0].strip().lower()
            for requirement in requirements
        }
        sspspace_spec_found = importlib.util.find_spec("sspspace") is not None
        if not imports_sspspace:
            raise RuntimeError("expected upstream sspspace import was not found")
        if "sspspace" in requirement_names:
            raise RuntimeError("sspspace unexpectedly appeared in locked requirements")
        if not label_lines:
            raise RuntimeError("expected test-output constructor dependency was not found")

        tree_sha256, file_count, source_bytes = hash_tree(source)
        status_sha256 = hashlib.sha256(args.preparation_status.read_bytes()).hexdigest()
        record.update(
            {
                "status": "passed",
                "source": {
                    "path": str(source),
                    "repository": "https://github.com/ijoffe/ARC-VSA-2025",
                    "locked_revision": LOCKED_REVISION,
                    "prepared_revision": observed_revision,
                    "preparation_status": str(args.preparation_status.resolve()),
                    "preparation_status_sha256": status_sha256,
                    "tree_sha256": tree_sha256,
                    "file_count": file_count,
                    "bytes": source_bytes,
                    "python_file_count": len(python_paths),
                    "root_license_file": "LICENSE",
                    "syntax_failures": syntax_failures,
                },
                "dependency_gate": {
                    "status": "blocked",
                    "imports_sspspace": imports_sspspace,
                    "sspspace_listed_in_requirements": False,
                    "sspspace_importable_in_probe_environment": sspspace_spec_found,
                    "requirements": requirements,
                    "reason": "Locked source imports sspspace, but the project does not identify or declare that dependency.",
                },
                "label_firewall_gate": {
                    "status": "blocked",
                    "source_file": "src/solver.py",
                    "test_output_dependency_lines": label_lines,
                    "reason": "ARCSolver.__init__ constructs test output grids in the solver process; the upstream entry path is not challenge-only.",
                },
                "solver_gate_passed": False,
                "fairness": {
                    "score_eligible_for_fair_main_board": False,
                    "reason": (
                        "No solver prediction was produced, and the locked upstream "
                        "path fails both dependency and test-label firewall gates."
                    ),
                },
                "limitations": [
                    "This audit parsed and inspected source without importing or executing the solver.",
                    "The missing dependency's intended implementation and compatibility cannot be inferred safely.",
                    "Removing the label dependency would be a new adaptation and would require a separate no-label smoke.",
                ],
            }
        )
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
            "ru_maxrss_before": rss_start,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "KiB on Linux",
            "memory_scope": "current process peak; children excluded",
        }
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "solver_gate_passed": record.get("solver_gate_passed"),
                "run_json": str(output_directory / "run.json"),
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
