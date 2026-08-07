"""Helpers for the trusted challenge-runtime core audit.

This module deliberately implements lifecycle evidence for trusted repository
code only.  It is not a security sandbox and it does not provide process-tree
resource accounting.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from .validation import Grid, load_task, validate_task


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def mutate_hidden_test_labels(
    task: Mapping[str, Any], *, offset: int = 1
) -> dict[str, Any]:
    """Return a valid task with every hidden test color shifted modulo ten."""

    if type(offset) is not int or offset % 10 == 0:
        raise ValueError("label-mutation offset must be a nonzero integer modulo 10")
    mutated = copy.deepcopy(dict(task))
    for pair in mutated.get("test", []):
        if "output" not in pair:
            raise ValueError("hidden-label mutation requires labeled test pairs")
        pair["output"] = [
            [(cell + offset) % 10 for cell in row] for row in pair["output"]
        ]
    return validate_task(mutated)


def sentinel_predictions(
    labeled_task_paths: Sequence[Path], *, top_k: int = 2
) -> dict[str, list[dict[str, Grid]]]:
    """Build a scoring-only oracle sentinel; never expose it to inference."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    result: dict[str, list[dict[str, Grid]]] = {}
    for path in sorted(labeled_task_paths):
        if path.stem in result:
            raise ValueError(f"duplicate task ID: {path.stem}")
        task = load_task(path)
        result[path.stem] = [
            {
                f"attempt_{attempt}": copy.deepcopy(pair["output"])
                for attempt in range(1, top_k + 1)
            }
            for pair in task["test"]
        ]
    return result


def tree_inventory(directory: Path) -> list[dict[str, object]]:
    """Hash every regular file in a tree using paths relative to that tree."""

    if not directory.is_dir():
        raise ValueError(f"tree does not exist: {directory}")
    records: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return records


def tree_sha256(directory: Path) -> str:
    return canonical_sha256(tree_inventory(directory))


@dataclass(frozen=True)
class ProcessEvent:
    name: str
    kind: str
    command: list[str]
    cwd: str
    pid: int
    process_group_id: int
    started_at_utc: str
    ended_at_utc: str
    wall_time_seconds: float
    timeout_seconds: float
    timed_out: bool
    return_code: int | None
    status: str
    stdout_path: str
    stderr_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_logged_process(
    command: Sequence[str],
    *,
    name: str,
    kind: str,
    cwd: Path,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    environment: Mapping[str, str],
    display_root: Path | None = None,
) -> ProcessEvent:
    """Run trusted code in a new process group and persist terminal streams.

    Streams go directly to files so an escaped descendant cannot keep a parent
    pipe open.  A process that deliberately creates a new session can still
    escape the process-group signal; callers must not use this for untrusted
    generated code.
    """

    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain nonempty strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if kind not in {"inference", "scoring"}:
        raise ValueError("kind must be inference or scoring")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open(
        "wb"
    ) as stderr_handle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        process_group_id = os.getpgid(process.pid)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            return_code = None
        stdout_handle.flush()
        os.fsync(stdout_handle.fileno())
        stderr_handle.flush()
        os.fsync(stderr_handle.fileno())
    wall = round(time.perf_counter() - started, 6)
    status = "timeout" if timed_out else ("passed" if return_code == 0 else "failed")

    def display(path: Path) -> str:
        if display_root is None:
            return path.as_posix()
        return path.relative_to(display_root).as_posix()

    return ProcessEvent(
        name=name,
        kind=kind,
        command=list(command),
        cwd=display(cwd),
        pid=process.pid,
        process_group_id=process_group_id,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        wall_time_seconds=wall,
        timeout_seconds=timeout_seconds,
        timed_out=timed_out,
        return_code=return_code,
        status=status,
        stdout_path=display(stdout_path),
        stderr_path=display(stderr_path),
    )
