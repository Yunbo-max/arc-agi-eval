from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ProcessResult:
    status: str
    return_code: int | None
    timed_out: bool
    timeout_seconds: float
    wall_time_seconds: float
    stdout: str
    stderr: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_process(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Run a command in its own process group and enforce a wall timeout.

    This provides lifecycle isolation only. It does not restrict filesystem or
    network access and must not be described as a security sandbox.
    """
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain nonempty strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    working_directory = Path(cwd)
    if not working_directory.is_dir():
        raise ValueError(f"working directory does not exist: {working_directory}")

    started = time.perf_counter()
    process = subprocess.Popen(
        list(command),
        cwd=working_directory,
        env=dict(environment) if environment is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    wall = round(time.perf_counter() - started, 6)
    if timed_out:
        status = "timeout"
    elif process.returncode == 0:
        status = "passed"
    else:
        status = "failed"
    return ProcessResult(
        status=status,
        return_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        wall_time_seconds=wall,
        stdout=stdout,
        stderr=stderr,
    )
