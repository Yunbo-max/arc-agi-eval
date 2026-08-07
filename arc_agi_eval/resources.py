"""Dependency-free, process-local resource measurements.

The standard library does not expose identical memory counters on every
platform, so this module reports both values and their sources:

* ``current_rss_bytes`` is the current process resident set when the host has a
  supported interface (Linux procfs or the Windows process API).  It is
  ``None`` rather than an invented estimate elsewhere.
* ``process_peak_rss_bytes`` is the operating system's process-lifetime high
  water mark.  It can include allocations made before a monitor was started.
* ``sampled_peak_current_rss_bytes`` is the largest current-RSS observation
  made by one monitor.  It is an interval-local lower bound, not a claim that
  every transient peak was observed.

CPU time is for the current process only and excludes child processes.  The
optional NVIDIA query likewise selects compute contexts belonging to one PID;
device-global memory or utilization is deliberately not attributed to that
process.  NVIDIA sampling invokes ``nvidia-smi`` and is disabled by default.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any

try:  # ``resource`` is unavailable on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on Windows
    _resource = None


MIB = 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class NvidiaProcessMemory:
    """Memory attributed by NVIDIA to this PID on one GPU context."""

    gpu_uuid: str
    used_memory_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_uuid": self.gpu_uuid,
            "used_memory_bytes": self.used_memory_bytes,
        }


@dataclass(frozen=True)
class NvidiaSnapshot:
    """Result of one optional ``nvidia-smi`` process-memory query.

    ``status`` is ``ok``, ``unavailable``, ``timeout``, or ``error``.  An
    ``ok`` snapshot with no contexts means that the PID had no visible NVIDIA
    compute allocation at the instant of the query.
    """

    status: str
    contexts: tuple[NvidiaProcessMemory, ...] = ()
    message: str | None = None
    ignored_rows: int = 0

    @property
    def total_used_memory_bytes(self) -> int | None:
        if self.status != "ok":
            return None
        return sum(context.used_memory_bytes for context in self.contexts)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "contexts": [context.to_dict() for context in self.contexts],
            "total_used_memory_bytes": self.total_used_memory_bytes,
            "message": self.message,
            "ignored_rows": self.ignored_rows,
        }


def _short_message(value: str, limit: int = 500) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:limit]


def query_nvidia_process_memory(
    *, pid: int | None = None, timeout_seconds: float = 2.0
) -> NvidiaSnapshot:
    """Query NVIDIA memory assigned to ``pid`` without importing GPU packages.

    Failure to have NVIDIA hardware, a driver, or ``nvidia-smi`` is represented
    in the returned status and is not an exception.  ``timeout_seconds`` must
    be positive.  The command reports MiB, which is converted to binary bytes.
    """

    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite positive number")
    selected_pid = os.getpid() if pid is None else pid
    if selected_pid <= 0:
        raise ValueError("pid must be positive")

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return NvidiaSnapshot(status="unavailable", message="nvidia-smi not found")

    command = [
        executable,
        "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return NvidiaSnapshot(
            status="timeout", message=f"nvidia-smi exceeded {timeout_seconds:g}s"
        )
    except OSError as error:
        return NvidiaSnapshot(status="unavailable", message=str(error))

    if completed.returncode != 0:
        return NvidiaSnapshot(
            status="error",
            message=_short_message(completed.stderr)
            or f"nvidia-smi exited with status {completed.returncode}",
        )

    contexts: list[NvidiaProcessMemory] = []
    ignored_rows = 0
    for row in csv.reader(io.StringIO(completed.stdout)):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 3:
            ignored_rows += 1
            continue
        try:
            row_pid = int(row[0].strip())
        except ValueError:
            ignored_rows += 1
            continue
        if row_pid != selected_pid:
            continue
        gpu_uuid = row[1].strip()
        try:
            used_mib = float(row[2].strip())
        except ValueError:
            ignored_rows += 1
            continue
        if not gpu_uuid or not math.isfinite(used_mib) or used_mib < 0:
            ignored_rows += 1
            continue
        contexts.append(
            NvidiaProcessMemory(
                gpu_uuid=gpu_uuid,
                used_memory_bytes=int(round(used_mib * MIB)),
            )
        )

    contexts.sort(key=lambda context: (context.gpu_uuid, context.used_memory_bytes))
    return NvidiaSnapshot(
        status="ok", contexts=tuple(contexts), ignored_rows=ignored_rows
    )


def _windows_memory_bytes() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    try:  # pragma: no cover - exercised on Windows
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not succeeded:
            return None
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError, ValueError):
        return None


def _current_rss_bytes() -> tuple[int | None, str]:
    """Return current RSS and a machine-readable source label."""

    try:
        fields = (os.path.join("/proc", "self", "statm"))
        with open(fields, encoding="ascii") as handle:
            values = handle.read().split()
        resident_pages = int(values[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if resident_pages >= 0 and page_size > 0:
            return resident_pages * page_size, "linux-procfs-statm"
    except (IndexError, OSError, TypeError, ValueError):
        pass

    windows = _windows_memory_bytes()
    if windows is not None:
        return windows[0], "windows-working-set"
    return None, "unavailable"


def _peak_rss_bytes() -> tuple[int | None, str]:
    """Return the process-lifetime RSS high water mark and its source."""

    windows = _windows_memory_bytes()
    if windows is not None:
        return windows[1], "windows-peak-working-set"
    if _resource is None:
        return None, "unavailable"
    try:
        maximum = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None, "unavailable"
    if not math.isfinite(maximum) or maximum < 0:
        return None, "unavailable"
    if sys.platform == "darwin":
        return int(maximum), "getrusage-ru_maxrss-bytes"
    return int(maximum * 1024), "getrusage-ru_maxrss-kib"


@dataclass(frozen=True)
class ResourceSample:
    """One instantaneous process-local resource observation."""

    monotonic_seconds: float
    process_cpu_seconds: float
    current_rss_bytes: int | None
    current_rss_source: str
    process_peak_rss_bytes: int | None
    process_peak_rss_source: str
    nvidia: NvidiaSnapshot | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "monotonic_seconds": self.monotonic_seconds,
            "process_cpu_seconds": self.process_cpu_seconds,
            "current_rss_bytes": self.current_rss_bytes,
            "current_rss_source": self.current_rss_source,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "process_peak_rss_source": self.process_peak_rss_source,
            "nvidia": None if self.nvidia is None else self.nvidia.to_dict(),
        }


def capture_resource_sample(
    *, include_nvidia: bool = False, nvidia_timeout_seconds: float = 2.0
) -> ResourceSample:
    """Capture current-process clocks, RSS counters, and optional GPU memory."""

    monotonic_seconds = time.perf_counter()
    process_cpu_seconds = time.process_time()
    current_rss, current_source = _current_rss_bytes()
    peak_rss, peak_source = _peak_rss_bytes()
    nvidia = (
        query_nvidia_process_memory(timeout_seconds=nvidia_timeout_seconds)
        if include_nvidia
        else None
    )
    return ResourceSample(
        monotonic_seconds=monotonic_seconds,
        process_cpu_seconds=process_cpu_seconds,
        current_rss_bytes=current_rss,
        current_rss_source=current_source,
        process_peak_rss_bytes=peak_rss,
        process_peak_rss_source=peak_source,
        nvidia=nvidia,
    )


@dataclass(frozen=True)
class ResourceUsage:
    """Summary and raw observations for one monitor interval."""

    started_at_utc: str
    ended_at_utc: str
    samples: tuple[ResourceSample, ...]

    @property
    def wall_time_seconds(self) -> float:
        return max(0.0, self.samples[-1].monotonic_seconds - self.samples[0].monotonic_seconds)

    @property
    def process_cpu_seconds(self) -> float:
        return max(0.0, self.samples[-1].process_cpu_seconds - self.samples[0].process_cpu_seconds)

    @property
    def current_rss_bytes(self) -> int | None:
        return self.samples[-1].current_rss_bytes

    @property
    def process_peak_rss_bytes(self) -> int | None:
        return self.samples[-1].process_peak_rss_bytes

    @property
    def sampled_peak_current_rss_bytes(self) -> int | None:
        values = [
            sample.current_rss_bytes
            for sample in self.samples
            if sample.current_rss_bytes is not None
        ]
        return max(values) if values else None

    @property
    def nvidia_current_process_memory_bytes(self) -> int | None:
        snapshot = self.samples[-1].nvidia
        return None if snapshot is None else snapshot.total_used_memory_bytes

    @property
    def nvidia_sampled_peak_process_memory_bytes(self) -> int | None:
        values = [
            sample.nvidia.total_used_memory_bytes
            for sample in self.samples
            if sample.nvidia is not None
            and sample.nvidia.total_used_memory_bytes is not None
        ]
        return max(values) if values else None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable audit record with explicit semantics."""

        return {
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "wall_time_seconds": self.wall_time_seconds,
            "process_cpu_seconds": self.process_cpu_seconds,
            "current_rss_bytes": self.current_rss_bytes,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "sampled_peak_current_rss_bytes": self.sampled_peak_current_rss_bytes,
            "nvidia_current_process_memory_bytes": self.nvidia_current_process_memory_bytes,
            "nvidia_sampled_peak_process_memory_bytes": self.nvidia_sampled_peak_process_memory_bytes,
            "sample_count": len(self.samples),
            "samples": [sample.to_dict() for sample in self.samples],
            "semantics": {
                "scope": "current process only; child processes are excluded",
                "process_cpu_seconds": "process CPU consumed during the monitor interval",
                "current_rss_bytes": "resident set at the final observation, or null if unavailable",
                "process_peak_rss_bytes": (
                    "OS process-lifetime high water mark at the final observation; "
                    "it may predate this interval"
                ),
                "sampled_peak_current_rss_bytes": (
                    "largest interval observation; a lower bound between samples"
                ),
                "nvidia_memory": (
                    "nvidia-smi compute memory attributed to this PID; sampled, not device-global"
                ),
            },
        }


class ResourceMonitor:
    """Explicit-sampling context manager for lightweight resource accounting.

    Start and stop are always sampled.  Call :meth:`sample` at meaningful phase
    boundaries when an interval-local RSS or NVIDIA-memory trace is desired.
    The class intentionally creates no background thread, making both sampling
    points and overhead auditable.
    """

    def __init__(
        self, *, include_nvidia: bool = False, nvidia_timeout_seconds: float = 2.0
    ) -> None:
        if nvidia_timeout_seconds <= 0 or not math.isfinite(nvidia_timeout_seconds):
            raise ValueError("nvidia_timeout_seconds must be a finite positive number")
        self.include_nvidia = include_nvidia
        self.nvidia_timeout_seconds = nvidia_timeout_seconds
        self._samples: list[ResourceSample] = []
        self._started_at_utc: str | None = None
        self._usage: ResourceUsage | None = None

    def _capture(self) -> ResourceSample:
        return capture_resource_sample(
            include_nvidia=self.include_nvidia,
            nvidia_timeout_seconds=self.nvidia_timeout_seconds,
        )

    def start(self) -> ResourceMonitor:
        if self._started_at_utc is not None:
            raise RuntimeError("resource monitor has already been started")
        self._started_at_utc = _utc_now()
        self._samples.append(self._capture())
        return self

    def sample(self) -> ResourceSample:
        if self._started_at_utc is None:
            raise RuntimeError("resource monitor has not been started")
        if self._usage is not None:
            raise RuntimeError("resource monitor has already been stopped")
        observation = self._capture()
        self._samples.append(observation)
        return observation

    def stop(self) -> ResourceUsage:
        if self._started_at_utc is None:
            raise RuntimeError("resource monitor has not been started")
        if self._usage is not None:
            return self._usage
        self._samples.append(self._capture())
        self._usage = ResourceUsage(
            started_at_utc=self._started_at_utc,
            ended_at_utc=_utc_now(),
            samples=tuple(self._samples),
        )
        return self._usage

    @property
    def result(self) -> ResourceUsage:
        if self._usage is None:
            raise RuntimeError("resource monitor has not been stopped")
        return self._usage

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()


@dataclass(frozen=True)
class ResourceCalibration:
    """Median incremental cost of one resource snapshot."""

    iterations: int
    repeats: int
    include_nvidia: bool
    wall_seconds_per_sample: float
    process_cpu_seconds_per_sample: float
    baseline_wall_seconds_per_iteration: float
    baseline_process_cpu_seconds_per_iteration: float
    wall_seconds_per_sample_trials: tuple[float, ...]
    process_cpu_seconds_per_sample_trials: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "repeats": self.repeats,
            "include_nvidia": self.include_nvidia,
            "wall_seconds_per_sample": self.wall_seconds_per_sample,
            "process_cpu_seconds_per_sample": self.process_cpu_seconds_per_sample,
            "baseline_wall_seconds_per_iteration": self.baseline_wall_seconds_per_iteration,
            "baseline_process_cpu_seconds_per_iteration": self.baseline_process_cpu_seconds_per_iteration,
            "wall_seconds_per_sample_trials": list(self.wall_seconds_per_sample_trials),
            "process_cpu_seconds_per_sample_trials": list(
                self.process_cpu_seconds_per_sample_trials
            ),
            "semantics": (
                "median incremental capture cost after subtracting a matched empty loop; "
                "negative clock noise is clamped to zero"
            ),
        }


def calibrate_resource_monitor(
    *,
    iterations: int = 16,
    repeats: int = 3,
    include_nvidia: bool = False,
    nvidia_timeout_seconds: float = 2.0,
) -> ResourceCalibration:
    """Measure the incremental wall/CPU overhead of one resource snapshot.

    The calibration uses matched empty loops and reports the median across
    ``repeats``.  Keep NVIDIA disabled for CPU/RSS calibration; enabling it
    deliberately includes the cost of launching and parsing ``nvidia-smi``.
    """

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if nvidia_timeout_seconds <= 0 or not math.isfinite(nvidia_timeout_seconds):
        raise ValueError("nvidia_timeout_seconds must be a finite positive number")

    # Warm procfs/resource code paths and imports before timed trials.
    capture_resource_sample(
        include_nvidia=include_nvidia,
        nvidia_timeout_seconds=nvidia_timeout_seconds,
    )

    baseline_wall_trials: list[float] = []
    baseline_cpu_trials: list[float] = []
    sample_wall_trials: list[float] = []
    sample_cpu_trials: list[float] = []

    for _ in range(repeats):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        for _ in range(iterations):
            pass
        baseline_cpu = time.process_time() - cpu_start
        baseline_wall = time.perf_counter() - wall_start

        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        for _ in range(iterations):
            capture_resource_sample(
                include_nvidia=include_nvidia,
                nvidia_timeout_seconds=nvidia_timeout_seconds,
            )
        measured_cpu = time.process_time() - cpu_start
        measured_wall = time.perf_counter() - wall_start

        baseline_wall_trials.append(baseline_wall / iterations)
        baseline_cpu_trials.append(baseline_cpu / iterations)
        sample_wall_trials.append(max(0.0, measured_wall - baseline_wall) / iterations)
        sample_cpu_trials.append(max(0.0, measured_cpu - baseline_cpu) / iterations)

    return ResourceCalibration(
        iterations=iterations,
        repeats=repeats,
        include_nvidia=include_nvidia,
        wall_seconds_per_sample=median(sample_wall_trials),
        process_cpu_seconds_per_sample=median(sample_cpu_trials),
        baseline_wall_seconds_per_iteration=median(baseline_wall_trials),
        baseline_process_cpu_seconds_per_iteration=median(baseline_cpu_trials),
        wall_seconds_per_sample_trials=tuple(sample_wall_trials),
        process_cpu_seconds_per_sample_trials=tuple(sample_cpu_trials),
    )


def dumps_resource_record(value: ResourceUsage | ResourceCalibration) -> str:
    """Serialize a resource result deterministically for immutable run records."""

    return json.dumps(value.to_dict(), indent=2, sort_keys=True) + "\n"


__all__ = [
    "NvidiaProcessMemory",
    "NvidiaSnapshot",
    "ResourceCalibration",
    "ResourceMonitor",
    "ResourceSample",
    "ResourceUsage",
    "calibrate_resource_monitor",
    "capture_resource_sample",
    "dumps_resource_record",
    "query_nvidia_process_memory",
]
