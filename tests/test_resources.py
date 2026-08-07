import json
import subprocess
import unittest
from unittest import mock

from arc_agi_eval import resources
from arc_agi_eval.resources import (
    ResourceMonitor,
    calibrate_resource_monitor,
    capture_resource_sample,
    dumps_resource_record,
    query_nvidia_process_memory,
)


class ResourceSampleTests(unittest.TestCase):
    def test_cpu_rss_snapshot_is_sane_and_json_serializable(self) -> None:
        with mock.patch.object(
            resources, "query_nvidia_process_memory", side_effect=AssertionError
        ):
            sample = capture_resource_sample(include_nvidia=False)

        self.assertGreater(sample.monotonic_seconds, 0)
        self.assertGreaterEqual(sample.process_cpu_seconds, 0)
        self.assertIsNone(sample.nvidia)
        self.assertIn(
            sample.current_rss_source,
            {"linux-procfs-statm", "windows-working-set", "unavailable"},
        )
        if sample.current_rss_bytes is not None:
            self.assertGreater(sample.current_rss_bytes, 0)
        if sample.process_peak_rss_bytes is not None:
            self.assertGreater(sample.process_peak_rss_bytes, 0)
        json.dumps(sample.to_dict(), sort_keys=True)

    def test_monitor_records_durations_samples_and_memory_semantics(self) -> None:
        monitor = ResourceMonitor()
        with monitor:
            sum(index * index for index in range(20_000))
            monitor.sample()

        usage = monitor.result
        self.assertEqual(len(usage.samples), 3)
        self.assertGreaterEqual(usage.wall_time_seconds, 0)
        self.assertGreaterEqual(usage.process_cpu_seconds, 0)
        self.assertIs(monitor.stop(), usage)
        if usage.sampled_peak_current_rss_bytes is not None:
            self.assertGreater(usage.sampled_peak_current_rss_bytes, 0)
        record = json.loads(dumps_resource_record(usage))
        self.assertEqual(record["sample_count"], 3)
        self.assertIn("process-lifetime high water mark", record["semantics"]["process_peak_rss_bytes"])
        self.assertEqual(
            record["semantics"]["scope"],
            "current process only; child processes are excluded",
        )

    def test_monitor_lifecycle_errors_are_explicit(self) -> None:
        monitor = ResourceMonitor()
        with self.assertRaisesRegex(RuntimeError, "not been started"):
            monitor.sample()
        with self.assertRaisesRegex(RuntimeError, "not been stopped"):
            _ = monitor.result
        monitor.start()
        with self.assertRaisesRegex(RuntimeError, "already been started"):
            monitor.start()
        monitor.stop()
        with self.assertRaisesRegex(RuntimeError, "already been stopped"):
            monitor.sample()


class NvidiaResourceTests(unittest.TestCase):
    def test_query_filters_pid_and_converts_mib_to_bytes(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "42, GPU-b, 12\n"
                "7, GPU-other, 900\n"
                "42, GPU-a, 1.5\n"
                "malformed\n"
            ),
            stderr="",
        )
        with mock.patch.object(resources.shutil, "which", return_value="/gpu/nvidia-smi"), mock.patch.object(
            resources.subprocess, "run", return_value=completed
        ) as run:
            snapshot = query_nvidia_process_memory(pid=42)

        self.assertEqual(snapshot.status, "ok")
        self.assertEqual(snapshot.total_used_memory_bytes, int(13.5 * 1024 * 1024))
        self.assertEqual([item.gpu_uuid for item in snapshot.contexts], ["GPU-a", "GPU-b"])
        self.assertEqual(snapshot.ignored_rows, 1)
        self.assertIn("used_gpu_memory", run.call_args.args[0][1])

    def test_missing_nvidia_tool_is_a_recorded_state(self) -> None:
        with mock.patch.object(resources.shutil, "which", return_value=None):
            snapshot = query_nvidia_process_memory(pid=42)
        self.assertEqual(snapshot.status, "unavailable")
        self.assertIsNone(snapshot.total_used_memory_bytes)


class ResourceCalibrationTests(unittest.TestCase):
    def test_calibration_is_fast_structured_and_nonnegative(self) -> None:
        calibration = calibrate_resource_monitor(iterations=3, repeats=2)
        self.assertEqual(calibration.iterations, 3)
        self.assertEqual(calibration.repeats, 2)
        self.assertFalse(calibration.include_nvidia)
        self.assertGreaterEqual(calibration.wall_seconds_per_sample, 0)
        self.assertGreaterEqual(calibration.process_cpu_seconds_per_sample, 0)
        self.assertEqual(len(calibration.wall_seconds_per_sample_trials), 2)
        record = json.loads(dumps_resource_record(calibration))
        self.assertEqual(record["iterations"], 3)
        self.assertIn("matched empty loop", record["semantics"])

    def test_calibration_rejects_invalid_counts(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    calibrate_resource_monitor(iterations=value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
