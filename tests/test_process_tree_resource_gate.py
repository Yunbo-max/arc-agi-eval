import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


from scripts.audit_process_tree_resource_gate import (
    DEFAULT_OUTPUT_DIRECTORY,
    evaluate_gate,
    parse_mountinfo,
    parse_proc_cgroup,
    probe_cgroup_environment,
    probe_nvidia,
    run_setsid_escape_fixture,
    write_immutable_report,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CgroupProbeTests(unittest.TestCase):
    def test_parses_v1_mount_and_namespaced_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary) / "memory"
            mount.mkdir()
            (mount / "cgroup.procs").write_text("1\n", encoding="utf-8")
            mountinfo = (
                f"11 1 0:11 /docker/unit {mount} rw,nosuid,nodev,noexec,relatime "
                "- cgroup cgroup rw,memory\n"
            )
            membership = "12:memory:/docker/unit\n"
            result = probe_cgroup_environment(
                mountinfo_text=mountinfo, membership_text=membership
            )

        self.assertEqual(result["hierarchy"], "v1")
        self.assertTrue(result["cgroup_v1_mounted"])
        self.assertFalse(result["cgroup_v2_mounted"])
        self.assertEqual(
            result["mounts"][0]["membership_directory"]["path"], str(mount)
        )
        self.assertFalse(result["delegation"]["delegation_verified"])

    def test_observes_v2_access_without_claiming_verified_delegation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary) / "unified"
            membership_directory = mount / "job"
            membership_directory.mkdir(parents=True)
            files = {
                "cgroup.controllers": "cpu memory pids\n",
                "cgroup.events": "populated 1\n",
                "cgroup.procs": f"{os.getpid()}\n",
                "cgroup.subtree_control": "\n",
                "cpu.stat": "usage_usec 1\n",
                "memory.current": "1\n",
                "memory.events": "oom 0\n",
                "memory.peak": "1\n",
                "pids.current": "1\n",
            }
            for name, value in files.items():
                (membership_directory / name).write_text(value, encoding="utf-8")
            mountinfo = (
                f"22 1 0:22 / {mount} rw,nosuid,nodev,noexec,relatime "
                "- cgroup2 cgroup rw\n"
            )
            result = probe_cgroup_environment(
                mountinfo_text=mountinfo, membership_text="0::/job\n"
            )

        delegation = result["delegation"]
        self.assertEqual(result["hierarchy"], "v2")
        self.assertTrue(delegation["required_controllers_visible"])
        self.assertTrue(delegation["writable_by_non_mutating_access_checks"])
        self.assertFalse(delegation["mutation_test_performed"])
        self.assertFalse(delegation["delegation_verified"])
        self.assertFalse(delegation["usable_for_gate"])

    def test_parsers_ignore_malformed_and_non_cgroup_lines(self) -> None:
        mounts = parse_mountinfo(
            "malformed\n"
            "1 0 0:1 / / rw - overlay overlay rw\n"
            "2 0 0:2 / /cg rw - cgroup2 cgroup rw\n"
        )
        memberships = parse_proc_cgroup("bad\n0::/unit\nnot-int:cpu:/x\n")
        self.assertEqual(len(mounts), 1)
        self.assertEqual(mounts[0]["filesystem_type"], "cgroup2")
        self.assertEqual(memberships, [
            {
                "line_number": 2,
                "hierarchy_id": 0,
                "controllers": [],
                "path": "/unit",
            }
        ])


class SetsidFixtureTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "setsid"), "setsid is POSIX-only")
    def test_fixture_exits_normally_after_escaping_the_root_pgid(self) -> None:
        result = run_setsid_escape_fixture()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["root_return_code"], 0)
        self.assertTrue(result["assertions"]["child_escaped_root_process_group"])
        self.assertTrue(result["assertions"]["setsid_preserved_cgroup_membership"])
        self.assertTrue(result["assertions"]["child_reaped_by_normal_wait"])
        self.assertEqual(result["signals_sent"], [])
        self.assertFalse(result["timeout_used"])
        self.assertFalse(result["gpu_used"])


class NvidiaProbeTests(unittest.TestCase):
    def test_read_only_snapshot_parses_occupancy_and_disabled_accounting(self) -> None:
        calls: list[list[str]] = []

        def runner(arguments: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(arguments)
            if arguments[1].startswith("--query-gpu="):
                stdout = "0, Test GPU, GPU-1, 100, 25, 75, 40, Default, Disabled\n"
            elif arguments[1].startswith("--query-compute-apps="):
                stdout = "GPU-1, 123, worker, 24\n"
            else:
                stdout = ""
            return subprocess.CompletedProcess(arguments, 0, stdout, "")

        result = probe_nvidia(
            nvidia_smi="/usr/bin/nvidia-smi",
            nvml_library="libnvidia-ml.so.1",
            python_binding="pynvml",
            runner=runner,
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["read_only_queries"])
        self.assertFalse(result["gpu_workload_launched"])
        self.assertFalse(result["accounting_enabled_on_all_devices"])
        self.assertTrue(result["gpu_occupied"])
        self.assertEqual(result["devices"][0]["memory_used_bytes"], 25 * 1024**2)
        self.assertEqual(result["compute_processes"][0]["pid"], 123)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[1].startswith("--query-") for call in calls))

    def test_missing_nvidia_smi_is_explicit_and_does_not_initialize_nvml(self) -> None:
        result = probe_nvidia(
            nvidia_smi="",
            nvml_library="",
            python_binding="",
            runner=lambda arguments: (_ for _ in ()).throw(AssertionError(arguments)),
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["nvml_initialized_by_audit"])
        self.assertFalse(result["gpu_workload_launched"])
        self.assertEqual(result["commands"], [])


class GateAndReportTests(unittest.TestCase):
    def test_prerequisite_audit_never_claims_child_inclusive_gate(self) -> None:
        cgroup = {
            "hierarchy": "v1",
            "delegation": {"writable_by_non_mutating_access_checks": False},
        }
        nvidia = {
            "status": "ok",
            "accounting_enabled_on_all_devices": False,
            "gpu_occupied": True,
        }
        fixture = {"status": "passed"}
        gate = evaluate_gate(cgroup, nvidia, fixture)
        blocker_codes = {item["code"] for item in gate["blockers"]}
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "blocked")
        self.assertIn("cgroup_v2_unified_unavailable", blocker_codes)
        self.assertIn("process_tree_backend_not_implemented_or_calibrated", blocker_codes)

    def test_immutable_writer_refuses_even_an_empty_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing_empty = Path(temporary) / "existing-empty"
            existing_empty.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_immutable_report(existing_empty, {"status": "passed"})

            output = Path(temporary) / "new-report"
            destination = write_immutable_report(output, {"status": "passed"})
            original = destination.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_immutable_report(output, {"status": "failed"})
            self.assertEqual(destination.read_bytes(), original)

    def test_script_contains_no_process_kill_or_gpu_mutation_calls(self) -> None:
        source = (
            ROOT / "scripts" / "audit_process_tree_resource_gate.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("os.kill(", source)
        self.assertNotIn("os.killpg(", source)
        self.assertNotIn(".terminate(", source)
        self.assertNotIn(".kill(", source)
        self.assertNotIn("nvmlInit", source)

    @unittest.skipUnless(
        (DEFAULT_OUTPUT_DIRECTORY / "run.json").is_file(),
        "immutable host report has not been materialized yet",
    )
    def test_materialized_report_is_blocked_and_hash_bound(self) -> None:
        run_path = DEFAULT_OUTPUT_DIRECTORY / "run.json"
        record = json.loads(run_path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["gate"]["status"], "blocked")
        self.assertFalse(record["gate"]["passed"])
        self.assertFalse(record["resource_claim"]["children_included"])
        self.assertFalse(record["resource_claim"]["process_tree_measurement_performed"])
        self.assertEqual(record["safety"]["signals_sent"], [])
        self.assertFalse(record["safety"]["cgroup_mutations_attempted"])
        self.assertFalse(record["safety"]["gpu_workload_launched"])
        self.assertTrue(
            record["setsid_escape_fixture"]["assertions"]
            ["child_escaped_root_process_group"]
        )
        for item in record["implementation_files"]:
            path = ROOT / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(sha256_file(path), item["sha256"])


if __name__ == "__main__":
    unittest.main()
