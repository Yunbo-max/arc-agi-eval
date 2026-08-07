from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_gridcoder_gates.py"
CONFIG = ROOT / "configs" / "gridcoder2024_gate_v3.json"


def load_auditor():
    spec = importlib.util.spec_from_file_location("gridcoder_gate_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(output_directory: Path, config: Path = CONFIG) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--output-directory",
            str(output_directory),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_static_gate_audit_passes_without_solver_execution(tmp_path: Path) -> None:
    output = tmp_path / "gridcoder-gate"
    completed = run_audit(output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))

    assert record["status"] == "passed"
    assert record["method_gate_status"] == "blocked"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert record["scope"] == "source-dependency-label-artifact-gate-audit-only"
    assert record["solver_gate_passed"] is False
    assert record["fairness"]["evidence_scope"] == "blocker_audit"
    assert record["fairness"]["performance_table_eligible"] is False
    assert record["source"]["python_file_count"] == 28
    assert len(record["source"]["tracked_file_inventory"]) == 29
    assert record["source"]["root_tracked_paths_match_snapshot"] is True
    assert record["source"]["independent_git_checkout"] is False
    assert record["source"]["revision_object_present_in_root_git"] is False
    assert record["arc_gym"]["observed_revision"] == (
        "740b443a955cdb31ee8209ee4d74af87b027926e"
    )
    assert len(record["arc_gym"]["tracked_python_inventory"]) == 7
    assert record["dependency_gate"]["status"] == "blocked"
    assert record["dependency_gate"]["missing_runner_modules"] == [
        "ARC_gym/utils/batching.py",
        "ARC_gym/utils/graphs.py",
    ]
    assert record["label_firewall_gate"]["flow_detected"] is True
    assert record["label_firewall_gate"]["test_output_subscript_lines"]
    assert record["label_firewall_gate"]["test_output_return_lines"]
    assert record["label_firewall_gate"]["all_tasks_append_lines"]
    assert record["label_firewall_gate"]["yq_assignment_lines"]
    assert record["checkpoint_gate"]["present_count"] == 0
    assert record["runtime_portability_gate"]["hardcoded_cuda_detected"] is True
    assert record["runtime_portability_gate"]["checkpoint_load_detected"] is True
    assert record["coverage_gate"]["task_count"] == 49
    assert record["coverage_gate"]["success_count"] == 33
    assert record["coverage_gate"]["failure_count"] == 16
    assert record["gate_summary"] == {"blocked": 7, "passed": 0}
    assert all(value is True for value in record["validation"].values())
    assert all(value is False for value in record["controls"].values())
    assert record["controls"]["checkpoint_bytes_read"] is False
    assert "results" not in record

    for item in record["evidence_manifest"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert any(
        item["path"] == "scripts/audit_gridcoder_gates.py"
        for item in record["evidence_manifest"]
    )

    second_output = tmp_path / "gridcoder-gate-replay"
    second = run_audit(second_output)
    assert second.returncode == 0, second.stderr or second.stdout
    replay = json.loads((second_output / "run.json").read_text(encoding="utf-8"))
    assert replay["observation_digest_sha256"] == record["observation_digest_sha256"]


def test_mismatched_snapshot_expectation_is_terminal_failure(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source"]["expected_python_file_count"] = 27
    bad_config = tmp_path / "bad-config.json"
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "failed-audit"

    completed = run_audit(output, bad_config)
    assert completed.returncode == 1
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["validation"]["source_snapshot_matches"] is False
    assert record["controls"]["upstream_code_executed"] is False


def test_nonempty_output_directory_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    completed = run_audit(output)
    assert completed.returncode == 2
    assert "output directory is not empty" in completed.stderr
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_auditor_has_no_upstream_or_accelerator_imports() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden_roots = {
        "ARC_gym",
        "GridCoder2024",
        "torch",
        "requests",
        "socket",
        "importlib",
    }
    assert not {name.split(".", 1)[0] for name in imported} & forbidden_roots


def test_config_path_traversal_is_rejected_before_evidence_reads(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source"]["critical_files"][0]["path"] = "../../etc/passwd"
    bad_config = tmp_path / "path-traversal.json"
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "path-traversal-run"
    completed = run_audit(output, bad_config)
    assert completed.returncode == 1
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["type"] == "ValueError"
    assert "contained relative file path" in record["error"]["message"]
    assert "evidence_manifest" not in record


def test_empty_controls_are_rejected_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["controls"] = {}
    bad_config = tmp_path / "empty-controls.json"
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "empty-controls-run"
    completed = run_audit(output, bad_config)
    assert completed.returncode == 1
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["type"] == "ValueError"
    assert "config.controls keys mismatch" in record["error"]["message"]


def test_contained_checkpoint_and_data_critical_paths_are_rejected() -> None:
    auditor = load_auditor()
    for injected in (
        "model_full.pth",
        "ARC/data/evaluation/private-task.json",
    ):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["source"]["critical_files"][0] = {
            "path": injected,
            "sha256": "0" * 64,
        }
        try:
            auditor.validate_config(config)
        except ValueError as error:
            assert "critical_files path set mismatch" in str(error)
        else:
            raise AssertionError(f"unsafe contained path accepted: {injected}")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["arc_gym"]["critical_files"][0] = {
        "path": "weights/model_full.pth",
        "sha256": "0" * 64,
    }
    try:
        auditor.validate_config(config)
    except ValueError as error:
        assert "critical_files path set mismatch" in str(error)
    else:
        raise AssertionError("unsafe ARC_gym critical path accepted")

    for section, injected in (
        ("source", "model_full.pth"),
        ("source", "ARC/data/evaluation/private-task.json"),
        ("arc_gym", "weights/model_full.pth"),
    ):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config[section]["expected_tracked_paths"].append(injected)
        try:
            auditor.validate_config(config)
        except ValueError as error:
            assert "expected_tracked_paths mismatch" in str(error)
        else:
            raise AssertionError(f"unsafe tracked path accepted: {section}/{injected}")


def test_forbidden_material_aborts_before_allowlist_reads() -> None:
    auditor = load_auditor()
    for checkpoints, data_present in (
        ([{"present": True}], False),
        ([{"present": False}], True),
    ):
        try:
            auditor.assert_forbidden_material_absent(checkpoints, data_present)
        except RuntimeError as error:
            assert "aborting before retained-source reads" in str(error)
        else:
            raise AssertionError("forbidden material did not abort the audit")


def test_tracked_allowlist_rejects_extra_path_before_file_inspection(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    content = repository / "source"
    content.mkdir(parents=True)
    (content / "safe.py").write_text("x = 1\n", encoding="utf-8")
    (content / "model_full.pth").write_text("must-not-read", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "add", "source/safe.py", "source/model_full.pth"],
        check=True,
    )
    try:
        auditor.tracked_allowlist(repository, "source", content, ["safe.py"])
    except ValueError as error:
        assert "allowlist mismatch" in str(error)
    else:
        raise AssertionError("extra tracked checkpoint path was accepted")


def test_tracked_allowlist_rejects_symlinks(tmp_path: Path) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    content = repository / "source"
    content.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (content / "safe.py").symlink_to(outside)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "add", "source/safe.py"], check=True
    )
    try:
        auditor.tracked_allowlist(repository, "source", content, ["safe.py"])
    except ValueError as error:
        assert "regular non-symlink file" in str(error)
    else:
        raise AssertionError("tracked symlink was accepted")
