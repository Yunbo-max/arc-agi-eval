from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import pytest


ROOT = Path(__file__).absolute().parents[1]
SCRIPT = ROOT / "scripts" / "audit_trm_gates.py"
LAUNCHER = ROOT / "scripts" / "launch_trm_gate.py"
SUPPORT = ROOT / "scripts" / "audit_batch_c_static_gates.py"
CONFIG = ROOT / "configs" / "trm_gate_v1.json"
RUNNER_MANIFEST = ROOT / "configs" / "trm_gate_runner_manifest_v1.json"
SAFE_TEST_ROOT = Path("/tmp/arc-agi-eval-trm-tests")
PYTHON = Path("/usr/bin/python3")


@pytest.fixture
def gate_tmp_path(tmp_path: Path):
    path = SAFE_TEST_ROOT / f"{os.getpid()}-{tmp_path.name}-{time.time_ns()}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def load_auditor():
    spec = importlib.util.spec_from_file_location("trm_gate_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads(RUNNER_MANIFEST.read_text(encoding="utf-8"))
    support_member = next(
        item for item in manifest["members"] if item["role"] == "support"
    )
    support_payload = SUPPORT.read_bytes()
    assert hashlib.sha256(support_payload).hexdigest() == support_member["sha256"]
    module.core = module._execute_verified_support(support_payload)
    return module


def attach_verified_launcher_context(auditor) -> None:
    spec = importlib.util.spec_from_file_location("trm_gate_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest, manifest_payload, payloads = launcher.load_verified_manifest(
        "configs/trm_gate_runner_manifest_v1.json"
    )
    del manifest
    auditor.__verified_runner_manifest_context__ = {
        "manifest_path": "configs/trm_gate_runner_manifest_v1.json",
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_payload": manifest_payload,
        "member_payloads": payloads,
        "executed_auditor_sha256": hashlib.sha256(
            payloads["scripts/audit_trm_gates.py"]
        ).hexdigest(),
        "operator_supplied_manifest_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "launcher_source_execution": {
            "mode": "canonical-direct-script-source",
            "name_is_main": True,
            "spec_is_none": True,
            "cached_is_none": True,
            "argv0_is_canonical_script": True,
            "python_executable": "/usr/bin/python3",
            "required_python_executable": "/usr/bin/python3",
            "isolated": True,
            "ignore_environment": True,
            "no_user_site": True,
            "no_site": True,
            "dont_write_bytecode": True,
            "sys_path_excludes_cwd_and_relative_entries": True,
        },
    }


def run_audit(
    output: Path, config: Path = CONFIG
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-B",
            "-S",
            str(LAUNCHER),
            "--manifest",
            "configs/trm_gate_runner_manifest_v1.json",
            "--expected-manifest-sha256",
            hashlib.sha256(RUNNER_MANIFEST.read_bytes()).hexdigest(),
            "--config",
            str(config),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_static_gate_passes_while_method_remains_blocked(
    gate_tmp_path: Path,
) -> None:
    output = gate_tmp_path / "trm-gate"
    completed = run_audit(output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))

    assert record["status"] == "passed"
    assert record["method_gate_status"] == "blocked"
    assert record["scope"] == (
        "source-artifact-dataset-label-resource-gate-audit-only"
    )
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["solver_gate_passed"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert "predictions" not in record
    assert "results" not in record
    assert record["gate_summary"] == {"blocked": 10, "passed": 2}
    assert len(record["blockers"]) == 10
    assert all(item["status"] == "blocked" for item in record["blockers"])
    assert all(record["validation"].values())
    assert sorted(path.name for path in output.iterdir()) == ["run.json"]

    source = record["source"]
    assert source["observed_revision"] == (
        "c01103738605ba39d1430519b1ee0c62f4c707f8"
    )
    assert source["observed_commit_tree"] == (
        "f7402e4124f7bd9a07f1b0ef392efeb9e0ee7649"
    )
    assert source["tracked_file_count"] == 40
    assert source["tracked_worktree_bytes"] == 11_503_764
    assert source["retained_byte_exact_file_count"] == 28
    assert source["retained_byte_exact_bytes"] == 160_840
    assert source["metadata_only_tracked_file_count"] == 12
    assert source["metadata_only_tracked_bytes"] == 11_342_924
    assert source["root_license_paths"] == ["LICENSE"]
    assert source["working_tree_all_files_byte_exact_verified"] is False
    restricted = source["metadata_only_inventory"]
    assert len(restricted) == 12
    assert sum(item["bytes"] for item in restricted) == 11_342_924
    assert all(item["worktree_bytes_read"] is False for item in restricted)
    assert all(item["worktree_sha256"] is None for item in restricted)
    assert all(Path(item["path"]).suffix in {".json", ".png"} for item in restricted)
    assert sum(item["path"].endswith("_solutions.json") for item in restricted) == 5
    assert sum(item["path"].endswith("_challenges.json") for item in restricted) == 5

    analysis = record["static_analysis"]
    expected = {
        "repository_archived_notice": True,
        "readme_citation_year": 2025,
        "asset_status_created_year": 2026,
        "official_arc_prize_entry_evidence_detected": False,
        "readme_scores_are_unverified_self_report": True,
        "reported_arc_agi_1_percent": 45,
        "reported_arc_agi_2_percent": 8,
        "paper_checkpoint_reference_detected": False,
        "asset_status_checkpoint_count": 0,
        "arc_builder_reads_solutions": True,
        "arc_builder_injects_solution_into_test_output": True,
        "arc_builder_missing_solution_fails_closed": False,
        "runtime_test_batches_include_labels": True,
        "loss_head_reads_current_labels": True,
        "primary_model_inner_reads_labels": False,
        "eval_save_outputs_default": [],
        "evaluation_return_keys_include_configured_save_outputs": True,
        "evaluation_collects_configured_keys_from_full_batch": True,
        "evaluation_persists_configured_outputs_with_torch_save": True,
        "evaluation_labels_can_be_persisted_via_eval_save_outputs": True,
        "evaluation_mode_uses_fixed_max_steps": True,
        "evaluator_required_outputs_include_labels": False,
        "evaluator_hashes_expected_output": True,
        "evaluator_submission_k_default": 2,
        "evaluator_aggregated_voting_default": True,
        "evaluator_aggregated_voting_state_checkpointed": False,
        "evaluator_missing_prediction_continues": True,
        "evaluator_explicit_final_tie_break": False,
        "evaluator_gather_object_unconditional": True,
        "distributed_init_requires_local_rank": True,
        "torch_load_call_count": 1,
        "torch_save_call_count": 2,
        "checkpoint_save_model_state_only": True,
        "checkpoint_save_includes_optimizer_rng_and_evaluator": False,
        "checkpoint_save_excludes_optimizer_state": True,
        "checkpoint_save_excludes_train_step": True,
        "checkpoint_save_excludes_rng_state": True,
        "checkpoint_save_excludes_evaluator_state": True,
        "puzzle_embedding_is_persistent_buffer": True,
        "prior_smoke_torch_requirement_match": False,
        "mixed_arc1_arc2_data_path_guard_detected": False,
        "training_sampler_uses_global_numpy_choice": True,
        "training_sampler_uses_seeded_philox": True,
        "pretrain_seeds_torch_rng": True,
        "pretrain_seeds_numpy_rng": False,
        "model_uses_torch_exploration_rng": True,
        "rng_state_resume_contract_complete": False,
        "dependency_hash_count": 0,
        "dependency_lock_has_transitive_hash_closure": False,
        "arc_builder_default_num_aug": 1000,
        "arc_flat_sequence_length": 900,
        "trm_hidden_size": 512,
        "trm_halt_max_steps": 16,
        "default_evaluation_interval_count": 10,
        "default_checkpoint_count_upper_bound": 10,
        "checkpoint_retention_policy_detected": False,
        "static_storage_lower_bound_available": False,
        "default_evaluates_from_first_interval": True,
        "fixed_checkpoint_selection_policy_documented": False,
        "fixed_seed_repetition_policy_documented": False,
        "root_license_mit": True,
    }
    for key, value in expected.items():
        assert analysis[key] == value
    assert analysis["torch_load_calls"] == [
        {"lineno": 249, "map_location": "cuda", "weights_only": "absent"}
    ]
    assert analysis["torch_save_calls"] == [
        {
            "lineno": 241,
            "payload_expression": "train_state.model.state_dict()",
            "purpose": "model-checkpoint",
        },
        {
            "lineno": 431,
            "payload_expression": "save_preds",
            "purpose": "configured-evaluation-outputs",
        },
    ]
    assert analysis["evaluator_state_fields"] == ["_local_hmap", "_local_preds"]
    assert analysis["evaluator_serialization_methods"] == []

    controls = record["controls"]
    for key in (
        "upstream_imported",
        "upstream_executed",
        "checkpoint_opened",
        "checkpoint_loaded",
        "provider_or_wandb_code_path_executed",
        "solver_executed",
        "solver_prediction_produced",
        "auditor_process_arc_json_worktree_leaf_bytes_read",
        "auditor_process_solution_json_worktree_leaf_bytes_read",
        "auditor_process_image_worktree_leaf_bytes_read",
        "metadata_only_worktree_leaf_content_reads",
    ):
        assert controls[key] in {False, 0}
    assert controls["network_used"] is None
    assert controls["gpu_used"] is None
    assert controls["network_usage_measurement"] == "not-instrumented"
    assert controls["gpu_usage_measurement"] == "not-instrumented"
    assert controls["network_namespace_enforced"] is False
    assert controls["gpu_device_namespace_enforced"] is False
    assert controls["retained_source_read_attempts"] == 56
    assert controls["bound_metadata_read_attempts"] == 4
    assert controls["git_subprocesses_started"] == 6
    assert controls["git_subprocess_source_local_config_available"] is False
    assert controls["git_subprocess_isolated_git_directory_used"] is True
    assert record["resources"]["provider_requests"] is None
    assert record["resources"]["currency_spend_usd"] is None
    assert record["resources"]["intentional_provider_requests"] == 0
    assert record["resources"]["intentional_currency_spend_usd"] == 0.0
    assert record["resources"]["gpu_used"] is None
    assert record["resources"]["network_used"] is None

    retained_reads = [
        item
        for item in record["read_ledger"]["file_read_attempts"]
        if item["category"] == "retained_source_text"
    ]
    assert len(retained_reads) == 56
    assert all(item["status"] == "completed" for item in retained_reads)
    assert not any(
        Path(item["path"]).suffix.lower() in {".json", ".png"}
        for item in retained_reads
    )
    assert all(
        item["worktree_content_requested"] is False
        for item in record["read_ledger"]["git_subprocesses"]
    )
    assert "not a solver smoke" in record["claim_boundary"]
    assert record["classification"]["method_paper_year"] == 2025
    assert record["classification"]["classification_year_bucket"] == (
        "paper_method_2025"
    )
    assert record["classification"]["asset_snapshot_year"] == 2026
    assert record["classification"]["official_arc_prize_entry_verified"] is False
    assert "unverified upstream README" in record["claim_boundary"]


def test_observation_digest_and_source_read_ledger_are_replay_stable(
    gate_tmp_path: Path,
) -> None:
    records = []
    for name in ("first", "second"):
        output = gate_tmp_path / name
        completed = run_audit(output)
        assert completed.returncode == 0, completed.stderr or completed.stdout
        records.append(json.loads((output / "run.json").read_text(encoding="utf-8")))
    assert records[0]["observation_digest_sha256"] == records[1][
        "observation_digest_sha256"
    ]
    observation_keys = (
        "method_id",
        "scope",
        "source",
        "static_analysis",
        "bound_evidence",
        "passed_gates",
        "blockers",
        "gate_summary",
        "controls",
        "classification",
        "benchmark_policy",
        "prior_evidence_interpretation",
    )
    assert records[0]["observation_digest_sha256"] == canonical_sha256(
        {key: records[0][key] for key in observation_keys}
    )
    assert records[0]["source"] == records[1]["source"]
    assert records[0]["static_analysis"] == records[1]["static_analysis"]


def test_config_is_canonical_closed_and_binds_support() -> None:
    auditor = load_auditor()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert canonical_sha256(config) == auditor.EXPECTED_CONFIG_CANONICAL_SHA256
    assert hashlib.sha256(SUPPORT.read_bytes()).hexdigest() == (
        auditor.EXPECTED_SUPPORT_SHA256
    )
    auditor.validate_config(config)
    mutated = copy.deepcopy(config)
    mutated["controls"]["gpu_allowed"] = True
    with pytest.raises(ValueError, match="hardcoded v1 contract"):
        auditor.validate_config(mutated)


def test_runner_manifest_is_internally_closed_and_requires_external_anchor() -> None:
    manifest = json.loads(RUNNER_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["manifest_id"] == "trm-gate-runner-manifest-v1"
    assert manifest["member_count"] == 5
    assert manifest["members_sha256"] == canonical_sha256(manifest["members"])
    expected = {
        "launcher": "scripts/launch_trm_gate.py",
        "auditor": "scripts/audit_trm_gates.py",
        "support": "scripts/audit_batch_c_static_gates.py",
        "config": "configs/trm_gate_v1.json",
        "source_lock": "configs/source_locks.json",
    }
    assert {item["role"]: item["path"] for item in manifest["members"]} == expected
    for item in manifest["members"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        if item["role"] == "config":
            assert item["canonical_sha256"] == canonical_sha256(
                json.loads(path.read_text(encoding="utf-8"))
            )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "auditor_external_lock" not in config
    assert "import scripts.audit_batch_c_static_gates" not in SCRIPT.read_text(
        encoding="utf-8"
    )


def test_direct_auditor_entry_is_rejected_before_output(
    gate_tmp_path: Path,
) -> None:
    output = gate_tmp_path / "direct-entry-must-not-exist"
    completed = subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-B",
            "-S",
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--runner-manifest",
            str(RUNNER_MANIFEST),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "must enter through the verified launcher" in completed.stderr
    assert not output.exists()


def test_imported_launcher_cannot_publish_even_with_support_poison(
    gate_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location("trm_gate_launcher_poison", LAUNCHER)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    poisoned = types.ModuleType("scripts.audit_batch_c_static_gates")
    poisoned.side_effect = "must-not-execute"
    monkeypatch.setitem(sys.modules, "scripts.audit_batch_c_static_gates", poisoned)
    output = gate_tmp_path / "imported-launcher-must-not-publish"
    assert (
        launcher.main(
            [
                "--manifest",
                "configs/trm_gate_runner_manifest_v1.json",
                "--expected-manifest-sha256",
                hashlib.sha256(RUNNER_MANIFEST.read_bytes()).hexdigest(),
                "--config",
                str(CONFIG),
                "--output-directory",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_unisolated_direct_launcher_is_rejected_before_output(
    gate_tmp_path: Path,
) -> None:
    output = gate_tmp_path / "unisolated-must-not-publish"
    completed = subprocess.run(
        [
            str(PYTHON),
            "-B",
            str(LAUNCHER),
            "--manifest",
            "configs/trm_gate_runner_manifest_v1.json",
            "--expected-manifest-sha256",
            hashlib.sha256(RUNNER_MANIFEST.read_bytes()).hexdigest(),
            "--config",
            str(CONFIG),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "canonical direct-source script entry" in completed.stderr
    assert not output.exists()


def test_wrong_operator_manifest_digest_is_rejected_before_output(
    gate_tmp_path: Path,
) -> None:
    output = gate_tmp_path / "wrong-manifest-digest-must-not-publish"
    completed = subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-B",
            "-S",
            str(LAUNCHER),
            "--manifest",
            "configs/trm_gate_runner_manifest_v1.json",
            "--expected-manifest-sha256",
            "0" * 64,
            "--config",
            str(CONFIG),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "operator-supplied runner-manifest SHA-256 mismatch" in completed.stderr
    assert not output.exists()


def test_runner_and_support_provenance_are_terminally_bound(
    gate_tmp_path: Path,
) -> None:
    output = gate_tmp_path / "provenance"
    completed = run_audit(output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads(RUNNER_MANIFEST.read_text(encoding="utf-8"))
    members = {item["role"]: item for item in manifest["members"]}
    assert record["runner_provenance"] == {
        "path": "scripts/audit_trm_gates.py",
        "bytes": SCRIPT.stat().st_size,
        "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "expected_sha256": members["auditor"]["sha256"],
        "manifest_bound": True,
        "executed_from_verified_source_bytes": True,
        "terminal_bytes_equal": True,
    }
    assert record["support_provenance"] == {
        "path": "scripts/audit_batch_c_static_gates.py",
        "bytes": SUPPORT.stat().st_size,
        "sha256": hashlib.sha256(SUPPORT.read_bytes()).hexdigest(),
        "expected_sha256": members["support"]["sha256"],
        "loaded_via": "preverified-source-bytes-compile-exec",
        "normal_import_used": False,
        "pyc_used": False,
        "terminal_bytes_equal": True,
    }
    assert record["runner_manifest_provenance"] == {
        "path": "configs/trm_gate_runner_manifest_v1.json",
        "sha256": hashlib.sha256(RUNNER_MANIFEST.read_bytes()).hexdigest(),
        "operator_supplied_expected_sha256": hashlib.sha256(
            RUNNER_MANIFEST.read_bytes()
        ).hexdigest(),
        "operator_supplied_digest_matched": True,
        "repository_external_signature_verified": False,
        "next_input_freeze_anchor_required": True,
        "members_sha256": manifest["members_sha256"],
        "member_count": 5,
        "launcher_sha256": members["launcher"]["sha256"],
        "terminal_bytes_equal": True,
    }
    assert record["launcher_provenance"] == {
        "path": "scripts/launch_trm_gate.py",
        "sha256": members["launcher"]["sha256"],
        "canonical_direct_script_source": True,
        "isolated_python_flags_required": True,
        "terminal_file_matches_manifest": True,
        "executed_launcher_bytes_preauthenticated": False,
        "interpreter_binary_digest_verified": False,
        "source_execution_context": {
            "mode": "canonical-direct-script-source",
            "name_is_main": True,
            "spec_is_none": True,
            "cached_is_none": True,
            "argv0_is_canonical_script": True,
            "python_executable": "/usr/bin/python3",
            "required_python_executable": "/usr/bin/python3",
            "isolated": True,
            "ignore_environment": True,
            "no_user_site": True,
            "no_site": True,
            "dont_write_bytecode": True,
            "sys_path_excludes_cwd_and_relative_entries": True,
        },
    }
    assert record["replay_consistency"]["complete_observation_count"] == 2
    assert record["replay_consistency"]["equal"] is True
    assert record["commit_consistency"]["toctou_eliminated"] is False
    assert record["validation"]["runner_sha256_manifest_bound"] is True
    assert (
        record["validation"]["support_loaded_only_after_sha256_verification"]
        is True
    )


def test_every_noncanonical_config_is_rejected_before_output(
    gate_tmp_path: Path,
) -> None:
    canary = gate_tmp_path / "arc-agi_evaluation_solutions.json"
    canary.write_bytes(b"must-never-open")
    output = gate_tmp_path / "must-not-exist"
    completed = run_audit(output, canary)
    assert completed.returncode == 2
    assert "production config path must equal" in completed.stderr
    assert not output.exists()
    assert canary.read_bytes() == b"must-never-open"


@pytest.mark.parametrize("kind", ["empty-dir", "nonempty-dir", "file", "fifo", "symlink"])
def test_output_requires_a_completely_fresh_leaf(
    gate_tmp_path: Path, kind: str
) -> None:
    output = gate_tmp_path / "occupied"
    marker = gate_tmp_path / "marker"
    marker.write_text("preserve", encoding="utf-8")
    if kind == "empty-dir":
        output.mkdir()
    elif kind == "nonempty-dir":
        output.mkdir()
        (output / "keep.txt").write_text("keep", encoding="utf-8")
    elif kind == "file":
        output.write_text("keep", encoding="utf-8")
    elif kind == "fifo":
        os.mkfifo(output)
    else:
        output.symlink_to(marker)
    completed = run_audit(output)
    assert completed.returncode == 2
    assert "output path must not exist" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    if kind == "nonempty-dir":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_output_parent_symlink_is_rejected_without_target_write(
    gate_tmp_path: Path,
) -> None:
    real = gate_tmp_path / "real"
    real.mkdir()
    linked = gate_tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    completed = run_audit(linked / "run")
    assert completed.returncode == 2
    assert not (real / "run").exists()


def test_failure_record_never_claims_solver_evidence() -> None:
    auditor = load_auditor()
    record = auditor.failure_record(
        "failed-run", ValueError("injected"), auditor.core.ReadLedger({})
    )
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["solver_prediction_produced"] is False
    assert record["solver_gate_passed"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert "predictions" not in record
    assert "results" not in record
    assert record["controls"]["network_used"] is None
    assert record["controls"]["gpu_used"] is None
    assert record["controls"]["measurement_status"] == (
        "unknown-after-audit-failure"
    )


def test_injected_static_failure_publishes_one_fail_closed_record(
    gate_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auditor = load_auditor()
    attach_verified_launcher_context(auditor)
    output = gate_tmp_path / "failed"

    def injected_failure(*args, **kwargs):
        raise ValueError("injected TRM static failure")

    monkeypatch.setattr(auditor, "run_static_audit", injected_failure)
    assert (
        auditor.main(
        [
            "--config",
            str(CONFIG),
            "--runner-manifest",
            str(RUNNER_MANIFEST),
            "--output-directory",
            str(output),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert '"status": "failed"' in captured.err
    assert sorted(path.name for path in output.iterdir()) == ["run.json"]
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["failure"] == {
        "type": "ValueError",
        "message": "injected TRM static failure",
    }
    assert record["solver_prediction_produced"] is False
    assert record["read_ledger"] == {
        "file_read_attempts": [],
        "git_subprocesses": [],
    }


def test_replay_difference_fails_closed_before_positive_attestation(
    gate_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auditor = load_auditor()
    attach_verified_launcher_context(auditor)
    original = auditor.run_static_audit
    calls = 0

    def differing_replay(*args, **kwargs):
        nonlocal calls
        calls += 1
        record = original(*args, **kwargs)
        if calls == 2:
            record["observation_digest_sha256"] = "0" * 64
        return record

    monkeypatch.setattr(auditor, "run_static_audit", differing_replay)
    output = gate_tmp_path / "replay-difference"
    assert (
        auditor.main(
            [
                "--config",
                str(CONFIG),
                "--runner-manifest",
                str(RUNNER_MANIFEST),
                "--output-directory",
                str(output),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "complete replay observation changed" in captured.err
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["solver_prediction_produced"] is False
    assert len(record["read_ledger"]["file_read_attempts"]) == 128
    assert len(record["read_ledger"]["git_subprocesses"]) == 12
