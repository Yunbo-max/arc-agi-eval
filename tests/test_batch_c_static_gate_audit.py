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

import pytest


ROOT = Path(__file__).absolute().parents[1]
SCRIPT = ROOT / "scripts" / "audit_batch_c_static_gates.py"
CONFIG = ROOT / "configs" / "batch_c_static_gate_v1.json"
METHOD_EXPECTATIONS = {
    "arcmemo": {
        "tracked": 848,
        "retained": 32,
        "metadata_only": 816,
        "blockers": 9,
        "analysis": {
            "default_problem_data_is_null": True,
            "default_long_cot_selection_disabled": True,
            "continual_strict_filter_precedes_memory_update": True,
            "continual_selection_dry_run_propagated": False,
            "continual_update_llm_dry_run_propagated": False,
            "executor_exec_call_count": 2,
            "official_attempt_limit_default_is_none": True,
        },
    },
    "arc-lang-public": {
        "tracked": 42,
        "retained": 27,
        "metadata_only": 15,
        "blockers": 8,
        "analysis": {
            "default_runner_passes_solution_path": True,
            "truth_read_precedes_solver_call": True,
            "challenge_only_none_branch_detected": True,
            "raw_input_extra_forbid_declared": False,
            "expected_eager_provider_clients_bound": True,
            "pre_request_budget_reservation_detected": False,
            "generated_code_execution_detected": False,
        },
    },
    "epang-arc-agi": {
        "tracked": 1256,
        "retained": 16,
        "metadata_only": 1240,
        "blockers": 9,
        "analysis": {
            "submission_eager_imports_all_data": True,
            "arc1_solution_paths_at_module_import": True,
            "inline_test_aggregate_metrics_detected": True,
            "pickle_load_call_count": 2,
            "llm_completion_to_generated_python_executor": True,
            "executor_isolation_terms": [],
            "pre_request_budget_reservation_detected": False,
        },
    },
}
SAFE_TEST_ROOT = Path("/tmp/arc-agi-eval-batch-c-tests")


@pytest.fixture
def gate_tmp_path(tmp_path: Path):
    path = SAFE_TEST_ROOT / f"{os.getpid()}-{tmp_path.name}-{time.time_ns()}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def load_auditor():
    spec = importlib.util.spec_from_file_location("batch_c_gate_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(
    method_id: str, output: Path, config: Path = CONFIG
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
            sys.executable,
            str(SCRIPT),
            "--method-id",
            method_id,
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


@pytest.mark.parametrize("method_id", list(METHOD_EXPECTATIONS))
def test_static_gate_passes_while_method_remains_blocked(
    gate_tmp_path: Path, method_id: str
) -> None:
    output = gate_tmp_path / method_id
    completed = run_audit(method_id, output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    expected = METHOD_EXPECTATIONS[method_id]

    assert record["status"] == "passed"
    assert record["method_gate_status"] == "blocked"
    assert record["scope"] == (
        "static-source-label-api-artifact-blocker-audit-only"
    )
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["solver_gate_passed"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert "predictions" not in record
    assert "results" not in record
    assert record["resources"]["provider_requests"] == 0
    assert record["resources"]["currency_spend_usd"] == 0.0
    assert record["resources"]["gpu_used"] is False
    assert record["resources"]["network_used"] is False
    assert record["resources"]["wall_seconds"] >= 0
    assert record["runner_provenance"] == {
        "path": "scripts/audit_batch_c_static_gates.py",
        "bytes": SCRIPT.stat().st_size,
        "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "terminal_bytes_equal": True,
    }
    assert sorted(item.name for item in output.iterdir()) == ["run.json"]

    source = record["source"]
    assert source["tracked_file_count"] == expected["tracked"]
    assert source["retained_byte_exact_file_count"] == expected["retained"]
    assert source["metadata_only_tracked_file_count"] == expected["metadata_only"]
    assert source["working_tree_all_files_byte_exact_verified"] is False
    assert source["root_license_paths"] == []
    assert len(record["blockers"]) == expected["blockers"]
    assert all(item["status"] == "blocked" for item in record["blockers"])
    for key, value in expected["analysis"].items():
        assert record["static_analysis"][key] == value
    assert all(record["validation"].values())

    controls = record["controls"]
    for key in (
        "network_used",
        "gpu_used",
        "upstream_imported",
        "upstream_executed",
        "provider_called",
        "generated_code_executed",
        "solver_prediction_produced",
        "auditor_process_arc_or_solution_worktree_leaf_bytes_read",
        "auditor_process_pickle_worktree_leaf_bytes_read",
        "metadata_only_worktree_leaf_content_reads",
    ):
        assert controls[key] in {False, 0}
    assert controls["retained_source_read_attempts"] == 2 * expected["retained"]
    assert controls["git_subprocesses_started"] == 6
    assert controls["git_subprocess_source_local_config_available"] is False
    assert controls["git_subprocess_isolated_git_directory_used"] is True
    assert controls["git_object_leaf_hardlink_aliases_allowed"] is False

    retained_reads = [
        item
        for item in record["read_ledger"]["file_read_attempts"]
        if item["category"] == "retained_source_text"
    ]
    assert len(retained_reads) == 2 * expected["retained"]
    forbidden = {
        ".json",
        ".jsonl",
        ".pkl",
        ".pickle",
        ".pdf",
        ".ipynb",
        ".pyc",
        ".png",
    }
    assert not any(Path(item["path"]).suffix.lower() in forbidden for item in retained_reads)
    assert all(item["status"] == "completed" for item in retained_reads)
    assert all(
        item["worktree_content_requested"] is False
        for item in record["read_ledger"]["git_subprocesses"]
    )
    assert "not a solver smoke" in record["claim_boundary"]


@pytest.mark.parametrize("method_id", list(METHOD_EXPECTATIONS))
def test_observation_digest_is_replay_stable(gate_tmp_path: Path, method_id: str) -> None:
    outputs = [gate_tmp_path / "first", gate_tmp_path / "second"]
    records = []
    for output in outputs:
        completed = run_audit(method_id, output)
        assert completed.returncode == 0, completed.stderr or completed.stdout
        records.append(json.loads((output / "run.json").read_text(encoding="utf-8")))
    assert records[0]["observation_digest_sha256"] == records[1][
        "observation_digest_sha256"
    ]
    assert records[0]["observation_digest_sha256"] == canonical_sha256(
        {
            key: records[0][key]
            for key in (
                "method_id",
                "scope",
                "source",
                "static_analysis",
                "prior_reports",
                "blockers",
                "controls",
                "benchmark_policy",
                "prior_evidence_interpretation",
            )
        }
    )


def test_config_is_canonical_and_closed() -> None:
    auditor = load_auditor()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert canonical_sha256(config) == auditor.EXPECTED_CONFIG_CANONICAL_SHA256
    assert [item["method_id"] for item in config["methods"]] == [
        "arcmemo",
        "arc-lang-public",
        "epang-arc-agi",
    ]
    auditor.validate_config(config)
    mutated = copy.deepcopy(config)
    mutated["controls"]["network_allowed"] = True
    with pytest.raises(ValueError, match="hardcoded v1 contract"):
        auditor.validate_config(mutated)


def test_every_noncanonical_config_is_rejected_before_output(gate_tmp_path: Path) -> None:
    copied = gate_tmp_path / "copied.json"
    copied.write_bytes(CONFIG.read_bytes())
    output = gate_tmp_path / "must-not-exist"
    completed = run_audit("arc-lang-public", output, copied)
    assert completed.returncode == 2
    assert "production config path must equal" in completed.stderr
    assert not output.exists()


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
    completed = run_audit("arc-lang-public", output)
    assert completed.returncode == 2
    assert "output path must not exist" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    if kind == "nonempty-dir":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_output_parent_symlink_is_rejected_without_target_write(gate_tmp_path: Path) -> None:
    real = gate_tmp_path / "real"
    real.mkdir()
    linked = gate_tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    completed = run_audit("arc-lang-public", linked / "run")
    assert completed.returncode == 2
    assert not (real / "run").exists()


@pytest.mark.parametrize(
    "forbidden_parent",
    [
        ROOT,
        ROOT / "configs",
        ROOT / "reports" / "e0-overlap",
        Path("/root/arc-paper-assets/sources/arcmemo/cache"),
        Path("/root/arc-paper-assets/sources/arc-lang-public/.git/hooks"),
    ],
)
def test_cli_rejects_output_outside_closed_namespaces_before_creation(
    forbidden_parent: Path,
) -> None:
    output = forbidden_parent / f"must-not-create-{os.getpid()}-{time.time_ns()}"
    assert not output.exists()
    completed = run_audit("arc-lang-public", output)
    assert completed.returncode == 2
    assert "output must be one fresh leaf" in completed.stderr
    assert not output.exists()


def test_output_policy_accepts_only_matching_method_report_leaf() -> None:
    auditor = load_auditor()
    auditor.validate_output_location(
        ROOT / "reports" / "arcmemo" / "fresh-run", "arcmemo"
    )
    with pytest.raises(ValueError):
        auditor.validate_output_location(
            ROOT / "reports" / "arc-lang-public" / "fresh-run", "arcmemo"
        )
    with pytest.raises(ValueError):
        auditor.validate_output_location(
            ROOT / "reports" / "arcmemo" / "nested" / "fresh-run", "arcmemo"
        )


def test_restricted_suffix_cannot_enter_retained_reader() -> None:
    auditor = load_auditor()
    ledger = auditor.ReadLedger({"data/solutions.json": "source_python"})
    with pytest.raises(ValueError, match="restricted suffix"):
        ledger.authorize_retained("data/solutions.json", "source_python")


def test_git_object_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    auditor = load_auditor()
    git_dir = tmp_path / ".git"
    object_dir = git_dir / "objects" / "aa"
    object_dir.mkdir(parents=True)
    object_leaf = object_dir / ("0" * 38)
    object_leaf.write_bytes(b"pinned-object")
    os.link(object_leaf, tmp_path / "restricted.json")
    descriptor = auditor.open_absolute_directory(git_dir)
    try:
        with pytest.raises(ValueError, match="hardlink alias"):
            auditor.git_object_metadata(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "payload",
    [
        b"100644 blob " + b"0" * 40 + b"\tgood.py\0",
        b"120000 blob " + b"0" * 40 + b"\tlink\0",
        b"100644 blob " + b"0" * 40 + b"\t../escape.py\0",
        b"100644 blob " + b"0" * 40 + b"\tmissing-nul.py",
    ],
)
def test_git_tree_parser_is_fail_closed(payload: bytes) -> None:
    auditor = load_auditor()
    if b"good.py" in payload:
        assert list(auditor.parse_git_tree(payload)) == ["good.py"]
    else:
        with pytest.raises(ValueError):
            auditor.parse_git_tree(payload)


def test_no_clobber_writer_preserves_first_record(tmp_path: Path) -> None:
    auditor = load_auditor()
    output = auditor.create_fresh_output(tmp_path / "fresh")
    try:
        auditor.write_json_no_clobber(output, {"value": "first"})
        with pytest.raises(OSError):
            auditor.write_json_no_clobber(output, {"value": "second"})
    finally:
        output.close()
    assert json.loads((tmp_path / "fresh" / "run.json").read_text())["value"] == "first"
    failed_temporaries = list((tmp_path / "fresh").glob(".run.json.tmp-*"))
    assert len(failed_temporaries) == 1
    assert failed_temporaries[0].stat().st_mode & 0o777 == 0o600


def test_writer_never_unlinks_replaced_temp_after_prepublish_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output = auditor.create_fresh_output(tmp_path / "fresh")
    original_stat = auditor.os.stat
    replaced_path: str | None = None

    def replace_before_stat(path, *args, **kwargs):
        nonlocal replaced_path
        if (
            replaced_path is None
            and isinstance(path, str)
            and path.startswith(".run.json.tmp-")
            and kwargs.get("dir_fd") == output.descriptor
        ):
            auditor.os.unlink(path, dir_fd=output.descriptor)
            attacker = auditor.os.open(
                path,
                auditor.os.O_WRONLY
                | auditor.os.O_CREAT
                | auditor.os.O_EXCL
                | auditor.os.O_NOFOLLOW,
                0o600,
                dir_fd=output.descriptor,
            )
            try:
                auditor.os.write(attacker, b"replacement-must-survive")
                auditor.os.fsync(attacker)
            finally:
                auditor.os.close(attacker)
            replaced_path = path
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "stat", replace_before_stat)
    try:
        with pytest.raises(RuntimeError, match="temporary report leaf changed"):
            auditor.write_json_no_clobber(output, {"value": "trusted"})
    finally:
        output.close()
    assert replaced_path is not None
    assert (tmp_path / "fresh" / replaced_path).read_bytes() == b"replacement-must-survive"


def test_writer_fails_closed_when_write_makes_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output = auditor.create_fresh_output(tmp_path / "fresh")
    monkeypatch.setattr(auditor.os, "write", lambda *_args, **_kwargs: 0)
    try:
        with pytest.raises(RuntimeError, match="write made no progress"):
            auditor.write_json_no_clobber(output, {"value": "trusted"})
    finally:
        output.close()
    temporaries = list((tmp_path / "fresh").glob(".run.json.tmp-*"))
    assert len(temporaries) == 1
    assert temporaries[0].stat().st_mode & 0o777 == 0o600


def test_writer_detects_temp_inode_swap_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output = auditor.create_fresh_output(tmp_path / "fresh")
    original = auditor.rename_noreplace

    def replace_then_rename(directory_fd: int, source: str, destination: str) -> None:
        os.unlink(source, dir_fd=directory_fd)
        descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, b"attacker")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        original(directory_fd, source, destination)

    monkeypatch.setattr(auditor, "rename_noreplace", replace_then_rename)
    try:
        with pytest.raises(RuntimeError, match="published report leaf differs"):
            auditor.write_json_no_clobber(output, {"value": "trusted"})
    finally:
        output.close()
    assert (tmp_path / "fresh" / "run.json").read_bytes() == b"attacker"


@pytest.mark.parametrize("attack_kind", ["replace-path", "overwrite-held-inode"])
def test_writer_rechecks_path_and_payload_after_final_directory_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack_kind: str
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "fresh"
    output = auditor.create_fresh_output(output_path)
    original_verify = output.verify
    trusted = {"value": "trusted"}
    trusted_payload = (
        json.dumps(
            trusted,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    replacement = b"X" * len(trusted_payload)
    injected = False

    def replace_after_verification() -> None:
        nonlocal injected
        original_verify()
        if injected:
            return
        try:
            auditor.os.stat(
                "run.json", dir_fd=output.descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        if attack_kind == "replace-path":
            auditor.os.unlink("run.json", dir_fd=output.descriptor)
            attacker = auditor.os.open(
                "run.json",
                auditor.os.O_WRONLY
                | auditor.os.O_CREAT
                | auditor.os.O_EXCL
                | auditor.os.O_NOFOLLOW,
                0o600,
                dir_fd=output.descriptor,
            )
        else:
            attacker = auditor.os.open(
                "run.json",
                auditor.os.O_WRONLY | auditor.os.O_NOFOLLOW,
                dir_fd=output.descriptor,
            )
        try:
            assert auditor.os.write(attacker, replacement) == len(replacement)
            auditor.os.fsync(attacker)
        finally:
            auditor.os.close(attacker)
        injected = True

    monkeypatch.setattr(output, "verify", replace_after_verification)
    try:
        with pytest.raises(RuntimeError, match="changed after commit sync"):
            auditor.write_json_no_clobber(output, trusted)
    finally:
        output.close()
    assert injected is True
    assert (output_path / "run.json").read_bytes() == replacement


def test_failure_record_never_claims_solver_evidence() -> None:
    auditor = load_auditor()
    record = auditor.failure_record(
        "arcmemo", "failed", ValueError("injected"), auditor.ReadLedger({})
    )
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert all(
        record["controls"][key] is False
        for key in (
            "network_used",
            "gpu_used",
            "upstream_imported",
            "upstream_executed",
            "provider_called",
            "generated_code_executed",
            "solver_prediction_produced",
        )
    )
