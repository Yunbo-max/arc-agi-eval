from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).absolute().parents[1]
SCRIPT = ROOT / "scripts" / "audit_barc_gates.py"
CONFIG = ROOT / "configs" / "barc_gate_v1.json"
SOURCE = Path("/root/arc-paper-assets/sources/barc")
SAFE_GIT_CONFIG = b"""[core]
\trepositoryformatversion = 1
\tfilemode = true
\tbare = false
\tlogallrefupdates = true
[remote \"origin\"]
\turl = https://github.com/xu3kev/BARC
\tfetch = +refs/heads/*:refs/remotes/origin/*
\tpromisor = true
\tpartialclonefilter = blob:none
[branch \"master\"]
\tremote = origin
\tmerge = refs/heads/master
"""


def load_auditor():
    spec = importlib.util.spec_from_file_location("barc_gate_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(output: Path, config: Path = CONFIG) -> subprocess.CompletedProcess[str]:
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
    )


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_static_gate_passes_while_all_method_gates_remain_blocked(
    tmp_path: Path,
) -> None:
    output = tmp_path / "barc-gate"
    completed = run_audit(output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))

    assert record["status"] == "passed"
    assert record["method_gate_status"] == "blocked"
    assert record["scope"] == "source-artifact-label-resource-gate-audit-only"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert record["solver_gate_passed"] is False
    assert record["source"]["observed_revision"] == (
        "a7b51a6b1ff969da3a78a71c533b6d79a93966e7"
    )
    assert record["source"]["observed_commit_tree"] == (
        "55ea72e3290ef7d3ec0ebed3554a9a60b83110ad"
    )
    assert record["source"]["tracked_file_count"] == 1477
    assert record["source"]["tracked_bytes_from_closed_worktree_metadata"] == 61938237
    assert record["source"]["retained_byte_exact_file_count"] == 13
    assert record["source"]["metadata_only_tracked_file_count"] == 1464
    assert record["source"]["ignored_pyc_metadata_only_count"] == 2
    assert record["source"]["working_tree_all_files_byte_exact_verified"] is False
    assert "does not claim byte-exact content for 1464" in record["claim_boundary"]
    assert "Only the 13 retained" in record["claim_boundary"]
    assert any("No git status or diff" in item for item in record["limitations"])

    assert record["license_gate"]["status"] == "blocked"
    assert record["license_gate"]["root_license_count"] == 0
    vendored = record["license_gate"]["vendored_license"]
    assert vendored["identifier"] == "Apache-2.0"
    assert vendored["applied_to_repository_root"] is False
    assert len(record["artifact_gate"]["base_models"]) == 4
    assert len(record["artifact_gate"]["lora_adapters"]) == 2
    assert all(
        item["provenance_verified"] is False
        for item in record["artifact_gate"]["base_models"]
        + record["artifact_gate"]["lora_adapters"]
    )
    assert record["safe_offline_model_load_gate"]["status"] == "blocked"
    assert record["safe_offline_model_load_gate"][
        "local_files_only_contract_validated"
    ] is False
    assert record["label_firewall_gate"]["pseudo_evaluation"][
        "pseudo_evaluation_flow_detected"
    ] is True
    assert record["label_firewall_gate"]["transduction_formatter"][
        "transduction_label_materialization_detected"
    ] is True
    assert record["label_firewall_gate"]["published_transduction_evaluation"][
        "published_runner_label_flow_detected"
    ] is True
    assert record["label_firewall_gate"]["published_transduction_reranking"][
        "label_aware_reranking_detected"
    ] is True
    assert record["label_firewall_gate"]["induction_jsonl_and_generated_execution"][
        "induction_generated_exec_label_flow_detected"
    ] is True
    candidate = record["label_firewall_gate"][
        "challenge_only_direct_transduction_candidate"
    ]
    assert candidate["status"] == "design-candidate-not-implemented"
    assert candidate["solver_prediction_validated"] is False
    assert record["dependency_gate"]["conflicting_vllm_paths_detected"] is True
    assert record["dependency_gate"]["reproducible_dependency_lock_detected"] is False
    assert record["resource_gate"]["target_gpu"] == "RTX 3090"
    assert record["resource_gate"]["single_base_planning_size_gib"] == 14.97
    assert record["resource_gate"]["four_base_planning_size_gib"] == 59.88
    assert record["resource_gate"][
        "paper_eight_process_training_supported_on_single_gpu"
    ] is False
    assert len(record["blockers"]) == 8
    assert all(item["status"] == "blocked" for item in record["blockers"])
    assert all(value is True for value in record["validation"].values())
    assert record["controls"]["retained_source_files_read"] == 13
    assert record["controls"]["retained_source_read_attempts"] == 13
    assert record["controls"]["git_local_config_read_attempts"] == 4
    assert record["controls"]["git_local_config_bytes_read"] == 1216
    assert record["controls"]["git_head_read_attempts"] == 4
    assert record["controls"]["git_head_bytes_read"] == 164
    assert record["controls"]["git_subprocesses_started"] == 6
    assert record["controls"][
        "git_subprocess_object_database_reads_possible"
    ] is True
    assert record["controls"][
        "git_subprocess_timeout_and_output_caps_enforced"
    ] is True
    assert record["controls"][
        "git_subprocess_object_database_bytes_measured"
    ] is False
    assert record["controls"]["git_subprocess_worktree_content_requested"] is False
    assert record["controls"][
        "git_subprocess_untrusted_local_config_available"
    ] is False
    assert "arc_or_label_bundle_bytes_read" not in record["controls"]
    assert "restricted_source_bytes_unread" not in record["validation"]
    assert "subprocess byte volume is unmeasured" in record["claim_boundary"]
    assert len(record["read_ledger"]["file_read_attempts"]) == 23
    assert len(record["read_ledger"]["git_subprocesses"]) == 6
    for key in (
        "network_used",
        "gpu_used",
        "upstream_imported",
        "upstream_executed",
        "generated_code_executed",
        "solver_prediction_produced",
        "auditor_process_arc_or_label_worktree_leaf_bytes_read",
        "auditor_process_pickle_worktree_leaf_bytes_read",
        "auditor_process_model_weight_worktree_leaf_bytes_read",
        "auditor_process_pyc_worktree_leaf_bytes_read",
    ):
        assert record["controls"][key] in {False, 0}
    assert "predictions" not in record
    assert "results" not in record

    replay_output = tmp_path / "barc-gate-replay"
    replay = run_audit(replay_output)
    assert replay.returncode == 0, replay.stderr or replay.stdout
    replay_record = json.loads(
        (replay_output / "run.json").read_text(encoding="utf-8")
    )
    assert replay_record["observation_digest_sha256"] == record[
        "observation_digest_sha256"
    ]


@pytest.mark.parametrize("kind", ["empty-dir", "nonempty-dir", "file", "fifo", "symlink"])
def test_output_requires_a_completely_fresh_leaf(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "occupied"
    marker = tmp_path / "marker"
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


def test_output_parent_symlink_is_rejected_without_target_write(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    completed = run_audit(linked / "run")
    assert completed.returncode == 2
    assert not (real / "run").exists()


def test_every_noncanonical_config_is_rejected_before_output(tmp_path: Path) -> None:
    canary = tmp_path / "arc-agi_evaluation_solutions.json"
    canary.write_bytes(b"must-never-open")
    for candidate in (
        canary,
        SOURCE / "data_processing/validation_transduction_prompt.jsonl",
    ):
        output = tmp_path / f"rejected-{candidate.name}"
        completed = run_audit(output, candidate)
        assert completed.returncode == 2
        assert "production config path must equal" in completed.stderr
        assert not output.exists()
    assert canary.read_bytes() == b"must-never-open"


def test_static_failure_publishes_failure_record_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()

    def injected_failure(*args, **kwargs):
        raise ValueError("injected static contract failure")

    monkeypatch.setattr(auditor, "run_static_audit", injected_failure)
    output = tmp_path / "failed"
    exit_code, returned = auditor.execute_audit(CONFIG, output)
    persisted = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert returned == persisted
    assert persisted["status"] == "failed"
    assert persisted["method_gate_status"] == "blocked"
    assert persisted["error"]["message"] == "injected static contract failure"
    assert persisted["solver_prediction_produced"] is False
    assert [path.name for path in output.iterdir()] == ["run.json"]


def test_main_returns_one_for_persisted_static_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auditor = load_auditor()

    def injected_failure(*args, **kwargs):
        raise RuntimeError("injected CLI failure")

    output = tmp_path / "failed-cli"
    monkeypatch.setattr(auditor, "run_static_audit", injected_failure)
    monkeypatch.setattr(
        auditor.sys,
        "argv",
        [
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--output-directory",
            str(output),
        ],
    )
    assert auditor.main() == 1
    captured = capsys.readouterr()
    assert '"status": "failed"' in captured.err
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_config_contract_cannot_redirect_or_weaken_policy() -> None:
    auditor = load_auditor()
    mutations = [
        lambda value: value["source"].__setitem__("repository_path", "/tmp/barc"),
        lambda value: value["source_lock"].__setitem__("path", "answers.jsonl"),
        lambda value: value["config_read_policy"].__setitem__(
            "alternate_paths_allowed", True
        ),
        lambda value: value["source"]["retained_text"][0].__setitem__(
            "path", "answers.jsonl"
        ),
        lambda value: value["source"]["git_metadata_contract"][
            "required_absent_paths"
        ].pop(),
        lambda value: value["source"]["git_metadata_contract"].__setitem__(
            "timeout_seconds", 0
        ),
        lambda value: value["artifacts"]["base_models"].pop(),
        lambda value: value["controls"].__setitem__("network_allowed", True),
        lambda value: value["resource_contract"].__setitem__(
            "target_gpu_vram_gib", True
        ),
    ]
    for mutate in mutations:
        config = read_config()
        mutate(config)
        with pytest.raises(ValueError, match="hardcoded v1 contract"):
            auditor.validate_config(config)


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    auditor = load_auditor()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        auditor.strict_json(b'{"a":1,"a":2}', "test")
    with pytest.raises(ValueError, match="non-finite"):
        auditor.strict_json(b'{"a":NaN}', "test")


def test_reader_roles_reject_sensitive_paths_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    sensitive = tmp_path / "answers.jsonl"
    sensitive.write_bytes(b"must-never-open")
    opened: list[object] = []
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        opened.append(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", tracking_open)
    ledger = auditor.ReadLedger()
    ledger.bind_config(CONFIG)
    with pytest.raises(ValueError, match="role/path binding mismatch"):
        auditor.secure_read_absolute(
            sensitive,
            max_bytes=1024,
            role="canonical_config",
            ledger=ledger,
        )
    assert opened == []
    assert ledger.records == []

    declaration = {
        "path": "safe.py",
        "role": "source_python",
        "bytes": 0,
        "sha256": "0" * 64,
        "blob_oid": "0" * 40,
    }
    ledger = auditor.ReadLedger()
    ledger.bind_source_policy([declaration])
    bad = {**declaration, "path": "answers.jsonl"}
    with pytest.raises(ValueError, match="metadata-only suffix"):
        auditor.secure_read_relative(-1, bad, {}, ledger)
    assert opened == []
    assert ledger.records == []


def test_restricted_source_and_ignored_pyc_leaves_are_never_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    forbidden_suffixes = {
        ".json",
        ".jsonl",
        ".pkl",
        ".pickle",
        ".pyc",
        ".png",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
    }
    forbidden_opens: list[str] = []
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            decoded = os.fsdecode(path)
            if (
                Path(decoded).suffix.lower() in forbidden_suffixes
                and Path(decoded).name
                not in {"barc_gate_v1.json", "source_locks.json"}
            ):
                forbidden_opens.append(decoded)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", tracking_open)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Path.read_text is forbidden in the auditor")
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Path.read_bytes is forbidden in the auditor")
        ),
    )
    record = auditor.run_static_audit(CONFIG, "unit-audit", auditor.ReadLedger())
    assert record["status"] == "passed"
    assert forbidden_opens == []
    assert record["controls"][
        "auditor_process_arc_or_label_worktree_leaf_bytes_read"
    ] == 0
    assert record["controls"][
        "auditor_process_pyc_worktree_leaf_bytes_read"
    ] == 0


def test_caller_tmpdir_cannot_redirect_isolated_git_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted-source"
    restricted.mkdir()
    canary = restricted / "keep.txt"
    canary.write_text("preserve", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(restricted))
    completed = run_audit(tmp_path / "output")
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert [path.name for path in restricted.iterdir()] == ["keep.txt"]
    assert canary.read_text(encoding="utf-8") == "preserve"


def mini_contract(files: dict[str, dict], ignored: list[dict] | None = None) -> dict:
    del files
    return {
        "source": {
            "ignored_metadata_only": ignored or [],
            "retained_text": [],
        }
    }


def tracked_file(path: Path) -> dict[str, object]:
    return {
        "mode": "100644",
        "blob_oid": "0" * 40,
        "bytes": path.stat().st_size,
    }


@pytest.mark.parametrize("unknown", ["extra.py", "answers.jsonl", "model.safetensors"])
def test_closed_world_rejects_unknown_file_without_opening_it(
    tmp_path: Path,
    unknown: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    safe = repository / "safe.py"
    safe.write_text("x = 1\n", encoding="utf-8")
    (repository / unknown).write_bytes(b"must-not-read")
    tracked = {"safe.py": tracked_file(safe)}
    leaf_opens: list[str] = []
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        decoded = os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else ""
        if decoded in {"safe.py", unknown}:
            leaf_opens.append(decoded)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", tracking_open)
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="unknown file"):
            auditor.closed_world_inventory(root_fd, tracked, mini_contract(tracked))
    finally:
        os.close(root_fd)
    assert leaf_opens == []


def test_closed_world_rejects_unknown_directory_special_and_hardlink(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    safe = repository / "safe.py"
    safe.write_text("x = 1\n", encoding="utf-8")
    tracked = {"safe.py": tracked_file(safe)}
    (repository / "unknown-dir").mkdir()
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="unknown directory"):
            auditor.closed_world_inventory(root_fd, tracked, mini_contract(tracked))
    finally:
        os.close(root_fd)
    (repository / "unknown-dir").rmdir()
    os.mkfifo(repository / "special")
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="nonregular"):
            auditor.closed_world_inventory(root_fd, tracked, mini_contract(tracked))
    finally:
        os.close(root_fd)
    (repository / "special").unlink()
    os.link(safe, repository / "alias.py")
    tracked["alias.py"] = tracked_file(repository / "alias.py")
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="hard-linked"):
            auditor.closed_world_inventory(root_fd, tracked, mini_contract(tracked))
    finally:
        os.close(root_fd)


def test_retained_reader_detects_replacement_after_inventory(tmp_path: Path) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    safe = repository / "safe.py"
    payload = b"x = 1\n"
    safe.write_bytes(payload)
    tracked = {"safe.py": tracked_file(safe)}
    declaration = {
        "path": "safe.py",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "blob_oid": "0" * 40,
        "role": "source_python",
    }
    root_fd = auditor.open_absolute_directory(repository)
    try:
        snapshot, _ = auditor.closed_world_inventory(
            root_fd, tracked, mini_contract(tracked)
        )
        replacement = repository / "replacement"
        replacement.write_bytes(b"y = 2\n")
        os.replace(replacement, safe)
        ledger = auditor.ReadLedger()
        ledger.bind_source_policy([declaration])
        with pytest.raises(RuntimeError, match="changed after inventory"):
            auditor.secure_read_relative(root_fd, declaration, snapshot, ledger)
    finally:
        os.close(root_fd)


def test_wrong_sha_failure_ledger_records_all_bytes_read(tmp_path: Path) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    safe = repository / "safe.py"
    payload = b"x = 1\n"
    safe.write_bytes(payload)
    tracked = {"safe.py": tracked_file(safe)}
    declaration = {
        "path": "safe.py",
        "bytes": len(payload),
        "sha256": "0" * 64,
        "blob_oid": "0" * 40,
        "role": "source_python",
    }
    ledger = auditor.ReadLedger()
    ledger.bind_source_policy([declaration])
    root_fd = auditor.open_absolute_directory(repository)
    try:
        snapshot, _ = auditor.closed_world_inventory(
            root_fd, tracked, mini_contract(tracked)
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch") as raised:
            auditor.secure_read_relative(root_fd, declaration, snapshot, ledger)
    finally:
        os.close(root_fd)
    attempt = ledger.records[0]
    assert attempt["bytes"] == len(payload)
    assert attempt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert attempt["read_status"] == "completed"
    assert attempt["validation_status"] == "failed"
    failure = auditor.failure_record("wrong-sha", "static-audit", raised.value, ledger)
    assert failure["controls"]["retained_source_files_read"] == 1
    assert failure["read_ledger"]["file_read_attempts"][0] == attempt


def test_mid_read_race_failure_ledger_records_consumed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    safe = repository / "safe.py"
    payload = b"0123456789abcdef\n"
    safe.write_bytes(payload)
    tracked = {"safe.py": tracked_file(safe)}
    declaration = {
        "path": "safe.py",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "blob_oid": "0" * 40,
        "role": "source_python",
    }
    ledger = auditor.ReadLedger()
    ledger.bind_source_policy([declaration])
    root_fd = auditor.open_absolute_directory(repository)
    mutator_fd = os.open(safe, os.O_WRONLY)
    real_read = auditor.os.read
    real_fstat = auditor.os.fstat
    mutated = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            os.pwrite(mutator_fd, b"X", 0)
            os.fsync(mutator_fd)
        return chunk

    def racing_fstat(descriptor: int):
        observed = real_fstat(descriptor)
        if mutated and descriptor != mutator_fd and auditor.stat.S_ISREG(
            observed.st_mode
        ):
            fields = list(observed)
            fields[9] = observed.st_ctime + 1
            return os.stat_result(fields)
        return observed

    try:
        snapshot, _ = auditor.closed_world_inventory(
            root_fd, tracked, mini_contract(tracked)
        )
        monkeypatch.setattr(auditor.os, "read", racing_read)
        monkeypatch.setattr(auditor.os, "fstat", racing_fstat)
        with pytest.raises(RuntimeError, match="changed while verified bytes") as raised:
            auditor.secure_read_relative(root_fd, declaration, snapshot, ledger)
    finally:
        os.close(mutator_fd)
        os.close(root_fd)
    attempt = ledger.records[0]
    assert mutated is True
    assert attempt["bytes"] == len(payload)
    assert attempt["read_status"] == "failed"
    assert attempt["validation_status"] == "failed"
    failure = auditor.failure_record("read-race", "static-audit", raised.value, ledger)
    assert failure["controls"]["retained_source_files_read"] == 1
    assert failure["read_ledger"]["file_read_attempts"][0]["bytes"] == len(payload)


def test_git_tree_parser_rejects_malformed_and_unsafe_records() -> None:
    auditor = load_auditor()
    with pytest.raises(ValueError, match="NUL-terminated"):
        auditor.parse_git_tree(b"")
    with pytest.raises(ValueError, match="contained POSIX"):
        auditor.parse_git_tree(
            b"100644 blob " + b"0" * 40 + b"\t../answer.json\0"
        )
    with pytest.raises(ValueError, match="unsupported entry"):
        auditor.parse_git_tree(
            b"040000 tree " + b"0" * 40 + b"\tdirectory\0"
        )


def test_git_config_include_solution_and_fifo_are_rejected_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    assert len(SAFE_GIT_CONFIG) == 304
    assert hashlib.sha256(SAFE_GIT_CONFIG).hexdigest() == (
        "b99b39f2e5e40142bc7030ddf55ad6db560d560360a7879a7ab3e04b521d8f6f"
    )
    repository = tmp_path / "repo"
    git_directory = repository / ".git"
    git_directory.mkdir(parents=True)
    solution = tmp_path / "solutions.json"
    solution.write_bytes(b"must-never-open")
    local_config = git_directory / "config"
    include_payload = f'[include]\n\tpath = {solution}\n'.encode("utf-8")
    include_payload += b"#" * (len(SAFE_GIT_CONFIG) - len(include_payload))
    local_config.write_bytes(include_payload)
    declaration = {
        "path": ".git/config",
        "bytes": len(SAFE_GIT_CONFIG),
        "sha256": hashlib.sha256(SAFE_GIT_CONFIG).hexdigest(),
        "mode": "0644",
    }
    git_fd = auditor.open_absolute_directory(git_directory)
    opened_solution = False
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        nonlocal opened_solution
        if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path) == str(
            solution
        ):
            opened_solution = True
        return real_open(path, flags, *args, **kwargs)

    try:
        monkeypatch.setattr(auditor.os, "open", tracking_open)
        ledger = auditor.ReadLedger()
        with pytest.raises(ValueError, match="byte/SHA-256 contract mismatch"):
            auditor.secure_read_git_config(git_fd, declaration, ledger)
        assert ledger.records[0]["bytes"] == local_config.stat().st_size
        assert ledger.git_subprocesses == []
        assert opened_solution is False
    finally:
        os.close(git_fd)

    local_config.unlink()
    os.mkfifo(local_config)
    git_fd = auditor.open_absolute_directory(git_directory)
    started = auditor.time.monotonic()
    try:
        with pytest.raises(ValueError, match="single-link regular file"):
            auditor.secure_read_git_config(
                git_fd, declaration, auditor.ReadLedger()
            )
    finally:
        os.close(git_fd)
    assert auditor.time.monotonic() - started < 1.0
    with pytest.raises(ValueError, match="forbidden or unknown section"):
        auditor._git_config_entries(
            f'[include]\n\tpath = {solution}\n'.encode("utf-8")
        )


@pytest.mark.parametrize(
    "relative_path",
    ["info/attributes", "commondir", "gitdir", "config.worktree"],
)
def test_auxiliary_git_config_paths_are_rejected_without_fifo_open(
    tmp_path: Path, relative_path: str,
) -> None:
    auditor = load_auditor()
    git_directory = tmp_path / ".git"
    target = git_directory / relative_path
    target.parent.mkdir(parents=True)
    os.mkfifo(target)
    git_fd = auditor.open_absolute_directory(git_directory)
    started = auditor.time.monotonic()
    try:
        with pytest.raises(ValueError, match="forbidden auxiliary"):
            auditor.require_git_metadata_path_absent(
                git_fd, f".git/{relative_path}"
            )
    finally:
        os.close(git_fd)
    assert auditor.time.monotonic() - started < 1.0


def test_git_object_store_nested_metadata_inventory_detects_changes(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    objects = tmp_path / "objects"
    (objects / "info").mkdir(parents=True)
    (objects / "pack").mkdir()
    (objects / "pack" / "pack-test.idx").write_bytes(b"index")
    objects_fd = auditor.open_absolute_directory(objects)
    try:
        before = auditor.git_object_store_metadata_inventory(objects_fd)
        (objects / "info" / "alternates").write_text(
            "/tmp/solutions.json\n", encoding="utf-8"
        )
        after = auditor.git_object_store_metadata_inventory(objects_fd)
    finally:
        os.close(objects_fd)
    assert before != after
    assert "info/alternates" not in before
    assert "info/alternates" in after


def test_fixed_git_allowlist_output_cap_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    git_directory = repository / ".git"
    git_directory.mkdir(parents=True)
    root_fd = auditor.open_absolute_directory(repository)
    git_fd = auditor.open_absolute_directory(git_directory)
    command = ("rev-parse", "--verify", auditor.EXPECTED_REVISION)
    try:
        ledger = auditor.ReadLedger()
        with pytest.raises(ValueError, match="outside the BARC metadata allowlist"):
            auditor._git(root_fd, git_fd, (), ledger, "config", "--list")
        assert ledger.git_subprocesses == []

        real_popen = auditor.subprocess.Popen

        def noisy_popen(_argv, **kwargs):
            assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
            assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
            assert kwargs["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
            assert kwargs["env"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
            assert kwargs["env"]["GIT_DIR"] == f"/proc/self/fd/{root_fd}"
            assert kwargs["env"]["GIT_OBJECT_DIRECTORY"] == (
                f"/proc/self/fd/{git_fd}"
            )
            assert "GIT_WORK_TREE" not in kwargs["env"]
            assert "HOME" not in kwargs["env"]
            return real_popen(["/usr/bin/printf", "123456789"], **kwargs)

        monkeypatch.setattr(auditor.subprocess, "Popen", noisy_popen)
        monkeypatch.setattr(
            auditor,
            "GIT_COMMAND_STDOUT_LIMITS",
            {**auditor.GIT_COMMAND_STDOUT_LIMITS, command: 8},
        )
        ledger = auditor.ReadLedger()
        with pytest.raises(RuntimeError, match="stdout limit"):
            auditor._git(root_fd, git_fd, (), ledger, *command)
        assert ledger.git_subprocesses[0]["status"] == "failed"
        assert ledger.git_subprocesses[0]["stdout_bytes"] == 9

        def sleeping_popen(_argv, **kwargs):
            return real_popen(["/usr/bin/sleep", "1"], **kwargs)

        monkeypatch.setattr(auditor.subprocess, "Popen", sleeping_popen)
        monkeypatch.setattr(auditor, "GIT_COMMAND_TIMEOUT_SECONDS", 0.05)
        ledger = auditor.ReadLedger()
        started = auditor.time.monotonic()
        with pytest.raises(TimeoutError, match="timeout") as raised:
            auditor._git(root_fd, git_fd, (), ledger, *command)
        assert auditor.time.monotonic() - started < 1.0
        assert ledger.git_subprocesses[0]["status"] == "failed"
        assert ledger.git_subprocesses[0]["failure_type"] == "TimeoutError"
        failure = auditor.failure_record(
            "git-timeout", "static-audit", raised.value, ledger
        )
        assert failure["controls"]["git_subprocesses_started"] == 1
        assert failure["controls"][
            "git_subprocess_object_database_reads_possible"
        ] is True
        assert failure["controls"][
            "git_subprocess_worktree_content_requested"
        ] is False
        assert failure["read_ledger"]["git_subprocesses"][0][
            "potential_repository_reads"
        ].startswith("auditor-owned isolated Git-dir")
    finally:
        os.close(git_fd)
        os.close(root_fd)


def test_label_flow_analyzers_distinguish_candidate_from_contaminated_runner() -> None:
    auditor = load_auditor()
    candidate = ast.parse(
        """
BASE_MODEL = "barc0/Llama-3.1-ARC-Heavy-Transduction-8B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=False, revision="abc")
llm = LLM(model=BASE_MODEL)
messages = d["messages"]
inputs = tokenizer.apply_chat_template([
    {"role": "system", "content": messages[0]["content"]},
    {"role": "user", "content": messages[1]["content"]},
], add_generation_prompt=True)
outputs = llm.generate(inputs, params)
"""
    )
    observation = auditor.analyze_vllm_runner(candidate)
    assert observation["challenge_only_prompt_pattern_detected"] is True
    assert observation["direct_transduction_structure_detected"] is True
    assert observation["safe_offline_tokenizer_load_detected"] is True
    assert observation["published_runner_label_flow_detected"] is False

    contaminated = ast.parse(
        """
BASE_MODEL = "barc0/Llama-3.1-ARC-Heavy-Transduction-8B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
llm = LLM(model=BASE_MODEL)
messages = d["messages"]
inputs = tokenizer.apply_chat_template([
    {"role": "system", "content": messages[0]["content"]},
    {"role": "user", "content": messages[1]["content"]},
    {"role": "assistant", "content": messages[2]["content"]},
], add_generation_prompt=False)
outputs = llm.generate(inputs, params)
correct = outputs[0] == d["answer"]
"""
    )
    observation = auditor.analyze_vllm_runner(contaminated)
    assert observation["challenge_only_prompt_pattern_detected"] is False
    assert observation["safe_offline_tokenizer_load_detected"] is False
    assert observation["published_runner_label_flow_detected"] is True


def test_uncalled_dead_decoys_cannot_satisfy_label_flow_contracts() -> None:
    auditor = load_auditor()
    runner = ast.parse(
        """
BASE_MODEL = "barc0/Llama-3.1-ARC-Heavy-Transduction-8B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
llm = LLM(model=BASE_MODEL)
messages = d["messages"]
inputs = tokenizer.apply_chat_template([
    {"role": "system", "content": messages[0]["content"]},
    {"role": "user", "content": messages[1]["content"]},
], add_generation_prompt=True)
outputs = llm.generate(inputs, params)

def dead_decoy():
    poisoned = tokenizer.apply_chat_template([
        {"role": "assistant", "content": messages[2]["content"]},
    ])
    bad = llm.generate(poisoned, params)
    return bad[0] == d["answer"]

unused_alias = dead_decoy
"""
    )
    observation = auditor.analyze_vllm_runner(runner)
    assert observation["challenge_only_prompt_pattern_detected"] is True
    assert observation["messages_index_2_access_detected"] is False
    assert observation["answer_access_detected"] is False
    assert observation["published_runner_label_flow_detected"] is False

    after_return = ast.parse(
        """
def main():
    return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template(
        [messages[2]], add_generation_prompt=False
    )
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
"""
    )
    assert not auditor.analyze_vllm_runner(after_return)[
        "published_runner_label_flow_detected"
    ]

    definitely_terminated_decoys = [
        """
def main():
    try:
        return
    finally:
        cleanup = True
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
""",
        """
def main():
    try:
        return
    except Exception:
        return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
""",
        """
def main():
    while True:
        return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
""",
        """
def main():
    while True:
        try:
            break
        finally:
            return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
""",
    ]
    for source in definitely_terminated_decoys:
        assert not auditor.analyze_vllm_runner(ast.parse(source))[
            "published_runner_label_flow_detected"
        ]

    structured_terminal_decoys = [
        """
def main():
    with manager():
        return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
""",
        """
def main():
    match 0:
        case _:
            return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
""",
    ]
    for source in structured_terminal_decoys:
        assert not auditor.analyze_vllm_runner(ast.parse(source))[
            "published_runner_label_flow_detected"
        ]

    suppressible_exception_control = ast.parse(
        """
def main():
    with suppress(Exception):
        raise RuntimeError()
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main()
"""
    )
    assert auditor.analyze_vllm_runner(suppressible_exception_control)[
        "published_runner_label_flow_detected"
    ]

    fallthrough_match_control = ast.parse(
        """
def main(subject):
    match subject:
        case 0:
            return
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]
main(subject)
"""
    )
    assert auditor.analyze_vllm_runner(fallthrough_match_control)[
        "published_runner_label_flow_detected"
    ]

    called_alias = ast.parse(
        """
def contaminated():
    messages = d["messages"]
    prompt = tokenizer.apply_chat_template([messages[2]])
    outputs = llm.generate(prompt, params)
    correct = outputs[0] == d["answer"]

alias = contaminated
alias()
"""
    )
    assert auditor.analyze_vllm_runner(called_alias)[
        "published_runner_label_flow_detected"
    ]

    expression_branch_decoys = [
        "correct = (outputs[0] == d['answer']) if False else False",
        "correct = False and (outputs[0] == d['answer'])",
        "correct = (() and (outputs[0] == d['answer'])) or False",
    ]
    for expression in expression_branch_decoys:
        source = f"""
messages = d["messages"]
prompt = tokenizer.apply_chat_template([messages[2]])
outputs = llm.generate(prompt, params)
{expression}
"""
        assert not auditor.analyze_vllm_runner(ast.parse(source))[
            "published_runner_label_flow_detected"
        ]

    formatter = ast.parse(
        """
def convert_chat_format_transduction(question, answer):
    return {"messages": [{"role": "assistant", "content": answer}]}

def main():
    with open(args.load_file) as source:
        rows = source.readlines()
    answer = problem.test_pairs[0].y
    train_data = []
    train_data.append(convert_chat_format_transduction(question, answer))
    print(train_data[i]["messages"][2]["content"])
    with open("dataset/transduction_formatted_test-time_finetune.jsonl", "w") as output:
        output.write("rows")

unused_main = main
"""
    )
    assert not auditor.analyze_transduction_formatter(formatter)[
        "transduction_label_materialization_detected"
    ]


def test_executable_scope_prunes_only_provably_unreachable_nodes() -> None:
    auditor = load_auditor()

    def calls_in_main(source: str) -> set[str]:
        tree = ast.parse(source)
        main = auditor.top_level_functions(tree)["main"]
        return {
            name
            for node in auditor.executable_scope_nodes(main)
            if isinstance(node, ast.Call)
            if (name := auditor.dotted_name(node.func)) is not None
        }

    reachable_controls = [
        """
def main():
    try:
        return
    except Exception:
        pass
    poison()
""",
        """
def main():
    while True:
        break
    poison()
""",
        """
def main():
    while True:
        try:
            break
        finally:
            pass
    poison()
""",
        """
def main():
    value = clean() if unknown else poison()
""",
        """
def main():
    value = True and poison()
""",
        """
def main():
    value = False or poison()
""",
    ]
    for source in reachable_controls:
        assert "poison" in calls_in_main(source)

    short_circuit_controls = [
        """
def main():
    value = unknown and False and poison()
""",
        """
def main():
    value = unknown or True or poison()
""",
    ]
    for source in short_circuit_controls:
        assert "poison" not in calls_in_main(source)

    empty_container_short_circuits = ["()", "[]", "{}"]
    for literal in empty_container_short_circuits:
        assert "poison" not in calls_in_main(
            f"def main():\n    value = {literal} and poison()\n"
        )
        assert "poison" in calls_in_main(
            f"def main():\n    value = {literal} or poison()\n"
        )

    nonempty_literal_containers = ["(1,)", "[1]", "{1}", "{1: 2}"]
    for literal in nonempty_literal_containers:
        assert "poison" in calls_in_main(
            f"def main():\n    value = {literal} and poison()\n"
        )

    unpack_only_containers = ["(*items,)", "[*items]", "{*items}", "{**mapping}"]
    for literal in unpack_only_containers:
        assert "poison" in calls_in_main(
            f"def main():\n    value = {literal} and poison()\n"
        )
        assert "poison" in calls_in_main(
            f"def main():\n    value = {literal} or poison()\n"
        )

    fixed_nonempty_with_unpack = [
        "(*items, 1)",
        "[*items, 1]",
        "{*items, 1}",
        "{**mapping, 1: 2}",
    ]
    for literal in fixed_nonempty_with_unpack:
        assert "poison" not in calls_in_main(
            f"def main():\n    value = {literal} or poison()\n"
        )

    suppressible_with_exception = """
def main():
    with suppress(Exception):
        raise RuntimeError()
    poison()
"""
    assert "poison" in calls_in_main(suppressible_with_exception)


def test_pseudo_eval_and_formatter_detectors_require_label_flow() -> None:
    auditor = load_auditor()
    pseudo = ast.parse(
        """
all_dataset = []
with open("arc_all_evaluation.json") as f:
    data = json.load(f)
for source_task in data:
    training_data = source_task["data"]["train"]
    for train in training_data:
        new_test_dataset = [train]
        all_dataset.append({"test": new_test_dataset})
with open("dataset/arc_all_evaluation_new_seperate.json", "w") as f:
    for task in all_dataset:
        json.dump(task, f)
"""
    )
    assert auditor.analyze_pseudo_eval(pseudo)["pseudo_evaluation_flow_detected"]
    no_promotion = ast.parse(
        """
with open("arc_all_evaluation.json") as f:
    data = json.load(f)
all_dataset = []
for source_task in data:
    training_data = source_task["data"]["train"]
    for train in training_data:
        new_test_dataset = []
        all_dataset.append({"test": new_test_dataset})
with open("dataset/arc_all_evaluation_new_seperate.json", "w") as f:
    for task in all_dataset:
        json.dump(task, f)
"""
    )
    assert not auditor.analyze_pseudo_eval(no_promotion)[
        "pseudo_evaluation_flow_detected"
    ]
    disconnected_sink = ast.parse(
        """
with open("arc_all_evaluation.json") as f:
    data = json.load(f)
for source_task in data:
    training_data = source_task["train"]
    for train in training_data:
        new_test_dataset = [train]
with open("dataset/arc_all_evaluation_new_seperate.json", "w") as f:
    json.dump(data, f)
"""
    )
    assert not auditor.analyze_pseudo_eval(disconnected_sink)[
        "pseudo_evaluation_flow_detected"
    ]

    formatter = ast.parse(
        """
def convert_chat_format_transduction(question, answer):
    messages = {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}
    return messages

def main():
    with open(args.load_file) as source:
        rows = source.readlines()
    answer = problem.test_pairs[0].y
    train_data = []
    train_data.append(convert_chat_format_transduction(question, answer))
    print(train_data[i]["messages"][2]["content"])
    with open("dataset/transduction_formatted_test-time_finetune.jsonl", "w") as output:
        output.write("\\n".join(json.dumps(data) for data in train_data))

main()
"""
    )
    assert auditor.analyze_transduction_formatter(formatter)[
        "transduction_label_materialization_detected"
    ]
    disconnected_write = ast.parse(ast.unparse(formatter))
    for node in ast.walk(disconnected_write):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
        ):
            node.args = [ast.Constant("rows")]
    assert not auditor.analyze_transduction_formatter(disconnected_write)[
        "transduction_label_materialization_detected"
    ]


def test_output_mkdir_open_swap_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output = tmp_path / "run"
    real_open = auditor.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and path == "run"
            and kwargs.get("dir_fd") is not None
            and flags & auditor.os.O_DIRECTORY
        ):
            parent_fd = kwargs["dir_fd"]
            os.rename("run", "original", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.mkdir("run", dir_fd=parent_fd)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", racing_open)
    with pytest.raises(auditor.OutputPathError, match="raced before pinning"):
        auditor.create_fresh_output(output)
    assert swapped is True
    assert list(output.iterdir()) == []


def test_output_parent_directory_sync_failure_cannot_return_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output = tmp_path / "run"
    calls = 0

    def failing_parent_sync(descriptor):
        nonlocal calls
        calls += 1
        raise OSError("injected parent directory sync failure")

    monkeypatch.setattr(auditor.os, "fsync", failing_parent_sync)
    with pytest.raises(OSError, match="parent directory sync failure"):
        auditor.create_fresh_output(output)
    assert calls == 1
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_output_publish_swap_never_writes_attacker_directory(tmp_path: Path) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    try:
        original = tmp_path / "original"
        output_path.rename(original)
        output_path.mkdir()
        with pytest.raises(auditor.OutputPathError, match="replaced after creation"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
        assert list(output_path.iterdir()) == []
        assert list(original.iterdir()) == []
    finally:
        output.close()


def test_output_parent_rename_replacement_is_detected_before_publish(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    requested_parent = tmp_path / "requested-parent"
    requested_parent.mkdir()
    output_path = requested_parent / "run"
    output = auditor.create_fresh_output(output_path)
    original_parent = tmp_path / "original-parent"
    requested_parent.rename(original_parent)
    requested_parent.mkdir()
    try:
        with pytest.raises(auditor.OutputPathError, match="parent path was replaced"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert list(requested_parent.iterdir()) == []
    assert list((original_parent / "run").iterdir()) == []


def test_temporary_name_swap_cannot_replace_report_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_rename = auditor.rename_noreplace
    swapped = False

    def swapping_rename(directory_fd, source, destination):
        nonlocal swapped
        if not swapped and isinstance(source, str) and source.startswith(".run.json."):
            os.unlink(source, dir_fd=directory_fd)
            attacker = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(attacker, b'{"status":"attacker"}\n')
            finally:
                os.close(attacker)
            swapped = True
        return real_rename(directory_fd, source, destination)

    monkeypatch.setattr(auditor, "rename_noreplace", swapping_rename)
    try:
        with pytest.raises(auditor.OutputPathError, match="published report identity"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert swapped is True
    assert (output_path / "run.json").read_bytes() == b'{"status":"attacker"}\n'
    assert list(output_path.glob(".run.json.*.tmp")) == []


def test_temporary_hardlink_injection_cannot_publish_mutable_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_rename = auditor.rename_noreplace
    injected = False

    def linking_rename(directory_fd, source, destination):
        nonlocal injected
        os.link(
            source,
            ".retained-link",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        injected = True
        return real_rename(directory_fd, source, destination)

    monkeypatch.setattr(auditor, "rename_noreplace", linking_rename)
    try:
        with pytest.raises(auditor.OutputPathError, match="identity or payload"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert injected is True
    assert (output_path / "run.json").stat().st_nlink == 2
    assert (output_path / ".retained-link").stat().st_ino == (
        output_path / "run.json"
    ).stat().st_ino


def test_successful_publication_never_calls_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    def forbidden_unlink(*args, **kwargs):
        del args, kwargs
        raise AssertionError("publication must not unlink any path")

    monkeypatch.setattr(auditor.os, "unlink", forbidden_unlink)
    try:
        auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert json.loads((output_path / "run.json").read_text()) == {
        "status": "passed"
    }
    assert list(output_path.glob(".run.json.*.tmp")) == []


def test_committed_report_ignores_temporary_descriptor_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_close = auditor.os.close
    injected = False

    def failing_close(descriptor):
        nonlocal injected
        if not injected and os.fstat(descriptor).st_mode & 0o170000 == 0o100000:
            injected = True
            real_close(descriptor)
            raise OSError("injected descriptor close failure")
        return real_close(descriptor)

    monkeypatch.setattr(auditor.os, "close", failing_close)
    auditor.write_json_no_clobber(output, {"status": "passed"})
    output.close(record_committed=True)
    assert injected is True
    assert json.loads((output_path / "run.json").read_text()) == {
        "status": "passed"
    }


def test_execute_close_failure_cannot_override_committed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    real_write = auditor.write_json_no_clobber
    real_close = auditor.os.close
    publication_finished = False
    injected = False

    def tracking_write(output, value):
        nonlocal publication_finished
        real_write(output, value)
        publication_finished = True

    def failing_close(descriptor):
        nonlocal injected
        if publication_finished and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected output close failure")
        return real_close(descriptor)

    monkeypatch.setattr(auditor, "write_json_no_clobber", tracking_write)
    monkeypatch.setattr(auditor.os, "close", failing_close)
    exit_code, returned = auditor.execute_audit(CONFIG, output_path)
    assert injected is True
    assert exit_code == 0
    assert returned["status"] == "passed"
    assert json.loads((output_path / "run.json").read_text()) == returned


def test_run_json_creation_race_is_no_clobber_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_rename = auditor.rename_noreplace
    injected = False

    def racing_rename(directory_fd, source, destination):
        nonlocal injected
        if destination == "run.json" and not injected:
            injected = True
            attacker = os.open(
                "run.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(attacker, b'{"status":"attacker"}\n')
            finally:
                os.close(attacker)
        return real_rename(directory_fd, source, destination)

    monkeypatch.setattr(auditor, "rename_noreplace", racing_rename)
    try:
        with pytest.raises(FileExistsError):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert injected is True
    assert (output_path / "run.json").read_bytes() == b'{"status":"attacker"}\n'
    temporary = list(output_path.glob(".run.json.*.tmp"))
    assert len(temporary) == 1
    assert json.loads(temporary[0].read_text()) == {"status": "passed"}


def test_published_bytes_are_exact_serialization(tmp_path: Path) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    value = {"z": [1, 2], "a": {"passed": True}, "unicode": "边界"}
    try:
        auditor.write_json_no_clobber(output, value)
    finally:
        output.close()
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    assert (output_path / "run.json").read_bytes() == expected


def test_precommit_leaf_swap_and_rollback_failure_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    original = tmp_path / "original"
    real_fsync = auditor.os.fsync
    real_unlink = auditor.os.unlink
    calls = 0

    def swapping_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            output_path.rename(original)
            output_path.mkdir()
            raise OSError("injected precommit sync failure")
        return real_fsync(descriptor)

    def refusing_rollback(path, *args, **kwargs):
        if path == "run.json" and kwargs.get("dir_fd") == output.descriptor:
            raise OSError("injected rollback unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "fsync", swapping_fsync)
    monkeypatch.setattr(auditor.os, "unlink", refusing_rollback)
    try:
        with pytest.raises(OSError, match="precommit sync failure"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert list(output_path.iterdir()) == []
    assert (original / "run.json").is_file()


def test_directory_sync_failure_preserves_racer_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_fsync = auditor.os.fsync
    calls = 0
    def replace_with_attacker(payload: bytes) -> None:
        os.unlink("run.json", dir_fd=output.descriptor)
        descriptor = os.open(
            "run.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=output.descriptor,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    def racing_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            replace_with_attacker(b"attacker-run\n")
            raise OSError("injected directory sync race")
        return real_fsync(descriptor)

    monkeypatch.setattr(auditor.os, "fsync", racing_fsync)
    try:
        with pytest.raises(OSError, match="directory sync race"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()

    assert (output_path / "run.json").read_bytes() == b"attacker-run\n"
    assert list(output_path.glob(".run.json.*.tmp")) == []


def test_directory_commit_sync_failure_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_fsync = auditor.os.fsync
    calls = 0

    def failing_commit_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory commit sync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(auditor.os, "fsync", failing_commit_fsync)
    try:
        with pytest.raises(OSError, match="directory commit sync failure"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()
    assert calls == 2
    assert json.loads((output_path / "run.json").read_text()) == {
        "status": "passed"
    }
    assert [path.name for path in output_path.iterdir()] == ["run.json"]


def test_rerank_and_generated_exec_detectors_fail_closed_on_mutation() -> None:
    auditor = load_auditor()
    rerank_fixture = """
def build_transformed_prompt(messages, tokenizer, examples, test, order):
    assistant_msg = messages[2]["content"]
    prompt = tokenizer.apply_chat_template([
        {"role": "assistant", "content": assistant_msg}
    ])
    return prompt

def generate_candidates(llm, prompt):
    outputs = llm.generate(prompt, params)
    responses = []
    scores = []
    for output in outputs:
        responses.append(output.text)
        scores.append(output.score)
    return responses, scores, None

def frequency_ranking(candidates_per_perm, scores_per_perm):
    candidate_stats = {}
    for candidates, scores in zip(candidates_per_perm, scores_per_perm):
        candidate_stats.update({candidate: scores for candidate in candidates})
    return candidate_stats

with open("validation_data.jsonl") as f:
    data = f.readlines()
messages = d["messages"]
transformed_prompt = build_transformed_prompt(messages, tokenizer, examples, test, order)
candidates, scores, _ = generate_candidates(llm, transformed_prompt)
candidates_per_perm = [candidates]
scores_per_perm = [scores]
original_candidates = candidates_per_perm[0]
frequency_ranked_candidates = frequency_ranking(candidates_per_perm, scores_per_perm)
answer = d["answer"].strip()
is_correct_original = any(cand == answer for cand in original_candidates)
is_correct_frequency = any(cand == answer for cand in frequency_ranked_candidates)
response = {
    "is_correct_original": is_correct_original,
    "is_correct_frequency": is_correct_frequency,
}
"""
    assert auditor.analyze_reranking(ast.parse(rerank_fixture))[
        "label_aware_reranking_detected"
    ]
    assert not auditor.analyze_reranking(
        ast.parse(rerank_fixture.replace('messages[2]["content"]', 'messages[1]["content"]'))
    )["label_aware_reranking_detected"]
    assert not auditor.analyze_reranking(
        ast.parse(rerank_fixture.replace("return prompt", 'return "SAFE CONSTANT"'))
    )["label_aware_reranking_detected"]

    eval_code = ast.parse(
        """
def main():
    parser.add_argument("--answer_file", help="Path to the answer file")
    with open(answer_file) as f:
        problem_answers = [json.loads(line) for line in f]
    codes = []
    for problem in problem_answers:
        for response in problem["responses"]:
            parsed_codes = parse_code(response)
            codes.append(parsed_codes[0])
    multi_validate(problem, codes)
def multi_validate(problem, codes):
    return multi_execute_transformation(codes, inputs)
main()
"""
    )
    execution = ast.parse(
        """
def _worker_with_id(args):
    task_id, source_code = args
    exec(source_code)
def multi_process_execute(codes):
    tasks = [(index, code) for index, code in enumerate(codes)]
    pool = ProcessPool()
    pool.map(_worker_with_id, tasks)
def multi_execute_transformation(sources, inputs):
    codes = [source for source in sources]
    return multi_process_execute(codes)
"""
    )
    evaluation = ast.parse(
        """
def main():
    folder = "induction_samples_with_execution_results/ARC-Potpourri/"
    all_data = []
    all_data.append(orjsonl.load(path=folder))
    data = all_data
    for d in data:
        for train_verdict, output_grids in zip(
            d["train_verdicts"], d["output_grids"]
        ):
            if train_verdict:
                test_outputs = output_grids
    for test_pair in test_pairs:
        ground_truth = test_pair.y
        matched = any(output == ground_truth for output in test_outputs)
main()
"""
    )
    assert auditor.analyze_generated_code_execution(
        eval_code, execution, evaluation
    )["induction_generated_exec_label_flow_detected"]
    no_exec = ast.parse(
        """
def _worker_with_id(args):
    return args
def multi_process_execute(codes):
    tasks = [(index, code) for index, code in enumerate(codes)]
    pool = ProcessPool()
    pool.map(_worker_with_id, tasks)
def multi_execute_transformation(sources, inputs):
    codes = [source for source in sources]
    return multi_process_execute(codes)
"""
    )
    assert not auditor.analyze_generated_code_execution(
        eval_code, no_exec, evaluation
    )["induction_generated_exec_label_flow_detected"]
    disconnected_eval = ast.parse(
        """
def main():
    parser.add_argument("--answer_file", help="Path to the answer file")
    with open(answer_file) as f:
        rows = [json.loads(line) for line in f]
    parse_code("constant")
    multi_execute_transformation(["constant"], [])
main()
"""
    )
    disconnected_evaluation = ast.parse(
        """
def main():
    folder = "induction_samples_with_execution_results/ARC-Potpourri/"
    field = "train_verdicts"
    ground_truth = test_pair.y
main()
"""
    )
    assert not auditor.analyze_generated_code_execution(
        disconnected_eval, execution, disconnected_evaluation
    )["induction_generated_exec_label_flow_detected"]


def test_dependency_detector_requires_pins_and_one_compatible_vllm_path() -> None:
    auditor = load_auditor()
    actual = auditor.analyze_dependencies(
        "git+https://github.com/xu3kev/arc-py.git\norjsonl\nfunc-timeout\n",
        "Induction vllm==0.6.0; Transduction vllm==0.5.4; --num_processes=8",
    )
    assert actual["conflicting_vllm_paths_detected"] is True
    assert actual["requirements_fully_pinned"] is False
    assert actual["git_dependency_revision_pinned"] is False
    assert actual["reproducible_dependency_lock_detected"] is False
    pinned = auditor.analyze_dependencies(
        "orjsonl==1.0\nfunc-timeout==4.3.5\n",
        "Induction and Transduction vllm==0.6.0; --num_processes=1",
    )
    assert pinned["requirements_fully_pinned"] is True
    assert pinned["conflicting_vllm_paths_detected"] is False


def test_terminal_source_path_identity_detects_replacement(tmp_path: Path) -> None:
    auditor = load_auditor()
    source = tmp_path / "source"
    source.mkdir()
    pinned = auditor.open_absolute_directory(source)
    try:
        source.rename(tmp_path / "original")
        source.mkdir()
        with pytest.raises(RuntimeError, match="directory path identity changed"):
            auditor.verify_directory_path_identity(source, pinned)
    finally:
        os.close(pinned)


def test_terminal_git_observation_must_equal_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    real_verify = auditor.verify_git_contract
    calls = 0

    def changing_verify(root_fd, config, ledger):
        nonlocal calls
        calls += 1
        observation, tracked, metadata = real_verify(root_fd, config, ledger)
        if calls == 2:
            observation = {**observation, "observed_commit_tree": "0" * 40}
        return observation, tracked, metadata

    monkeypatch.setattr(auditor, "verify_git_contract", changing_verify)
    with pytest.raises(RuntimeError, match="Git revision/tree/listing/config"):
        auditor.run_static_audit(CONFIG, "git-race", auditor.ReadLedger())
    assert calls == 2


def test_terminal_git_config_signature_must_equal_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    real_verify = auditor.verify_git_contract
    calls = 0

    def changing_verify(root_fd, config, ledger):
        nonlocal calls
        calls += 1
        observation, tracked, metadata = real_verify(root_fd, config, ledger)
        if calls == 2:
            metadata = {**metadata, "local_config_signature": (0,) * 7}
        return observation, tracked, metadata

    monkeypatch.setattr(auditor, "verify_git_contract", changing_verify)
    with pytest.raises(RuntimeError, match="Git revision/tree/listing/config"):
        auditor.run_static_audit(CONFIG, "git-config-race", auditor.ReadLedger())
    assert calls == 2


def test_terminal_inventory_must_equal_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    real_inventory = auditor.closed_world_inventory
    calls = 0

    def changing_inventory(root_fd, tracked, config):
        nonlocal calls
        calls += 1
        internal, public = real_inventory(root_fd, tracked, config)
        if calls == 2:
            internal = {**internal}
            first = next(path for path in internal if internal[path]["kind"] == "tracked")
            internal[first] = {**internal[first], "signature": (0,) * 7}
        return internal, public

    monkeypatch.setattr(auditor, "closed_world_inventory", changing_inventory)
    with pytest.raises(RuntimeError, match="worktree metadata changed"):
        auditor.run_static_audit(CONFIG, "inventory-race", auditor.ReadLedger())
    assert calls == 2


def test_second_git_hook_leaf_replacement_is_caught_by_terminal_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    metadata_only = repository / "labels.jsonl"
    metadata_only.write_bytes(b"original-label-metadata")
    tracked = {"labels.jsonl": tracked_file(metadata_only)}
    config = mini_contract(tracked)
    root_fd = auditor.open_absolute_directory(repository)
    try:
        initial_inventory, initial_public = auditor.closed_world_inventory(
            root_fd, tracked, config
        )
        initial_git = {
            "observed_revision": "a" * 40,
            "observed_commit_tree": "b" * 40,
            "git_tree_listing_sha256": "c" * 64,
        }
        initial_git_metadata = {
            "git_directory_signature": (1,) * 7,
            "local_config_signature": (2,) * 7,
        }
        ledger = auditor.ReadLedger()

        def mutating_final_git(observed_root_fd, observed_config, observed_ledger):
            del observed_root_fd, observed_config, observed_ledger
            replacement = repository / "replacement"
            replacement.write_bytes(b"changed--label-metadata")
            assert replacement.stat().st_size == metadata_only.stat().st_size
            os.replace(replacement, metadata_only)
            return initial_git, tracked, initial_git_metadata

        monkeypatch.setattr(auditor, "verify_git_contract", mutating_final_git)
        with pytest.raises(RuntimeError, match="worktree metadata changed"):
            auditor.verify_terminal_state(
                root_fd,
                repository,
                tracked,
                config,
                initial_git,
                initial_git_metadata,
                initial_inventory,
                initial_public,
                ledger,
            )
    finally:
        os.close(root_fd)


def test_auditor_has_no_upstream_dynamic_network_gpu_or_git_clean_hooks() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported: set[str] = set()
    forbidden_calls: list[str] = []
    forbidden_git_subcommands: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "eval",
                "exec",
                "compile",
                "__import__",
            }:
                forbidden_calls.append(node.func.id)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "resolve",
                "read_text",
                "read_bytes",
                "glob",
                "rglob",
                "chdir",
            }:
                forbidden_calls.append(node.func.attr)
            if isinstance(node.func, ast.Name) and node.func.id == "_git":
                for argument in node.args[1:]:
                    if isinstance(argument, ast.Constant) and argument.value in {
                        "status",
                        "diff",
                        "diff-files",
                    }:
                        forbidden_git_subcommands.append(argument.value)
    assert not imported & {
        "torch",
        "transformers",
        "vllm",
        "datasets",
        "requests",
        "socket",
        "urllib",
        "wandb",
        "arc",
    }
    assert forbidden_calls == []
    assert forbidden_git_subcommands == []
    source = SCRIPT.read_text(encoding="utf-8")
    assert "shell=False" in source
    assert "nvidia-smi" not in source
    assert "curl" not in source
    assert "wget" not in source
