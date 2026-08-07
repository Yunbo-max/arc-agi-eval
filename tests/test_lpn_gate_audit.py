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
SCRIPT = ROOT / "scripts" / "audit_lpn_gates.py"
CONFIG = ROOT / "configs" / "lpn_gate_v1.json"


def load_auditor():
    spec = importlib.util.spec_from_file_location("lpn_gate_auditor", SCRIPT)
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
    }
    arguments = [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--output-directory",
            str(output),
        ]
    return subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_static_gate_passes_but_method_stays_blocked(tmp_path: Path) -> None:
    output = tmp_path / "lpn-gate"
    completed = run_audit(output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))

    assert record["status"] == "passed"
    assert record["scope"] == "source-artifact-data-label-gate-audit-only"
    assert record["method_gate_status"] == "blocked"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert record["solver_gate_passed"] is False
    assert record["fairness"] == {
        "evidence_scope": "blocker_audit",
        "performance_table_eligible": False,
        "promotion_effect": "none",
    }
    assert record["source"]["observed_revision"] == (
        "0adfe56b86d2cba5ae5794edb02da6399a96d98a"
    )
    assert record["source"]["observed_commit_tree"] == (
        "5793cc33c7a1166b5d9e0e61b5774c0ebe534c58"
    )
    assert record["source"]["tracked_file_count"] == 79
    assert record["source"]["python_file_count"] == 21
    assert record["source"]["python_bytes"] == 785886
    assert record["source"]["ignored_pyc_count"] == 14
    assert record["source"]["ignored_pyc_bytes_read"] is False
    assert record["ast_gate"]["status"] == "passed"
    assert record["ast_gate"]["evaluator_challenge_only_candidate"][
        "challenge_only_candidate_detected"
    ] is True
    assert record["ast_gate"]["official_train_same_process_label_flow"][
        "same_function_and_process_label_flow"
    ] is True
    assert record["ast_gate"]["official_evaluate_checkpoint_flow"][
        "official_network_and_paired_label_flow_detected"
    ] is True
    assert record["checkpoint_gate"]["present_count"] == 0
    assert "excluding opaque .git" in record["checkpoint_gate"]["presence_scope"]
    assert "no external artifact/cache roots inspected" in record[
        "checkpoint_gate"
    ]["presence_scope"]
    assert record["license_gate"]["code_status"] == "passed"
    assert record["license_gate"]["code_identifier"] == "Apache-2.0"
    assert record["license_gate"]["artifact_status"] == "blocked-unverified"
    assert len(record["artifact_gate"]["source_declared_wandb_artifact_ids"]) == 7
    assert record["artifact_gate"]["identifier_status"] == (
        "source_declared_unverified"
    )
    assert all(value is True for value in record["validation"].values())
    assert record["controls"]["retained_python_files_read"] == 21
    for key, value in record["controls"].items():
        if key != "retained_python_files_read":
            assert value is False
    assert "results" not in record
    assert "predictions" not in record

    metadata = record["source"]["metadata_only_inventory"]
    assert sum(item["path"].endswith((".yaml", ".yml")) for item in metadata) == 29
    assert sum(item["path"].endswith(".json") for item in metadata) == 6
    assert sum(item["path"].endswith(".ipynb") for item in metadata) == 5
    assert sum(item["path"].endswith(".pyc") for item in metadata) == 14
    assert all(item["sha256"] is None for item in metadata)
    assert all(item["bytes_read"] is False for item in metadata)
    assert all(item["manifest_included"] is False for item in metadata)

    replay_output = tmp_path / "lpn-gate-replay"
    replay = run_audit(replay_output)
    assert replay.returncode == 0, replay.stderr or replay.stdout
    replay_record = json.loads(
        (replay_output / "run.json").read_text(encoding="utf-8")
    )
    assert replay_record["observation_digest_sha256"] == record[
        "observation_digest_sha256"
    ]


@pytest.mark.parametrize("kind", ["empty-dir", "nonempty-dir", "file", "fifo", "symlink"])
def test_output_must_be_a_completely_fresh_leaf(
    tmp_path: Path, kind: str
) -> None:
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
    elif kind == "file":
        assert output.read_text(encoding="utf-8") == "keep"
    elif kind == "symlink":
        assert output.is_symlink()


def test_output_parent_symlink_is_rejected_without_writing_target(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    completed = run_audit(linked_parent / "run")
    assert completed.returncode == 2
    assert not (real_parent / "run").exists()


def test_config_failure_writes_one_conservative_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output = tmp_path / "failed-run"

    def injected_failure(*args, **kwargs):
        raise ValueError("injected static-audit failure")

    monkeypatch.setattr(auditor, "run_static_audit", injected_failure)
    exit_code, returned = auditor.execute_audit(CONFIG, output)
    assert exit_code == 1
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert returned == record
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert record["solver_gate_passed"] is False
    assert record["error"]["type"] == "ValueError"
    assert record["error"]["message"] == "injected static-audit failure"
    assert record["controls"]["retained_python_files_read"] == 0
    assert "results" not in record
    assert "evidence_manifest" not in record
    assert [path.name for path in output.iterdir()] == ["run.json"]


def test_production_cli_rejects_every_noncanonical_config_before_output(
    tmp_path: Path,
) -> None:
    temporary_solution = tmp_path / "arc-agi_evaluation_solutions.json"
    temporary_solution.write_bytes(b"canary-solution-bytes")
    for config in (
        temporary_solution,
        ROOT
        / "external/LPN/src/datasets/json/arc-agi_evaluation_solutions.json",
    ):
        output = tmp_path / f"rejected-{len(list(tmp_path.iterdir()))}"
        completed = run_audit(output, config)
        assert completed.returncode == 2
        assert "production config path must equal" in completed.stderr or (
            "may not be inside the LPN source tree" in completed.stderr
        )
        assert not output.exists()
    assert temporary_solution.read_bytes() == b"canary-solution-bytes"


def test_noncanonical_config_is_rejected_before_reader_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    sensitive = tmp_path / "arc-agi_evaluation_solutions.json"
    sensitive.write_bytes(b"must-never-open")
    output = tmp_path / "output"
    calls: list[Path] = []

    def forbidden_reader(path, **kwargs):
        calls.append(path)
        raise AssertionError("reader must not be reached")

    monkeypatch.setattr(auditor, "secure_read_absolute", forbidden_reader)
    with pytest.raises(ValueError, match="production config path must equal"):
        auditor.validate_config_location(sensitive, output)
    assert calls == []
    assert not output.exists()


def test_reader_roles_cannot_spoof_sensitive_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    sensitive = tmp_path / "arc-agi_evaluation_solutions.json"
    sensitive.write_bytes(b"must-never-open")
    opened: list[object] = []
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        opened.append(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", tracking_open)
    ledger = auditor.ReadLedger()
    with pytest.raises(ValueError, match="canonical config path"):
        ledger.bind_config(sensitive)
    ledger.bind_config(CONFIG)
    with pytest.raises(ValueError, match="role/path binding mismatch"):
        auditor.secure_read_absolute(
            sensitive,
            max_bytes=1024,
            role="canonical_config",
            ledger=ledger,
        )
    assert opened == []

    ledger = auditor.ReadLedger()
    ledger.bind_source_policy({"safe.py"})
    with pytest.raises(ValueError, match="metadata-only suffix"):
        auditor.secure_read_relative(
            -1,
            "arc-agi_evaluation_solutions.json",
            expected_bytes=0,
            expected_sha256="0" * 64,
            snapshot={},
            role="retained_python",
            ledger=ledger,
        )
    assert opened == []


@pytest.mark.parametrize(
    "injected",
    [
        'open("labels.json")',
        "self.evaluate_generations({}, {})",
        "wandb.init()",
        "solutions = {}",
    ],
)
def test_evaluator_forbidden_flow_is_rejected_outside_test_loop(
    injected: str,
) -> None:
    auditor = load_auditor()
    fixture = f"""
class Evaluator:
    def json_submission(self, challenges, params, only_n_tasks, overfit_task, progress_bar, key, train):
        results = {{}}
        {injected}
        for task_id, task in challenges.items():
            for example in task["train"]:
                train_input = example["input"]
                train_output = example["output"]
            for example in task["test"]:
                test_input = example["input"]
                attempts = {{"attempt_1": [], "attempt_2": []}}
        return results
"""
    observation = auditor.analyze_evaluator(ast.parse(fixture))
    assert observation["challenge_only_candidate_detected"] is False
    assert observation["function_forbidden_calls"] or observation[
        "forbidden_reference_lines"
    ]


def test_output_mkdir_open_rename_swap_is_detected(
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
            os.rename(
                "run",
                "original",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir("run", dir_fd=parent_fd)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", racing_open)
    with pytest.raises(auditor.OutputPathError, match="raced before pinning"):
        auditor.create_fresh_output(output)
    assert swapped is True
    assert list(output.iterdir()) == []
    assert (tmp_path / "original").is_dir()


def test_output_publish_rename_swap_never_writes_attacker_directory(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    try:
        original = tmp_path / "original"
        output_path.rename(original)
        output_path.mkdir()
        with pytest.raises(
            auditor.OutputPathError, match="replaced after creation"
        ):
            auditor.write_json_no_clobber(output, {"status": "passed"})
        assert list(output_path.iterdir()) == []
        assert list(original.iterdir()) == []
    finally:
        output.close()


def test_temporary_name_swap_before_link_cannot_replace_report_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_link = auditor.os.link
    swapped = False

    def swapping_link(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and isinstance(source, str) and source.startswith(".run.json."):
            source_fd = kwargs["src_dir_fd"]
            os.unlink(source, dir_fd=source_fd)
            attacker = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_fd,
            )
            try:
                os.write(attacker, b'{"status":"attacker"}\n')
            finally:
                os.close(attacker)
            swapped = True
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "link", swapping_link)
    try:
        with pytest.raises(auditor.OutputPathError, match="linked report identity"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()

    assert swapped is True
    assert list(output_path.iterdir()) == []


def test_leaf_swap_with_rollback_unlink_failure_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    original_path = tmp_path / "original"
    real_fsync = auditor.os.fsync
    real_unlink = auditor.os.unlink
    fsync_calls = 0

    def swapping_fsync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            output_path.rename(original_path)
            output_path.mkdir()
            raise OSError("injected precommit directory sync failure")
        return real_fsync(descriptor)

    def refusing_run_rollback(path, *args, **kwargs):
        if path == "run.json" and kwargs.get("dir_fd") == output.descriptor:
            raise OSError("injected rollback unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "fsync", swapping_fsync)
    monkeypatch.setattr(auditor.os, "unlink", refusing_run_rollback)
    try:
        with pytest.raises(OSError, match="precommit directory sync failure"):
            auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()

    assert fsync_calls >= 2
    assert list(output_path.iterdir()) == []
    assert (original_path / "run.json").is_file()


def test_post_commit_temporary_cleanup_failure_keeps_result_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_unlink = auditor.os.unlink

    def failing_temporary_unlink(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".run.json."):
            raise OSError("injected temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "unlink", failing_temporary_unlink)
    try:
        auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()

    assert json.loads((output_path / "run.json").read_text(encoding="utf-8")) == {
        "status": "passed"
    }
    assert len(list(output_path.glob(".run.json.*.tmp"))) == 1


def test_post_commit_cleanup_sync_failure_keeps_result_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    output_path = tmp_path / "run"
    output = auditor.create_fresh_output(output_path)
    real_fsync = auditor.os.fsync
    calls = 0

    def failing_cleanup_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected cleanup sync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(auditor.os, "fsync", failing_cleanup_fsync)
    try:
        auditor.write_json_no_clobber(output, {"status": "passed"})
    finally:
        output.close()

    assert calls == 3
    assert json.loads((output_path / "run.json").read_text(encoding="utf-8")) == {
        "status": "passed"
    }
    assert [path.name for path in output_path.iterdir()] == ["run.json"]


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
        if not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected temporary descriptor close failure")
        return real_close(descriptor)

    monkeypatch.setattr(auditor.os, "close", failing_close)
    auditor.write_json_no_clobber(output, {"status": "passed"})
    output.close(record_committed=True)

    assert injected is True
    assert json.loads((output_path / "run.json").read_text(encoding="utf-8")) == {
        "status": "passed"
    }


def test_execute_audit_output_close_failure_cannot_override_committed_record(
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
            raise OSError("injected output descriptor close failure")
        return real_close(descriptor)

    monkeypatch.setattr(auditor, "write_json_no_clobber", tracking_write)
    monkeypatch.setattr(auditor.os, "close", failing_close)
    exit_code, returned = auditor.execute_audit(CONFIG, output_path)

    assert injected is True
    assert exit_code == 0
    assert returned["status"] == "passed"
    assert json.loads((output_path / "run.json").read_text(encoding="utf-8")) == returned


def test_terminal_source_path_identity_detects_root_replacement(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    source = tmp_path / "source"
    source.mkdir()
    pinned = auditor.open_absolute_directory(source)
    try:
        source.rename(tmp_path / "original-source")
        source.mkdir()
        with pytest.raises(RuntimeError, match="directory path identity changed"):
            auditor.verify_directory_path_identity(source, pinned)
    finally:
        os.close(pinned)


def test_terminal_git_observation_must_equal_initial_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    real_verify = auditor.verify_git_contract
    calls = 0

    def changing_verify(root_fd, config):
        nonlocal calls
        calls += 1
        observation = real_verify(root_fd, config)
        if calls == 2:
            observation = {**observation, "observed_commit_tree": "0" * 40}
        return observation

    monkeypatch.setattr(auditor, "verify_git_contract", changing_verify)
    with pytest.raises(RuntimeError, match="Git revision/tree metadata changed"):
        auditor.run_static_audit(CONFIG, "git-race", auditor.ReadLedger())
    assert calls == 2


def test_config_paths_and_allowlists_cannot_redirect_reads() -> None:
    auditor = load_auditor()
    mutations = [
        ("source.repository_path", lambda c: c["source"].__setitem__("repository_path", "/etc")),
        ("source_lock.path", lambda c: c["source_lock"].__setitem__("path", "external/LPN/src/datasets/json/arc-agi_evaluation_solutions.json")),
        ("license.path", lambda c: c["license"].__setitem__("path", "src/datasets/json/arc-agi_evaluation_solutions.json")),
        ("ast.path", lambda c: c["ast_contract"].__setitem__("evaluator_path", "src/datasets/json/arc-agi_evaluation_solutions.json")),
        ("python.path", lambda c: c["source"]["retained_python"][0].__setitem__("path", "src/datasets/json/arc-agi_evaluation_solutions.json")),
        ("tracked.reorder", lambda c: c["source"]["tracked_files"].reverse()),
    ]
    for name, mutate in mutations:
        config = read_config()
        mutate(config)
        with pytest.raises(ValueError, match="mismatch|hardcoded|contract|paths"):
            auditor.validate_config(config)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    auditor = load_auditor()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        auditor.strict_json(b'{"a": 1, "a": 2}', "test")
    with pytest.raises(ValueError, match="non-finite"):
        auditor.strict_json(b'{"a": NaN}', "test")


def test_boolean_values_cannot_impersonate_integer_contract_fields() -> None:
    auditor = load_auditor()
    for mutate in (
        lambda config: config.__setitem__("schema_version", True),
        lambda config: config["source"].__setitem__(
            "expected_unknown_entry_count", False
        ),
        lambda config: config["artifact"].__setitem__(
            "expected_locked_source_tree_checkpoint_count", False
        ),
    ):
        config = read_config()
        mutate(config)
        with pytest.raises(ValueError):
            auditor.validate_config(config)


def test_upstream_sensitive_and_pyc_paths_are_never_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = load_auditor()
    config = read_config()
    forbidden_names = {
        Path(item["path"]).name
        for item in config["source"]["tracked_files"]
        if item["path"].endswith((".json", ".yaml", ".yml", ".ipynb"))
    }
    forbidden_names.update(
        Path(item["path"]).name for item in config["source"]["ignored_pyc"]
    )
    forbidden_names.add("state.msgpack")
    opened_forbidden: list[str] = []
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            name = os.fsdecode(path)
            if Path(name).name in forbidden_names:
                opened_forbidden.append(name)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", tracking_open)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Path.read_text is forbidden")
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Path.read_bytes is forbidden")
        ),
    )
    record = auditor.run_static_audit(CONFIG, "unit-audit", auditor.ReadLedger())
    assert record["status"] == "passed"
    assert opened_forbidden == []


def mini_config(files: list[dict], pyc: list[dict] | None = None) -> dict:
    return {
        "source": {
            "tracked_files": files,
            "retained_python": [
                {
                    "path": item["path"],
                    "bytes": item["bytes"],
                    "sha256": item.get("sha256", "0" * 64),
                }
                for item in files
                if item["path"].endswith(".py")
            ],
            "ignored_pyc": pyc or [],
            "opaque_directories": [".git"],
        }
    }


@pytest.mark.parametrize("unknown", ["extra.py", "answers.json", "state.msgpack"])
def test_closed_world_rejects_unknown_file_before_any_file_read(
    tmp_path: Path, unknown: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    safe = repository / "safe.py"
    safe.write_text("x = 1\n", encoding="utf-8")
    (repository / unknown).write_text("must-not-read", encoding="utf-8")
    config = mini_config(
        [{"path": "safe.py", "bytes": 6, "mode": "100644", "blob_oid": "0" * 40}]
    )
    leaf_file_opens: list[str] = []
    real_open = auditor.os.open

    def tracking_open(path, flags, *args, **kwargs):
        name = os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else ""
        if name in {"safe.py", unknown}:
            leaf_file_opens.append(name)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", tracking_open)
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="unknown"):
            auditor.closed_world_inventory(root_fd, config)
    finally:
        os.close(root_fd)
    assert leaf_file_opens == []


def test_closed_world_rejects_unknown_directory_and_special_file(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    (repository / "safe.py").write_text("x = 1\n", encoding="utf-8")
    config = mini_config(
        [{"path": "safe.py", "bytes": 6, "mode": "100644", "blob_oid": "0" * 40}]
    )
    (repository / "unknown-dir").mkdir()
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="unknown directory"):
            auditor.closed_world_inventory(root_fd, config)
    finally:
        os.close(root_fd)
    (repository / "unknown-dir").rmdir()
    os.mkfifo(repository / "special")
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="nonregular"):
            auditor.closed_world_inventory(root_fd, config)
    finally:
        os.close(root_fd)


def test_parent_symlink_and_hardlink_aliases_fail_closed(tmp_path: Path) -> None:
    auditor = load_auditor()
    real = tmp_path / "real"
    real.mkdir()
    (real / "child").mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        descriptor = auditor.open_absolute_directory(linked / "child")
        os.close(descriptor)

    repository = tmp_path / "hardlinks"
    repository.mkdir()
    (repository / ".git").mkdir()
    solution = repository / "solution.json"
    solution.write_text("x = 1\n", encoding="utf-8")
    os.link(solution, repository / "safe.py")
    config = mini_config(
        [
            {"path": "safe.py", "bytes": 6, "mode": "100644", "blob_oid": "0" * 40},
            {"path": "solution.json", "bytes": 6, "mode": "100644", "blob_oid": "1" * 40},
        ]
    )
    root_fd = auditor.open_absolute_directory(repository)
    try:
        with pytest.raises(ValueError, match="hard-linked"):
            auditor.closed_world_inventory(root_fd, config)
    finally:
        os.close(root_fd)


def test_retained_reader_detects_replacement_and_same_inode_mutation(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    for mode in ("replace", "rewrite"):
        repository = tmp_path / mode
        repository.mkdir()
        (repository / ".git").mkdir()
        safe = repository / "safe.py"
        original = b"x = 1\n"
        safe.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        config = mini_config(
            [{"path": "safe.py", "bytes": 6, "mode": "100644", "blob_oid": "0" * 40}]
        )
        root_fd = auditor.open_absolute_directory(repository)
        try:
            snapshot = auditor.closed_world_inventory(root_fd, config)
            if mode == "replace":
                replacement = repository / "replacement"
                replacement.write_bytes(b"y = 2\n")
                os.replace(replacement, safe)
            else:
                descriptor = os.open(safe, os.O_WRONLY)
                try:
                    os.pwrite(descriptor, b"y = 2\n", 0)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            ledger = auditor.ReadLedger()
            ledger.bind_source_policy({"safe.py"})
            with pytest.raises(
                (RuntimeError, ValueError), match="changed after inventory|SHA-256 mismatch"
            ):
                auditor.secure_read_relative(
                    root_fd,
                    "safe.py",
                    expected_bytes=6,
                    expected_sha256=digest,
                    snapshot=snapshot,
                    role="retained_python",
                    ledger=ledger,
                )
        finally:
            os.close(root_fd)


def test_config_reader_cannot_follow_noncanonical_aliases(tmp_path: Path) -> None:
    auditor = load_auditor()
    regular = tmp_path / "regular.json"
    regular.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(regular)
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    hardlink = tmp_path / "hardlink.json"
    os.link(regular, hardlink)
    ledger = auditor.ReadLedger()
    ledger.bind_config(CONFIG)
    for path in (symlink, fifo, hardlink):
        with pytest.raises(ValueError, match="role/path binding mismatch"):
            auditor.secure_read_absolute(
                path,
                max_bytes=1024,
                role="canonical_config",
                ledger=ledger,
            )


def test_auditor_has_no_upstream_dynamic_network_or_gpu_hooks() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported: set[str] = set()
    forbidden_calls: list[str] = []
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
    assert not imported & {
        "jax",
        "flax",
        "wandb",
        "torch",
        "requests",
        "socket",
        "urllib",
        "http",
        "importlib",
        "marshal",
        "src",
    }
    assert forbidden_calls == []
    source = SCRIPT.read_text(encoding="utf-8")
    assert "shell=False" in source
    assert "curl" not in source
    assert "wget" not in source
    assert "nvidia-smi" not in source
