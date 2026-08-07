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
SCRIPT = ROOT / "scripts" / "audit_architects_2024_gates.py"
CONFIG = ROOT / "configs" / "architects_2024_gate_v1.json"
SOURCE_LOCK = ROOT / "configs" / "source_locks.json"
SOURCE = Path("/root/arc-paper-assets/sources/architects-2024")
CHECKPOINT_REPORT = (
    ROOT
    / "reports"
    / "architects-2024"
    / "20260806-4bit-checkpoint-integrity"
    / "run.json"
)
PREFLIGHT_REPORT = (
    ROOT
    / "reports"
    / "architects-2024"
    / "20260806-forward-preflight-gpu-occupied"
    / "run.json"
)


def load_auditor():
    spec = importlib.util.spec_from_file_location("architects_gate_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def auditor():
    return load_auditor()


def config_document() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def run_cli(output: Path, config: Path = CONFIG) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
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


def mutate_notebook(payload: bytes, old: str, new: str) -> bytes:
    document = json.loads(payload.decode("utf-8"))
    replacements = 0
    for cell in document["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        joined = "".join(source) if isinstance(source, list) else source
        if old in joined:
            replacements += joined.count(old)
            joined = joined.replace(old, new)
            cell["source"] = joined.splitlines(keepends=True)
    assert replacements > 0
    return json.dumps(document, ensure_ascii=False).encode("utf-8")


@pytest.fixture(scope="module")
def successful_record(tmp_path_factory, auditor):
    output = tmp_path_factory.mktemp("architects-gate") / "run"
    exit_code, record = auditor.execute_audit(CONFIG, output)
    assert exit_code == 0
    assert json.loads((output / "run.json").read_text(encoding="utf-8")) == record
    return record


def test_config_is_strict_and_valid(auditor):
    validated = auditor.validate_config(config_document())
    assert validated["method_id"] == "architects-2024"
    assert validated["counted_toward_smoke"] is False
    assert validated["source"]["repository_path"] == str(SOURCE)
    assert validated["source"]["expected_revision"] == (
        "d3ac3f6ebf6fb609bfdc782561ee99977ca35d95"
    )
    assert validated["source"]["expected_commit_tree"] == (
        "46eedc2d304d60bcb3a393063528d2a694223199"
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"a":1,"a":2}', "duplicate"),
        (b'{"a":NaN}', "non-finite"),
        (b"\xff", "UTF-8"),
    ],
)
def test_strict_json_rejects_ambiguous_documents(auditor, payload, message):
    with pytest.raises(ValueError, match=message):
        auditor.strict_json(payload, "adversarial")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value["expected_blocker_ids"].reverse(),
        lambda value: value["controls"].update(network_allowed=True),
        lambda value: value["source"].update(repository_path="/tmp/other"),
        lambda value: value["source"].update(expected_revision="0" * 40),
        lambda value: value["source"].update(expected_git_config_sha256="0" * 64),
        lambda value: value["source"].update(expected_git_head_sha256="0" * 64),
        lambda value: value["source"]["tracked_files"][0].update(blob_oid="0" * 40),
        lambda value: value["source"]["tracked_files"][0].update(
            bytes=value["source"]["tracked_files"][0]["bytes"] + 1
        ),
        lambda value: value["notebooks"]["updated"].update(
            sha256=value["notebooks"]["official"]["sha256"]
        ),
        lambda value: value["prior_reports"]["checkpoint_integrity"].update(
            path="arc-agi_evaluation_solutions.json"
        ),
        lambda value: value["source"]["read_files"][0].update(
            role="label_bearing_runner"
        ),
    ],
)
def test_config_mutations_fail_closed(auditor, mutation):
    document = config_document()
    mutation(document)
    with pytest.raises(ValueError):
        auditor.validate_config(document)


def test_matching_manifest_and_worktree_tamper_cannot_relabel_locked_tree(auditor):
    document = config_document()
    replacement = b"attacker-controlled matching worktree bytes"
    entry = document["source"]["tracked_files"][0]
    entry["bytes"] = len(replacement)
    entry["blob_oid"] = auditor.git_blob_oid(replacement)
    with pytest.raises(ValueError, match="metadata digest|Git tree"):
        auditor.validate_config(document)
    assert auditor.git_tree_oid_from_manifest(
        config_document()["source"]["tracked_files"]
    ) == "46eedc2d304d60bcb3a393063528d2a694223199"


def test_alternate_config_is_rejected_before_any_read_or_output(
    auditor, tmp_path, monkeypatch
):
    fake = tmp_path / "arc-agi_evaluation_solutions.json"
    fake.write_text('{"sensitive":"must-not-read"}', encoding="utf-8")
    output = tmp_path / "output"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("reader/output creation must not run")

    monkeypatch.setattr(auditor, "secure_read_absolute", forbidden)
    monkeypatch.setattr(auditor, "create_fresh_output", forbidden)
    with pytest.raises(ValueError, match="canonical"):
        auditor.execute_audit(fake, output)
    assert calls == []
    assert not output.exists()


def test_cli_rejects_solution_named_config_without_creating_output(tmp_path):
    fake = tmp_path / "arc-agi_evaluation_solutions.json"
    payload = b'{"do_not_read":"sentinel-architects-config"}'
    fake.write_bytes(payload)
    output = tmp_path / "output"
    result = run_cli(output, fake)
    assert result.returncode == 2
    assert "canonical" in result.stderr
    assert fake.read_bytes() == payload
    assert not output.exists()


def test_read_ledger_roles_are_closed(auditor):
    ledger = auditor.ReadLedger()
    assert ledger.authorize_absolute(CONFIG, "canonical_config") == "gate_config"
    with pytest.raises(ValueError):
        ledger.authorize_absolute(CHECKPOINT_REPORT, "canonical_config")
    with pytest.raises(ValueError):
        ledger.authorize_absolute(CONFIG, "attacker_category")
    assert (
        ledger.authorize_relative("LICENSE.txt", "code_license") == "source_text"
    )
    with pytest.raises(ValueError):
        ledger.authorize_relative(
            "training_code/run_evaluation_Llama-rearc_with_ttt.py",
            "code_license",
        )
    with pytest.raises(ValueError):
        ledger.authorize_relative("../LICENSE.txt", "code_license")


def test_source_lock_and_closed_world_are_exact(successful_record):
    source = successful_record["source"]
    assert source["git"]["revision"] == "d3ac3f6ebf6fb609bfdc782561ee99977ca35d95"
    assert source["git"]["commit_tree"] == "46eedc2d304d60bcb3a393063528d2a694223199"
    assert source["git"]["tracked_file_count"] == 18
    assert source["git"]["worktree_status_command_used"] is False
    assert source["git"]["git_may_read_worktree_bytes"] is False
    assert source["git"]["subprocess_used"] is False
    assert source["git"]["object_database_read"] is False
    assert source["git"]["object_database_reads_possible"] is False
    assert source["git"]["object_database_bytes_measured"] is True
    assert source["git"]["detached_head"] == {
        "path": ".git/HEAD",
        "bytes": 41,
        "sha256": "721b0545affeb16fe2d3fdf5079cbc5cb9ce2d9b713a94556187a998ad7f3f8a",
        "detached_revision": "d3ac3f6ebf6fb609bfdc782561ee99977ca35d95",
    }
    assert source["git"]["local_config"] == {
        "path": ".git/config",
        "bytes": 309,
        "sha256": "1737a734def03c3a14f4af03bf84e452d3a4fd431e28073ba808670873e296fc",
        "origin_url": "https://github.com/da-fr/arc-prize-2024",
        "dangerous_include_filter_helper_directives": False,
    }
    assert source["git"]["forbidden_auxiliary_paths"] == {
        "required_absent_paths": [
            ".git/commondir",
            ".git/config.worktree",
            ".git/info/attributes",
            ".git/objects/info/alternates",
            ".git/objects/info/http-alternates",
        ],
        "all_absent": True,
    }
    assert source["filesystem"]["file_count"] == 18
    assert source["filesystem"]["unknown_entry_count"] == 0
    assert len(source["filesystem"]["tracked_blob_bindings"]) == 18
    assert source["filesystem"]["tracked_bytes"] == 3071252
    assert source["initial_terminal_match"] is True
    assert source["terminal_verification_order"] == [
        "detached-head-local-config-and-locked-manifest",
        "closed-world-metadata-and-18-blob-binding",
        "source-root-path-identity",
    ]
    assert source["python_syntax"] == {
        "file_count": 9,
        "bytes": 60690,
        "failures": [],
    }


@pytest.mark.parametrize(
    ("target_path", "retained_role"),
    [
        (".github/overview.png", None),
        ("README.md", "source_readme"),
    ],
    ids=["previously-unread-leaf", "previously-retained-leaf"],
)
def test_terminal_blob_binding_rejects_same_size_leaf_replacement(
    auditor, tmp_path, target_path, retained_role
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".git").mkdir()
    document = config_document()
    declarations = []
    payloads = {}
    for locked in document["source"]["tracked_files"]:
        relative = locked["path"]
        payload = ("locked:" + relative).encode("utf-8")
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        payloads[relative] = payload
        declarations.append(
            {
                "path": relative,
                "mode": "100644",
                "blob_oid": auditor.git_blob_oid(payload),
                "bytes": len(payload),
            }
        )
    document["source"]["tracked_files"] = declarations
    ledger = auditor.ReadLedger()
    ledger.bind_tracked_policy(declarations)
    root_fd = auditor.open_absolute_directory(source_root)
    try:
        initial = auditor.closed_world_inventory(
            root_fd, document, ledger, phase="initial"
        )
        if retained_role is not None:
            retained_payload = payloads[target_path]
            auditor.secure_read_relative(
                root_fd,
                target_path,
                retained_role,
                len(retained_payload),
                hashlib.sha256(retained_payload).hexdigest(),
                ledger,
            )
        target = source_root / target_path
        original = target.read_bytes()
        replacement = bytes([original[0] ^ 1]) + original[1:]
        temporary = target.with_name(target.name + ".replacement")
        temporary.write_bytes(replacement)
        os.replace(temporary, target)
        assert target.stat().st_size == len(original)
        with pytest.raises(ValueError, match="Git blob lock mismatch"):
            auditor.closed_world_inventory(
                root_fd, document, ledger, phase="terminal"
            )
    finally:
        os.close(root_fd)
    assert initial["file_count"] == 18
    assert ledger.entries[-1]["path"] == target_path
    assert ledger.entries[-1]["outcome"] == "failed"


def test_source_relative_reader_rejects_symlink_and_hardlink(auditor, tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"license")
    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    (symlink_root / "LICENSE.txt").symlink_to(target)
    root_fd = auditor.open_absolute_directory(symlink_root)
    try:
        with pytest.raises((OSError, ValueError)):
            auditor.secure_read_relative(
                root_fd,
                "LICENSE.txt",
                "code_license",
                len(b"license"),
                hashlib.sha256(b"license").hexdigest(),
                auditor.ReadLedger(),
            )
    finally:
        os.close(root_fd)

    hardlink_root = tmp_path / "hardlink-root"
    hardlink_root.mkdir()
    os.link(target, hardlink_root / "LICENSE.txt")
    root_fd = auditor.open_absolute_directory(hardlink_root)
    try:
        with pytest.raises(ValueError, match="private regular"):
            auditor.secure_read_relative(
                root_fd,
                "LICENSE.txt",
                "code_license",
                len(b"license"),
                hashlib.sha256(b"license").hexdigest(),
                auditor.ReadLedger(),
            )
    finally:
        os.close(root_fd)


def test_source_relative_reader_records_wrong_sha_failure(auditor, tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = b"exact bytes were read before the digest mismatch"
    (source_root / "LICENSE.txt").write_bytes(payload)
    ledger = auditor.ReadLedger()
    root_fd = auditor.open_absolute_directory(source_root)
    try:
        with pytest.raises(ValueError, match="byte lock mismatch"):
            auditor.secure_read_relative(
                root_fd,
                "LICENSE.txt",
                "code_license",
                len(payload),
                "0" * 64,
                ledger,
            )
    finally:
        os.close(root_fd)
    assert ledger.entries == [
        {
            "path": "LICENSE.txt",
            "role": "code_license",
            "category": "source_text",
            "outcome": "failed",
            "read_started": True,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "error_type": "ValueError",
        }
    ]
    controls = auditor.failure_record(
        "wrong-sha", "static-audit", RuntimeError("probe"), ledger
    )["controls"]
    assert controls["explicit_read_file_count"] == 1
    assert controls["explicit_read_bytes"] == len(payload)
    assert controls["explicit_read_failed_count"] == 1


def test_source_relative_reader_records_mid_read_path_race(
    auditor, tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = b"a" * (1024 * 1024 + 17)
    target = source_root / "LICENSE.txt"
    target.write_bytes(payload)
    ledger = auditor.ReadLedger()
    original_read = auditor.os.read
    replaced = False

    def racing_read(fd, count):
        nonlocal replaced
        chunk = original_read(fd, count)
        if chunk and not replaced:
            replaced = True
            replacement = source_root / "replacement"
            replacement.write_bytes(b"b" * len(payload))
            os.replace(replacement, target)
        return chunk

    root_fd = auditor.open_absolute_directory(source_root)
    monkeypatch.setattr(auditor.os, "read", racing_read)
    try:
        with pytest.raises(RuntimeError, match="changed while (?:being )?read"):
            auditor.secure_read_relative(
                root_fd,
                "LICENSE.txt",
                "code_license",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                ledger,
            )
    finally:
        os.close(root_fd)
    assert replaced is True
    assert ledger.entries[0]["outcome"] == "failed"
    assert ledger.entries[0]["read_started"] is True
    assert ledger.entries[0]["bytes"] == len(payload)
    assert ledger.entries[0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert ledger.entries[0]["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "forbidden_path",
    ["arc-agi_evaluation_solutions.json", "checkpoint/model.safetensors"],
)
def test_arbitrary_relative_reader_is_rejected_before_open(
    auditor, tmp_path, monkeypatch, forbidden_path
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    root_fd = auditor.open_absolute_directory(source_root)
    open_calls = []

    def forbidden_open(*args, **kwargs):
        open_calls.append((args, kwargs))
        raise AssertionError("unauthorized path must be rejected before os.open")

    monkeypatch.setattr(auditor.os, "open", forbidden_open)
    try:
        with pytest.raises(ValueError, match="untrusted relative reader"):
            auditor.secure_read_relative(
                root_fd,
                forbidden_path,
                "python_source",
                0,
                hashlib.sha256(b"").hexdigest(),
                auditor.ReadLedger(),
            )
        with pytest.raises(ValueError, match="capability"):
            auditor._secure_read_locked_relative(
                root_fd, auditor.ReadLedger(), object()
            )
    finally:
        os.close(root_fd)
    assert open_calls == []


def test_audit_uses_no_git_or_other_subprocess(auditor, tmp_path, monkeypatch):
    calls = []

    def forbidden_popen(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("the audit must not start a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)
    exit_code, record = auditor.execute_audit(CONFIG, tmp_path / "run")
    assert exit_code == 0
    assert record["status"] == "passed"
    assert record["source"]["git"]["subprocess_used"] is False
    assert record["controls"]["git_query_count"] == 0
    assert calls == []


@pytest.mark.parametrize(
    "dangerous_fragment",
    [
        '[include]\n\tpath = /tmp/architects-malicious-include\n',
        '[filter "arc"]\n\tclean = /tmp/architects-malicious-filter\n',
        '[credential]\n\thelper = /tmp/architects-malicious-helper\n',
    ],
    ids=["include", "filter", "credential-helper"],
)
def test_malicious_local_git_config_is_rejected_before_git_execution(
    auditor, tmp_path, monkeypatch, dangerous_fragment
):
    source_root = tmp_path / "source"
    git_directory = source_root / ".git"
    git_directory.mkdir(parents=True)
    marker = tmp_path / "must-not-execute"
    payload = (
        "[core]\n"
        "\trepositoryformatversion = 1\n"
        '[remote "origin"]\n'
        "\turl = https://github.com/da-fr/arc-prize-2024\n"
        + dangerous_fragment.replace("/tmp/architects-malicious-filter", str(marker))
    ).encode("utf-8")
    (git_directory / "config").write_bytes(payload)
    document = config_document()
    document["source"]["expected_git_config_bytes"] = len(payload)
    document["source"]["expected_git_config_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    subprocess_calls = []

    def forbidden_popen(*args, **kwargs):
        subprocess_calls.append((args, kwargs))
        raise AssertionError("unsafe local config must never reach a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)
    root_fd = auditor.open_absolute_directory(source_root)
    try:
        with pytest.raises(ValueError, match="dangerous local Git config"):
            auditor.verify_git_contract(
                root_fd,
                document,
                auditor.ReadLedger(),
                phase="initial",
            )
    finally:
        os.close(root_fd)
    assert subprocess_calls == []
    assert not marker.exists()


@pytest.mark.parametrize(
    "forbidden_relative",
    [
        ".git/commondir",
        ".git/config.worktree",
        ".git/info/attributes",
        ".git/objects/info/alternates",
        ".git/objects/info/http-alternates",
    ],
)
def test_forbidden_git_indirection_path_is_rejected_without_execution(
    auditor, tmp_path, monkeypatch, forbidden_relative
):
    source_root = tmp_path / "source"
    forbidden = source_root / forbidden_relative
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("attacker-controlled Git indirection\n", encoding="utf-8")
    subprocess_calls = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )
    root_fd = auditor.open_absolute_directory(source_root)
    try:
        with pytest.raises(ValueError, match="forbidden Git auxiliary path"):
            auditor.verify_git_contract(
                root_fd,
                config_document(),
                auditor.ReadLedger(),
                phase="initial",
            )
    finally:
        os.close(root_fd)
    assert subprocess_calls == []


def test_terminal_git_observation_drift_after_retained_reads_is_rejected(
    auditor, monkeypatch
):
    original = auditor.verify_git_contract
    phases = []

    def drifting(root_fd, config, ledger, *, phase):
        phases.append(phase)
        observation = original(root_fd, config, ledger, phase=phase)
        if phase == "terminal":
            observation = {**observation, "commit_tree": "0" * 40}
        return observation

    monkeypatch.setattr(auditor, "verify_git_contract", drifting)
    with pytest.raises(RuntimeError, match="Git metadata drifted"):
        auditor.run_static_audit(CONFIG, "drift-probe", auditor.ReadLedger())
    assert phases == ["initial", "terminal"]


def test_terminal_verification_order_follows_all_retained_reads(
    auditor, monkeypatch
):
    events = []
    original_git = auditor.verify_git_contract
    original_inventory = auditor.closed_world_inventory
    original_identity = auditor.verify_directory_path_identity
    original_read = auditor.secure_read_relative

    def observing_git(root_fd, config, ledger, *, phase):
        events.append(f"git:{phase}")
        return original_git(root_fd, config, ledger, phase=phase)

    def observing_inventory(root_fd, config, ledger, *, phase):
        events.append(f"inventory:{phase}")
        return original_inventory(root_fd, config, ledger, phase=phase)

    def observing_identity(path, root_fd):
        events.append("root-identity")
        return original_identity(path, root_fd)

    def observing_read(*args, **kwargs):
        events.append("retained-read")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(auditor, "verify_git_contract", observing_git)
    monkeypatch.setattr(auditor, "closed_world_inventory", observing_inventory)
    monkeypatch.setattr(auditor, "verify_directory_path_identity", observing_identity)
    monkeypatch.setattr(auditor, "secure_read_relative", observing_read)
    record = auditor.run_static_audit(CONFIG, "order-probe", auditor.ReadLedger())
    assert record["status"] == "passed"
    last_retained = max(index for index, item in enumerate(events) if item == "retained-read")
    assert events[last_retained + 1 : last_retained + 4] == [
        "git:terminal",
        "inventory:terminal",
        "root-identity",
    ]


def test_full_ledger_contains_no_checkpoint_solution_or_pickle_path(successful_record):
    ledger = successful_record["read_ledger"]
    assert len(ledger) == 62
    assert all(item["outcome"] == "verified" for item in ledger)
    assert all(item["error_type"] is None for item in ledger)
    forbidden = (
        "_solutions.json",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
        ".pickle",
        ".bz2",
    )
    assert not any(item["path"].lower().endswith(forbidden) for item in ledger)
    assert not any("snapshots/" in item["path"] for item in ledger)
    controls = successful_record["controls"]
    assert controls["arc_solution_bytes_read"] is False
    assert controls["checkpoint_bytes_read"] is False
    assert controls["pickle_cache_bytes_read"] is False
    assert controls["source_notebook_files_read"] == 8
    assert controls["source_python_files_read"] == 27
    assert controls["retained_source_notebook_files_read"] == 2
    assert controls["retained_source_python_files_read"] == 9
    assert controls["prior_report_files_read"] == 2
    assert controls["source_worktree_binding_files_read"] == 36
    assert controls["git_local_config_files_read"] == 4
    assert controls["git_head_files_read"] == 4
    assert controls["explicit_read_verified_count"] == 62
    assert controls["explicit_read_failed_count"] == 0
    assert controls["git_worktree_verification_used"] is False
    assert controls["git_may_read_locked_source_worktree_bytes"] is False
    assert controls["full_tracked_worktree_blob_binding_used"] is True
    assert controls["git_subprocess_used"] is False
    assert controls["git_subprocess_object_database_reads_possible"] is False
    assert controls["git_subprocess_object_database_bytes_measured"] is True
    assert controls["git_worktree_content_command_used"] is False
    assert "no Git subprocess" in controls["explicit_read_bytes_scope"]


def test_absolute_reader_enforces_role_specific_size_cap(
    auditor, monkeypatch
):
    ledger = auditor.ReadLedger()
    monkeypatch.setitem(
        auditor.ABSOLUTE_MAX_BYTES, "canonical_config", CONFIG.stat().st_size - 1
    )
    with pytest.raises(ValueError, match="byte cap"):
        auditor.secure_read_absolute(CONFIG, "canonical_config", ledger)
    assert ledger.entries == [
        {
            "path": str(CONFIG),
            "role": "canonical_config",
            "category": "gate_config",
            "outcome": "failed",
            "read_started": False,
            "bytes": 0,
            "sha256": None,
            "error_type": "ValueError",
        }
    ]


def test_absolute_reader_records_mid_read_path_race(
    auditor, tmp_path, monkeypatch
):
    payload = CONFIG.read_bytes()
    target = tmp_path / "config.json"
    target.write_bytes(payload)
    original_parent = auditor.open_absolute_parent
    original_read = auditor.os.read
    replaced = False

    def redirected_parent(path):
        if path == CONFIG:
            return auditor.open_absolute_directory(tmp_path), target.name, tmp_path
        return original_parent(path)

    def racing_read(fd, count):
        nonlocal replaced
        chunk = original_read(fd, count)
        if chunk and not replaced:
            replaced = True
            replacement = tmp_path / "replacement.json"
            replacement.write_bytes(b" " * len(payload))
            os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(auditor, "open_absolute_parent", redirected_parent)
    monkeypatch.setattr(auditor.os, "read", racing_read)
    ledger = auditor.ReadLedger()
    with pytest.raises(RuntimeError, match="changed while (?:being )?read"):
        auditor.secure_read_absolute(CONFIG, "canonical_config", ledger)
    assert replaced is True
    assert ledger.entries[0]["outcome"] == "failed"
    assert ledger.entries[0]["read_started"] is True
    assert ledger.entries[0]["bytes"] == len(payload)
    assert ledger.entries[0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert ledger.entries[0]["error_type"] == "RuntimeError"


def test_absolute_bytes_can_be_verified_before_semantic_contract_fails(auditor):
    ledger = auditor.ReadLedger()
    payload = auditor.secure_read_absolute(SOURCE_LOCK, "source_lock", ledger)
    document = config_document()
    document["source_lock"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        auditor.verify_source_lock(payload, document)
    assert ledger.entries[0]["outcome"] == "verified"
    assert ledger.entries[0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_official_and_updated_notebook_roles_are_distinct(successful_record):
    notebooks = successful_record["notebook_roles"]
    assert notebooks["role_hashes_distinct"] is True
    assert notebooks["official"]["sha256"] == (
        "4ecb52b2811226711b656f40bf3ecff62509c2a189a222c1394cb14180b2bfe9"
    )
    assert notebooks["updated"]["sha256"] == (
        "28e76d2c888ab98a0169d9e6889d1b4ed4d03dc54ebd877401e055658d41d6dd"
    )
    assert notebooks["official"]["role"] == "official-kaggle-53.5-submission"
    assert notebooks["updated"]["role"] == (
        "updated-local-candidate-not-official-submission"
    )


def test_official_notebook_candidate_contract(successful_record):
    official = successful_record["notebook_roles"]["official"]
    assert official["true_test_challenge_only_candidate"] is True
    assert official["challenge_assignment"][0].endswith(
        "arc-agi_test_challenges.json"
    )
    assert official["fake_reply_assignment"][0].endswith(
        "arc-agi_training_solutions.json"
    )
    assert all(item[2] is True for item in official["load_replies_lines"])
    assert all(item[2] is True for item in official["validation_lines"])
    assert official["remove_replies_lines"]
    assert official["challenge_load_lines"]
    assert official["prepare_dataset"]["remove_replies_before_train_consumers"] is True
    assert official["prepare_dataset"]["contract_passed"] is True
    assert official["training_consumer"]["contract_passed"] is True
    assert official["pickle_cache_read_present"] is True


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "if arc_test_set.is_fake: arc_test_set.load_replies(arc_solutions_file)",
            "arc_test_set.load_replies(arc_solutions_file)",
        ),
        ("ds = ds.remove_replies()", "ds = ds.change_keys(ds.keys)"),
        (
            "arc_test_set = ArcDataset.from_file(arc_challenge_file)",
            "arc_test_set = ArcDataset.from_file(arc_solutions_file)",
        ),
        (
            "arc_test_set = ArcDataset.from_file(arc_challenge_file)",
            "decoy_test_set = ArcDataset.from_file(arc_challenge_file)",
        ),
        (
            "    if arc_test_set.is_fake:\n        decoder.benchmark_selection_algos",
            "    if True:\n        decoder.benchmark_selection_algos",
        ),
        ("arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json"),
    ],
)
def test_official_notebook_semantic_mutations_fail_candidate(
    auditor, old, new
):
    payload = (SOURCE / "kaggle_notebooks/arc-prize-2024_kaggle.ipynb").read_bytes()
    mutated = mutate_notebook(payload, old, new)
    observation = auditor.analyze_official_notebook(mutated, "mutated.ipynb")
    assert observation["true_test_challenge_only_candidate"] is False


def test_dead_remove_replies_decoy_does_not_replace_real_training_flow(auditor):
    payload = (SOURCE / "kaggle_notebooks/arc-prize-2024_kaggle.ipynb").read_bytes()
    mutated = mutate_notebook(
        payload,
        "        ds = ds.remove_replies()\n",
        "        if False:\n            ds = ds.remove_replies()\n",
    )
    observation = auditor.analyze_official_notebook(mutated, "mutated.ipynb")
    assert observation["from_file_lines"]
    assert observation["prepare_dataset"]["remove_replies_lines"] == []
    assert observation["prepare_dataset"]["contract_passed"] is False
    assert observation["true_test_challenge_only_candidate"] is False


def test_unguarded_call_in_any_notebook_cell_fails_candidate(auditor):
    payload = (SOURCE / "kaggle_notebooks/arc-prize-2024_kaggle.ipynb").read_bytes()
    document = json.loads(payload.decode("utf-8"))
    document["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["arc_test_set.load_replies(arc_solutions_file)\n"],
        }
    )
    mutated = json.dumps(document).encode("utf-8")
    observation = auditor.analyze_official_notebook(mutated, "mutated.ipynb")
    assert any(item[2] is False for item in observation["load_replies_lines"])
    assert observation["true_test_challenge_only_candidate"] is False


def test_local_runners_are_unconditional_label_bearing_paths(successful_record):
    runners = successful_record["ast_observations"]["local_label_bearing_runners"]
    assert [item["path"] for item in runners] == [
        "training_code/run_evaluation_Llama-rearc_with_ttt.py",
        "training_code/run_evaluation_Llama-rearc_without_ttt.py",
    ]
    assert all(item["unconditional_solution_read"] for item in runners)
    assert all(item["same_process_validation_call_lines"] for item in runners)
    assert all(item["contract_passed"] for item in runners)


def test_local_runner_control_flow_mutation_is_detected(auditor):
    path = SOURCE / "training_code/run_evaluation_Llama-rearc_without_ttt.py"
    text = path.read_text(encoding="utf-8")
    old = "arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))"
    assert old in text
    mutated = text.replace(old, "if is_fake:\n    " + old)
    observation = auditor.analyze_local_runner(ast.parse(mutated), path.name)
    assert observation["solution_load_call_lines"]
    assert observation["unconditional_solution_load_call_lines"] == []
    assert observation["contract_passed"] is False


def test_arc1_training_contamination_ast_contract(successful_record):
    contamination = successful_record["ast_observations"][
        "arc1_training_contamination"
    ]
    assert contamination["arc1_training_contamination_confirmed"] is True
    assert contamination["evaluation_solution_suffix_in_train_branch"] is True
    assert contamination["load_solutions_lines"]
    assert contamination["move_test_to_train_lines"]
    assert contamination["repeat_lines"]
    assert contamination["arceval_mix_key_lines"]


def test_arc1_contamination_mutation_is_detected(auditor):
    path = SOURCE / "training_code/run_finetuning_Nemo-full.py"
    text = path.read_text(encoding="utf-8")
    mutated = text.replace(
        "arc-agi_evaluation_solutions.json", "arc-agi_training_solutions.json"
    )
    observation = auditor.analyze_contamination(ast.parse(mutated))
    assert observation["arc1_training_contamination_confirmed"] is False


def test_model_loader_safe_offline_gate_is_blocked(successful_record):
    loader = successful_record["ast_observations"]["model_loader"]
    assert loader["local_files_only_explicit_true"] is False
    assert loader["trust_remote_code_explicit_false"] is False
    assert loader["safe_offline_contract_passed"] is False


def test_model_loader_safe_keyword_mutation_would_pass_unit_contract(auditor):
    path = SOURCE / "training_code/model_tools.py"
    text = path.read_text(encoding="utf-8")
    old = "        load_in_4bit=True,\n"
    assert old in text
    mutated = text.replace(
        old,
        old
        + "        local_files_only=True,\n"
        + "        trust_remote_code=False,\n",
        1,
    )
    observation = auditor.analyze_model_loader(ast.parse(mutated))
    assert observation["safe_offline_contract_passed"] is True


def test_prior_checkpoint_report_is_exact_and_weights_are_not_reopened(
    successful_record
):
    artifact = successful_record["artifact"]
    assert artifact["report_sha256"] == (
        "6cf23296baf789502b990be884e0420898b067ba49e041dfade1396c0a5ef8f3"
    )
    assert artifact["revision"] == "6de719999a213e717fe339fb5a29177ddc4310d9"
    assert artifact["file_count"] == 9
    assert artifact["total_bytes"] == 3790920477
    assert artifact["model_file"]["sha256"] == (
        "96ae74f8955c5bf3e84d5732494525d13076410710a060641893359db13300c5"
    )
    assert artifact["checkpoint_bytes_reopened"] is False


def test_prior_checkpoint_report_mutation_fails_exact_hash(auditor):
    payload = CHECKPOINT_REPORT.read_bytes()
    mutated = payload.replace(b'"status": "passed"', b'"status": "failed"')
    assert mutated != payload
    with pytest.raises(ValueError, match="hash"):
        auditor.verify_checkpoint_report(mutated, config_document())


def test_prior_preflight_report_is_exact_without_new_gpu_query(successful_record):
    preflight = successful_record["resource_preflight"]
    assert preflight == {
        "report_sha256": "3c1ca104f4f5317404196308e515f2e42f7161be63f45395e7df2be93aff74c6",
        "status": "blocked",
        "minimum_free_vram_bytes": 10737418240,
        "observed_free_vram_bytes": 5335154688,
        "gpu_query_repeated": False,
    }


def test_success_report_is_static_blocker_evidence(successful_record):
    assert successful_record["status"] == "passed"
    assert successful_record["audit_status"] == "passed"
    assert successful_record["counted_toward_smoke"] is False
    assert successful_record["strict_runtime_promotion"] is False
    assert successful_record["performance_claim"] is False
    assert successful_record["prediction_produced"] is False
    assert successful_record["solver_prediction_produced"] is False
    assert successful_record["strict_runtime_promoted"] is False
    assert successful_record["performance_table_eligible"] is False
    assert successful_record["solver_gate_passed"] is False
    assert successful_record["method_gate_status"] == "blocked"
    assert successful_record["evidence_scope"] == "blocker_audit"
    assert successful_record["fairness"] == successful_record["benchmark_policy"]
    assert successful_record["method_gate"]["status"] == "blocked"
    assert successful_record["gate_summary"] == {
        "passed": 4,
        "blocked": 8,
        "failed": 0,
    }
    assert successful_record["method_gate"]["blocking_gate_ids"] == [
        "model-license-review",
        "arc1-training-contamination",
        "local-runner-label-firewall",
        "safe-offline-model-load",
        "pickle-cache-isolation",
        "dependency-environment-parity",
        "resource-capacity",
        "solver-prediction-and-parity",
    ]
    assert successful_record["benchmark_policy"] == {
        "arc_agi_1": "training-contaminated-ineligible-for-clean-main-board",
        "arc_agi_2": "potential-new-transfer-after-challenge-only-adapter-and-all-runtime-gates",
    }


def test_failure_record_keeps_common_gate_fields_conservative(auditor):
    record = auditor.failure_record(
        "failure-probe", "static-audit", RuntimeError("probe"), auditor.ReadLedger()
    )
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["solver_prediction_produced"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert record["solver_gate_passed"] is False
    assert record["evidence_scope"] == "blocker_audit"


def test_preexisting_output_is_preserved(auditor, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(auditor.OutputPathError, match="must not exist"):
        auditor.execute_audit(CONFIG, output)
    assert sentinel.read_bytes() == b"preserve"


def test_output_symlink_and_symlink_parent_are_rejected(auditor, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(auditor.OutputPathError):
        auditor.execute_audit(CONFIG, output)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        auditor.execute_audit(CONFIG, linked_parent / "output")
    assert not (real_parent / "output").exists()


def test_output_creation_inode_swap_is_rejected(
    auditor, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    original_open = auditor.os.open
    swapped = False

    def racing_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and path == output.name
            and dir_fd is not None
            and flags & auditor.os.O_DIRECTORY
        ):
            swapped = True
            os.rename(
                output.name,
                "created-by-auditor",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.mkdir(output.name, dir_fd=dir_fd)
        return original_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(auditor.os, "open", racing_open)
    with pytest.raises(auditor.OutputPathError, match="raced"):
        auditor.create_fresh_output(output)
    assert swapped


def test_output_parent_directory_fsync_failure_propagates(
    auditor, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    calls = 0

    def failing_fsync(_fd):
        nonlocal calls
        calls += 1
        raise OSError("injected parent directory fsync failure")

    monkeypatch.setattr(auditor.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="parent directory fsync failure"):
        auditor.create_fresh_output(output)
    assert calls == 1
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_output_swap_before_publish_is_rejected(auditor, tmp_path):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    try:
        os.rename(
            output.leaf,
            "pinned-original",
            src_dir_fd=output.parent_descriptor,
            dst_dir_fd=output.parent_descriptor,
        )
        os.mkdir(output.leaf, dir_fd=output.parent_descriptor)
        with pytest.raises(auditor.OutputPathError):
            auditor.write_json_no_clobber(output, {"status": "test"})
        assert not (output_path / "run.json").exists()
    finally:
        output.close(record_committed=False)


def test_output_swap_during_publish_is_rejected(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    original_rename = auditor.rename_no_replace
    swapped = False

    def racing_rename(*args, **kwargs):
        nonlocal swapped
        result = original_rename(*args, **kwargs)
        if not swapped:
            swapped = True
            os.rename(
                output.leaf,
                "pinned-original",
                src_dir_fd=output.parent_descriptor,
                dst_dir_fd=output.parent_descriptor,
            )
            os.mkdir(output.leaf, dir_fd=output.parent_descriptor)
        return result

    monkeypatch.setattr(auditor, "rename_no_replace", racing_rename)
    try:
        with pytest.raises(auditor.OutputPathError):
            auditor.write_json_no_clobber(output, {"status": "test"})
        assert swapped
        assert not (output_path / "run.json").exists()
    finally:
        output.close(record_committed=False)


def test_serialized_bytes_are_verified_before_commit(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    original_read = auditor.os.read
    corrupted = False

    def corrupting_read(fd, count):
        nonlocal corrupted
        payload = original_read(fd, count)
        if payload and not corrupted:
            corrupted = True
            return bytes([payload[0] ^ 1]) + payload[1:]
        return payload

    monkeypatch.setattr(auditor.os, "read", corrupting_read)
    try:
        with pytest.raises(auditor.OutputPathError, match="serialized"):
            auditor.write_json_no_clobber(output, {"status": "test"})
        assert corrupted
        assert not (output_path / "run.json").exists()
    finally:
        output.close(record_committed=False)


def test_run_json_no_clobber_preserves_raced_file(auditor, tmp_path):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    attacker = output_path / "run.json"
    attacker.write_bytes(b"attacker-sentinel")
    try:
        with pytest.raises(auditor.OutputPathError, match="already exists"):
            auditor.write_json_no_clobber(output, {"status": "test"})
        assert attacker.read_bytes() == b"attacker-sentinel"
    finally:
        output.close(record_committed=False)


def test_atomic_publication_never_unlinks_and_consumes_temporary_name(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    value = {"status": "atomic-no-unlink"}
    unlink_calls = []

    def forbidden_unlink(*args, **kwargs):
        unlink_calls.append((args, kwargs))
        raise AssertionError("publication recovery must never unlink a pathname")

    monkeypatch.setattr(auditor.os, "unlink", forbidden_unlink)
    committed = False
    try:
        auditor.write_json_no_clobber(output, value)
        committed = True
        assert (output_path / "run.json").read_bytes() == auditor.canonical_bytes(
            value, pretty=True
        )
        assert sorted(path.name for path in output_path.iterdir()) == ["run.json"]
        assert unlink_calls == []
    finally:
        output.close(record_committed=committed)


def test_directory_fsync_failure_propagates_without_deleting_renamed_record(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    value = {"status": "committed-at-atomic-rename"}
    original_fsync = auditor.os.fsync
    directory_syncs = 0

    def failing_directory_fsync(fd):
        nonlocal directory_syncs
        if fd == output.descriptor:
            directory_syncs += 1
            raise OSError("injected directory fsync failure after atomic rename")
        return original_fsync(fd)

    monkeypatch.setattr(auditor.os, "fsync", failing_directory_fsync)
    try:
        with pytest.raises(OSError, match="directory fsync failure"):
            auditor.write_json_no_clobber(output, value)
        assert directory_syncs == 1
        assert (output_path / "run.json").read_bytes() == auditor.canonical_bytes(
            value, pretty=True
        )
    finally:
        output.close(record_committed=False)


def test_racer_replacement_during_commit_check_is_preserved_without_unlink(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    value = {"status": "owned-record"}
    attacker_payload = b"attacker-raced-run-json"
    original_fsync = auditor.os.fsync
    swapped = False

    def racing_directory_fsync(fd):
        nonlocal swapped
        if fd == output.descriptor and not swapped:
            swapped = True
            os.rename(
                "run.json",
                "auditor-owned-record",
                src_dir_fd=output.descriptor,
                dst_dir_fd=output.descriptor,
            )
            attacker_fd = os.open(
                "run.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o644,
                dir_fd=output.descriptor,
            )
            try:
                os.write(attacker_fd, attacker_payload)
                original_fsync(attacker_fd)
            finally:
                os.close(attacker_fd)
        return original_fsync(fd)

    monkeypatch.setattr(auditor.os, "fsync", racing_directory_fsync)
    try:
        with pytest.raises(auditor.OutputPathError, match="changed"):
            auditor.write_json_no_clobber(output, value)
        assert swapped is True
        assert (output_path / "run.json").read_bytes() == attacker_payload
        assert (output_path / "auditor-owned-record").read_bytes() == (
            auditor.canonical_bytes(value, pretty=True)
        )
    finally:
        output.close(record_committed=False)


def test_temp_descriptor_close_failure_after_commit_is_suppressed(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    value = {"status": "committed-before-temp-close"}
    original_open = auditor.os.open
    original_close = auditor.os.close
    temp_descriptor = None
    close_failure_injected = False

    def capturing_open(path, flags, *args, **kwargs):
        nonlocal temp_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if isinstance(path, str) and path.startswith("._run-"):
            temp_descriptor = descriptor
        return descriptor

    def failing_close(fd):
        nonlocal close_failure_injected
        if fd == temp_descriptor and not close_failure_injected:
            close_failure_injected = True
            original_close(fd)
            raise OSError("injected temp descriptor close failure")
        return original_close(fd)

    monkeypatch.setattr(auditor.os, "open", capturing_open)
    monkeypatch.setattr(auditor.os, "close", failing_close)
    committed = False
    try:
        auditor.write_json_no_clobber(output, value)
        committed = True
        assert close_failure_injected
        assert (output_path / "run.json").read_bytes() == auditor.canonical_bytes(
            value, pretty=True
        )
    finally:
        output.close(record_committed=committed)


def test_fresh_output_close_failure_after_commit_is_suppressed(
    auditor, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    auditor.write_json_no_clobber(output, {"status": "committed"})
    original_close = auditor.os.close
    injected = False

    def failing_output_close(fd):
        nonlocal injected
        if fd == output.descriptor and not injected:
            injected = True
            original_close(fd)
            raise OSError("injected output close failure")
        return original_close(fd)

    monkeypatch.setattr(auditor.os, "close", failing_output_close)
    output.close(record_committed=True)
    assert injected
    assert json.loads((output_path / "run.json").read_text(encoding="utf-8")) == {
        "status": "committed"
    }


def test_successful_publication_has_exact_serialized_record_and_no_temp(
    auditor, tmp_path
):
    output_path = tmp_path / "output"
    output = auditor.create_fresh_output(output_path)
    value = {"z": [1, 2], "a": {"ok": True}}
    committed = False
    try:
        auditor.write_json_no_clobber(output, value)
        committed = True
        assert (output_path / "run.json").read_bytes() == auditor.canonical_bytes(
            value, pretty=True
        )
        assert sorted(path.name for path in output_path.iterdir()) == ["run.json"]
    finally:
        output.close(record_committed=committed)


def test_script_and_config_have_no_formatting_or_syntax_error():
    compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")
    json.loads(CONFIG.read_text(encoding="utf-8"))
