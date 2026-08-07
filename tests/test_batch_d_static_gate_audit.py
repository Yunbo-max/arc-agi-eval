from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import pytest


ROOT = Path(__file__).absolute().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_batch_d_static_gate.py"
AUDITOR = ROOT / "scripts" / "audit_batch_d_static_gates.py"
SUPPORT = ROOT / "scripts" / "audit_batch_c_static_gates.py"
SOURCE_LOCK = ROOT / "configs" / "source_locks.json"
PYTHON = Path("/usr/bin/python3")
SAFE_TEST_ROOT = Path("/tmp/arc-agi-eval-batch-d-tests")

LAUNCHER_SHA256 = "385d3f26fd9cf35a67f6e3bd2a15059a0db7faff6cf2cb3e3036931aeceff347"
AUDITOR_SHA256 = "d5557b71deb4a9f2b9cc664b3a821f2e95a498719a5cfc029f1bb102917bc0f5"
SUPPORT_SHA256 = "8860877257cf2864ddf8304fdef407d76de72339b6aba9d47391db5a57c7626e"
SOURCE_LOCK_SHA256 = "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"

PROFILES = {
    "soar": {
        "manifest": ROOT / "configs" / "soar_gate_runner_manifest_v1.json",
        "manifest_relative": "configs/soar_gate_runner_manifest_v1.json",
        "manifest_id": "soar-gate-runner-manifest-v1",
        "manifest_sha256": "e841e5f579ce6eae221ff986f7c2eefad2575c520693379e8c0e4aeeb6718999",
        "members_sha256": "40ca616e6215693e06ca5eb541ec1d593740ac98c604961f28d633b61f2a12f9",
        "config": ROOT / "configs" / "soar_gate_v1.json",
        "config_relative": "configs/soar_gate_v1.json",
        "config_file_sha256": "9e002b67c3cd74d2518309e528c4243dfacae503869af2a26abad3a528247fd3",
        "config_canonical_sha256": "9867f1c85644eb2a9a2a981191742bac248144124143991902fa728e7735889b",
        "scope": "source-artifact-dataset-label-api-code-resource-gate-audit-only",
        "tracked_entry_count": 46,
        "tracked_blob_file_count": 46,
        "tracked_blob_bytes": 94_998_756,
        "retained_count": 28,
        "retained_bytes": 309_288,
        "metadata_only_count": 18,
        "metadata_only_bytes": 94_689_468,
        "gitlink_count": 0,
        "blocker_count": 13,
        "passed_gate_count": 2,
    },
    "nvarc": {
        "manifest": ROOT / "configs" / "nvarc_gate_runner_manifest_v1.json",
        "manifest_relative": "configs/nvarc_gate_runner_manifest_v1.json",
        "manifest_id": "nvarc-gate-runner-manifest-v1",
        "manifest_sha256": "137c1c79dae3f4c00d10a12d4157991149794897860545223d2e11d6294622ae",
        "members_sha256": "777d6d09be3fe83196be86264ae7cec5639acceee8dfdb003c13e2ccd634e0a4",
        "config": ROOT / "configs" / "nvarc_gate_v1.json",
        "config_relative": "configs/nvarc_gate_v1.json",
        "config_file_sha256": "b11e93afc9805b846c21ab2eda6af83bd4cf6c1d8446e6cfcd9a23d903bd17cb",
        "config_canonical_sha256": "f2766a0fdd049bb951d19951c46751ceb3979be17c4c94ad3871503a5a11365f",
        "scope": "source-gitlink-artifact-dataset-label-code-resource-gate-audit-only",
        "tracked_entry_count": 46,
        "tracked_blob_file_count": 39,
        "tracked_blob_bytes": 1_918_149,
        "retained_count": 24,
        "retained_bytes": 101_686,
        "metadata_only_count": 15,
        "metadata_only_bytes": 1_816_463,
        "gitlink_count": 7,
        "blocker_count": 12,
        "passed_gate_count": 1,
    },
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_auditor():
    spec = importlib.util.spec_from_file_location("batch_d_gate_auditor_test", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    support_payload = SUPPORT.read_bytes()
    assert hashlib.sha256(support_payload).hexdigest() == SUPPORT_SHA256
    module.core = module._execute_verified_support(support_payload)
    return module


@pytest.fixture
def batch_d_tmp_roots(tmp_path: Path):
    token = f"{os.getpid()}-{tmp_path.name}-{time.time_ns()}"
    roots = {
        method: SAFE_TEST_ROOT / method / token for method in sorted(PROFILES)
    }
    for path in roots.values():
        path.mkdir(parents=True)
    try:
        yield roots
    finally:
        for path in roots.values():
            shutil.rmtree(path)


def launcher_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def run_launcher(
    method: str,
    output: Path,
    *,
    config: Path | None = None,
    expected_manifest_sha256: str | None = None,
    isolated: bool = True,
) -> subprocess.CompletedProcess[str]:
    profile = PROFILES[method]
    flags = ["-I", "-B", "-S"] if isolated else ["-B"]
    return subprocess.run(
        [
            str(PYTHON),
            *flags,
            str(LAUNCHER),
            "--manifest",
            profile["manifest_relative"],
            "--expected-manifest-sha256",
            expected_manifest_sha256 or profile["manifest_sha256"],
            "--config",
            str(config or profile["config"]),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        env=launcher_environment(),
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("method", ["soar", "nvarc"])
def test_config_and_runner_manifest_are_canonical_hash_closed(method: str) -> None:
    profile = PROFILES[method]
    manifest_path = profile["manifest"]
    config_path = profile["config"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert file_sha256(LAUNCHER) == LAUNCHER_SHA256
    assert file_sha256(AUDITOR) == AUDITOR_SHA256
    assert file_sha256(SUPPORT) == SUPPORT_SHA256
    assert file_sha256(SOURCE_LOCK) == SOURCE_LOCK_SHA256
    assert file_sha256(manifest_path) == profile["manifest_sha256"]
    assert file_sha256(config_path) == profile["config_file_sha256"]
    assert canonical_sha256(config) == profile["config_canonical_sha256"]

    assert set(manifest) == {
        "schema_version",
        "manifest_id",
        "method_id",
        "member_count",
        "members",
        "members_sha256",
    }
    assert manifest["schema_version"] == 1
    assert manifest["manifest_id"] == profile["manifest_id"]
    assert manifest["method_id"] == method
    assert manifest["member_count"] == 5
    assert len(manifest["members"]) == 5
    assert manifest["members_sha256"] == profile["members_sha256"]
    assert manifest["members_sha256"] == canonical_sha256(manifest["members"])
    expected_members = {
        "launcher": "scripts/launch_batch_d_static_gate.py",
        "auditor": "scripts/audit_batch_d_static_gates.py",
        "support": "scripts/audit_batch_c_static_gates.py",
        "config": profile["config_relative"],
        "source_lock": "configs/source_locks.json",
    }
    assert {item["role"]: item["path"] for item in manifest["members"]} == (
        expected_members
    )
    for member in manifest["members"]:
        path = ROOT / member["path"]
        payload = path.read_bytes()
        assert len(payload) == member["bytes"]
        assert hashlib.sha256(payload).hexdigest() == member["sha256"]
        if member["role"] == "config":
            assert member["canonical_sha256"] == profile[
                "config_canonical_sha256"
            ]

    auditor = load_auditor()
    auditor_profile = auditor.PROFILES[profile["manifest_relative"]]
    assert auditor_profile["method_id"] == method
    assert auditor_profile["expected_config_canonical_sha256"] == profile[
        "config_canonical_sha256"
    ]
    assert auditor.validate_config(config, auditor_profile) == config


@pytest.mark.parametrize(
    ("method", "mutation", "message"),
    [
        ("soar", "tracked-count", "blob count"),
        ("soar", "tracked-bytes", "blob bytes"),
        ("nvarc", "tracked-entry-count", "entry count"),
        ("nvarc", "shallow", "shallow-repository"),
        ("nvarc", "opaque", "opaque worktree"),
        ("nvarc", "blocker-ids", "expected blocker IDs"),
    ],
)
def test_rebound_config_rejects_unconsumed_contract_drift(
    method: str, mutation: str, message: str
) -> None:
    profile = PROFILES[method]
    config = json.loads(profile["config"].read_text(encoding="utf-8"))
    source = config["source"]
    if mutation == "tracked-count":
        source["expected_tracked_file_count"] += 1
    elif mutation == "tracked-bytes":
        source["expected_tracked_bytes"] += 1
    elif mutation == "tracked-entry-count":
        source["expected_tracked_entry_count"] += 1
    elif mutation == "shallow":
        source["expected_shallow_repository"] = False
    elif mutation == "opaque":
        source["opaque_worktree_paths"] = source["opaque_worktree_paths"][:-1]
    elif mutation == "blocker-ids":
        config["expected_blocker_ids"] = config["expected_blocker_ids"][:-1]
    else:  # pragma: no cover - parameter table is closed above
        raise AssertionError(mutation)

    auditor = load_auditor()
    auditor_profile = dict(
        auditor.PROFILES[profile["manifest_relative"]],
        expected_config_canonical_sha256=canonical_sha256(config),
    )
    with pytest.raises(ValueError, match=message):
        auditor.validate_config(config, auditor_profile)


@pytest.mark.parametrize("method", ["soar", "nvarc"])
def test_isolated_launcher_passes_static_gate_while_method_remains_blocked(
    method: str, batch_d_tmp_roots: dict[str, Path]
) -> None:
    profile = PROFILES[method]
    output = batch_d_tmp_roots[method] / "happy-path"
    completed = run_launcher(method, output)
    assert completed.returncode == 0, completed.stderr or completed.stdout

    assert sorted(item.name for item in output.iterdir()) == ["run.json"]
    run_path = output / "run.json"
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(run_path.stat().st_mode) == 0o600
    assert run_path.stat().st_nlink == 1
    record = json.loads(run_path.read_text(encoding="utf-8"))

    assert record["method_id"] == method
    assert record["status"] == "passed"
    assert record["method_gate_status"] == "blocked"
    assert record["scope"] == profile["scope"]
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["solver_gate_passed"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert "predictions" not in record
    assert "results" not in record
    assert "score" not in record
    assert all(record["validation"].values())
    assert record["gate_summary"] == {
        "blocked": profile["blocker_count"],
        "passed": profile["passed_gate_count"],
    }
    assert len(record["blockers"]) == profile["blocker_count"]
    assert all(item["status"] == "blocked" for item in record["blockers"])

    config = json.loads(profile["config"].read_text(encoding="utf-8"))
    source = record["source"]
    assert source["repository_shallow"] is True
    assert source["shallow_boundary_sha256"] == config["source"][
        "git_metadata_contract"
    ]["shallow"]["sha256"]
    assert source["tracked_entry_count"] == profile["tracked_entry_count"]
    assert source["tracked_blob_file_count"] == profile["tracked_blob_file_count"]
    assert source["tracked_blob_bytes"] == profile["tracked_blob_bytes"]
    assert source["retained_text_file_count"] == profile["retained_count"]
    assert source["retained_text_bytes"] == profile["retained_bytes"]
    assert source["metadata_only_file_count"] == profile["metadata_only_count"]
    assert source["metadata_only_bytes"] == profile["metadata_only_bytes"]
    assert source["tracked_gitlink_count"] == profile["gitlink_count"]
    assert source["gitlink_root_directory_enumerated_count"] == profile[
        "gitlink_count"
    ]
    assert source["gitlink_descendant_entries_opened_count"] == 0
    assert source["gitlink_descendant_leaf_bytes_read"] is False
    assert source["working_tree_all_files_byte_exact_verified"] is False
    assert len(source["metadata_only_inventory"]) == profile[
        "metadata_only_count"
    ]
    assert all(
        item["worktree_bytes_read"] is False
        and item["worktree_sha256"] is None
        for item in source["metadata_only_inventory"]
    )

    resources = record["resources"]
    for key in (
        "provider_requests",
        "currency_spend_usd",
        "gpu_used",
        "network_used",
    ):
        assert resources[key] is None
    assert resources["intentional_provider_requests"] == 0
    assert resources["intentional_currency_spend_usd"] == 0.0
    assert resources["network_usage_measurement"] == "not-instrumented"
    assert resources["gpu_usage_measurement"] == "not-instrumented"
    assert resources["wall_and_cpu_deltas_include_bootstrap"] is False
    assert resources["max_rss_includes_premeasurement_process_lifetime"] is True

    restricted_paths = set(config["source"]["metadata_only_paths"])
    file_reads = record["read_ledger"]["file_read_attempts"]
    assert not restricted_paths.intersection(item["path"] for item in file_reads)
    retained_reads = [
        item for item in file_reads if item["category"] == "retained_source_text"
    ]
    assert len(retained_reads) == 2 * profile["retained_count"]
    assert all(item["status"] == "completed" for item in retained_reads)
    assert all(
        item["worktree_content_requested"] is False
        for item in record["read_ledger"]["git_subprocesses"]
    )
    assert record["replay_consistency"]["complete_observation_count"] == 2
    assert record["replay_consistency"]["equal"] is True
    assert record["read_ledger_scope"].startswith("second_complete_observation")
    assert set(record["phase_read_ledgers"]) == {
        "scope",
        "first_complete_observation",
        "second_complete_observation",
    }
    assert record["phase_read_ledgers"]["second_complete_observation"] == record[
        "read_ledger"
    ]
    assert sum(item["role"] == "git_shallow" for item in file_reads) == 1
    assert sum(item["role"] == "git_shallow_terminal" for item in file_reads) == 1
    assert record["commit_consistency"]["toctou_eliminated"] is False

    if method == "soar":
        assert source["gitlinks"] == []
        assert source["gitlink_worktree"] == []
        assert source["root_license_paths"] == ["LICENSE.md"]
    else:
        expected_gitlinks = config["source"]["gitlinks"]
        assert source["gitlinks"] == expected_gitlinks
        assert len(source["gitlink_worktree"]) == 7
        assert {item["path"] for item in source["gitlink_worktree"]} == {
            item["path"] for item in expected_gitlinks
        }
        assert all(
            item["kind"] == "empty-directory" and item["entry_count"] == 0
            for item in source["gitlink_worktree"]
        )
        assert all(
            item["gitlink_oid"]
            == {link["path"]: link["object_oid"] for link in expected_gitlinks}[
                item["path"]
            ]
            for item in source["gitlink_worktree"]
        )
        assert record["validation"]["gitlink_descendant_entries_not_opened"] is True
        assert record["validation"]["gitlink_descendant_leaf_bytes_not_read"] is True
        assert not any(
            "submodule" in argument
            for process in record["read_ledger"]["git_subprocesses"]
            for argument in process["argv"]
        )
        assert not any(
            any(
                item["path"] == link["path"]
                or item["path"].startswith(link["path"] + "/")
                for link in expected_gitlinks
            )
            for item in file_reads
        )


@pytest.mark.parametrize(
    ("method", "other_method"), [("soar", "nvarc"), ("nvarc", "soar")]
)
def test_cross_method_config_is_rejected_before_output_creation(
    method: str,
    other_method: str,
    batch_d_tmp_roots: dict[str, Path],
) -> None:
    output = batch_d_tmp_roots[method] / "cross-config-must-not-exist"
    completed = run_launcher(method, output, config=PROFILES[other_method]["config"])
    assert completed.returncode == 2
    assert "production config path must equal" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("method", ["soar", "nvarc"])
def test_wrong_operator_manifest_digest_is_rejected_before_output_creation(
    method: str, batch_d_tmp_roots: dict[str, Path]
) -> None:
    output = batch_d_tmp_roots[method] / "wrong-digest-must-not-exist"
    completed = run_launcher(
        method, output, expected_manifest_sha256="0" * 64
    )
    assert completed.returncode == 2
    assert "operator-supplied runner-manifest SHA-256 mismatch" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("method", ["soar", "nvarc"])
def test_unisolated_launcher_is_rejected_before_output_creation(
    method: str, batch_d_tmp_roots: dict[str, Path]
) -> None:
    output = batch_d_tmp_roots[method] / "unisolated-must-not-exist"
    completed = run_launcher(method, output, isolated=False)
    assert completed.returncode == 2
    assert "canonical direct-source script entry" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("method", ["soar", "nvarc"])
def test_direct_auditor_without_verified_context_is_rejected_before_output(
    method: str, batch_d_tmp_roots: dict[str, Path]
) -> None:
    profile = PROFILES[method]
    output = batch_d_tmp_roots[method] / "direct-auditor-must-not-exist"
    completed = subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-B",
            "-S",
            str(AUDITOR),
            "--config",
            str(profile["config"]),
            "--runner-manifest",
            str(profile["manifest"]),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        env=launcher_environment(),
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "must enter through the verified launcher" in completed.stderr
    assert not output.exists()


def tree_record(
    mode: str, object_type: str, oid: str, path: str | bytes
) -> bytes:
    encoded_path = path.encode("utf-8") if isinstance(path, str) else path
    return (
        f"{mode} {object_type} {oid}\t".encode("ascii") + encoded_path + b"\0"
    )


def test_batch_d_tree_parser_accepts_exact_nvarc_gitlinks() -> None:
    auditor = load_auditor()
    config = json.loads(PROFILES["nvarc"]["config"].read_text(encoding="utf-8"))
    expected_gitlinks = config["source"]["gitlinks"]
    payload = tree_record("100644", "blob", "a" * 40, ".gitmodules")
    for item in expected_gitlinks:
        payload += tree_record(
            item["mode"], item["object_type"], item["object_oid"], item["path"]
        )

    blobs, gitlinks, entries = auditor._parse_batch_d_tree(payload)
    assert blobs == {
        ".gitmodules": {
            "path": ".gitmodules",
            "mode": "100644",
            "blob_oid": "a" * 40,
        }
    }
    assert gitlinks == {item["path"]: item for item in expected_gitlinks}
    assert len(entries) == 8
    assert entries == sorted(entries, key=lambda item: item["path"])


@pytest.mark.parametrize(
    ("case", "payload"),
    [
        (
            "unknown-mode",
            tree_record("120000", "blob", "a" * 40, "unsafe-link"),
        ),
        (
            "unknown-type",
            tree_record("100644", "tree", "a" * 40, "nested"),
        ),
        (
            "duplicate-path",
            tree_record("100644", "blob", "a" * 40, "same.py")
            + tree_record("100644", "blob", "b" * 40, "same.py"),
        ),
        (
            "non-utf8-path",
            tree_record("100644", "blob", "a" * 40, b"bad-\xff.py"),
        ),
    ],
)
def test_batch_d_tree_parser_rejects_unsupported_or_ambiguous_entries(
    case: str, payload: bytes
) -> None:
    del case
    auditor = load_auditor()
    with pytest.raises(ValueError):
        auditor._parse_batch_d_tree(payload)


@pytest.mark.parametrize("method", ["soar", "nvarc"])
def test_failure_record_never_claims_solver_evidence(method: str) -> None:
    auditor = load_auditor()
    profile = auditor.PROFILES[PROFILES[method]["manifest_relative"]]
    record = auditor.failure_record(
        profile,
        "failed-run",
        ValueError("injected Batch D failure"),
        auditor.core.ReadLedger({}),
    )
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["solver_gate_passed"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert "predictions" not in record
    assert "results" not in record
    assert record["controls"]["network_used"] is None
    assert record["controls"]["gpu_used"] is None
    assert record["controls"]["solver_executed"] is None
    assert record["controls"]["solver_prediction_produced"] is False
    assert record["controls"]["measurement_status"] == (
        "unknown-after-audit-failure"
    )
