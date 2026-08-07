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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_2d_ngpt_gates.py"
CONFIG = ROOT / "configs" / "ngpt2d_gate_v1.json"
EXPECTED_FULL_CHAIN_LINES = {
    "test_solution_to_test_sample_output_lines": [113, 122],
    "get_color_perm_sample_output_lines": [132],
    "evaluation_solution_to_eval_train_lines": [201],
    "evaluation_solution_to_eval_test_lines": [202],
    "eval_train_and_eval_test_samples_lines": [223],
    "eval_solution_color_perm_lines": [225],
    "eval_aug_color_dataset_write_lines": [230],
    "tune_train_dataset_constructor_lines": [268],
    "tune_valid_dataset_constructor_lines": [269],
    "arc_eval_dataset_eval_aug_color_lines": [416],
    "arc_eval_dataset_eval_aug_color_use_lines": [465],
    "arc_valid_dataset_eval_aug_color_lines": [538],
    "arc_valid_dataset_eval_test_lines": [504],
    "run_model_update_lines": [1793],
    "run_train_model_lines": [1801],
    "arc_valid_sample_output_read_lines": [564],
    "arc_valid_output_batch_value_lines": [599, 624, 654, 673],
    "arc_model_prepare_batch_output_lines": [1114],
    "arc_model_forward_output_loss_lines": [1173],
    "valid_epoch_return_loss_model_lines": [1293],
    "valid_epoch_output_metric_helper_lines": [1299, 1303, 1306, 1309, 1312],
    "metric_helper_output_read_lines": [1351, 1440, 1486],
    "metric_helper_output_loss_lines": [1420, 1471, 1526],
    "metric_helper_output_accuracy_lines": [1426, 1476, 1531],
}


def load_auditor():
    spec = importlib.util.spec_from_file_location("ngpt2d_gate_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(
    output_directory: Path, config: Path = CONFIG
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
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
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_static_gate_audit_passes_but_method_remains_blocked(tmp_path: Path) -> None:
    output = tmp_path / "ngpt2d-gate"
    completed = run_audit(output)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))

    assert record["status"] == "passed"
    assert record["method_gate_status"] == "blocked"
    assert record["counted_toward_smoke"] is False
    assert record["solver_prediction_produced"] is False
    assert record["strict_runtime_promoted"] is False
    assert record["performance_table_eligible"] is False
    assert record["solver_gate_passed"] is False
    assert record["fairness"]["evidence_scope"] == "blocker_audit"
    assert record["gate_summary"] == {"blocked": 7, "passed": 1}
    assert "results" not in record
    assert all(value is True for value in record["validation"].values())
    assert all(value is False for value in record["controls"].values())

    source = record["source"]
    assert source["tracked_file_count"] == 10
    assert len(source["tracked_file_inventory"]) == 10
    assert source["total_bytes"] == 1_091_417
    assert source["tree_sha256"] == (
        "05bb50747e8cf0b98c149f75f65dee8a37bde3b7fe5c1926ad661bf1513e7c21"
    )
    assert source["external_sources_prefixed_tree_sha256"] == (
        "f7d595edbc89619d83f1570532ff5ed58f155accac1dd53a6d11ea268a04d1dd"
    )
    assert source["historical_four_python_file_tree_sha256"] == (
        "0576711088613bfb292caf1538e27db4c1137332494f9f59f3ba1fa09be0aff9"
    )
    assert "source_root_relative_path" in source["tree_digest_algorithm"]

    filesystem = source["filesystem_inventory"]
    assert filesystem["closed_world"] is True
    assert filesystem["filesystem_has_ignored_cache"] is True
    assert filesystem["ignored_cache_bytes_read"] is False
    assert filesystem["ignored_cache_executable_content_trusted"] is False
    assert filesystem["runtime_source_tree_approved"] is False
    assert [
        (item["path"], item["bytes"])
        for item in filesystem["ignored_cache_inventory"]
    ] == [
        ("cfg/__pycache__/cfg_064.cpython-310.pyc", 2044),
        ("code/__pycache__/064.cpython-310.pyc", 44484),
    ]
    assert all(item["sha256"] is None for item in filesystem["ignored_cache_inventory"])
    assert all(item["bytes_read"] is False for item in filesystem["ignored_cache_inventory"])
    assert all(
        item["manifest_included"] is False
        for item in filesystem["ignored_cache_inventory"]
    )
    assert record["ignored_cache_bytes_read"] is False
    assert record["ignored_cache_executable_content_trusted"] is False
    assert record["filesystem_has_ignored_cache"] is True
    assert record["runtime_source_tree_approved"] is False

    assert all(
        item["present"] is False
        and item["bytes_read"] is False
        and item["sha256"] is None
        for item in record["forbidden_artifact_observations"]
    )
    declared_artifacts = {
        item["declared_path"] for item in record["forbidden_artifact_observations"]
    }
    assert {
        "/usr/paper-assets/arc/sources/2d-ngpt/re-arc/gen10000/tasks",
        "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/fixed_size.pkl",
        "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/arc-agi_training_solutions.json",
        "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    }.issubset(declared_artifacts)
    label = record["label_firewall_gate"]
    assert label["flow_detected"] is True
    assert label["exact_full_chain_lines_match"] is True
    assert {
        key: label[key] for key in EXPECTED_FULL_CHAIN_LINES
    } == EXPECTED_FULL_CHAIN_LINES
    assert all(
        line is not None
        for line in label["expected_tune_flow_first_call_lines"].values()
    )

    runtime = record["runtime_portability_gate"]
    assert runtime["checkpoint_load_detected"] is True
    assert runtime["torch_load_calls"] == [
        {
            "line": 1595,
            "map_location_literals": ["cpu"],
            "weights_only_declared": False,
        }
    ]
    assert runtime["hardcoded_cuda_detected"] is True
    assert runtime["global_cfg_blocker_detected"] is True
    global_scopes = {item["scope"] for item in runtime["global_cfg_scopes"]}
    assert {"ARCModel.forward", "add_validation_data"}.issubset(global_scopes)

    prior = record["prior_architecture_evidence"]
    assert prior["classification"] == "component-evidence-only"
    assert prior["component_smoke_counted"] is True
    assert prior["solver_prediction_smoke"] is False
    assert prior["solver_evidence"] is False

    cache_paths = {
        "external/ARC-AGI-Challenge-2024/cfg/__pycache__/cfg_064.cpython-310.pyc",
        "external/ARC-AGI-Challenge-2024/code/__pycache__/064.cpython-310.pyc",
    }
    manifested = {item["path"] for item in record["evidence_manifest"]}
    assert not cache_paths & manifested
    for item in record["evidence_manifest"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    replay_output = tmp_path / "ngpt2d-gate-replay"
    replay_completed = run_audit(replay_output)
    assert replay_completed.returncode == 0
    replay = json.loads((replay_output / "run.json").read_text(encoding="utf-8"))
    assert replay["observation_digest_sha256"] == record["observation_digest_sha256"]


def test_nonempty_output_directory_is_rejected_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    completed = run_audit(output)
    assert completed.returncode == 2
    assert "output directory must not already exist" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_even_empty_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    completed = run_audit(output)
    assert completed.returncode == 2
    assert "output directory must not already exist" in completed.stderr
    assert not list(output.iterdir())


def test_output_directory_rejects_leaf_and_parent_redirects(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    leaf_link = tmp_path / "leaf-link"
    leaf_link.symlink_to(target, target_is_directory=True)
    leaf = run_audit(leaf_link)
    assert leaf.returncode == 2
    assert "leaf must not be a symlink" in leaf.stderr
    assert not (target / "run.json").exists()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    parent = run_audit(parent_link / "new-output")
    assert parent.returncode == 2
    assert "symlink/non-directory component" in parent.stderr
    assert not (real_parent / "new-output").exists()

    regular_parent = tmp_path / "regular-file"
    regular_parent.write_text("not a directory", encoding="utf-8")
    non_directory = run_audit(regular_parent / "new-output")
    assert non_directory.returncode == 2
    assert "symlink/non-directory component" in non_directory.stderr


def test_config_and_output_paths_cannot_enter_sensitive_source_tree(
    tmp_path: Path,
) -> None:
    source = ROOT / "external" / "ARC-AGI-Challenge-2024"
    source_output = source / "must-not-create-gate-output"
    assert not source_output.exists()
    output_attempt = run_audit(source_output)
    assert output_attempt.returncode == 2
    assert "must remain outside source" in output_attempt.stderr
    assert not source_output.exists()

    config_attempt = run_audit(tmp_path / "unused-output", source / "README.md")
    assert config_attempt.returncode == 2
    assert "config path must remain outside source" in config_attempt.stderr


def test_config_snapshot_mutation_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source"]["expected_files"][0]["sha256"] = "0" * 64
    bad_config = tmp_path / "bad-config.json"
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "failed-gate"
    completed = run_audit(output, bad_config)
    assert completed.returncode == 1
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["method_gate_status"] == "blocked"
    assert record["error"]["type"] == "ValueError"
    assert "expected_files mismatch" in record["error"]["message"]
    assert "evidence_manifest" not in record
    assert all(value is False for value in record["controls"].values())


@pytest.mark.parametrize(
    ("field_path", "expected_message"),
    [
        (
            ("source", "source_lock_sha256"),
            "config.source.source_lock_sha256 mismatch",
        ),
        (
            ("prior_architecture_evidence", "run_sha256"),
            "config.prior_architecture_evidence.run_sha256 mismatch",
        ),
        (
            ("prior_architecture_evidence", "runner_sha256"),
            "config.prior_architecture_evidence.runner_sha256 mismatch",
        ),
    ],
)
def test_config_cannot_mutate_locked_evidence_hashes(
    field_path: tuple[str, str], expected_message: str
) -> None:
    auditor = load_auditor()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    section, field = field_path
    config[section][field] = "0" * 64
    with pytest.raises(ValueError, match=expected_message):
        auditor.validate_config(config)


def test_prepared_layout_artifact_lists_are_exactly_locked() -> None:
    auditor = load_auditor()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "/usr/paper-assets/arc/sources/2d-ngpt/re-arc/gen10000/tasks" in config[
        "forbidden_artifacts"
    ]["re_arc_paths"]
    assert (
        "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/fixed_size.pkl"
        in config["forbidden_artifacts"]["fixed_size_paths"]
    )
    assert sum(
        path.startswith(
            "/usr/paper-assets/arc/sources/2d-ngpt/input/arc-prize-2024/"
        )
        for path in config["forbidden_artifacts"]["solution_paths"]
    ) == 2
    mutated = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutated["forbidden_artifacts"]["re_arc_paths"].remove(
        "/usr/paper-assets/arc/sources/2d-ngpt/re-arc/gen10000/tasks"
    )
    with pytest.raises(ValueError, match="re_arc_paths mismatch"):
        auditor.validate_config(mutated)


def test_config_cannot_redirect_forbidden_or_retained_paths(tmp_path: Path) -> None:
    auditor = load_auditor()
    for mutate, expected_message in (
        (
            lambda config: config["forbidden_artifacts"]["solution_paths"].append(
                str(tmp_path / "solution.json")
            ),
            "solution_paths mismatch",
        ),
        (
            lambda config: config["source"]["expected_files"].append(
                {"path": "../../etc/passwd", "bytes": 1, "sha256": "0" * 64}
            ),
            "expected_files mismatch",
        ),
    ):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutate(config)
        with pytest.raises(ValueError, match=expected_message):
            auditor.validate_config(config)


def test_forbidden_material_aborts_before_retained_source_reads() -> None:
    auditor = load_auditor()
    observations = [
        {
            "kind": "solution",
            "path": "/not/opened/solution.json",
            "present": True,
        }
    ]
    with pytest.raises(RuntimeError, match="aborting before retained-source reads"):
        auditor.assert_forbidden_material_absent(observations)


def test_forbidden_observation_never_opens_solution_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    solution = tmp_path / "arc-agi_evaluation_solutions.json"
    solution.write_text("must-not-read", encoding="utf-8")

    def forbidden_open(*args, **kwargs):
        raise AssertionError("metadata-only observation attempted to open bytes")

    monkeypatch.setattr(auditor.os, "open", forbidden_open)
    observation = auditor.observe_forbidden_path(solution)
    assert observation["present"] is True
    assert observation["file_type"] == "regular_file"
    with pytest.raises(RuntimeError, match="aborting before retained-source reads"):
        auditor.assert_forbidden_material_absent([observation])


def test_forbidden_observation_rejects_symlink_parent_without_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "exp_54.pt").write_text("must-not-read", encoding="utf-8")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("metadata-only observation attempted to open bytes")

    monkeypatch.setattr(auditor.os, "open", forbidden_open)
    observation = auditor.observe_forbidden_path(linked_parent / "exp_54.pt")
    assert observation["present"] is True
    assert observation["file_type"] == "forbidden_parent_component"
    assert observation["forbidden_parent_component"]["file_type"] == "symlink"


def test_metadata_inventory_rejects_unknown_file_before_source_read(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    root = tmp_path / "source"
    root.mkdir()
    (root / "safe.py").write_text("x = 1\n", encoding="utf-8")
    (root / "z-extra.json").write_text("must-not-read", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown file.*z-extra.json"):
        auditor.metadata_only_source_inventory(root, ["safe.py"], [], [], [])


def test_metadata_inventory_rejects_unknown_directory_before_source_read(
    tmp_path: Path,
) -> None:
    auditor = load_auditor()
    root = tmp_path / "source"
    root.mkdir()
    (root / "safe.py").write_text("x = 1\n", encoding="utf-8")
    (root / "checkpoint").mkdir()
    with pytest.raises(ValueError, match="unknown directory.*checkpoint"):
        auditor.metadata_only_source_inventory(root, ["safe.py"], [], [], [])


def test_metadata_inventory_rejects_symlink_before_source_read(tmp_path: Path) -> None:
    auditor = load_auditor()
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("must-not-read", encoding="utf-8")
    (root / "safe.py").symlink_to(outside)
    with pytest.raises(ValueError, match="contains symlink"):
        auditor.metadata_only_source_inventory(root, ["safe.py"], [], [], [])


def test_metadata_inventory_binds_ignored_cache_size(tmp_path: Path) -> None:
    auditor = load_auditor()
    root = tmp_path / "source"
    cache = root / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (root / "safe.py").write_text("x = 1\n", encoding="utf-8")
    (cache / "safe.pyc").write_bytes(b"123")
    with pytest.raises(ValueError, match="cache size mismatch"):
        auditor.metadata_only_source_inventory(
            root,
            ["safe.py"],
            ["pkg"],
            ["pkg/__pycache__"],
            [{"path": "pkg/__pycache__/safe.pyc", "bytes": 4}],
        )


def test_verified_reader_rejects_leaf_and_parent_symlinks(tmp_path: Path) -> None:
    auditor = load_auditor()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    target = real_parent / "source.py"
    target.write_text("x = 1\n", encoding="utf-8")
    leaf_link = tmp_path / "leaf.py"
    leaf_link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        auditor.read_regular_bytes(leaf_link)

    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        auditor.read_regular_bytes(parent_link / "source.py")


def test_verified_reader_rejects_leaf_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auditor = load_auditor()
    source = tmp_path / "source.py"
    source.write_text("safe = True\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("must_not_read = True\n", encoding="utf-8")
    real_open = auditor.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == source.name and kwargs.get("dir_fd") is not None and not swapped:
            source.unlink()
            source.symlink_to(outside)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(auditor.os, "open", swapping_open)
    with pytest.raises(OSError):
        auditor.read_regular_bytes(source)
    assert swapped is True


def test_auditor_has_no_upstream_runtime_or_dynamic_code_hooks() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported: set[str] = set()
    calls: list[ast.Call] = []
    sys_path_mutations: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            calls.append(node)
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "sys"
                and child.attr == "path"
                for target in targets
                for child in ast.walk(target)
            ):
                sys_path_mutations.append(node)
    forbidden_roots = {
        "torch",
        "requests",
        "socket",
        "importlib",
        "marshal",
        "ARC_AGI_Challenge_2024",
    }
    assert not {name.split(".", 1)[0] for name in imported} & forbidden_roots
    forbidden_calls = {
        node.func.id
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id in {"compile", "eval", "exec", "__import__"}
    }
    assert not forbidden_calls
    assert not sys_path_mutations
    assert not any(
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr in {"chdir", "fchdir"}
        for node in calls
    )
