#!/usr/bin/env python3
"""Reproduce the metadata-first TinyRecursiveModels blocker gate.

The auditor never imports or executes upstream code, opens bundled ARC JSON or
image leaves, loads a checkpoint, initializes a model/provider/GPU, or uses the
network.  A passing report means that the locked static observations and
blockers were reproduced.  It is not a solver smoke, prediction, or score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time
import types
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
AUDITOR_PATH = ROOT / "scripts" / "audit_trm_gates.py"
SUPPORT_PATH = ROOT / "scripts" / "audit_batch_c_static_gates.py"
CANONICAL_CONFIG_RELATIVE = "configs/trm_gate_v1.json"
SOURCE_LOCK_RELATIVE = "configs/source_locks.json"
RUNNER_MANIFEST_RELATIVE = "configs/trm_gate_runner_manifest_v1.json"
CONFIG_ID = "trm-source-artifact-dataset-label-resource-gate-v1"
METHOD_ID = "tiny-recursive-models"
SCOPE = "source-artifact-dataset-label-resource-gate-audit-only"
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "3f8846344565228e43b8a5dd62c7a7eb0bceb84139b4693cee3423a47313afc9"
)
EXPECTED_SOURCE_LOCK_SHA256 = (
    "a785b89743dc06c1296dbfa9691081035bd062ae7f97c5d80c9cfbb38f76a5b4"
)
EXPECTED_SUPPORT_SHA256 = (
    "8860877257cf2864ddf8304fdef407d76de72339b6aba9d47391db5a57c7626e"
)
TEST_OUTPUT_ROOT = Path("/tmp/arc-agi-eval-trm-tests")
MAX_AUDITOR_BYTES = 4 * 1024 * 1024
MAX_BOUND_METADATA_BYTES = 4 * 1024 * 1024
core: Any = None


def _require_file_safety_flags() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    if missing:
        raise RuntimeError(
            "TRM auditor requires non-degrading file-safety flags: "
            + ", ".join(missing)
        )


def _bootstrap_stat_signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _bootstrap_secure_read(path: Path, *, max_bytes: int, field: str) -> bytes:
    """Read one regular file without following its final component.

    This deliberately uses only the Python standard library so the shared
    support module can be authenticated before any of its bytes are executed.
    """

    if not path.is_absolute():
        raise ValueError(f"{field} path must be absolute")
    _require_file_safety_flags()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{field} must be a single-link regular file")
        if before.st_size > max_bytes:
            raise ValueError(f"{field} exceeds the byte limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{field} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{field} grew while reading")
        after = os.fstat(descriptor)
        if _bootstrap_stat_signature(before) != _bootstrap_stat_signature(after):
            raise ValueError(f"{field} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _bootstrap_strict_json(payload: bytes, field: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant forbidden in {field}: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {field}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from error


def _execute_verified_support(payload: bytes) -> types.ModuleType:
    digest = hashlib.sha256(payload).hexdigest()
    module = types.ModuleType(f"_verified_batch_c_support_{digest}")
    module.__file__ = str(SUPPORT_PATH)
    module.__package__ = "scripts"
    module.__dict__["__verified_source_sha256__"] = digest
    code = compile(payload, str(SUPPORT_PATH), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def bootstrap_verified_runtime(
    config_path: Path,
    runner_manifest_path: Path,
) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, dict[str, Any]]:
    """Consume launcher-verified bytes before executing the support module."""

    global core
    canonical_config = ROOT / CANONICAL_CONFIG_RELATIVE
    canonical_manifest = ROOT / RUNNER_MANIFEST_RELATIVE
    if os.path.abspath(config_path) != os.path.abspath(canonical_config):
        raise ValueError(f"production config path must equal {CANONICAL_CONFIG_RELATIVE}")
    if os.path.abspath(runner_manifest_path) != os.path.abspath(canonical_manifest):
        raise ValueError(
            f"production runner manifest must equal {RUNNER_MANIFEST_RELATIVE}"
        )
    context = globals().get("__verified_runner_manifest_context__")
    if not isinstance(context, dict):
        raise ValueError("TRM production audit must enter through the verified launcher")
    manifest_payload = context.get("manifest_payload")
    payloads = context.get("member_payloads")
    executed_auditor_sha256 = context.get("executed_auditor_sha256")
    if not isinstance(manifest_payload, bytes) or not isinstance(payloads, dict):
        raise ValueError("TRM verified-launcher context is incomplete")
    manifest = _bootstrap_strict_json(manifest_payload, "TRM runner manifest")
    if not isinstance(manifest, dict):
        raise ValueError("TRM runner manifest must be an object")
    expected_paths = {
        "scripts/launch_trm_gate.py",
        "scripts/audit_trm_gates.py",
        "scripts/audit_batch_c_static_gates.py",
        CANONICAL_CONFIG_RELATIVE,
        SOURCE_LOCK_RELATIVE,
    }
    members = manifest.get("members")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("manifest_id") != "trm-gate-runner-manifest-v1"
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("member_count") != 5
        or not isinstance(members, list)
        or len(members) != 5
        or {item.get("path") for item in members if isinstance(item, dict)}
        != expected_paths
        or set(payloads) != expected_paths
    ):
        raise ValueError("TRM runner manifest closure mismatch")
    by_path = {item["path"]: item for item in members}
    for path, payload in payloads.items():
        member = by_path[path]
        if (
            not isinstance(payload, bytes)
            or member.get("bytes") != len(payload)
            or member.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"TRM runner member mismatch: {path}")
    if context.get("manifest_path") != RUNNER_MANIFEST_RELATIVE:
        raise ValueError("TRM verified launcher used a noncanonical manifest")
    if context.get("manifest_sha256") != hashlib.sha256(manifest_payload).hexdigest():
        raise ValueError("TRM runner-manifest digest context mismatch")

    config_payload = payloads[CANONICAL_CONFIG_RELATIVE]
    auditor_payload = payloads["scripts/audit_trm_gates.py"]
    support_payload = payloads["scripts/audit_batch_c_static_gates.py"]
    source_lock_payload = payloads[SOURCE_LOCK_RELATIVE]
    auditor_digest = hashlib.sha256(auditor_payload).hexdigest()
    support_digest = hashlib.sha256(support_payload).hexdigest()
    if executed_auditor_sha256 != auditor_digest:
        raise ValueError("executed TRM auditor bytes differ from the runner manifest")
    if context.get("launcher_source_execution") != {
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
    }:
        raise ValueError("TRM launcher was not entered as canonical direct source")
    if (
        context.get("operator_supplied_manifest_sha256")
        != hashlib.sha256(manifest_payload).hexdigest()
    ):
        raise ValueError("operator-supplied TRM runner-manifest digest mismatch")
    if support_digest != EXPECTED_SUPPORT_SHA256:
        raise ValueError("TRM static-audit support SHA-256 mismatch")

    verified_core = _execute_verified_support(support_payload)
    if verified_core.__dict__.get("__verified_source_sha256__") != support_digest:
        raise RuntimeError("verified support execution provenance was lost")
    core = verified_core
    config = _bootstrap_strict_json(config_payload, "TRM gate config")
    config = validate_config(config)
    if by_path[CANONICAL_CONFIG_RELATIVE].get("canonical_sha256") != (
        core.canonical_sha256(config)
    ):
        raise ValueError("TRM config canonical digest differs from runner manifest")
    return (
        config,
        config_payload,
        auditor_payload,
        support_payload,
        source_lock_payload,
        manifest,
    )


def _text(payloads: dict[str, bytes], path: str) -> str:
    return core.source_text(payloads[path])


def _all_calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        call for call in _all_calls(tree) if core.dotted_name(call.func) == name
    ]


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    return ast.literal_eval(node)


def _yaml_scalar(text: str, key: str) -> Any:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+?)\s*(?:#.*)?$", text)
    if match is None:
        raise ValueError(f"missing YAML scalar: {key}")
    raw = match.group(1).strip()
    if raw in {"True", "true"}:
        return True
    if raw in {"False", "false"}:
        return False
    if re.fullmatch(r"-?[0-9]+", raw):
        return int(raw)
    return raw.strip("'\"")


def _requirement_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _requirement_name(line: str) -> str:
    return line.split("==", 1)[0].strip().lower().replace("_", "-")


def _annotated_default(class_node: ast.ClassDef, name: str) -> Any:
    for node in class_node.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return _literal(node.value)
    raise ValueError(f"missing annotated default: {class_node.name}.{name}")


def _annotated_fields(class_node: ast.ClassDef) -> list[str]:
    return [
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


def _keyword_state(call: ast.Call, name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        try:
            return ast.literal_eval(keyword.value)
        except (TypeError, ValueError):
            return "<dynamic>"
    return "absent"


def _torch_load_records(tree: ast.AST) -> list[dict[str, Any]]:
    return [
        {
            "lineno": call.lineno,
            "map_location": _keyword_state(call, "map_location"),
            "weights_only": _keyword_state(call, "weights_only"),
        }
        for call in _calls_named(tree, "torch.load")
    ]


def _torch_save_records(tree: ast.AST) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for call in _calls_named(tree, "torch.save"):
        if not call.args:
            raise ValueError("torch.save call has no payload")
        expression = ast.unparse(call.args[0])
        if expression == "train_state.model.state_dict()":
            purpose = "model-checkpoint"
        elif expression == "save_preds":
            purpose = "configured-evaluation-outputs"
        else:
            purpose = "unclassified"
        records.append(
            {
                "lineno": call.lineno,
                "payload_expression": expression,
                "purpose": purpose,
            }
        )
    return sorted(records, key=lambda item: item["lineno"])


def _checkpoint_asset_count(assets: Any) -> int:
    if not isinstance(assets, dict):
        return 0
    count = 0
    for name, value in assets.items():
        kind = value.get("kind") if isinstance(value, dict) else None
        if "checkpoint" in str(name).lower() or kind == "checkpoint":
            count += 1
    return count


def analyze_trm(
    parsed: dict[str, ast.Module],
    payloads: dict[str, bytes],
    asset_status: dict[str, Any],
    prior_report: dict[str, Any],
) -> dict[str, Any]:
    readme = _text(payloads, "README.md")
    license_text = _text(payloads, "LICENSE")
    builder_text = _text(payloads, "dataset/build_arc_dataset.py")
    puzzle_text = _text(payloads, "puzzle_dataset.py")
    loss_text = _text(payloads, "models/losses.py")
    trm_text = _text(payloads, "models/recursive_reasoning/trm.py")
    evaluator_text = _text(payloads, "evaluators/arc.py")
    pretrain_text = _text(payloads, "pretrain.py")
    config_text = _text(payloads, "config/cfg_pretrain.yaml")
    sparse_text = _text(payloads, "models/sparse_embedding.py")
    utility_text = _text(payloads, "utils/functions.py")

    score_match = re.search(
        r"([0-9]+)% on ARC-AGI-1 and ([0-9]+)% on ARC-AGI-2", readme
    )
    if score_match is None:
        raise ValueError("README score statement is missing")

    evaluator_init = core.function_def(
        core.class_def(parsed["evaluators/arc.py"], "ARC"), "__init__"
    )
    submission_default = _literal(
        core.argument_default(evaluator_init, "submission_K")
    )
    pass_ks_default = list(
        _literal(core.argument_default(evaluator_init, "pass_Ks"))
    )
    aggregated_default = _literal(
        core.argument_default(evaluator_init, "aggregated_voting")
    )

    pretrain_tree = parsed["pretrain.py"]
    torch_load_records = _torch_load_records(pretrain_tree)
    torch_save_records = _torch_save_records(pretrain_tree)
    pretrain_config_class = core.class_def(pretrain_tree, "PretrainConfig")
    train_state_class = core.class_def(pretrain_tree, "TrainState")
    train_state_fields = _annotated_fields(train_state_class)
    model_checkpoint_payloads = [
        item["payload_expression"]
        for item in torch_save_records
        if item["purpose"] == "model-checkpoint"
    ]

    evaluator_class = core.class_def(parsed["evaluators/arc.py"], "ARC")
    evaluator_state_fields = sorted(
        {
            target.attr
            for node in ast.walk(core.function_def(evaluator_class, "__init__"))
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr in {"_local_hmap", "_local_preds"}
        }
    )
    serialization_names = {
        "state_dict",
        "load_state_dict",
        "__getstate__",
        "__setstate__",
    }
    evaluator_serialization_methods = sorted(
        node.name
        for node in evaluator_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in serialization_names
    )

    cuda_transfers = [
        call
        for call in _all_calls(pretrain_tree)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "cuda"
    ]
    cuda_contexts = [
        call
        for call in _calls_named(pretrain_tree, "torch.device")
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "cuda"
    ]

    arc_hf_calls = _calls_named(
        parsed["dataset/build_arc_dataset.py"], "hf_hub_download"
    )
    non_arc_hf_calls = []
    for path in (
        "dataset/build_maze_dataset.py",
        "dataset/build_sudoku_dataset.py",
    ):
        non_arc_hf_calls.extend(_calls_named(parsed[path], "hf_hub_download"))

    requirements = _requirement_lines(_text(payloads, "requirements.txt"))
    pinned = _requirement_lines(_text(payloads, "specific_requirements.txt"))
    pinned_names = {_requirement_name(line) for line in pinned}
    torch_requirement = next(
        line for line in pinned if _requirement_name(line) == "torch"
    )
    pinned_torch_version = torch_requirement.split("==", 1)[1]

    inner_forward = core.function_def(
        core.class_def(
            parsed["models/recursive_reasoning/trm.py"],
            "TinyRecursiveReasoningModel_ACTV1_Inner",
        ),
        "forward",
    )
    inner_strings = core.string_constants(inner_forward)

    concept_lines = [
        line.strip() for line in readme.splitlines() if "concept" in line.lower()
    ]
    concept_explanation_terms = re.compile(
        r"\b(source|provenance|derived|generated|origin|license)\b", re.I
    )
    citation_match = re.search(
        r"@misc\{jolicoeurmartineau2025[\s\S]{0,500}?year=\{([0-9]{4})\}",
        readme,
    )
    if citation_match is None:
        raise ValueError("TRM paper citation year is missing")
    created_at = asset_status.get("created_at")
    if not isinstance(created_at, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T.*", created_at
    ):
        raise ValueError("TRM asset-status creation time is missing")
    assets = asset_status.get("assets", {})
    arch_text = _text(payloads, "config/arch/trm.yaml")
    builder_tree = parsed["dataset/build_arc_dataset.py"]
    data_process_config = core.class_def(builder_tree, "DataProcessConfig")
    arc_max_grid_assign = next(
        node
        for node in builder_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ARCMaxGridSize"
            for target in node.targets
        )
    )
    arc_max_grid_size = _literal(arc_max_grid_assign.value)
    epochs = _yaml_scalar(config_text, "epochs")
    eval_interval = _yaml_scalar(config_text, "eval_interval")
    default_evaluation_count = epochs // eval_interval

    return {
        "repository_archived_notice": "archive (make read-only)" in readme,
        "readme_citation_year": int(citation_match.group(1)),
        "asset_status_created_year": int(created_at[:4]),
        "official_arc_prize_entry_evidence_detected": (
            "official ARC Prize entry" in readme
            or "official ARC Prize submission" in readme
        ),
        "readme_scores_are_unverified_self_report": True,
        "reported_arc_agi_1_percent": int(score_match.group(1)),
        "reported_arc_agi_2_percent": int(score_match.group(2)),
        "readme_arc_gpu_count": 4
        if readme.count("assuming 4 H-100 GPUs") == 2
        else None,
        "readme_arc_gpu_model": "H-100"
        if "assuming 4 H-100 GPUs" in readme
        else None,
        "readme_arc_runtime_days_each": 3
        if readme.count("*Runtime:* ~3 days") == 2
        else None,
        "readme_separate_arc1_arc2_overlap_warning": (
            "ARC-AGI-2 training data contains some ARC-AGI-1 eval data" in readme
        ),
        "readme_concept_subset_provenance_explained": any(
            concept_explanation_terms.search(line) for line in concept_lines
        ),
        "paper_checkpoint_reference_detected": bool(
            re.search(r"\b(checkpoint|pretrained weights|model weights)\b", readme, re.I)
        ),
        "asset_status_total_asset_count": len(assets)
        if isinstance(assets, dict)
        else 0,
        "asset_status_checkpoint_count": _checkpoint_asset_count(assets),
        "arc_builder_reads_challenges": (
            'f"{config.input_file_prefix}_{subset_name}_challenges.json"'
            in builder_text
            and "puzzles = json.load(f)" in builder_text
        ),
        "arc_builder_reads_solutions": (
            'f"{config.input_file_prefix}_{subset_name}_solutions.json"'
            in builder_text
            and "sols = json.load(f)" in builder_text
        ),
        "arc_builder_injects_solution_into_test_output": (
            'puzzles[puzzle_id]["test"][idx]["output"] = sol_grid'
            in builder_text
        ),
        "arc_builder_missing_solution_dummy_output": (
            'example.setdefault("output", [[0]])' in builder_text
        ),
        "arc_builder_missing_solution_fails_closed": False
        if 'example.setdefault("output", [[0]])' in builder_text
        else None,
        "arc_builder_retains_test_puzzles_with_outputs": (
            "test_puzzles[name] = puzzle" in builder_text
        ),
        "arc_builder_serializes_test_labels": (
            'results["labels"].append(out)' in builder_text
            and 'f"{subset_name}__{k}.npy"' in builder_text
        ),
        "arc_builder_writes_test_puzzles": (
            'os.path.join(config.output_dir, "test_puzzles.json")'
            in builder_text
            and "json.dump(test_puzzles, f)" in builder_text
        ),
        "runtime_test_batches_include_labels": (
            '"labels": dataset["labels"][local_start: local_end]'
            in puzzle_text
        ),
        "loss_head_reads_current_labels": (
            'labels = new_carry.current_data["labels"]' in loss_text
        ),
        "primary_model_inner_reads_labels": "labels" in inner_strings,
        "evaluation_runner_passes_label_bearing_batch_to_model": (
            "carry=carry, batch=batch, return_keys=return_keys" in pretrain_text
            and "evaluator.update_batch(batch, preds)" in pretrain_text
        ),
        "eval_save_outputs_default": list(
            _annotated_default(pretrain_config_class, "eval_save_outputs")
        ),
        "evaluation_return_keys_include_configured_save_outputs": (
            "return_keys = set(config.eval_save_outputs)" in pretrain_text
        ),
        "evaluation_collects_configured_keys_from_full_batch": (
            "for collection in (batch, preds):" in pretrain_text
            and "if k in config.eval_save_outputs:" in pretrain_text
            and "save_preds.setdefault(k, [])" in pretrain_text
        ),
        "evaluation_persists_configured_outputs_with_torch_save": any(
            item["purpose"] == "configured-evaluation-outputs"
            for item in torch_save_records
        ),
        "evaluation_labels_can_be_persisted_via_eval_save_outputs": (
            '"labels": dataset["labels"][local_start: local_end]' in puzzle_text
            and "return_keys = set(config.eval_save_outputs)" in pretrain_text
            and "for collection in (batch, preds):" in pretrain_text
            and "if k in config.eval_save_outputs:" in pretrain_text
            and any(
                item["purpose"] == "configured-evaluation-outputs"
                for item in torch_save_records
            )
        ),
        "evaluation_mode_uses_fixed_max_steps": (
            "if self.training and (self.config.halt_max_steps > 1):" in trm_text
            and "halted = is_last_step" in trm_text
        ),
        "evaluator_required_outputs_include_labels": (
            "labels" in _literal(
                next(
                    node.value
                    for node in core.class_def(parsed["evaluators/arc.py"], "ARC").body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "required_outputs"
                        for target in node.targets
                    )
                )
            )
        ),
        "evaluator_reads_test_puzzles": (
            'os.path.join(data_path, "test_puzzles.json")' in evaluator_text
            and "self.test_puzzles = json.load(f)" in evaluator_text
        ),
        "evaluator_hashes_expected_output": (
            'label_hash = grid_hash(arc_grid_to_np(pair["output"]))'
            in evaluator_text
        ),
        "evaluator_submission_k_default": submission_default,
        "evaluator_pass_ks_default": pass_ks_default,
        "evaluator_aggregated_voting_default": aggregated_default,
        "evaluator_clears_predictions_when_aggregating": False
        if "if not self.aggregated_voting:" in evaluator_text
        else None,
        "evaluator_state_fields": evaluator_state_fields,
        "evaluator_serialization_methods": evaluator_serialization_methods,
        "evaluator_aggregated_voting_state_checkpointed": (
            bool(evaluator_serialization_methods)
            or any(
                "evaluator" in item["payload_expression"]
                or "_local_hmap" in item["payload_expression"]
                or "_local_preds" in item["payload_expression"]
                for item in torch_save_records
            )
        ),
        "evaluator_missing_prediction_continues": (
            'print (f"Puzzle {name} has no predictions.")' in evaluator_text
            and "if not len(p_map):" in evaluator_text
            and "continue" in evaluator_text
        ),
        "evaluator_single_prediction_is_duplicated": (
            "while len(pred_grids) < self.submission_K:" in evaluator_text
            and "pred_grids.append(pred_grids[0])" in evaluator_text
        ),
        "evaluator_explicit_final_tie_break": False
        if "p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)"
        in evaluator_text
        else None,
        "evaluator_writes_submission": (
            'os.path.join(save_path, "submission.json")' in evaluator_text
            and "json.dump(submission, f)" in evaluator_text
        ),
        "evaluator_gather_object_unconditional": (
            "dist.gather_object(" in evaluator_text
            and evaluator_text.index("dist.gather_object(")
            < evaluator_text.index("if rank != 0:")
        ),
        "distributed_init_requires_local_rank": (
            'if "LOCAL_RANK" in os.environ:' in pretrain_text
            and 'dist.init_process_group(backend="nccl")' in pretrain_text
        ),
        "torch_load_call_count": len(torch_load_records),
        "torch_load_calls": torch_load_records,
        "torch_save_call_count": len(torch_save_records),
        "torch_save_calls": torch_save_records,
        "checkpoint_load_resets_puzzle_embedding_on_shape_mismatch": (
            "Resetting puzzle embedding as shape is different" in pretrain_text
            and "torch.mean(puzzle_emb, dim=0, keepdim=True)" in pretrain_text
        ),
        "train_state_fields": train_state_fields,
        "checkpoint_model_payload_expressions": model_checkpoint_payloads,
        "checkpoint_save_model_state_only": model_checkpoint_payloads
        == ["train_state.model.state_dict()"],
        "checkpoint_save_includes_optimizer_rng_and_evaluator": any(
            any(term in expression for term in ("optimizer", "rng", "evaluator"))
            for expression in model_checkpoint_payloads
        ),
        "checkpoint_save_excludes_optimizer_state": (
            model_checkpoint_payloads == ["train_state.model.state_dict()"]
            and "optimizers" in train_state_fields
        ),
        "checkpoint_save_excludes_train_step": (
            model_checkpoint_payloads == ["train_state.model.state_dict()"]
            and "step" in train_state_fields
        ),
        "checkpoint_save_excludes_rng_state": (
            model_checkpoint_payloads == ["train_state.model.state_dict()"]
            and not any("rng" in item for item in model_checkpoint_payloads)
        ),
        "checkpoint_save_excludes_evaluator_state": (
            model_checkpoint_payloads == ["train_state.model.state_dict()"]
            and not evaluator_serialization_methods
        ),
        "puzzle_embedding_is_persistent_buffer": (
            "self.weights = nn.Buffer(" in sparse_text
            and "std=init_std), persistent=True" in sparse_text
        ),
        "hardcoded_cuda_tensor_transfer_call_count": len(cuda_transfers),
        "hardcoded_cuda_device_context_count": len(cuda_contexts),
        "wandb_init_call_count": len(_calls_named(pretrain_tree, "wandb.init")),
        "wandb_log_call_count": len(_calls_named(pretrain_tree, "wandb.log")),
        "wandb_log_code_call_count": len(
            _calls_named(pretrain_tree, "wandb.run.log_code")
        ),
        "wandb_finish_call_count": len(_calls_named(pretrain_tree, "wandb.finish")),
        "wandb_disabled_or_offline_guard_detected": any(
            marker in pretrain_text
            for marker in ("WANDB_MODE", 'mode="offline"', 'mode="disabled"')
        ),
        "arc_builder_huggingface_download_call_count": len(arc_hf_calls),
        "non_arc_builder_huggingface_download_call_count": len(non_arc_hf_calls),
        "dynamic_model_import_call_count": utility_text.count(
            "importlib.import_module(prefix + module_path)"
        ),
        "mixed_arc1_arc2_data_path_guard_detected": bool(
            re.search(r"raise|assert", builder_text)
            and re.search(r"training2.*evaluation|evaluation.*training2", builder_text)
        ),
        "training_sampler_uses_global_numpy_choice": (
            "np.random.choice(puzzle_size, append_size, replace=False)"
            in puzzle_text
            and "rng.integers(" in puzzle_text
        ),
        "training_sampler_uses_seeded_philox": (
            "np.random.Philox(seed=self.config.seed + self._iters)" in puzzle_text
            and "rng.permutation(" in puzzle_text
        ),
        "pretrain_seeds_torch_rng": "torch.random.manual_seed(" in pretrain_text,
        "pretrain_seeds_numpy_rng": "np.random.seed(" in pretrain_text,
        "model_uses_torch_exploration_rng": (
            "torch.rand_like(q_halt_logits)" in trm_text
            and "torch.randint_like(new_steps" in trm_text
        ),
        "rng_state_resume_contract_complete": False
        if model_checkpoint_payloads == ["train_state.model.state_dict()"]
        else None,
        "unpinned_requirement_count": sum("==" not in line for line in requirements),
        "pinned_requirement_count": sum("==" in line for line in pinned),
        "pinned_requirement_unique_count": len(pinned_names),
        "pinned_torch_requirement": torch_requirement,
        "dependency_hash_count": sum("--hash=" in line for line in pinned),
        "dependency_lock_has_transitive_hash_closure": all(
            "==" in line and "--hash=" in line for line in pinned
        ),
        "prior_smoke_torch_requirement_match": (
            prior_report.get("torch_version") == pinned_torch_version
        ),
        "arc_builder_default_num_aug": _annotated_default(
            data_process_config, "num_aug"
        ),
        "arc_max_grid_size": arc_max_grid_size,
        "arc_flat_sequence_length": arc_max_grid_size * arc_max_grid_size,
        "trm_hidden_size": _yaml_scalar(arch_text, "hidden_size"),
        "trm_halt_max_steps": _yaml_scalar(arch_text, "halt_max_steps"),
        "trm_puzzle_embedding_length": _yaml_scalar(
            arch_text, "puzzle_emb_len"
        ),
        "default_global_batch_size": _yaml_scalar(
            config_text, "global_batch_size"
        ),
        "default_epochs": _yaml_scalar(config_text, "epochs"),
        "default_eval_interval": _yaml_scalar(config_text, "eval_interval"),
        "default_checkpoint_every_eval": _yaml_scalar(
            config_text, "checkpoint_every_eval"
        ),
        "default_seed": _yaml_scalar(config_text, "seed"),
        "default_ema": _yaml_scalar(config_text, "ema"),
        "default_min_eval_interval": _yaml_scalar(
            config_text, "min_eval_interval"
        ),
        "default_evaluates_from_first_interval": (
            _yaml_scalar(config_text, "min_eval_interval") == 0
            and "if _iter_id >= config.min_eval_interval:" in pretrain_text
        ),
        "default_evaluation_interval_count": default_evaluation_count,
        "default_checkpoint_count_upper_bound": default_evaluation_count
        if _yaml_scalar(config_text, "checkpoint_every_eval") is True
        else 1,
        "checkpoint_retention_policy_detected": any(
            name in {"os.remove", "os.unlink", "Path.unlink", "shutil.rmtree"}
            for name in (
                core.dotted_name(call.func) for call in _all_calls(pretrain_tree)
            )
        ),
        "static_storage_lower_bound_available": False,
        "fixed_checkpoint_selection_policy_documented": bool(
            re.search(r"checkpoint selection|select(?:ed|ing)? checkpoint", readme, re.I)
        ),
        "fixed_seed_repetition_policy_documented": bool(
            re.search(r"(?:repeat|replicate).{0,30}seed|seeds?\s*[:=]\s*\[", readme, re.I)
        ),
        "root_license_mit": license_text.startswith("MIT License\n"),
    }


def validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("TRM gate config must be an object")
    if EXPECTED_CONFIG_CANONICAL_SHA256 == "TO_BE_PINNED":
        raise ValueError("auditor config digest has not been pinned")
    if core.canonical_sha256(value) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise ValueError("TRM gate config differs from the hardcoded v1 contract")
    if value.get("schema_version") != 1 or value.get("config_id") != CONFIG_ID:
        raise ValueError("TRM gate config identity mismatch")
    if value.get("method_id") != METHOD_ID or value.get("scope") != SCOPE:
        raise ValueError("TRM method or scope mismatch")
    if value.get("counted_toward_smoke") is not False:
        raise ValueError("TRM static gate cannot count toward smoke")
    if value.get("config_read_policy") != {
        "canonical_path": CANONICAL_CONFIG_RELATIVE,
        "alternate_paths_allowed": False,
    }:
        raise ValueError("TRM canonical config policy mismatch")
    if value.get("static_audit_support") != {
        "path": "scripts/audit_batch_c_static_gates.py",
        "sha256": EXPECTED_SUPPORT_SHA256,
    }:
        raise ValueError("TRM static-audit support binding mismatch")
    if value.get("source_lock") != {
        "path": SOURCE_LOCK_RELATIVE,
        "sha256": EXPECTED_SOURCE_LOCK_SHA256,
    }:
        raise ValueError("TRM source-lock binding mismatch")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("TRM source contract is missing")
    if source.get("repository_path") != (
        "/usr/paper-assets/arc/sources/tiny-recursive-models"
    ):
        raise ValueError("TRM source path mismatch")
    core.validate_hash(source.get("expected_revision"), "TRM revision", core.GIT_SHA_RE)
    core.validate_hash(
        source.get("expected_commit_tree"), "TRM commit tree", core.GIT_SHA_RE
    )
    for field in (
        "git_tree_manifest_sha256",
        "worktree_metadata_sha256",
        "retained_manifest_sha256",
    ):
        core.validate_hash(source.get(field), f"TRM {field}")
    retained = source.get("retained_text")
    metadata_only = source.get("metadata_only_paths")
    if not isinstance(retained, list) or len(retained) != 28:
        raise ValueError("TRM retained-text policy must contain 28 leaves")
    if not isinstance(metadata_only, list) or len(metadata_only) != 12:
        raise ValueError("TRM metadata-only policy must contain 12 leaves")
    retained_paths: set[str] = set()
    for item in retained:
        if not isinstance(item, dict) or set(item) != {"path", "role"}:
            raise ValueError("TRM retained declaration malformed")
        path = core.safe_relative_path(item["path"], "TRM retained path")
        if path in retained_paths or not item["role"]:
            raise ValueError("TRM retained path duplicated or role-less")
        if PurePosixPath(path).suffix.lower() in core.FORBIDDEN_RETAINED_SUFFIXES:
            raise ValueError("TRM restricted leaf entered retained-text policy")
        retained_paths.add(path)
    if len(set(metadata_only)) != len(metadata_only):
        raise ValueError("TRM metadata-only paths must be unique")
    for path in metadata_only:
        core.safe_relative_path(path, "TRM metadata-only path")
        if path in retained_paths or PurePosixPath(path).suffix.lower() not in {
            ".json",
            ".png",
        }:
            raise ValueError("TRM metadata-only policy contains an unsafe path")
    if source.get("expected_retained_file_count") != len(retained):
        raise ValueError("TRM retained count mismatch")
    if source.get("expected_metadata_only_file_count") != len(metadata_only):
        raise ValueError("TRM metadata-only count mismatch")
    if source.get("opaque_worktree_paths") != []:
        raise ValueError("TRM v1 has no opaque worktree directory")
    if source.get("expected_root_license_paths") != ["LICENSE"]:
        raise ValueError("TRM root-license expectation mismatch")
    blockers = value.get("blockers")
    if not isinstance(blockers, list) or len(blockers) != 10:
        raise ValueError("TRM gate must preserve ten blockers")
    blocker_ids = [item.get("id") for item in blockers if isinstance(item, dict)]
    if len(blocker_ids) != 10 or len(set(blocker_ids)) != 10:
        raise ValueError("TRM blocker IDs must be unique")
    if not all(set(item) == {"id", "gate", "detail"} for item in blockers):
        raise ValueError("TRM blocker declaration fields mismatch")
    if not isinstance(value.get("expected_analysis"), dict) or not value[
        "expected_analysis"
    ]:
        raise ValueError("TRM expected analysis is missing")
    if value.get("classification_contract") != {
        "classification_year_bucket": "paper_method_2025",
        "method_paper_year": 2025,
        "asset_snapshot_year": 2026,
        "asset_snapshot_created_at": "2026-08-06T02:49:44+00:00",
        "official_arc_prize_entry_verified": False,
        "readme_score_evidence_class": "unverified-upstream-readme-self-report",
        "claim": (
            "TinyRecursiveModels is classified in the 2025 paper/method year "
            "bucket with a locked asset/source snapshot created in 2026; this "
            "year bucket does not assert ARC Prize participation, the bound "
            "workspace evidence does not verify an official ARC Prize entry, "
            "and the README percentages are not official or independently "
            "reproduced scores."
        ),
    }:
        raise ValueError("TRM classification and score boundary changed")
    expected_controls = {
        "network_allowed": False,
        "gpu_allowed": False,
        "upstream_import_allowed": False,
        "upstream_execution_allowed": False,
        "arc_challenge_or_solution_byte_read_allowed": False,
        "checkpoint_byte_read_allowed": False,
        "checkpoint_load_allowed": False,
        "provider_or_wandb_call_allowed": False,
        "solver_execution_allowed": False,
        "prediction_allowed": False,
    }
    if value.get("controls") != expected_controls:
        raise ValueError("TRM controls differ from the fail-closed v1 policy")
    for group in (value.get("external_metadata"), value.get("prior_reports")):
        if not isinstance(group, list) or not group:
            raise ValueError("TRM bound evidence groups must be non-empty")
        for item in group:
            if not isinstance(item, dict) or set(item) != {
                "path",
                "sha256",
                "role",
                "assertions",
            }:
                raise ValueError("TRM bound evidence declaration malformed")
            core.validate_hash(item["sha256"], "TRM bound evidence SHA-256")
            if not isinstance(item["assertions"], dict) or not item["role"]:
                raise ValueError("TRM bound evidence assertion is missing")
    return value


def _source_lock_entry(value: dict[str, Any]) -> dict[str, Any]:
    entries = value.get("sources")
    if not isinstance(entries, dict) or METHOD_ID not in entries:
        raise ValueError("source lock has no TinyRecursiveModels entry")
    entry = entries[METHOD_ID]
    if not isinstance(entry, dict):
        raise ValueError("TinyRecursiveModels source-lock entry is malformed")
    return entry


def _read_bound_evidence(
    declarations: list[dict[str, Any]], ledger: core.ReadLedger
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    objects: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        declared = declaration["path"]
        path = Path(declared)
        if not path.is_absolute():
            path = ROOT / path
        payload = core.secure_read_absolute(
            path,
            role=declaration["role"],
            category="bound_metadata",
            max_bytes=MAX_BOUND_METADATA_BYTES,
            ledger=ledger,
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != declaration["sha256"]:
            raise ValueError(f"bound evidence SHA-256 mismatch: {declared}")
        parsed = core.strict_json(payload, declared)
        if not isinstance(parsed, dict):
            raise ValueError(f"bound evidence is not an object: {declared}")
        core.assert_subset(parsed, declaration["assertions"], declared)
        payloads[declared] = payload
        objects[declared] = parsed
        observations.append(
            {
                "path": declared,
                "role": declaration["role"],
                "sha256": digest,
                "assertions_passed": True,
            }
        )
    return observations, payloads, objects


def _verify_git_file(
    payload: bytes,
    signature: tuple[int, ...],
    declaration: dict[str, Any],
    field: str,
) -> None:
    core.verify_exact_file_contract(payload, signature, declaration, field)


def run_static_audit(
    config: dict[str, Any],
    config_payload: bytes,
    source_lock_payload: bytes,
    ledger: core.ReadLedger,
    run_id: str,
) -> dict[str, Any]:
    source = config["source"]
    source_path = Path(source["repository_path"])
    source_lock = core.strict_json(source_lock_payload, "source lock")
    lock_entry = _source_lock_entry(source_lock)
    if lock_entry != config["source_lock_entry"]:
        raise ValueError("TRM source-lock entry differs from the gate contract")
    if lock_entry.get("revision") != source["expected_revision"]:
        raise ValueError("TRM source-lock revision mismatch")

    declarations = config["external_metadata"] + config["prior_reports"]
    bound_observations, bound_payloads, bound_objects = _read_bound_evidence(
        declarations, ledger
    )
    asset_path = config["external_metadata"][0]["path"]
    prior_path = config["prior_reports"][0]["path"]

    root_fd = core.open_absolute_directory(source_path)
    git_fd: int | None = None
    object_fd: int | None = None
    isolated_fd: int | None = None
    isolated_temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        core.verify_directory_identity(source_path, root_fd)
        git_path_info = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(git_path_info.st_mode):
            raise ValueError("TRM .git is not a real directory")
        git_fd = os.open(".git", core.directory_flags(), dir_fd=root_fd)
        git_signature = core.stat_signature(git_path_info)
        if git_signature != core.stat_signature(os.fstat(git_fd)):
            raise RuntimeError("TRM .git raced before pinning")
        for forbidden in core.FORBIDDEN_GIT_PATHS:
            core.require_git_path_absent(git_fd, forbidden)

        git_contract = source["git_metadata_contract"]
        git_config, git_config_signature = core.secure_read_pinned_relative(
            git_fd,
            "config",
            role="git_local_config",
            category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        git_head, git_head_signature = core.secure_read_pinned_relative(
            git_fd,
            "HEAD",
            role="git_head",
            category="git_metadata",
            max_bytes=core.MAX_CONFIG_BYTES,
            ledger=ledger,
        )
        _verify_git_file(
            git_config,
            git_config_signature,
            git_contract["local_config"],
            "TRM Git local config",
        )
        _verify_git_file(
            git_head,
            git_head_signature,
            git_contract["head"],
            "TRM Git HEAD",
        )
        core.validate_git_config_entries(
            git_config, source["repository_url"], config["source_lock_entry"]["branch"]
        )
        revision = source["expected_revision"]
        allowed_heads = {
            f"{revision}\n".encode("ascii"),
            f"ref: refs/heads/{config['source_lock_entry']['branch']}\n".encode(
                "ascii"
            ),
        }
        if git_head not in allowed_heads:
            raise ValueError("TRM Git HEAD is not at the locked revision/branch")
        for token in (
            b"command =",
            b"insteadof",
            b"include.path",
            b"fsmonitor",
            b"sshcommand",
        ):
            if token in git_config.lower():
                raise ValueError("TRM Git config contains an execution/redirection key")

        object_info = os.stat("objects", dir_fd=git_fd, follow_symlinks=False)
        if not stat.S_ISDIR(object_info.st_mode):
            raise ValueError("TRM Git object store is not a directory")
        object_fd = os.open("objects", core.directory_flags(), dir_fd=git_fd)
        object_signature = core.stat_signature(object_info)
        if object_signature != core.stat_signature(os.fstat(object_fd)):
            raise RuntimeError("TRM Git object directory raced before pinning")
        isolated_temp = tempfile.TemporaryDirectory(
            prefix="arc-agi-eval-trm-git-", dir="/tmp"
        )
        isolated_fd = core.initialize_isolated_git_directory(
            Path(isolated_temp.name)
        )
        allowed_git = {
            ("rev-parse", "--verify", revision),
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            ("ls-tree", "-rz", "--full-tree", revision),
        }
        objects_before = core.git_object_metadata(git_fd)
        observed_revision = core.run_git(
            isolated_fd,
            object_fd,
            ("rev-parse", "--verify", revision),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        observed_tree = core.run_git(
            isolated_fd,
            object_fd,
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        tree_payload = core.run_git(
            isolated_fd,
            object_fd,
            ("ls-tree", "-rz", "--full-tree", revision),
            allowed_git,
            ledger,
        )
        if observed_revision != revision or observed_tree != source[
            "expected_commit_tree"
        ]:
            raise ValueError("TRM Git revision/tree mismatch")
        tree = core.parse_git_tree(tree_payload)
        tree_manifest = core.git_tree_manifest(tree)
        if core.canonical_sha256(tree_manifest) != source[
            "git_tree_manifest_sha256"
        ]:
            raise ValueError("TRM Git tree manifest mismatch")
        if len(tree) != source["expected_tracked_file_count"]:
            raise ValueError("TRM tracked file count mismatch")
        if core.extension_counts(set(tree)) != source["expected_extension_counts"]:
            raise ValueError("TRM extension inventory mismatch")

        worktree, initial_signatures, opaque = core.walk_worktree_metadata(
            root_fd, tree, set()
        )
        if opaque:
            raise ValueError("TRM worktree unexpectedly contains an opaque path")
        worktree_manifest = [worktree[path] for path in sorted(worktree)]
        if core.canonical_sha256(worktree_manifest) != source[
            "worktree_metadata_sha256"
        ]:
            raise ValueError("TRM worktree metadata manifest mismatch")
        if sum(item["bytes"] for item in worktree_manifest) != source[
            "expected_tracked_bytes"
        ]:
            raise ValueError("TRM tracked byte total mismatch")
        for path, item in tree.items():
            expected_mode = "0755" if item["mode"] == "100755" else "0644"
            if worktree[path]["mode"] != expected_mode:
                raise ValueError(f"TRM worktree/Git mode mismatch: {path}")

        retained_policy = {
            item["path"]: item["role"] for item in source["retained_text"]
        }
        metadata_only_paths = set(source["metadata_only_paths"])
        if set(tree) - set(retained_policy) != metadata_only_paths:
            raise ValueError("TRM retained/metadata-only partition is not closed")
        metadata_only_inventory = [
            {
                "path": path,
                "mode": worktree[path]["mode"],
                "bytes": worktree[path]["bytes"],
                "git_blob_oid": tree[path]["blob_oid"],
                "worktree_bytes_read": False,
                "worktree_sha256": None,
            }
            for path in sorted(metadata_only_paths)
        ]
        if len(metadata_only_inventory) != source[
            "expected_metadata_only_file_count"
        ] or sum(item["bytes"] for item in metadata_only_inventory) != source[
            "expected_metadata_only_bytes"
        ]:
            raise ValueError("TRM metadata-only count/byte contract mismatch")

        first_payloads: dict[str, bytes] = {}
        for path, role in retained_policy.items():
            first_payloads[path] = core.secure_read_retained(
                root_fd, path, role, initial_signatures[path], ledger
            )
        first_manifest = core.retained_manifest(
            source["retained_text"], first_payloads, tree
        )
        if len(first_manifest) != source["expected_retained_file_count"]:
            raise ValueError("TRM retained manifest count mismatch")
        if sum(item["bytes"] for item in first_manifest) != source[
            "expected_retained_bytes"
        ]:
            raise ValueError("TRM retained byte total mismatch")
        if core.canonical_sha256(first_manifest) != source[
            "retained_manifest_sha256"
        ]:
            raise ValueError("TRM retained manifest digest mismatch")
        parsed = {
            path: core.parse_python(payload, path)
            for path, payload in first_payloads.items()
            if PurePosixPath(path).suffix.lower() == ".py"
        }
        analysis = analyze_trm(
            parsed,
            first_payloads,
            bound_objects[asset_path],
            bound_objects[prior_path],
        )
        if analysis != config["expected_analysis"]:
            raise ValueError(
                "TRM static analysis differs from the pinned expectation: "
                + json.dumps(analysis, sort_keys=True)
            )
        root_license_paths = sorted(
            path
            for path in tree
            if "/" not in path
            and PurePosixPath(path).name.lower() in core.ROOT_LICENSE_NAMES
        )
        if root_license_paths != source["expected_root_license_paths"]:
            raise ValueError("TRM root-license path observation mismatch")

        terminal_worktree, terminal_signatures, terminal_opaque = (
            core.walk_worktree_metadata(root_fd, tree, set())
        )
        if (
            terminal_worktree != worktree
            or terminal_signatures != initial_signatures
            or terminal_opaque != opaque
        ):
            raise RuntimeError("TRM worktree metadata changed during audit")
        terminal_payloads: dict[str, bytes] = {}
        for path, role in retained_policy.items():
            terminal_payloads[path] = core.secure_read_retained(
                root_fd, path, role, terminal_signatures[path], ledger
            )
        terminal_manifest = core.retained_manifest(
            source["retained_text"], terminal_payloads, tree
        )
        if terminal_payloads != first_payloads or terminal_manifest != first_manifest:
            raise RuntimeError("TRM retained source changed during audit")

        terminal_git_config, terminal_git_config_signature = (
            core.secure_read_pinned_relative(
                git_fd,
                "config",
                role="git_local_config_terminal",
                category="git_metadata",
                max_bytes=core.MAX_CONFIG_BYTES,
                ledger=ledger,
            )
        )
        terminal_git_head, terminal_git_head_signature = (
            core.secure_read_pinned_relative(
                git_fd,
                "HEAD",
                role="git_head_terminal",
                category="git_metadata",
                max_bytes=core.MAX_CONFIG_BYTES,
                ledger=ledger,
            )
        )
        if (
            terminal_git_config != git_config
            or terminal_git_head != git_head
            or terminal_git_config_signature != git_config_signature
            or terminal_git_head_signature != git_head_signature
        ):
            raise RuntimeError("TRM Git metadata changed during audit")
        terminal_revision = core.run_git(
            isolated_fd,
            object_fd,
            ("rev-parse", "--verify", revision),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        terminal_tree = core.run_git(
            isolated_fd,
            object_fd,
            ("rev-parse", "--verify", f"{revision}^{{tree}}"),
            allowed_git,
            ledger,
        ).decode("ascii").strip()
        terminal_tree_payload = core.run_git(
            isolated_fd,
            object_fd,
            ("ls-tree", "-rz", "--full-tree", revision),
            allowed_git,
            ledger,
        )
        if (
            terminal_revision != observed_revision
            or terminal_tree != observed_tree
            or terminal_tree_payload != tree_payload
            or core.git_object_metadata(git_fd) != objects_before
        ):
            raise RuntimeError("TRM Git object/tree state changed during audit")
        if git_signature != core.stat_signature(
            os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        ) or git_signature != core.stat_signature(os.fstat(git_fd)):
            raise RuntimeError("TRM .git identity changed during audit")
        if object_signature != core.stat_signature(
            os.stat("objects", dir_fd=git_fd, follow_symlinks=False)
        ) or object_signature != core.stat_signature(os.fstat(object_fd)):
            raise RuntimeError("TRM object-store identity changed during audit")
        if set(os.listdir(isolated_fd)) != {"HEAD", "objects", "refs"}:
            raise RuntimeError("isolated TRM Git directory gained an entry")
        for directory in ("objects", "refs"):
            child_fd = os.open(directory, core.directory_flags(), dir_fd=isolated_fd)
            try:
                if os.listdir(child_fd):
                    raise RuntimeError(
                        f"isolated TRM Git {directory} directory was modified"
                    )
            finally:
                os.close(child_fd)
        core.verify_directory_identity(source_path, root_fd)
    finally:
        if isolated_fd is not None:
            os.close(isolated_fd)
        if isolated_temp is not None:
            isolated_temp.cleanup()
        if object_fd is not None:
            os.close(object_fd)
        if git_fd is not None:
            os.close(git_fd)
        os.close(root_fd)

    _, terminal_bound_payloads, _ = _read_bound_evidence(declarations, ledger)
    if terminal_bound_payloads != bound_payloads:
        raise RuntimeError("TRM bound metadata changed during audit")

    blockers = [{**item, "status": "blocked"} for item in config["blockers"]]
    controls = {
        "network_used": None,
        "gpu_used": None,
        "network_usage_measurement": "not-instrumented",
        "gpu_usage_measurement": "not-instrumented",
        "network_namespace_enforced": False,
        "gpu_device_namespace_enforced": False,
        "negative_runtime_claims_enforced_by_os": False,
        "intentional_network_code_path_executed": False,
        "intentional_gpu_code_path_executed": False,
        "upstream_imported": False,
        "upstream_executed": False,
        "checkpoint_opened": False,
        "checkpoint_loaded": False,
        "provider_or_wandb_code_path_executed": False,
        "solver_executed": False,
        "solver_prediction_produced": False,
        "auditor_process_arc_json_worktree_leaf_bytes_read": 0,
        "auditor_process_solution_json_worktree_leaf_bytes_read": 0,
        "auditor_process_image_worktree_leaf_bytes_read": 0,
        "metadata_only_worktree_leaf_content_reads": 0,
        "retained_source_read_passes": 2,
        "retained_source_read_attempts": 2 * len(source["retained_text"]),
        "bound_metadata_read_passes": 2,
        "bound_metadata_read_attempts": 2 * len(declarations),
        "git_subprocesses_started": len(ledger.git_processes),
        "git_subprocess_worktree_content_requested": False,
        "git_subprocess_object_database_reads_possible": True,
        "git_subprocess_object_database_bytes_measured": False,
        "git_subprocess_source_local_config_available": False,
        "git_subprocess_isolated_git_directory_used": True,
        "git_system_and_global_config_disabled": True,
        "git_lazy_fetch_disabled": True,
    }
    source_observation = {
        "repository_path": str(source_path),
        "expected_revision": source["expected_revision"],
        "observed_revision": observed_revision,
        "observed_commit_tree": observed_tree,
        "tracked_file_count": len(tree),
        "tracked_worktree_bytes": sum(item["bytes"] for item in worktree_manifest),
        "git_tree_manifest_sha256": core.canonical_sha256(tree_manifest),
        "worktree_metadata_sha256": core.canonical_sha256(worktree_manifest),
        "extension_counts": core.extension_counts(set(tree)),
        "retained_byte_exact_file_count": len(first_manifest),
        "retained_byte_exact_bytes": sum(item["bytes"] for item in first_manifest),
        "retained_manifest_sha256": core.canonical_sha256(first_manifest),
        "metadata_only_tracked_file_count": len(metadata_only_inventory),
        "metadata_only_tracked_bytes": sum(
            item["bytes"] for item in metadata_only_inventory
        ),
        "metadata_only_inventory": metadata_only_inventory,
        "root_license_paths": root_license_paths,
        "working_tree_all_files_byte_exact_verified": False,
    }
    passed_gates = [
        {
            "id": "locked-source-integrity",
            "gate": "source_provenance",
            "status": "passed",
            "blocking": False,
            "detail": "The exact commit/tree, closed worktree metadata, and 28 retained text leaves reproduced without opening 12 restricted leaves.",
        },
        {
            "id": "root-code-license",
            "gate": "license_clearance",
            "status": "passed",
            "blocking": False,
            "detail": "The locked root LICENSE is MIT; this does not clear bundled dataset or future checkpoint licenses.",
        },
    ]
    observation = {
        "method_id": METHOD_ID,
        "scope": SCOPE,
        "source": source_observation,
        "static_analysis": analysis,
        "bound_evidence": bound_observations,
        "passed_gates": passed_gates,
        "blockers": blockers,
        "gate_summary": {"passed": len(passed_gates), "blocked": len(blockers)},
        "controls": controls,
        "classification": config["classification_contract"],
        "benchmark_policy": config["benchmark_policy"],
        "prior_evidence_interpretation": config[
            "prior_evidence_interpretation"
        ],
    }
    return {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "runner": "scripts.audit_trm_gates",
        "status": "passed",
        "method_gate_status": "blocked",
        "scope": SCOPE,
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "solver_gate_passed": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "config": {
            "path": CANONICAL_CONFIG_RELATIVE,
            "file_sha256": hashlib.sha256(config_payload).hexdigest(),
            "canonical_sha256": core.canonical_sha256(config),
        },
        **observation,
        "observation_digest_sha256": core.canonical_sha256(observation),
        "read_ledger": ledger.snapshot(),
        "validation": {
            "canonical_config_bound": True,
            "support_source_bound": True,
            "source_lock_bound": True,
            "revision_and_tree_bound": True,
            "isolated_git_metadata_view_bound": True,
            "closed_worktree_metadata_bound": True,
            "retained_bytes_bound_twice": True,
            "restricted_worktree_content_not_opened": True,
            "bound_metadata_verified_twice": True,
            "static_analysis_matches": True,
            "terminal_state_matched_at_last_observation": True,
            "all_method_blockers_preserved": True,
        },
        "claim_boundary": (
            "This metadata-first audit makes byte-exact claims only for 28 retained text leaves. "
            "It makes path/mode/Git-OID/size metadata claims, not worktree-content claims, for 10 bundled ARC JSON and two image leaves. "
            "The method/paper is classified in the 2025 year bucket and the locked asset/source snapshot was created in 2026; the year bucket does not assert ARC Prize participation and no bound evidence verifies an official ARC Prize entry. "
            "The 45%/8% values are unverified upstream README self-reports, not official or independently reproduced scores. "
            "Git object-database byte volume and network/GPU syscalls are unmeasured. The operator-supplied runner-manifest digest is checked, but no repository-external signature or transparency-log anchor is verified during this run. Passing reproduces blockers; it is not a solver smoke, prediction, score, benchmark, checkpoint run, paper reproduction, or cryptographic attestation."
        ),
        "limitations": [
            "The ten bundled JSON and two image leaves were not opened; their worktree bytes were not verified against the pinned Git blobs.",
            "The root MIT license does not by itself establish the provenance or reuse terms of bundled ARC/concept data or any future checkpoint.",
            "Git metadata subprocesses may read pinned local object files; kernel-level object byte volume is not measured.",
            "Network and GPU use are not observed by a syscall/device monitor or prevented by an OS namespace; false fields describe only intentional audited code paths and measured use remains unknown.",
            "Input equality is checked in two complete observations plus a final byte re-read, but the filesystem is not an immutable mount; mutation after the last check and before report commit cannot be eliminated.",
            "The launcher requires /usr/bin/python3 -I -B -S semantics and a canonical direct-source entry, but its executing bytes and the interpreter binary are not preauthenticated by this Python process.",
            "The operator-supplied manifest SHA-256 must be anchored by the subsequent input-freeze chain; no repository-external signature or transparency log is verified during this run.",
            "Static source signatures characterize only the exact pinned tree and do not prove a future challenge-only adapter or runtime safe.",
        ],
    }


def failure_record(
    run_id: str, error: BaseException, ledger: core.ReadLedger
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "runner": "scripts.audit_trm_gates",
        "status": "failed",
        "method_gate_status": "blocked",
        "scope": SCOPE,
        "counted_toward_smoke": False,
        "solver_prediction_produced": False,
        "solver_gate_passed": False,
        "strict_runtime_promoted": False,
        "performance_table_eligible": False,
        "failure": {"type": type(error).__name__, "message": str(error)},
        "read_ledger": ledger.snapshot(),
        "controls": {
            "network_used": None,
            "gpu_used": None,
            "upstream_imported": None,
            "upstream_executed": None,
            "checkpoint_opened": None,
            "checkpoint_loaded": None,
            "provider_or_wandb_called": None,
            "solver_executed": None,
            "solver_prediction_produced": False,
            "measurement_status": "unknown-after-audit-failure",
        },
        "claim_boundary": "The static audit failed; no solver or prediction was run.",
    }


def validate_output_location(path: Path) -> None:
    parts = core._absolute_parts(path)
    if len(parts) < 2 or parts[-1] in {"", ".", ".."} or parts[-1].startswith("."):
        raise core.OutputPathError("output must use a visible fresh leaf name")
    absolute = Path(*parts)
    production_parent = ROOT / "reports" / METHOD_ID
    if Path(*parts[:-1]) == Path(*core._absolute_parts(production_parent)):
        return
    if core.lexical_within(absolute, TEST_OUTPUT_ROOT) and not core.lexical_equal(
        absolute, TEST_OUTPUT_ROOT
    ):
        return
    raise core.OutputPathError(
        "output must be a fresh TinyRecursiveModels report leaf or enter the dedicated TRM test root"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    canonical_config = ROOT / CANONICAL_CONFIG_RELATIVE
    canonical_manifest = ROOT / RUNNER_MANIFEST_RELATIVE
    try:
        (
            config,
            config_payload,
            auditor_payload,
            support_payload,
            source_lock_payload,
            runner_manifest,
        ) = bootstrap_verified_runtime(args.config, args.runner_manifest)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    started_at = core.utc_now()
    started_monotonic = time.monotonic()
    started_usage = core.usage_snapshot()

    output_path = (
        args.output_directory
        if args.output_directory.is_absolute()
        else ROOT / args.output_directory
    )
    try:
        validate_output_location(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    protected = {
        canonical_config,
        canonical_manifest,
        ROOT / SOURCE_LOCK_RELATIVE,
        AUDITOR_PATH,
        SUPPORT_PATH,
        ROOT / "scripts" / "launch_trm_gate.py",
    }
    if any(core.lexical_equal(output_path, path) for path in protected):
        parser.error("output path overlaps a protected input")

    try:
        retained_policy = {
            item["path"]: item["role"] for item in config["source"]["retained_text"]
        }
        if hashlib.sha256(source_lock_payload).hexdigest() != EXPECTED_SOURCE_LOCK_SHA256:
            raise ValueError("TRM source-lock SHA-256 mismatch")
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    try:
        output = core.create_fresh_output(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    ledger = core.ReadLedger(retained_policy)
    phase_ledgers: list[core.ReadLedger] = []
    try:
        try:
            first_ledger = core.ReadLedger(retained_policy)
            phase_ledgers.append(first_ledger)
            ledger = first_ledger
            first_record = run_static_audit(
                config,
                config_payload,
                source_lock_payload,
                first_ledger,
                output_path.name,
            )
            first_digest = first_record["observation_digest_sha256"]

            second_config = _bootstrap_secure_read(
                canonical_config,
                max_bytes=core.MAX_CONFIG_BYTES,
                field="TRM gate config replay",
            )
            second_lock = _bootstrap_secure_read(
                ROOT / SOURCE_LOCK_RELATIVE,
                max_bytes=core.MAX_CONFIG_BYTES,
                field="TRM source lock replay",
            )
            if second_config != config_payload or second_lock != source_lock_payload:
                raise RuntimeError("TRM config or source lock changed between audits")
            ledger = core.ReadLedger(retained_policy)
            phase_ledgers.append(ledger)
            record = run_static_audit(
                config,
                config_payload,
                source_lock_payload,
                ledger,
                output_path.name,
            )
            second_digest = record["observation_digest_sha256"]
            if first_digest != second_digest:
                raise RuntimeError(
                    "TRM complete replay observation changed before commit"
                )

            context = globals()["__verified_runner_manifest_context__"]
            manifest_payload = context["manifest_payload"]
            terminal_manifest = _bootstrap_secure_read(
                canonical_manifest,
                max_bytes=core.MAX_CONFIG_BYTES,
                field="TRM runner manifest terminal",
            )
            terminal_config = _bootstrap_secure_read(
                canonical_config,
                max_bytes=core.MAX_CONFIG_BYTES,
                field="TRM gate config terminal",
            )
            terminal_lock = _bootstrap_secure_read(
                ROOT / SOURCE_LOCK_RELATIVE,
                max_bytes=core.MAX_CONFIG_BYTES,
                field="TRM source lock terminal",
            )
            terminal_auditor = _bootstrap_secure_read(
                AUDITOR_PATH,
                max_bytes=MAX_AUDITOR_BYTES,
                field="TRM auditor terminal",
            )
            terminal_support = _bootstrap_secure_read(
                SUPPORT_PATH,
                max_bytes=MAX_AUDITOR_BYTES,
                field="TRM support terminal",
            )
            launcher_path = ROOT / "scripts" / "launch_trm_gate.py"
            terminal_launcher = _bootstrap_secure_read(
                launcher_path,
                max_bytes=MAX_AUDITOR_BYTES,
                field="TRM launcher terminal",
            )
            member_payloads = context["member_payloads"]
            if (
                terminal_manifest != manifest_payload
                or terminal_config != config_payload
                or terminal_lock != source_lock_payload
                or terminal_auditor != auditor_payload
                or terminal_support != support_payload
                or terminal_launcher
                != member_payloads["scripts/launch_trm_gate.py"]
            ):
                raise RuntimeError(
                    "TRM runner manifest or one of its members changed during audit"
                )
            runner_members = {
                item["role"]: item for item in runner_manifest["members"]
            }
            record["runner_provenance"] = {
                "path": "scripts/audit_trm_gates.py",
                "bytes": len(auditor_payload),
                "sha256": hashlib.sha256(auditor_payload).hexdigest(),
                "expected_sha256": runner_members["auditor"]["sha256"],
                "manifest_bound": True,
                "executed_from_verified_source_bytes": True,
                "terminal_bytes_equal": True,
            }
            record["support_provenance"] = {
                "path": "scripts/audit_batch_c_static_gates.py",
                "bytes": len(support_payload),
                "sha256": hashlib.sha256(support_payload).hexdigest(),
                "expected_sha256": runner_members["support"]["sha256"],
                "loaded_via": "preverified-source-bytes-compile-exec",
                "normal_import_used": False,
                "pyc_used": False,
                "terminal_bytes_equal": True,
            }
            launcher_context = context["launcher_source_execution"]
            record["launcher_provenance"] = {
                "path": "scripts/launch_trm_gate.py",
                "sha256": runner_members["launcher"]["sha256"],
                "canonical_direct_script_source": True,
                "isolated_python_flags_required": True,
                "terminal_file_matches_manifest": True,
                "executed_launcher_bytes_preauthenticated": False,
                "interpreter_binary_digest_verified": False,
                "source_execution_context": launcher_context,
            }
            record["runner_manifest_provenance"] = {
                "path": RUNNER_MANIFEST_RELATIVE,
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "operator_supplied_expected_sha256": context[
                    "operator_supplied_manifest_sha256"
                ],
                "operator_supplied_digest_matched": True,
                "repository_external_signature_verified": False,
                "next_input_freeze_anchor_required": True,
                "members_sha256": runner_manifest["members_sha256"],
                "member_count": runner_manifest["member_count"],
                "launcher_sha256": runner_members["launcher"]["sha256"],
                "terminal_bytes_equal": True,
            }
            record["replay_consistency"] = {
                "complete_observation_count": 2,
                "first_observation_digest_sha256": first_digest,
                "second_observation_digest_sha256": second_digest,
                "equal": True,
                "first_read_ledger_sha256": core.canonical_sha256(
                    first_ledger.snapshot()
                ),
                "second_read_ledger_sha256": core.canonical_sha256(
                    ledger.snapshot()
                ),
            }
            record["commit_consistency"] = {
                "strategy": (
                    "two-complete-source-observations-plus-terminal-byte-recheck"
                ),
                "immutable_mount": False,
                "toctou_eliminated": False,
                "claim": "Inputs matched at the final observation before commit.",
            }
            record["validation"].update(
                {
                    "runner_sha256_manifest_bound": True,
                    "runner_manifest_members_verified_by_launcher": True,
                    "operator_supplied_manifest_digest_matched": True,
                    "launcher_direct_source_and_isolated_flags_required": True,
                    "launcher_terminal_file_matches_manifest": True,
                    "support_loaded_only_after_sha256_verification": True,
                    "double_complete_observation_equal": True,
                    "runner_manifest_terminally_stable_at_last_observation": True,
                }
            )
            returncode = 0
        except BaseException as error:
            aggregate_ledger = core.ReadLedger(retained_policy)
            for phase_ledger in phase_ledgers:
                aggregate_ledger.file_reads.extend(phase_ledger.file_reads)
                aggregate_ledger.git_processes.extend(phase_ledger.git_processes)
            ledger = aggregate_ledger
            record = failure_record(output_path.name, error, ledger)
            returncode = 1
        record["read_ledger"] = ledger.snapshot()
        record["started_at_utc"] = started_at
        record["ended_at_utc"] = core.utc_now()
        resources = core.usage_record(
            started_usage,
            core.usage_snapshot(),
            time.monotonic() - started_monotonic,
        )
        resources["provider_requests"] = None
        resources["currency_spend_usd"] = None
        resources["gpu_used"] = None
        resources["network_used"] = None
        resources["network_usage_measurement"] = "not-instrumented"
        resources["gpu_usage_measurement"] = "not-instrumented"
        resources["intentional_provider_requests"] = 0
        resources["intentional_currency_spend_usd"] = 0.0
        resources["bootstrap_resources_included"] = False
        record["resources"] = resources
        core.write_json_no_clobber(output, record)
        stream = sys.stdout if returncode == 0 else sys.stderr
        print(json.dumps(record, indent=2, sort_keys=True), file=stream)
        return returncode
    finally:
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())
