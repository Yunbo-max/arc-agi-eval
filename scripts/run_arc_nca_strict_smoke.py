#!/usr/bin/env python3
"""Run one frozen ARC_NCA CPU smoke through a method-specific A/B firewall."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi_eval.challenge_runtime import (  # noqa: E402
    canonical_sha256,
    mutate_hidden_test_labels,
    pretty_json_bytes,
    run_logged_process,
    sentinel_predictions,
    sha256_bytes,
    sha256_file,
    tree_inventory,
    tree_sha256,
    utc_now,
)
from arc_agi_eval.firewall import challenge_only  # noqa: E402
from arc_agi_eval.protocol import build_protocol_manifest  # noqa: E402
from arc_agi_eval.resources import ResourceMonitor  # noqa: E402
from arc_agi_eval.run_schema import validate_run_file  # noqa: E402
from arc_agi_eval.validation import load_task  # noqa: E402
from scripts.run_challenge_runtime_core import (  # noqa: E402
    answers_from_directory,
    assert_score_contract,
    atomic_bytes,
    atomic_json,
    cpu_model,
    file_record,
    git_dirty,
    git_revision,
    output_count,
    parse_utc,
    read_json,
    score_reference_payload,
    validate_prediction_shape,
)


SCHEMA_PATH = ROOT / "schemas" / "protocol-v1-run.schema.json"
PROTOCOL_CONFIG = ROOT / "configs" / "protocol_v1_draft.json"
DEFAULT_CONFIG = ROOT / "configs" / "arc_nca_cpu_dev_smoke_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "reports" / "arc-nca" / "20260806-cpu-dev-6150a2bd-strict-v1"
)


def _require_inside_repository(path: Path, parser: argparse.ArgumentParser) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        parser.error(f"path must remain inside repository: {error}")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pre-run-input-bundle", required=True, type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.config = _require_inside_repository(args.config, parser)
    args.pre_run_input_bundle = _require_inside_repository(
        args.pre_run_input_bundle, parser
    )
    args.output_directory = _require_inside_repository(args.output_directory, parser)
    if not args.config.is_file():
        parser.error(f"config does not exist: {args.config}")
    if not args.pre_run_input_bundle.is_file():
        parser.error(
            f"pre-run input-bundle run does not exist: {args.pre_run_input_bundle}"
        )
    if args.pre_run_input_bundle.name != "run.json":
        parser.error("--pre-run-input-bundle must name an immutable run.json")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error(f"output directory is not empty: {args.output_directory}")
    return args


def _inventory_map(inventory_path: Path) -> dict[str, dict[str, Any]]:
    inventory = read_json(inventory_path)
    files = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(files, list):
        raise ValueError(f"invalid inventory: {inventory_path}")
    result: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"invalid inventory item: {inventory_path}")
        if item["path"] in result:
            raise ValueError(f"duplicate inventory path: {item['path']}")
        result[item["path"]] = item
    return result


def verify_pre_run_bundle(
    run_path: Path, *, config_path: Path, code_paths: list[Path]
) -> dict[str, Any]:
    validation = validate_run_file(
        run_path,
        schema_path=SCHEMA_PATH,
        repo_root=ROOT,
        verify_files=True,
    )
    run = read_json(run_path)
    if run.get("status") != "passed" or run.get("method_id") != "e0-input-freeze":
        raise ValueError("pre-run input bundle is not a passed e0-input-freeze run")
    directory = run_path.parent
    bundle_path = directory / "bundle-manifest.json"
    bundle = read_json(bundle_path)
    if bundle["authorization"]["locked_public_solver_run"] is not False:
        raise ValueError("pre-run bundle unexpectedly authorizes public execution")
    if bundle["methods"]["strict_runtime_passed"] != 0:
        raise ValueError("pre-run bundle must precede this strict promotion")
    if bundle["methods"]["admitted_configuration_count"] != 0:
        raise ValueError("pre-run bundle must have an empty public admitted set")

    declared_path = directory / "declared-input-inventory.json"
    code_path = directory / "code-inventory.json"
    declared = _inventory_map(declared_path)
    code = _inventory_map(code_path)

    required_declared = [config_path, ROOT / "configs" / "source_locks.json"]
    required_code = code_paths
    verified: list[dict[str, str]] = []
    for path, inventory, kind in [
        *((item, declared, "declared") for item in required_declared),
        *((item, code, "code") for item in required_code),
    ]:
        relative = path.relative_to(ROOT).as_posix()
        item = inventory.get(relative)
        if item is None:
            raise ValueError(f"pre-run bundle omits {kind} path: {relative}")
        observed = sha256_file(path)
        if observed != item.get("sha256"):
            raise ValueError(f"pre-run bundle hash mismatch: {relative}")
        snapshot = ROOT / item["snapshot_path"]
        if sha256_file(snapshot) != observed or snapshot.read_bytes() != path.read_bytes():
            raise ValueError(f"pre-run snapshot mismatch: {relative}")
        verified.append({"path": relative, "sha256": observed, "kind": kind})
    return {
        "run_path": run_path.relative_to(ROOT).as_posix(),
        "run_sha256": sha256_file(run_path),
        "record_sha256": validation.record_sha256,
        "bundle_manifest_path": bundle_path.relative_to(ROOT).as_posix(),
        "bundle_manifest_sha256": sha256_file(bundle_path),
        "bundle_root_sha256": bundle["bundle_root_sha256"],
        "verified_inputs": verified,
    }


def safe_inference_config(config: dict[str, Any]) -> dict[str, Any]:
    adapter = config["adapter"]
    optimization = config["optimization"]
    determinism = config["determinism"]
    return {
        "schema_version": 1,
        "config_id": config["config_id"],
        "task_id": config["task_id"],
        "expected_challenge_sha256": config["expected_challenge_sha256"],
        "expected_nca_source_sha256": config["upstream"][
            "expected_nca_source_sha256"
        ],
        "expected_arc_utils_sha256": config["upstream"][
            "expected_arc_utils_sha256"
        ],
        "steps": optimization["steps"],
        "rollout_steps": optimization["rollout_steps"],
        "pool_size": optimization["pool_size"],
        "batch_size": optimization["batch_size"],
        "seed": determinism["seed"],
        "top_k": config["attempt_budget"]["top_k"],
        "threads": determinism["threads"],
        "channels": adapter["channels"],
        "hidden_channels": adapter["hidden_channels"],
        "gene_size": adapter["gene_size"],
        "color_count": adapter["color_count"],
        "expected_parameter_count": adapter["expected_parameter_count"],
        "learning_rate": optimization["learning_rate"],
        "noise_level": optimization["noise_level"],
        "update_rate": optimization["update_rate"],
    }


def copy_locked_tree(source: Path, destination: Path) -> list[dict[str, object]]:
    inventory = tree_inventory(source)
    for item in inventory:
        relative = Path(str(item["path"]))
        atomic_bytes(destination / relative, (source / relative).read_bytes())
    if tree_inventory(destination) != inventory:
        raise ValueError("copied upstream tree differs from locked source")
    return inventory


def _full_task_from_frozen_record(
    config: dict[str, Any], frozen_manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    representatives = [
        item
        for item in frozen_manifest["representatives"]
        if item.get("task_id") == config["task_id"]
    ]
    if len(representatives) != 1:
        raise ValueError("frozen task representative is not unique")
    representative = representatives[0]
    expected_pairs = {
        "order_index": config["representative_order_index"],
        "cluster_id": config["representative_cluster_id"],
        "challenge_sha256": config["expected_challenge_sha256"],
        "solution_sha256": config["expected_solution_sha256"],
        "source_sha256": config["expected_source_task_sha256"],
        "source_path": config["source_task_path"],
    }
    for field, expected in expected_pairs.items():
        if representative.get(field) != expected:
            raise ValueError(f"frozen representative mismatch: {field}")
    frozen_directory = ROOT / config["frozen_runtime_directory"]
    challenge_path = frozen_directory / representative["challenge_path"]
    solution_path = frozen_directory / representative["solution_path"]
    source_path = ROOT / representative["source_path"]
    for path, expected in (
        (challenge_path, config["expected_challenge_sha256"]),
        (solution_path, config["expected_solution_sha256"]),
        (source_path, config["expected_source_task_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen task file digest mismatch: {path}")
    task = load_task(source_path)
    challenge = challenge_only(task)
    if pretty_json_bytes(challenge) != challenge_path.read_bytes():
        raise ValueError("source task does not reproduce the frozen challenge bytes")
    solution = read_json(solution_path)
    expected_outputs = [pair["output"] for pair in task["test"]]
    if solution != {"task_id": config["task_id"], "test_outputs": expected_outputs}:
        raise ValueError("frozen solution does not match the source task")
    return task, representative, challenge_path, solution_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    monitor = ResourceMonitor(include_nvidia=False).start()
    events: list[dict[str, object]] = []
    status = "failed"
    failure: dict[str, str] | None = None
    usage = None
    try:
        config = read_json(args.config)
        if config.get("method_id") != "arc-nca":
            raise ValueError("method config identity mismatch")
        if config.get("public_execution_authorized") is not False:
            raise ValueError("this development smoke cannot authorize public execution")
        if config.get("analyst_test_label_exposure") is not True:
            raise ValueError("known analyst exposure must remain disclosed")
        if config.get("performance_tuning_from_test_labels") is not False:
            raise ValueError("test-label tuning is forbidden")

        code_paths = [
            ROOT / "scripts" / "infer_arc_nca_cpu.py",
            ROOT / "scripts" / "run_arc_nca_strict_smoke.py",
            ROOT / "scripts" / "run_challenge_runtime_core.py",
            ROOT / "arc_agi_eval" / "challenge_runtime.py",
            ROOT / "arc_agi_eval" / "firewall.py",
            ROOT / "arc_agi_eval" / "reference_scoring.py",
            ROOT / "arc_agi_eval" / "resources.py",
            ROOT / "arc_agi_eval" / "run_schema.py",
            ROOT / "arc_agi_eval" / "scoring.py",
            ROOT / "arc_agi_eval" / "validation.py",
        ]
        pre_run_bundle = verify_pre_run_bundle(
            args.pre_run_input_bundle,
            config_path=args.config,
            code_paths=code_paths,
        )

        frozen_directory = ROOT / config["frozen_runtime_directory"]
        frozen_manifest_path = frozen_directory / "manifest.json"
        frozen_run_path = frozen_directory / "run.json"
        if sha256_file(frozen_manifest_path) != config[
            "expected_frozen_manifest_sha256"
        ]:
            raise ValueError("frozen development manifest digest mismatch")
        if sha256_file(frozen_run_path) != config["expected_frozen_run_sha256"]:
            raise ValueError("frozen development run digest mismatch")
        frozen_manifest = read_json(frozen_manifest_path)
        task, representative, frozen_challenge_path, frozen_solution_path = (
            _full_task_from_frozen_record(config, frozen_manifest)
        )
        task_id = config["task_id"]
        task_count = 1
        outputs_total = len(task["test"])
        top_k = config["attempt_budget"]["top_k"]
        timeout_seconds = config["attempt_budget"][
            "timeout_seconds_per_process"
        ]
        if outputs_total != 1 or top_k != 2:
            raise ValueError("frozen single-output Top-2 contract changed")

        upstream_root = ROOT / config["upstream"]["repository_path"]
        upstream_inventory = tree_inventory(upstream_root)
        if len(upstream_inventory) != config["upstream"]["expected_file_count"]:
            raise ValueError("upstream file count mismatch")
        if tree_sha256(upstream_root) != config["upstream"]["expected_tree_sha256"]:
            raise ValueError("upstream retained-tree digest mismatch")
        license_path = ROOT / config["upstream"]["license_path"]
        if sha256_file(license_path) != config["upstream"][
            "expected_license_sha256"
        ]:
            raise ValueError("upstream license digest mismatch")

        inputs_directory = output_directory / "inputs"
        staged_upstream = inputs_directory / "upstream"
        copied_inventory = copy_locked_tree(upstream_root, staged_upstream)
        adapter_snapshot = inputs_directory / "adapter" / "infer_arc_nca_cpu.py"
        orchestrator_snapshot = (
            inputs_directory / "adapter" / "run_arc_nca_strict_smoke.py"
        )
        atomic_bytes(
            adapter_snapshot, (ROOT / "scripts" / "infer_arc_nca_cpu.py").read_bytes()
        )
        atomic_bytes(orchestrator_snapshot, Path(__file__).read_bytes())
        if sha256_file(staged_upstream / "NCA.py") != config["upstream"][
            "expected_nca_source_sha256"
        ]:
            raise ValueError("staged NCA.py digest mismatch")
        if sha256_file(staged_upstream / "arc_agi_utils.py") != config["upstream"][
            "expected_arc_utils_sha256"
        ]:
            raise ValueError("staged arc_agi_utils.py digest mismatch")

        config_copy = output_directory / "config.json"
        atomic_bytes(config_copy, args.config.read_bytes())
        protocol_live = build_protocol_manifest(ROOT, PROTOCOL_CONFIG)
        protocol_snapshot = output_directory / "protocol-manifest.json"
        atomic_json(
            protocol_snapshot,
            {
                "schema_version": 1,
                "protocol_id": "arc-rebench-protocol-v1-draft",
                "protocol_status": protocol_live["protocol_status"],
                "protocol_config_path": PROTOCOL_CONFIG.relative_to(ROOT).as_posix(),
                "protocol_config_sha256_at_execution": sha256_file(PROTOCOL_CONFIG),
                "protocol_root_sha256_at_execution": protocol_live[
                    "protocol_root_sha256"
                ],
                "freeze_ready_at_execution": protocol_live["readiness"][
                    "freeze_ready"
                ],
                "required_unmet_gate_ids": protocol_live["readiness"][
                    "required_unmet_gate_ids"
                ],
                "public_solver_execution_authorized": False,
                "pre_run_input_bundle": pre_run_bundle,
            },
        )

        runtime_a = output_directory / "runtime" / "a"
        runtime_b = output_directory / "runtime" / "b"
        inference_a = runtime_a / "inference"
        inference_b = runtime_b / "inference"
        scoring_a = runtime_a / "scoring"
        scoring_b = runtime_b / "scoring"
        challenge_payload = frozen_challenge_path.read_bytes()
        atomic_bytes(inference_a / f"{task_id}.json", challenge_payload)
        atomic_bytes(inference_b / f"{task_id}.json", challenge_payload)
        safe_manifest = {
            "format": "arc-agi-challenge-tree-v1",
            "source_id": "frozen-dev-audit-single-task-v1",
            "tasks_total": 1,
            "files": [
                {
                    "path": f"{task_id}.json",
                    "sha256": config["expected_challenge_sha256"],
                }
            ],
        }
        atomic_json(inference_a / "MANIFEST", safe_manifest)
        atomic_json(inference_b / "MANIFEST", safe_manifest)
        inference_config = safe_inference_config(config)
        inference_config_a = runtime_a / "inference-config.json"
        inference_config_b = runtime_b / "inference-config.json"
        atomic_json(inference_config_a, inference_config)
        atomic_json(inference_config_b, inference_config)
        if inference_config_a.read_bytes() != inference_config_b.read_bytes():
            raise ValueError("A/B inference-safe config bytes differ")

        mutated = mutate_hidden_test_labels(
            task, offset=config["hidden_label_mutation"]["offset"]
        )
        if pretty_json_bytes(challenge_only(mutated)) != challenge_payload:
            raise ValueError("hidden-label mutation changed inference-visible task")
        atomic_json(scoring_a / f"{task_id}.json", task)
        atomic_json(scoring_b / f"{task_id}.json", mutated)
        mutation_changed_outputs = 0
        mutation_changed_cells = 0
        for original_pair, mutated_pair in zip(task["test"], mutated["test"]):
            mutation_changed_outputs += int(
                original_pair["output"] != mutated_pair["output"]
            )
            mutation_changed_cells += sum(
                original != changed
                for original_row, changed_row in zip(
                    original_pair["output"], mutated_pair["output"]
                )
                for original, changed in zip(original_row, changed_row)
            )
        if mutation_changed_outputs != outputs_total:
            raise ValueError("hidden-label mutation did not change every output")
        if tree_inventory(inference_a) != tree_inventory(inference_b):
            raise ValueError("A/B inference-visible trees differ")
        if tree_sha256(scoring_a) == tree_sha256(scoring_b):
            raise ValueError("A/B hidden scoring trees are identical")

        sentinel_path = output_directory / "scoring-only" / "sentinel.json"
        sentinel = sentinel_predictions([scoring_a / f"{task_id}.json"], top_k=top_k)
        atomic_json(sentinel_path, sentinel)

        child_environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "LC_ALL": "C.UTF-8",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONHASHSEED": str(config["determinism"]["python_hash_seed"]),
            "PYTHONNOUSERSITE": "1",
            "TZ": "UTC",
        }
        scoring_environment = dict(child_environment)
        scoring_environment["PYTHONPATH"] = str(ROOT)
        environment_path = output_directory / "environment-lock.json"
        atomic_json(
            environment_path,
            {
                "schema_version": 1,
                "expected": config["expected_environment"],
                "inference_environment_allowlist": child_environment,
                "scoring_environment_addition": {"PYTHONPATH": str(ROOT)},
                "provider_credentials_forwarded": False,
                "network_used": False,
                "network_namespace_enforced": False,
                "filesystem_namespace_enforced": False,
                "gpu_device_namespace_enforced": False,
            },
        )
        event_log_path = output_directory / "process-events.json"

        def run_child(
            command: list[str],
            *,
            name: str,
            kind: str,
            runtime: Path,
            environment: dict[str, str],
            stdout_path: Path | None = None,
        ) -> dict[str, object]:
            destination = stdout_path or runtime / "logs" / f"{name}.stdout.log"
            event = run_logged_process(
                command,
                name=name,
                kind=kind,
                cwd=runtime / "work",
                timeout_seconds=timeout_seconds,
                stdout_path=destination,
                stderr_path=runtime / "logs" / f"{name}.stderr.log",
                environment=environment,
                display_root=ROOT,
            ).as_dict()
            events.append(event)
            atomic_json(event_log_path, {"schema_version": 1, "events": events})
            if event["status"] != "passed" or event["return_code"] != 0:
                raise RuntimeError(
                    f"{name} failed: status={event['status']} "
                    f"return_code={event['return_code']}"
                )
            return event

        for runtime in (runtime_a, runtime_b):
            (runtime / "work").mkdir(parents=True, exist_ok=True)
        prediction_a_path = runtime_a / "predictions.json"
        prediction_b_path = runtime_b / "predictions.json"
        metadata_a_path = runtime_a / "inference-metadata.json"
        metadata_b_path = runtime_b / "inference-metadata.json"

        def inference_command(
            runtime: Path,
            inference_directory: Path,
            inference_config_path: Path,
            prediction_path: Path,
            metadata_path: Path,
        ) -> list[str]:
            return [
                sys.executable,
                str(adapter_snapshot),
                "--challenge",
                str(inference_directory / f"{task_id}.json"),
                "--config",
                str(inference_config_path),
                "--upstream-root",
                str(staged_upstream),
                "--output",
                str(prediction_path),
                "--metadata",
                str(metadata_path),
                "--write-root",
                str(runtime),
            ]

        inference_command_a = inference_command(
            runtime_a,
            inference_a,
            inference_config_a,
            prediction_a_path,
            metadata_a_path,
        )
        inference_command_b = inference_command(
            runtime_b,
            inference_b,
            inference_config_b,
            prediction_b_path,
            metadata_b_path,
        )
        forbidden_inference_strings = {
            str(scoring_a),
            str(scoring_b),
            str(frozen_solution_path),
            config["expected_solution_sha256"],
        }
        if any(
            forbidden in part
            for command in (inference_command_a, inference_command_b)
            for part in command
            for forbidden in forbidden_inference_strings
        ):
            raise ValueError("inference command exposes a hidden-label locator")
        inference_event_a = run_child(
            inference_command_a,
            name="inference-a",
            kind="inference",
            runtime=runtime_a,
            environment=child_environment,
        )
        inference_event_b = run_child(
            inference_command_b,
            name="inference-b",
            kind="inference",
            runtime=runtime_b,
            environment=child_environment,
        )

        predictions_a_bytes = prediction_a_path.read_bytes()
        predictions_b_bytes = prediction_b_path.read_bytes()
        if predictions_a_bytes != predictions_b_bytes:
            raise ValueError("A/B prediction bytes differ")
        predictions_a = read_json(prediction_a_path)
        predictions_b = read_json(prediction_b_path)
        validate_prediction_shape(
            predictions_a,
            task_count=task_count,
            outputs_total=outputs_total,
            top_k=top_k,
        )
        validate_prediction_shape(
            predictions_b,
            task_count=task_count,
            outputs_total=outputs_total,
            top_k=top_k,
        )
        metadata_a = read_json(metadata_a_path)
        metadata_b = read_json(metadata_b_path)
        for metadata in (metadata_a, metadata_b):
            required_metadata = {
                "method_id": "arc-nca",
                "config_id": config["config_id"],
                "device": "cpu",
                "gpu_api_called": False,
                "test_output_fields_received": 0,
                "scorer_imported": False,
                "parameter_count": config["adapter"]["expected_parameter_count"],
                "cuda_visible_devices": "",
            }
            for field, expected in required_metadata.items():
                if metadata.get(field) != expected:
                    raise ValueError(f"inference metadata mismatch: {field}")
            for field in ("python", "torch", "numpy"):
                if metadata.get(field) != config["expected_environment"][field]:
                    raise ValueError(f"inference environment mismatch: {field}")

        score_a_path = runtime_a / "score.json"
        score_b_path = runtime_b / "score.json"
        sentinel_score_a_path = output_directory / "scoring-only" / "sentinel-a.json"
        sentinel_score_b_path = output_directory / "scoring-only" / "sentinel-b.json"
        score_specs = [
            (prediction_a_path, scoring_a, score_a_path, "score-a", runtime_a),
            (prediction_b_path, scoring_b, score_b_path, "score-b", runtime_b),
            (sentinel_path, scoring_a, sentinel_score_a_path, "sentinel-a", runtime_a),
            (sentinel_path, scoring_b, sentinel_score_b_path, "sentinel-b", runtime_b),
        ]
        scoring_events: list[dict[str, object]] = []
        for prediction, answer_tree, destination, name, runtime in score_specs:
            command = [
                sys.executable,
                "-m",
                "arc_agi_eval",
                "score",
                str(prediction),
                str(answer_tree),
                "--top-k",
                str(top_k),
                "--json",
            ]
            scoring_events.append(
                run_child(
                    command,
                    name=name,
                    kind="scoring",
                    runtime=runtime,
                    environment=scoring_environment,
                    stdout_path=destination,
                )
            )

        latest_inference_end = max(
            parse_utc(inference_event_a["ended_at_utc"]),
            parse_utc(inference_event_b["ended_at_utc"]),
        )
        if any(
            parse_utc(event["started_at_utc"]) < latest_inference_end
            for event in scoring_events
        ):
            raise ValueError("scoring began before all inference exited")
        inference_pids = {inference_event_a["pid"], inference_event_b["pid"]}
        scoring_pids = {event["pid"] for event in scoring_events}
        if inference_pids & scoring_pids:
            raise ValueError("inference and scoring process IDs overlap")

        score_a = read_json(score_a_path)
        score_b = read_json(score_b_path)
        sentinel_score_a = read_json(sentinel_score_a_path)
        sentinel_score_b = read_json(sentinel_score_b_path)
        for score in (score_a, score_b, sentinel_score_a, sentinel_score_b):
            assert_score_contract(
                score, tasks=task_count, outputs=outputs_total, top_k=top_k
            )
        if sentinel_score_a["outputs_exact"] != outputs_total:
            raise ValueError("original-label sentinel is not perfect")
        if sentinel_score_b["outputs_exact"] != 0:
            raise ValueError("mutated-label sentinel did not change to zero exact")

        reference_a = score_reference_payload(
            predictions_a, answers_from_directory(scoring_a), top_k=top_k
        )
        reference_b = score_reference_payload(
            predictions_b, answers_from_directory(scoring_b), top_k=top_k
        )
        if not reference_a["agree"] or not reference_b["agree"]:
            raise ValueError("production and reference exact scorers disagree")
        reference_path = output_directory / "reference-scorer-check.json"
        atomic_json(
            reference_path,
            {
                "schema_version": 1,
                "a": reference_a,
                "b": reference_b,
                "all_agree": True,
                "computed_after_all_inference": True,
            },
        )

        checks = {
            "all_child_processes_exit_zero": all(
                event["status"] == "passed" and event["return_code"] == 0
                for event in events
            ),
            "all_child_processes_terminal": all(
                event["ended_at_utc"] and not event["timed_out"] for event in events
            ),
            "analyst_exposure_disclosed": config["analyst_test_label_exposure"],
            "configuration_frozen_before_execution": True,
            "hidden_label_trees_differ": tree_sha256(scoring_a)
            != tree_sha256(scoring_b),
            "inference_and_scoring_pids_disjoint": not (
                inference_pids & scoring_pids
            ),
            "inference_commands_exclude_hidden_label_locators": True,
            "inference_received_test_output_fields": 0,
            "inference_visible_trees_equal": tree_inventory(inference_a)
            == tree_inventory(inference_b),
            "label_mutation_changed_all_outputs": mutation_changed_outputs
            == outputs_total,
            "prediction_bytes_a_b_equal": predictions_a_bytes
            == predictions_b_bytes,
            "production_reference_scorers_agree": True,
            "scoring_after_all_inference": True,
            "sentinel_label_sensitivity": sentinel_score_a["outputs_exact"]
            == outputs_total
            and sentinel_score_b["outputs_exact"] == 0,
        }
        if not all(
            value is True
            or (
                key == "inference_received_test_output_fields" and value == 0
            )
            for key, value in checks.items()
        ):
            raise ValueError("one or more strict runtime checks failed")

        challenge_comparison_path = output_directory / "challenge-comparison.json"
        atomic_json(
            challenge_comparison_path,
            {
                "schema_version": 1,
                "task_id": task_id,
                "task_count": task_count,
                "test_output_denominator": outputs_total,
                "frozen_challenge_path": frozen_challenge_path.relative_to(
                    ROOT
                ).as_posix(),
                "frozen_challenge_sha256": sha256_file(frozen_challenge_path),
                "visible_tree_a_sha256": tree_sha256(inference_a),
                "visible_tree_b_sha256": tree_sha256(inference_b),
                "visible_trees_equal": tree_inventory(inference_a)
                == tree_inventory(inference_b),
                "hidden_scoring_tree_a_sha256": tree_sha256(scoring_a),
                "hidden_scoring_tree_b_sha256": tree_sha256(scoring_b),
                "hidden_scoring_trees_differ": tree_sha256(scoring_a)
                != tree_sha256(scoring_b),
                "label_mutation_changed_outputs": mutation_changed_outputs,
                "label_mutation_changed_cells": mutation_changed_cells,
            },
        )

        results_path = output_directory / "results-summary.json"
        atomic_json(
            results_path,
            {
                "schema_version": 1,
                "status": "passed",
                "scope": "arc-nca-reduced-method-specific-cpu-firewall-smoke",
                "checks": checks,
                "score_a": score_a,
                "score_b": score_b,
                "sentinel_score_a": sentinel_score_a,
                "sentinel_score_b": sentinel_score_b,
                "prediction_sha256": sha256_bytes(predictions_a_bytes),
                "prediction_bytes": len(predictions_a_bytes),
                "process_event_count": len(events),
                "inference_process_count": 2,
                "scoring_process_count": len(scoring_events),
                "performance_table_eligibility_changed": False,
                "public_execution_authorized": False,
                "analyst_test_label_exposure": True,
                "score_used_for_tuning": False,
            },
        )

        upstream_inventory_path = output_directory / "upstream-inventory.json"
        atomic_json(
            upstream_inventory_path,
            {
                "schema_version": 1,
                "source_path": config["upstream"]["repository_path"],
                "source_tree_sha256": canonical_sha256(upstream_inventory),
                "copied_tree_sha256": canonical_sha256(copied_inventory),
                "file_count": len(upstream_inventory),
                "files": upstream_inventory,
            },
        )
        source_patch_path = output_directory / "source-patch-manifest.json"
        supporting_sources = [*code_paths, args.config]
        atomic_json(
            source_patch_path,
            {
                "schema_version": 1,
                "scope": "ARC_NCA trusted reduced CPU adapter and strict A/B orchestrator",
                "upstream_ca_location": config["adapter"]["upstream_location"],
                "semantic_device_change": config["adapter"][
                    "semantic_device_change"
                ],
                "files": [
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in supporting_sources
                ],
                "executed_adapter_snapshot": {
                    "path": adapter_snapshot.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(adapter_snapshot),
                    "bytes": adapter_snapshot.stat().st_size,
                },
            },
        )
        source_lock_path = output_directory / "source-lock.json"
        atomic_json(
            source_lock_path,
            {
                "schema_version": 1,
                "repository": config["upstream"]["repository"],
                "revision": config["upstream"]["revision"],
                "source_tree_sha256": tree_sha256(upstream_root),
                "copied_tree_sha256": tree_sha256(staged_upstream),
                "inventory_path": upstream_inventory_path.relative_to(ROOT).as_posix(),
                "inventory_sha256": sha256_file(upstream_inventory_path),
                "license_path": (
                    staged_upstream / "LICENSE"
                ).relative_to(ROOT).as_posix(),
                "license_sha256": sha256_file(staged_upstream / "LICENSE"),
                "repository_revision": git_revision(),
                "repository_dirty": git_dirty(),
                "patch_manifest_path": source_patch_path.relative_to(ROOT).as_posix(),
                "patch_manifest_sha256": sha256_file(source_patch_path),
            },
        )
        artifact_manifest_path = output_directory / "artifact-manifest.json"
        atomic_json(
            artifact_manifest_path,
            {
                "schema_version": 1,
                "artifact_count": 0,
                "licenses_verified": True,
                "model_checkpoints": [],
                "source_license": {
                    "spdx": "Apache-2.0",
                    "path": (staged_upstream / "LICENSE").relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(staged_upstream / "LICENSE"),
                },
            },
        )
        data_manifest_path = output_directory / "data-manifest.json"
        atomic_json(
            data_manifest_path,
            {
                "schema_version": 1,
                "dataset_id": config["dataset_id"],
                "split": config["split"],
                "task_count": task_count,
                "test_output_denominator": outputs_total,
                "contamination_policy": "historical",
                "known_overlap_excluded_partition": True,
                "analyst_test_label_exposure": True,
                "score_used_for_tuning": False,
                "representative": representative,
                "frozen_manifest_path": frozen_manifest_path.relative_to(
                    ROOT
                ).as_posix(),
                "frozen_manifest_sha256": sha256_file(frozen_manifest_path),
                "frozen_run_path": frozen_run_path.relative_to(ROOT).as_posix(),
                "frozen_run_sha256": sha256_file(frozen_run_path),
                "frozen_solution_path": frozen_solution_path.relative_to(
                    ROOT
                ).as_posix(),
                "frozen_solution_sha256": sha256_file(frozen_solution_path),
                "visible_tree_sha256": tree_sha256(inference_a),
                "hidden_scoring_tree_a_sha256": tree_sha256(scoring_a),
                "hidden_scoring_tree_b_sha256": tree_sha256(scoring_b),
            },
        )
        hardware_path = output_directory / "hardware-manifest.json"
        atomic_json(
            hardware_path,
            {
                "schema_version": 1,
                "profile_id": "host-20260806-arc-nca-cpu-smoke",
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
                "gpu_requested": False,
                "cuda_visible_devices": "",
            },
        )
        status = "passed"
    except BaseException as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    usage = monitor.stop()

    if status != "passed":
        failed_record = {
            "schema_version": 1,
            "method_id": "arc-nca",
            "run_id": output_directory.name,
            "runner": "scripts.run_arc_nca_strict_smoke",
            "status": "failed",
            "scope": "arc-nca-reduced-method-specific-cpu-firewall-smoke",
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "error": failure,
            "process_events": events,
            "resources": usage.to_dict(),
        }
        atomic_json(output_directory / "run.json", failed_record)
        print(json.dumps(failed_record, indent=2, sort_keys=True))
        return 1

    try:
        resource_trace_path = output_directory / "resource-trace.json"
        atomic_json(
            resource_trace_path,
            {
                "schema_version": 1,
                "scope": "orchestrator-current-process-only",
                "children_included": False,
                "usage": usage.to_dict(),
            },
        )
        content_manifest_path = output_directory / "content-manifest.json"
        content_entries = []
        for path in sorted(output_directory.rglob("*")):
            if path.is_file() and path.name not in {
                "run.json",
                "candidate-run.json",
                "content-manifest.json",
            }:
                content_entries.append(
                    {
                        "path": path.relative_to(output_directory).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
        atomic_json(
            content_manifest_path,
            {
                "schema_version": 1,
                "file_count": len(content_entries),
                "files": content_entries,
            },
        )

        role_paths: dict[str, list[Path]] = {
            "protocol_manifest": [protocol_snapshot],
            "source_lock": [source_lock_path],
            "source_patch": [source_patch_path],
            "artifact_manifest": [artifact_manifest_path],
            "data_manifest": [data_manifest_path],
            "config": [config_copy],
            "environment_lock": [environment_path],
            "hardware_manifest": [hardware_path],
            "challenge_manifest": [inference_a / "MANIFEST"],
            "predictions": [prediction_a_path, prediction_b_path, sentinel_path],
            "results": [
                score_a_path,
                score_b_path,
                sentinel_score_a_path,
                sentinel_score_b_path,
                results_path,
                reference_path,
                challenge_comparison_path,
            ],
            "event_log": [event_log_path],
            "resource_trace": [resource_trace_path],
            "content_manifest": [content_manifest_path],
        }
        files = [file_record(SCHEMA_PATH, role="schema")]
        used_paths = {SCHEMA_PATH.resolve()}
        for role, paths in role_paths.items():
            for path in paths:
                files.append(file_record(path, role=role))
                used_paths.add(path.resolve())
        for path in sorted(output_directory.rglob("*")):
            if (
                path.is_file()
                and path.name not in {"run.json", "candidate-run.json"}
                and path.resolve() not in used_paths
            ):
                role = (
                    "stdout"
                    if path.name.endswith(".stdout.log")
                    else "stderr"
                    if path.name.endswith(".stderr.log")
                    else "other"
                )
                files.append(file_record(path, role=role))

        primary_metric = {
            "name": "output_exact_pass_at_k",
            "role": "primary",
            "top_k": top_k,
            "numerator": score_a["outputs_exact"],
            "denominator": score_a["outputs_total"],
            "value": score_a["output_exact_accuracy"],
            "denominator_policy": "the one predeclared frozen development test output",
        }
        secondary_metrics = [
            {
                "name": "strict_task_exact_pass_at_k",
                "role": "secondary",
                "top_k": top_k,
                "numerator": score_a["tasks_exact"],
                "denominator": score_a["tasks_total"],
                "value": score_a["task_exact_accuracy"],
                "denominator_policy": "the one predeclared frozen development task",
            },
            {
                "name": "micro_cell_accuracy",
                "role": "diagnostic",
                "top_k": top_k,
                "numerator": score_a["cells_correct"],
                "denominator": score_a["cells_total"],
                "value": score_a["cell_accuracy"],
                "denominator_policy": "all cells in the predeclared development output",
            },
        ]
        record = {
            "schema_version": "protocol-v1-run-1.0.0",
            "schema_digest_sha256": sha256_file(SCHEMA_PATH),
            "protocol_id": "arc-rebench-protocol-v1-draft",
            "protocol_digest_sha256": sha256_file(protocol_snapshot),
            "method_id": "arc-nca",
            "config_id": config["config_id"],
            "run_id": output_directory.name,
            "status": "passed",
            "evidence_scope": "solver_prediction_smoke",
            "parity_class": "reduced",
            "resource_class": "local_cpu",
            "code_trust_class": "trusted_locked",
            "claim": (
                "A frozen two-step CPU adaptation of ARC_NCA produced byte-identical "
                "Top-2 predictions in two inference-only processes over byte-identical, "
                "test-label-free copies of one predeclared development task. Independent "
                "scoring began only after both inference processes exited and passed "
                "hidden-label mutation, sentinel, and reference-scorer checks."
            ),
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "source": {
                "lock_digest_sha256": sha256_file(source_lock_path),
                "revision": config["upstream"]["revision"],
                "dirty": True,
                "patch_digest_sha256": sha256_file(source_patch_path),
            },
            "artifacts": {
                "manifest_digest_sha256": sha256_file(artifact_manifest_path),
                "artifact_count": 0,
                "licenses_verified": True,
            },
            "data": {
                "manifest_digest_sha256": sha256_file(data_manifest_path),
                "dataset_id": config["dataset_id"],
                "split": config["split"],
                "task_count": task_count,
                "contamination_policy": "historical",
            },
            "config": {
                "digest_sha256": sha256_file(config_copy),
                "seed": config["determinism"]["seed"],
                "deterministic": True,
            },
            "hardware": {
                "profile_id": "host-20260806-arc-nca-cpu-smoke",
                "manifest_digest_sha256": sha256_file(hardware_path),
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
            "execution": {
                "runner": "scripts.run_arc_nca_strict_smoke",
                "command": [
                    "python3",
                    "scripts/run_arc_nca_strict_smoke.py",
                    "--config",
                    args.config.relative_to(ROOT).as_posix(),
                    "--pre-run-input-bundle",
                    args.pre_run_input_bundle.relative_to(ROOT).as_posix(),
                    "--output-directory",
                    output_directory.relative_to(ROOT).as_posix(),
                ],
                "working_directory": ".",
                "environment_digest_sha256": sha256_file(environment_path),
                "claim_execution_started": True,
                "target_code_executed": True,
                "network_used": False,
                "gpu_requested": False,
            },
            "challenge_firewall": {
                "challenge_manifest_digest_sha256": sha256_file(
                    inference_a / "MANIFEST"
                ),
                "inference_received_test_labels": False,
                "inference_started": True,
                "scoring_after_inference": True,
                "label_mutation_check": "passed",
                "network_policy": "denied",
                "write_policy": "run_directory_only",
                "security_isolation": "trusted_process",
            },
            "attempt_budget": {
                "top_k": top_k,
                "timeout_seconds": timeout_seconds,
                "max_retries": config["attempt_budget"]["max_retries"],
                "max_candidates": config["attempt_budget"]["max_candidates"],
                "api_call_cap": 0,
                "input_token_cap": 0,
                "output_token_cap": 0,
                "cost_cap_usd": 0,
            },
            "resources": {
                "accounting_scope": "current_process",
                "child_processes_observed": True,
                "children_included": False,
                "sampling": "sampled",
                "wall_time_seconds": usage.wall_time_seconds,
                "cpu_seconds": usage.process_cpu_seconds,
                "peak_rss_bytes": usage.sampled_peak_current_rss_bytes,
                "peak_vram_bytes": None,
                "energy_joules": None,
                "disk_delta_bytes": None,
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
            },
            "results": {
                "kind": "arc_predictions",
                "predictions_path": prediction_a_path.relative_to(ROOT).as_posix(),
                "score_path": score_a_path.relative_to(ROOT).as_posix(),
                "primary_metric": primary_metric,
                "secondary_metrics": secondary_metrics,
                "checks": [
                    {
                        "name": "ab-label-firewall",
                        "status": "passed",
                        "detail": (
                            "A/B inference trees and prediction bytes match; all hidden "
                            "outputs changed, and the scoring-only sentinel changed from "
                            f"{outputs_total}/{outputs_total} to 0/{outputs_total}."
                        ),
                    },
                    {
                        "name": "independent-scoring-lifecycle",
                        "status": "passed",
                        "detail": (
                            "Two inference and four scoring processes have terminal, "
                            "disjoint PIDs; every scorer started after both inference "
                            "processes exited."
                        ),
                    },
                    {
                        "name": "reference-exact-score",
                        "status": "passed",
                        "detail": (
                            "Production and independently implemented exact-match scorers "
                            "agree on original and mutated hidden labels."
                        ),
                    },
                    {
                        "name": "analyst-exposure-boundary",
                        "status": "passed",
                        "detail": (
                            "The public training label exposure is disclosed; the frozen "
                            "configuration was not changed from the observed score, and this "
                            "record makes no benchmark or performance claim."
                        ),
                    },
                ],
            },
            "failures": {"count": 0, "items": []},
            "files": files,
            "limitations": [
                "This is a two-step CPU mechanism/firewall smoke on one public training task, not an ARC-AGI benchmark or paper reproduction.",
                "The public training answer became visible to the analyst after task selection; the score was not used to tune this frozen configuration and cannot support a performance claim.",
                "The adapter copies the upstream CA architecture but changes its hard-coded CUDA update-mask allocation to the input tensor device and fixes the ARC palette at ten colors.",
                "The run uses a CUDA-enabled PyTorch build on CPU, not the upstream environment or a CPU-only wheel.",
                "No filesystem, network, or GPU-device namespace is available; label, network, and write restrictions are trusted-code policies rather than kernel-enforced isolation.",
                "Resource counters cover the orchestrator only; child inference and scoring CPU/RSS are excluded, so the required process-tree resource gate remains pending.",
                "A/B equality proves independence from the hidden labels mutated here, not absence of historical, analyst, or pretrained-model contamination.",
                "The pre-run input bundle authorizes no locked-public solver execution, and this development smoke does not change that authorization.",
            ],
        }
        candidate_path = output_directory / "candidate-run.json"
        atomic_json(candidate_path, record)
        validation = validate_run_file(
            candidate_path,
            schema_path=SCHEMA_PATH,
            repo_root=ROOT,
            verify_files=True,
        )
        os.replace(candidate_path, output_directory / "run.json")
        print(json.dumps(validation.as_dict(), indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure_path = output_directory / "validation-failure.json"
        atomic_json(
            failure_path,
            {
                "schema_version": 1,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        failed_record = {
            "schema_version": 1,
            "method_id": "arc-nca",
            "run_id": output_directory.name,
            "runner": "scripts.run_arc_nca_strict_smoke",
            "status": "failed",
            "scope": "terminal-record-validation",
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "error": {"type": type(error).__name__, "message": str(error)},
            "process_events": events,
            "resources": usage.to_dict(),
        }
        atomic_json(output_directory / "run.json", failed_record)
        print(json.dumps(failed_record, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
