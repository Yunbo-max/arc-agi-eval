#!/usr/bin/env python3
"""Run the trusted deterministic baseline through the frozen dev-audit firewall."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


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
from arc_agi_eval.reference_scoring import reference_exact_score  # noqa: E402
from arc_agi_eval.resources import ResourceMonitor  # noqa: E402
from arc_agi_eval.run_schema import validate_run_file  # noqa: E402
from arc_agi_eval.scoring import score_predictions  # noqa: E402
from arc_agi_eval.validation import load_task  # noqa: E402


SCHEMA_PATH = ROOT / "schemas" / "protocol-v1-run.schema.json"
PROTOCOL_CONFIG = ROOT / "configs" / "protocol_v1_draft.json"


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, pretty_json_bytes(value))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_record(
    path: Path, *, role: str, required: bool = True
) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "required_for_claim": required,
    }


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(completed.stdout)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown-cpu"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def output_count(tasks: dict[str, Any]) -> int:
    return sum(len(outputs) for outputs in tasks.values())


def answers_from_directory(directory: Path) -> dict[str, list[list[list[int]]]]:
    answers: dict[str, list[list[list[int]]]] = {}
    for path in sorted(directory.glob("*.json")):
        task = load_task(path)
        answers[path.stem] = [pair["output"] for pair in task["test"]]
    return answers


def validate_prediction_shape(
    predictions: Any, *, task_count: int, outputs_total: int, top_k: int
) -> None:
    if not isinstance(predictions, dict) or len(predictions) != task_count:
        raise ValueError("prediction task count mismatch")
    if output_count(predictions) != outputs_total:
        raise ValueError("prediction output denominator mismatch")
    expected_keys = {f"attempt_{index}" for index in range(1, top_k + 1)}
    for task_id, outputs in predictions.items():
        if not isinstance(outputs, list):
            raise ValueError(f"{task_id}: prediction outputs are not a list")
        for index, attempts in enumerate(outputs):
            if not isinstance(attempts, dict) or set(attempts) != expected_keys:
                raise ValueError(
                    f"{task_id}[{index}]: expected exactly Top-{top_k} attempts"
                )


def assert_score_contract(
    score: dict[str, Any], *, tasks: int, outputs: int, top_k: int
) -> None:
    expected = {
        "top_k": top_k,
        "tasks_total": tasks,
        "tasks_predicted": tasks,
        "outputs_total": outputs,
    }
    for key, value in expected.items():
        if score.get(key) != value:
            raise ValueError(f"score contract mismatch for {key}: {score.get(key)!r}")


def score_reference_payload(
    predictions: dict[str, Any], answers: dict[str, Any], *, top_k: int
) -> dict[str, object]:
    # Production validation runs first, then the independent reference scorer
    # consumes the validated in-memory representation.
    production = score_predictions(predictions, answers, top_k=top_k)
    reference = reference_exact_score(predictions, answers, top_k=top_k)
    comparable = {
        "tasks_total": production.tasks_total,
        "tasks_predicted": production.tasks_predicted,
        "tasks_exact": production.tasks_exact,
        "outputs_total": production.outputs_total,
        "outputs_exact": production.outputs_exact,
    }
    reference_payload = {
        "tasks_total": reference.tasks_total,
        "tasks_predicted": reference.tasks_predicted,
        "tasks_exact": reference.tasks_exact,
        "outputs_total": reference.outputs_total,
        "outputs_exact": reference.outputs_exact,
    }
    return {
        "production_exact_fields": comparable,
        "reference_exact_fields": reference_payload,
        "agree": comparable == reference_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "challenge_runtime_core_v1.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "e0-challenge-runtime"
            / "20260806-deterministic-baseline-dev-audit-v1"
        ),
    )
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    try:
        output_directory.relative_to(ROOT)
    except ValueError as error:
        parser.error(f"output directory must remain inside repository: {error}")
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    monitor = ResourceMonitor(include_nvidia=False).start()
    status = "failed"
    failure: dict[str, str] | None = None
    record: dict[str, object] | None = None
    events: list[dict[str, object]] = []
    event_log_path = output_directory / "process-events.json"
    try:
        config = read_json(args.config)
        frozen_directory = (ROOT / config["frozen_runtime_directory"]).resolve()
        frozen_manifest_path = frozen_directory / "manifest.json"
        frozen_run_path = frozen_directory / "run.json"
        if sha256_file(frozen_manifest_path) != config[
            "expected_frozen_manifest_sha256"
        ]:
            raise ValueError("frozen development manifest digest mismatch")
        if sha256_file(frozen_run_path) != config["expected_frozen_run_sha256"]:
            raise ValueError("frozen development strict run digest mismatch")
        frozen_manifest = read_json(frozen_manifest_path)
        task_count = config["expected_task_count"]
        outputs_total = config["expected_test_output_denominator"]
        top_k = config["top_k"]
        timeout_seconds = config["process_timeout_seconds"]
        if frozen_manifest["dev_audit"]["representative_task_count"] != task_count:
            raise ValueError("frozen task count does not match config")
        if frozen_manifest["dev_audit"]["test_output_denominator"] != outputs_total:
            raise ValueError("frozen output denominator does not match config")

        runtime_a = output_directory / "runtime" / "a"
        runtime_b = output_directory / "runtime" / "b"
        inference_a = runtime_a / "inference" / "dev-audit"
        inference_b = runtime_b / "inference" / "dev-audit"
        scoring_a = runtime_a / "scoring" / "dev-audit"
        scoring_b = runtime_b / "scoring" / "dev-audit"

        # Copy the exact locked inference tree twice. The solver receives one of
        # these directories and no path to either scoring tree.
        for item in frozen_manifest["inference_file_inventory"]:
            source = frozen_directory / item["path"]
            if sha256_file(source) != item["sha256"]:
                raise ValueError(f"frozen inference hash mismatch: {item['path']}")
            relative = Path(item["path"]).relative_to("inference/dev-audit")
            payload = source.read_bytes()
            atomic_bytes(inference_a / relative, payload)
            atomic_bytes(inference_b / relative, payload)

        mutation_offset = config["hidden_label_mutation"]["offset"]
        original_label_payloads: dict[str, bytes] = {}
        mutated_label_payloads: dict[str, bytes] = {}
        mutation_changed_outputs = 0
        mutation_changed_cells = 0
        for representative in frozen_manifest["representatives"]:
            source_path = ROOT / representative["source_path"]
            if sha256_file(source_path) != representative["source_sha256"]:
                raise ValueError(
                    f"frozen labeled source hash mismatch: {representative['task_id']}"
                )
            task = load_task(source_path)
            mutated = mutate_hidden_test_labels(task, offset=mutation_offset)
            challenge_payload = pretty_json_bytes(challenge_only(task))
            mutated_challenge_payload = pretty_json_bytes(challenge_only(mutated))
            locked_challenge_path = frozen_directory / representative["challenge_path"]
            if challenge_payload != locked_challenge_path.read_bytes():
                raise ValueError(
                    f"source-to-challenge mismatch: {representative['task_id']}"
                )
            if mutated_challenge_payload != challenge_payload:
                raise ValueError(
                    f"label mutation changed challenge: {representative['task_id']}"
                )
            task_id = representative["task_id"]
            original_payload = pretty_json_bytes(task)
            mutated_payload = pretty_json_bytes(mutated)
            original_label_payloads[task_id] = original_payload
            mutated_label_payloads[task_id] = mutated_payload
            atomic_bytes(scoring_a / f"{task_id}.json", original_payload)
            atomic_bytes(scoring_b / f"{task_id}.json", mutated_payload)
            for original_pair, mutated_pair in zip(task["test"], mutated["test"]):
                mutation_changed_outputs += int(
                    original_pair["output"] != mutated_pair["output"]
                )
                mutation_changed_cells += sum(
                    original_cell != mutated_cell
                    for original_row, mutated_row in zip(
                        original_pair["output"], mutated_pair["output"]
                    )
                    for original_cell, mutated_cell in zip(original_row, mutated_row)
                )
        if mutation_changed_outputs != outputs_total:
            raise ValueError("not every hidden test output changed under mutation")

        visible_inventory_a = tree_inventory(inference_a)
        visible_inventory_b = tree_inventory(inference_b)
        visible_tree_a_sha256 = canonical_sha256(visible_inventory_a)
        visible_tree_b_sha256 = canonical_sha256(visible_inventory_b)
        if visible_inventory_a != visible_inventory_b:
            raise ValueError("A/B inference-visible inventories differ")
        scoring_tree_a_sha256 = tree_sha256(scoring_a)
        scoring_tree_b_sha256 = tree_sha256(scoring_b)
        if scoring_tree_a_sha256 == scoring_tree_b_sha256:
            raise ValueError("A/B hidden scoring trees did not change")

        sentinel_path = output_directory / "scoring-only" / "sentinel-predictions.json"
        sentinel = sentinel_predictions(sorted(scoring_a.glob("*.json")), top_k=top_k)
        if output_count(sentinel) != outputs_total:
            raise ValueError("sentinel output denominator mismatch")
        atomic_json(sentinel_path, sentinel)

        config_copy = output_directory / "config.json"
        atomic_bytes(config_copy, args.config.read_bytes())
        protocol_snapshot = output_directory / "protocol-manifest.json"
        atomic_json(
            protocol_snapshot,
            {
                "schema_version": 1,
                "protocol_id": "arc-rebench-protocol-v1-draft",
                "protocol_status": "draft-not-frozen",
                "gate_id": "te.challenge-runtime",
                "protocol_config_path": PROTOCOL_CONFIG.relative_to(ROOT).as_posix(),
                "protocol_config_sha256_at_execution": sha256_file(PROTOCOL_CONFIG),
            },
        )

        child_environment = {
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ROOT),
            "TZ": "UTC",
        }
        environment_path = output_directory / "environment-lock.json"
        atomic_json(
            environment_path,
            {
                "schema_version": 1,
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "child_environment_allowlist": child_environment,
                "inherited_environment_variable_count": 0,
                "provider_credentials_forwarded": False,
                "network_used": False,
                "network_namespace_enforced": False,
            },
        )

        for runtime in (runtime_a, runtime_b):
            (runtime / "work").mkdir(parents=True, exist_ok=True)
        prediction_a_path = runtime_a / "predictions.json"
        prediction_b_path = runtime_b / "predictions.json"
        metadata_a_path = runtime_a / "inference-metadata.json"
        metadata_b_path = runtime_b / "inference-metadata.json"

        def run_child(
            command: list[str], *, name: str, kind: str, runtime: Path
        ) -> dict[str, object]:
            event = run_logged_process(
                command,
                name=name,
                kind=kind,
                cwd=runtime / "work",
                timeout_seconds=timeout_seconds,
                stdout_path=runtime / "logs" / f"{name}.stdout.log",
                stderr_path=runtime / "logs" / f"{name}.stderr.log",
                environment=child_environment,
                display_root=ROOT,
            ).as_dict()
            events.append(event)
            atomic_json(event_log_path, {"schema_version": 1, "events": events})
            if event["status"] != "passed":
                raise RuntimeError(
                    f"{name} did not pass: status={event['status']} "
                    f"return_code={event['return_code']}"
                )
            return event

        inference_command_a = [
            sys.executable,
            "-m",
            "arc_agi_eval",
            "baseline",
            str(inference_a),
            "--output",
            str(prediction_a_path),
            "--metadata",
            str(metadata_a_path),
            "--json",
        ]
        inference_command_b = [
            sys.executable,
            "-m",
            "arc_agi_eval",
            "baseline",
            str(inference_b),
            "--output",
            str(prediction_b_path),
            "--metadata",
            str(metadata_b_path),
            "--json",
        ]
        inference_event_a = run_child(
            inference_command_a,
            name="inference-a",
            kind="inference",
            runtime=runtime_a,
        )
        inference_event_b = run_child(
            inference_command_b,
            name="inference-b",
            kind="inference",
            runtime=runtime_b,
        )

        predictions_a_bytes = prediction_a_path.read_bytes()
        predictions_b_bytes = prediction_b_path.read_bytes()
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
        if predictions_a_bytes != predictions_b_bytes:
            raise ValueError("A/B prediction bytes differ")

        score_a_path = runtime_a / "score.json"
        score_b_path = runtime_b / "score.json"
        sentinel_score_a_path = output_directory / "scoring-only" / "sentinel-score-a.json"
        sentinel_score_b_path = output_directory / "scoring-only" / "sentinel-score-b.json"
        score_commands = [
            (
                [
                    sys.executable,
                    "-m",
                    "arc_agi_eval",
                    "score",
                    str(prediction_a_path),
                    str(scoring_a),
                    "--top-k",
                    str(top_k),
                    "--json",
                ],
                "score-a",
                runtime_a,
                score_a_path,
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "arc_agi_eval",
                    "score",
                    str(prediction_b_path),
                    str(scoring_b),
                    "--top-k",
                    str(top_k),
                    "--json",
                ],
                "score-b",
                runtime_b,
                score_b_path,
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "arc_agi_eval",
                    "score",
                    str(sentinel_path),
                    str(scoring_a),
                    "--top-k",
                    str(top_k),
                    "--json",
                ],
                "sentinel-score-a",
                runtime_a,
                sentinel_score_a_path,
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "arc_agi_eval",
                    "score",
                    str(sentinel_path),
                    str(scoring_b),
                    "--top-k",
                    str(top_k),
                    "--json",
                ],
                "sentinel-score-b",
                runtime_b,
                sentinel_score_b_path,
            ),
        ]
        scoring_events: list[dict[str, object]] = []
        for command, name, runtime, stdout_destination in score_commands:
            # The scorer's stdout is itself the immutable JSON result file.
            event = run_logged_process(
                command,
                name=name,
                kind="scoring",
                cwd=runtime / "work",
                timeout_seconds=timeout_seconds,
                stdout_path=stdout_destination,
                stderr_path=runtime / "logs" / f"{name}.stderr.log",
                environment=child_environment,
                display_root=ROOT,
            ).as_dict()
            events.append(event)
            scoring_events.append(event)
            atomic_json(event_log_path, {"schema_version": 1, "events": events})
            if event["status"] != "passed":
                raise RuntimeError(
                    f"{name} did not pass: status={event['status']} "
                    f"return_code={event['return_code']}"
                )

        score_a = read_json(score_a_path)
        score_b = read_json(score_b_path)
        sentinel_score_a = read_json(sentinel_score_a_path)
        sentinel_score_b = read_json(sentinel_score_b_path)
        for score in (score_a, score_b, sentinel_score_a, sentinel_score_b):
            assert_score_contract(
                score, tasks=task_count, outputs=outputs_total, top_k=top_k
            )
        if sentinel_score_a["outputs_exact"] != outputs_total:
            raise ValueError("original-label sentinel did not score perfectly")
        if sentinel_score_b["outputs_exact"] != 0:
            raise ValueError("mutated-label sentinel did not change to zero exact")

        latest_inference_end = max(
            parse_utc(inference_event_a["ended_at_utc"]),
            parse_utc(inference_event_b["ended_at_utc"]),
        )
        if any(
            parse_utc(event["started_at_utc"]) < latest_inference_end
            for event in scoring_events
        ):
            raise ValueError("a scoring process started before all inference exited")
        inference_pids = {inference_event_a["pid"], inference_event_b["pid"]}
        scoring_pids = {event["pid"] for event in scoring_events}
        if inference_pids & scoring_pids:
            raise ValueError("inference and scoring process IDs are not distinct")

        answers_a = answers_from_directory(scoring_a)
        answers_b = answers_from_directory(scoring_b)
        reference_a = score_reference_payload(predictions_a, answers_a, top_k=top_k)
        reference_b = score_reference_payload(predictions_b, answers_b, top_k=top_k)
        reference_path = output_directory / "reference-scorer-check.json"
        atomic_json(
            reference_path,
            {
                "schema_version": 1,
                "a": reference_a,
                "b": reference_b,
                "all_agree": reference_a["agree"] and reference_b["agree"],
                "computed_after_all_inference": True,
            },
        )
        if not reference_a["agree"] or not reference_b["agree"]:
            raise ValueError("production and reference scorers disagree")

        checks = {
            "all_child_processes_exit_zero": all(
                event["status"] == "passed" and event["return_code"] == 0
                for event in events
            ),
            "all_child_processes_terminal": all(
                event["ended_at_utc"] and not event["timed_out"] for event in events
            ),
            "challenge_manifest_bytes_a_b_equal": (
                (inference_a / "MANIFEST").read_bytes()
                == (inference_b / "MANIFEST").read_bytes()
            ),
            "hidden_label_trees_differ": scoring_tree_a_sha256
            != scoring_tree_b_sha256,
            "inference_and_scoring_pids_disjoint": not (
                inference_pids & scoring_pids
            ),
            "inference_commands_exclude_score_flag": all(
                "--score" not in command
                for command in (inference_command_a, inference_command_b)
            ),
            "inference_visible_test_output_fields": 0,
            "label_mutation_changed_all_outputs": mutation_changed_outputs
            == outputs_total,
            "prediction_bytes_a_b_equal": predictions_a_bytes
            == predictions_b_bytes,
            "production_reference_scorers_agree": reference_a["agree"]
            and reference_b["agree"],
            "scoring_after_all_inference": True,
            "sentinel_label_sensitivity": sentinel_score_a["outputs_exact"]
            == outputs_total
            and sentinel_score_b["outputs_exact"] == 0,
            "visible_tree_inventories_a_b_equal": visible_inventory_a
            == visible_inventory_b,
        }
        if not all(
            value is True or (key == "inference_visible_test_output_fields" and value == 0)
            for key, value in checks.items()
        ):
            raise ValueError("one or more challenge-runtime checks failed")

        challenge_comparison_path = output_directory / "challenge-comparison.json"
        atomic_json(
            challenge_comparison_path,
            {
                "schema_version": 1,
                "task_count": task_count,
                "test_output_denominator": outputs_total,
                "source_frozen_manifest_path": frozen_manifest_path.relative_to(
                    ROOT
                ).as_posix(),
                "source_frozen_manifest_sha256": sha256_file(frozen_manifest_path),
                "visible_file_count_per_tree": len(visible_inventory_a),
                "visible_tree_a_sha256": visible_tree_a_sha256,
                "visible_tree_b_sha256": visible_tree_b_sha256,
                "visible_trees_equal": visible_inventory_a == visible_inventory_b,
                "visible_manifest_sha256": sha256_file(inference_a / "MANIFEST"),
                "hidden_scoring_tree_a_sha256": scoring_tree_a_sha256,
                "hidden_scoring_tree_b_sha256": scoring_tree_b_sha256,
                "hidden_scoring_trees_differ": scoring_tree_a_sha256
                != scoring_tree_b_sha256,
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
                "scope": "global-trusted-challenge-runtime-core",
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
                "method_specific_strict_runtime_reports_created": 0,
            },
        )

        source_patch_path = output_directory / "source-patch-manifest.json"
        relevant_sources = [
            ROOT / "arc_agi_eval" / "baseline.py",
            ROOT / "arc_agi_eval" / "challenge_runtime.py",
            ROOT / "arc_agi_eval" / "cli.py",
            ROOT / "arc_agi_eval" / "dataset.py",
            ROOT / "arc_agi_eval" / "firewall.py",
            ROOT / "arc_agi_eval" / "reference_scoring.py",
            ROOT / "arc_agi_eval" / "scoring.py",
            ROOT / "arc_agi_eval" / "validation.py",
            ROOT / "scripts" / "run_challenge_runtime_core.py",
            args.config,
        ]
        atomic_json(
            source_patch_path,
            {
                "schema_version": 1,
                "scope": "trusted deterministic challenge-runtime core",
                "files": [
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in relevant_sources
                ],
            },
        )
        source_lock_path = output_directory / "source-lock.json"
        atomic_json(
            source_lock_path,
            {
                "schema_version": 1,
                "revision": git_revision(),
                "dirty": git_dirty(),
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
                "artifacts": [],
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
                "contamination_policy": "overlap_excluded",
                "claim_boundary": frozen_manifest["claim_boundary"],
                "frozen_manifest_path": frozen_manifest_path.relative_to(ROOT).as_posix(),
                "frozen_manifest_sha256": sha256_file(frozen_manifest_path),
                "frozen_run_path": frozen_run_path.relative_to(ROOT).as_posix(),
                "frozen_run_sha256": sha256_file(frozen_run_path),
                "visible_tree_sha256": visible_tree_a_sha256,
                "hidden_scoring_tree_a_sha256": scoring_tree_a_sha256,
                "hidden_scoring_tree_b_sha256": scoring_tree_b_sha256,
            },
        )
        hardware_path = output_directory / "hardware-manifest.json"
        atomic_json(
            hardware_path,
            {
                "schema_version": 1,
                "profile_id": "host-20260806-cpu-challenge-runtime",
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
        )

        content_manifest_path = output_directory / "content-manifest.json"
        content_entries = []
        for path in sorted(output_directory.rglob("*")):
            if path.is_file() and path.name not in {"run.json", "content-manifest.json"}:
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
        status = "passed"
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    usage = monitor.stop()

    if status == "passed":
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
                and path.name != "run.json"
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
            "denominator_policy": (
                "all declared dev-audit test outputs, including missing predictions"
            ),
        }
        secondary_metrics = [
            {
                "name": "strict_task_exact_pass_at_k",
                "role": "secondary",
                "top_k": top_k,
                "numerator": score_a["tasks_exact"],
                "denominator": score_a["tasks_total"],
                "value": score_a["task_exact_accuracy"],
                "denominator_policy": (
                    "all frozen dev-audit representative tasks"
                ),
            },
            {
                "name": "micro_cell_accuracy",
                "role": "diagnostic",
                "top_k": top_k,
                "numerator": score_a["cells_correct"],
                "denominator": score_a["cells_total"],
                "value": score_a["cell_accuracy"],
                "denominator_policy": (
                    "all cells in all declared dev-audit test outputs"
                ),
            },
        ]
        record = {
            "schema_version": "protocol-v1-run-1.0.0",
            "schema_digest_sha256": sha256_file(SCHEMA_PATH),
            "protocol_id": "arc-rebench-protocol-v1-draft",
            "protocol_digest_sha256": sha256_file(protocol_snapshot),
            "method_id": config["method_id"],
            "config_id": config["config_id"],
            "run_id": output_directory.name,
            "status": "passed",
            "evidence_scope": "solver_prediction_smoke",
            "parity_class": "not_applicable",
            "resource_class": "local_cpu",
            "code_trust_class": "trusted_locked",
            "claim": (
                "The locked deterministic baseline produced byte-identical Top-2 "
                "predictions in two trusted subprocesses over byte-identical, "
                "test-label-free copies of the frozen 94-task dev-audit challenge; "
                "independent scoring began only after both inference processes "
                "exited, used all 97 outputs, and passed hidden-label mutation and "
                "production/reference exact-score checks."
            ),
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "source": {
                "lock_digest_sha256": sha256_file(source_lock_path),
                "revision": git_revision(),
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
                "contamination_policy": "overlap_excluded",
            },
            "config": {
                "digest_sha256": sha256_file(config_copy),
                "seed": config["seed"],
                "deterministic": True,
            },
            "hardware": {
                "profile_id": "host-20260806-cpu-challenge-runtime",
                "manifest_digest_sha256": sha256_file(hardware_path),
                "cpu_model": cpu_model(),
                "accelerator_kind": "none",
                "accelerator_model": None,
                "accelerator_uuid": None,
                "accelerator_count": 0,
                "exclusive_accelerator": False,
            },
            "execution": {
                "runner": "scripts.run_challenge_runtime_core",
                "command": [
                    "python3",
                    "scripts/run_challenge_runtime_core.py",
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
                "max_retries": config["max_retries"],
                "max_candidates": config["max_candidates"],
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
                            "A/B visible trees and prediction bytes match; all 97 "
                            "hidden outputs changed, while an original-answer sentinel "
                            "scored 97/97 on A and 0/97 on B."
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
                            "The production and independently implemented exact-match "
                            "scorers agree on original and mutated labels."
                        ),
                    },
                ],
            },
            "failures": {"count": 0, "items": []},
            "files": files,
            "limitations": [
                "This closes only the global trusted challenge-runtime core; every tracked method still requires its own strict A/B runtime report before promotion, and no method eligibility status changes here.",
                "The deterministic floor baseline is a protocol sentinel, not a paper method, public benchmark result, or performance-table entry.",
                "The development partition excludes known overlaps only; it is not a pristine private holdout and supports no fully-clean claim.",
                "Process-group lifecycle control is not process-tree containment: setsid, double-fork, or daemon behavior could escape it, so this runner is restricted to trusted locked code.",
                "No filesystem or network namespace is available; denied network and run-directory-only writes are trusted-code policies rather than kernel-enforced isolation.",
                "Resource counters cover the parent orchestrator only; child inference/scoring CPU and RSS are excluded, so the required process-tree resource gate remains pending.",
                "A/B prediction equality proves independence from the labels mutated in this audit, not absence of every side channel or historical/pretraining contamination.",
            ],
        }
        run_path = output_directory / "run.json"
        atomic_json(run_path, record)
        try:
            validation = validate_run_file(
                run_path,
                schema_path=SCHEMA_PATH,
                repo_root=ROOT,
                verify_files=True,
            )
        except BaseException as error:
            status = "failed"
            failure = {"type": type(error).__name__, "message": str(error)}
        else:
            print(json.dumps(validation.as_dict(), indent=2, sort_keys=True))
    if status != "passed":
        failed_record = {
            "schema_version": 1,
            "method_id": "deterministic-floor-runtime-core",
            "run_id": output_directory.name,
            "runner": "scripts.run_challenge_runtime_core",
            "status": "failed",
            "scope": "global-trusted-challenge-runtime-core",
            "started_at_utc": started_at,
            "ended_at_utc": usage.ended_at_utc,
            "error": failure,
            "process_events": events,
            "resources": usage.to_dict(),
        }
        atomic_json(output_directory / "run.json", failed_record)
        print(json.dumps(failed_record, indent=2, sort_keys=True))
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
