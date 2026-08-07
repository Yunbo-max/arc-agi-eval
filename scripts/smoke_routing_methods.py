#!/usr/bin/env python3
"""Run zero-dollar, fail-closed source component checks for routing papers."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator


SPECS = {
    "graphplanner": {
        "revision": "56010b1f43c6096ac1a87736c7a2e110e368ecb7",
        "paper_year": 2026,
        "native_scope": "text/code/QA routing; arc_challenge is AI2 ARC-Challenge, not ARC-AGI grids",
    },
    "routemoa": {
        "revision": "8d07c48747da0e25adbe3df11e53fb6422b40bc7",
        "paper_year": 2026,
        "native_scope": "language/QA/reasoning/generation routing, not ARC-AGI grids",
    },
    "maca": {
        "revision": "62bd012aa28785237eef09d8a4c251695ab67c01",
        "paper_year": 2026,
        "native_scope": "multi-agent code/QA/math orchestration; ARC-C is AI2 ARC-Challenge, not ARC-AGI grids",
    },
}

PROVIDER_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NVIDIA_API_KEY",
    "NVIDIA_API_KEYS",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
)

LICENSE_NAMES = {
    "copying",
    "copying.md",
    "copying.txt",
    "license",
    "license.md",
    "license.txt",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def ensure_no_provider_credentials() -> None:
    present = sorted(key for key in PROVIDER_CREDENTIALS if os.environ.get(key))
    if present:
        raise RuntimeError(
            "provider credentials must be absent for this smoke: " + ", ".join(present)
        )


@contextmanager
def network_guard() -> Iterator[None]:
    original_create_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("network disabled by zero-dollar routing smoke")

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.socket.connect = blocked  # type: ignore[assignment]
    socket.socket.connect_ex = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.create_connection = original_create_connection
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


def license_record(source: Path) -> dict[str, object]:
    paths = sorted(
        path for path in source.iterdir() if path.is_file() and path.name.lower() in LICENSE_NAMES
    )
    detected = "unresolved"
    if paths:
        text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
        if "Apache License" in text and "Version 2.0" in text:
            detected = "Apache-2.0"
    return {
        "root_files": [path.name for path in paths],
        "detected": detected,
        "reuse_blocked_pending_clarification": not paths,
    }


def source_record(method_id: str, source: Path) -> dict[str, object]:
    expected = SPECS[method_id]["revision"]
    revision = git(source, "rev-parse", "HEAD")
    dirty_paths = git(source, "status", "--porcelain").splitlines()
    if revision != expected:
        raise RuntimeError(f"expected revision {expected}, found {revision}")
    if dirty_paths:
        raise RuntimeError(f"locked source is dirty: {dirty_paths[:5]}")
    return {
        "path": str(source),
        "expected_revision": expected,
        "observed_revision": revision,
        "dirty_paths": dirty_paths,
        "license": license_record(source),
    }


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graphplanner_smoke(source: Path) -> dict[str, object]:
    expected_columns = {
        "task_name",
        "query",
        "gt",
        "metric",
        "choices",
        "query_embedding",
        "task_id",
    }
    datasets: dict[str, object] = {}
    for name in ("router_data_train.csv", "router_data_test.csv"):
        path = source / "data" / name
        counts: Counter[str] = Counter()
        rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if columns != expected_columns:
                raise RuntimeError(f"{name}: unexpected columns {sorted(columns)}")
            for row in reader:
                rows += 1
                counts[row["task_name"]] += 1
        datasets[name] = {
            "rows": rows,
            "task_counts": dict(sorted(counts.items())),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "contains_native_ground_truth_column": True,
        }

    prompting = import_file(
        "graphplanner_task_prompting_smoke",
        source / "router_planner" / "shared" / "task_prompting.py",
    )
    secret = "SCHEMA_SMOKE_SECRET_LABEL"
    prompt = prompting.generate_task_query(
        "arc_challenge",
        {
            "query": "Which option is correct?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["one", "two", "three", "four"],
            },
            "gt": secret,
        },
    )
    if secret in prompt:
        raise RuntimeError("ground-truth sentinel leaked into generated prompt")
    return {
        "scope": "zero-dollar-bundled-router-data-schema-and-prompt-component",
        "datasets": datasets,
        "prompt_component": {
            "task": "arc_challenge",
            "prompt_characters": len(prompt),
            "ground_truth_sentinel_present": False,
        },
        "fairness": {
            "arc_agi_labels_loaded": False,
            "native_csv_ground_truth_present_but_not_scored": True,
            "labels_used_for_inference_or_selection": False,
            "solver_score_produced": False,
            "score_eligible_for_fair_main_board": False,
        },
        "limitations": [
            "This parses bundled native router CSVs and one pure prompt formatter only.",
            "It does not import the router environment, download Longformer/GPT-2, call NIM, train PPO, or solve an ARC task.",
            "The bundled arc_challenge rows are AI2 science multiple choice, not ARC-AGI grid tasks.",
            "No trained router checkpoint is bundled.",
        ],
    }


def cost_analysis(items: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, object]:
    total = sum(float(item.get("cost") or 0.0) for item in items)
    positive = sum(float(item.get("cost") or 0.0) > 0 for item in items)
    stored_average = float(summary["avg_cost_per_item"])
    declared_divisor = total / stored_average if stored_average else None
    return {
        "total_cost_matches_items": math.isclose(
            total, float(summary["total_cost"]), rel_tol=0.0, abs_tol=1e-12
        ),
        "stored_total_items": int(summary["total_items"]),
        "actual_items": len(items),
        "positive_cost_items": positive,
        "stored_avg_cost_per_item": stored_average,
        "true_avg_cost_per_item": total / len(items),
        "inferred_stored_divisor": declared_divisor,
        "stored_average_uses_all_items": math.isclose(
            stored_average, total / len(items), rel_tol=0.0, abs_tol=1e-12
        ),
    }


def routemoa_smoke(source: Path) -> dict[str, object]:
    base = source / "emoa_large" / "eval"
    filenames = (
        "benchmark_questions.json",
        "moa.json",
        "routemoa.json",
        "smoa.json",
        "summary_all.json",
    )
    documents = {
        name: json.loads((base / name).read_text(encoding="utf-8")) for name in filenames
    }
    benchmark = documents["benchmark_questions.json"]
    summary_all = documents["summary_all.json"]
    benchmark_lookup = {
        (category, dataset, str(item["item_id"])): item
        for category, datasets in benchmark["data"].items()
        for dataset, dataset_info in datasets.items()
        for item in dataset_info["items"]
    }
    methods: dict[str, object] = {}
    for method in ("moa", "routemoa", "smoa"):
        result = documents[f"{method}.json"]
        keys: list[tuple[str, str, str]] = []
        primary_scores: list[float] = []
        scored_dataset_count = 0
        summary_differences: list[dict[str, object]] = []
        benchmark_mismatches: list[dict[str, str]] = []
        cost_anomalies: list[dict[str, object]] = []
        missing_responses = 0
        for category, datasets in result["results"].items():
            for dataset, dataset_info in datasets.items():
                stored = dataset_info["summary"]
                items = dataset_info["items"]
                primary_scores.append(float(stored["accuracy"]))
                reference_summary = summary_all["per_dataset"][category][dataset][method]
                for field in (
                    "accuracy",
                    "latency_avg",
                    "cost_normalized",
                    "avg_cost_per_item",
                    "total_items",
                    "metric_name",
                ):
                    if stored[field] != reference_summary[field]:
                        summary_differences.append(
                            {"dataset": dataset, "field": field}
                        )
                correctness = [item.get("is_correct") for item in items]
                if all(value is not None for value in correctness):
                    scored_dataset_count += 1
                    recomputed = sum(float(value) for value in correctness) / len(correctness)
                    if not math.isclose(
                        recomputed, float(stored["accuracy"]), rel_tol=0.0, abs_tol=1e-12
                    ):
                        summary_differences.append(
                            {"dataset": dataset, "field": "mean(is_correct)"}
                        )
                cost = cost_analysis(items, stored)
                if not cost["total_cost_matches_items"]:
                    summary_differences.append(
                        {"dataset": dataset, "field": "total_cost"}
                    )
                if not cost["stored_average_uses_all_items"]:
                    cost_anomalies.append({"dataset": dataset, **cost})
                for item in items:
                    key = (category, dataset, str(item["item_id"]))
                    keys.append(key)
                    expected = benchmark_lookup.get(key)
                    if (
                        expected is None
                        or item.get("ground_truth") != expected.get("ground_truth")
                        or item.get("full_prompt") != expected.get("full_prompt")
                    ):
                        benchmark_mismatches.append(
                            {"category": category, "dataset": dataset, "item_id": key[2]}
                        )
                    missing_responses += item.get("model_response") is None
        recomputed_macro = sum(primary_scores) / len(primary_scores)
        declared_macro = float(
            summary_all["global_avg"][method]["accuracy_macro_avg"]
        )
        if summary_differences or benchmark_mismatches or missing_responses:
            raise RuntimeError(
                f"{method}: inconsistent precomputed result bundle"
            )
        methods[method] = {
            "categories": len(result["results"]),
            "datasets": len(primary_scores),
            "items": len(keys),
            "unique_items": len(set(keys)),
            "missing_model_responses": missing_responses,
            "benchmark_prompt_and_label_mismatches": len(benchmark_mismatches),
            "datasets_with_complete_stored_is_correct": scored_dataset_count,
            "stored_summary_differences": len(summary_differences),
            "stored_macro_recomputed": recomputed_macro,
            "stored_macro_declared_rounded": declared_macro,
            "stored_macro_rounding_delta": recomputed_macro - declared_macro,
            "cost_average_anomalies": cost_anomalies,
        }

    evaluator_text = (base / "evaluate.py").read_text(encoding="utf-8")
    bundled_result_has_data = "data" in documents["routemoa.json"]
    evaluator_expects_data = 'predictions["data"]' in evaluator_text
    return {
        "scope": "zero-dollar-precomputed-results-integrity-and-stored-score-aggregation-only",
        "files": {
            name: {
                "bytes": (base / name).stat().st_size,
                "sha256": sha256(base / name),
                "top_level_keys": list(documents[name]),
            }
            for name in filenames
        },
        "methods": methods,
        "provided_evaluator_contract": {
            "bundled_routemoa_has_data_key": bundled_result_has_data,
            "evaluator_expects_data_key": evaluator_expects_data,
            "directly_compatible": bundled_result_has_data or not evaluator_expects_data,
        },
        "fairness": {
            "arc_agi_labels_loaded": False,
            "native_precomputed_labels_loaded": True,
            "raw_model_inference_performed": False,
            "raw_responses_regraded": False,
            "stored_is_correct_used_for_scorer_only_audit": True,
            "solver_score_produced": False,
            "score_eligible_for_fair_main_board": False,
        },
        "limitations": [
            "This verifies upstream precomputed file integrity and internal stored-score aggregation only; labels are present, so it is scorer-only evidence.",
            "It does not run RouteMoA inference, a router checkpoint, any routed model, an API judge, or an ARC solver.",
            "The bundled result files use a results key while the provided evaluator requires data, so they are not directly accepted by that evaluator.",
            "Some stored avg_cost_per_item values divide by positive-cost rows rather than all 15 declared items; anomalies are recorded and not normalized away.",
            "Generative and judge metrics are not independently recomputed by this smoke.",
        ],
    }


def parse_dataset_registry(path: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("  ") and line.endswith(":") and not line.startswith("    "):
            current = line.strip()[:-1]
        elif current and line.strip().startswith("implemented:"):
            result[current] = line.split(":", 1)[1].strip().lower() == "true"
    return result


def maca_smoke(source: Path) -> dict[str, object]:
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
    }
    mismatches = {
        key: os.environ.get(key) for key, value in expected_environment.items() if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"offline deterministic environment missing: {mismatches}")

    sys.path.insert(0, str(source / "src"))
    try:
        import torch
        from MACA.graphspec.model import GraphSpecModel

        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = GraphSpecModel(
            n_agents=6,
            embed_dim=384,
            hidden_dim=256,
            st_model=None,
            device="cpu",
        )
        edge_probs, node_probs = model.predict_probs(
            ["zero dollar graph prior component", "second synthetic task"]
        )
        finite = bool(torch.isfinite(edge_probs).all() and torch.isfinite(node_probs).all())
        bounded = bool(
            (edge_probs >= 0).all()
            and (edge_probs <= 1).all()
            and (node_probs >= 0).all()
            and (node_probs <= 1).all()
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if list(edge_probs.shape) != [2, 6, 6] or list(node_probs.shape) != [2, 6]:
            raise RuntimeError("unexpected GraphSpec output shape")
        if not finite or not bounded:
            raise RuntimeError("GraphSpec output is non-finite or outside [0, 1]")
        component = {
            "torch_version": torch.__version__,
            "device": str(next(model.parameters()).device),
            "parameters": parameters,
            "estimated_fp32_parameter_bytes": parameters * 4,
            "edge_probability_shape": list(edge_probs.shape),
            "node_probability_shape": list(node_probs.shape),
            "finite": finite,
            "bounded_zero_one": bounded,
            "sentence_transformer_used": model.text_embedder.use_st,
            "random_untrained_weights": True,
        }
    finally:
        try:
            sys.path.remove(str(source / "src"))
        except ValueError:
            pass

    rollout_text = (source / "src" / "MACA" / "grpo" / "rollout" / "mas_rollout.py").read_text(
        encoding="utf-8"
    )
    verl_text = (source / "src" / "MACA" / "grpo" / "backend" / "verl" / "trainer.py").read_text(
        encoding="utf-8"
    )
    registry = parse_dataset_registry(source / "configs" / "datasets" / "registry.yaml")
    return {
        "scope": "zero-dollar-graphspec-random-weight-cpu-component-only",
        "component": component,
        "dataset_registry": {
            "implemented": sorted(name for name, value in registry.items() if value),
            "not_implemented": sorted(name for name, value in registry.items() if not value),
        },
        "training_wiring": {
            "native_backend_uses_mock_rollout_adapter": "MockRolloutAdapter" in rollout_text,
            "mock_reward_matches_graph_prior_argmax": "argmax(node_probs)" in rollout_text,
            "verl_backend_declares_placeholder": "Placeholder integration point" in verl_text,
        },
        "fairness": {
            "arc_agi_labels_loaded": False,
            "native_labels_loaded": False,
            "checkpoint_loaded": False,
            "model_or_api_inference_performed": False,
            "solver_score_produced": False,
            "score_eligible_for_fair_main_board": False,
        },
        "limitations": [
            "This is a deterministic CPU forward through GraphSpec with random fallback embeddings and untrained weights only.",
            "It does not execute the mock GRPO adapter, VERL, provider/local LLMs, generated code, or any benchmark task.",
            "The repository-wide syntax audit failed separately on two upstream Python files.",
            "The native GRPO path is a deterministic stand-in and the VERL path is a placeholder, so neither is paper training evidence.",
            "No paper checkpoint is bundled and most paper benchmark adapters are marked unimplemented.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=sorted(SPECS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": args.method_id,
        "run_id": output_directory.name,
        "runner": "scripts.smoke_routing_methods",
        "status": "failed",
        "started_at_utc": utc_now(),
        "paper_year": SPECS[args.method_id]["paper_year"],
        "native_scope": SPECS[args.method_id]["native_scope"],
    }
    try:
        ensure_no_provider_credentials()
        record["source"] = source_record(args.method_id, source)
        with network_guard():
            if args.method_id == "graphplanner":
                result = graphplanner_smoke(source)
            elif args.method_id == "routemoa":
                result = routemoa_smoke(source)
            else:
                result = maca_smoke(source)
        record.update(result)
        record["status"] = "passed"
        record["security"] = {
            "network_guard": "socket connect/create_connection fail closed",
            "provider_credentials_present": False,
            "api_requests": 0,
            "generated_code_executed": False,
            "untrusted_checkpoint_or_pickle_loaded": False,
        }
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "python_version": platform.python_version(),
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
            "ru_maxrss_before": rss_start,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "KiB on Linux",
            "gpu_used": False,
            "network_used": False,
            "external_weights_loaded": False,
            "external_datasets_downloaded": False,
            "estimated_cost_usd": 0.0,
        }
        record["claim_boundary"] = (
            "This run is limited to the declared zero-dollar schema, precomputed-scorer, "
            "or random-weight component scope. It is not an ARC-AGI solver run, native "
            "benchmark reproduction, paper-parity result, or fair-main-board score."
        )
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "run_json": str(output_directory / "run.json"),
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
