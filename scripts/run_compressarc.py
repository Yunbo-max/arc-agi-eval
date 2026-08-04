#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import operator
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "CompressARC"
UPSTREAM_REVISION = "83a22218024d46273eb32b769a906340202ffb4d"
UPSTREAM_TREE_SHA256 = "db41685bb9161aa2aa9727dea2a48a601285c04e821ca1b8e9e7d76251d202e7"
UPSTREAM_TORCH_VERSION = "2.5.1"
TASK_ID_PATTERN = re.compile(r"[0-9a-f]{8}\Z")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi_eval.scoring import score_predictions
from arc_agi_eval.validation import load_json, load_task, validate_grid


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _task_id(value: str) -> str:
    if TASK_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected an 8-character lowercase hexadecimal task ID")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one pinned CompressARC task without upstream artifacts."
    )
    parser.add_argument("--split", required=True, choices=("training", "evaluation", "test"))
    parser.add_argument("--task-id", required=True, type=_task_id)
    parser.add_argument("--steps", required=True, type=_positive_int)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log-interval", type=_positive_int)
    args = parser.parse_args(argv)

    args.task_dir = args.task_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.task_dir.is_dir():
        parser.error(f"task directory does not exist: {args.task_dir}")
    if args.task_dir.name != args.split:
        parser.error(
            f"split {args.split!r} does not match task directory name {args.task_dir.name!r}"
        )
    task_path = args.task_dir / f"{args.task_id}.json"
    if not task_path.is_file():
        parser.error(f"canonical task file does not exist: {task_path}")
    if args.run_dir.exists() and not args.run_dir.is_dir():
        parser.error(f"run directory is not a directory: {args.run_dir}")
    if args.run_dir.is_dir() and any(args.run_dir.iterdir()):
        parser.error(f"run directory must be new or empty: {args.run_dir}")
    return args


def challenge_without_test_outputs(task: dict[str, Any]) -> dict[str, Any]:
    def copy_grid(grid: Sequence[Sequence[int]]) -> list[list[int]]:
        return [list(row) for row in grid]

    return {
        "train": [
            {"input": copy_grid(pair["input"]), "output": copy_grid(pair["output"])}
            for pair in task["train"]
        ],
        "test": [{"input": copy_grid(pair["input"])} for pair in task["test"]],
    }


def _json_grid(grid: Any, where: str) -> list[list[int]]:
    try:
        converted = [
            [operator.index(cell) for cell in row]
            for row in grid
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: upstream solution is not an integer grid") from exc
    return validate_grid(converted, where)


def solutions_to_predictions(
    task_id: str,
    first: Sequence[Any] | None,
    second: Sequence[Any] | None,
) -> dict[str, list[dict[str, list[list[int]]]]]:
    if first is None or second is None:
        raise ValueError("upstream Logger did not produce two solutions")
    if len(first) != len(second):
        raise ValueError("upstream Logger solutions have different output counts")
    if not first:
        raise ValueError("upstream Logger produced no test outputs")

    outputs = []
    for index, (attempt_1, attempt_2) in enumerate(zip(first, second)):
        outputs.append(
            {
                "attempt_1": _json_grid(attempt_1, f"{task_id}[{index}].attempt_1"),
                "attempt_2": _json_grid(attempt_2, f"{task_id}[{index}].attempt_2"),
            }
        )
    return {task_id: outputs}


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    torch_version = _distribution_version("torch")
    torch_base_version = torch_version.split("+", 1)[0] if torch_version else None
    deviations = []
    if torch_base_version != UPSTREAM_TORCH_VERSION:
        deviations.append(
            f"torch {torch_version or 'not installed'} instead of upstream torch=={UPSTREAM_TORCH_VERSION}"
        )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: _distribution_version(name)
            for name in ("torch", "numpy", "matplotlib", "tqdm")
        },
        "upstream_torch_version": UPSTREAM_TORCH_VERSION,
        "deviations": deviations,
        "cuda_available": None,
        "torch_cuda_runtime": None,
        "gpu": None,
    }


def _source_tree_sha256() -> str:
    files: set[Path] = set()
    for pattern in ("*.py", "*.md", "LICENSE", "requirements.txt", "dataset/*.json"):
        files.update(path for path in UPSTREAM_ROOT.glob(pattern) if path.is_file())
    manifest = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(UPSTREAM_ROOT.parent).as_posix()}\n"
        for path in sorted(files, key=lambda item: item.relative_to(UPSTREAM_ROOT.parent).as_posix())
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def _import_upstream() -> tuple[Any, Any, Any, Any]:
    observed_hash = _source_tree_sha256()
    if observed_hash != UPSTREAM_TREE_SHA256:
        raise RuntimeError(
            "pinned CompressARC source verification failed: "
            f"expected {UPSTREAM_TREE_SHA256}, got {observed_hash}"
        )

    module_names = {path.stem for path in UPSTREAM_ROOT.glob("*.py")}
    for name in sorted(module_names):
        existing = sys.modules.get(name)
        existing_file = getattr(existing, "__file__", None) if existing else None
        if existing_file and Path(existing_file).resolve().parent != UPSTREAM_ROOT:
            raise RuntimeError(f"refusing to shadow existing module {name!r} from {existing_file}")

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.dont_write_bytecode = True
    source_entry = str(UPSTREAM_ROOT)
    sys.path.insert(0, source_entry)
    try:
        preprocessing = importlib.import_module("preprocessing")
        arc_compressor = importlib.import_module("arc_compressor")
        train = importlib.import_module("train")
        solution_selection = importlib.import_module("solution_selection")
    finally:
        if sys.path[0] == source_entry:
            sys.path.pop(0)
        else:
            sys.path.remove(source_entry)

    for module in (preprocessing, arc_compressor, train, solution_selection):
        if Path(module.__file__).resolve().parent != UPSTREAM_ROOT:
            raise RuntimeError(f"imported {module.__name__!r} outside the pinned upstream tree")
    return preprocessing, arc_compressor, train, solution_selection


def _empty_metrics() -> dict[str, Any]:
    return {
        "top_k": 2,
        "tasks_total": None,
        "tasks_predicted": None,
        "tasks_exact": None,
        "task_exact_accuracy": None,
        "outputs_total": None,
        "outputs_exact": None,
        "output_exact_accuracy": None,
        "cells_total": None,
        "cells_correct": None,
        "cell_accuracy": None,
    }


def _initial_run(args: argparse.Namespace, command: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner": "scripts.run_compressarc",
        "runner_version": 1,
        "run_id": args.run_dir.name,
        "status": "running",
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "source": {
            "repository": "https://github.com/iliao2345/CompressARC",
            "revision": UPSTREAM_REVISION,
            "local_path": str(UPSTREAM_ROOT.relative_to(PROJECT_ROOT)),
            "expected_retained_tree_sha256": UPSTREAM_TREE_SHA256,
            "observed_retained_tree_sha256": None,
            "verified": False,
        },
        "configuration": {
            "split": args.split,
            "task_id": args.task_id,
            "steps_requested": args.steps,
            "log_interval": args.log_interval,
            "task_directory": str(args.task_dir),
            "top_k": 2,
            "seed": 0,
            "optimizer": "torch.optim.Adam(lr=0.01, betas=(0.5, 0.9))",
            "test_outputs_available_to_optimizer": False,
        },
        "command": list(command),
        "environment": _environment(),
        "artifacts": {
            "predictions": "predictions.json",
            "prediction_bytes": None,
            "prediction_sha256": None,
            "checkpoint": None,
            "plots": [],
        },
        "counts": {
            "tasks_requested": 1,
            "tasks_predicted": None,
            "test_outputs_predicted": None,
            "attempts_generated": None,
            "steps_completed": 0,
        },
        "timing": {
            "prediction_wall_time_seconds": None,
            "scoring_wall_time_seconds": None,
            "wall_time_seconds": None,
        },
        "final_loss": None,
        "peak_process_vram_bytes": None,
        "metrics": _empty_metrics(),
        "error_traceback": None,
    }


def run(args: argparse.Namespace, command: Sequence[str] | None = None) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.run_dir / "run.json"
    prediction_path = args.run_dir / "predictions.json"
    if run_path.exists() or prediction_path.exists():
        raise ValueError(f"run directory contains runner artifacts: {args.run_dir}")

    started = time.perf_counter()
    command = command if command is not None else sys.argv
    record = _initial_run(args, command)
    _atomic_write_json(run_path, record)

    torch = None
    logger = None
    prediction_started = time.perf_counter()
    try:
        task_path = args.task_dir / f"{args.task_id}.json"
        canonical_task = load_task(task_path, require_test_outputs=False)
        challenge = challenge_without_test_outputs(canonical_task)
        del canonical_task

        preprocessing, arc_compressor, train, solution_selection = _import_upstream()
        record["source"]["observed_retained_tree_sha256"] = _source_tree_sha256()
        record["source"]["verified"] = True

        torch = importlib.import_module("torch")
        record["environment"].update(
            {
                "packages": {
                    **record["environment"]["packages"],
                    "torch": str(torch.__version__),
                },
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_runtime": torch.version.cuda,
            }
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CompressARC requires CUDA, but torch.cuda.is_available() is false")
        device = torch.cuda.current_device()
        record["environment"]["gpu"] = {
            "device_index": device,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        }

        torch.cuda.reset_peak_memory_stats(device)
        task = preprocessing.Task(args.task_id, challenge, None)
        model = arc_compressor.ARCCompressor(task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        logger = solution_selection.Logger(task)

        for train_step in range(args.steps):
            train.take_step(task, model, optimizer, train_step, logger)
            record["counts"]["steps_completed"] = train_step + 1
            if args.log_interval and (
                (train_step + 1) % args.log_interval == 0 or train_step + 1 == args.steps
            ):
                print(
                    f"step {train_step + 1}/{args.steps} loss={logger.loss_curve[-1]:.6f}",
                    flush=True,
                )

        torch.cuda.synchronize(device)
        predictions = solutions_to_predictions(
            args.task_id,
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
        )
        _atomic_write_json(prediction_path, predictions)
        prediction_finished = time.perf_counter()

        prediction_bytes = prediction_path.read_bytes()
        output_count = len(predictions[args.task_id])
        record["artifacts"].update(
            {
                "prediction_bytes": len(prediction_bytes),
                "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            }
        )
        record["counts"].update(
            {
                "tasks_predicted": 1,
                "test_outputs_predicted": output_count,
                "attempts_generated": output_count * 2,
            }
        )
        record["final_loss"] = logger.loss_curve[-1]
        record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
        record["timing"]["prediction_wall_time_seconds"] = round(
            prediction_finished - prediction_started, 6
        )

        scoring_started = time.perf_counter()
        scoring_task = load_task(task_path, require_test_outputs=False)
        labels_available = all("output" in pair for pair in scoring_task["test"])
        if labels_available:
            answers = {args.task_id: [pair["output"] for pair in scoring_task["test"]]}
            persisted_predictions = load_json(prediction_path)
            record["metrics"] = score_predictions(
                persisted_predictions,
                answers,
                top_k=2,
                source=str(prediction_path),
            ).as_dict()
        record["timing"]["scoring_wall_time_seconds"] = round(
            time.perf_counter() - scoring_started, 6
        )
        record["status"] = "passed"
        record["ended_at_utc"] = _utc_now()
        record["timing"]["wall_time_seconds"] = round(time.perf_counter() - started, 6)
        _atomic_write_json(run_path, record)
        print(f"wrote {prediction_path} and {run_path}", flush=True)
        return 0
    except BaseException:
        record["status"] = "failed"
        record["ended_at_utc"] = _utc_now()
        record["error_traceback"] = traceback.format_exc()
        record["timing"]["wall_time_seconds"] = round(time.perf_counter() - started, 6)
        if logger is not None and logger.loss_curve:
            record["final_loss"] = logger.loss_curve[-1]
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        _atomic_write_json(run_path, record)
        print(record["error_traceback"], file=sys.stderr, end="")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = [str(Path(__file__).relative_to(PROJECT_ROOT)), *(argv if argv is not None else sys.argv[1:])]
    return run(args, command)


if __name__ == "__main__":
    raise SystemExit(main())
