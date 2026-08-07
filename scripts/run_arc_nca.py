#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "ARC_NCA"
UPSTREAM_REVISION = "25d522bc766f9ddaebbf7dad63f58790fe7aa884"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi_eval.firewall import challenge_only
from arc_agi_eval.scoring import score_predictions
from arc_agi_eval.validation import load_json, load_task


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one reduced, pinned ARC_NCA task")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=positive_int)
    parser.add_argument("--rollout-steps", type=positive_int, default=32)
    parser.add_argument("--pool-size", type=positive_int, default=32)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.task_dir = args.task_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    task_path = args.task_dir / f"{args.task_id}.json"
    if not task_path.is_file():
        parser.error(f"task does not exist: {task_path}")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error(f"run directory must be new or empty: {args.run_dir}")
    return args


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def source_hash() -> str:
    entries = []
    for path in sorted(UPSTREAM_ROOT.glob("*.py")):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    return hashlib.sha256("".join(entries).encode()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def same_size(task: dict[str, Any]) -> bool:
    return all(
        len(pair["input"]) == len(pair["output"])
        and len(pair["input"][0]) == len(pair["output"][0])
        for pair in task["train"]
    )


def load_upstream() -> tuple[Any, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(UPSTREAM_ROOT))
    try:
        nca_module = importlib.import_module("NCA")
        arc_utils = importlib.import_module("arc_agi_utils")
    finally:
        sys.path.pop(0)
    return nca_module, arc_utils


def get_batch(
    pool: torch.Tensor, pristine: torch.Tensor, batch_size: int, noise_level: float
) -> tuple[torch.Tensor, np.ndarray]:
    """Equivalent training-only subset of upstream utils.get_batch."""
    indices = np.random.randint(0, pool.shape[0], batch_size)
    batch = pool[indices].clone()
    mask = torch.rand_like(batch) < noise_level
    midpoint = pool.shape[1] // 2
    batch[:, midpoint:] = (
        batch[:, midpoint:] * (~mask[:, midpoint:]).float()
        + torch.randn_like(batch[:, midpoint:]) * mask[:, midpoint:].float()
    )
    batch[0:1] = pristine
    return batch, indices


def decode_grid(state: torch.Tensor, palette: torch.Tensor) -> list[list[int]]:
    pixels = state[:4].permute(1, 2, 0)
    distances = ((pixels[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(dim=-1)
    return distances.argmin(dim=-1).cpu().tolist()


def run(args: argparse.Namespace, command: Sequence[str]) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.run_dir / "run.json"
    prediction_path = args.run_dir / "predictions.json"
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema_version": 1,
        "runner": "scripts.run_arc_nca",
        "run_id": args.run_dir.name,
        "status": "running",
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "command": list(command),
        "source": {
            "repository": "https://github.com/etimush/ARC_NCA",
            "revision": UPSTREAM_REVISION,
            "tree_sha256": source_hash(),
        },
        "configuration": {
            "task_id": args.task_id,
            "steps": args.steps,
            "rollout_steps": args.rollout_steps,
            "pool_size": args.pool_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "color_count": 10,
            "model": "CA",
            "channels": 50,
            "hidden_channels": 264,
            "test_outputs_available_to_optimizer": False,
            "protocol_deviations": [
                "scripted wrapper for notebook-only upstream",
                "inlined get_batch/pool update to avoid upstream's visualization-only OpenCV import",
                "fixed ARC color_count=10 instead of deriving it from evaluation outputs",
                "reduced configurable pool, optimization, and rollout budgets",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "matplotlib": package_version("matplotlib"),
        },
        "counts": {"steps_completed": 0, "attempts_generated": 0},
        "timing": {"wall_time_seconds": None},
        "peak_process_vram_bytes": None,
        "final_training_loss": None,
        "metrics": None,
        "artifacts": {"predictions": None, "prediction_sha256": None},
        "error_traceback": None,
    }
    atomic_json(run_path, record)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("ARC_NCA upstream requires CUDA")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(0)

        task_path = args.task_dir / f"{args.task_id}.json"
        labeled = load_task(task_path)
        if not same_size(labeled):
            raise ValueError("ignore-size-change protocol requires equal train input/output shapes")
        task = challenge_only(labeled)
        del labeled

        nca_module, arc_utils = load_upstream()
        channels = 50
        gene_size = 25
        color_count = 10
        train_inputs = [torch.tensor(pair["input"], device="cuda:0") for pair in task["train"]]
        train_outputs = [torch.tensor(pair["output"], device="cuda:0") for pair in task["train"]]
        test_inputs = [torch.tensor(pair["input"], device="cuda:0") for pair in task["test"]]
        encode = lambda grid: arc_utils.arc_to_nca_space(
            color_count, grid, channels, gene_size, device="cuda:0", mode="rgb",
            gene_location=list(range(gene_size)), is_invis=1,
        )
        encoded_inputs = [encode(grid) for grid in train_inputs]
        encoded_outputs = [encode(grid) for grid in train_outputs]
        pools = [grid.tile(args.pool_size, 1, 1, 1) for grid in encoded_inputs]

        model = nca_module.CA(channels, 264).to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(args.steps):
            model.train()
            demo = step % len(pools)
            with torch.no_grad():
                x, indices = get_batch(
                    pools[demo], encoded_inputs[demo].clone(), args.batch_size, noise_level=0.2
                )
                target = encoded_outputs[demo].tile(args.batch_size, 1, 1, 1)
            for _ in range(args.rollout_steps):
                x = model(x, 0.5)
            loss = (target[:, :4] - x[:, :4]).pow(2).mean()
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad /= parameter.grad.norm() + 1e-8
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                pools[demo][indices] = x.detach()
            record["counts"]["steps_completed"] = step + 1
            record["final_training_loss"] = float(loss.detach())

        palette = torch.stack(
            [encode(torch.tensor([[color]], device="cuda:0"))[:4, 0, 0] for color in range(10)]
        )
        outputs = []
        model.eval()
        for test_index, grid in enumerate(test_inputs):
            attempts = {}
            for attempt in (1, 2):
                torch.manual_seed(args.seed + 1000 * attempt + test_index)
                state = encode(grid)[None]
                with torch.no_grad():
                    for _ in range(args.rollout_steps):
                        state = model(state, 0.5)
                attempts[f"attempt_{attempt}"] = decode_grid(state[0], palette)
            outputs.append(attempts)
        predictions = {args.task_id: outputs}
        atomic_json(prediction_path, predictions)
        prediction_bytes = prediction_path.read_bytes()
        record["artifacts"] = {
            "predictions": "predictions.json",
            "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        }
        record["counts"]["attempts_generated"] = 2 * len(outputs)

        # Labels are loaded only after prediction bytes have been persisted.
        scoring_task = load_task(task_path)
        answers = {args.task_id: [pair["output"] for pair in scoring_task["test"]]}
        record["metrics"] = score_predictions(load_json(prediction_path), answers).as_dict()
        record["status"] = "passed"
        record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))
    except BaseException:
        record["status"] = "failed"
        record["error_traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))
    record["ended_at_utc"] = utc_now()
    record["timing"]["wall_time_seconds"] = round(time.perf_counter() - started, 6)
    atomic_json(run_path, record)
    if record["status"] == "failed":
        print(record["error_traceback"], file=sys.stderr, end="")
        return 1
    print(f"wrote {prediction_path} and {run_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = ["scripts/run_arc_nca.py", *(argv if argv is not None else sys.argv[1:])]
    return run(args, command)


if __name__ == "__main__":
    raise SystemExit(main())
