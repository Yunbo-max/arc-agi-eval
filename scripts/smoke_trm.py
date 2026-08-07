#!/usr/bin/env python3
"""Run a CPU-only, no-checkpoint TinyRecursiveModels architecture smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
LOCKED_REVISION = "c01103738605ba39d1430519b1ee0c62f4c707f8"


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


def git_output(source: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            os.environ.get(
                "ARC_TRM_SOURCE",
                "/usr/paper-assets/arc/sources/tiny-recursive-models",
            )
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT
        / "reports"
        / "tiny-recursive-models"
        / "20260806-cpu-architecture-forward-smoke",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")

    started_at = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record: dict[str, object] = {
        "schema_version": 1,
        "method_id": "tiny-recursive-models",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_trm",
        "status": "failed",
        "scope": "architecture_only",
        "started_at_utc": started_at,
        "source": str(source),
        "locked_revision": LOCKED_REVISION,
        "checkpoint_loaded": False,
        "benchmark_executed": False,
        "training_executed": False,
        "arc_data_loaded": False,
        "labels_accessible": False,
        "gpu_requested": False,
    }

    try:
        if not source.is_dir():
            raise FileNotFoundError(f"source checkout not found: {source}")
        revision = git_output(source, "rev-parse", "HEAD")
        dirty = bool(git_output(source, "status", "--porcelain"))
        if revision != LOCKED_REVISION:
            raise RuntimeError(
                f"source revision {revision} does not match lock {LOCKED_REVISION}"
            )
        if dirty:
            raise RuntimeError("source checkout is dirty")

        # This must be set before importing torch. It makes accidental CUDA use
        # impossible for this architecture-only probe.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))

        import einops
        import pydantic
        import torch
        from models.recursive_reasoning.trm import (
            TinyRecursiveReasoningModel_ACTV1,
        )

        torch.manual_seed(0)
        config = {
            "batch_size": 1,
            "seq_len": 16,
            "puzzle_emb_ndim": 512,
            "num_puzzle_identifiers": 1,
            "vocab_size": 12,
            "H_cycles": 3,
            "L_cycles": 4,
            "H_layers": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "expansion": 4,
            "num_heads": 8,
            "pos_encodings": "rope",
            "halt_max_steps": 16,
            "halt_exploration_prob": 0.0,
            "forward_dtype": "bfloat16",
            "mlp_t": False,
            "puzzle_emb_len": 16,
            "no_ACT_continue": True,
        }
        model = TinyRecursiveReasoningModel_ACTV1(config).cpu().eval()
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        inputs = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6]],
            dtype=torch.int64,
        )
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int64),
        }
        carry = model.initial_carry(batch)
        with torch.no_grad():
            new_carry, outputs = model(carry, batch)
        logits = outputs["logits"]
        logits_bytes = logits.to(torch.float32).contiguous().numpy().tobytes()

        expected_shape = [1, 16, 12]
        if list(logits.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected logits shape {list(logits.shape)} != {expected_shape}"
            )
        if trainable_parameters != 6_829_058:
            raise RuntimeError(
                "unexpected trainable parameter count "
                f"{trainable_parameters} != 6829058"
            )

        record.update(
            {
                "status": "passed",
                "source_revision": revision,
                "source_dirty": dirty,
                "source_license": "MIT",
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "pydantic_version": pydantic.__version__,
                "einops_version": einops.__version__,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "torch_cuda_available": torch.cuda.is_available(),
                "device": str(logits.device),
                "model_config": config,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "input_shape": list(inputs.shape),
                "logits_shape": list(logits.shape),
                "halt_shape": list(outputs["q_halt_logits"].shape),
                "carry_steps": new_carry.steps.tolist(),
                "logits_float32_sha256": hashlib.sha256(logits_bytes).hexdigest(),
                "limitations": [
                    "Synthetic 16-token input; no ARC task was loaded.",
                    "No checkpoint, training step, accuracy measurement, or benchmark.",
                    "The paper ARC configurations train separate ARC-AGI-1 and ARC-AGI-2 models for about three days on four H100 GPUs.",
                ],
            }
        )
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["ended_at_utc"] = utc_now()
        record["resources"] = {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
            "ru_maxrss_before": rss_start,
            "ru_maxrss_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ru_maxrss_unit": "KiB on Linux",
            "memory_scope": "current process peak; children excluded",
        }
        atomic_json(output_directory / "run.json", record)

    print(
        json.dumps(
            {
                "status": record["status"],
                "run_json": str(output_directory / "run.json"),
                "resources": record["resources"],
                "error": record.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
