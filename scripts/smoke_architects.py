#!/usr/bin/env python3
"""Load the locked ARChitects 4-bit checkpoint and run one token forward."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import traceback


EXPECTED_MODEL_REVISION = "6de719999a213e717fe339fb5a29177ddc4310d9"
GIB = 1024**3


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


def gpu_state() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    state: dict[str, object] = {
        "command": command,
        "exit_code": result.returncode,
        "stderr": result.stderr.strip(),
    }
    if result.returncode == 0:
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) == 5:
            state.update(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "total_memory_bytes": int(fields[2]) * 1024**2,
                    "free_memory_bytes": int(fields[3]) * 1024**2,
                    "driver_version": fields[4],
                }
            )
    return state


def base_record(
    *, output_directory: Path, snapshot: Path, started_at: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method_id": "architects-2024",
        "run_id": output_directory.name,
        "runner": "scripts.smoke_architects",
        "scope": "published-4bit-checkpoint-one-token-forward",
        "started_at_utc": started_at,
        "model": {
            "repo_id": "da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit",
            "revision": EXPECTED_MODEL_REVISION,
            "snapshot": str(snapshot),
        },
        "environment": {"python": platform.python_version()},
        "fairness": {
            "arc_agi_1_public_evaluation": "training-contaminated; ineligible for label-clean main board",
            "arc_agi_2": "eligible only as a new transfer experiment after challenge-only adaptation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=10.0)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    output_directory = args.output_directory.resolve()
    if snapshot.name != EXPECTED_MODEL_REVISION:
        parser.error(f"unexpected model revision: {snapshot.name}")
    if not (snapshot / "model.safetensors").is_file():
        parser.error(f"missing model.safetensors: {snapshot}")
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(f"output directory is not empty: {output_directory}")
    if args.minimum_free_vram_gib <= 0:
        parser.error("minimum free VRAM must be positive")

    started_at = utc_now()
    record = base_record(
        output_directory=output_directory, snapshot=snapshot, started_at=started_at
    )
    before = gpu_state()
    required_bytes = int(args.minimum_free_vram_gib * GIB)
    record["preflight"] = {
        "gpu": before,
        "minimum_free_vram_bytes": required_bytes,
    }
    free_bytes = before.get("free_memory_bytes")
    if not isinstance(free_bytes, int) or free_bytes < required_bytes:
        record.update(
            {
                "status": "blocked",
                "ended_at_utc": utc_now(),
                "blocker": (
                    "GPU free-memory gate not met; no model import or allocation attempted"
                ),
            }
        )
        atomic_json(output_directory / "run.json", record)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 2

    wall_start = time.perf_counter()
    try:
        import accelerate
        import bitsandbytes
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        record["environment"].update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "bitsandbytes": bitsandbytes.__version__,
            }
        )
        torch.cuda.reset_peak_memory_stats(0)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.eval()
        token_id = tokenizer.bos_token_id
        if token_id is None:
            token_id = 1
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=False)
        shape = list(outputs.logits.shape)
        passed = shape == [1, 1, int(model.config.vocab_size)]
        record.update(
            {
                "status": "passed" if passed else "failed",
                "ended_at_utc": utc_now(),
                "result": {
                    "input_shape": list(input_ids.shape),
                    "logits_shape": shape,
                    "vocab_size": int(model.config.vocab_size),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
                "resources": {
                    "wall_time_seconds": time.perf_counter() - wall_start,
                    "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "gpu_after": gpu_state(),
                },
                "claim_boundary": (
                    "A one-token architecture forward passed; no ARC prompt, TTT, "
                    "prediction, benchmark, or paper-reproduction claim is made"
                ),
            }
        )
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "ended_at_utc": utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "resources": {"wall_time_seconds": time.perf_counter() - wall_start},
            }
        )
    atomic_json(output_directory / "run.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
