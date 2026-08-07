#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "ARC-AGI-Challenge-2024"
RUN_DIR = ROOT / "reports" / "2d-ngpt" / "20260806-large-architecture-forward-smoke"
REVISION = "e5420b10b9470b3b5c6548572768d2d4c15130f6"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tree_hash() -> str:
    lines = []
    for path in sorted(SOURCE.rglob("*.py")):
        lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(SOURCE).as_posix()}\n"
        )
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise SystemExit(f"run directory is not empty: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {
        "schema_version": 1,
        "runner": "scripts.smoke_2d_ngpt",
        "run_id": RUN_DIR.name,
        "status": "running",
        "started_at_utc": now(),
        "ended_at_utc": None,
        "source": {"revision": REVISION, "tree_sha256": tree_hash()},
        "configuration": {
            "version": 64,
            "size": "large",
            "input_shape": [1, 30, 30],
            "synthetic_untrained_weights": True,
            "checkpoint_present": any(SOURCE.rglob("*.pt")),
            "adapter": "inject config module global required by ARCModel.forward",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "parameters": None,
        "output_shape": None,
        "peak_process_vram_bytes": None,
        "timing": {"wall_time_seconds": None},
        "error_traceback": None,
    }
    atomic(RUN_DIR / "run.json", record)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("2D nGPT architecture smoke requires CUDA")
        config_module = load("ngpt_cfg_064", SOURCE / "cfg" / "cfg_064.py")
        model_module = load("ngpt_code_064", SOURCE / "code" / "064.py")
        cfg = config_module.cfg
        cfg.device = torch.device("cuda")
        model_module.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats(0)
        model = model_module.ARCModel(cfg).cuda().eval()
        record["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        batch = {
            "input": torch.zeros((1, 30, 30), dtype=torch.long, device="cuda"),
            "output": torch.zeros((1, 30, 30), dtype=torch.long, device="cuda"),
            "task": torch.zeros((1, 1, 1), dtype=torch.long, device="cuda"),
        }
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()
        record["output_shape"] = list(output.shape)
        record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))
        record["status"] = "passed"
    except BaseException:
        record["status"] = "failed"
        record["error_traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))
    record["ended_at_utc"] = now()
    record["timing"]["wall_time_seconds"] = round(time.perf_counter() - started, 6)
    atomic(RUN_DIR / "run.json", record)
    if record["status"] == "failed":
        print(record["error_traceback"], file=sys.stderr, end="")
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
