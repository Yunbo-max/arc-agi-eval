#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "GridCoder2024"
ARC_GYM = ROOT / "external" / "ARC_gym"
RUN_DIR = ROOT / "reports" / "gridcoder2024" / "20260806-architecture-forward-smoke"


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


def tree_hash(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*.py")):
        lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
        )
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise SystemExit(f"run directory is not empty: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {
        "schema_version": 1,
        "runner": "scripts.smoke_gridcoder",
        "run_id": RUN_DIR.name,
        "status": "running",
        "started_at_utc": now(),
        "ended_at_utc": None,
        "source": {
            "gridcoder_revision": "bf6136e5f57029dcbbb85242b8ffd8a1a241bb5f",
            "gridcoder_tree_sha256": tree_hash(SOURCE),
            "arc_gym_revision": revision(ARC_GYM),
            "arc_gym_tree_sha256": tree_hash(ARC_GYM / "ARC_gym"),
            "arc_gym_license_file_present": bool(list(ARC_GYM.glob("LICENSE*"))),
        },
        "configuration": {
            "input_shape": [1, 13, 30, 30],
            "max_sequence_length": 1,
            "synthetic_untrained_weights": True,
            "checkpoint_present": (SOURCE / "model_full.pth").is_file(),
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
            raise RuntimeError("GridCoder architecture smoke requires CUDA")
        sys.path.insert(0, str(SOURCE))
        from Hodel_primitives_atomicV3 import semantics
        from model.LVM import LVM

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(0)
        model = LVM(13, len(semantics) + 4, emb_dim=512, max_seq_length=1).cuda().eval()
        record["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        x1 = torch.zeros((1, 13, 30, 30), device="cuda")
        x2 = torch.zeros_like(x1)
        with torch.no_grad():
            output = model(x1, x2)
        torch.cuda.synchronize()
        record["output_shape"] = list(output.shape)
        record["peak_process_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))
        record["status"] = "passed"
    except BaseException:
        record["status"] = "failed"
        record["error_traceback"] = traceback.format_exc()
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
