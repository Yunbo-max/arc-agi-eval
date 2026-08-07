#!/usr/bin/env python3
"""Run the reduced ARC_NCA adapter on one label-free task using CPU only.

This process is deliberately inference-only.  It accepts no answer or scoring
path, rejects every test ``output`` field, does not import the repository
scorer, and writes only predictions plus deterministic execution metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

import numpy as np
import torch


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InferenceInputError(ValueError):
    """Raised when inference-visible input violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InferenceInputError(f"{path}: invalid JSON: {error}") from error


def atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_grid(value: Any, where: str) -> list[list[int]]:
    if not isinstance(value, list) or not value or len(value) > 30:
        raise InferenceInputError(f"{where}: expected 1..30 rows")
    width: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row or len(row) > 30:
            raise InferenceInputError(f"{where}[{row_index}]: expected 1..30 cells")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise InferenceInputError(f"{where}: ragged grid")
        for column_index, cell in enumerate(row):
            if type(cell) is not int or not 0 <= cell <= 9:
                raise InferenceInputError(
                    f"{where}[{row_index}][{column_index}]: invalid ARC color"
                )
    return value


def load_label_free_task(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise InferenceInputError("task root must be an object")
    if not {"train", "test"}.issubset(value) or not set(value).issubset(
        {"train", "test", "name"}
    ):
        raise InferenceInputError("task root has missing or unknown fields")
    if not isinstance(value["train"], list) or not value["train"]:
        raise InferenceInputError("task must contain training demonstrations")
    if not isinstance(value["test"], list) or not value["test"]:
        raise InferenceInputError("task must contain test inputs")
    for index, pair in enumerate(value["train"]):
        if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
            raise InferenceInputError(f"train[{index}]: expected input and output")
        _validate_grid(pair["input"], f"train[{index}].input")
        _validate_grid(pair["output"], f"train[{index}].output")
        if (
            len(pair["input"]) != len(pair["output"])
            or len(pair["input"][0]) != len(pair["output"][0])
        ):
            raise InferenceInputError(
                f"train[{index}]: CPU smoke requires equal input/output shape"
            )
    for index, pair in enumerate(value["test"]):
        if not isinstance(pair, dict):
            raise InferenceInputError(f"test[{index}]: expected an object")
        if "output" in pair:
            raise InferenceInputError(
                f"test[{index}]: hidden output supplied to inference"
            )
        if set(pair) != {"input"}:
            raise InferenceInputError(f"test[{index}]: expected only input")
        _validate_grid(pair["input"], f"test[{index}].input")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise InferenceInputError(f"{field}: expected a positive integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceInputError(f"{field}: expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise InferenceInputError(f"{field}: expected a finite number")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    required = {
        "schema_version",
        "config_id",
        "task_id",
        "expected_challenge_sha256",
        "expected_nca_source_sha256",
        "expected_arc_utils_sha256",
        "steps",
        "rollout_steps",
        "pool_size",
        "batch_size",
        "seed",
        "top_k",
        "threads",
        "channels",
        "hidden_channels",
        "gene_size",
        "color_count",
        "expected_parameter_count",
        "learning_rate",
        "noise_level",
        "update_rate",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise InferenceInputError("inference config fields do not match contract")
    if value["schema_version"] != 1:
        raise InferenceInputError("unsupported inference config schema")
    for field in (
        "config_id",
        "task_id",
        "expected_challenge_sha256",
        "expected_nca_source_sha256",
        "expected_arc_utils_sha256",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise InferenceInputError(f"{field}: expected a nonempty string")
    if not re.fullmatch(r"[0-9a-f]{8}", value["task_id"]):
        raise InferenceInputError("task_id must be eight lowercase hexadecimal digits")
    for field in (
        "expected_challenge_sha256",
        "expected_nca_source_sha256",
        "expected_arc_utils_sha256",
    ):
        if SHA256_RE.fullmatch(value[field]) is None:
            raise InferenceInputError(f"{field}: invalid SHA-256")
    for field in (
        "steps",
        "rollout_steps",
        "pool_size",
        "batch_size",
        "top_k",
        "threads",
        "channels",
        "hidden_channels",
        "gene_size",
        "color_count",
        "expected_parameter_count",
    ):
        _positive_int(value[field], field)
    if type(value["seed"]) is not int:
        raise InferenceInputError("seed: expected an integer")
    if value["top_k"] != 2:
        raise InferenceInputError("this frozen adapter requires Top-2")
    if value["batch_size"] > value["pool_size"]:
        raise InferenceInputError("batch_size cannot exceed pool_size")
    learning_rate = _finite_float(value["learning_rate"], "learning_rate")
    noise_level = _finite_float(value["noise_level"], "noise_level")
    update_rate = _finite_float(value["update_rate"], "update_rate")
    if learning_rate <= 0:
        raise InferenceInputError("learning_rate must be positive")
    if not 0 <= noise_level <= 1 or not 0 <= update_rate <= 1:
        raise InferenceInputError("noise_level and update_rate must be in [0, 1]")
    return value


def load_arc_utils(upstream_root: Path, expected_sha256: str) -> Any:
    source = upstream_root / "arc_agi_utils.py"
    if sha256_file(source) != expected_sha256:
        raise InferenceInputError("upstream arc_agi_utils.py digest mismatch")
    sys.dont_write_bytecode = True
    specification = importlib.util.spec_from_file_location(
        "arc_nca_locked_arc_agi_utils", source
    )
    if specification is None or specification.loader is None:
        raise InferenceInputError("cannot load locked arc_agi_utils.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class DevicePortableCA(torch.nn.Module):
    """Audited CPU port of upstream ``NCA.py`` lines 271--299.

    Layer structure and operations are retained.  The only semantic device
    change is allocating the stochastic update mask on ``x.device`` rather
    than the upstream hard-coded ``cuda:0``.
    """

    def __init__(self, channels: int, hidden_channels: int, mask_n: int = 0):
        super().__init__()
        self.channels = channels
        self.perc = torch.nn.Conv2d(
            channels,
            8 * channels,
            3,
            padding=1,
            padding_mode="zeros",
            bias=False,
        )
        self.dropout = torch.nn.Dropout2d(p=0.4)
        self.dropout2 = torch.nn.Dropout2d(p=0.4)
        self.w1 = torch.nn.Conv2d(9 * channels, hidden_channels, 1)
        self.w2 = torch.nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.mask_n = mask_n

    def forward(self, x: torch.Tensor, update_rate: float = 0.5) -> torch.Tensor:
        y = self.perc(x)
        y = torch.cat((y, x), dim=1)
        y = self.dropout(y)
        y = self.w1(y)
        y = torch.relu(y)
        y = self.dropout2(y)
        y = self.w2(y)
        batch, _channels, height, width = y.shape
        update_mask = (
            torch.rand(batch, 1, height, width, device=x.device) + update_rate
        ).floor()
        return x + y * update_mask


def get_batch(
    pool: torch.Tensor,
    pristine: torch.Tensor,
    batch_size: int,
    noise_level: float,
) -> tuple[torch.Tensor, np.ndarray]:
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
    distances = ((pixels[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(
        dim=-1
    )
    return distances.argmin(dim=-1).cpu().tolist()


def run_inference(
    task: dict[str, Any], config: dict[str, Any], arc_utils: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    try:
        torch.set_num_interop_threads(config["threads"])
    except RuntimeError:
        if torch.get_num_interop_threads() != config["threads"]:
            raise
    torch.use_deterministic_algorithms(True)

    device = torch.device("cpu")
    channels = config["channels"]
    gene_size = config["gene_size"]
    color_count = config["color_count"]

    def encode(grid: torch.Tensor) -> torch.Tensor:
        return arc_utils.arc_to_nca_space(
            color_count,
            grid,
            channels,
            gene_size,
            device="cpu",
            mode="rgb",
            gene_location=list(range(gene_size)),
            is_invis=1,
        )

    train_inputs = [
        torch.tensor(pair["input"], dtype=torch.int64, device=device)
        for pair in task["train"]
    ]
    train_outputs = [
        torch.tensor(pair["output"], dtype=torch.int64, device=device)
        for pair in task["train"]
    ]
    test_inputs = [
        torch.tensor(pair["input"], dtype=torch.int64, device=device)
        for pair in task["test"]
    ]
    encoded_inputs = [encode(grid) for grid in train_inputs]
    encoded_outputs = [encode(grid) for grid in train_outputs]
    pools = [
        grid.tile(config["pool_size"], 1, 1, 1) for grid in encoded_inputs
    ]

    model = DevicePortableCA(channels, config["hidden_channels"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != config["expected_parameter_count"]:
        raise InferenceInputError(
            f"parameter count mismatch: {parameter_count}"
        )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise InferenceInputError("model parameter escaped CPU")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"])
    )
    final_loss: float | None = None
    for step in range(config["steps"]):
        model.train()
        demo = step % len(pools)
        with torch.no_grad():
            state, indices = get_batch(
                pools[demo],
                encoded_inputs[demo].clone(),
                config["batch_size"],
                float(config["noise_level"]),
            )
            target = encoded_outputs[demo].tile(
                config["batch_size"], 1, 1, 1
            )
        for _ in range(config["rollout_steps"]):
            state = model(state, float(config["update_rate"]))
        loss = (target[:, :4] - state[:, :4]).pow(2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite ARC_NCA training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad /= parameter.grad.norm() + 1e-8
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pools[demo][indices] = state.detach()
        final_loss = float(loss.detach())

    palette = torch.stack(
        [
            encode(torch.tensor([[color]], dtype=torch.int64, device=device))[
                :4, 0, 0
            ]
            for color in range(color_count)
        ]
    )
    outputs: list[dict[str, list[list[int]]]] = []
    model.eval()
    for test_index, grid in enumerate(test_inputs):
        attempts: dict[str, list[list[int]]] = {}
        for attempt in range(1, config["top_k"] + 1):
            torch.manual_seed(config["seed"] + 1000 * attempt + test_index)
            state = encode(grid)[None]
            with torch.no_grad():
                for _ in range(config["rollout_steps"]):
                    state = model(state, float(config["update_rate"]))
            decoded = decode_grid(state[0], palette)
            _validate_grid(decoded, f"prediction[{test_index}].attempt_{attempt}")
            attempts[f"attempt_{attempt}"] = decoded
        outputs.append(attempts)
    predictions = {config["task_id"]: outputs}
    metadata = {
        "schema_version": 1,
        "method_id": "arc-nca",
        "config_id": config["config_id"],
        "task_id": config["task_id"],
        "device": "cpu",
        "gpu_api_called": False,
        "test_output_fields_received": 0,
        "scorer_imported": False,
        "steps_completed": config["steps"],
        "rollout_steps": config["rollout_steps"],
        "attempts_generated": config["top_k"] * len(test_inputs),
        "parameter_count": parameter_count,
        "final_training_loss": final_loss,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return predictions, metadata


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--write-root", required=True, type=Path)
    args = parser.parse_args(argv)
    for name in ("challenge", "config", "upstream_root", "output", "metadata", "write_root"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.challenge.is_file() or not args.config.is_file():
        parser.error("challenge and config must be existing files")
    if not args.upstream_root.is_dir():
        parser.error("upstream root must be an existing directory")
    if args.output == args.metadata:
        parser.error("output and metadata paths must differ")
    if not _inside(args.output, args.write_root) or not _inside(
        args.metadata, args.write_root
    ):
        parser.error("all writes must remain inside --write-root")
    if args.output.exists() or args.metadata.exists():
        parser.error("output and metadata must not already exist")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.challenge.stem != config["task_id"]:
        raise InferenceInputError("challenge task ID does not match config")
    challenge_sha256 = sha256_file(args.challenge)
    if challenge_sha256 != config["expected_challenge_sha256"]:
        raise InferenceInputError("challenge digest mismatch")
    nca_source = args.upstream_root / "NCA.py"
    if sha256_file(nca_source) != config["expected_nca_source_sha256"]:
        raise InferenceInputError("upstream NCA.py digest mismatch")
    task = load_label_free_task(args.challenge)
    arc_utils = load_arc_utils(
        args.upstream_root, config["expected_arc_utils_sha256"]
    )
    predictions, metadata = run_inference(task, config, arc_utils)
    atomic_json(args.output, predictions)
    metadata["challenge_sha256"] = challenge_sha256
    metadata["config_sha256"] = sha256_file(args.config)
    metadata["prediction_sha256"] = sha256_file(args.output)
    metadata["upstream_nca_sha256"] = config["expected_nca_source_sha256"]
    metadata["upstream_arc_utils_sha256"] = config[
        "expected_arc_utils_sha256"
    ]
    atomic_json(args.metadata, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
