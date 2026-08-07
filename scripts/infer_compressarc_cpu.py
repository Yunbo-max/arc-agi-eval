#!/usr/bin/env python3
"""Run a reduced CompressARC adaptation in an inference-only CPU process.

The process accepts no answer/scoring path, rejects every test ``output``, and
does not import the repository evaluator.  It loads a hash-locked, label-free
subset of the upstream source tree.  The sole source transformation replaces
CompressARC's module-level CUDA default with CPU before the module is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import types
from typing import Any, Sequence

import numpy as np
import torch


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CUDA_DEFAULT_LINE = "torch.set_default_device('cuda')"
CPU_DEFAULT_LINE = "torch.set_default_device('cpu')"


class InferenceInputError(ValueError):
    """Raised when an inference-visible input violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, int]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    manifest = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest(), len(files)


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
        "expected_safe_tree_sha256",
        "expected_safe_file_count",
        "expected_arc_compressor_sha256",
        "steps",
        "seed",
        "top_k",
        "threads",
        "learning_rate",
        "beta_1",
        "beta_2",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise InferenceInputError("inference config fields do not match contract")
    if value["schema_version"] != 1:
        raise InferenceInputError("unsupported inference config schema")
    for field in (
        "config_id",
        "task_id",
        "expected_challenge_sha256",
        "expected_safe_tree_sha256",
        "expected_arc_compressor_sha256",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise InferenceInputError(f"{field}: expected a nonempty string")
    if re.fullmatch(r"[0-9a-f]{8}", value["task_id"]) is None:
        raise InferenceInputError("task_id must be eight lowercase hexadecimal digits")
    for field in (
        "expected_challenge_sha256",
        "expected_safe_tree_sha256",
        "expected_arc_compressor_sha256",
    ):
        if SHA256_RE.fullmatch(value[field]) is None:
            raise InferenceInputError(f"{field}: invalid SHA-256")
    for field in (
        "expected_safe_file_count",
        "steps",
        "top_k",
        "threads",
    ):
        _positive_int(value[field], field)
    if type(value["seed"]) is not int:
        raise InferenceInputError("seed: expected an integer")
    if value["top_k"] != 2:
        raise InferenceInputError("this frozen adapter requires Top-2")
    learning_rate = _finite_float(value["learning_rate"], "learning_rate")
    beta_1 = _finite_float(value["beta_1"], "beta_1")
    beta_2 = _finite_float(value["beta_2"], "beta_2")
    if learning_rate <= 0:
        raise InferenceInputError("learning_rate must be positive")
    if not 0 <= beta_1 < beta_2 < 1:
        raise InferenceInputError("optimizer betas must satisfy 0 <= beta_1 < beta_2 < 1")
    return value


def load_upstream_modules(
    upstream_root: Path,
    *,
    expected_tree_sha256: str,
    expected_file_count: int,
    expected_arc_compressor_sha256: str,
) -> tuple[Any, Any, Any, Any, str]:
    observed_tree_sha256, observed_file_count = tree_sha256(upstream_root)
    if observed_tree_sha256 != expected_tree_sha256:
        raise InferenceInputError("safe upstream tree digest mismatch")
    if observed_file_count != expected_file_count:
        raise InferenceInputError("safe upstream file count mismatch")

    compressor_path = upstream_root / "arc_compressor.py"
    if sha256_file(compressor_path) != expected_arc_compressor_sha256:
        raise InferenceInputError("upstream arc_compressor.py digest mismatch")
    compressor_source = compressor_path.read_text(encoding="utf-8")
    if compressor_source.count(CUDA_DEFAULT_LINE) != 1:
        raise InferenceInputError("upstream CUDA default line does not match contract")
    portable_source = compressor_source.replace(
        CUDA_DEFAULT_LINE,
        CPU_DEFAULT_LINE,
        1,
    )
    portable_sha256 = hashlib.sha256(portable_source.encode("utf-8")).hexdigest()

    sys.dont_write_bytecode = True
    source_entry = str(upstream_root)
    sys.path.insert(0, source_entry)
    try:
        preprocessing = importlib.import_module("preprocessing")
        compressor = types.ModuleType("arc_compressor")
        compressor.__file__ = str(compressor_path)
        compressor.__package__ = ""
        sys.modules["arc_compressor"] = compressor
        exec(compile(portable_source, str(compressor_path), "exec"), compressor.__dict__)
        train = importlib.import_module("train")
        selection = importlib.import_module("solution_selection")
    finally:
        if sys.path and sys.path[0] == source_entry:
            sys.path.pop(0)
        elif source_entry in sys.path:
            sys.path.remove(source_entry)
    return preprocessing, compressor, train, selection, portable_sha256


def require_cpu_tensors(named_values: Sequence[tuple[str, Any]]) -> None:
    """Fail closed unless every declared runtime tensor is resident on CPU."""

    for name, value in named_values:
        if not isinstance(value, torch.Tensor):
            raise InferenceInputError(f"{name}: expected a torch Tensor")
        if value.device.type != "cpu":
            raise InferenceInputError(f"{name}: expected CPU, got {value.device}")


def require_optimizer_state_cpu(optimizer: Any) -> None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        if not isinstance(state, dict):
            raise InferenceInputError("optimizer state does not match contract")
        for field, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise InferenceInputError(
                    f"optimizer[{parameter_index}].{field}: expected CPU, "
                    f"got {value.device}"
                )


def _prediction_grid(value: Any, where: str) -> list[list[int]]:
    if isinstance(value, tuple):
        value = [list(row) if isinstance(row, tuple) else row for row in value]
    return _validate_grid(value, where)


def predictions_from_logger(task_id: str, logger: Any) -> dict[str, Any]:
    first = logger.solution_most_frequent
    second = logger.solution_second_most_frequent
    if not isinstance(first, tuple) or not isinstance(second, tuple):
        raise InferenceInputError("upstream logger did not emit two solution tuples")
    if len(first) != len(second) or not first:
        raise InferenceInputError("upstream logger output count is invalid")
    outputs = []
    for index, (attempt_1, attempt_2) in enumerate(zip(first, second)):
        outputs.append(
            {
                "attempt_1": _prediction_grid(
                    attempt_1, f"{task_id}[{index}].attempt_1"
                ),
                "attempt_2": _prediction_grid(
                    attempt_2, f"{task_id}[{index}].attempt_2"
                ),
            }
        )
    return {task_id: outputs}


def run_inference(
    challenge_path: Path,
    config_path: Path,
    upstream_root: Path,
    output_path: Path,
    metadata_path: Path,
    write_root: Path,
) -> None:
    config = load_config(config_path)
    task = load_label_free_task(challenge_path)
    if sha256_file(challenge_path) != config["expected_challenge_sha256"]:
        raise InferenceInputError("challenge digest mismatch")

    threads = config["threads"]
    required_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": str(config["seed"]),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MPLBACKEND": "Agg",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, expected in required_environment.items():
        if os.environ.get(name) != expected:
            raise InferenceInputError(
                f"{name}: expected frozen value {expected!r}"
            )
    mpl_config = os.environ.get("MPLCONFIGDIR")
    if not mpl_config or not _inside(Path(mpl_config).resolve(), write_root):
        raise InferenceInputError("MPLCONFIGDIR must remain inside --write-root")
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(threads)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_default_device("cpu")

    preprocessing, compressor, train, selection, portable_sha256 = (
        load_upstream_modules(
            upstream_root,
            expected_tree_sha256=config["expected_safe_tree_sha256"],
            expected_file_count=config["expected_safe_file_count"],
            expected_arc_compressor_sha256=config[
                "expected_arc_compressor_sha256"
            ],
        )
    )
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if str(torch.get_default_device()) != "cpu":
        raise InferenceInputError("portable upstream module did not retain CPU default")

    upstream_task = preprocessing.Task(config["task_id"], task, None)
    model = compressor.ARCCompressor(upstream_task)
    optimizer = torch.optim.Adam(
        model.weights_list,
        lr=config["learning_rate"],
        betas=(config["beta_1"], config["beta_2"]),
    )
    logger = selection.Logger(upstream_task)
    require_cpu_tensors(
        [
            ("task.problem", upstream_task.problem),
            ("task.masks", upstream_task.masks),
            *((f"model.weights_list[{index}]", weight) for index, weight in enumerate(model.weights_list)),
            ("logger.current_logits", logger.current_logits),
            ("logger.current_x_mask", logger.current_x_mask),
            ("logger.current_y_mask", logger.current_y_mask),
            ("logger.ema_logits", logger.ema_logits),
            ("logger.ema_x_mask", logger.ema_x_mask),
            ("logger.ema_y_mask", logger.ema_y_mask),
        ]
    )
    for step in range(config["steps"]):
        train.take_step(upstream_task, model, optimizer, step, logger)
        require_optimizer_state_cpu(optimizer)

    predictions = predictions_from_logger(config["task_id"], logger)
    atomic_json(output_path, predictions)
    prediction_sha256 = sha256_file(output_path)
    parameter_count = sum(int(weight.numel()) for weight in model.weights_list)
    metadata = {
        "schema_version": 1,
        "method_id": "compressarc",
        "config_id": config["config_id"],
        "task_id": config["task_id"],
        "challenge_sha256": config["expected_challenge_sha256"],
        "prediction_sha256": prediction_sha256,
        "attempts_generated": sum(
            len(outputs) * config["top_k"] for outputs in predictions.values()
        ),
        "steps_completed": config["steps"],
        "final_training_loss": float(logger.loss_curve[-1]),
        "parameter_count": parameter_count,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_environment_verified": True,
        "gpu_api_called": False,
        "test_output_fields_received": 0,
        "scorer_imported": False,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "tqdm": importlib.metadata.version("tqdm"),
        "upstream_arc_compressor_sha256": config[
            "expected_arc_compressor_sha256"
        ],
        "portable_arc_compressor_sha256": portable_sha256,
        "device_change": f"{CUDA_DEFAULT_LINE} -> {CPU_DEFAULT_LINE}",
    }
    atomic_json(metadata_path, metadata)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--write-root", type=Path, required=True)
    args = parser.parse_args(argv)
    for field in ("challenge", "config", "upstream_root", "write_root"):
        setattr(args, field, getattr(args, field).resolve())
    for field in ("output", "metadata"):
        setattr(args, field, getattr(args, field).resolve())
        if not _inside(getattr(args, field), args.write_root):
            parser.error(f"--{field.replace('_', '-')} must be inside --write-root")
    if args.output == args.metadata:
        parser.error("--output and --metadata must differ")
    for field in ("challenge", "config"):
        if not getattr(args, field).is_file():
            parser.error(f"--{field.replace('_', '-')} is not a file")
    if not args.upstream_root.is_dir():
        parser.error("--upstream-root is not a directory")
    if not args.write_root.is_dir():
        parser.error("--write-root is not a directory")
    if args.output.exists() or args.metadata.exists():
        parser.error("output paths must not already exist")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    args = parse_args(argv)
    run_inference(
        args.challenge,
        args.config,
        args.upstream_root,
        args.output,
        args.metadata,
        args.write_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
