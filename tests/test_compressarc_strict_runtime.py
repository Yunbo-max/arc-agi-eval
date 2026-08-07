from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts import infer_compressarc_cpu as inference


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = ROOT / "scripts" / "infer_compressarc_cpu.py"
UPSTREAM_ROOT = ROOT / "external" / "CompressARC"
FROZEN_CHALLENGE = (
    ROOT
    / "reports"
    / "e0-development-split"
    / "20260806-frozen-known-overlap-excluded-dev-audit-v1"
    / "inference"
    / "dev-audit"
    / "3c9b0459.json"
)
VENV_PYTHON = ROOT / ".venvs" / "compressarc" / "bin" / "python"
SAFE_UPSTREAM_FILES = (
    "LICENSE",
    "README.md",
    "requirements.txt",
    "arc_compressor.py",
    "initializers.py",
    "layers.py",
    "multitensor_systems.py",
    "preprocessing.py",
    "solution_selection.py",
    "train.py",
    "visualization.py",
)
EXPECTED_SAFE_TREE_SHA256 = (
    "d89f1b1fbc8a2567e8f62bacc6b88abb510f7c6bec5c3230bfa1aa7297fecb7d"
)
EXPECTED_ARC_COMPRESSOR_SHA256 = (
    "7a89e502c1106f419dc34374aea858fa4182eb605ecbf9a473ca47da655a97ce"
)
EXPECTED_PORTABLE_SHA256 = (
    "dc57fa3bea5a8a30352fdcf5ea76dda73ce41fc71972eac389be52c7cbdd90d1"
)
EXPECTED_CHALLENGE_SHA256 = (
    "9fb838b74e287bb9fd223822f2e287a07df58ab9ea0e29daaaf1ea2093c7e6ab"
)


def safe_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "config_id": "compressarc-cpu-dev-smoke-v1",
        "task_id": "3c9b0459",
        "expected_challenge_sha256": EXPECTED_CHALLENGE_SHA256,
        "expected_safe_tree_sha256": EXPECTED_SAFE_TREE_SHA256,
        "expected_safe_file_count": len(SAFE_UPSTREAM_FILES),
        "expected_arc_compressor_sha256": EXPECTED_ARC_COMPRESSOR_SHA256,
        "steps": 2,
        "seed": 0,
        "top_k": 2,
        "threads": 1,
        "learning_rate": 0.01,
        "beta_1": 0.5,
        "beta_2": 0.9,
    }


def copy_safe_tree(destination: Path) -> None:
    for relative in SAFE_UPSTREAM_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(UPSTREAM_ROOT / relative, target)


class CompressArcInferenceBoundaryTests(unittest.TestCase):
    def test_source_imports_no_scorer_repository_or_cuda_api(self) -> None:
        source = INFERENCE_PATH.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(any(name.startswith("arc_agi_eval") for name in imported))
        self.assertFalse(any("scor" in name.lower() for name in imported))
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn("preprocess_tasks(", source)
        for forbidden in ("--solution", "--answer", "--score", "--scoring"):
            self.assertNotIn(forbidden, source)

    def test_label_free_loader_rejects_hidden_output_without_writing(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "3c9b0459.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(safe_config()), encoding="utf-8")
            output = root / "predictions.json"
            metadata = root / "metadata.json"
            with self.assertRaisesRegex(
                inference.InferenceInputError, "hidden output supplied"
            ):
                inference.run_inference(
                    path,
                    config_path,
                    root,
                    output,
                    metadata,
                    root,
                )
            self.assertFalse(output.exists())
            self.assertFalse(metadata.exists())

    def test_config_contract_rejects_unknown_fields(self) -> None:
        config = safe_config()
        config["solution_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                inference.InferenceInputError, "fields do not match contract"
            ):
                inference.load_config(path)

    def test_code_only_tree_and_portable_patch_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "upstream"
            copy_safe_tree(stage)
            digest, count = inference.tree_sha256(stage)
        self.assertEqual(digest, EXPECTED_SAFE_TREE_SHA256)
        self.assertEqual(count, 11)
        self.assertEqual(
            inference.sha256_file(UPSTREAM_ROOT / "arc_compressor.py"),
            EXPECTED_ARC_COMPRESSOR_SHA256,
        )
        source = (UPSTREAM_ROOT / "arc_compressor.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(inference.CUDA_DEFAULT_LINE), 1)
        portable = source.replace(
            inference.CUDA_DEFAULT_LINE, inference.CPU_DEFAULT_LINE, 1
        )
        self.assertEqual(
            hashlib.sha256(portable.encode("utf-8")).hexdigest(),
            EXPECTED_PORTABLE_SHA256,
        )
        forbidden = {
            "dataset",
            "scoring.py",
            "solve_task.py",
            "parallel_train.py",
        }
        self.assertTrue(forbidden.isdisjoint(SAFE_UPSTREAM_FILES))

    def test_frozen_target_is_label_free_and_shape_selected(self) -> None:
        self.assertEqual(
            inference.sha256_file(FROZEN_CHALLENGE), EXPECTED_CHALLENGE_SHA256
        )
        task = inference.load_label_free_task(FROZEN_CHALLENGE)
        self.assertEqual((len(task["train"]), len(task["test"])), (4, 1))
        for pair in [*task["train"], *task["test"]]:
            self.assertEqual((len(pair["input"]), len(pair["input"][0])), (3, 3))
        self.assertTrue(
            all(
                (len(pair["output"]), len(pair["output"][0])) == (3, 3)
                for pair in task["train"]
            )
        )
        self.assertTrue(all("output" not in pair for pair in task["test"]))

    @unittest.skipUnless(VENV_PYTHON.is_file(), "locked CompressARC venv absent")
    def test_target_cpu_inference_replays_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "upstream"
            copy_safe_tree(stage)
            config_path = root / "inference-config.json"
            config_path.write_text(
                json.dumps(safe_config(), sort_keys=True), encoding="utf-8"
            )
            outputs: list[bytes] = []
            metadata_records: list[dict[str, object]] = []
            for replay in ("a", "b"):
                runtime = root / replay
                work = runtime / "work"
                mpl = runtime / "mplconfig"
                work.mkdir(parents=True)
                mpl.mkdir(parents=True)
                output = runtime / "predictions.json"
                metadata = runtime / "metadata.json"
                environment = {
                    "CUDA_VISIBLE_DEVICES": "",
                    "LC_ALL": "C.UTF-8",
                    "MPLBACKEND": "Agg",
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MPLCONFIGDIR": str(mpl),
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TZ": "UTC",
                }
                completed = subprocess.run(
                    [
                        str(VENV_PYTHON),
                        str(INFERENCE_PATH),
                        "--challenge",
                        str(FROZEN_CHALLENGE),
                        "--config",
                        str(config_path),
                        "--upstream-root",
                        str(stage),
                        "--output",
                        str(output),
                        "--metadata",
                        str(metadata),
                        "--write-root",
                        str(runtime),
                    ],
                    cwd=work,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"stdout={completed.stdout}\nstderr={completed.stderr}",
                )
                outputs.append(output.read_bytes())
                metadata_records.append(json.loads(metadata.read_text()))

            self.assertEqual(outputs[0], outputs[1])
            for metadata in metadata_records:
                self.assertEqual(metadata["device"], "cpu")
                self.assertEqual(metadata["cuda_visible_devices"], "")
                self.assertFalse(metadata["gpu_api_called"])
                self.assertFalse(metadata["scorer_imported"])
                self.assertEqual(metadata["test_output_fields_received"], 0)
                self.assertEqual(metadata["steps_completed"], 2)
                self.assertTrue(metadata["required_environment_verified"])
                self.assertEqual(metadata["matplotlib"], "3.10.0")
                self.assertEqual(metadata["tqdm"], "4.70.0")
                self.assertEqual(
                    metadata["portable_arc_compressor_sha256"],
                    EXPECTED_PORTABLE_SHA256,
                )


if __name__ == "__main__":
    unittest.main()
