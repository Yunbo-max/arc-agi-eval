from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from arc_agi_eval.challenge_runtime import tree_inventory, tree_sha256
from scripts import infer_arc_nca_cpu as inference
from scripts.run_arc_nca_strict_smoke import safe_inference_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "arc_nca_cpu_dev_smoke_v1.json"
INFERENCE_PATH = ROOT / "scripts" / "infer_arc_nca_cpu.py"
UPSTREAM_ROOT = ROOT / "external" / "ARC_NCA"


def read_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class ArcNcaInferenceBoundaryTests(unittest.TestCase):
    def test_inference_source_imports_no_scorer_or_repository_package(self) -> None:
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
        self.assertNotIn("torch.cuda.", source)
        self.assertNotIn("--solution", source)
        self.assertNotIn("--score", source)

    def test_label_free_loader_rejects_hidden_output(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deadbeef.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            with self.assertRaisesRegex(
                inference.InferenceInputError, "hidden output supplied"
            ):
                inference.load_label_free_task(path)

    def test_safe_child_config_contains_no_label_locator_or_digest(self) -> None:
        config = read_config()
        safe = safe_inference_config(config)
        encoded = json.dumps(safe, sort_keys=True)
        self.assertNotIn("solution", encoded.lower())
        self.assertNotIn("scoring", encoded.lower())
        self.assertNotIn(config["expected_solution_sha256"], encoded)
        self.assertNotIn(config["source_task_path"], encoded)
        self.assertNotIn(config["frozen_runtime_directory"], encoded)
        self.assertEqual(safe["task_id"], "6150a2bd")
        self.assertEqual(safe["top_k"], 2)

    def test_portable_ca_forward_is_cpu_and_parameter_exact(self) -> None:
        model = inference.DevicePortableCA(50, 264)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 312264)
        state = torch.zeros((1, 50, 3, 3), dtype=torch.float32)
        torch.manual_seed(0)
        output = model(state, 0.5)
        self.assertEqual(tuple(output.shape), (1, 50, 3, 3))
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_synthetic_cpu_inference_replays_deterministically(self) -> None:
        full = read_config()
        safe = safe_inference_config(full)
        safe.update(
            {
                "steps": 1,
                "rollout_steps": 1,
                "pool_size": 2,
                "batch_size": 2,
            }
        )
        task = {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
            ],
            "test": [{"input": [[1, 1], [0, 0]]}],
        }
        arc_utils = inference.load_arc_utils(
            UPSTREAM_ROOT, safe["expected_arc_utils_sha256"]
        )
        first, first_metadata = inference.run_inference(task, safe, arc_utils)
        second, second_metadata = inference.run_inference(task, safe, arc_utils)
        self.assertEqual(first, second)
        self.assertEqual(first_metadata["final_training_loss"], second_metadata["final_training_loss"])
        self.assertEqual(set(first), {"6150a2bd"})
        self.assertEqual(
            set(first["6150a2bd"][0]), {"attempt_1", "attempt_2"}
        )
        self.assertEqual(first_metadata["device"], "cpu")
        self.assertFalse(first_metadata["gpu_api_called"])


class ArcNcaFrozenConfigTests(unittest.TestCase):
    def test_target_and_upstream_hashes_are_frozen(self) -> None:
        config = read_config()
        challenge = (
            ROOT
            / config["frozen_runtime_directory"]
            / "inference"
            / "dev-audit"
            / f"{config['task_id']}.json"
        )
        self.assertEqual(
            hashlib.sha256(challenge.read_bytes()).hexdigest(),
            config["expected_challenge_sha256"],
        )
        task = inference.load_label_free_task(challenge)
        self.assertEqual(len(task["train"]), 2)
        self.assertEqual(len(task["test"]), 1)
        self.assertTrue(
            all(
                len(pair["input"]) == len(pair["output"]) == 3
                and len(pair["input"][0]) == len(pair["output"][0]) == 3
                for pair in task["train"]
            )
        )
        self.assertEqual(config["optimization"]["steps"], len(task["train"]))
        self.assertTrue(config["analyst_test_label_exposure"])
        self.assertFalse(config["performance_tuning_from_test_labels"])
        self.assertFalse(config["public_execution_authorized"])

        inventory = tree_inventory(UPSTREAM_ROOT)
        self.assertEqual(len(inventory), config["upstream"]["expected_file_count"])
        self.assertEqual(
            tree_sha256(UPSTREAM_ROOT),
            config["upstream"]["expected_tree_sha256"],
        )
        for name, field in (
            ("NCA.py", "expected_nca_source_sha256"),
            ("arc_agi_utils.py", "expected_arc_utils_sha256"),
            ("LICENSE", "expected_license_sha256"),
        ):
            self.assertEqual(
                hashlib.sha256((UPSTREAM_ROOT / name).read_bytes()).hexdigest(),
                config["upstream"][field],
            )

    def test_source_lock_points_to_the_retained_tree(self) -> None:
        source_locks = json.loads(
            (ROOT / "configs" / "source_locks.json").read_text(encoding="utf-8")
        )
        entry = source_locks["sources"]["arc-nca"]
        self.assertEqual(entry["repository_path"], "external/ARC_NCA")
        self.assertNotIn("asset_subpath", entry)
        self.assertEqual(entry["revision"], read_config()["upstream"]["revision"])


if __name__ == "__main__":
    unittest.main()
