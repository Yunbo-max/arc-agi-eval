from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.challenge_runtime import tree_inventory, tree_sha256
from scripts import infer_compressarc_cpu as inference
from scripts import run_compressarc_strict_smoke as orchestrator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "compressarc_cpu_dev_smoke_v1.json"
UPSTREAM_ROOT = ROOT / "external" / "CompressARC"
FROZEN_ROOT = (
    ROOT
    / "reports"
    / "e0-development-split"
    / "20260806-frozen-known-overlap-excluded-dev-audit-v1"
)


def read_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class CompressArcStrictConfigTests(unittest.TestCase):
    def test_safe_child_config_contains_no_hidden_label_locator(self) -> None:
        config = read_config()
        safe = orchestrator.safe_inference_config(config)
        encoded = json.dumps(safe, sort_keys=True)
        for forbidden in (
            "solution",
            "scoring",
            config["expected_solution_sha256"],
            config["source_task_path"],
            config["frozen_runtime_directory"],
            config["upstream"]["expected_full_tree_sha256"],
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(set(safe), {
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
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(safe), encoding="utf-8")
            self.assertEqual(inference.load_config(path), safe)

    def test_frozen_target_matches_only_label_free_selection_metadata(self) -> None:
        config = read_config()
        manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text())
        challenge, representative, challenge_path = (
            orchestrator._label_free_target_from_frozen_record(config, manifest)
        )
        self.assertEqual(representative["task_id"], "3c9b0459")
        self.assertEqual(hashlib.sha256(challenge_path.read_bytes()).hexdigest(),
                         config["expected_challenge_sha256"])
        self.assertEqual((len(challenge["train"]), len(challenge["test"])), (4, 1))
        self.assertTrue(all("output" not in pair for pair in challenge["test"]))
        self.assertFalse(config["analyst_test_label_exposure"])
        self.assertFalse(config["performance_tuning_from_test_labels"])
        self.assertFalse(config["public_execution_authorized"])

    def test_upstream_full_and_code_only_hash_domains_are_frozen(self) -> None:
        config = read_config()
        upstream = config["upstream"]
        inventory = tree_inventory(UPSTREAM_ROOT)
        self.assertEqual(len(inventory), upstream["expected_full_file_count"])
        self.assertEqual(tree_sha256(UPSTREAM_ROOT), upstream["expected_full_tree_sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "safe-upstream"
            copied = orchestrator.copy_safe_upstream_tree(
                UPSTREAM_ROOT, stage, upstream["safe_file_allowlist"]
            )
            observed, count = orchestrator.inference_tree_sha256(stage)
        self.assertEqual(count, upstream["expected_safe_file_count"])
        self.assertEqual(observed, upstream["expected_safe_tree_sha256"])
        self.assertEqual(len(copied), 11)
        allowed = set(upstream["safe_file_allowlist"])
        self.assertTrue(
            {"dataset", "scoring.py", "solve_task.py", "parallel_train.py"}
            .isdisjoint(allowed)
        )

    def test_environment_deviations_are_explicit(self) -> None:
        config = read_config()
        expected = config["expected_environment"]
        deviations = config["compatibility_deviations"]
        self.assertEqual(expected["python_path"], ".venvs/compressarc/bin/python")
        self.assertEqual(expected["cuda_visible_devices"], "")
        self.assertNotEqual(
            deviations["upstream_requirements"]["torch"],
            deviations["observed_environment"]["torch"],
        )
        self.assertNotEqual(
            deviations["upstream_requirements"]["numpy"],
            deviations["observed_environment"]["numpy"],
        )
        self.assertIn("does not establish", deviations["parity_effect"])

    def test_source_lock_matches_frozen_repository(self) -> None:
        config = read_config()
        locks = json.loads(
            (ROOT / "configs" / "source_locks.json").read_text(encoding="utf-8")
        )
        lock = locks["sources"]["compressarc"]
        self.assertEqual(lock["revision"], config["upstream"]["revision"])
        self.assertEqual(
            lock["repository_path"], config["upstream"]["repository_path"]
        )


class CompressArcLifecycleStructureTests(unittest.TestCase):
    def test_scoring_payload_loader_is_called_only_after_both_inference_calls(self) -> None:
        source = inspect.getsource(orchestrator.main)
        second_inference = source.index("inference_event_b = run_child(")
        prediction_check = source.index("A/B prediction bytes differ")
        materialization = source.index("_load_scoring_payload_after_inference(")
        score_loop = source.index("for prediction, answer_tree")
        self.assertLess(second_inference, prediction_check)
        self.assertLess(prediction_check, materialization)
        self.assertLess(materialization, score_loop)

    def test_pre_inference_helper_does_not_load_source_or_solution_payload(self) -> None:
        source = inspect.getsource(
            orchestrator._label_free_target_from_frozen_record
        )
        self.assertNotIn("load_task", source)
        self.assertNotIn("read_json(solution", source)
        scoring_source = inspect.getsource(
            orchestrator._load_scoring_payload_after_inference
        )
        self.assertIn("load_task", scoring_source)
        self.assertIn("read_json(solution_path)", scoring_source)

    def test_pre_run_contract_keeps_public_execution_closed(self) -> None:
        config = read_config()
        self.assertEqual(
            config["pre_run_contract"],
            {
                "strict_runtime_passed": 1,
                "admitted_configuration_count": 0,
                "performance_eligible_count": 0,
                "locked_public_solver_run": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
