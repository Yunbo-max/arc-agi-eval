import json
from pathlib import Path
import sys
import tempfile
import unittest

from arc_agi_eval.challenge_runtime import (
    mutate_hidden_test_labels,
    run_logged_process,
    sentinel_predictions,
    tree_inventory,
    tree_sha256,
)
from arc_agi_eval.firewall import challenge_only


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports"
    / "e0-challenge-runtime"
    / "20260806-deterministic-baseline-dev-audit-v1"
)


class ChallengeRuntimeTests(unittest.TestCase):
    def test_hidden_label_mutation_preserves_challenge(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3, 9]], "output": [[4, 0]]}],
        }
        mutated = mutate_hidden_test_labels(task, offset=1)
        self.assertEqual(mutated["test"][0]["output"], [[5, 1]])
        self.assertEqual(challenge_only(task), challenge_only(mutated))

    def test_hidden_label_mutation_rejects_noop(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        with self.assertRaisesRegex(ValueError, "nonzero"):
            mutate_hidden_test_labels(task, offset=10)

    def test_sentinel_copies_all_labeled_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_path = Path(temporary) / "task0001.json"
            task_path.write_text(
                json.dumps(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]], "output": [[4]]}],
                    }
                ),
                encoding="utf-8",
            )
            predictions = sentinel_predictions([task_path], top_k=2)
        self.assertEqual(
            predictions["task0001"],
            [{"attempt_1": [[4]], "attempt_2": [[4]]}],
        )

    def test_tree_digest_is_path_and_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "a" / "one.txt").write_text("one", encoding="utf-8")
            first = tree_sha256(root)
            self.assertEqual(len(tree_inventory(root)), 1)
            (root / "a" / "one.txt").write_text("two", encoding="utf-8")
            self.assertNotEqual(first, tree_sha256(root))

    def test_logged_process_records_terminal_pid_and_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = run_logged_process(
                [sys.executable, "-c", "print('ok')"],
                name="fixture-inference",
                kind="inference",
                cwd=ROOT,
                timeout_seconds=10,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                environment={"PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(event.status, "passed")
            self.assertEqual(event.return_code, 0)
            self.assertGreater(event.pid, 0)
            self.assertEqual((root / "stdout.log").read_text(), "ok\n")

    def test_saved_report_runtime_invariants(self) -> None:
        if not (REPORT / "run.json").is_file():
            self.skipTest("challenge-runtime report has not been generated")
        run = json.loads((REPORT / "run.json").read_text(encoding="utf-8"))
        summary = json.loads(
            (REPORT / "results-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run["status"], "passed")
        self.assertEqual(run["evidence_scope"], "solver_prediction_smoke")
        self.assertEqual(run["data"]["task_count"], 94)
        self.assertEqual(run["results"]["primary_metric"]["denominator"], 97)
        self.assertTrue(summary["checks"]["prediction_bytes_a_b_equal"])
        self.assertTrue(summary["checks"]["scoring_after_all_inference"])
        self.assertEqual(summary["sentinel_score_a"]["outputs_exact"], 97)
        self.assertEqual(summary["sentinel_score_b"]["outputs_exact"], 0)


if __name__ == "__main__":
    unittest.main()
