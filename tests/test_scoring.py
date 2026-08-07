import tempfile
import unittest
from pathlib import Path

from arc_agi_eval.scoring import PredictionValidationError, score_prediction_file, score_predictions


FIXTURES = Path(__file__).parent / "fixtures"
TASK_DIR = FIXTURES / "data" / "evaluation"


class ScoringTests(unittest.TestCase):
    def test_top_one_requires_every_test_output(self) -> None:
        score = score_prediction_file(FIXTURES / "predictions.json", TASK_DIR, top_k=1)
        self.assertEqual(score.tasks_exact, 0)
        self.assertEqual(score.outputs_exact, 1)
        self.assertEqual(score.cells_correct, 5)
        self.assertEqual(score.cells_total, 6)

    def test_top_two_can_solve_with_different_attempts_per_output(self) -> None:
        score = score_prediction_file(FIXTURES / "predictions.json", TASK_DIR)
        self.assertEqual(score.tasks_exact, 1)
        self.assertEqual(score.outputs_exact, 2)
        self.assertEqual(score.cells_correct, 6)
        self.assertEqual(score.cell_accuracy, 1.0)

    def test_top_k_uses_the_configured_attempt_budget(self) -> None:
        answers = {"task": [[[9]]]}
        predictions = {
            "task": [
                {
                    "attempt_1": [[0]],
                    "attempt_2": [[0]],
                    "attempt_3": [[9]],
                }
            ]
        }
        top_two = score_predictions(predictions, answers, top_k=2)
        top_three = score_predictions(predictions, answers, top_k=3)
        self.assertEqual(top_two.tasks_exact, 0)
        self.assertEqual(top_three.tasks_exact, 1)

    def test_missing_task_and_wrong_shape_receive_zero_credit(self) -> None:
        answers = {
            "missing": [[[1]]],
            "shape": [[[1, 1]]],
        }
        predictions = {"shape": [{"attempt_1": [[1], [1]]}]}
        score = score_predictions(predictions, answers, top_k=1)
        self.assertEqual(score.tasks_exact, 0)
        self.assertEqual(score.cells_correct, 0)
        self.assertEqual(score.cells_total, 3)
        self.assertEqual(score.tasks_predicted, 1)

    def test_unknown_task_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "unknown task ID"):
            score_predictions({"typo": [{"attempt_1": [[0]]}]}, {"known": [[[0]]]})

    def test_task_and_output_weighting_are_reported_separately(self) -> None:
        answers = {"multi": [[[1]], [[2]]], "single": [[[3]]]}
        predictions = {
            "multi": [{"attempt_1": [[1]]}, {"attempt_1": [[2]]}],
            "single": [{"attempt_1": [[0]]}],
        }
        score = score_predictions(predictions, answers, top_k=1)
        self.assertEqual((score.outputs_exact, score.outputs_total), (2, 3))
        self.assertEqual((score.tasks_exact, score.tasks_total), (1, 2))
        self.assertEqual(score.output_exact_accuracy, 2 / 3)
        self.assertEqual(score.task_exact_accuracy, 1 / 2)

    def test_metric_contract_declares_output_exact_as_primary(self) -> None:
        answers = {"multi": [[[1]], [[2]]], "single": [[[3]]]}
        predictions = {
            "multi": [{"attempt_1": [[1]]}, {"attempt_1": [[2]]}],
            "single": [{"attempt_1": [[0]]}],
        }
        payload = score_predictions(predictions, answers, top_k=2).as_dict()
        self.assertEqual(payload["score_schema_version"], 2)
        primary = payload["metric_contract"]["primary"]
        self.assertEqual(primary["name"], "output_exact_pass_at_k")
        self.assertEqual(primary["top_k"], 2)
        self.assertEqual((primary["numerator"], primary["denominator"]), (2, 3))
        strict = payload["metric_contract"]["secondary"][
            "strict_task_exact_pass_at_k"
        ]
        self.assertEqual((strict["numerator"], strict["denominator"]), (1, 2))
        self.assertEqual(
            payload["metric_contract"]["secondary"]["micro_cell_accuracy"][
                "role"
            ],
            "diagnostic_only",
        )

    def test_malformed_attempt_keys_and_noncontiguous_attempts_are_rejected(self) -> None:
        answers = {"task": [[[1]]]}
        with self.assertRaisesRegex(PredictionValidationError, "invalid attempt key"):
            score_predictions({"task": [{"guess_1": [[1]]}]}, answers)
        with self.assertRaisesRegex(PredictionValidationError, "contiguous"):
            score_predictions({"task": [{"attempt_2": [[1]]}]}, answers)

    def test_duplicate_prediction_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction = Path(temporary) / "predictions.json"
            prediction.write_text(
                '{"multi": [{"attempt_1": [[0]], "attempt_1": [[1]]}, '
                '{"attempt_1": [[0]]}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PredictionValidationError, "duplicate object key"):
                score_prediction_file(prediction, TASK_DIR)


if __name__ == "__main__":
    unittest.main()
