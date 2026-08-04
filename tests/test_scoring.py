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


if __name__ == "__main__":
    unittest.main()
