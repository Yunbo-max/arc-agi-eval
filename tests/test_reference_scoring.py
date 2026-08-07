import random
import unittest

from arc_agi_eval.reference_scoring import reference_exact_score
from arc_agi_eval.scoring import score_predictions


def random_grid(rng: random.Random) -> list[list[int]]:
    height = rng.randint(1, 4)
    width = rng.randint(1, 4)
    return [[rng.randrange(4) for _ in range(width)] for _ in range(height)]


class ReferenceScoringTests(unittest.TestCase):
    def test_independent_scorers_agree_on_generated_cases(self) -> None:
        rng = random.Random(20260806)
        for _ in range(500):
            answers = {
                f"task-{task_index}": [
                    random_grid(rng) for _ in range(rng.randint(1, 3))
                ]
                for task_index in range(rng.randint(1, 6))
            }
            predictions = {}
            for task_id, outputs in answers.items():
                if rng.random() < 0.25:
                    continue
                predictions[task_id] = []
                for expected in outputs:
                    attempts = {}
                    for attempt in range(1, rng.randint(2, 5)):
                        attempts[f"attempt_{attempt}"] = (
                            expected if rng.random() < 0.35 else random_grid(rng)
                        )
                    predictions[task_id].append(attempts)

            for top_k in (1, 2, 3):
                production = score_predictions(predictions, answers, top_k=top_k)
                reference = reference_exact_score(predictions, answers, top_k=top_k)
                self.assertEqual(production.tasks_total, reference.tasks_total)
                self.assertEqual(production.tasks_predicted, reference.tasks_predicted)
                self.assertEqual(production.tasks_exact, reference.tasks_exact)
                self.assertEqual(production.outputs_total, reference.outputs_total)
                self.assertEqual(production.outputs_exact, reference.outputs_exact)


if __name__ == "__main__":
    unittest.main()
