import random
import unittest

from arc_agi_eval.isoarc import (
    D4,
    INVERSE_D4,
    color_transform,
    inverse_color_mapping,
    restore_test_order,
    transform_predictions,
    transform_task,
)
from arc_agi_eval.validation import validate_task


class IsoArcTests(unittest.TestCase):
    def test_all_d4_round_trips_on_random_rectangles_and_predictions(self) -> None:
        rng = random.Random(20260806)
        for _ in range(200):
            height = rng.randint(1, 30)
            width = rng.randint(1, 30)
            grid = [[rng.randrange(10) for _ in range(width)] for _ in range(height)]
            predictions = [{"attempt_1": grid, "attempt_2": [row[::-1] for row in grid]}]
            for name, transform in D4.items():
                inverse = D4[INVERSE_D4[name]]
                self.assertEqual(inverse(transform(grid)), grid)
                self.assertEqual(
                    transform_predictions(transform_predictions(predictions, transform), inverse),
                    predictions,
                )

    def test_color_bijection_round_trip(self) -> None:
        mapping = {0: 0, 1: 7, 7: 1, 2: 9, 9: 2}
        grid = [[0, 1, 2], [7, 8, 9]]
        transformed = color_transform(mapping)(grid)
        self.assertEqual(color_transform(inverse_color_mapping(mapping))(transformed), grid)
        with self.assertRaisesRegex(ValueError, "injective"):
            color_transform({1: 2, 3: 2})
        with self.assertRaisesRegex(ValueError, "closed color set"):
            color_transform({1: 2})
        with self.assertRaisesRegex(ValueError, "closed color set"):
            color_transform({0: 0, 1: 2, 2: 3})

    def test_task_transform_is_valid_and_permutations_restore(self) -> None:
        task = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 1]]},
                {"input": [[3], [4]], "output": [[4], [3]]},
                {"input": [[5]], "output": [[5]]},
            ],
            "test": [
                {"input": [[6, 7]], "output": [[7, 6]]},
                {"input": [[8]], "output": [[8]]},
            ],
        }
        transformed = transform_task(
            task, D4["rotate_90"], train_order=[2, 0, 1], test_order=[1, 0]
        )
        validate_task(transformed)
        self.assertEqual(restore_test_order(["second", "first"], [1, 0]), ["first", "second"])

    def test_unlabeled_transformed_task_remains_valid(self) -> None:
        task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}
        transformed = transform_task(
            task,
            color_transform({0: 0, 1: 4, 4: 1, 2: 5, 5: 2, 3: 6, 6: 3}),
        )
        validate_task(transformed, require_test_outputs=False)
        self.assertNotIn("output", transformed["test"][0])


if __name__ == "__main__":
    unittest.main()
