import copy
import json
import tempfile
import unittest
from pathlib import Path

from arc_agi_eval.baseline import (
    GEOMETRIC_TRANSFORMS,
    constant_dominant_color,
    copy_input,
    generate_predictions,
    learn_color_mapping,
    rank_candidates,
    run_baseline,
    solve_color_mapping,
    solve_geometric,
)


FIXTURES = Path(__file__).parent / "fixtures"
TASK_DIR = FIXTURES / "data" / "evaluation"


class BasicBaselineTests(unittest.TestCase):
    def test_copy_input_returns_an_independent_grid(self) -> None:
        source = [[1, 2], [3, 4]]
        result = copy_input(source)
        self.assertEqual(result, source)
        self.assertIsNot(result, source)
        self.assertIsNot(result[0], source[0])

    def test_dominant_color_uses_lowest_color_to_break_ties(self) -> None:
        self.assertEqual(
            constant_dominant_color([[3, 1], [1, 3]]),
            [[1, 1], [1, 1]],
        )


class GeometricSolverTests(unittest.TestCase):
    def test_all_transforms_support_rectangular_grids(self) -> None:
        source = [[1, 2, 3], [4, 5, 6]]
        expected = {
            "identity": [[1, 2, 3], [4, 5, 6]],
            "rotate_90": [[4, 1], [5, 2], [6, 3]],
            "rotate_180": [[6, 5, 4], [3, 2, 1]],
            "rotate_270": [[3, 6], [2, 5], [1, 4]],
            "flip_horizontal": [[3, 2, 1], [6, 5, 4]],
            "flip_vertical": [[4, 5, 6], [1, 2, 3]],
            "transpose": [[1, 4], [2, 5], [3, 6]],
            "anti_transpose": [[6, 3], [5, 2], [4, 1]],
        }
        self.assertEqual(
            {name: transform(source) for name, transform in GEOMETRIC_TRANSFORMS},
            expected,
        )

    def test_solver_returns_only_train_validated_transform(self) -> None:
        train = [
            {
                "input": [[1, 2, 3], [4, 5, 6]],
                "output": [[4, 1], [5, 2], [6, 3]],
            }
        ]
        candidates = solve_geometric(train, [[7, 8], [9, 0], [1, 2]])
        self.assertEqual([candidate.solver for candidate in candidates], ["geometric:rotate_90"])
        self.assertEqual(candidates[0].grid, [[1, 9, 7], [2, 0, 8]])


class ColorMappingSolverTests(unittest.TestCase):
    def test_learns_mapping_and_leaves_unseen_colors_unchanged(self) -> None:
        train = [{"input": [[0, 1]], "output": [[2, 3]]}]
        self.assertEqual(learn_color_mapping(train), {0: 2, 1: 3})
        candidate = solve_color_mapping(train, [[1, 0, 9]])
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.grid, [[3, 2, 9]])

    def test_rejects_inconsistent_and_shape_changing_examples(self) -> None:
        inconsistent = [{"input": [[1, 1]], "output": [[2, 3]]}]
        shape_changing = [{"input": [[1, 2]], "output": [[1], [2]]}]
        self.assertIsNone(learn_color_mapping(inconsistent))
        self.assertIsNone(learn_color_mapping(shape_changing))


class CandidateRankingTests(unittest.TestCase):
    def test_validated_solver_ranks_before_deduplicated_fallbacks(self) -> None:
        train = [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 2], [2, 1]],
            }
        ]
        first = rank_candidates(train, [[0, 1, 1]], top_k=2)
        second = rank_candidates(train, [[0, 1, 1]], top_k=2)
        self.assertEqual(first, second)
        self.assertEqual(
            [candidate.solver for candidate in first],
            ["color_mapping", "copy_input"],
        )
        self.assertEqual(first[0].grid, [[1, 2, 2]])

    def test_uniform_input_still_gets_two_distinct_attempts(self) -> None:
        train = [{"input": [[0]], "output": [[1]]}]
        candidates = rank_candidates(train, [[0, 0]], top_k=2)
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[0].grid, candidates[1].grid)

    def test_test_labels_do_not_affect_generation(self) -> None:
        original = json.loads((TASK_DIR / "multi.json").read_text(encoding="utf-8"))
        changed = copy.deepcopy(original)
        for pair in changed["test"]:
            pair["output"] = [[9]]
        from_original = [
            [candidate.grid for candidate in rank_candidates(original["train"], pair["input"])]
            for pair in original["test"]
        ]
        from_changed = [
            [candidate.grid for candidate in rank_candidates(changed["train"], pair["input"])]
            for pair in changed["test"]
        ]
        self.assertEqual(from_original, from_changed)


class BaselineRunnerTests(unittest.TestCase):
    def test_runner_writes_predictions_metadata_and_post_hoc_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction_path = Path(temporary) / "predictions.json"
            metadata_path = Path(temporary) / "run.json"
            metadata = run_baseline(
                TASK_DIR, prediction_path, metadata_path, score=True
            )
            predictions, _ = generate_predictions(TASK_DIR)
            self.assertEqual(
                json.loads(prediction_path.read_text(encoding="utf-8")), predictions
            )
            self.assertEqual(metadata["tasks_total"], 1)
            self.assertEqual(metadata["test_outputs_total"], 2)
            self.assertEqual(metadata["attempts_total"], 4)
            self.assertEqual(metadata["score"]["outputs_exact"], 1)
            self.assertTrue(metadata_path.is_file())


if __name__ == "__main__":
    unittest.main()
