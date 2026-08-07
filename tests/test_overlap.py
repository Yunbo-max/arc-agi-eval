import unittest
from pathlib import Path

from arc_agi_eval.overlap import (
    analyze_split_overlap,
    load_task_split,
    normalized_task,
)


ROOT = Path(__file__).resolve().parents[1]


class OverlapTests(unittest.TestCase):
    def test_name_and_train_order_do_not_change_semantic_task(self) -> None:
        first = {
            "name": "metadata-a",
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[3]], "output": [[4]]},
            ],
            "test": [{"input": [[5]], "output": [[6]]}],
        }
        second = {
            "name": "metadata-b",
            "train": list(reversed(first["train"])),
            "test": first["test"],
        }
        self.assertEqual(normalized_task(first), normalized_task(second))

    def test_challenge_fingerprint_ignores_only_test_outputs(self) -> None:
        first = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        second = {
            "train": first["train"],
            "test": [{"input": [[3]], "output": [[9]]}],
        }
        self.assertNotEqual(normalized_task(first), normalized_task(second))
        self.assertEqual(
            normalized_task(first, include_test_outputs=False),
            normalized_task(second, include_test_outputs=False),
        )

    def test_test_example_order_does_not_change_semantic_task(self) -> None:
        first = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [
                {"input": [[3]], "output": [[4]]},
                {"input": [[5]], "output": [[6]]},
            ],
        }
        second = {"train": first["train"], "test": list(reversed(first["test"]))}
        self.assertEqual(normalized_task(first), normalized_task(second))

    def test_canonical_arc1_eval_arc2_train_overlap_counts(self) -> None:
        overlap = analyze_split_overlap(
            load_task_split(
                ROOT / "third_party" / "arc-agi-1" / "data" / "evaluation"
            ),
            load_task_split(
                ROOT / "third_party" / "arc-agi-2" / "data" / "training"
            ),
        )
        self.assertEqual(overlap["left_task_count"], 400)
        self.assertEqual(overlap["right_task_count"], 1000)
        self.assertEqual(overlap["id_overlap_count"], 376)
        self.assertEqual(overlap["semantic_labeled_exact_count"], 375)
        self.assertEqual(overlap["test_io_exact_count"], 376)
        self.assertEqual(overlap["ordered_test_io_exact_count"], 357)


if __name__ == "__main__":
    unittest.main()
