import unittest
from pathlib import Path

from arc_agi_eval.dataset import enumerate_dataset, group_by_split


FIXTURES = Path(__file__).parent / "fixtures"


class DatasetTests(unittest.TestCase):
    def test_enumerates_splits_and_task_ids(self) -> None:
        refs = enumerate_dataset(FIXTURES / "data")
        grouped = group_by_split(refs)
        self.assertEqual(list(grouped), ["evaluation"])
        self.assertEqual([ref.task_id for ref in grouped["evaluation"]], ["multi"])


if __name__ == "__main__":
    unittest.main()
