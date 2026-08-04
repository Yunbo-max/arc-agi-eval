import copy
import unittest
from pathlib import Path

from arc_agi_eval.validation import TaskValidationError, load_task, validate_task


FIXTURES = Path(__file__).parent / "fixtures"


class ValidationTests(unittest.TestCase):
    def test_valid_task_loads(self) -> None:
        task = load_task(FIXTURES / "data" / "evaluation" / "multi.json")
        self.assertEqual(len(task["test"]), 2)

    def test_ragged_grid_is_rejected(self) -> None:
        task = load_task(FIXTURES / "data" / "evaluation" / "multi.json")
        invalid = copy.deepcopy(task)
        invalid["train"][0]["input"] = [[0, 1], [0]]
        with self.assertRaisesRegex(TaskValidationError, "does not match width"):
            validate_task(invalid)

    def test_boolean_color_is_rejected(self) -> None:
        task = load_task(FIXTURES / "data" / "evaluation" / "multi.json")
        invalid = copy.deepcopy(task)
        invalid["test"][0]["output"][0][0] = True
        with self.assertRaisesRegex(TaskValidationError, "integer from 0 to 9"):
            validate_task(invalid)

    def test_optional_name_metadata_is_validated(self) -> None:
        task = load_task(FIXTURES / "data" / "evaluation" / "multi.json")
        task["name"] = "multi"
        self.assertEqual(validate_task(task)["name"], "multi")
        task["name"] = 1
        with self.assertRaisesRegex(TaskValidationError, "nonempty string"):
            validate_task(task)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(TaskValidationError, "duplicate object key"):
            load_task(FIXTURES / "duplicate.json")


if __name__ == "__main__":
    unittest.main()
