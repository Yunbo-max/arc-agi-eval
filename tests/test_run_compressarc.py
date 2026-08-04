import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_compressarc import (
    challenge_without_test_outputs,
    parse_args,
    solutions_to_predictions,
)


class CompressArcRunnerTests(unittest.TestCase):
    def test_solution_conversion_produces_standard_top_two_json(self) -> None:
        predictions = solutions_to_predictions(
            "007bbfb7",
            (((1, 2), (3, 4)),),
            (((4, 3), (2, 1)),),
        )
        self.assertEqual(
            predictions,
            {
                "007bbfb7": [
                    {
                        "attempt_1": [[1, 2], [3, 4]],
                        "attempt_2": [[4, 3], [2, 1]],
                    }
                ]
            },
        )
        json.dumps(predictions)

    def test_solution_conversion_rejects_mismatched_output_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "different output counts"):
            solutions_to_predictions("007bbfb7", (((1,),),), ())

    def test_challenge_removes_test_outputs(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        challenge = challenge_without_test_outputs(task)
        self.assertEqual(challenge["test"], [{"input": [[3]]}])
        self.assertEqual(challenge["train"], task["train"])
        self.assertIsNot(challenge["train"][0]["output"], task["train"][0]["output"])

    def test_parse_args_accepts_valid_task_and_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "training"
            task_dir.mkdir()
            (task_dir / "007bbfb7.json").write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            args = parse_args(
                [
                    "--split",
                    "training",
                    "--task-id",
                    "007bbfb7",
                    "--steps",
                    "2",
                    "--task-dir",
                    str(task_dir),
                    "--run-dir",
                    str(run_dir),
                    "--log-interval",
                    "1",
                ]
            )
        self.assertEqual(args.steps, 2)
        self.assertEqual(args.log_interval, 1)

    def test_parse_args_rejects_nonpositive_counts(self) -> None:
        for option in ("--steps", "--log-interval"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--split",
                        "training",
                        "--task-id",
                        "007bbfb7",
                        "--steps",
                        "1" if option == "--log-interval" else "0",
                        "--task-dir",
                        "/does/not/matter",
                        "--run-dir",
                        "/does/not/matter",
                        *(["--log-interval", "0"] if option == "--log-interval" else []),
                    ]
                )

    def test_parse_args_rejects_invalid_task_id_before_touching_paths(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--split",
                    "training",
                    "--task-id",
                    "not-a-task",
                    "--steps",
                    "1",
                    "--task-dir",
                    "/does/not/matter",
                    "--run-dir",
                    "/does/not/matter",
                ]
            )


if __name__ == "__main__":
    unittest.main()
