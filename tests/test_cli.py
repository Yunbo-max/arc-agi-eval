import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from arc_agi_eval.cli import main


FIXTURES = Path(__file__).parent / "fixtures"


class CliTests(unittest.TestCase):
    def test_validate_and_list_commands(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["validate", str(FIXTURES / "data")])
        self.assertEqual(status, 0)
        self.assertIn("validated 1 task(s)", output.getvalue())

        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["list", str(FIXTURES / "data")])
        self.assertEqual(status, 0)
        self.assertIn("evaluation\t1", output.getvalue())

    def test_baseline_command_writes_requested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "predictions.json"
            metadata = Path(temporary) / "metadata.json"
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "baseline",
                        str(FIXTURES / "data" / "evaluation"),
                        "--output",
                        str(predictions),
                        "--metadata",
                        str(metadata),
                        "--score",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue(predictions.is_file())
            run = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(run["tasks_total"], 1)
            self.assertEqual(run["score"]["outputs_exact"], 1)
            self.assertIn("primary_output_exact_pass@2", output.getvalue())
            self.assertEqual(
                run["score"]["metric_contract"]["primary"]["name"],
                "output_exact_pass_at_k",
            )

    def test_score_command_prints_primary_before_secondary(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "score",
                    str(FIXTURES / "predictions.json"),
                    str(FIXTURES / "data" / "evaluation"),
                ]
            )
        self.assertEqual(status, 0)
        lines = output.getvalue().splitlines()
        primary_index = next(
            index for index, line in enumerate(lines) if line.startswith("primary_")
        )
        secondary_index = next(
            index for index, line in enumerate(lines) if line.startswith("secondary_")
        )
        self.assertLess(primary_index, secondary_index)


if __name__ == "__main__":
    unittest.main()
