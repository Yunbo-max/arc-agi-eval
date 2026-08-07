import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "baselines.json"


class BaselineManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_ids_and_summary_counts_are_consistent(self) -> None:
        entries = self.manifest["entries"]
        ids = [entry["id"] for entry in entries]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), self.manifest["summary"]["entry_count"])
        for level, summary_key in (
            ("smoke", "smoke_passed_count"),
            ("benchmark", "benchmark_passed_count"),
            ("full", "full_reproduction_passed_count"),
        ):
            observed = sum(
                entry["reproduction"][level]["status"] == "passed"
                for entry in entries
            )
            self.assertEqual(observed, self.manifest["summary"][summary_key])

    def test_every_passed_status_has_matching_run_evidence(self) -> None:
        for entry in self.manifest["entries"]:
            for level, state in entry["reproduction"].items():
                if state["status"] != "passed":
                    continue
                with self.subTest(baseline=entry["id"], level=level):
                    self.assertIn("evidence", state)
                    evidence = ROOT / state["evidence"]
                    self.assertTrue(evidence.is_file(), evidence)
                    run = json.loads(evidence.read_text(encoding="utf-8"))
                    self.assertEqual(run["status"], "passed")
                    self.assertIn(entry["id"], evidence.parts)

    def test_status_and_feasibility_values_use_declared_enums(self) -> None:
        statuses = set(self.manifest["enums"]["execution_status"])
        feasibility = set(self.manifest["enums"]["feasibility"])
        for entry in self.manifest["entries"]:
            for state in entry["reproduction"].values():
                self.assertIn(state["status"], statuses)
                self.assertIn(state["feasibility"], feasibility)


if __name__ == "__main__":
    unittest.main()
