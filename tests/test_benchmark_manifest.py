import unittest
from pathlib import Path
import json


from arc_agi_eval.benchmark_manifest import build_benchmark_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark_sources.json"
SAVED = (
    ROOT
    / "reports"
    / "e0-benchmark-data"
    / "20260806-public-snapshot-integrity-v1"
    / "manifest.json"
)


class BenchmarkManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_benchmark_manifest(ROOT, CONFIG)

    def test_all_vendored_tasks_are_schema_valid_and_counted(self) -> None:
        self.assertEqual(self.manifest["summary"]["task_file_count"], 1920)
        self.assertEqual(
            self.manifest["benchmarks"]["arc_agi_1"]["splits"]["training"]["task_count"],
            400,
        )
        self.assertEqual(
            self.manifest["benchmarks"]["arc_agi_1"]["splits"]["evaluation"]["task_count"],
            400,
        )
        self.assertEqual(
            self.manifest["benchmarks"]["arc_agi_2"]["splits"]["training"]["task_count"],
            1000,
        )
        self.assertEqual(
            self.manifest["benchmarks"]["arc_agi_2"]["splits"]["evaluation"]["task_count"],
            120,
        )

    def test_public_evaluation_denominators_match_scorer_contract(self) -> None:
        arc1 = self.manifest["benchmarks"]["arc_agi_1"]["splits"]["evaluation"]
        arc2 = self.manifest["benchmarks"]["arc_agi_2"]["splits"]["evaluation"]
        self.assertEqual(arc1["test_output_count"], 419)
        self.assertEqual(arc2["test_output_count"], 167)

    def test_manifest_is_deterministic(self) -> None:
        rebuilt = build_benchmark_manifest(ROOT, CONFIG)
        self.assertEqual(
            rebuilt["manifest_payload_sha256"],
            self.manifest["manifest_payload_sha256"],
        )

    def test_saved_manifest_matches_live_rebuild(self) -> None:
        saved = json.loads(SAVED.read_text(encoding="utf-8"))
        self.assertEqual(saved, self.manifest)


if __name__ == "__main__":
    unittest.main()
