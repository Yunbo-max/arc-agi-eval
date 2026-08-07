import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.development_runtime import (
    build_development_runtime,
    canonical_sha256,
)
from arc_agi_eval.validation import load_task


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "development_partition_v1.json"
REPORT = (
    ROOT
    / "reports"
    / "e0-development-split"
    / "20260806-frozen-known-overlap-excluded-dev-audit-v1"
)


class DevelopmentRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build = build_development_runtime(ROOT, CONFIG)
        cls.manifest = cls.build.manifest

    def test_partition_counts_and_claim_boundary_are_frozen(self) -> None:
        self.assertEqual(self.manifest["freeze_status"], "frozen")
        self.assertEqual(
            self.manifest["claim_boundary"], "known-overlap-excluded-only"
        )
        self.assertEqual(self.manifest["partition"]["general_cluster_count"], 1008)
        self.assertEqual(
            self.manifest["partition"]["known_overlap_excluded_cluster_count"],
            376,
        )
        self.assertEqual(self.manifest["partition"]["remaining_cluster_count"], 632)
        self.assertTrue(self.manifest["partition"]["no_cluster_reallocation"])

    def test_dev_audit_uses_one_unique_representative_per_cluster(self) -> None:
        audit = self.manifest["dev_audit"]
        self.assertEqual(audit["cluster_count"], 94)
        self.assertEqual(audit["representative_task_count"], 94)
        self.assertEqual(audit["source_record_count"], 159)
        self.assertEqual(audit["test_output_denominator"], 97)
        reps = self.manifest["representatives"]
        self.assertEqual(len({item["cluster_id"] for item in reps}), 94)
        self.assertEqual(len({item["task_id"] for item in reps}), 94)
        self.assertEqual(canonical_sha256(reps), audit["representative_order_sha256"])

    def test_inference_tree_is_label_free_and_separate_from_solutions(self) -> None:
        self.assertEqual(len(self.build.inference_files), 95)
        self.assertEqual(len(self.build.solution_files), 94)
        for path, payload in self.build.inference_files.items():
            self.assertTrue(path.startswith("inference/dev-audit/"))
            if path.endswith(".json"):
                value = json.loads(payload)
                self.assertTrue(all("output" not in pair for pair in value["test"]))
        for path in self.build.solution_files:
            self.assertTrue(path.startswith("scoring/dev-audit/"))
            self.assertNotIn(path, self.build.inference_files)
        self.assertEqual(
            self.manifest["checks"]["inference_visible_test_output_fields"], 0
        )
        self.assertTrue(
            self.manifest["checks"]["scoring_data_outside_inference_tree"]
        )

    def test_every_source_and_emitted_file_hash_is_bound(self) -> None:
        for record in self.manifest["representatives"]:
            source = ROOT / record["source_path"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                record["source_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(
                    self.build.inference_files[record["challenge_path"]]
                ).hexdigest(),
                record["challenge_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(
                    self.build.solution_files[record["solution_path"]]
                ).hexdigest(),
                record["solution_sha256"],
            )

    def test_hidden_label_mutation_and_scorer_sensitivity_pass(self) -> None:
        checks = self.manifest["checks"]
        self.assertEqual(checks["hidden_label_mutation_task_count"], 94)
        self.assertTrue(checks["hidden_label_mutation_stable"])
        self.assertTrue(checks["scorer_label_sensitivity_passed"])
        self.assertEqual(checks["scorer_original_outputs_exact"], 97)
        self.assertEqual(checks["scorer_mutated_outputs_exact"], 0)

    def test_rebuild_and_saved_manifest_are_identical(self) -> None:
        rebuilt = build_development_runtime(ROOT, CONFIG)
        self.assertEqual(rebuilt, self.build)
        if (REPORT / "manifest.json").is_file():
            saved = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, self.manifest)

    def test_claim_boundary_cannot_be_broadened(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["claim_boundary"] = "fully-clean"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=ROOT, encoding="utf-8"
        ) as handle:
            json.dump(config, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "broadened"):
                build_development_runtime(ROOT, Path(handle.name))


if __name__ == "__main__":
    unittest.main()
