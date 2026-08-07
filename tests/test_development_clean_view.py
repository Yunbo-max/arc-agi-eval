from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from arc_agi_eval.development_clean_view import (
    SPLIT_NAMES,
    build_arc1_clean_development_view,
    load_locked_source_manifest,
    validate_arc1_clean_development_view,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REFERENCE = (
    "reports/e0-development-split/"
    "20260806-training-only-deterministic-split-retry1/manifest.json"
)
SOURCE_PATH = ROOT / SOURCE_REFERENCE
SOURCE_SHA256 = "b21f60c7fc381977b3ac7ba2655270f037c5ddb6172cbdcf155b8e7c62ff7313"
EVIDENCE = (
    ROOT
    / "reports"
    / "e0-development-split"
    / "20260806-arc1-clean-overlap-excluded-draft-view"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Arc1CleanDevelopmentViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_locked_source_manifest(
            SOURCE_PATH, expected_sha256=SOURCE_SHA256
        )
        cls.manifest_path = EVIDENCE / "manifest.json"
        cls.run_path = EVIDENCE / "run.json"
        cls.view = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.run_record = json.loads(cls.run_path.read_text(encoding="utf-8"))

    def test_source_manifest_hash_is_immutable(self) -> None:
        self.assertEqual(_sha256_file(SOURCE_PATH), SOURCE_SHA256)
        self.assertEqual(
            self.view["source_manifest"]["file_sha256"], SOURCE_SHA256
        )
        self.assertEqual(
            self.run_record["source_manifest"]["file_sha256"], SOURCE_SHA256
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_locked_source_manifest(SOURCE_PATH, expected_sha256="0" * 64)

    def test_view_is_deterministic_and_all_digests_validate(self) -> None:
        rebuilt = build_arc1_clean_development_view(
            self.source,
            source_manifest_reference=SOURCE_REFERENCE,
            source_manifest_file_sha256=SOURCE_SHA256,
        )
        self.assertEqual(rebuilt, self.view)
        validate_arc1_clean_development_view(
            self.view,
            source_manifest=self.source,
            observed_source_manifest_file_sha256=SOURCE_SHA256,
        )
        self.assertEqual(
            self.run_record["manifest"]["file_sha256"],
            _sha256_file(self.manifest_path),
        )

    def test_expected_exclusion_and_remaining_counts(self) -> None:
        self.assertEqual(self.view["protocol_status"], "draft-not-frozen")
        self.assertEqual(self.view["summary"]["source_cluster_count"], 1008)
        self.assertEqual(self.view["summary"]["source_record_count"], 1400)
        self.assertEqual(self.view["summary"]["excluded_cluster_count"], 376)
        self.assertEqual(
            self.view["summary"]["excluded_source_record_count"], 377
        )
        self.assertEqual(self.view["summary"]["remaining_cluster_count"], 632)
        self.assertEqual(
            self.view["summary"]["remaining_source_record_count"], 1023
        )
        self.assertEqual(
            self.view["summary"]["remaining_source_record_counts"],
            {"arc-agi-1-training": 399, "arc-agi-2-training": 624},
        )
        self.assertEqual(
            {
                name: (
                    split["cluster_count"],
                    split["source_record_count"],
                )
                for name, split in self.view["splits"].items()
            },
            {
                "dev-build": (432, 703),
                "dev-select": (106, 161),
                "dev-audit": (94, 159),
            },
        )

    def test_no_flagged_cluster_or_record_remains(self) -> None:
        flagged_clusters = set(
            self.source["contamination"]["flagged_cluster_ids"]
        )
        flagged_records = set(
            self.source["contamination"]["flagged_cluster_member_record_ids"]
        )
        retained_clusters = {cluster["cluster_id"] for cluster in self.view["clusters"]}
        retained_records = {
            record["record_id"] for record in self.view["task_records"]
        }
        self.assertTrue(flagged_clusters.isdisjoint(retained_clusters))
        self.assertTrue(flagged_records.isdisjoint(retained_records))
        self.assertEqual(
            set(self.view["exclusion"]["excluded_cluster_ids"]), flagged_clusters
        )
        self.assertEqual(
            set(self.view["exclusion"]["excluded_task_record_ids"]),
            flagged_records,
        )

    def test_remaining_clusters_keep_source_split_without_crossing(self) -> None:
        flagged = set(self.view["exclusion"]["excluded_cluster_ids"])
        observed: set[str] = set()
        for split_name in SPLIT_NAMES:
            expected = [
                cluster_id
                for cluster_id in self.source["splits"][split_name]["cluster_ids"]
                if cluster_id not in flagged
            ]
            actual = self.view["splits"][split_name]["cluster_ids"]
            self.assertEqual(actual, expected)
            self.assertTrue(observed.isdisjoint(actual))
            observed.update(actual)
        self.assertEqual(
            observed, {cluster["cluster_id"] for cluster in self.view["clusters"]}
        )

    def test_task_ids_records_and_counts_are_complete_per_split(self) -> None:
        records = {
            record["record_id"]: record for record in self.view["task_records"]
        }
        for split_name, split in self.view["splits"].items():
            with self.subTest(split=split_name):
                self.assertEqual(
                    sum(split["source_record_counts"].values()),
                    split["source_record_count"],
                )
                listed_ids = {
                    source: sorted(
                        records[record_id]["task_id"]
                        for record_id in split["task_record_ids"]
                        if records[record_id]["source"] == source
                    )
                    for source in self.view["sources"]
                }
                self.assertEqual(split["task_ids_by_source"], listed_ids)

    def test_no_task_or_evaluation_file_was_read_and_no_reallocation_occurred(
        self,
    ) -> None:
        policy = self.view["data_policy"]
        self.assertEqual(policy["source_manifest_files_read"], 1)
        self.assertEqual(policy["training_task_files_read"], 0)
        self.assertEqual(policy["evaluation_task_files_read"], 0)
        self.assertEqual(policy["evaluation_label_files_read"], 0)
        self.assertEqual(policy["solution_files_read"], 0)
        self.assertFalse(policy["cluster_reallocation_performed"])

    def test_validator_rejects_reintroduced_flagged_cluster(self) -> None:
        tampered = deepcopy(self.view)
        flagged = tampered["exclusion"]["excluded_cluster_ids"][0]
        tampered["splits"]["dev-build"]["cluster_ids"].append(flagged)
        tampered["splits"]["dev-build"]["cluster_ids"].sort()
        with self.assertRaisesRegex(ValueError, "flagged cluster remains"):
            validate_arc1_clean_development_view(tampered)


if __name__ == "__main__":
    unittest.main()
