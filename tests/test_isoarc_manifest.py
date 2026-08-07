import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.isoarc_manifest import (
    FEATURE_NAMES,
    audit_labeled_references,
    build_fixed64_design,
    canonical_sha256,
)
from arc_agi_eval.validation import load_task


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "isoarc_fixed64.json"
REPORT = ROOT / "reports" / "e0-isoarc" / "20260806-fixed64-design-v1"


class Fixed64IsoArcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build = build_fixed64_design(ROOT, CONFIG)
        cls.manifest = cls.build.manifest

    def test_selection_is_exact_unique_and_inventory_complete(self) -> None:
        summary = self.manifest["summary"]
        self.assertEqual(summary["source_task_count"], 520)
        self.assertEqual(summary["selected_base_task_count"], 64)
        self.assertEqual(
            summary["selected_by_benchmark"], {"arc_agi_1": 32, "arc_agi_2": 32}
        )
        self.assertEqual(summary["unique_task_id_count"], 64)
        self.assertEqual(len(self.manifest["candidate_inventory"]), 520)
        selected_ids = [item["task_id"] for item in self.manifest["assignments"]]
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertEqual(
            canonical_sha256(
                [
                    f"{item['benchmark_generation']}:{item['task_id']}"
                    for item in self.manifest["assignments"]
                ]
            ),
            self.manifest["digests"]["selection_sha256"],
        )

    def test_only_frozen_input_visible_features_are_used(self) -> None:
        self.assertEqual(self.manifest["feature_contract"]["features"], FEATURE_NAMES)
        self.assertEqual(
            self.manifest["feature_contract"]["cell_values_used"],
            "train inputs and test inputs only",
        )
        self.assertTrue(
            self.manifest["selection_contract"]["rank_excludes_task_content_hashes"]
        )
        for item in self.manifest["candidate_inventory"]:
            self.assertEqual(set(item["selection_features"]), set(FEATURE_NAMES))
            task = load_task(ROOT / item["path"], require_test_outputs=False)
            self.assertTrue(all("output" not in pair for pair in task["test"]))

    def test_color_maps_are_full_distinct_zero_fixed_bijections(self) -> None:
        signatures = []
        for record in self.manifest["color_maps"]:
            mapping = {int(key): value for key, value in record["mapping"].items()}
            self.assertEqual(set(mapping), set(range(10)))
            self.assertEqual(set(mapping.values()), set(range(10)))
            self.assertEqual(mapping[0], 0)
            self.assertNotEqual([mapping[i] for i in range(10)], list(range(10)))
            signatures.append(tuple(mapping[i] for i in range(10)))
        self.assertEqual(len(set(signatures)), 4)

    def test_variant_design_round_trips_and_balance(self) -> None:
        summary = self.manifest["summary"]
        self.assertEqual(summary["logical_variant_count"], 1135)
        self.assertEqual(len(self.build.variant_files), 1135)
        self.assertTrue(self.manifest["contract_checks"]["all_task_round_trips"])
        self.assertTrue(self.manifest["contract_checks"]["all_prediction_round_trips"])
        self.assertTrue(self.manifest["contract_checks"]["test_order_restoration"])
        for counts in self.manifest["balance_tables"][
            "latin_pair_counts_by_benchmark"
        ].values():
            self.assertEqual(len(counts), 16)
            self.assertEqual(set(counts.values()), {8})
        for item in self.manifest["variant_file_inventory"]:
            payload = self.build.variant_files[item["path"]]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            self.assertEqual(len(payload), item["bytes"])

    def test_rebuild_and_saved_manifest_are_identical(self) -> None:
        rebuilt = build_fixed64_design(ROOT, CONFIG)
        self.assertEqual(self.manifest, rebuilt.manifest)
        self.assertEqual(self.build.variant_files, rebuilt.variant_files)
        if (REPORT / "manifest.json").is_file():
            saved = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, self.manifest)

    def test_labeled_auditor_is_mutation_sensitive_and_selection_inert(self) -> None:
        audit = audit_labeled_references(ROOT, self.manifest)
        self.assertEqual(audit["hidden_label_mutation_task_count"], 520)
        self.assertTrue(audit["hidden_label_mutation_stable"])
        self.assertEqual(audit["labeled_task_round_trip_count"], 1135)
        self.assertEqual(audit["labeled_prediction_round_trip_count"], 1135)
        self.assertTrue(audit["scorer_label_sensitivity_passed"])
        self.assertFalse(audit["selection_or_assignment_modified_by_auditor"])

    def test_official_quota_cannot_be_weakened_in_config(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["selection"]["quota_by_benchmark"]["arc_agi_1"] = 31
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=ROOT, encoding="utf-8"
        ) as handle:
            json.dump(config, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "quotas"):
                build_fixed64_design(ROOT, Path(handle.name))


if __name__ == "__main__":
    unittest.main()
