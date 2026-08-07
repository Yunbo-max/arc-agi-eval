import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.development_split import (
    IsomorphismWitness,
    SOURCE_NAMES,
    build_development_manifest,
    find_verified_isomorphism,
    fingerprint,
    validate_development_manifest,
    verify_isomorphism_witness,
)
from arc_agi_eval.isoarc import D4, color_transform, transform_task
from arc_agi_eval.validation import load_task


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "reports"
    / "e0-development-split"
    / "20260806-training-only-deterministic-split-retry1"
)


def _base_task() -> dict[str, object]:
    return {
        "train": [
            {
                "input": [[1, 2, 1], [2, 1, 2]],
                "output": [[2, 1, 2], [1, 2, 1]],
            },
            {"input": [[3]], "output": [[1]]},
        ],
        "test": [{"input": [[1, 3]], "output": [[2, 3]]}],
    }


def _transformed_base_task() -> dict[str, object]:
    mapping = color_transform({1: 7, 7: 1, 2: 4, 4: 2, 3: 9, 9: 3})
    return transform_task(
        _base_task(),
        lambda grid: mapping(D4["rotate_90"](grid)),
        train_order=[1, 0],
        test_order=[0],
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifiedIsomorphismTests(unittest.TestCase):
    def test_global_d4_color_and_example_permutation_has_replayable_witness(
        self,
    ) -> None:
        left = _base_task()
        right = _transformed_base_task()
        search = find_verified_isomorphism(left, right)
        self.assertEqual(search.status, "matched")
        self.assertIsNotNone(search.witness)
        self.assertTrue(verify_isomorphism_witness(left, right, search.witness))
        self.assertEqual(search.witness.d4_transform, "rotate_90")
        self.assertEqual(search.witness.train_permutation, (1, 0))

        tampered = IsomorphismWitness(
            d4_transform=search.witness.d4_transform,
            color_mapping={**search.witness.color_mapping, 1: 8},
            train_permutation=search.witness.train_permutation,
            test_permutation=search.witness.test_permutation,
        )
        self.assertFalse(verify_isomorphism_witness(left, right, tampered))

    def test_per_example_color_maps_cannot_replace_one_global_bijection(self) -> None:
        left = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[1, 1]], "output": [[2, 2]]},
            ],
            "test": [{"input": [[1, 1, 1]], "output": [[2, 2, 2]]}],
        }
        right = {
            "train": [
                {"input": [[3]], "output": [[4]]},
                {"input": [[5, 5]], "output": [[4, 4]]},
            ],
            "test": [{"input": [[3, 3, 3]], "output": [[4, 4, 4]]}],
        }
        search = find_verified_isomorphism(left, right)
        self.assertEqual(search.status, "not-isomorphic")
        self.assertIsNone(search.witness)

    def test_per_example_d4_transforms_cannot_replace_one_global_d4(self) -> None:
        horizontal = {"input": [[1, 2]], "output": [[2, 1]]}
        vertical = {"input": [[3], [4]], "output": [[4], [3]]}
        left = {
            "train": [horizontal, vertical],
            "test": [{"input": [[5]], "output": [[6]]}],
        }
        right = {
            "train": [
                {
                    "input": D4["rotate_90"](horizontal["input"]),
                    "output": D4["rotate_90"](horizontal["output"]),
                },
                vertical,
            ],
            "test": left["test"],
        }
        search = find_verified_isomorphism(left, right)
        self.assertEqual(search.status, "not-isomorphic")

    def test_state_cap_is_conservative_instead_of_merging(self) -> None:
        search = find_verified_isomorphism(
            _base_task(), _transformed_base_task(), max_search_states=1
        )
        self.assertEqual(search.status, "inconclusive")
        self.assertIsNone(search.witness)


class DevelopmentManifestTests(unittest.TestCase):
    def _write_task(self, path: Path, task: dict[str, object]) -> None:
        path.write_text(json.dumps(task), encoding="utf-8")

    def test_toy_manifest_is_deterministic_and_clusters_never_cross_splits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arc1 = root / "arc1" / "training"
            arc2 = root / "arc2" / "training"
            arc1.mkdir(parents=True)
            arc2.mkdir(parents=True)
            self._write_task(arc1 / "base.json", _base_task())
            self._write_task(
                arc1 / "one.json",
                {
                    "train": [{"input": [[1]], "output": [[1]]}],
                    "test": [{"input": [[2]], "output": [[2]]}],
                },
            )
            self._write_task(arc2 / "renamed.json", _transformed_base_task())
            self._write_task(
                arc2 / "two.json",
                {
                    "train": [{"input": [[3]], "output": [[4]]}],
                    "test": [{"input": [[5]], "output": [[6]]}],
                },
            )
            sources = {SOURCE_NAMES[0]: arc1, SOURCE_NAMES[1]: arc2}
            first = build_development_manifest(
                sources,
                expected_counts={SOURCE_NAMES[0]: 2, SOURCE_NAMES[1]: 2},
            )
            second = build_development_manifest(
                sources,
                expected_counts={SOURCE_NAMES[0]: 2, SOURCE_NAMES[1]: 2},
            )
            self.assertEqual(first, second)
            validate_development_manifest(first)
            self.assertEqual(first["summary"]["source_record_count"], 4)
            self.assertEqual(first["summary"]["deduplicated_cluster_count"], 3)
            self.assertEqual(
                first["summary"]["d4_color_nonexact_verified_edge_count"], 1
            )

            seen: set[str] = set()
            for split in first["splits"].values():
                cluster_ids = set(split["cluster_ids"])
                self.assertTrue(seen.isdisjoint(cluster_ids))
                seen.update(cluster_ids)
            self.assertEqual(seen, {c["cluster_id"] for c in first["clusters"]})

    def test_non_training_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "arc1" / "evaluation"
            training = root / "arc2" / "training"
            evaluation.mkdir(parents=True)
            training.mkdir(parents=True)
            self._write_task(evaluation / "a.json", _base_task())
            self._write_task(training / "b.json", _base_task())
            with self.assertRaisesRegex(ValueError, "named training"):
                build_development_manifest(
                    {SOURCE_NAMES[0]: evaluation, SOURCE_NAMES[1]: training},
                    expected_counts={SOURCE_NAMES[0]: 1, SOURCE_NAMES[1]: 1},
                )

    def test_public_seed_and_split_weights_are_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "public seed is fixed"):
            build_development_manifest({}, public_seed=7)
        with self.assertRaisesRegex(ValueError, "split weights are fixed"):
            build_development_manifest(
                {},
                split_weights=(
                    ("dev-build", 60),
                    ("dev-select", 20),
                    ("dev-audit", 20),
                ),
            )


class CanonicalDevelopmentManifestEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = EVIDENCE / "manifest.json"
        cls.run_path = EVIDENCE / "run.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.run_record = json.loads(cls.run_path.read_text(encoding="utf-8"))

    def test_locked_counts_digests_and_split_quotas(self) -> None:
        validate_development_manifest(self.manifest)
        self.assertEqual(self.manifest["protocol_status"], "draft-not-frozen")
        self.assertEqual(self.manifest["summary"]["source_record_count"], 1400)
        self.assertEqual(
            self.manifest["summary"]["deduplicated_cluster_count"], 1008
        )
        self.assertEqual(
            self.manifest["summary"]["exact_verified_edge_count"], 392
        )
        self.assertEqual(
            self.manifest["summary"]["d4_color_nonexact_verified_edge_count"], 0
        )
        self.assertEqual(self.manifest["summary"]["inconclusive_pair_count"], 0)
        self.assertEqual(
            self.manifest["summary"]["known_arc1_eval_overlap_task_id_count"],
            376,
        )
        self.assertEqual(
            self.manifest["summary"]["known_arc1_eval_overlap_cluster_count"],
            376,
        )
        self.assertEqual(
            {
                name: split["cluster_count"]
                for name, split in self.manifest["splits"].items()
            },
            {"dev-build": 706, "dev-select": 151, "dev-audit": 151},
        )
        self.assertEqual(
            self.run_record["manifest"]["file_sha256"],
            _sha256_file(self.manifest_path),
        )

    def test_every_source_sha_and_verified_edge_replays_from_training_only(
        self,
    ) -> None:
        sources = {
            name: Path(metadata["resolved_path"])
            for name, metadata in self.run_record["sources"].items()
        }
        self.assertTrue(all(path.name == "training" for path in sources.values()))
        tasks: dict[str, dict[str, object]] = {}
        tree_entries: dict[str, list[dict[str, str]]] = {
            name: [] for name in sources
        }
        for record in self.manifest["task_records"]:
            task_path = sources[record["source"]] / record["relative_path"]
            observed_sha = _sha256_file(task_path)
            self.assertEqual(observed_sha, record["source_file_sha256"])
            tree_entries[record["source"]].append(
                {
                    "relative_path": record["relative_path"],
                    "file_sha256": observed_sha,
                }
            )
            tasks[record["record_id"]] = load_task(task_path)
        for name, entries in tree_entries.items():
            self.assertEqual(
                fingerprint(sorted(entries, key=lambda entry: entry["relative_path"])),
                self.manifest["sources"][name]["tree_sha256"],
            )

        for edge in self.manifest["verified_edges"]:
            encoded = edge["witness"]
            witness = IsomorphismWitness(
                d4_transform=encoded["d4_transform"],
                color_mapping={
                    int(source): target
                    for source, target in encoded["color_mapping"].items()
                },
                train_permutation=tuple(encoded["train_permutation"]),
                test_permutation=tuple(encoded["test_permutation"]),
            )
            self.assertTrue(
                verify_isomorphism_witness(
                    tasks[edge["left_record_id"]],
                    tasks[edge["right_record_id"]],
                    witness,
                )
            )

    def test_arc1_overlap_is_annotation_only_and_clean_view_remains_blocked(
        self,
    ) -> None:
        contamination = self.manifest["contamination"]
        self.assertEqual(
            contamination["status"], "annotated-contamination-aware-for-arc1"
        )
        self.assertFalse(contamination["assignment_influence"])
        self.assertEqual(contamination["overlap_task_id_count"], 376)
        self.assertEqual(contamination["flagged_arc2_training_record_count"], 376)
        self.assertEqual(contamination["flagged_cluster_count"], 376)
        self.assertEqual(contamination["flagged_cluster_member_record_count"], 377)
        self.assertEqual(
            {
                name: state["flagged_cluster_count"]
                for name, state in contamination["by_split"].items()
            },
            {"dev-build": 274, "dev-select": 45, "dev-audit": 57},
        )
        self.assertEqual(
            contamination["arc1_public_evaluation_claim"],
            "contamination-aware-only",
        )
        self.assertEqual(
            contamination["arc1_clean_view"]["status"],
            "required-not-materialized",
        )
        self.assertEqual(
            self.manifest["data_policy"]["evaluation_tasks_read"], 0
        )
        self.assertFalse(
            self.manifest["data_policy"]["overlap_ledger_used_for_assignment"]
        )


if __name__ == "__main__":
    unittest.main()
