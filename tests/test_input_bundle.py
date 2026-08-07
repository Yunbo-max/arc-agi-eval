import json
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.input_bundle import (
    build_code_inventory,
    build_public_task_orders,
    deterministic_task_order,
    verify_challenge_view,
    verify_declared_inputs,
)
from scripts.freeze_input_bundle import bundle_leaf_paths


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "input_freeze_v1.json"
REPORT = (
    ROOT / "reports" / "e0-freeze" / "20260806-input-bundle-v1-retry16"
)


class InputBundleTests(unittest.TestCase):
    def test_bundle_leaf_exclusions_match_only_exact_root_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "inputs" / "declared" / "reports" / "evidence"
            nested.mkdir(parents=True)
            for path in (
                root / "run.json",
                root / "content-manifest.json",
                root / "bundle-manifest.json",
                nested / "run.json",
                nested / "content-manifest.json",
            ):
                path.write_text("{}\n", encoding="utf-8")

            content_paths = {
                path.relative_to(root).as_posix()
                for path in bundle_leaf_paths(
                    root,
                    excluded_relative_paths={"run.json", "content-manifest.json"},
                )
            }
            self.assertEqual(
                content_paths,
                {
                    "bundle-manifest.json",
                    "inputs/declared/reports/evidence/run.json",
                    "inputs/declared/reports/evidence/content-manifest.json",
                },
            )

            run_file_paths = {
                path.relative_to(root).as_posix()
                for path in bundle_leaf_paths(
                    root, excluded_relative_paths={"run.json"}
                )
            }
            self.assertEqual(
                run_file_paths,
                {
                    "content-manifest.json",
                    "bundle-manifest.json",
                    "inputs/declared/reports/evidence/run.json",
                    "inputs/declared/reports/evidence/content-manifest.json",
                },
            )

    def test_bundle_leaf_inventory_refuses_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            (root / "linked.json").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "refuses symlink"):
                bundle_leaf_paths(root, excluded_relative_paths={"run.json"})

    def test_declared_inputs_are_hash_verified(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        inventory = verify_declared_inputs(ROOT, config["declared_input_files"])
        self.assertEqual(len(inventory), len(config["declared_input_files"]))
        self.assertEqual(len({item["path"] for item in inventory}), len(inventory))

    def test_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.json"
            path.write_text("{}\n", encoding="utf-8")
            record = {
                "role": "fixture",
                "path": "input.json",
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_declared_inputs(root, [record])

    def test_task_orders_are_stable_and_seed_separated(self) -> None:
        ids = ["a", "b", "c", "d"]
        first = deterministic_task_order(
            ids, domain="fixture-v1", benchmark="arc1", seed=1
        )
        self.assertEqual(
            first,
            deterministic_task_order(
                reversed(ids), domain="fixture-v1", benchmark="arc1", seed=1
            ),
        )
        second = deterministic_task_order(
            ids, domain="fixture-v1", benchmark="arc1", seed=2
        )
        self.assertEqual(set(first), set(second))
        self.assertNotEqual(first, second)

    def test_public_task_order_manifest_binds_every_order(self) -> None:
        views = {
            "arc1": {"records": [{"task_id": "a"}, {"task_id": "b"}]},
            "arc2": {"records": [{"task_id": "c"}, {"task_id": "d"}]},
        }
        manifest = build_public_task_orders(
            views, domain="fixture-v1", seeds=[1, 2, 3]
        )
        self.assertEqual(len(manifest["benchmarks"]["arc1"]["orders"]), 3)
        self.assertEqual(len(manifest["public_task_orders_sha256"]), 64)

    def test_real_public_views_are_label_free_and_complete(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        for benchmark, view in config["public_task_order"]["views"].items():
            audit = verify_challenge_view(
                ROOT / view["directory"],
                expected_task_count=view["expected_task_count"],
            )
            self.assertEqual(audit["test_output_fields_present"], 0, benchmark)

    def test_saved_bundle_has_zero_admitted_methods(self) -> None:
        if not (REPORT / "run.json").is_file():
            self.skipTest("input bundle has not been generated")
        run = json.loads((REPORT / "run.json").read_text(encoding="utf-8"))
        bundle = json.loads(
            (REPORT / "bundle-manifest.json").read_text(encoding="utf-8")
        )
        orders = json.loads(
            (REPORT / "task-orders.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run["status"], "passed")
        self.assertEqual(bundle["methods"]["entry_count"], 24)
        self.assertEqual(bundle["methods"]["strict_runtime_passed"], 2)
        self.assertEqual(bundle["methods"]["performance_eligible"], 0)
        self.assertEqual(bundle["methods"]["admitted_configuration_count"], 0)
        self.assertFalse(bundle["authorization"]["locked_public_solver_run"])
        self.assertEqual(orders["benchmarks"]["arc_agi_1"]["task_count"], 400)
        self.assertEqual(orders["benchmarks"]["arc_agi_2"]["task_count"], 120)

    def test_saved_bundle_code_snapshot_matches_current_tree(self) -> None:
        if not (REPORT / "code-inventory.json").is_file():
            self.skipTest("input bundle has not been generated")
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        observed = build_code_inventory(
            ROOT, config["code_inventory"]["include_globs"]
        )
        saved = json.loads(
            (REPORT / "code-inventory.json").read_text(encoding="utf-8")
        )
        frozen_sources = [
            {key: item[key] for key in ("path", "sha256", "bytes")}
            for item in saved["files"]
        ]
        self.assertEqual(observed, frozen_sources)

    def test_saved_bundle_manifests_cover_every_leaf_exactly(self) -> None:
        if not (REPORT / "run.json").is_file():
            self.skipTest("input bundle has not been generated")
        run = json.loads((REPORT / "run.json").read_text(encoding="utf-8"))
        content = json.loads(
            (REPORT / "content-manifest.json").read_text(encoding="utf-8")
        )

        expected_content_paths = {
            path.relative_to(REPORT).as_posix()
            for path in bundle_leaf_paths(
                REPORT,
                excluded_relative_paths={"run.json", "content-manifest.json"},
            )
        }
        observed_content_paths = {item["path"] for item in content["files"]}
        self.assertEqual(observed_content_paths, expected_content_paths)
        self.assertEqual(content["file_count"], len(expected_content_paths))

        report_prefix = REPORT.relative_to(ROOT)
        observed_run_bundle_paths = set()
        observed_run_external_paths = set()
        for item in run["files"]:
            path = Path(item["path"])
            try:
                observed_run_bundle_paths.add(
                    path.relative_to(report_prefix).as_posix()
                )
            except ValueError:
                observed_run_external_paths.add(path.as_posix())
        expected_run_bundle_paths = {
            path.relative_to(REPORT).as_posix()
            for path in bundle_leaf_paths(
                REPORT, excluded_relative_paths={"run.json"}
            )
        }
        self.assertEqual(observed_run_bundle_paths, expected_run_bundle_paths)
        self.assertEqual(
            observed_run_external_paths, {"schemas/protocol-v1-run.schema.json"}
        )
        nested_declared_runs = {
            path
            for path in expected_run_bundle_paths
            if path.startswith("inputs/declared/") and path.endswith("/run.json")
        }
        self.assertTrue(nested_declared_runs)
        self.assertTrue(nested_declared_runs <= observed_run_bundle_paths)


if __name__ == "__main__":
    unittest.main()
