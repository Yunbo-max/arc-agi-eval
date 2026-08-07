import json
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.prior_exposure import (
    build_prior_exposure_manifest,
    classify_path,
    exclusion_for_path,
)


ROOT = Path(__file__).resolve().parents[1]


class PriorExposureTests(unittest.TestCase):
    def test_first_matching_rule_is_authoritative(self) -> None:
        rules = [
            {"glob": "reports/special/**", "tier": "historical_published", "reason": "specific"},
            {"glob": "reports/**", "tier": "development", "reason": "default"},
        ]
        self.assertEqual(
            classify_path("reports/special/run.json", rules)["tier"],
            "historical_published",
        )

    def test_current_workspace_inventory_is_complete_and_draft(self) -> None:
        manifest = build_prior_exposure_manifest(
            ROOT, ROOT / "configs" / "prior_exposure.json"
        )
        exclusions = json.loads(
            (ROOT / "configs" / "prior_exposure.json").read_text(encoding="utf-8")
        )["inventory_exclusions"]
        observed_runs = len(
            [
                path
                for path in (ROOT / "reports").glob("**/run.json")
                if exclusion_for_path(path.relative_to(ROOT).as_posix(), exclusions)
                is None
            ]
        )
        observed_results = len(
            [path for path in (ROOT / "results").glob("*") if path.is_file()]
        )
        self.assertEqual(manifest["protocol_status"], "draft-not-frozen")
        self.assertEqual(manifest["summary"]["run_record_count"], observed_runs)
        self.assertEqual(
            manifest["summary"]["result_artifact_count"], observed_results
        )
        self.assertEqual(manifest["summary"]["disclosure_count"], 8)
        self.assertEqual(
            manifest["summary"]["external_private_workspace_record_count"], 0
        )

    def test_control_plane_attestations_do_not_change_locked_inventory(self) -> None:
        manifest = build_prior_exposure_manifest(
            ROOT, ROOT / "configs" / "prior_exposure.json"
        )
        saved_run = json.loads(
            (
                ROOT
                / "reports"
                / "e0-prior-exposure"
                / "20260806-workspace-disclosure-draft-retry16"
                / "run.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["inventory_sha256"], saved_run["manifest"]["inventory_sha256"]
        )
        self.assertGreaterEqual(
            manifest["summary"]["excluded_control_plane_record_count_at_cutoff"],
            2,
        )

    def test_secret_like_run_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reports" / "m" / "r").mkdir(parents=True)
            (root / "results").mkdir()
            (root / "evidence.txt").write_text("evidence", encoding="utf-8")
            (root / "reports" / "m" / "r" / "run.json").write_text(
                json.dumps({"run_id": "r", "status": "passed", "api_key": "x"}),
                encoding="utf-8",
            )
            (root / "results" / "x.json").write_text("{}", encoding="utf-8")
            config = {
                "schema_version": "x",
                "protocol_status": "draft-not-frozen",
                "scope": "test",
                "evidence_tiers": ["development"],
                "inventory_scope": "test inventory",
                "inventory_exclusions": [
                    {
                        "glob": "reports/e0-prior-exposure/**/run.json",
                        "reason": "self attestation",
                    },
                    {
                        "glob": "reports/e0-protocol/**/run.json",
                        "reason": "protocol attestation",
                    },
                ],
                "classification_rules": [
                    {"glob": "reports/**/run.json", "tier": "development", "reason": "test"},
                    {"glob": "results/*", "tier": "development", "reason": "test"},
                ],
                "disclosures": [
                    {"id": "x", "tier": "development", "evidence": ["evidence.txt"]}
                ],
                "leaderboard_submissions": {"workspace_record_count": 0, "records": []},
                "external_private_evaluation": {"workspace_record_count": 0, "records": []},
                "limitations": [],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "secret-like keys"):
                build_prior_exposure_manifest(root, config_path)

    def test_control_plane_exclusion_breaks_self_reference(self) -> None:
        exclusions = [
            {
                "glob": "reports/e0-prior-exposure/**/run.json",
                "reason": "self attestation",
            }
        ]
        self.assertIsNotNone(
            exclusion_for_path(
                "reports/e0-prior-exposure/cutoff/run.json", exclusions
            )
        )
        self.assertIsNone(
            exclusion_for_path("reports/e0-scoring/audit/run.json", exclusions)
        )

    def test_control_plane_path_cannot_hide_solver_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hidden = root / "reports" / "e0-protocol" / "solver-public-eval"
            hidden.mkdir(parents=True)
            (root / "results").mkdir()
            (hidden / "run.json").write_text(
                json.dumps(
                    {
                        "method_id": "solver",
                        "runner": "solver.public_eval",
                        "run_id": "solver-public-eval",
                        "scope": "full-public-benchmark",
                        "status": "passed",
                    }
                ),
                encoding="utf-8",
            )
            (root / "reports" / "ordinary" / "r").mkdir(parents=True)
            (root / "reports" / "ordinary" / "r" / "run.json").write_text(
                json.dumps({"run_id": "r", "status": "passed"}), encoding="utf-8"
            )
            (root / "results" / "x.json").write_text("{}", encoding="utf-8")
            (root / "evidence.txt").write_text("x", encoding="utf-8")
            config = {
                "schema_version": "x",
                "protocol_status": "draft-not-frozen",
                "scope": "test",
                "inventory_scope": "test",
                "inventory_exclusions": [
                    {
                        "glob": "reports/e0-prior-exposure/**/run.json",
                        "reason": "self",
                    },
                    {
                        "glob": "reports/e0-protocol/**/run.json",
                        "reason": "protocol",
                    },
                ],
                "evidence_tiers": ["development"],
                "classification_rules": [
                    {
                        "glob": "reports/**/run.json",
                        "tier": "development",
                        "reason": "test",
                    },
                    {
                        "glob": "results/*",
                        "tier": "development",
                        "reason": "test",
                    },
                ],
                "disclosures": [
                    {"id": "x", "tier": "development", "evidence": ["evidence.txt"]}
                ],
                "leaderboard_submissions": {"workspace_record_count": 0, "records": []},
                "external_private_evaluation": {"workspace_record_count": 0, "records": []},
                "limitations": [],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_prior_exposure_manifest(root, config_path)


if __name__ == "__main__":
    unittest.main()
