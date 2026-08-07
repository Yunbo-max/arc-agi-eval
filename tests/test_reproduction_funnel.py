import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_reproduction_funnel import audit_funnel


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "generated_at": "test",
        "enums": {
            "execution_status": [
                "not_started",
                "running",
                "passed",
                "failed",
                "blocked",
                "not_applicable",
            ]
        },
        "summary": {
            "entry_count": 2,
            "public_candidate_count": 2,
            "partial_complex_count": 0,
            "unavailable_blocked_count": 0,
            "smoke_passed_count": 1,
            "benchmark_passed_count": 0,
            "full_reproduction_passed_count": 0,
        },
        "entries": [
            {
                "id": "alpha",
                "name": "Alpha",
                "availability": "public_candidate",
                "reproduction": {
                    "smoke": {
                        "status": "passed",
                        "feasibility": "feasible",
                        "evidence": "reports/alpha/smoke/run.json",
                    },
                    "benchmark": {
                        "status": "not_started",
                        "feasibility": "likely",
                    },
                    "full": {"status": "blocked", "feasibility": "blocked"},
                },
            },
            {
                "id": "beta",
                "name": "Beta",
                "availability": "public_candidate",
                "reproduction": {
                    "smoke": {
                        "status": "not_started",
                        "feasibility": "feasible",
                    },
                    "benchmark": {
                        "status": "not_started",
                        "feasibility": "likely",
                    },
                    "full": {"status": "blocked", "feasibility": "blocked"},
                },
                "auxiliary_evidence": [
                    {
                        "status": "passed",
                        "scope": "metadata-only",
                        "evidence": "reports/beta/aux/run.json",
                    }
                ],
            },
        ],
    }


class ReproductionFunnelTests(unittest.TestCase):
    def test_auxiliary_pass_does_not_promote_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = synthetic_manifest()
            manifest_path = root / "configs" / "baselines.json"
            write_json(manifest_path, manifest)
            write_json(
                root / "reports" / "alpha" / "smoke" / "run.json",
                {"status": "passed", "method_id": "alpha", "run_id": "smoke"},
            )
            write_json(
                root / "reports" / "beta" / "aux" / "run.json",
                {"status": "passed", "method_id": "beta", "run_id": "aux"},
            )
            write_json(
                root / "reports" / "alpha" / "source" / "run.json",
                {
                    "status": "passed",
                    "method_id": "alpha",
                    "runner": "scripts.audit_source",
                    "run_id": "source",
                    "ended_at_utc": "2026-01-01T00:00:00Z",
                    "source": {"observed_revision": "abc"},
                },
            )

            audit = audit_funnel(root, manifest_path)

        self.assertEqual(audit["validation"]["error_count"], 0)
        self.assertTrue(audit["validation"]["all_manifest_passed_evidence_valid"])
        self.assertEqual(
            audit["summary"]["main_reproduction_funnel_passed"]["smoke"], 1
        )
        self.assertEqual(
            audit["summary"]["auxiliary_passed_excluded_from_smoke"], 1
        )
        methods = {method["id"]: method for method in audit["methods"]}
        self.assertEqual(methods["beta"]["layers"]["smoke"]["status"], "not_started")
        self.assertFalse(
            methods["beta"]["auxiliary_evidence"]["items"][0][
                "counted_toward_smoke"
            ]
        )
        self.assertEqual(methods["alpha"]["layers"]["source_audit"]["status"], "passed")
        self.assertFalse(
            methods["alpha"]["layers"]["source_audit"]["counted_toward_smoke"]
        )

    def test_passed_manifest_state_rejects_nonpassed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = synthetic_manifest()
            manifest_path = root / "configs" / "baselines.json"
            write_json(manifest_path, manifest)
            write_json(
                root / "reports" / "alpha" / "smoke" / "run.json",
                {"status": "failed", "method_id": "alpha"},
            )
            write_json(
                root / "reports" / "beta" / "aux" / "run.json",
                {"status": "passed", "method_id": "beta"},
            )

            audit = audit_funnel(root, manifest_path)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("passed-evidence-run-status-mismatch", kinds)
        self.assertFalse(audit["validation"]["all_manifest_passed_evidence_valid"])
        self.assertEqual(
            audit["summary"]["manifest_passed_evidence"][
                "valid_reproduction_claim_count"
            ],
            0,
        )

    def test_passed_manifest_state_requires_existing_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = synthetic_manifest()
            manifest_path = root / "configs" / "baselines.json"
            write_json(manifest_path, manifest)
            write_json(
                root / "reports" / "beta" / "aux" / "run.json",
                {"status": "passed", "method_id": "beta"},
            )

            audit = audit_funnel(root, manifest_path)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("passed-evidence-missing", kinds)

    def test_passed_manifest_state_requires_run_json_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = synthetic_manifest()
            manifest["entries"][0]["reproduction"]["smoke"]["evidence"] = (
                "reports/alpha/smoke/evidence.json"
            )
            manifest_path = root / "configs" / "baselines.json"
            write_json(manifest_path, manifest)
            write_json(
                root / "reports" / "alpha" / "smoke" / "evidence.json",
                {"status": "passed", "method_id": "alpha"},
            )
            write_json(
                root / "reports" / "beta" / "aux" / "run.json",
                {"status": "passed", "method_id": "beta"},
            )

            audit = audit_funnel(root, manifest_path)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("passed-evidence-is-not-run-json", kinds)

    def test_current_manifest_has_24_valid_method_rows(self) -> None:
        manifest = json.loads(
            (ROOT / "configs" / "baselines.json").read_text(encoding="utf-8")
        )
        audit = audit_funnel(ROOT, ROOT / "configs" / "baselines.json")
        self.assertEqual(audit["summary"]["method_count"], 24)
        self.assertEqual(audit["validation"]["error_count"], 0)
        self.assertTrue(audit["validation"]["all_manifest_passed_evidence_valid"])
        self.assertEqual(
            audit["summary"]["main_reproduction_funnel_passed"],
            {
                "smoke": manifest["summary"]["smoke_passed_count"],
                "public_benchmark": manifest["summary"][
                    "benchmark_passed_count"
                ],
                "full_reproduction": manifest["summary"][
                    "full_reproduction_passed_count"
                ],
            },
        )
        self.assertTrue(audit["validation"]["auxiliary_excluded_from_smoke"])
        self.assertEqual(
            audit["summary"]["main_reproduction_funnel_passed"]["smoke"], 17
        )
        self.assertEqual(
            audit["summary"]["auxiliary_passed_excluded_from_smoke"], 10
        )
        self.assertEqual(
            audit["summary"]["manifest_passed_evidence"]["valid_total_claim_count"],
            27,
        )

    def test_primary_smoke_rejects_explicit_blocker_audit_evidence(self) -> None:
        cases = (
            (
                "counted-toward-smoke-false",
                {"counted_toward_smoke": False},
                ["counted_toward_smoke=false"],
            ),
            (
                "blocker-audit-scope",
                {
                    "scope": (
                        "source-dependency-label-artifact-gate-audit-only"
                    )
                },
                [
                    "scope="
                    "source-dependency-label-artifact-gate-audit-only"
                ],
            ),
            (
                "blocked-blocker-audit",
                {
                    "method_gate_status": "blocked",
                    "evidence_scope": "blocker_audit",
                },
                [
                    "method_gate_status=blocked,"
                    "evidence_scope=blocker_audit"
                ],
            ),
            (
                "blocked-nested-blocker-audit",
                {
                    "method_gate_status": "blocked",
                    "fairness": {"evidence_scope": "blocker_audit"},
                },
                [
                    "method_gate_status=blocked,"
                    "evidence_scope=blocker_audit"
                ],
            ),
        )
        for name, report_fields, expected_reasons in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = synthetic_manifest()
                manifest_path = root / "configs" / "baselines.json"
                write_json(manifest_path, manifest)
                write_json(
                    root / "reports" / "alpha" / "smoke" / "run.json",
                    {
                        "status": "passed",
                        "method_id": "alpha",
                        "run_id": "smoke",
                        **report_fields,
                    },
                )
                write_json(
                    root / "reports" / "beta" / "aux" / "run.json",
                    {"status": "passed", "method_id": "beta"},
                )

                audit = audit_funnel(root, manifest_path)

            errors = audit["validation"]["errors"]
            matching = [
                error
                for error in errors
                if error["kind"]
                == "passed-smoke-evidence-explicitly-excluded"
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["method_id"], "alpha")
            self.assertEqual(matching[0]["layer"], "smoke")
            self.assertFalse(
                audit["validation"]["all_manifest_passed_evidence_valid"]
            )
            self.assertEqual(
                audit["summary"]["main_reproduction_funnel_passed"]["smoke"],
                0,
            )
            self.assertEqual(
                audit["summary"]["manifest_passed_evidence"][
                    "valid_reproduction_claim_count"
                ],
                0,
            )
            alpha = next(
                method for method in audit["methods"] if method["id"] == "alpha"
            )
            validation = alpha["layers"]["smoke"]["evidence_validation"]
            self.assertFalse(validation["valid"])
            self.assertFalse(validation["primary_smoke_accepted"])
            self.assertEqual(
                validation["primary_smoke_exclusion_reasons"], expected_reasons
            )

    def test_legacy_metadata_gaps_warn_without_invalidating_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = synthetic_manifest()
            manifest_path = root / "configs" / "baselines.json"
            write_json(manifest_path, manifest)
            write_json(
                root / "reports" / "alpha" / "smoke" / "run.json",
                {"status": "passed", "run_id": "smoke"},
            )
            write_json(
                root / "reports" / "beta" / "aux" / "run.json",
                {
                    "status": "passed",
                    "method_id": "beta",
                    "runner": "test.runner",
                    "scope": "test-scope",
                },
            )

            audit = audit_funnel(root, manifest_path)

        kinds = {warning["kind"] for warning in audit["validation"]["warnings"]}
        self.assertEqual(audit["validation"]["error_count"], 0)
        self.assertTrue(audit["validation"]["all_manifest_passed_evidence_valid"])
        self.assertIn("passed-evidence-method-id-not-declared", kinds)
        self.assertIn("passed-evidence-runner-not-declared", kinds)
        self.assertIn("passed-evidence-scope-not-declared", kinds)

    def test_auxiliary_fair_eligibility_must_match_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = synthetic_manifest()
            manifest["entries"][1]["auxiliary_evidence"][0][
                "score_eligible_for_fair_main_board"
            ] = False
            manifest_path = root / "configs" / "baselines.json"
            write_json(manifest_path, manifest)
            write_json(
                root / "reports" / "alpha" / "smoke" / "run.json",
                {"status": "passed", "method_id": "alpha"},
            )
            write_json(
                root / "reports" / "beta" / "aux" / "run.json",
                {
                    "status": "passed",
                    "method_id": "beta",
                    "fairness": {
                        "score_eligible_for_fair_main_board": True
                    },
                },
            )

            audit = audit_funnel(root, manifest_path)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("auxiliary-fair-eligibility-mismatch", kinds)


if __name__ == "__main__":
    unittest.main()
