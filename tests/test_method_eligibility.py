import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_method_eligibility import (
    _validate_prediction_run,
    _validate_strict_runtime_run,
    audit_inventory,
    immutable_json,
)
from tests.test_run_schema import materialize_declared_files, prediction_record


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "configs" / "method_eligibility.json"
BASELINE_PATH = ROOT / "configs" / "baselines.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def load_inventory() -> dict[str, object]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def write_temporary_inventory(
    directory: str, inventory: dict[str, object]
) -> Path:
    path = Path(directory) / "method_eligibility.json"
    write_json(path, inventory)
    return path


class MethodEligibilityTests(unittest.TestCase):
    def test_current_inventory_is_exact_and_evidence_validated(self) -> None:
        audit = audit_inventory(ROOT, INVENTORY_PATH, BASELINE_PATH)

        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["validation"]["error_count"], 0)
        self.assertTrue(audit["validation"]["exact_baseline_id_set"])
        self.assertTrue(audit["validation"]["baseline_order_preserved"])
        self.assertTrue(audit["validation"]["all_declared_evidence_paths_exist"])
        self.assertEqual(audit["summary"]["entry_count"], 24)
        self.assertEqual(
            audit["summary"]["solver_prediction_artifact_validated_count"], 2
        )
        self.assertEqual(
            audit["summary"]["component_plus_solver_prediction_count"], 17
        )
        self.assertEqual(
            audit["summary"]["performance_table_eligible"],
            {"eligible": 0, "ineligible": 24},
        )
        self.assertEqual(
            audit["summary"]["solver_prediction_smoke_status"]["passed"], 2
        )
        self.assertEqual(
            audit["summary"]["strict_runtime_promotion_status"],
            {
                "passed": 2,
                "failed": 0,
                "blocked": 0,
                "not_run": 22,
                "unavailable": 0,
            },
        )
        self.assertEqual(
            audit["summary"]["strict_runtime_artifact_validated_count"], 2
        )
        self.assertTrue(
            audit["validation"]["global_runtime_core_cannot_promote_method"]
        )
        self.assertTrue(
            audit["validation"][
                "strict_runtime_requires_protocol_v1_file_validation"
            ]
        )

    def test_strict_runtime_promotion_is_method_specific(self) -> None:
        inventory = load_inventory()

        self.assertEqual(len(inventory["entries"]), 24)
        expected_arc_nca = {
            "status": "passed",
            "config_id": "arc-nca-cpu-dev-smoke-v1",
            "evidence": [
                "reports/arc-nca/20260806-cpu-dev-6150a2bd-strict-v1/run.json"
            ],
        }
        expected_compressarc = {
            "status": "passed",
            "config_id": "compressarc-cpu-dev-smoke-v1",
            "evidence": [
                "reports/compressarc/20260806-cpu-dev-3c9b0459-strict-v1/run.json"
            ],
        }
        for entry in inventory["entries"]:
            if entry["id"] == "arc-nca":
                self.assertEqual(entry["strict_runtime_promotion"], expected_arc_nca)
            elif entry["id"] == "compressarc":
                self.assertEqual(
                    entry["strict_runtime_promotion"], expected_compressarc
                )
            else:
                self.assertEqual(
                    entry["strict_runtime_promotion"],
                    {"status": "not_run", "config_id": None, "evidence": []},
                    entry["id"],
                )
            self.assertFalse(entry["performance_table_eligible"], entry["id"])

    def test_component_and_scorer_evidence_are_not_solver_smokes(self) -> None:
        audit = audit_inventory(ROOT, INVENTORY_PATH, BASELINE_PATH)
        methods = {method["id"]: method for method in audit["methods"]}

        for method_id in (
            "barc",
            "arcmemo",
            "arc-lang-public",
            "epang-arc-agi",
        ):
            self.assertEqual(methods[method_id]["evidence_scope"], "component")
            self.assertEqual(
                methods[method_id]["solver_prediction_smoke_status"], "not_run"
            )
            self.assertFalse(methods[method_id]["performance_table_eligible"])
        self.assertEqual(methods["routemoa"]["evidence_scope"], "scorer_only")
        self.assertEqual(
            methods["routemoa"]["solver_prediction_smoke_status"], "not_run"
        )
        self.assertFalse(methods["routemoa"]["performance_table_eligible"])
        self.assertEqual(
            audit["summary"]["code_trust_class"]["generated_untrusted"], 5
        )
        self.assertEqual(audit["summary"]["code_trust_class"]["api_network"], 3)

    def test_inventory_must_match_every_baseline_id(self) -> None:
        inventory = load_inventory()
        inventory["entries"] = inventory["entries"][:-1]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_temporary_inventory(temporary, inventory)
            audit = audit_inventory(ROOT, path, BASELINE_PATH)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("baseline-inventory-id-set-mismatch", kinds)
        self.assertIn("baseline-inventory-order-mismatch", kinds)
        self.assertIn("declared-summary-mismatch", kinds)

    def test_component_scope_cannot_be_promoted_by_a_boolean(self) -> None:
        inventory = load_inventory()
        barc = inventory["entries"][0]
        self.assertEqual(barc["evidence_scope"], "component")
        barc["performance_table_eligible"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = write_temporary_inventory(temporary, inventory)
            audit = audit_inventory(ROOT, path, BASELINE_PATH)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("non-solver-evidence-promoted-to-performance-table", kinds)
        self.assertIn("eligible-row-lacks-passed-solver-prediction", kinds)
        self.assertIn("eligible-row-lacks-passed-strict-runtime", kinds)
        self.assertIn("declared-summary-mismatch", kinds)

    def test_not_run_strict_runtime_cannot_carry_evidence(self) -> None:
        inventory = load_inventory()
        inventory["entries"][0]["strict_runtime_promotion"]["evidence"] = [
            "reports/e0-challenge-runtime/20260806-deterministic-baseline-dev-audit-v1/run.json"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_temporary_inventory(temporary, inventory)
            audit = audit_inventory(ROOT, path, BASELINE_PATH)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("not-run-strict-runtime-has-evidence", kinds)

    def test_missing_evidence_path_fails_closed(self) -> None:
        inventory = load_inventory()
        inventory["entries"][0]["solver_prediction_smoke"]["evidence"] = [
            "reports/barc/does-not-exist/run.json"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_temporary_inventory(temporary, inventory)
            audit = audit_inventory(ROOT, path, BASELINE_PATH)

        kinds = {error["kind"] for error in audit["validation"]["errors"]}
        self.assertIn("evidence-file-missing", kinds)
        self.assertIn("component-passing-evidence-not-found", kinds)
        self.assertFalse(audit["validation"]["all_declared_evidence_paths_exist"])

    def test_prediction_artifact_hash_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "reports" / "alpha" / "solver-smoke"
            prediction = b'{"alpha": [[[1]]]}\n'
            prediction_path = run_dir / "predictions.json"
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_bytes(prediction)
            write_json(
                run_dir / "run.json",
                {
                    "status": "passed",
                    "runner": "tests.synthetic_solver",
                    "configuration": {
                        "test_outputs_available_to_optimizer": False
                    },
                    "metrics": {"tasks_predicted": 1},
                    "source": {"revision": "locked-revision"},
                    "artifacts": {
                        "predictions": "predictions.json",
                        "prediction_sha256": "0" * 64,
                    },
                },
            )
            errors: list[dict[str, object]] = []
            observed = _validate_prediction_run(
                root,
                "alpha",
                "reports/alpha/solver-smoke/run.json",
                errors,
            )

        kinds = {error["kind"] for error in errors}
        self.assertIn("solver-prediction-artifact-hash-mismatch", kinds)
        self.assertEqual(
            observed["prediction_sha256"], hashlib.sha256(prediction).hexdigest()
        )

    def test_prediction_smoke_requires_explicit_optimizer_label_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "reports" / "alpha" / "solver-smoke"
            prediction = b"{}\n"
            prediction_path = run_dir / "predictions.json"
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_bytes(prediction)
            write_json(
                run_dir / "run.json",
                {
                    "status": "passed",
                    "runner": "tests.synthetic_solver",
                    "configuration": {},
                    "metrics": {"tasks_predicted": 1},
                    "source": {"revision": "locked-revision"},
                    "artifacts": {
                        "predictions": "predictions.json",
                        "prediction_sha256": hashlib.sha256(prediction).hexdigest(),
                    },
                },
            )
            errors: list[dict[str, object]] = []
            _validate_prediction_run(
                root,
                "alpha",
                "reports/alpha/solver-smoke/run.json",
                errors,
            )

        kinds = {error["kind"] for error in errors}
        self.assertIn("solver-prediction-label-exclusion-not-proven", kinds)

    def test_strict_runtime_validates_protocol_v1_files_and_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = prediction_record()
            materialize_declared_files(root, record)
            run_path = root / "reports" / "solver-alpha" / "smoke" / "run.json"
            write_json(run_path, record)
            errors: list[dict[str, object]] = []

            observed = _validate_strict_runtime_run(
                root,
                "solver-alpha",
                "solver-alpha-reduced",
                "reports/solver-alpha/smoke/run.json",
                errors,
            )

        self.assertEqual(errors, [])
        self.assertIsNotNone(observed)
        self.assertEqual(observed["method_id"], "solver-alpha")
        self.assertEqual(observed["config_id"], "solver-alpha-reduced")
        self.assertEqual(
            observed["declared_file_count"], observed["verified_file_count"]
        )
        self.assertEqual(observed["label_mutation_check"], "passed")
        self.assertFalse(observed["inference_received_test_labels"])
        self.assertTrue(observed["scoring_after_inference"])

    def test_strict_runtime_rejects_tampered_declared_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = prediction_record()
            materialize_declared_files(root, record)
            run_path = root / "reports" / "solver-alpha" / "smoke" / "run.json"
            write_json(run_path, record)
            (root / record["results"]["predictions_path"]).write_text(
                "tampered\n", encoding="utf-8"
            )
            errors: list[dict[str, object]] = []

            observed = _validate_strict_runtime_run(
                root,
                "solver-alpha",
                "solver-alpha-reduced",
                "reports/solver-alpha/smoke/run.json",
                errors,
            )

        self.assertIsNone(observed)
        self.assertIn(
            "strict-runtime-protocol-v1-validation-failed",
            {error["kind"] for error in errors},
        )

    def test_strict_runtime_rejects_method_and_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = prediction_record()
            record["method_id"] = "solver-beta"
            materialize_declared_files(root, record)
            run_path = root / "reports" / "solver-alpha" / "smoke" / "run.json"
            write_json(run_path, record)
            errors: list[dict[str, object]] = []

            observed = _validate_strict_runtime_run(
                root,
                "solver-alpha",
                "wrong-config",
                "reports/solver-alpha/smoke/run.json",
                errors,
            )

        self.assertIsNone(observed)
        kinds = {error["kind"] for error in errors}
        self.assertIn("strict-runtime-method-id-mismatch", kinds)
        self.assertIn("strict-runtime-config-id-mismatch", kinds)

    def test_global_runtime_core_cannot_be_reused_for_method_promotion(self) -> None:
        errors: list[dict[str, object]] = []
        observed = _validate_strict_runtime_run(
            ROOT,
            "barc",
            "challenge-runtime-core-v1",
            "reports/e0-challenge-runtime/20260806-deterministic-baseline-dev-audit-v1/run.json",
            errors,
        )

        self.assertIsNone(observed)
        self.assertIn(
            "strict-runtime-global-core-cannot-promote-method",
            {error["kind"] for error in errors},
        )

    def test_immutable_report_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run.json"
            immutable_json(output, {"status": "passed"})
            with self.assertRaises(FileExistsError):
                immutable_json(output, {"status": "failed"})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "passed"},
            )


if __name__ == "__main__":
    unittest.main()
