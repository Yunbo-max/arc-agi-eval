from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from arc_agi_eval.run_schema import (
    DEFAULT_SCHEMA_PATH,
    RunSchemaValidationError,
    load_schema,
    validate_run_file,
    validate_run_record,
)
from scripts.audit_protocol_v1_run import main as audit_main


SHA = {
    letter: letter * 64 for letter in "0123456789abcdef"
}
SCHEMA_BYTES = DEFAULT_SCHEMA_PATH.read_bytes()
SCHEMA_FILE_SHA256 = hashlib.sha256(SCHEMA_BYTES).hexdigest()


def provenance_files() -> list[dict[str, object]]:
    declarations = (
        ("schema", "contracts/run-schema.json", SCHEMA_FILE_SHA256),
        ("protocol_manifest", "contracts/protocol.json", SHA["a"]),
        ("source_lock", "manifests/source-lock.json", SHA["0"]),
        ("artifact_manifest", "manifests/artifacts.json", SHA["1"]),
        ("data_manifest", "manifests/data.json", SHA["2"]),
        ("config", "manifests/config.json", SHA["3"]),
        ("environment_lock", "manifests/environment.json", SHA["4"]),
        ("hardware_manifest", "manifests/hardware.json", SHA["b"]),
    )
    return [
        {
            "role": role,
            "path": path,
            "sha256": digest,
            "bytes": 1,
            "required_for_claim": True,
        }
        for role, path, digest in declarations
    ]


def check_record() -> dict[str, object]:
    return {
        "schema_version": "protocol-v1-run-1.0.0",
        "schema_digest_sha256": SCHEMA_FILE_SHA256,
        "protocol_id": "arc-rebench-protocol-v1-draft",
        "protocol_digest_sha256": SHA["a"],
        "method_id": "e0-protocol-v1-schema",
        "config_id": "strict-validator-v1",
        "run_id": "20260806-schema-test",
        "status": "passed",
        "evidence_scope": "e0_audit",
        "parity_class": "not_applicable",
        "resource_class": "local_cpu",
        "code_trust_class": "trusted_locked",
        "claim": "The pinned schema and validator pass their declared checks.",
        "started_at_utc": "2026-08-06T04:00:00Z",
        "ended_at_utc": "2026-08-06T04:00:01.000001Z",
        "source": {
            "lock_digest_sha256": SHA["0"],
            "revision": "a" * 40,
            "dirty": False,
            "patch_digest_sha256": None,
        },
        "artifacts": {
            "manifest_digest_sha256": SHA["1"],
            "artifact_count": 0,
            "licenses_verified": True,
        },
        "data": {
            "manifest_digest_sha256": SHA["2"],
            "dataset_id": "schema-tests",
            "split": "synthetic",
            "task_count": 0,
            "contamination_policy": "not_applicable",
        },
        "config": {
            "digest_sha256": SHA["3"],
            "seed": 20260806,
            "deterministic": True,
        },
        "hardware": {
            "profile_id": "test-cpu-host",
            "manifest_digest_sha256": SHA["b"],
            "cpu_model": "synthetic-test-cpu",
            "accelerator_kind": "none",
            "accelerator_model": None,
            "accelerator_uuid": None,
            "accelerator_count": 0,
            "exclusive_accelerator": False,
        },
        "execution": {
            "runner": "scripts.audit_protocol_v1_run",
            "command": ["python3", "-m", "unittest", "tests.test_run_schema"],
            "working_directory": ".",
            "environment_digest_sha256": SHA["4"],
            "claim_execution_started": True,
            "target_code_executed": True,
            "network_used": False,
            "gpu_requested": False,
        },
        "challenge_firewall": {
            "challenge_manifest_digest_sha256": None,
            "inference_received_test_labels": False,
            "inference_started": False,
            "scoring_after_inference": False,
            "label_mutation_check": "not_applicable",
            "network_policy": "denied",
            "write_policy": "no_writes",
            "security_isolation": "trusted_process",
        },
        "attempt_budget": {
            "top_k": 0,
            "timeout_seconds": 30,
            "max_retries": 0,
            "max_candidates": 0,
            "api_call_cap": 0,
            "input_token_cap": 0,
            "output_token_cap": 0,
            "cost_cap_usd": 0,
        },
        "resources": {
            "accounting_scope": "current_process",
            "child_processes_observed": False,
            "children_included": False,
            "sampling": "sampled",
            "wall_time_seconds": 1.000001,
            "cpu_seconds": 0.1,
            "peak_rss_bytes": 1024,
            "peak_vram_bytes": None,
            "energy_joules": None,
            "disk_delta_bytes": 0,
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0,
        },
        "results": {
            "kind": "check",
            "predictions_path": None,
            "score_path": None,
            "primary_metric": None,
            "secondary_metrics": [],
            "checks": [
                {
                    "name": "schema-self-test",
                    "status": "passed",
                    "detail": "All declared positive and negative cases passed.",
                }
            ],
        },
        "failures": {"count": 0, "items": []},
        "files": provenance_files()
        + [
            {
                "role": "test",
                "path": "evidence/check.txt",
                "sha256": SHA["5"],
                "bytes": 1,
                "required_for_claim": True,
            }
        ],
        "limitations": [
            "This validates the run-record contract, not the research protocol."
        ],
    }


def prediction_record() -> dict[str, object]:
    record = check_record()
    record.update(
        {
            "method_id": "solver-alpha",
            "config_id": "solver-alpha-reduced",
            "run_id": "20260806-solver-alpha-prediction-smoke",
            "evidence_scope": "solver_prediction_smoke",
            "parity_class": "reduced",
            "claim": "One frozen task produced schema-valid Top-2 predictions.",
        }
    )
    record["data"].update(
        {
            "dataset_id": "arc-agi-2-training",
            "split": "dev-audit",
            "task_count": 1,
            "contamination_policy": "clean",
        }
    )
    record["challenge_firewall"].update(
        {
            "challenge_manifest_digest_sha256": SHA["6"],
            "inference_started": True,
            "scoring_after_inference": True,
            "label_mutation_check": "passed",
            "write_policy": "run_directory_only",
        }
    )
    record["attempt_budget"].update(
        {
            "top_k": 2,
            "timeout_seconds": 120,
            "max_candidates": 2,
        }
    )
    record["results"] = {
        "kind": "arc_predictions",
        "predictions_path": "reports/solver-alpha/smoke/predictions.json",
        "score_path": "reports/solver-alpha/smoke/score.json",
        "primary_metric": {
            "name": "output_exact_pass_at_k",
            "role": "primary",
            "top_k": 2,
            "numerator": 1,
            "denominator": 2,
            "value": 0.5,
            "denominator_policy": "all declared test outputs",
        },
        "secondary_metrics": [],
        "checks": [],
    }
    record["files"] = [
        item for item in record["files"] if item["role"] != "test"
    ] + [
        {
            "role": "challenge_manifest",
            "path": "reports/solver-alpha/smoke/challenge-manifest.json",
            "sha256": SHA["6"],
            "bytes": 12,
            "required_for_claim": True,
        },
        {
            "role": "predictions",
            "path": "reports/solver-alpha/smoke/predictions.json",
            "sha256": SHA["7"],
            "bytes": 12,
            "required_for_claim": True,
        },
        {
            "role": "results",
            "path": "reports/solver-alpha/smoke/score.json",
            "sha256": SHA["8"],
            "bytes": 12,
            "required_for_claim": True,
        },
    ]
    return record


def blocked_record() -> dict[str, object]:
    record = check_record()
    record.update(
        {
            "status": "blocked",
            "evidence_scope": "solver_prediction_smoke",
            "code_trust_class": "generated_untrusted",
            "resource_class": "non_execution",
            "results": None,
        }
    )
    record["files"] = [
        item for item in record["files"] if item["role"] != "test"
    ]
    record["execution"].update(
        {
            "claim_execution_started": False,
            "target_code_executed": False,
        }
    )
    record["challenge_firewall"].update(
        {
            "network_policy": "not_applicable",
            "write_policy": "not_applicable",
            "security_isolation": "no_execution",
        }
    )
    record["resources"].update(
        {
            "accounting_scope": "not_applicable",
            "sampling": "not_applicable",
            "cpu_seconds": None,
            "peak_rss_bytes": None,
        }
    )
    record["failures"] = {
        "count": 1,
        "items": [
            {
                "kind": "blocker",
                "code": "strict-sandbox-unavailable",
                "message": "Generated code was not executed without a strict sandbox.",
                "retryable": True,
            }
        ],
    }
    return record


def materialize_declared_files(root: Path, record: dict[str, object]) -> None:
    role_digests: dict[str, str] = {}
    for index, file_record in enumerate(record["files"]):
        path = root / file_record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            SCHEMA_BYTES
            if file_record["role"] == "schema"
            else f"{index}:{file_record['role']}:{file_record['path']}\n".encode()
        )
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        file_record["sha256"] = digest
        file_record["bytes"] = len(payload)
        role_digests[file_record["role"]] = digest

    record["schema_digest_sha256"] = role_digests["schema"]
    record["protocol_digest_sha256"] = role_digests["protocol_manifest"]
    record["source"]["lock_digest_sha256"] = role_digests["source_lock"]
    record["artifacts"]["manifest_digest_sha256"] = role_digests[
        "artifact_manifest"
    ]
    record["data"]["manifest_digest_sha256"] = role_digests["data_manifest"]
    record["config"]["digest_sha256"] = role_digests["config"]
    record["execution"]["environment_digest_sha256"] = role_digests[
        "environment_lock"
    ]
    record["hardware"]["manifest_digest_sha256"] = role_digests[
        "hardware_manifest"
    ]
    if "challenge_manifest" in role_digests:
        record["challenge_firewall"][
            "challenge_manifest_digest_sha256"
        ] = role_digests["challenge_manifest"]
    if "source_patch" in role_digests:
        record["source"]["patch_digest_sha256"] = role_digests["source_patch"]


def assert_invalid(
    test: unittest.TestCase, record: dict[str, object], expected: str
) -> None:
    with test.assertRaises(RunSchemaValidationError) as caught:
        validate_run_record(record)
    test.assertTrue(
        any(expected in issue for issue in caught.exception.issues),
        f"expected {expected!r} in {caught.exception.issues!r}",
    )


class ProtocolV1RunSchemaTests(unittest.TestCase):
    def test_schema_and_valid_check_record_pass(self) -> None:
        schema = load_schema()
        result = validate_run_record(check_record(), schema=schema)
        self.assertEqual(result.run_id, "20260806-schema-test")
        self.assertEqual(result.declared_file_count, 9)
        self.assertEqual(result.verified_file_count, 0)

    def test_validator_rejects_schema_keywords_it_does_not_implement(self) -> None:
        schema = load_schema()
        schema["allOf"] = []
        with self.assertRaisesRegex(ValueError, "unsupported JSON Schema keyword"):
            validate_run_record(check_record(), schema=schema)

        schema = load_schema()
        schema["properties"]["method_id"]["minimum"] = 1
        with self.assertRaisesRegex(ValueError, "ref siblings"):
            validate_run_record(check_record(), schema=schema)

        schema = load_schema()
        schema["additionalProperties"] = {}
        with self.assertRaisesRegex(ValueError, "only boolean values"):
            validate_run_record(check_record(), schema=schema)

        schema = load_schema()
        schema["properties"]["started_at_utc"] = {
            "type": "string",
            "format": "uri",
        }
        with self.assertRaisesRegex(ValueError, "unsupported format"):
            validate_run_record(check_record(), schema=schema)

        schema = load_schema()
        schema["properties"]["method_id"]["$ref"] = "#/$defs/missing"
        with self.assertRaisesRegex(ValueError, "unresolvable JSON Schema reference"):
            validate_run_record(check_record(), schema=schema)

    def test_record_schema_digest_must_match_selected_schema_file(self) -> None:
        record = check_record()
        record["schema_digest_sha256"] = SHA["e"]
        schema_file = next(
            item for item in record["files"] if item["role"] == "schema"
        )
        schema_file["sha256"] = SHA["e"]
        with self.assertRaises(RunSchemaValidationError) as caught:
            validate_run_record(record, schema_sha256=SHA["f"])
        self.assertTrue(
            any("validator's schema file" in issue for issue in caught.exception.issues)
        )

    def test_valid_prediction_record_passes(self) -> None:
        result = validate_run_record(prediction_record())
        self.assertEqual(result.declared_file_count, 11)

    def test_blocked_generated_code_without_results_is_valid(self) -> None:
        result = validate_run_record(blocked_record())
        self.assertEqual(result.run_id, "20260806-schema-test")

    def test_failed_record_without_results_is_valid(self) -> None:
        record = check_record()
        record["status"] = "failed"
        record["results"] = None
        record["failures"] = {
            "count": 1,
            "items": [
                {
                    "kind": "error",
                    "code": "import-failed",
                    "message": "The declared import command returned nonzero.",
                    "retryable": True,
                }
            ],
        }
        validate_run_record(record)

    def test_unknown_fields_are_rejected_at_every_level(self) -> None:
        record = check_record()
        record["execution"]["surprise"] = True
        assert_invalid(self, record, "unknown field")

    def test_secret_like_field_names_and_values_are_rejected(self) -> None:
        record = check_record()
        record["execution"]["api_key"] = "redacted"
        assert_invalid(self, record, "secret-like field name")

        record = check_record()
        record["limitations"].append("accidentally copied sk-abcdefghijklmnopq")
        assert_invalid(self, record, "value resembles a secret")

    def test_generated_code_execution_requires_strict_sandbox(self) -> None:
        record = prediction_record()
        record["code_trust_class"] = "generated_untrusted"
        assert_invalid(self, record, "requires strict_sandbox")

        record["challenge_firewall"]["security_isolation"] = "strict_sandbox"
        validate_run_record(record)

    def test_every_provenance_digest_is_bound_to_a_required_file(self) -> None:
        mutations = (
            (
                lambda record: record.__setitem__(
                    "schema_digest_sha256", SHA["e"]
                ),
                "schema",
            ),
            (
                lambda record: record.__setitem__(
                    "protocol_digest_sha256", SHA["e"]
                ),
                "protocol_manifest",
            ),
            (
                lambda record: record["source"].__setitem__(
                    "lock_digest_sha256", SHA["e"]
                ),
                "source_lock",
            ),
            (
                lambda record: record["artifacts"].__setitem__(
                    "manifest_digest_sha256", SHA["e"]
                ),
                "artifact_manifest",
            ),
            (
                lambda record: record["data"].__setitem__(
                    "manifest_digest_sha256", SHA["e"]
                ),
                "data_manifest",
            ),
            (
                lambda record: record["config"].__setitem__(
                    "digest_sha256", SHA["e"]
                ),
                "config",
            ),
            (
                lambda record: record["execution"].__setitem__(
                    "environment_digest_sha256", SHA["e"]
                ),
                "environment_lock",
            ),
            (
                lambda record: record["hardware"].__setitem__(
                    "manifest_digest_sha256", SHA["e"]
                ),
                "hardware_manifest",
            ),
        )
        for mutate, role in mutations:
            with self.subTest(role=role):
                record = check_record()
                mutate(record)
                assert_invalid(self, record, f"no required {role} file")

        record = prediction_record()
        record["challenge_firewall"]["challenge_manifest_digest_sha256"] = SHA["e"]
        assert_invalid(self, record, "no required challenge_manifest file")

        record = check_record()
        record["source"].update(
            {"dirty": True, "patch_digest_sha256": SHA["e"]}
        )
        assert_invalid(self, record, "no required source_patch file")

    def test_passed_prediction_requires_label_mutation_check(self) -> None:
        for scope in (
            "solver_prediction_smoke",
            "fixed_subset_benchmark",
            "full_public_benchmark",
        ):
            with self.subTest(scope=scope):
                record = prediction_record()
                record["evidence_scope"] = scope
                record["challenge_firewall"]["label_mutation_check"] = "not_run"
                assert_invalid(self, record, "label_mutation_check")

    def test_passed_prediction_rejects_inference_test_labels(self) -> None:
        for scope in (
            "solver_prediction_smoke",
            "fixed_subset_benchmark",
            "full_public_benchmark",
        ):
            with self.subTest(scope=scope):
                record = prediction_record()
                record["evidence_scope"] = scope
                record["challenge_firewall"][
                    "inference_received_test_labels"
                ] = True
                assert_invalid(self, record, "inference_received_test_labels")

    def test_passed_prediction_requires_prediction_and_score_files(self) -> None:
        record = prediction_record()
        record["files"] = []
        assert_invalid(self, record, "no matching required predictions")

    def test_blocked_and_failed_terminal_states_must_be_coherent(self) -> None:
        record = blocked_record()
        record["execution"]["claim_execution_started"] = True
        assert_invalid(self, record, "blocked requires false")

        record = check_record()
        record["status"] = "failed"
        assert_invalid(self, record, "failed terminal records require a failure")

    def test_failure_count_and_metric_ratio_are_checked(self) -> None:
        record = blocked_record()
        record["failures"]["count"] = 2
        assert_invalid(self, record, "must equal the number")

        record = prediction_record()
        record["results"]["primary_metric"]["value"] = 0.25
        assert_invalid(self, record, "numerator/denominator")

    def test_resource_scope_and_api_caps_are_cross_checked(self) -> None:
        record = check_record()
        record["resources"]["children_included"] = True
        assert_invalid(self, record, "current_process scope excludes children")

        record = check_record()
        record["resources"]["api_calls"] = 1
        assert_invalid(self, record, "exceeds attempt_budget.api_call_cap")

    def test_hardware_identity_and_benchmark_exclusivity_are_checked(self) -> None:
        record = prediction_record()
        record["evidence_scope"] = "fixed_subset_benchmark"
        record["resource_class"] = "local_gpu"
        record["execution"]["gpu_requested"] = True
        record["hardware"].update(
            {
                "profile_id": "test-rtx3090",
                "cpu_model": "synthetic-test-cpu",
                "accelerator_kind": "nvidia_gpu",
                "accelerator_model": "NVIDIA GeForce RTX 3090",
                "accelerator_uuid": "GPU-test",
                "accelerator_count": 1,
                "exclusive_accelerator": False,
            }
        )
        assert_invalid(self, record, "passed local benchmark requires true")

        record["hardware"]["exclusive_accelerator"] = True
        validate_run_record(record)

    def test_repo_paths_reject_absolute_parent_and_windows_forms(self) -> None:
        for unsafe in ("/tmp/evidence", "../evidence", "a/../evidence", "C:/x"):
            with self.subTest(path=unsafe):
                record = check_record()
                record["files"][0]["path"] = unsafe
                assert_invalid(self, record, "path")

    def test_file_integrity_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = check_record()
            materialize_declared_files(root, record)
            result = validate_run_record(
                record, repo_root=root, verify_files=True
            )
            self.assertEqual(result.verified_file_count, len(record["files"]))

            first_path = root / record["files"][0]["path"]
            first_path.write_bytes(b"tampered")
            assert_invalid_with_root = self.assertRaises(RunSchemaValidationError)
            with assert_invalid_with_root as caught:
                validate_run_record(record, repo_root=root, verify_files=True)
            self.assertTrue(
                any("declared digest" in issue for issue in caught.exception.issues)
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_cannot_escape_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            record = check_record()
            materialize_declared_files(root, record)
            target = Path(outside) / "outside.txt"
            target.write_bytes(b"x")
            evidence = root / "evidence" / "check.txt"
            evidence.unlink()
            os.symlink(target, evidence)
            assert_invalid_with_root = self.assertRaises(RunSchemaValidationError)
            with assert_invalid_with_root as caught:
                validate_run_record(record, repo_root=root, verify_files=True)
            self.assertTrue(
                any("escapes the repository root" in issue for issue in caught.exception.issues)
            )

    def test_duplicate_json_keys_are_rejected_before_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_path = Path(temporary) / "run.json"
            run_path.write_text('{"run_id":"one","run_id":"two"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                validate_run_file(run_path, verify_files=False)

    def test_cli_reports_success_and_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = check_record()
            materialize_declared_files(root, record)
            run_path = root / "run.json"
            run_path.write_text(json.dumps(record), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = audit_main(
                    [str(run_path), "--repo-root", str(root), "--json"]
                )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "passed")

            record["unexpected"] = True
            run_path.write_text(json.dumps(record), encoding="utf-8")
            output = io.StringIO()
            errors = io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                status = audit_main(
                    [str(run_path), "--repo-root", str(root), "--json"]
                )
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
