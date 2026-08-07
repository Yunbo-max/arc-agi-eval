import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


from arc_agi_eval.protocol import build_protocol_manifest


class ProtocolRootTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "evidence").mkdir()
        (self.root / "evidence" / "scorer.json").write_text("{}\n", encoding="utf-8")
        (self.root / "evidence" / "gate.json").write_text(
            '{"semantic_check": true, "status": "passed"}\n', encoding="utf-8"
        )
        (self.root / "configs").mkdir()
        self.config = {
            "schema_version": "1.0.0-draft",
            "protocol_id": "test-protocol-v1-draft",
            "protocol_status": "draft-not-frozen",
            "comparison_axes": {
                "evidence_scope": ["development", "locked_public"],
                "parity_class": ["local", "paper_exact"],
                "resource_class": ["cpu", "single_gpu"],
                "code_trust_class": ["trusted_locked", "generated_untrusted"],
            },
            "scoring_contract": {
                "primary": {
                    "name": "output_exact_pass_at_k",
                    "top_k": 2,
                    "denominator_policy": "all_declared_test_outputs",
                },
                "missing_output_policy": "zero_credit_in_declared_denominator",
                "evidence": ["evidence/scorer.json"],
            },
            "benchmark_contracts": {
                "arc_agi_1": {
                    "benchmark_generation": "arc_agi_1",
                    "top_k": 2,
                    "declared_task_count": 400,
                    "declared_output_count": 419,
                    "evidence": ["evidence/scorer.json"],
                },
                "arc_agi_2": {
                    "benchmark_generation": "arc_agi_2",
                    "top_k": 2,
                    "declared_task_count": 120,
                    "declared_output_count": 167,
                    "evidence": ["evidence/scorer.json"],
                },
            },
            "freeze_gates": [
                {
                    "id": "scorer",
                    "class": "core_measurement",
                    "status": "passed",
                    "required_for_freeze": True,
                    "evidence": ["evidence/gate.json"],
                    "acceptance_assertions": [
                        {
                            "path": "evidence/gate.json",
                            "pointer": "/status",
                            "equals": "passed",
                        },
                        {
                            "path": "evidence/gate.json",
                            "pointer": "/semantic_check",
                            "equals": True,
                        },
                    ],
                    "blocker": None,
                },
                {
                    "id": "budget",
                    "class": "locked_public",
                    "status": "pending",
                    "required_for_freeze": True,
                    "evidence": [],
                    "acceptance_assertions": [],
                    "blocker": "budget is not frozen",
                },
            ],
            "unresolved_p0": [
                {"gate_id": "budget", "resolution": "freeze the budget"}
            ],
            "limitations": ["synthetic fixture"],
        }
        self.config_path = self.root / "configs" / "protocol.json"
        self._write(self.config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, value: dict) -> None:
        self.config_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_valid_draft_has_one_mechanical_blocker_and_stable_digest(self) -> None:
        first = build_protocol_manifest(self.root, self.config_path)
        second = build_protocol_manifest(self.root, self.config_path)
        self.assertFalse(first["readiness"]["freeze_ready"])
        self.assertEqual(first["readiness"]["required_unmet_gate_ids"], ["budget"])
        self.assertEqual(first["protocol_root_sha256"], second["protocol_root_sha256"])
        self.assertEqual(len(first["evidence_inventory"]), 2)

    def test_passed_gate_without_evidence_is_rejected(self) -> None:
        config = deepcopy(self.config)
        config["freeze_gates"][0]["evidence"] = []
        self._write(config)
        with self.assertRaisesRegex(ValueError, "passed gate requires evidence"):
            build_protocol_manifest(self.root, self.config_path)

    def test_semantic_assertion_is_evaluated(self) -> None:
        config = deepcopy(self.config)
        config["freeze_gates"][0]["acceptance_assertions"][1]["equals"] = False
        self._write(config)
        with self.assertRaisesRegex(ValueError, "gate assertion failed"):
            build_protocol_manifest(self.root, self.config_path)

    def test_status_only_assertion_is_not_enough(self) -> None:
        config = deepcopy(self.config)
        config["freeze_gates"][0]["acceptance_assertions"] = [
            config["freeze_gates"][0]["acceptance_assertions"][0]
        ]
        self._write(config)
        with self.assertRaisesRegex(ValueError, "semantic assertion beyond"):
            build_protocol_manifest(self.root, self.config_path)

    def test_passed_gate_cannot_assert_blocked_run_status(self) -> None:
        config = deepcopy(self.config)
        run_path = self.root / "evidence" / "run.json"
        run_path.write_text(
            '{"semantic_check": true, "status": "blocked"}\n', encoding="utf-8"
        )
        config["freeze_gates"][0]["evidence"] = ["evidence/run.json"]
        for assertion in config["freeze_gates"][0]["acceptance_assertions"]:
            assertion["path"] = "evidence/run.json"
        config["freeze_gates"][0]["acceptance_assertions"][0][
            "equals"
        ] = "blocked"
        self._write(config)
        with self.assertRaisesRegex(ValueError, "equals passed"):
            build_protocol_manifest(self.root, self.config_path)

    def test_at_least_one_gate_must_be_required(self) -> None:
        config = deepcopy(self.config)
        for gate in config["freeze_gates"]:
            gate["required_for_freeze"] = False
        config["unresolved_p0"] = []
        self._write(config)
        with self.assertRaisesRegex(ValueError, "at least one freeze gate"):
            build_protocol_manifest(self.root, self.config_path)

    def test_official_protocol_cannot_omit_mandatory_roster(self) -> None:
        config = deepcopy(self.config)
        config["protocol_id"] = "arc-rebench-protocol-v1-draft"
        self._write(config)
        with self.assertRaisesRegex(ValueError, "missing mandatory gate IDs"):
            build_protocol_manifest(self.root, self.config_path)

    def test_p0_must_exactly_match_required_unmet_gates(self) -> None:
        config = deepcopy(self.config)
        config["unresolved_p0"] = []
        self._write(config)
        with self.assertRaisesRegex(ValueError, "must match every required"):
            build_protocol_manifest(self.root, self.config_path)

    def test_frozen_protocol_cannot_retain_blocker(self) -> None:
        config = deepcopy(self.config)
        config["protocol_status"] = "frozen"
        self._write(config)
        with self.assertRaisesRegex(ValueError, "frozen protocol cannot"):
            build_protocol_manifest(self.root, self.config_path)

    def test_evidence_mutation_changes_protocol_root(self) -> None:
        before = build_protocol_manifest(self.root, self.config_path)
        (self.root / "evidence" / "gate.json").write_text(
            '{"changed": true, "semantic_check": true, "status": "passed"}\n',
            encoding="utf-8",
        )
        after = build_protocol_manifest(self.root, self.config_path)
        self.assertNotEqual(before["protocol_root_sha256"], after["protocol_root_sha256"])


class CurrentProtocolDraftTests(unittest.TestCase):
    def test_current_root_is_valid_but_not_freeze_ready(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = build_protocol_manifest(
            root, root / "configs" / "protocol_v1_draft.json"
        )
        self.assertEqual(manifest["protocol_status"], "draft-not-frozen")
        self.assertFalse(manifest["readiness"]["freeze_ready"])
        self.assertEqual(manifest["readiness"]["required_unmet_count"], 1)
        self.assertEqual(
            set(manifest["readiness"]["required_unmet_gate_ids"]),
            {
                "lp.process-tree-resources",
            },
        )


if __name__ == "__main__":
    unittest.main()
