import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "analysis_plan_v1.json"
REPORT = ROOT / "reports" / "e0-analysis" / "20260806-analysis-plan-v1"


class AnalysisPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_target_cycle_is_frozen_without_invented_dates(self) -> None:
        venue = self.config["venue_freeze"]
        self.assertEqual(venue["target_venue"], "NeurIPS")
        self.assertEqual(venue["target_track"], "Evaluations & Datasets")
        self.assertEqual(venue["target_cycle"], 2027)
        self.assertEqual(
            venue["official_2027_author_cfp_status"],
            "not_published_as_of_2026-08-06",
        )
        for field in (
            "official_2027_author_cfp_url",
            "submission_portal_url",
            "abstract_deadline",
            "full_paper_deadline",
            "deadline_timezone",
        ):
            self.assertIsNone(venue[field])

    def test_scientific_decisions_and_campaign_cap_are_frozen(self) -> None:
        self.assertEqual(self.config["freeze_status"], "frozen-v1")
        self.assertEqual(self.config["metrics"]["top_k"], 2)
        self.assertEqual(self.config["multiplicity"]["primary_adjustment"], "Holm")
        self.assertEqual(
            self.config["campaign_cap"]["local_gpu_hour_cap_including_contingency"],
            1500,
        )
        self.assertEqual(self.config["campaign_cap"]["api_spend_cap_usd"], 0)
        self.assertFalse(self.config["campaign_cap"]["api_execution_authorized"])
        self.assertEqual(
            self.config["hypotheses"][2]["status"],
            "declared_infeasible_for_protocol_v1_at_freeze",
        )

    def test_power_plan_does_not_use_locked_public_outcomes(self) -> None:
        power = self.config["power_simulation"]
        self.assertFalse(power["uses_locked_public_outcomes"])
        self.assertEqual(power["task_counts"], [64, 120, 400])
        self.assertEqual(power["simulations_per_cell"], 2000)

    def test_saved_report_passes_all_frozen_checks(self) -> None:
        if not (REPORT / "run.json").is_file():
            self.skipTest("analysis-plan report has not been generated")
        run = json.loads((REPORT / "run.json").read_text(encoding="utf-8"))
        results = json.loads(
            (REPORT / "results-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run["status"], "passed")
        self.assertEqual(run["schema_version"], "protocol-v1-run-1.0.0")
        self.assertEqual(results["target_cycle"], 2027)
        self.assertTrue(results["statistical_self_test_passed"])
        self.assertTrue(results["power_simulation_deterministic"])
        self.assertEqual(results["new_locked_public_method_scores_used"], 0)


if __name__ == "__main__":
    unittest.main()
