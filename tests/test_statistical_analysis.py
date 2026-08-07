import math
import unittest

import numpy as np

from arc_agi_eval.statistical_analysis import (
    AnalysisFallback,
    cluster_bootstrap_mean_interval,
    fit_cluster_robust_logistic,
    holm_adjust,
    minimum_detectable_effect_grid,
    paired_binary_exact_pvalue,
    paired_randomization_test,
    wald_block_test,
)


class StatisticalAnalysisTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust([0.04, 0.01, 0.03])
        self.assertEqual(adjusted, [0.06, 0.03, 0.06])
        self.assertEqual(holm_adjust([]), [])

    def test_exact_paired_binary_test(self) -> None:
        self.assertEqual(paired_binary_exact_pvalue([0, 1], [0, 1]), 1.0)
        first = [0] * 10
        second = [1] * 10
        self.assertAlmostEqual(paired_binary_exact_pvalue(first, second), 2 / 1024)

    def test_sign_flip_and_cluster_bootstrap_are_deterministic(self) -> None:
        differences = [1.0] * 25 + [-0.25] * 5
        first = paired_randomization_test(
            differences, seed=20260806, monte_carlo_resamples=2000
        )
        second = paired_randomization_test(
            differences, seed=20260806, monte_carlo_resamples=2000
        )
        self.assertEqual(first, second)
        interval = cluster_bootstrap_mean_interval(
            differences, seed=20260806, resamples=1000
        )
        self.assertLessEqual(interval["lower"], interval["estimate"])
        self.assertGreaterEqual(interval["upper"], interval["estimate"])

    def test_cluster_robust_logistic_and_wald(self) -> None:
        rng = np.random.default_rng(20260806)
        design = []
        outcomes = []
        clusters = []
        for cluster in range(80):
            cluster_shift = rng.normal(0, 0.25)
            for condition in (0, 1):
                design.append([1.0, float(condition)])
                probability = 1 / (1 + math.exp(-(-0.7 + 0.8 * condition + cluster_shift)))
                outcomes.append(int(rng.random() < probability))
                clusters.append(cluster)
        fit = fit_cluster_robust_logistic(design, outcomes, clusters)
        self.assertTrue(fit.converged)
        self.assertEqual(fit.cluster_count, 80)
        test = wald_block_test(fit, [1])
        self.assertGreaterEqual(test["p_value"], 0)
        self.assertLessEqual(test["p_value"], 1)

    def test_frozen_fallback_triggers_are_exercised(self) -> None:
        with self.assertRaises(AnalysisFallback) as separation:
            fit_cluster_robust_logistic(
                [[1, 0], [1, 1], [1, 0], [1, 1]],
                [1, 1, 1, 1],
                [0, 1, 2, 3],
            )
        self.assertEqual(separation.exception.code, "separation")

        with self.assertRaises(AnalysisFallback) as singular:
            fit_cluster_robust_logistic(
                [[1, 1], [1, 1], [1, 1], [1, 1]],
                [0, 1, 0, 1],
                [0, 1, 2, 3],
            )
        self.assertEqual(singular.exception.code, "singular-design")

        with self.assertRaises(AnalysisFallback) as nonconvergence:
            fit_cluster_robust_logistic(
                [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1]],
                [0, 1, 0, 1, 1, 0],
                [0, 1, 2, 3, 4, 5],
                max_iterations=1,
                tolerance=1e-30,
            )
        self.assertEqual(nonconvergence.exception.code, "non-convergence")

    def test_power_grid_is_deterministic(self) -> None:
        kwargs = {
            "task_counts": [64, 120],
            "effects": [0.05, 0.1],
            "baseline_probability": 0.2,
            "discordance_probability": 0.2,
            "simulations": 100,
            "alpha": 0.05,
            "target_power": 0.8,
            "seed": 20260806,
        }
        self.assertEqual(
            minimum_detectable_effect_grid(**kwargs),
            minimum_detectable_effect_grid(**kwargs),
        )


if __name__ == "__main__":
    unittest.main()
