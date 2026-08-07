"""Frozen statistical primitives for ARC-REBench analysis plan v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import random
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import chi2


class AnalysisFallback(RuntimeError):
    """Raised when the prespecified clustered model must use its fallback."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ClusterRobustLogisticFit:
    coefficients: list[float]
    covariance: list[list[float]]
    converged: bool
    iterations: int
    observation_count: int
    cluster_count: int
    design_rank: int
    condition_number: float
    working_correlation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    if not p_values:
        return []
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("p-values must be finite numbers in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_binary_exact_pvalue(first: Sequence[int], second: Sequence[int]) -> float:
    """Two-sided exact paired randomization/McNemar p-value."""

    if len(first) != len(second) or not first:
        raise ValueError("paired binary samples must have equal positive length")
    if any(value not in {0, 1} for value in itertools.chain(first, second)):
        raise ValueError("paired binary samples must contain only zero and one")
    first_only = sum(a == 1 and b == 0 for a, b in zip(first, second))
    second_only = sum(a == 0 and b == 1 for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower = min(first_only, second_only)
    probability = sum(
        math.comb(discordant, successes) for successes in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_randomization_test(
    cluster_differences: Sequence[float],
    *,
    seed: int,
    monte_carlo_resamples: int,
    exact_nonzero_limit: int = 20,
) -> dict[str, object]:
    """Two-sided sign-flip test over base-task cluster effects."""

    values = [float(value) for value in cluster_differences]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cluster differences must be finite and nonempty")
    nonzero = [value for value in values if value != 0]
    observed = abs(sum(nonzero))
    if not nonzero:
        return {
            "method": "exact-enumeration",
            "cluster_count": len(values),
            "nonzero_cluster_count": 0,
            "statistic_abs_sum": 0.0,
            "p_value": 1.0,
            "resamples": 1,
        }
    tolerance = 1e-12
    if len(nonzero) <= exact_nonzero_limit:
        total = 1 << len(nonzero)
        extreme = 0
        for mask in range(total):
            permuted = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(nonzero)
            )
            extreme += abs(permuted) + tolerance >= observed
        return {
            "method": "exact-enumeration",
            "cluster_count": len(values),
            "nonzero_cluster_count": len(nonzero),
            "statistic_abs_sum": observed,
            "p_value": extreme / total,
            "resamples": total,
        }
    if type(monte_carlo_resamples) is not int or monte_carlo_resamples < 1:
        raise ValueError("monte_carlo_resamples must be a positive integer")
    rng = random.Random(seed)
    extreme = 0
    for _ in range(monte_carlo_resamples):
        permuted = sum(
            value if rng.getrandbits(1) else -value for value in nonzero
        )
        extreme += abs(permuted) + tolerance >= observed
    return {
        "method": "monte-carlo-sign-flip",
        "cluster_count": len(values),
        "nonzero_cluster_count": len(nonzero),
        "statistic_abs_sum": observed,
        "p_value": (extreme + 1) / (monte_carlo_resamples + 1),
        "resamples": monte_carlo_resamples,
    }


def cluster_bootstrap_mean_interval(
    cluster_values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Percentile interval resampling whole base-task clusters."""

    values = [float(value) for value in cluster_values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cluster values must be finite and nonempty")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("resamples must be an integer of at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    alpha = 1.0 - confidence
    lower_index = max(0, math.floor((alpha / 2) * resamples))
    upper_index = min(resamples - 1, math.ceil((1 - alpha / 2) * resamples) - 1)
    return {
        "cluster_count": count,
        "resamples": resamples,
        "confidence": confidence,
        "estimate": sum(values) / count,
        "lower": samples[lower_index],
        "upper": samples[upper_index],
    }


def fit_cluster_robust_logistic(
    design: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    clusters: Sequence[str | int],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
    condition_limit: float = 1e12,
) -> ClusterRobustLogisticFit:
    """Fit independence-working-correlation logistic GEE with cluster sandwich.

    This is the prespecified implementation for H1/H2.  Algebraically it is a
    logistic mean model fitted by IRLS with a base-task cluster-robust sandwich.
    The plan freezes explicit fallback triggers rather than repairing unstable
    fits after results are seen.
    """

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    cluster_values = np.asarray(clusters, dtype=object)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("design must be a nonempty two-dimensional matrix")
    if y.shape != (x.shape[0],) or cluster_values.shape != (x.shape[0],):
        raise ValueError("outcomes and clusters must match design rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("design and outcomes must be finite")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("outcomes must be binary")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if tolerance <= 0 or condition_limit <= 1:
        raise ValueError("tolerance and condition limit must be positive")
    if np.all(y == y[0]):
        raise AnalysisFallback("separation", "all simulated outcomes are identical")
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise AnalysisFallback("singular-design", "design matrix is rank deficient")
    unique_clusters = list(dict.fromkeys(cluster_values.tolist()))
    if len(unique_clusters) <= x.shape[1]:
        raise AnalysisFallback(
            "insufficient-clusters",
            "cluster count must exceed the number of fitted coefficients",
        )

    beta = np.zeros(x.shape[1], dtype=float)
    converged = False
    condition_number = math.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        linear = np.clip(x @ beta, -30.0, 30.0)
        mean = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(mean * (1.0 - mean), 1e-12)
        information = x.T @ (weights[:, None] * x)
        condition_number = float(np.linalg.cond(information))
        if not math.isfinite(condition_number) or condition_number > condition_limit:
            raise AnalysisFallback(
                "singular-covariance",
                f"information condition number {condition_number!r} exceeds limit",
            )
        try:
            delta = np.linalg.solve(information, x.T @ (y - mean))
        except np.linalg.LinAlgError as error:
            raise AnalysisFallback(
                "singular-covariance", "information matrix is not invertible"
            ) from error
        beta = beta + delta
        if float(np.max(np.abs(beta))) > 30:
            raise AnalysisFallback(
                "separation", "coefficient magnitude exceeded the frozen limit"
            )
        if float(np.max(np.abs(delta))) <= tolerance:
            converged = True
            break
    if not converged:
        raise AnalysisFallback(
            "non-convergence",
            f"IRLS did not converge within {max_iterations} iterations",
        )

    linear = np.clip(x @ beta, -30.0, 30.0)
    mean = 1.0 / (1.0 + np.exp(-linear))
    weights = np.maximum(mean * (1.0 - mean), 1e-12)
    information = x.T @ (weights[:, None] * x)
    bread = np.linalg.inv(information)
    meat = np.zeros_like(information)
    for cluster in unique_clusters:
        mask = cluster_values == cluster
        score = x[mask].T @ (y[mask] - mean[mask])
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    observations, coefficients = x.shape
    cluster_count = len(unique_clusters)
    correction = (cluster_count / (cluster_count - 1)) * (
        (observations - 1) / (observations - coefficients)
    )
    covariance *= correction
    if not np.isfinite(covariance).all():
        raise AnalysisFallback("singular-covariance", "sandwich covariance is nonfinite")
    return ClusterRobustLogisticFit(
        coefficients=beta.tolist(),
        covariance=covariance.tolist(),
        converged=True,
        iterations=iterations,
        observation_count=observations,
        cluster_count=cluster_count,
        design_rank=rank,
        condition_number=condition_number,
        working_correlation="independence",
    )


def wald_block_test(
    fit: ClusterRobustLogisticFit, coefficient_indices: Sequence[int]
) -> dict[str, object]:
    indices = list(coefficient_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("coefficient indices must be nonempty and unique")
    beta = np.asarray(fit.coefficients, dtype=float)
    covariance = np.asarray(fit.covariance, dtype=float)
    if any(index < 0 or index >= len(beta) for index in indices):
        raise ValueError("coefficient index out of range")
    selected = beta[indices]
    selected_covariance = covariance[np.ix_(indices, indices)]
    try:
        statistic = float(selected.T @ np.linalg.solve(selected_covariance, selected))
    except np.linalg.LinAlgError as error:
        raise AnalysisFallback(
            "singular-covariance", "selected Wald covariance is not invertible"
        ) from error
    if not math.isfinite(statistic) or statistic < 0:
        raise AnalysisFallback("singular-covariance", "Wald statistic is invalid")
    degrees = len(indices)
    return {
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": float(chi2.sf(statistic, degrees)),
    }


def simulate_paired_binary_power(
    *,
    task_count: int,
    baseline_probability: float,
    discordance_probability: float,
    effect: float,
    simulations: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    """Power of the frozen exact paired fallback under explicit assumptions."""

    if type(task_count) is not int or task_count < 2:
        raise ValueError("task_count must be an integer of at least two")
    if type(simulations) is not int or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 <= baseline_probability <= 1:
        raise ValueError("baseline_probability must be in [0, 1]")
    if not 0 <= discordance_probability <= 1:
        raise ValueError("discordance_probability must be in [0, 1]")
    first_only = (discordance_probability - effect) / 2
    second_only = (discordance_probability + effect) / 2
    both = baseline_probability - first_only
    neither = 1 - both - first_only - second_only
    probabilities = [neither, second_only, first_only, both]
    if any(value < -1e-12 for value in probabilities):
        raise ValueError("effect assumptions imply an invalid paired distribution")
    probabilities = [max(0.0, value) for value in probabilities]
    cumulative = list(itertools.accumulate(probabilities))
    cumulative[-1] = 1.0
    rng = random.Random(seed)
    rejections = 0
    for _ in range(simulations):
        first: list[int] = []
        second: list[int] = []
        for _ in range(task_count):
            draw = rng.random()
            category = next(
                index for index, threshold in enumerate(cumulative) if draw <= threshold
            )
            pair = ((0, 0), (0, 1), (1, 0), (1, 1))[category]
            first.append(pair[0])
            second.append(pair[1])
        rejections += paired_binary_exact_pvalue(first, second) <= alpha
    return {
        "task_count": task_count,
        "baseline_probability": baseline_probability,
        "discordance_probability": discordance_probability,
        "effect": effect,
        "simulations": simulations,
        "alpha": alpha,
        "seed": seed,
        "rejections": rejections,
        "power": rejections / simulations,
    }


def minimum_detectable_effect_grid(
    *,
    task_counts: Iterable[int],
    effects: Sequence[float],
    baseline_probability: float,
    discordance_probability: float,
    simulations: int,
    alpha: float,
    target_power: float,
    seed: int,
) -> dict[str, object]:
    if not effects or any(effect <= 0 for effect in effects):
        raise ValueError("effects must be positive and nonempty")
    if sorted(effects) != list(effects) or len(set(effects)) != len(effects):
        raise ValueError("effects must be unique and ascending")
    if not 0 < target_power < 1:
        raise ValueError("target_power must be in (0, 1)")
    results: dict[str, object] = {}
    for task_offset, task_count in enumerate(task_counts):
        grid = []
        for effect_offset, effect in enumerate(effects):
            grid.append(
                simulate_paired_binary_power(
                    task_count=task_count,
                    baseline_probability=baseline_probability,
                    discordance_probability=discordance_probability,
                    effect=effect,
                    simulations=simulations,
                    alpha=alpha,
                    seed=seed + task_offset * 1000 + effect_offset,
                )
            )
        qualifying = [item["effect"] for item in grid if item["power"] >= target_power]
        results[str(task_count)] = {
            "grid": grid,
            "minimum_grid_effect_at_target_power": min(qualifying)
            if qualifying
            else None,
        }
    return {
        "target_power": target_power,
        "results": results,
    }
