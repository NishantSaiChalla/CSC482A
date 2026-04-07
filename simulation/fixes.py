"""Three robustness modifications to the baseline peer prediction scoring rule."""

import numpy as np
from environment import PopulationState, SimConfig
from scoring import log_score_payment
from typing import List, Optional


def baseline_payments(pop_state: PopulationState, cfg: SimConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """Standard log peer prediction: each worker scored against one random reference."""
    N = cfg.N
    payments = np.zeros(N)
    for i in range(N):
        candidates = np.delete(np.arange(N), i)
        ref_idx = rng.choice(candidates)
        payments[i] = log_score_payment(
            int(pop_state.reports[i]), int(pop_state.reports[ref_idx]), cfg
        )
    return payments


def fix_a_multipeer(pop_state: PopulationState, cfg: SimConfig,
                    rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """Fix A: average log score over k reference workers to reduce payment variance."""
    N = cfg.N
    payments = np.zeros(N)
    for i in range(N):
        candidates = np.delete(np.arange(N), i)
        refs = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        reported = int(pop_state.reports[i])
        scores = [log_score_payment(reported, int(pop_state.reports[r]), cfg) for r in refs]
        payments[i] = float(np.mean(scores))
    return payments


def fix_b_popularity_penalty(pop_state: PopulationState, cfg: SimConfig,
                              rng: np.random.Generator,
                              penalty_weight: float = 0.5) -> np.ndarray:
    """
    Fix B: tau_B(i) = tau*(i) - penalty_weight * P(reports = a_i)

    Penalizes reporting the popular answer, directly countering consensus-seeking.
    """
    N = cfg.N
    payments = np.zeros(N)
    report_freq = np.bincount(pop_state.reports, minlength=2) / N

    for i in range(N):
        candidates = np.delete(np.arange(N), i)
        ref_idx = rng.choice(candidates)
        reported = int(pop_state.reports[i])
        base_payment = log_score_payment(reported, int(pop_state.reports[ref_idx]), cfg)
        payments[i] = base_payment - penalty_weight * report_freq[reported]

    return payments


def fix_c_variance_aware(pop_states_history: List[PopulationState],
                          current_pop_state: PopulationState,
                          cfg: SimConfig,
                          rng: np.random.Generator,
                          min_rounds: int = 10,
                          variance_threshold: float = 0.1) -> np.ndarray:
    """
    Fix C: weight reference workers by their historical report variance.

    LLM workers' reports have near-zero variance across rounds (always consensus);
    honest workers' reports vary with the signal. Falls back to uniform weighting
    until min_rounds of history are available.
    """
    N = cfg.N
    payments = np.zeros(N)

    if len(pop_states_history) >= min_rounds:
        historical_reports = np.stack([ps.reports for ps in pop_states_history[-min_rounds:]])
        worker_variances = np.var(historical_reports, axis=0)
        weights = np.maximum(worker_variances, variance_threshold)
    else:
        weights = np.ones(N)

    for i in range(N):
        reported = int(current_pop_state.reports[i])
        candidates = np.delete(np.arange(N), i)
        candidate_weights = weights[candidates]
        total_weight = candidate_weights.sum()
        probs = (candidate_weights / total_weight if total_weight > 1e-15
                 else np.ones(len(candidates)) / len(candidates))
        ref_idx = candidates[rng.choice(len(candidates), p=probs)]
        payments[i] = log_score_payment(
            reported, int(current_pop_state.reports[ref_idx]), cfg
        )

    return payments
