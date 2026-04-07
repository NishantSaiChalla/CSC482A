"""
fixes.py — Three robustness modifications to the peer prediction mechanism.

The baseline mechanism degrades when LLM workers dominate because their
consensus-seeking reports inflate agreement without increasing information.
Three fixes are implemented here:

Fix A — Multi-peer scoring: Average the log score over k reference workers.
  Rationale: More reference workers = more stable estimate of the reference
  distribution. Reduces noise but doesn't directly address the LLM problem.
  Works best when LLM workers are a minority (reduces variance for honest workers).

Fix B — Popularity penalty: Subtract a penalty proportional to how common
  the reported answer is. If everyone reports 1, reporting 1 is penalized.
  Rationale: Directly counters the consensus-seeking behavior by making the
  "popular" answer less profitable to report.

Fix C — Variance-aware scoring: Track per-worker report variance across
  multiple rounds. Down-weight the reference value of low-variance workers
  (suspiciously consistent = likely LLM).
  Rationale: Honest workers' reports vary with the signal; LLM workers always
  report the same consensus answer, so their variance is near 0.

Each fix is implemented as a function that takes a PopulationState and returns
an array of modified payments (one per worker), replacing the baseline
log_score_payment.
"""

import numpy as np
from environment import PopulationState, SimConfig
from scoring import log_score_payment, predict_reference_report
from typing import List, Optional


def fix_a_multipeer(pop_state: PopulationState, cfg: SimConfig,
                    rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """
    Fix A: Multi-peer scoring — average log score over k reference workers.

    For each worker i, draw k reference workers (without replacement) and
    average their log-score payments. This reduces payment variance compared
    to the baseline (k=1) without changing the expected payment structure.

    Under the baseline, a single bad reference draw can swing the payment
    significantly. With k>1 references, honest workers get more stable rewards,
    while LLM workers' advantage from inflated agreement is not meaningfully
    reduced (they still benefit from consensus).

    Parameters
    ----------
    pop_state : PopulationState
        Population state with reports filled in
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For selecting reference workers
    k : int
        Number of reference workers to average over (default 3)

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing modified payments for each worker
    """
    N = cfg.N
    payments = np.zeros(N)

    for i in range(N):
        # Sample k reference workers without replacement, excluding i
        candidates = np.delete(np.arange(N), i)
        refs = rng.choice(candidates, size=min(k, len(candidates)), replace=False)

        reported = int(pop_state.reports[i])

        # Average log score over all k references
        scores = [log_score_payment(reported, int(pop_state.reports[r]), cfg)
                  for r in refs]
        payments[i] = float(np.mean(scores))

    return payments


def fix_b_popularity_penalty(pop_state: PopulationState, cfg: SimConfig,
                              rng: np.random.Generator,
                              penalty_weight: float = 0.5) -> np.ndarray:
    """
    Fix B: Popularity penalty — subtract a penalty for reporting the modal answer.

    The penalty is proportional to how common the reported answer is in the
    current population. If the report matches the mode, the worker is penalized;
    if it's the minority answer, they receive a bonus.

    Modified payment:
      τ_B(i) = τ*(i) - penalty_weight * P(reports = aⁱ)

    where P(reports = aⁱ) is the empirical fraction of workers who submitted
    the same report as worker i.

    This directly counteracts consensus-seeking: an LLM worker reports the modal
    answer and gets penalized exactly when everyone else does the same, reducing
    the attractiveness of consensus reporting.

    Parameters
    ----------
    pop_state : PopulationState
        Population state with reports filled in
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For selecting reference workers
    penalty_weight : float
        Strength of the popularity penalty (default 0.5)

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing penalized payments for each worker
    """
    N = cfg.N
    payments = np.zeros(N)

    # Compute fraction of workers who reported each value
    report_freq = np.bincount(pop_state.reports, minlength=2) / N  # [P(r=0), P(r=1)]

    for i in range(N):
        # Baseline log score with random reference
        candidates = np.delete(np.arange(N), i)
        ref_idx = rng.choice(candidates)
        reported = int(pop_state.reports[i])
        reference_report = int(pop_state.reports[ref_idx])

        base_payment = log_score_payment(reported, reference_report, cfg)

        # Penalty proportional to popularity of the reported answer
        popularity = report_freq[reported]
        penalty = penalty_weight * popularity

        payments[i] = base_payment - penalty

    return payments


def fix_c_variance_aware(pop_states_history: List[PopulationState],
                          current_pop_state: PopulationState,
                          cfg: SimConfig,
                          rng: np.random.Generator,
                          min_rounds: int = 10,
                          variance_threshold: float = 0.1) -> np.ndarray:
    """
    Fix C: Variance-aware scoring — down-weight low-variance reference workers.

    LLM workers report the same consensus answer repeatedly, so their per-round
    variance is near zero. Honest workers' signals flip with the true state,
    giving them higher variance across rounds.

    For each worker i, we estimate their historical report variance. When
    computing worker i's payment, we weight reference workers by their variance
    (low variance = less reliable reference = lower weight).

    Modified scoring: instead of picking a uniform random reference, we pick
    references with probability proportional to their variance.

    If fewer than min_rounds of history are available, fall back to baseline
    (uniform reference selection).

    Parameters
    ----------
    pop_states_history : List[PopulationState]
        Previous simulation rounds (for estimating variance)
    current_pop_state : PopulationState
        Current round's population state
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For selecting reference workers
    min_rounds : int
        Minimum rounds of history needed before applying variance weighting
    variance_threshold : float
        Minimum variance weight to avoid completely excluding workers

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing variance-weighted payments
    """
    N = cfg.N
    payments = np.zeros(N)

    # Estimate per-worker report variance from history
    use_variance_weighting = len(pop_states_history) >= min_rounds

    if use_variance_weighting:
        # Stack reports across historical rounds: shape (rounds, N)
        historical_reports = np.stack([ps.reports for ps in pop_states_history[-min_rounds:]])
        # Variance of reports for each worker across rounds
        worker_variances = np.var(historical_reports, axis=0)
        # Clip to minimum to avoid zero weights
        weights = np.maximum(worker_variances, variance_threshold)
    else:
        # No history: use uniform weights (baseline behavior)
        weights = np.ones(N)

    for i in range(N):
        reported = int(current_pop_state.reports[i])

        # Exclude worker i from candidate references
        candidates = np.delete(np.arange(N), i)
        candidate_weights = weights[candidates]

        # Normalize weights
        total_weight = candidate_weights.sum()
        if total_weight < 1e-15:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = candidate_weights / total_weight

        # Sample reference weighted by variance (high variance = more trustworthy)
        ref_idx = candidates[rng.choice(len(candidates), p=probs)]

        reference_report = int(current_pop_state.reports[ref_idx])
        payments[i] = log_score_payment(reported, reference_report, cfg)

    return payments


def baseline_payments(pop_state: PopulationState, cfg: SimConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Baseline scoring: each worker scored against one random reference.

    This is the standard log peer prediction rule with no modifications.
    Used as the control condition in all comparisons.

    Parameters
    ----------
    pop_state : PopulationState
        Population state with reports filled in
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For selecting reference workers

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing log-score payments
    """
    N = cfg.N
    payments = np.zeros(N)

    for i in range(N):
        candidates = np.delete(np.arange(N), i)
        ref_idx = rng.choice(candidates)
        reported = int(pop_state.reports[i])
        reference_report = int(pop_state.reports[ref_idx])
        payments[i] = log_score_payment(reported, reference_report, cfg)

    return payments
