"""
scoring.py — Bayesian update, prediction distribution, and log scoring rule.

This is the mathematical core of the peer prediction mechanism. The key insight
is that honest reporting is a strict Nash equilibrium: a worker maximizes their
expected payment by reporting their true signal, because signals are conditionally
independent given θ and the log scoring rule is strictly proper.

The three-step scoring process:
  1. Worker i observes signal sⁱ and updates beliefs: p(θ | sⁱ)
  2. Worker i uses updated beliefs to predict reference worker's report: g(sʲ | sⁱ)
  3. Worker i is paid ln g(aʳ⁽ⁱ⁾ | aⁱ) where aʳ⁽ⁱ⁾ is the reference report

Because the log scoring rule is strictly proper, the worker maximizes expected
payment by reporting p_predicted honestly — and honest reporting of signal leads
to honest prediction, making honest reporting a Nash equilibrium.
"""

import numpy as np
from typing import Tuple
from environment import SimConfig, signal_likelihood


def bayesian_update(s: int, cfg: SimConfig) -> np.ndarray:
    """
    Compute posterior P(θ | sⁱ) after observing signal s.

    Applies Bayes' rule:
      p(θ=1 | s) = P(s | θ=1) · P(θ=1) / Pr(s)
      p(θ=0 | s) = P(s | θ=0) · P(θ=0) / Pr(s)

    where Pr(s) = sum_θ P(s | θ) · P(θ)

    Parameters
    ----------
    s : int
        Observed signal (0 or 1)
    cfg : SimConfig
        Contains prior p(θ=1) and signal accuracy q

    Returns
    -------
    np.ndarray
        [p(θ=0 | s), p(θ=1 | s)] — posterior over θ given signal s
    """
    prior = np.array([1.0 - cfg.prior, cfg.prior])  # [P(θ=0), P(θ=1)]

    # Likelihood of observing s under each θ
    likelihoods = np.array([signal_likelihood(s, 0, cfg.q),
                             signal_likelihood(s, 1, cfg.q)])

    # Unnormalized posterior
    unnorm = likelihoods * prior

    # Normalize
    posterior = unnorm / unnorm.sum()

    return posterior  # [p(θ=0|s), p(θ=1|s)]


def predict_reference_report(s_i: int, cfg: SimConfig) -> np.ndarray:
    """
    Compute g(sʲ | sⁱ): predicted distribution over reference worker's report.

    Given worker i's signal sⁱ, the honest prediction of reference worker j's
    report integrates over the unknown true state:
      g(sʲ=1 | sⁱ) = sum_θ P(sʲ=1 | θ) · p(θ | sⁱ)
      g(sʲ=0 | sⁱ) = 1 - g(sʲ=1 | sⁱ)

    This is the honest prediction: given my signal, what is the probability
    that a randomly chosen other worker will report 1?

    Because signals are conditionally independent given θ, knowing sⁱ tells
    us about θ, which tells us about sʲ — but sⁱ and sʲ are not directly
    dependent (no common noise). This conditional independence is what makes
    the peer prediction mechanism work.

    Parameters
    ----------
    s_i : int
        Signal of worker i (the reporter, 0 or 1)
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    np.ndarray
        [g(sʲ=0 | sⁱ), g(sʲ=1 | sⁱ)] — predicted probability of reference report
    """
    posterior = bayesian_update(s_i, cfg)  # [p(θ=0|s_i), p(θ=1|s_i)]

    # P(sʲ=1 | θ) for each θ
    p_sj1_given_theta = np.array([signal_likelihood(1, 0, cfg.q),
                                   signal_likelihood(1, 1, cfg.q)])

    # g(sʲ=1 | sⁱ) = sum_θ P(sʲ=1|θ) * p(θ|sⁱ)
    g_1 = np.dot(p_sj1_given_theta, posterior)
    g_0 = 1.0 - g_1

    return np.array([g_0, g_1])


def log_score_payment(reported: int, reference_report: int, cfg: SimConfig) -> float:
    """
    Compute the log scoring rule payment τ*(i).

    τ*(i) = ln g(aʳ⁽ⁱ⁾ | aⁱ)

    where aⁱ is worker i's report and aʳ⁽ⁱ⁾ is the reference worker's report.

    The worker is paid based on how well their reported signal predicts the
    reference worker's report. Under honest reporting, this equals
    ln g(aʳ⁽ⁱ⁾ | sⁱ) — the log probability assigned to the reference's
    actual report.

    Parameters
    ----------
    reported : int
        Worker i's submitted report (0 or 1), used to compute g(· | reported)
    reference_report : int
        Reference worker's actual report (0 or 1) — what we're scoring against
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    float
        Log score payment (always ≤ 0, since log probabilities are ≤ 0)
    """
    g = predict_reference_report(reported, cfg)
    prob = g[reference_report]

    # Clamp to avoid log(0) — shouldn't happen with well-defined q, but safe
    prob = np.clip(prob, 1e-15, 1.0)
    return float(np.log(prob))


def expected_payment_honest(s_i: int, cfg: SimConfig) -> float:
    """
    Compute E[τ*(i) | honest reporting, signal=s_i].

    An honest worker with signal s_i reports s_i and is scored against
    the reference worker's report, which follows distribution g(sʲ | s_i).

    E[τ* | honest, s_i] = sum_{s_j} g(s_j | s_i) · ln g(s_j | s_i)
                        = -H(g(· | s_i))  [negative entropy of prediction]

    This equals the negative entropy of the honest prediction distribution.
    The more informative the signal, the tighter the prediction, the higher
    (less negative) the expected payment.

    Parameters
    ----------
    s_i : int
        Worker's true signal (0 or 1)
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    float
        Expected log score payment under honest reporting
    """
    g = predict_reference_report(s_i, cfg)  # honest prediction given s_i

    # E[ln g(sʲ | s_i)] = sum_{s_j} g(s_j | s_i) * ln g(s_j | s_i)
    expected = 0.0
    for sj in [0, 1]:
        p = g[sj]
        if p > 1e-15:
            expected += p * np.log(p)

    return expected


def expected_payment_consensus(s_i: int, consensus_prob: float, cfg: SimConfig) -> float:
    """
    Compute E[τ*(i) | consensus reporting, true signal=s_i].

    An LLM worker with true signal s_i reports 1 with probability consensus_prob
    (and 0 with probability 1-consensus_prob), regardless of their actual signal.

    The reference worker follows the true conditional distribution given the true
    state, which depends on s_i (since s_i tells us about θ).

    To compute this properly, we need to integrate over:
    1. Which report the LLM worker submits (r ∈ {0,1} with prob [1-c, c])
    2. What score they receive for that report given the reference's actual distribution

    E[τ* | consensus, s_i] = sum_r P(report=r|LLM) * E[ln g(sʲ | r) | reference dist]

    The reference worker's report distribution — what the reference actually does —
    is g(sʲ | s_i) because the reference worker is honest and their signal
    correlates with s_i through the true state θ.

    Parameters
    ----------
    s_i : int
        Worker's true (but ignored) signal
    consensus_prob : float
        P(LLM reports 1), the consensus probability used by LLM workers
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    float
        Expected log score payment under consensus (LLM) reporting
    """
    # Distribution of LLM's submitted report
    p_report = np.array([1.0 - consensus_prob, consensus_prob])  # [P(r=0), P(r=1)]

    # True distribution of reference worker's report, given s_i
    # (reference is honest, so their report distribution is g(sʲ | s_i))
    g_reference = predict_reference_report(s_i, cfg)

    # E[τ*] = sum_r P(r) * sum_{sʲ} g(sʲ | s_i) * ln g(sʲ | r)
    #       = sum_r P(r) * [g(0|s_i)*ln g(0|r) + g(1|s_i)*ln g(1|r)]
    expected = 0.0
    for r in [0, 1]:
        if p_report[r] < 1e-15:
            continue

        # g(· | r): prediction distribution IF the LLM submitted report r
        g_pred = predict_reference_report(r, cfg)

        # Expected log score under this submitted report
        inner = 0.0
        for sj in [0, 1]:
            p_ref = g_reference[sj]
            p_pred = g_pred[sj]
            if p_ref > 1e-15 and p_pred > 1e-15:
                inner += p_ref * np.log(p_pred)

        expected += p_report[r] * inner

    return expected


def compute_equilibrium_gap(alpha: float, cfg: SimConfig) -> Tuple[float, float]:
    """
    Compute expected payments for honest vs consensus strategies, averaged
    over signals with probability 0.5 each (symmetric prior).

    The 'equilibrium gap' is E[honest] - E[consensus]. When this is positive,
    honest reporting is more profitable and the mechanism is working. When it
    crosses zero, the equilibrium collapses — this is α*.

    Parameters
    ----------
    alpha : float
        Fraction of LLM workers (affects consensus_prob indirectly through
        the actual report distribution — but consensus_prob is a fixed
        model parameter here, not dependent on α)
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    Tuple[float, float]
        (avg_honest_payment, avg_consensus_payment) averaged over signals
    """
    from environment import compute_consensus_prob

    consensus_prob = compute_consensus_prob(cfg)

    # Average over both possible signal values (symmetric by design)
    honest_pay = 0.0
    consensus_pay = 0.0

    for s in [0, 1]:
        # Probability of observing signal s (marginal)
        p_s = (signal_likelihood(s, 1, cfg.q) * cfg.prior
               + signal_likelihood(s, 0, cfg.q) * (1.0 - cfg.prior))

        honest_pay += p_s * expected_payment_honest(s, cfg)
        consensus_pay += p_s * expected_payment_consensus(s, consensus_prob, cfg)

    return float(honest_pay), float(consensus_pay)
