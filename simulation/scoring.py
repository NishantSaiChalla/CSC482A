"""Log peer prediction scoring rule and equilibrium analysis."""

import numpy as np
from typing import Tuple
from environment import SimConfig, signal_likelihood


def bayesian_update(s: int, cfg: SimConfig) -> np.ndarray:
    """
    Posterior P(theta | s) via Bayes' rule.
    Returns [p(theta=0|s), p(theta=1|s)].
    """
    prior = np.array([1.0 - cfg.prior, cfg.prior])
    likelihoods = np.array([signal_likelihood(s, 0, cfg.q),
                             signal_likelihood(s, 1, cfg.q)])
    unnorm = likelihoods * prior
    return unnorm / unnorm.sum()


def predict_reference_report(s_i: int, cfg: SimConfig) -> np.ndarray:
    """
    g(sj | si): predicted distribution over reference worker's report.

    Integrates over theta using the posterior from s_i:
      g(sj=1 | si) = sum_theta P(sj=1 | theta) * p(theta | si)

    Returns [g(sj=0|si), g(sj=1|si)].
    """
    posterior = bayesian_update(s_i, cfg)
    p_sj1_given_theta = np.array([signal_likelihood(1, 0, cfg.q),
                                   signal_likelihood(1, 1, cfg.q)])
    g_1 = np.dot(p_sj1_given_theta, posterior)
    return np.array([1.0 - g_1, g_1])


def log_score_payment(reported: int, reference_report: int, cfg: SimConfig) -> float:
    """tau*(i) = ln g(a_ref | a_i): log probability assigned to reference's actual report."""
    g = predict_reference_report(reported, cfg)
    prob = np.clip(g[reference_report], 1e-15, 1.0)
    return float(np.log(prob))


def expected_payment_honest(s_i: int, cfg: SimConfig) -> float:
    """
    E[tau* | honest, signal=s_i] = sum_sj g(sj|si) * ln g(sj|si)
    Negative entropy of the prediction distribution.
    """
    g = predict_reference_report(s_i, cfg)
    return sum(p * np.log(p) for p in g if p > 1e-15)


def expected_payment_consensus(s_i: int, consensus_prob: float, cfg: SimConfig) -> float:
    """
    E[tau* | LLM, signal=s_i]: LLM reports r ~ Bernoulli(consensus_prob),
    scored against a reference whose distribution is g(sj | s_i).

    E = sum_r P(r|LLM) * sum_sj g(sj|si) * ln g(sj|r)
    """
    p_report = np.array([1.0 - consensus_prob, consensus_prob])
    g_reference = predict_reference_report(s_i, cfg)

    expected = 0.0
    for r in [0, 1]:
        if p_report[r] < 1e-15:
            continue
        g_pred = predict_reference_report(r, cfg)
        inner = sum(g_reference[sj] * np.log(g_pred[sj])
                    for sj in [0, 1]
                    if g_reference[sj] > 1e-15 and g_pred[sj] > 1e-15)
        expected += p_report[r] * inner
    return expected


def mixed_reference_distribution(s_i: int, alpha: float,
                                   consensus_prob: float,
                                   cfg: SimConfig) -> np.ndarray:
    """
    g_mixed(sj | si) = (1-alpha)*g_honest(sj|si) + alpha*[1-cp, cp]

    Reference pool is a mixture: honest workers use their signal,
    LLM workers report from Bernoulli(consensus_prob).
    """
    g_honest = predict_reference_report(s_i, cfg)
    llm_dist = np.array([1.0 - consensus_prob, consensus_prob])
    return (1.0 - alpha) * g_honest + alpha * llm_dist


def expected_payment_honest_mixed(s_i: int, alpha: float,
                                   consensus_prob: float,
                                   cfg: SimConfig) -> float:
    """E[payment | honest reporter, mixed reference]: uses g_honest for prediction, g_mixed as reference."""
    g_mixed = mixed_reference_distribution(s_i, alpha, consensus_prob, cfg)
    g_pred = predict_reference_report(s_i, cfg)
    return sum(g_mixed[sj] * np.log(g_pred[sj])
               for sj in [0, 1]
               if g_mixed[sj] > 1e-15 and g_pred[sj] > 1e-15)


def expected_payment_consensus_mixed(s_i: int, consensus_prob: float,
                                      alpha: float, cfg: SimConfig) -> float:
    """E[payment | LLM reporter, mixed reference]: LLM submits r ~ Bernoulli(cp), scored against g_mixed."""
    p_report = np.array([1.0 - consensus_prob, consensus_prob])
    g_mixed = mixed_reference_distribution(s_i, alpha, consensus_prob, cfg)

    expected = 0.0
    for r in [0, 1]:
        if p_report[r] < 1e-15:
            continue
        g_pred = predict_reference_report(r, cfg)
        inner = sum(g_mixed[sj] * np.log(g_pred[sj])
                    for sj in [0, 1]
                    if g_mixed[sj] > 1e-15 and g_pred[sj] > 1e-15)
        expected += p_report[r] * inner
    return expected


def compute_equilibrium_gap(alpha: float, cfg: SimConfig) -> Tuple[float, float]:
    """
    E[honest payment] and E[consensus payment] under a mixed reference population,
    averaged over signals. Gap > 0 means honest reporting is individually rational.
    """
    from environment import compute_consensus_prob
    consensus_prob = compute_consensus_prob(cfg)

    honest_pay = 0.0
    consensus_pay = 0.0
    for s in [0, 1]:
        p_s = (signal_likelihood(s, 1, cfg.q) * cfg.prior
               + signal_likelihood(s, 0, cfg.q) * (1.0 - cfg.prior))
        honest_pay += p_s * expected_payment_honest_mixed(s, alpha, consensus_prob, cfg)
        consensus_pay += p_s * expected_payment_consensus_mixed(s, consensus_prob, alpha, cfg)

    return float(honest_pay), float(consensus_pay)
