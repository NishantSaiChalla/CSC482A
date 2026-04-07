"""Evaluation metrics: truthfulness, mutual information, score variance, equilibrium gap."""

import numpy as np
from typing import List, Tuple
from environment import PopulationState, SimConfig
from scoring import log_score_payment, compute_equilibrium_gap


def truthfulness_rate(pop_state: PopulationState) -> float:
    """Fraction of workers whose report matches their private signal."""
    return float(np.mean(pop_state.reports == pop_state.signals))


def mutual_information(pop_states: List[PopulationState], cfg: SimConfig) -> float:
    """
    I(majority_report; theta) estimated from the 2x2 joint distribution
    P(r_bar, theta) across rounds, where r_bar = round(mean(reports)).
    """
    if len(pop_states) == 0:
        return 0.0

    joint = np.zeros((2, 2))
    for ps in pop_states:
        r_bar = int(np.mean(ps.reports) >= 0.5)
        joint[r_bar, ps.theta] += 1.0

    joint /= joint.sum()
    p_r = joint.sum(axis=1)
    p_t = joint.sum(axis=0)

    mi = 0.0
    for r in range(2):
        for t in range(2):
            if joint[r, t] > 1e-15 and p_r[r] > 1e-15 and p_t[t] > 1e-15:
                mi += joint[r, t] * np.log(joint[r, t] / (p_r[r] * p_t[t]))

    return float(max(mi, 0.0))


def score_variance(pop_state: PopulationState, cfg: SimConfig,
                   rng: np.random.Generator) -> float:
    """Variance of log-score payments across workers in one round."""
    N = cfg.N
    payments = np.zeros(N)
    for i in range(N):
        candidates = np.arange(N)
        candidates = candidates[candidates != i]
        ref_idx = rng.choice(candidates)
        payments[i] = log_score_payment(
            int(pop_state.reports[i]), int(pop_state.reports[ref_idx]), cfg
        )
    return float(np.var(payments))


def compute_all_payments(pop_state: PopulationState, cfg: SimConfig,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Log-score payments split by worker type. Returns (honest_payments, llm_payments)."""
    N = cfg.N
    honest_payments, llm_payments = [], []

    for i in range(N):
        candidates = np.arange(N)
        candidates = candidates[candidates != i]
        ref_idx = rng.choice(candidates)
        payment = log_score_payment(
            int(pop_state.reports[i]), int(pop_state.reports[ref_idx]), cfg
        )
        if pop_state.worker_types[i] == 0:
            honest_payments.append(payment)
        else:
            llm_payments.append(payment)

    return np.array(honest_payments), np.array(llm_payments)


def equilibrium_gap_empirical(pop_states: List[PopulationState], cfg: SimConfig,
                               rng: np.random.Generator) -> Tuple[float, float]:
    """Mean payments for honest vs LLM workers across all rounds. Returns (honest, llm)."""
    all_honest, all_llm = [], []
    for ps in pop_states:
        h, l = compute_all_payments(ps, cfg, rng)
        all_honest.extend(h.tolist())
        all_llm.extend(l.tolist())

    honest_mean = float(np.mean(all_honest)) if all_honest else float('nan')
    llm_mean = float(np.mean(all_llm)) if all_llm else float('nan')
    return honest_mean, llm_mean


def compute_metrics_for_rounds(pop_states: List[PopulationState], cfg: SimConfig,
                                rng: np.random.Generator) -> dict:
    """All four metrics averaged over a list of completed rounds."""
    tr = float(np.mean([truthfulness_rate(ps) for ps in pop_states]))
    mi = mutual_information(pop_states, cfg)
    sv = float(np.mean([score_variance(ps, cfg, rng) for ps in pop_states]))
    h_pay, l_pay = equilibrium_gap_empirical(pop_states, cfg, rng)
    gap = (h_pay - l_pay) if not (np.isnan(h_pay) or np.isnan(l_pay)) else float('nan')

    return {
        'truthfulness_rate': tr,
        'mutual_information': mi,
        'score_variance': sv,
        'honest_payment': h_pay,
        'llm_payment': l_pay,
        'equilibrium_gap': gap,
    }
