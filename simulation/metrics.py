"""
metrics.py — Evaluation metrics for the peer prediction mechanism.

Four metrics are tracked as α (fraction of LLM workers) varies:

1. Truthfulness rate: fraction of workers reporting their true signal.
   Under honest workers this is 1.0. LLM workers report consensus, so
   their truthfulness depends on whether consensus matches their signal.

2. Mutual information I(reports; θ): how much information the aggregate
   report distribution contains about the true state. This is the key
   quantity — high agreement with low MI signals mechanism degradation.

3. Score variance: variance of per-worker log-score payments. LLM workers
   produce highly correlated reports, which should reduce variance.

4. Equilibrium robustness: comparison of E[payment|honest] vs E[payment|LLM].
   When LLM payment exceeds honest payment, the truthful equilibrium has collapsed.
   The crossover point is α*.

All metrics operate on a single simulation round's PopulationState.
The experiment runner averages them over M rounds.
"""

import numpy as np
from typing import List, Tuple
from scipy.stats import entropy as scipy_entropy
from environment import PopulationState, SimConfig
from scoring import log_score_payment, compute_equilibrium_gap


def truthfulness_rate(pop_state: PopulationState) -> float:
    """
    Compute fraction of workers whose report matches their private signal.

    For honest workers this is always 1.0 by definition. For LLM workers
    this depends on whether the consensus answer matches their signal — with
    symmetric prior and q=0.7, this happens roughly 50-70% of the time.

    Parameters
    ----------
    pop_state : PopulationState
        Population state with reports and signals filled in

    Returns
    -------
    float
        Fraction in [0, 1] of workers where report == signal
    """
    return float(np.mean(pop_state.reports == pop_state.signals))


def mutual_information(pop_states: List[PopulationState], cfg: SimConfig) -> float:
    """
    Estimate I(reports; θ) from a list of simulation rounds.

    Mutual information between the aggregate report (fraction of 1s) and the
    true state θ tells us how informative the report distribution actually is.

    We estimate this as:
      I(report_fraction; θ) ≈ H(report_fraction) - H(report_fraction | θ)

    In practice, we use a simpler estimator: count the average report per θ
    value and compute MI directly from the joint distribution of
    (majority_report, θ), where majority_report = round(mean(reports)).

    The 2x2 joint distribution approach:
      P(r̄, θ) for r̄ ∈ {0,1}, θ ∈ {0,1}
      I(r̄; θ) = sum_{r̄, θ} P(r̄, θ) * log(P(r̄, θ) / (P(r̄) * P(θ)))

    Parameters
    ----------
    pop_states : List[PopulationState]
        List of simulation rounds
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    float
        Estimated mutual information in nats (use / log(2) for bits)
    """
    if len(pop_states) == 0:
        return 0.0

    # Build joint distribution: P(majority_report=r̄, θ=t)
    joint = np.zeros((2, 2))  # joint[r_bar, theta]

    for ps in pop_states:
        theta = ps.theta
        # Use majority vote as the aggregate report
        r_bar = int(np.mean(ps.reports) >= 0.5)
        joint[r_bar, theta] += 1.0

    joint /= joint.sum()

    # Marginals
    p_r = joint.sum(axis=1)  # P(r̄)
    p_t = joint.sum(axis=0)  # P(θ)

    # MI = sum P(r̄,θ) * log(P(r̄,θ) / (P(r̄) * P(θ)))
    mi = 0.0
    for r in range(2):
        for t in range(2):
            if joint[r, t] > 1e-15 and p_r[r] > 1e-15 and p_t[t] > 1e-15:
                mi += joint[r, t] * np.log(joint[r, t] / (p_r[r] * p_t[t]))

    return float(max(mi, 0.0))  # clip numerical noise to 0


def score_variance(pop_state: PopulationState, cfg: SimConfig,
                   rng: np.random.Generator) -> float:
    """
    Compute variance of log-score payments across all workers in one round.

    Each worker i is scored against a randomly chosen reference worker r(i).
    The variance measures how spread out the payments are. Under LLM workers,
    high agreement leads to similar payments (low variance) even when reports
    contain little information.

    Parameters
    ----------
    pop_state : PopulationState
        Population state with reports filled in
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        Random number generator for pairing workers

    Returns
    -------
    float
        Variance of log-score payments across workers
    """
    N = cfg.N
    payments = np.zeros(N)

    for i in range(N):
        # Pick a reference worker different from i
        candidates = np.arange(N)
        candidates = candidates[candidates != i]
        ref_idx = rng.choice(candidates)

        reported = int(pop_state.reports[i])
        reference_report = int(pop_state.reports[ref_idx])

        payments[i] = log_score_payment(reported, reference_report, cfg)

    return float(np.var(payments))


def compute_all_payments(pop_state: PopulationState, cfg: SimConfig,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute log-score payments separately for honest and LLM workers.

    This is used to directly compare which strategy is more profitable —
    the key quantity for detecting equilibrium collapse.

    Parameters
    ----------
    pop_state : PopulationState
        Population state with reports and worker_types filled in
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For reference worker pairing

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (honest_payments, llm_payments) — arrays of log-score payments,
        one per honest/LLM worker respectively. Empty arrays if no workers
        of that type exist.
    """
    N = cfg.N
    honest_payments = []
    llm_payments = []

    for i in range(N):
        # Pick a reference worker different from i
        candidates = np.arange(N)
        candidates = candidates[candidates != i]
        ref_idx = rng.choice(candidates)

        reported = int(pop_state.reports[i])
        reference_report = int(pop_state.reports[ref_idx])
        payment = log_score_payment(reported, reference_report, cfg)

        if pop_state.worker_types[i] == 0:
            honest_payments.append(payment)
        else:
            llm_payments.append(payment)

    return np.array(honest_payments), np.array(llm_payments)


def equilibrium_gap_empirical(pop_states: List[PopulationState], cfg: SimConfig,
                               rng: np.random.Generator) -> Tuple[float, float]:
    """
    Empirically estimate average payments for honest vs LLM workers.

    Averages over all simulation rounds. If no LLM workers exist (α=0),
    returns (honest_mean, NaN).

    Parameters
    ----------
    pop_states : List[PopulationState]
        List of simulation rounds (all with same α)
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For reference pairing

    Returns
    -------
    Tuple[float, float]
        (mean_honest_payment, mean_llm_payment) across all rounds
    """
    all_honest = []
    all_llm = []

    for ps in pop_states:
        h_pay, l_pay = compute_all_payments(ps, cfg, rng)
        all_honest.extend(h_pay.tolist())
        all_llm.extend(l_pay.tolist())

    honest_mean = float(np.mean(all_honest)) if all_honest else float('nan')
    llm_mean = float(np.mean(all_llm)) if all_llm else float('nan')

    return honest_mean, llm_mean


def compute_metrics_for_rounds(pop_states: List[PopulationState], cfg: SimConfig,
                                rng: np.random.Generator) -> dict:
    """
    Compute all four metrics averaged over a list of simulation rounds.

    This is the main entry point called by experiments.py for each (α, fix)
    combination.

    Parameters
    ----------
    pop_states : List[PopulationState]
        List of completed simulation rounds
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        Random number generator (for score variance pairing)

    Returns
    -------
    dict with keys:
        truthfulness_rate : float
        mutual_information : float  (in nats)
        score_variance : float
        honest_payment : float
        llm_payment : float
        equilibrium_gap : float   (honest - llm, positive = mechanism working)
    """
    # Truthfulness rate: average over rounds
    tr = float(np.mean([truthfulness_rate(ps) for ps in pop_states]))

    # Mutual information: computed from joint distribution across all rounds
    mi = mutual_information(pop_states, cfg)

    # Score variance: average over rounds
    sv = float(np.mean([score_variance(ps, cfg, rng) for ps in pop_states]))

    # Equilibrium gap: empirical payments per worker type
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
