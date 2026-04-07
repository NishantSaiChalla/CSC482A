"""
experiments.py — Experiment runners that sweep α and compare baseline vs. fixes.

This module is the orchestration layer. It:
  1. Sweeps α from 0.0 to 1.0 in steps of 0.05
  2. For each α, runs M=500 simulation rounds
  3. Computes all 4 metrics per round and averages them
  4. Repeats for: baseline, Fix A (k=3), Fix A (k=5), Fix B, Fix C
  5. Returns results as a pandas DataFrame for plotting

The key design constraint: each simulation round is independent (no sequential
dynamics), so we can generate all rounds in advance and parallelize if needed.
All randomness is seeded for reproducibility.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from environment import SimConfig, generate_round, PopulationState
from workers import populate_reports
from fixes import (baseline_payments, fix_a_multipeer,
                   fix_b_popularity_penalty, fix_c_variance_aware)
from scoring import compute_equilibrium_gap


# Default experiment parameters
ALPHA_VALUES = np.round(np.arange(0.0, 1.05, 0.05), 2)
M_ROUNDS = 500
SEED_BASE = 42


def run_single_round(cfg: SimConfig, rng: np.random.Generator,
                     llm_strength: float = 1.0) -> PopulationState:
    """
    Run one simulation round: generate state, signals, and all worker reports.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        Seeded random number generator
    llm_strength : float
        Consensus-following strength for LLM workers

    Returns
    -------
    PopulationState
        Completed round with reports filled in
    """
    pop_state = generate_round(cfg, rng)
    pop_state = populate_reports(pop_state, cfg, rng, llm_strength=llm_strength)
    return pop_state


def compute_round_metrics(pop_state: PopulationState, cfg: SimConfig,
                           rng: np.random.Generator,
                           history: Optional[List[PopulationState]] = None,
                           scoring_method: str = 'baseline',
                           fix_k: int = 3,
                           fix_penalty: float = 0.5) -> Dict:
    """
    Compute all metrics for a single round under a given scoring method.

    Parameters
    ----------
    pop_state : PopulationState
        Completed simulation round
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        For reference pairing
    history : List[PopulationState], optional
        Previous rounds (needed for Fix C)
    scoring_method : str
        One of 'baseline', 'fix_a', 'fix_b', 'fix_c'
    fix_k : int
        Number of peers for Fix A
    fix_penalty : float
        Penalty weight for Fix B

    Returns
    -------
    dict
        Per-round metrics: truthfulness_rate, score_variance, honest_payment,
        llm_payment, theta, report_mean
    """
    # Truthfulness rate
    truthfulness = float(np.mean(pop_state.reports == pop_state.signals))

    # Select scoring method and compute payments
    if scoring_method == 'baseline':
        payments = baseline_payments(pop_state, cfg, rng)
    elif scoring_method == 'fix_a':
        payments = fix_a_multipeer(pop_state, cfg, rng, k=fix_k)
    elif scoring_method == 'fix_b':
        payments = fix_b_popularity_penalty(pop_state, cfg, rng,
                                             penalty_weight=fix_penalty)
    elif scoring_method == 'fix_c':
        hist = history if history is not None else []
        payments = fix_c_variance_aware(hist, pop_state, cfg, rng)
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    # Score variance across all workers
    score_var = float(np.var(payments))

    # Separate payments by worker type
    honest_mask = pop_state.worker_types == 0
    llm_mask = pop_state.worker_types == 1

    honest_pay = float(np.mean(payments[honest_mask])) if honest_mask.any() else float('nan')
    llm_pay = float(np.mean(payments[llm_mask])) if llm_mask.any() else float('nan')

    return {
        'truthfulness_rate': truthfulness,
        'score_variance': score_var,
        'honest_payment': honest_pay,
        'llm_payment': llm_pay,
        'theta': pop_state.theta,
        'report_mean': float(np.mean(pop_state.reports)),
    }


def compute_mi_from_rounds(pop_states: List[PopulationState]) -> float:
    """
    Estimate I(majority_report; θ) from a list of rounds.

    Builds the 2x2 joint distribution and computes MI directly.

    Parameters
    ----------
    pop_states : List[PopulationState]
        Completed simulation rounds

    Returns
    -------
    float
        Mutual information in nats
    """
    joint = np.zeros((2, 2))  # joint[r_bar, theta]
    for ps in pop_states:
        theta = ps.theta
        r_bar = int(np.mean(ps.reports) >= 0.5)
        joint[r_bar, theta] += 1.0

    joint /= joint.sum()
    p_r = joint.sum(axis=1)
    p_t = joint.sum(axis=0)

    mi = 0.0
    for r in range(2):
        for t in range(2):
            if joint[r, t] > 1e-15 and p_r[r] > 1e-15 and p_t[t] > 1e-15:
                mi += joint[r, t] * np.log(joint[r, t] / (p_r[r] * p_t[t]))

    return float(max(mi, 0.0))


def run_experiment_for_alpha(alpha: float, cfg_base: SimConfig, M: int,
                              scoring_method: str = 'baseline',
                              fix_k: int = 3,
                              fix_penalty: float = 0.5,
                              llm_strength: float = 1.0,
                              seed_offset: int = 0) -> Dict:
    """
    Run M simulation rounds for a single (α, scoring_method) configuration.

    Parameters
    ----------
    alpha : float
        Fraction of LLM workers
    cfg_base : SimConfig
        Base configuration (alpha will be overridden)
    M : int
        Number of simulation rounds
    scoring_method : str
        Scoring method to use
    fix_k : int
        Peer count for Fix A
    fix_penalty : float
        Penalty weight for Fix B
    llm_strength : float
        LLM consensus following strength
    seed_offset : int
        Add to base seed for independent seeds across methods

    Returns
    -------
    dict
        Aggregated metrics averaged over M rounds:
        alpha, scoring_method, truthfulness_rate, mutual_information,
        score_variance, honest_payment, llm_payment, equilibrium_gap
    """
    # Override alpha in config
    cfg = SimConfig(N=cfg_base.N, q=cfg_base.q, prior=cfg_base.prior,
                    alpha=alpha, seed=cfg_base.seed)

    rng = np.random.default_rng(SEED_BASE + seed_offset + int(alpha * 1000))

    pop_states = []
    round_metrics = []

    for m in range(M):
        pop_state = run_single_round(cfg, rng, llm_strength=llm_strength)
        metrics = compute_round_metrics(
            pop_state, cfg, rng,
            history=pop_states[-50:] if scoring_method == 'fix_c' else None,
            scoring_method=scoring_method,
            fix_k=fix_k,
            fix_penalty=fix_penalty,
        )
        pop_states.append(pop_state)
        round_metrics.append(metrics)

    # Aggregate
    mi = compute_mi_from_rounds(pop_states)

    truthfulness = float(np.nanmean([m['truthfulness_rate'] for m in round_metrics]))
    score_var = float(np.nanmean([m['score_variance'] for m in round_metrics]))
    honest_pay_vals = [m['honest_payment'] for m in round_metrics
                       if not np.isnan(m['honest_payment'])]
    honest_pay = float(np.mean(honest_pay_vals)) if honest_pay_vals else float('nan')
    llm_pay_vals = [m['llm_payment'] for m in round_metrics
                    if not np.isnan(m['llm_payment'])]
    llm_pay = float(np.mean(llm_pay_vals)) if llm_pay_vals else float('nan')

    gap = (honest_pay - llm_pay) if not np.isnan(llm_pay) else float('nan')

    return {
        'alpha': alpha,
        'scoring_method': scoring_method,
        'truthfulness_rate': truthfulness,
        'mutual_information': mi,
        'score_variance': score_var,
        'honest_payment': honest_pay,
        'llm_payment': llm_pay,
        'equilibrium_gap': gap,
    }


def run_all_experiments(cfg: Optional[SimConfig] = None,
                        M: int = M_ROUNDS,
                        alpha_values: Optional[np.ndarray] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Run the full experiment suite: all α values × all scoring methods.

    Scoring methods run:
      - baseline
      - fix_a_k3 (Fix A with k=3 peers)
      - fix_a_k5 (Fix A with k=5 peers)
      - fix_b (Fix B popularity penalty)
      - fix_c (Fix C variance-aware)

    Parameters
    ----------
    cfg : SimConfig, optional
        Base simulation config. Defaults to SimConfig() with all defaults.
    M : int
        Number of simulation rounds per (α, method) combination
    alpha_values : np.ndarray, optional
        Array of α values to sweep. Defaults to ALPHA_VALUES.
    verbose : bool
        Print progress to stdout

    Returns
    -------
    pd.DataFrame
        Results with columns: alpha, scoring_method, truthfulness_rate,
        mutual_information, score_variance, honest_payment, llm_payment,
        equilibrium_gap
    """
    if cfg is None:
        cfg = SimConfig()
    if alpha_values is None:
        alpha_values = ALPHA_VALUES

    # Define all (method, params) combinations to run
    methods = [
        ('baseline', dict(fix_k=3, fix_penalty=0.5, seed_offset=0)),
        ('fix_a',    dict(fix_k=3, fix_penalty=0.5, seed_offset=100)),
        ('fix_a',    dict(fix_k=5, fix_penalty=0.5, seed_offset=200)),
        ('fix_b',    dict(fix_k=3, fix_penalty=0.5, seed_offset=300)),
        ('fix_c',    dict(fix_k=3, fix_penalty=0.5, seed_offset=400)),
    ]

    # Method labels for output
    method_labels = ['baseline', 'fix_a_k3', 'fix_a_k5', 'fix_b', 'fix_c']

    all_results = []
    total = len(alpha_values) * len(methods)
    done = 0

    for i, (method, params) in enumerate(methods):
        label = method_labels[i]

        if verbose:
            print(f"\n[{label}] Running {len(alpha_values)} alpha values x {M} rounds each...")

        for alpha in alpha_values:
            result = run_experiment_for_alpha(
                alpha=alpha,
                cfg_base=cfg,
                M=M,
                scoring_method=method,
                fix_k=params['fix_k'],
                fix_penalty=params['fix_penalty'],
                seed_offset=params['seed_offset'],
            )
            # Override scoring_method with descriptive label
            result['scoring_method'] = label
            all_results.append(result)

            done += 1
            if verbose:
                pct = done / total * 100
                llm_str = (f"{result['llm_payment']:.4f}"
                           if not np.isnan(result['llm_payment']) else "N/A")
                print(f"  alpha={alpha:.2f} | "
                      f"TR={result['truthfulness_rate']:.3f} "
                      f"MI={result['mutual_information']:.4f} "
                      f"SV={result['score_variance']:.5f} "
                      f"H={result['honest_payment']:.4f} "
                      f"L={llm_str}",
                      flush=True)

    df = pd.DataFrame(all_results)
    return df


def find_alpha_star(df: pd.DataFrame, method: str = 'baseline') -> Optional[float]:
    """
    Find α* where honest payment ≈ LLM payment (equilibrium collapses).

    Scans through α values for the given method and returns the first α
    where equilibrium_gap crosses zero (honest payment becomes less than
    or equal to LLM payment).

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    method : str
        Scoring method to analyze (default 'baseline')

    Returns
    -------
    float or None
        The α value at which the gap first becomes non-positive,
        or None if the gap never crosses zero.
    """
    sub = df[df['scoring_method'] == method].sort_values('alpha')

    for _, row in sub.iterrows():
        if not np.isnan(row['equilibrium_gap']) and row['equilibrium_gap'] <= 0:
            return float(row['alpha'])

    return None  # Mechanism stays robust across full range


def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print a formatted summary table of results.

    For each (α, method), shows all four metrics. Also marks where α* occurs
    for the baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    """
    alpha_star = find_alpha_star(df, 'baseline')

    print("\n" + "="*90)
    print("SIMULATION RESULTS SUMMARY")
    print("="*90)
    print(f"{'alpha':>6} {'Method':<12} {'TR':>8} {'MI(nats)':>10} {'Var':>10} "
          f"{'H Pay':>9} {'L Pay':>9} {'Gap':>9} {'a*?':>5}")
    print("-"*90)

    for _, row in df.sort_values(['alpha', 'scoring_method']).iterrows():
        star = " <--" if (alpha_star is not None and
                          row['scoring_method'] == 'baseline' and
                          abs(row['alpha'] - alpha_star) < 0.001) else ""
        llm_str = f"{row['llm_payment']:9.4f}" if not np.isnan(row['llm_payment']) else "     N/A"
        gap_str = f"{row['equilibrium_gap']:9.4f}" if not np.isnan(row['equilibrium_gap']) else "     N/A"

        print(f"{row['alpha']:6.2f} {row['scoring_method']:<12} "
              f"{row['truthfulness_rate']:8.3f} "
              f"{row['mutual_information']:10.4f} "
              f"{row['score_variance']:10.5f} "
              f"{row['honest_payment']:9.4f} "
              f"{llm_str} "
              f"{gap_str}"
              f"{star}")

    print("="*90)
    if alpha_star is not None:
        print(f"\nalpha* (equilibrium collapse) ~= {alpha_star:.2f} for baseline mechanism")
    else:
        print("\nalpha* not reached -- mechanism robust across full alpha range")
    print()
