"""
environment.py — State, signal generation, and worker population setup.

This module models the information environment in the peer prediction simulation:
- A binary true state θ is drawn from a prior
- Each worker receives a noisy private signal about θ
- Workers are classified as honest or LLM-assisted based on fraction α

Key design choice: The LLM worker's consensus probability is computed from the
MARGINAL distribution of reports (averaging over θ), not conditioned on θ. This
captures the key threat: the LLM predicts what the population will say, not what
is actually true.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    """
    Simulation configuration parameters. Centralizing these here
    makes it easy to run sweeps or change defaults without touching
    core logic in other modules.
    """
    N: int = 200          # Number of workers
    q: float = 0.7        # Signal accuracy P(s=θ | θ)
    prior: float = 0.5    # P(θ=1)
    alpha: float = 0.0    # Fraction of LLM-assisted workers
    seed: Optional[int] = None


@dataclass
class PopulationState:
    """
    Snapshot of the population for one simulation round.

    Attributes:
        theta: True state (0 or 1)
        signals: Array of private signals, shape (N,)
        worker_types: Array of 0 (honest) or 1 (LLM), shape (N,)
        reports: Array of submitted reports, shape (N,) — filled after workers act
        consensus_prob: P(report=1) marginalized over θ, used by LLM workers
    """
    theta: int
    signals: np.ndarray
    worker_types: np.ndarray
    reports: np.ndarray = field(default_factory=lambda: np.array([]))
    consensus_prob: float = 0.0


def signal_likelihood(s: int, theta: int, q: float) -> float:
    """
    P(s | θ) — likelihood of signal s given true state θ.

    With signal accuracy q:
      P(s=θ | θ) = q
      P(s≠θ | θ) = 1 - q

    Parameters
    ----------
    s : int
        Signal value (0 or 1)
    theta : int
        True state (0 or 1)
    q : float
        Signal accuracy in (0.5, 1)

    Returns
    -------
    float
        P(s | θ)
    """
    if s == theta:
        return q
    else:
        return 1.0 - q


def compute_consensus_prob(cfg: SimConfig) -> float:
    """
    Compute P(report=1) marginalized over θ for use by LLM workers.

    The LLM predicts what the population will report on average, not what
    is true. This marginal probability is:
      P(report=1) = P(s=1|θ=1)·P(θ=1) + P(s=1|θ=0)·P(θ=0)
                  = q · prior + (1-q) · (1-prior)

    When prior=0.5, this equals 0.5 always — the LLM is unbiased about which
    answer is "popular" in the long run. But within a single round, the actual
    population leans toward θ, so honest workers' reports are informative
    and correlated with θ while LLM workers' reports are not.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration

    Returns
    -------
    float
        Marginal probability that a random worker reports 1
    """
    # P(report=1) = sum_θ P(s=1|θ) * P(θ)
    p_report1 = (signal_likelihood(1, 1, cfg.q) * cfg.prior
                 + signal_likelihood(1, 0, cfg.q) * (1.0 - cfg.prior))
    return p_report1


def generate_round(cfg: SimConfig, rng: np.random.Generator) -> PopulationState:
    """
    Generate one simulation round: draw θ, signals, and worker type assignments.

    The true state θ is drawn from the prior. Each worker receives a signal
    that equals θ with probability q. Worker types are assigned independently
    with P(LLM) = α.

    Note: Reports are NOT generated here — that happens in workers.py where
    worker behavior logic lives. This keeps environment and behavior separate.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration
    rng : np.random.Generator
        Seeded random number generator for reproducibility

    Returns
    -------
    PopulationState
        A fresh population snapshot with theta, signals, and worker_types set.
        reports is empty until workers act.
    """
    # Draw true state from prior
    theta = int(rng.random() < cfg.prior)

    # Generate noisy private signals
    # Each signal = θ with prob q, flipped with prob (1-q)
    noise = rng.random(cfg.N)
    signals = np.where(noise < cfg.q, theta, 1 - theta).astype(int)

    # Assign worker types: 0 = honest, 1 = LLM-assisted
    worker_types = (rng.random(cfg.N) < cfg.alpha).astype(int)

    # Consensus probability: P(report=1) marginalized over θ
    # LLM workers use this to decide their report
    consensus_prob = compute_consensus_prob(cfg)

    return PopulationState(
        theta=theta,
        signals=signals,
        worker_types=worker_types,
        reports=np.zeros(cfg.N, dtype=int),
        consensus_prob=consensus_prob,
    )
