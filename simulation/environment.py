"""State, signal generation, and worker population setup."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    N: int = 200
    q: float = 0.7        # signal accuracy P(s=theta | theta)
    prior: float = 0.5    # P(theta=1)
    alpha: float = 0.0    # fraction of LLM-assisted workers
    seed: Optional[int] = None


@dataclass
class PopulationState:
    """
    Snapshot of one simulation round.

    Attributes:
        theta: true state (0 or 1)
        signals: private signals, shape (N,)
        worker_types: 0=honest, 1=LLM, shape (N,)
        reports: submitted reports, shape (N,) — filled by populate_reports()
        consensus_prob: observed honest majority fraction, used by LLM workers
    """
    theta: int
    signals: np.ndarray
    worker_types: np.ndarray
    reports: np.ndarray = field(default_factory=lambda: np.array([]))
    consensus_prob: float = 0.0


def signal_likelihood(s: int, theta: int, q: float) -> float:
    """P(s | theta): q if s==theta, else 1-q."""
    return q if s == theta else 1.0 - q


def compute_consensus_prob(cfg: SimConfig) -> float:
    """Marginal P(report=1) averaged over theta — used as LLM prior when no honest reports exist."""
    return (signal_likelihood(1, 1, cfg.q) * cfg.prior
            + signal_likelihood(1, 0, cfg.q) * (1.0 - cfg.prior))


def generate_round(cfg: SimConfig, rng: np.random.Generator) -> PopulationState:
    """Draw theta, generate noisy signals, assign worker types. Reports filled separately."""
    theta = int(rng.random() < cfg.prior)
    noise = rng.random(cfg.N)
    signals = np.where(noise < cfg.q, theta, 1 - theta).astype(int)
    worker_types = (rng.random(cfg.N) < cfg.alpha).astype(int)

    return PopulationState(
        theta=theta,
        signals=signals,
        worker_types=worker_types,
        reports=np.zeros(cfg.N, dtype=int),
        consensus_prob=compute_consensus_prob(cfg),
    )
