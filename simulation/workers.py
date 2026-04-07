"""Honest and LLM-assisted worker behavior models."""

import numpy as np
from environment import PopulationState, SimConfig


class HonestWorker:
    """Reports the true private signal."""

    def generate_report(self, signal: int, pop_state: PopulationState,
                        rng: np.random.Generator) -> int:
        return int(signal)


class LLMWorker:
    """
    Ignores private signal; reports the observed majority with probability `strength`.
    strength=1.0 is fully deterministic consensus-following.
    """

    def __init__(self, strength: float = 1.0):
        if not 0.5 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0.5, 1.0], got {strength}")
        self.strength = strength

    def generate_report(self, signal: int, pop_state: PopulationState,
                        rng: np.random.Generator) -> int:
        modal_answer = 1 if pop_state.consensus_prob >= 0.5 else 0
        if rng.random() < self.strength:
            return int(modal_answer)
        return int(1 - modal_answer)


def populate_reports(pop_state: PopulationState, cfg: SimConfig,
                     rng: np.random.Generator,
                     llm_strength: float = 1.0) -> PopulationState:
    """
    Two-pass report generation.

    Pass 1: honest workers report their signal.
    Pass 2: compute the observed honest majority, store it as consensus_prob,
    then LLM workers report based on that majority.

    This fixes the degenerate case where compute_consensus_prob() = 0.5 for a
    symmetric prior, which previously caused LLM workers to always report 1.
    With this approach the LLM tracks the actual round's honest majority,
    which correlates with theta.
    """
    honest_worker = HonestWorker()
    llm_worker = LLMWorker(strength=llm_strength)

    reports = np.zeros(cfg.N, dtype=int)
    honest_mask = pop_state.worker_types == 0

    for i in np.where(honest_mask)[0]:
        reports[i] = honest_worker.generate_report(pop_state.signals[i], pop_state, rng)

    honest_indices = np.where(honest_mask)[0]
    pop_state.consensus_prob = (float(np.mean(reports[honest_indices]))
                                if len(honest_indices) > 0 else 0.5)

    for i in np.where(~honest_mask)[0]:
        reports[i] = llm_worker.generate_report(pop_state.signals[i], pop_state, rng)

    pop_state.reports = reports
    return pop_state
