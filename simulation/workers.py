"""
workers.py — Worker behavior models: honest reporting and LLM-assisted reporting.

Two worker types are implemented:
  - HonestWorker: reports their true private signal
  - LLMWorker: ignores private signal, reports based on consensus prediction

Both workers share the same interface: generate_report(signal, pop_state, rng).
This abstraction allows experiments.py to drive a population without knowing
which type each worker is.

The LLM model here is deliberately simple and interpretable: the LLM predicts
the modal answer in the population and reports it with high (configurable)
probability. The strength parameter controls how deterministically the LLM
follows the consensus vs. adding noise.
"""

import numpy as np
from environment import PopulationState, SimConfig


class HonestWorker:
    """
    An honest worker who reports their private signal truthfully.

    This is the baseline behavior that peer prediction is designed to
    incentivize. No parameters needed — this worker's strategy is deterministic
    given the signal.
    """

    def generate_report(self, signal: int, pop_state: PopulationState,
                        rng: np.random.Generator) -> int:
        """
        Report the true private signal.

        Parameters
        ----------
        signal : int
            Worker's private signal (0 or 1)
        pop_state : PopulationState
            Current population state (not used by honest workers)
        rng : np.random.Generator
            Random number generator (not used by honest workers)

        Returns
        -------
        int
            The reported value = signal
        """
        return int(signal)


class LLMWorker:
    """
    An LLM-assisted worker who ignores their private signal and instead
    reports based on what the LLM predicts is the consensus answer.

    The LLM is modeled as observing the marginal distribution of reports
    in the population and reporting the modal answer with probability
    `strength` (and the minority answer with probability 1-strength).

    When strength=1.0: always reports consensus (modal) answer deterministically
    When strength=0.5: reports randomly (indistinguishable from noise)

    Parameters
    ----------
    strength : float
        How strongly the LLM follows the consensus. Default 1.0 (fully
        deterministic consensus-following).
    """

    def __init__(self, strength: float = 1.0):
        """
        Parameters
        ----------
        strength : float
            P(LLM reports modal answer) in [0.5, 1.0].
            At 1.0, always reports consensus. At 0.5, random.
        """
        if not 0.5 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0.5, 1.0], got {strength}")
        self.strength = strength

    def generate_report(self, signal: int, pop_state: PopulationState,
                        rng: np.random.Generator) -> int:
        """
        Generate a consensus-seeking report, ignoring the private signal.

        The LLM predicts what most people will say (consensus_prob from pop_state)
        and reports accordingly. If consensus_prob > 0.5, modal answer is 1;
        otherwise modal answer is 0. The LLM then reports the modal answer with
        probability `strength`.

        Parameters
        ----------
        signal : int
            Worker's private signal (IGNORED by LLM workers)
        pop_state : PopulationState
            Contains consensus_prob: P(random worker reports 1) marginalized
            over θ — the LLM's population-level prediction
        rng : np.random.Generator
            Random number generator for stochastic LLM behavior

        Returns
        -------
        int
            Consensus-based report (0 or 1)
        """
        # Determine the modal (consensus) answer
        consensus_prob = pop_state.consensus_prob
        modal_answer = 1 if consensus_prob >= 0.5 else 0
        minority_answer = 1 - modal_answer

        # Report modal answer with probability `strength`
        if rng.random() < self.strength:
            return int(modal_answer)
        else:
            return int(minority_answer)


def populate_reports(pop_state: PopulationState, cfg: SimConfig,
                     rng: np.random.Generator,
                     llm_strength: float = 1.0) -> PopulationState:
    """
    Generate reports for all workers in the population.

    Iterates over all workers, creates the appropriate worker type, and
    calls generate_report. Fills pop_state.reports in-place.

    Parameters
    ----------
    pop_state : PopulationState
        Population state with signals and worker_types already set
    cfg : SimConfig
        Simulation configuration (used to create workers)
    rng : np.random.Generator
        Seeded random number generator
    llm_strength : float
        Consensus-following strength for LLM workers (default 1.0)

    Returns
    -------
    PopulationState
        Same pop_state with reports array filled in
    """
    honest_worker = HonestWorker()
    llm_worker = LLMWorker(strength=llm_strength)

    reports = np.zeros(cfg.N, dtype=int)
    for i in range(cfg.N):
        signal = pop_state.signals[i]
        worker_type = pop_state.worker_types[i]

        if worker_type == 0:  # honest
            reports[i] = honest_worker.generate_report(signal, pop_state, rng)
        else:  # LLM-assisted
            reports[i] = llm_worker.generate_report(signal, pop_state, rng)

    pop_state.reports = reports
    return pop_state
