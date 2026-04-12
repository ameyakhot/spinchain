"""Simulated Annealing solver using D-Wave's Neal."""

from __future__ import annotations

import dimod
import neal

from spinchain.solvers.base import BaseSolver


class SimulatedAnnealingSolver(BaseSolver):
    """Simulated annealing solver via dwave-neal.

    This is the MVP solver. Same QUBO formulation will transfer
    directly to Ising chips (COBI, Quanfluence) or QAOA (IBM Quantum).
    """

    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int = 1000,
        beta_range: tuple[float, float] | None = None,
    ):
        self.sampler = neal.SimulatedAnnealingSampler()
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range

    def solve(self, bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
        """Solve using simulated annealing."""
        kwargs = {
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
        }
        if self.beta_range is not None:
            kwargs["beta_range"] = self.beta_range

        return self.sampler.sample(bqm, **kwargs)

    @property
    def name(self) -> str:
        return "simulated_annealing"
