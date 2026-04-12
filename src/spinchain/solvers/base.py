"""Base solver interface. All solver backends implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod

import dimod


class BaseSolver(ABC):
    """Abstract base for QUBO solvers.

    Implementations:
    - SimulatedAnnealingSolver (MVP — dwave-neal)
    - QiskitQAOASolver (future — IBM Quantum validation)
    - COBISolver (future — COBI Ising chip)
    - QuanfluenceSolver (future — optical CIM)
    """

    @abstractmethod
    def solve(self, bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
        """Solve the BQM and return a sample set.

        Args:
            bqm: Binary quadratic model representing the reasoning optimization.

        Returns:
            dimod.SampleSet with solutions ranked by energy.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Solver identifier for logging/benchmarking."""
        ...
