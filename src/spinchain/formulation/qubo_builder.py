"""Build QUBO from linear and quadratic weights.

Converts the HUBO formulation into a QUBO compatible with any
Ising/QUBO solver (SA, D-Wave, COBI, Quanfluence, Qiskit QAOA).
"""

from __future__ import annotations

import numpy as np
import dimod


class QUBOBuilder:
    """Builds a dimod BinaryQuadraticModel from fragment coefficients.

    The QUBO encodes: minimize H(x) = sum_i w_i * x_i + sum_{i<j} w_ij * x_i * x_j
    with an optional cardinality constraint to select ~K fragments.
    """

    def __init__(self, penalty_strength: float = 5.0):
        self.penalty_strength = penalty_strength

    def build(
        self,
        linear_weights: np.ndarray,
        quadratic_weights: np.ndarray,
        target_fragments: int | None = None,
    ) -> dimod.BinaryQuadraticModel:
        """Build BQM from weights.

        Args:
            linear_weights: 1D array of shape (R,) — linear coefficients.
            quadratic_weights: 2D array of shape (R, R) — quadratic coefficients.
            target_fragments: If set, add penalty to select approximately K fragments.

        Returns:
            dimod.BinaryQuadraticModel ready for any solver.
        """
        r = len(linear_weights)
        linear = {}
        quadratic = {}

        for i in range(r):
            linear[i] = float(linear_weights[i])

        for i in range(r):
            for j in range(i + 1, r):
                if abs(quadratic_weights[i, j]) > 1e-10:
                    quadratic[(i, j)] = float(quadratic_weights[i, j])

        # Add cardinality constraint: penalty * (sum(x_i) - K)^2
        if target_fragments is not None:
            k = target_fragments
            for i in range(r):
                linear[i] = linear.get(i, 0.0) + self.penalty_strength * (1 - 2 * k)
            for i in range(r):
                for j in range(i + 1, r):
                    key = (i, j)
                    quadratic[key] = quadratic.get(key, 0.0) + 2 * self.penalty_strength

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
        return bqm

    def bqm_to_qubo(self, bqm: dimod.BinaryQuadraticModel) -> tuple[dict, float]:
        """Convert BQM to raw QUBO dict format (for custom solvers)."""
        return bqm.to_qubo()
