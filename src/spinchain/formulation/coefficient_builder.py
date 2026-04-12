"""Build HUBO/QUBO coefficients from fragment statistics.

Implements the coefficient design from QCR-LLM (arXiv 2510.24509):
- Linear: w_i = -mu * p_i + alpha * risk_i
- Quadratic: w_ij = -beta * (corr_ij - lambda^2 * sim(i,j))
"""

from __future__ import annotations

import numpy as np
from scipy.stats import zscore


class CoefficientBuilder:
    """Computes linear and quadratic coefficients for the QUBO formulation.

    Args:
        mu: Weight for fragment popularity (higher = prefer common fragments).
        alpha: Weight for fragment stability/risk penalty.
        beta: Weight for pairwise coherence terms.
        lambda_sim: Semantic similarity penalty factor.
        regularization: Small constant for z-score stability.
    """

    def __init__(
        self,
        mu: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.0,
        lambda_sim: float = 0.3,
        regularization: float = 1e-6,
    ):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lambda_sim = lambda_sim
        self.regularization = regularization

    def compute_linear_weights(
        self,
        fragment_sources: list[set[int]],
        num_completions: int,
    ) -> np.ndarray:
        """Compute linear (1-body) weights for each fragment.

        w_i = -mu * p_i + alpha * risk_i
        where p_i = popularity, risk_i = p_i * (1 - p_i)
        """
        r = len(fragment_sources)
        weights = np.zeros(r)

        for i in range(r):
            p_i = len(fragment_sources[i]) / num_completions
            risk_i = p_i * (1.0 - p_i)
            weights[i] = -self.mu * p_i + self.alpha * risk_i

        return weights

    def compute_quadratic_weights(
        self,
        fragment_sources: list[set[int]],
        num_completions: int,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute quadratic (2-body) weights for fragment pairs.

        w_ij = -beta * (z_score(corr_ij) - lambda^2 * sim(i,j))
        where corr_ij = co-occurrence correlation, sim = cosine similarity
        """
        r = len(fragment_sources)
        n = num_completions

        # Popularity
        p = np.array([len(s) / n for s in fragment_sources])

        # Co-occurrence matrix
        co_occurrence = np.zeros((r, r))
        for i in range(r):
            for j in range(i + 1, r):
                n_ij = len(fragment_sources[i] & fragment_sources[j])
                co_occurrence[i, j] = n_ij / n - p[i] * p[j]
                co_occurrence[j, i] = co_occurrence[i, j]

        # Standardize correlations (z-scores)
        upper_tri = co_occurrence[np.triu_indices(r, k=1)]
        if len(upper_tri) > 1 and np.std(upper_tri) > self.regularization:
            mean_c = np.mean(upper_tri)
            std_c = np.std(upper_tri)
            z_corr = (co_occurrence - mean_c) / (std_c + self.regularization)
        else:
            z_corr = co_occurrence

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Quadratic weights
        weights = np.zeros((r, r))
        for i in range(r):
            for j in range(i + 1, r):
                weights[i, j] = -self.beta * (
                    z_corr[i, j] - self.lambda_sim**2 * sim_matrix[i, j]
                )
                weights[j, i] = weights[i, j]

        return weights
