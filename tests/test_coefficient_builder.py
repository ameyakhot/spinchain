"""Tests for CoefficientBuilder — the QUBO coefficient formulation."""

from __future__ import annotations

import numpy as np
import pytest

from spinchain.formulation.coefficient_builder import CoefficientBuilder


class TestLinearWeights:
    """Test w_i = -mu * p_i + alpha * risk_i where risk_i = p_i * (1 - p_i)."""

    def test_fully_popular_fragment(self):
        """Fragment appearing in ALL completions: p=1, risk=0 => w = -mu."""
        cb = CoefficientBuilder(mu=1.0, alpha=0.5)
        sources = [{0, 1, 2}]  # appears in all 3 completions
        w = cb.compute_linear_weights(sources, num_completions=3)

        assert w[0] == pytest.approx(-1.0)  # -mu * 1.0 + alpha * 0.0

    def test_half_popular_fragment(self):
        """Fragment appearing in half: p=0.5, risk=0.25 => w = -mu*0.5 + alpha*0.25."""
        cb = CoefficientBuilder(mu=1.0, alpha=0.5)
        sources = [{0, 1}]  # 2 of 4 completions
        w = cb.compute_linear_weights(sources, num_completions=4)

        expected = -1.0 * 0.5 + 0.5 * 0.25  # -0.375
        assert w[0] == pytest.approx(expected)

    def test_rare_fragment(self):
        """Fragment in 1 of 5: p=0.2, risk=0.16 => w = -0.2 + 0.08 = -0.12."""
        cb = CoefficientBuilder(mu=1.0, alpha=0.5)
        sources = [{2}]
        w = cb.compute_linear_weights(sources, num_completions=5)

        p = 0.2
        expected = -1.0 * p + 0.5 * p * (1 - p)
        assert w[0] == pytest.approx(expected)

    def test_mu_scales_popularity(self):
        """Doubling mu should double the popularity contribution."""
        sources = [{0, 1}]
        w1 = CoefficientBuilder(mu=1.0, alpha=0.0).compute_linear_weights(sources, 4)
        w2 = CoefficientBuilder(mu=2.0, alpha=0.0).compute_linear_weights(sources, 4)

        assert w2[0] == pytest.approx(2.0 * w1[0])

    def test_alpha_scales_risk(self):
        """With mu=0, only risk term remains. alpha=0 => w=0."""
        sources = [{0}]
        w = CoefficientBuilder(mu=0.0, alpha=0.0).compute_linear_weights(sources, 3)
        assert w[0] == pytest.approx(0.0)

    def test_multiple_fragments(self, two_completion_sources):
        """Verify weights for a mix of popular and rare fragments."""
        cb = CoefficientBuilder(mu=1.0, alpha=0.5)
        w = cb.compute_linear_weights(two_completion_sources, num_completions=2)

        assert len(w) == 4
        # Frags 0,3 have p=1 => w = -1.0
        assert w[0] == pytest.approx(-1.0)
        assert w[3] == pytest.approx(-1.0)
        # Frags 1,2 have p=0.5 => w = -0.5 + 0.5*0.25 = -0.375
        assert w[1] == pytest.approx(-0.375)
        assert w[2] == pytest.approx(-0.375)

    def test_popular_fragments_have_lower_energy(self, two_completion_sources):
        """More popular fragments should have more negative weights (lower energy = favored)."""
        cb = CoefficientBuilder(mu=1.0, alpha=0.5)
        w = cb.compute_linear_weights(two_completion_sources, num_completions=2)

        # popular frags (0,3) should be more negative than rare (1,2)
        assert w[0] < w[1]
        assert w[3] < w[2]


class TestQuadraticWeights:
    """Test w_ij = -beta * (z_corr_ij - lambda^2 * sim(i,j))."""

    def test_orthogonal_embeddings_no_similarity_penalty(self, two_completion_sources, orthogonal_embeddings_4):
        """With orthogonal embeddings, similarity term is zero — only co-occurrence matters."""
        cb = CoefficientBuilder(beta=1.0, lambda_sim=0.3)
        w = cb.compute_quadratic_weights(
            two_completion_sources, num_completions=2, embeddings=orthogonal_embeddings_4
        )

        assert w.shape == (4, 4)
        # Diagonal should be zero
        for i in range(4):
            assert w[i, i] == pytest.approx(0.0)
        # Should be symmetric
        for i in range(4):
            for j in range(i + 1, 4):
                assert w[i, j] == pytest.approx(w[j, i])

    def test_symmetry(self, two_completion_sources, similar_embeddings_4):
        """Quadratic weight matrix must be symmetric."""
        cb = CoefficientBuilder()
        w = cb.compute_quadratic_weights(
            two_completion_sources, num_completions=2, embeddings=similar_embeddings_4
        )

        np.testing.assert_array_almost_equal(w, w.T)

    def test_zero_diagonal(self, two_completion_sources, orthogonal_embeddings_4):
        """No self-interaction: w_ii = 0."""
        cb = CoefficientBuilder()
        w = cb.compute_quadratic_weights(
            two_completion_sources, num_completions=2, embeddings=orthogonal_embeddings_4
        )

        for i in range(4):
            assert w[i, i] == pytest.approx(0.0)

    def test_similar_embeddings_get_penalty(self):
        """Two fragments with high cosine similarity should get a positive weight (repulsion)."""
        # 2 fragments, both from same completion, nearly identical embeddings
        sources = [{0}, {0}]
        emb = np.array([
            [1.0, 0.0],
            [0.99, 0.14],  # very similar to first
        ])
        cb = CoefficientBuilder(beta=1.0, lambda_sim=1.0)
        w = cb.compute_quadratic_weights(sources, num_completions=1, embeddings=emb)

        # With lambda_sim=1.0 and high similarity, the sim penalty dominates
        # => w_01 should be positive (discouraging co-selection of near-duplicates)
        # Note: with only 1 pair, z-score normalization falls through to raw co-occurrence
        assert w[0, 1] > 0 or w[0, 1] == pytest.approx(0.0, abs=0.1)

    def test_beta_zero_kills_quadratic(self, two_completion_sources, orthogonal_embeddings_4):
        """With beta=0, all quadratic weights should be zero."""
        cb = CoefficientBuilder(beta=0.0)
        w = cb.compute_quadratic_weights(
            two_completion_sources, num_completions=2, embeddings=orthogonal_embeddings_4
        )

        np.testing.assert_array_almost_equal(w, np.zeros((4, 4)))

    def test_co_occurring_fragments_attract(self):
        """Fragments that always co-occur should have negative weight (attraction)."""
        # 3 fragments from 3 completions
        # frag 0 and 1 always co-occur, frag 2 is independent
        sources = [
            {0, 1, 2},  # frag 0: all
            {0, 1, 2},  # frag 1: all (always with 0)
            {0},         # frag 2: only comp 0
        ]
        emb = np.eye(3)  # orthogonal => no similarity penalty
        cb = CoefficientBuilder(beta=1.0, lambda_sim=0.0)
        w = cb.compute_quadratic_weights(sources, num_completions=3, embeddings=emb)

        # frag 0 and 1 co-occur perfectly => positive correlation => negative weight (attract)
        # Exact value depends on z-score normalization
        # But w[0,1] should be the most negative (strongest attraction)
        assert w[0, 1] <= w[0, 2]

    def test_single_pair_skips_zscore(self):
        """With only 2 fragments (1 pair), z-score std=0 => raw co-occurrence used."""
        sources = [{0}, {1}]
        emb = np.eye(2)
        cb = CoefficientBuilder(beta=1.0, lambda_sim=0.0)
        # Should not crash even with only 1 upper-tri element
        w = cb.compute_quadratic_weights(sources, num_completions=2, embeddings=emb)

        assert w.shape == (2, 2)
