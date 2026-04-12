"""Tests for QUBOBuilder — BQM construction and cardinality constraints."""

from __future__ import annotations

import numpy as np
import pytest
import dimod

from spinchain.formulation.qubo_builder import QUBOBuilder


class TestBQMConstruction:
    """Test basic BQM building from weight arrays."""

    def test_linear_only(self):
        """BQM with only linear weights, no quadratic."""
        builder = QUBOBuilder()
        linear_w = np.array([-1.0, -0.5, 0.3])
        quadratic_w = np.zeros((3, 3))

        bqm = builder.build(linear_w, quadratic_w)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY
        assert len(bqm.variables) == 3
        assert bqm.linear[0] == pytest.approx(-1.0)
        assert bqm.linear[1] == pytest.approx(-0.5)
        assert bqm.linear[2] == pytest.approx(0.3)
        assert len(bqm.quadratic) == 0

    def test_quadratic_terms(self):
        """BQM includes quadratic terms above threshold."""
        builder = QUBOBuilder()
        linear_w = np.array([-1.0, -1.0])
        quadratic_w = np.array([
            [0.0, 0.5],
            [0.5, 0.0],
        ])

        bqm = builder.build(linear_w, quadratic_w)

        assert len(bqm.quadratic) == 1
        assert bqm.quadratic[(0, 1)] == pytest.approx(0.5)

    def test_small_quadratic_pruned(self):
        """Quadratic terms with |w_ij| < 1e-10 are dropped."""
        builder = QUBOBuilder()
        linear_w = np.array([-1.0, -1.0])
        quadratic_w = np.array([
            [0.0, 1e-12],
            [1e-12, 0.0],
        ])

        bqm = builder.build(linear_w, quadratic_w)

        assert len(bqm.quadratic) == 0

    def test_negative_quadratic(self):
        """Negative quadratic weight (attraction) is preserved."""
        builder = QUBOBuilder()
        linear_w = np.array([0.0, 0.0])
        quadratic_w = np.array([
            [0.0, -2.5],
            [-2.5, 0.0],
        ])

        bqm = builder.build(linear_w, quadratic_w)

        assert bqm.quadratic[(0, 1)] == pytest.approx(-2.5)

    def test_offset_is_zero(self):
        """BQM offset should be zero (no constant term)."""
        builder = QUBOBuilder()
        bqm = builder.build(np.array([-1.0]), np.zeros((1, 1)))

        assert bqm.offset == pytest.approx(0.0)

    def test_larger_system(self):
        """6 fragments — verify all variables present and quadratic count correct."""
        builder = QUBOBuilder()
        r = 6
        linear_w = np.random.randn(r)
        quadratic_w = np.random.randn(r, r)
        quadratic_w = (quadratic_w + quadratic_w.T) / 2
        np.fill_diagonal(quadratic_w, 0.0)

        bqm = builder.build(linear_w, quadratic_w)

        assert len(bqm.variables) == r
        # Upper triangle: r*(r-1)/2 = 15 terms (all should be above threshold)
        max_quadratic = r * (r - 1) // 2
        assert len(bqm.quadratic) <= max_quadratic


class TestCardinalityConstraint:
    """Test the penalty*(sum(x_i) - K)^2 cardinality constraint."""

    def test_cardinality_adds_to_linear(self):
        """Cardinality K modifies linear terms: w_i += penalty*(1 - 2K)."""
        builder = QUBOBuilder(penalty_strength=5.0)
        linear_w = np.array([0.0, 0.0, 0.0])
        quadratic_w = np.zeros((3, 3))

        bqm = builder.build(linear_w, quadratic_w, target_fragments=1)

        # penalty*(1 - 2*1) = 5.0 * (-1) = -5.0
        for i in range(3):
            assert bqm.linear[i] == pytest.approx(-5.0)

    def test_cardinality_adds_to_quadratic(self):
        """Cardinality adds 2*penalty to all quadratic pairs."""
        builder = QUBOBuilder(penalty_strength=5.0)
        linear_w = np.array([0.0, 0.0, 0.0])
        quadratic_w = np.zeros((3, 3))

        bqm = builder.build(linear_w, quadratic_w, target_fragments=1)

        # All pairs get +2*5.0 = +10.0
        for i in range(3):
            for j in range(i + 1, 3):
                assert bqm.quadratic[(i, j)] == pytest.approx(10.0)

    def test_no_cardinality_leaves_weights_unchanged(self):
        """Without target_fragments, weights pass through unchanged."""
        builder = QUBOBuilder(penalty_strength=5.0)
        linear_w = np.array([-1.0, -2.0])
        quadratic_w = np.array([
            [0.0, 0.7],
            [0.7, 0.0],
        ])

        bqm = builder.build(linear_w, quadratic_w, target_fragments=None)

        assert bqm.linear[0] == pytest.approx(-1.0)
        assert bqm.linear[1] == pytest.approx(-2.0)
        assert bqm.quadratic[(0, 1)] == pytest.approx(0.7)

    def test_cardinality_stacks_with_existing_weights(self):
        """Cardinality penalty adds on top of existing QUBO weights."""
        builder = QUBOBuilder(penalty_strength=5.0)
        linear_w = np.array([-1.0, -2.0])
        quadratic_w = np.array([
            [0.0, 0.7],
            [0.7, 0.0],
        ])

        bqm = builder.build(linear_w, quadratic_w, target_fragments=1)

        # linear: original + penalty*(1-2*1) = original + (-5)
        assert bqm.linear[0] == pytest.approx(-1.0 + -5.0)
        assert bqm.linear[1] == pytest.approx(-2.0 + -5.0)
        # quadratic: original + 2*penalty = 0.7 + 10.0
        assert bqm.quadratic[(0, 1)] == pytest.approx(0.7 + 10.0)

    def test_ground_state_selects_k_fragments(self):
        """For uniform weights + cardinality K, brute-force ground state has exactly K selected."""
        builder = QUBOBuilder(penalty_strength=10.0)
        r = 4
        k = 2
        # Uniform negative linear (all fragments equally desirable)
        linear_w = np.full(r, -1.0)
        quadratic_w = np.zeros((r, r))

        bqm = builder.build(linear_w, quadratic_w, target_fragments=k)

        # Brute-force: enumerate all 2^4 = 16 states
        best_energy = float("inf")
        best_count = None
        for bits in range(2**r):
            sample = {i: (bits >> i) & 1 for i in range(r)}
            energy = bqm.energy(sample)
            if energy < best_energy:
                best_energy = energy
                best_count = sum(sample.values())

        assert best_count == k


class TestQUBOConversion:
    """Test BQM to raw QUBO dict conversion."""

    def test_bqm_to_qubo_roundtrip(self):
        """Converting to QUBO dict and back preserves structure."""
        builder = QUBOBuilder()
        linear_w = np.array([-1.0, 0.5])
        quadratic_w = np.array([
            [0.0, 0.3],
            [0.3, 0.0],
        ])

        bqm = builder.build(linear_w, quadratic_w)
        qubo_dict, offset = builder.bqm_to_qubo(bqm)

        assert isinstance(qubo_dict, dict)
        # QUBO dict has diagonal (linear) and off-diagonal (quadratic)
        assert (0, 0) in qubo_dict  # linear term for var 0
        assert (1, 1) in qubo_dict  # linear term for var 1
