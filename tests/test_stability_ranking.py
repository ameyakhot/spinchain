"""Tests for _stability_ranking — fragment selection from SA solutions."""

from __future__ import annotations

import dimod
import numpy as np
import pytest

from spinchain.server import _stability_ranking


def _make_sample_set(samples: list[dict], energies: list[float]) -> dimod.SampleSet:
    """Helper: build a SampleSet from explicit samples and energies."""
    return dimod.SampleSet.from_samples(
        samples, vartype=dimod.BINARY, energy=energies
    )


class TestStabilityRanking:
    """Test the QCR-LLM stability ranking algorithm."""

    def test_unanimous_fragment_selected(self):
        """Fragment appearing in ALL low-energy solutions should be selected."""
        samples = [
            {0: 1, 1: 1, 2: 0},
            {0: 1, 1: 0, 2: 1},
            {0: 1, 1: 1, 2: 1},
            {0: 1, 1: 0, 2: 0},
        ]
        energies = [-4.0, -3.5, -3.0, -2.0]  # sorted low to high

        selected = _stability_ranking(
            _make_sample_set(samples, energies),
            num_fragments=3,
            selection_threshold=0.5,   # take bottom 50% = 2 samples
            inclusion_threshold=0.5,
        )

        # Fragment 0 is in all samples including the 2 lowest-energy ones
        assert 0 in selected

    def test_rare_fragment_excluded(self):
        """Fragment appearing in few low-energy solutions should be excluded."""
        samples = [
            {0: 1, 1: 1, 2: 0},
            {0: 1, 1: 1, 2: 0},
            {0: 1, 1: 0, 2: 1},
            {0: 0, 1: 0, 2: 1},
        ]
        energies = [-4.0, -3.5, -3.0, -2.0]

        selected = _stability_ranking(
            _make_sample_set(samples, energies),
            num_fragments=3,
            selection_threshold=0.5,   # bottom 2 samples
            inclusion_threshold=0.5,
        )

        # Fragment 2 appears in 0 of the 2 lowest-energy samples
        assert 2 not in selected

    def test_threshold_boundary(self):
        """Fragment at exactly inclusion_threshold should be included."""
        # 4 samples, selection_threshold=1.0 => use all 4
        samples = [
            {0: 1, 1: 0},
            {0: 1, 1: 0},
            {0: 0, 1: 1},
            {0: 0, 1: 1},
        ]
        energies = [-4.0, -3.0, -2.0, -1.0]

        selected = _stability_ranking(
            _make_sample_set(samples, energies),
            num_fragments=2,
            selection_threshold=1.0,   # use all samples
            inclusion_threshold=0.5,   # need >=50% => 2/4 = 50% exactly
        )

        # Both fragments appear in exactly 50% => both should be included
        assert 0 in selected
        assert 1 in selected

    def test_sorted_by_frequency(self):
        """Selected fragments should be sorted descending by inclusion frequency."""
        samples = [
            {0: 1, 1: 1, 2: 1},
            {0: 0, 1: 1, 2: 1},
            {0: 0, 1: 0, 2: 1},
            {0: 0, 1: 0, 2: 1},
        ]
        energies = [-4.0, -3.0, -2.0, -1.0]

        selected = _stability_ranking(
            _make_sample_set(samples, energies),
            num_fragments=3,
            selection_threshold=1.0,
            inclusion_threshold=0.25,
        )

        # freq: frag2=100%, frag1=50%, frag0=25%
        assert selected[0] == 2  # highest frequency first

    def test_empty_selection_when_nothing_passes(self):
        """If no fragment meets inclusion_threshold, return empty list."""
        samples = [
            {0: 1, 1: 0},
            {0: 0, 1: 1},
        ]
        energies = [-2.0, -1.0]

        selected = _stability_ranking(
            _make_sample_set(samples, energies),
            num_fragments=2,
            selection_threshold=1.0,
            inclusion_threshold=0.99,  # need 99% but each is only 50%
        )

        assert selected == []

    def test_single_sample(self):
        """With one sample, selection_threshold ensures at least 1 is used."""
        samples = [{0: 1, 1: 0, 2: 1}]
        energies = [-3.0]

        selected = _stability_ranking(
            _make_sample_set(samples, energies),
            num_fragments=3,
            selection_threshold=0.25,
            inclusion_threshold=0.5,
        )

        # Only 1 sample, cutoff = max(1, 0.25*1) = 1
        # Frags 0,2 have freq 1.0, frag 1 has freq 0.0
        assert 0 in selected
        assert 2 in selected
        assert 1 not in selected
