"""Shared fixtures for SpinChain tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def two_completion_sources() -> list[set[int]]:
    """4 fragments from 2 completions with known overlap.

    Fragment 0: appears in both completions (popular)
    Fragment 1: only in completion 0
    Fragment 2: only in completion 1
    Fragment 3: appears in both completions (popular)
    """
    return [
        {0, 1},  # frag 0: both
        {0},     # frag 1: comp 0 only
        {1},     # frag 2: comp 1 only
        {0, 1},  # frag 3: both
    ]


@pytest.fixture
def orthogonal_embeddings_4() -> np.ndarray:
    """4 orthogonal unit vectors — zero cosine similarity between all pairs."""
    return np.eye(4)


@pytest.fixture
def similar_embeddings_4() -> np.ndarray:
    """4 embeddings where 0,1 are similar and 2,3 are similar."""
    emb = np.array([
        [1.0, 0.1, 0.0, 0.0],
        [0.9, 0.2, 0.0, 0.0],  # similar to 0
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 0.0, 0.9, 0.2],  # similar to 2
    ])
    return emb
