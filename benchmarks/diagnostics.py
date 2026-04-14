"""QUBO coefficient diagnostics — dump linear vs. quadratic magnitudes."""

from __future__ import annotations

import numpy as np

from spinchain.formulation.coefficient_builder import CoefficientBuilder
from spinchain.formulation.fragment_extractor import FragmentExtractor

from benchmarks.config import BenchmarkConfig


def analyze_coefficients(
    chains: list[str],
    config: BenchmarkConfig,
    mu: float = 1.0,
    alpha: float = 0.5,
    beta: float = 1.0,
    lambda_sim: float = 0.3,
) -> dict:
    """Run the formulation pipeline and return coefficient diagnostics.

    This mirrors the first half of server.py's optimize_reasoning() but
    captures all intermediate data instead of solving.
    """
    extractor = FragmentExtractor(similarity_threshold=config.similarity_threshold)
    fragments = extractor.extract_fragments(chains)

    if len(fragments) < 2:
        return {"num_fragments": len(fragments), "skipped": True}

    sources = extractor.fragment_sources
    embeddings = extractor.fragment_embeddings
    n = extractor.num_completions
    r = len(fragments)

    # Compute coefficients (same as server.py lines 158-167)
    builder = CoefficientBuilder(mu=mu, alpha=alpha, beta=beta, lambda_sim=lambda_sim)
    linear_w = builder.compute_linear_weights(sources, n)
    quadratic_w = builder.compute_quadratic_weights(sources, n, embeddings)

    # Fragment popularity
    popularity = [len(s) / n for s in sources]

    # Linear magnitude stats
    abs_linear = np.abs(linear_w)
    linear_mag = _stats(abs_linear)

    # Quadratic magnitude stats (upper triangle only, skip zeros)
    upper_indices = np.triu_indices(r, k=1)
    quad_upper = quadratic_w[upper_indices]
    abs_quad = np.abs(quad_upper)
    nonzero_mask = abs_quad > 1e-10
    abs_quad_nonzero = abs_quad[nonzero_mask]
    quadratic_mag = _stats(abs_quad_nonzero) if len(abs_quad_nonzero) > 0 else _empty_stats()

    # The key diagnostic: ratio of linear to quadratic magnitude
    ratio = (
        linear_mag["mean"] / quadratic_mag["mean"]
        if quadratic_mag["mean"] > 0
        else float("inf")
    )

    # Co-occurrence density
    total_pairs = r * (r - 1) // 2
    co_occurring_pairs = 0
    for i in range(r):
        for j in range(i + 1, r):
            if len(sources[i] & sources[j]) > 0:
                co_occurring_pairs += 1
    co_occurrence_density = co_occurring_pairs / total_pairs if total_pairs > 0 else 0

    # Similarity stats (cosine similarity upper triangle)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    sim_upper = sim_matrix[upper_indices]
    similarity_stats = _stats(sim_upper)

    return {
        "num_fragments": r,
        "skipped": False,
        "popularity": [round(p, 4) for p in popularity],
        "linear_weights": [round(float(w), 6) for w in linear_w],
        "linear_magnitude": linear_mag,
        "quadratic_magnitude": quadratic_mag,
        "linear_vs_quadratic_ratio": round(ratio, 1),
        "nonzero_quadratic_pairs": int(nonzero_mask.sum()),
        "total_pairs": total_pairs,
        "co_occurrence_density": round(co_occurrence_density, 4),
        "similarity_stats": similarity_stats,
    }


def _stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return _empty_stats()
    return {
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
    }


def _empty_stats() -> dict:
    return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
