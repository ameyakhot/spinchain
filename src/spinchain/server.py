"""SpinChain MCP server.

Exposes the optimize_reasoning tool via stdio transport.
Claude calls this mid-conversation to optimize its own reasoning chains.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from mcp.server.fastmcp import FastMCP

from spinchain.formulation.fragment_extractor import FragmentExtractor
from spinchain.formulation.coefficient_builder import CoefficientBuilder
from spinchain.formulation.qubo_builder import QUBOBuilder
from spinchain.solvers.simulated_annealing import SimulatedAnnealingSolver
from spinchain.tracing import get_tracer

logger = logging.getLogger("spinchain.server")

mcp = FastMCP(
    "SpinChain",
    instructions=(
        "SpinChain optimizes LLM reasoning by formulating reasoning path selection "
        "as a QUBO problem solved by simulated annealing. Pass multiple diverse "
        "reasoning chains to optimize_reasoning and get back the best fragments."
    ),
)


def _stability_ranking(
    sample_set,
    num_fragments: int,
    selection_threshold: float = 0.25,
    inclusion_threshold: float = 0.50,
) -> list[int]:
    """Select fragments via stability ranking over low-energy solutions.

    From QCR-LLM: take the bottom selection_threshold fraction of solutions
    by energy, compute inclusion frequency of each fragment, keep those
    above inclusion_threshold.
    """
    samples = list(sample_set.samples())
    energies = [float(d.energy) for d in sample_set.data()]

    sorted_indices = np.argsort(energies)
    cutoff = max(1, int(len(sorted_indices) * selection_threshold))
    low_energy_indices = sorted_indices[:cutoff]

    inclusion_freq = np.zeros(num_fragments)
    for idx in low_energy_indices:
        sample = samples[idx]
        for var_idx in range(num_fragments):
            if sample.get(var_idx, 0) == 1:
                inclusion_freq[var_idx] += 1

    inclusion_freq /= cutoff

    selected = [
        i for i in range(num_fragments)
        if inclusion_freq[i] >= inclusion_threshold
    ]
    selected.sort(key=lambda i: -inclusion_freq[i])

    return selected


@mcp.tool()
def optimize_reasoning(
    completions: list[str],
    num_reads: int = 100,
    num_sweeps: int = 1000,
    similarity_threshold: float = 0.85,
    selection_threshold: float = 0.25,
    inclusion_threshold: float = 0.50,
    cardinality_k: int | None = None,
) -> str:
    """Optimize LLM reasoning chains using QUBO/Ising formulation and simulated annealing.

    Takes multiple diverse reasoning chains (e.g., chain-of-thought completions),
    extracts and deduplicates reasoning fragments, formulates a QUBO problem that
    balances fragment popularity, stability, co-occurrence, and semantic diversity,
    then solves it with simulated annealing to select the optimal reasoning subset.

    Args:
        completions: List of diverse reasoning chain strings to optimize.
            Generate these by thinking through the problem multiple times
            with different approaches. Minimum 2 completions recommended,
            5-20 for best results.
        num_reads: Number of SA samples (higher = more thorough search).
        num_sweeps: Number of SA sweeps per sample (higher = better convergence).
        similarity_threshold: Cosine similarity threshold for fragment deduplication (0-1).
        selection_threshold: Fraction of low-energy solutions used for stability ranking.
        inclusion_threshold: Min frequency in low-energy set to include a fragment.
        cardinality_k: If set, adds a penalty to select approximately this many fragments.

    Returns:
        JSON string with optimized fragments and solver metadata.
    """
    tracer = get_tracer()
    trace_id = tracer.start_trace({
        "num_completions": len(completions),
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "similarity_threshold": similarity_threshold,
        "selection_threshold": selection_threshold,
        "inclusion_threshold": inclusion_threshold,
        "cardinality_k": cardinality_k,
    })

    try:
        if len(completions) < 2:
            result = {
                "selected_fragments": completions,
                "all_fragments": completions,
                "selected_indices": list(range(len(completions))),
                "num_completions": len(completions),
                "num_fragments": len(completions),
                "solver": "simulated_annealing",
                "min_energy": None,
                "energies": [],
                "fallback": True,
                "reason": "Need at least 2 completions to optimize.",
            }
            tracer.finish_trace(trace_id, {"fallback": True, "reason": result["reason"]})
            return json.dumps(result)

        # Fragment extraction
        stage = tracer.start_stage(trace_id, "fragment_extraction")
        extractor = FragmentExtractor(similarity_threshold=similarity_threshold)
        fragments = extractor.extract_fragments(completions)
        stage.metadata["num_raw_fragments"] = sum(
            len(s) for s in extractor.fragment_sources
        )
        stage.metadata["num_merged_fragments"] = len(fragments)
        tracer.end_stage(trace_id, stage)

        if len(fragments) < 2:
            result = {
                "selected_fragments": fragments if fragments else [completions[0]],
                "all_fragments": fragments,
                "selected_indices": list(range(len(fragments))),
                "num_completions": len(completions),
                "num_fragments": len(fragments),
                "solver": "simulated_annealing",
                "min_energy": None,
                "energies": [],
                "fallback": True,
                "reason": "Fewer than 2 unique fragments extracted.",
            }
            tracer.finish_trace(trace_id, {"fallback": True, "reason": result["reason"]})
            return json.dumps(result)

        # QUBO formulation
        stage = tracer.start_stage(trace_id, "qubo_formulation")
        coeff_builder = CoefficientBuilder()
        linear_w = coeff_builder.compute_linear_weights(
            extractor.fragment_sources,
            extractor.num_completions,
        )
        quadratic_w = coeff_builder.compute_quadratic_weights(
            extractor.fragment_sources,
            extractor.num_completions,
            extractor.fragment_embeddings,
        )
        qubo_builder = QUBOBuilder()
        bqm = qubo_builder.build(linear_w, quadratic_w, target_fragments=cardinality_k)
        stage.metadata["num_linear_terms"] = len(bqm.linear)
        stage.metadata["num_quadratic_terms"] = len(bqm.quadratic)
        tracer.end_stage(trace_id, stage)

        # Solve
        stage = tracer.start_stage(trace_id, "simulated_annealing")
        solver = SimulatedAnnealingSolver(num_reads=num_reads, num_sweeps=num_sweeps)
        sample_set = solver.solve(bqm)
        energies = [float(d.energy) for d in sample_set.data()]
        stage.metadata["num_samples"] = len(energies)
        stage.metadata["min_energy"] = min(energies) if energies else None
        stage.metadata["max_energy"] = max(energies) if energies else None
        tracer.end_stage(trace_id, stage)

        # Stability ranking
        stage = tracer.start_stage(trace_id, "stability_ranking")
        selected_indices = _stability_ranking(
            sample_set, len(fragments), selection_threshold, inclusion_threshold
        )
        stage.metadata["num_selected"] = len(selected_indices)
        tracer.end_stage(trace_id, stage)

        selected_fragments = [fragments[i] for i in selected_indices]

        result = {
            "selected_fragments": selected_fragments,
            "all_fragments": fragments,
            "selected_indices": selected_indices,
            "num_completions": len(completions),
            "num_fragments": len(fragments),
            "solver": "simulated_annealing",
            "min_energy": min(energies) if energies else None,
            "energies": sorted(energies)[:10],
            "fallback": False,
        }

        tracer.finish_trace(trace_id, {
            "fallback": False,
            "num_fragments": len(fragments),
            "num_selected": len(selected_indices),
            "min_energy": result["min_energy"],
        })

        return json.dumps(result)

    except Exception as e:
        tracer.finish_trace(trace_id, {}, error=str(e))
        raise


def main():
    """Start the SpinChain MCP server with stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
