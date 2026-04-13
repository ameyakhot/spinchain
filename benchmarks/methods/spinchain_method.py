"""SpinChain QUBO/SA optimization method."""

from __future__ import annotations

import json

from spinchain.server import optimize_reasoning

from benchmarks.config import BenchmarkConfig
from benchmarks.datasets.base import Problem
from benchmarks.extractors import extract_answer
from benchmarks.methods.base import MethodResult
from benchmarks.scoring import score


class SpinChainMethod:
    name = "spinchain"

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()

    def run(self, chains: list[str], problem: Problem) -> MethodResult:
        result_json = optimize_reasoning(
            completions=chains,
            num_reads=self.config.num_reads,
            num_sweeps=self.config.num_sweeps,
            similarity_threshold=self.config.similarity_threshold,
            selection_threshold=self.config.selection_threshold,
            inclusion_threshold=self.config.inclusion_threshold,
        )
        result = json.loads(result_json)

        if result.get("fallback"):
            text = " ".join(chains)
        else:
            text = " ".join(result["selected_fragments"])

        predicted = extract_answer(text, problem.dataset, problem.choices)

        return MethodResult(
            method=self.name,
            predicted_answer=predicted,
            correct=score(predicted, problem.ground_truth, problem.dataset) if predicted else False,
            metadata={
                "min_energy": result.get("min_energy"),
                "num_fragments": result.get("num_fragments"),
                "num_selected": len(result.get("selected_indices", [])),
                "fallback": result.get("fallback", False),
            },
        )
