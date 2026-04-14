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

    def __init__(self, config: BenchmarkConfig | None = None, diagnostics: bool = False):
        self.config = config or BenchmarkConfig()
        self.diagnostics = diagnostics

    def run(self, chains: list[str], problem: Problem) -> MethodResult:
        chain_answers = [
            extract_answer(chain, problem.dataset, problem.choices)
            for chain in chains
        ]
        result_json = optimize_reasoning(
            completions=chains,
            num_reads=self.config.num_reads,
            num_sweeps=self.config.num_sweeps,
            similarity_threshold=self.config.similarity_threshold,
            selection_threshold=self.config.selection_threshold,
            inclusion_threshold=self.config.inclusion_threshold,
            question=problem.question,
            chain_answers=chain_answers,
        )
        result = json.loads(result_json)

        if result.get("fallback"):
            text = " ".join(chains)
        else:
            text = " ".join(result["selected_fragments"])

        predicted = extract_answer(text, problem.dataset, problem.choices)

        metadata = {
            "min_energy": result.get("min_energy"),
            "num_fragments": result.get("num_fragments"),
            "num_selected": len(result.get("selected_indices", [])),
            "fallback": result.get("fallback", False),
        }

        if self.diagnostics:
            from benchmarks.diagnostics import analyze_coefficients
            diag = analyze_coefficients(chains, self.config)
            metadata["diagnostics"] = diag

        return MethodResult(
            method=self.name,
            predicted_answer=predicted,
            correct=score(predicted, problem.ground_truth, problem.dataset) if predicted else False,
            metadata=metadata,
        )
