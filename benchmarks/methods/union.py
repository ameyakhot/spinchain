"""Union baseline: all deduplicated fragments, no QUBO optimization."""

from __future__ import annotations

from spinchain.formulation.fragment_extractor import FragmentExtractor

from benchmarks.datasets.base import Problem
from benchmarks.extractors import extract_answer
from benchmarks.methods.base import MethodResult, count_tokens, total_chain_tokens
from benchmarks.scoring import score


class UnionMethod:
    name = "union"

    def __init__(self, similarity_threshold: float = 0.85):
        self.extractor = FragmentExtractor(similarity_threshold=similarity_threshold)

    def run(self, chains: list[str], problem: Problem) -> MethodResult:
        fragments = self.extractor.extract_fragments(chains)
        text = " ".join(fragments)
        predicted = extract_answer(text, problem.dataset, problem.choices)

        return MethodResult(
            method=self.name,
            predicted_answer=predicted,
            correct=score(predicted, problem.ground_truth, problem.dataset) if predicted else False,
            metadata={
                "num_fragments": len(fragments),
                "input_tokens": total_chain_tokens(chains),
                "output_tokens": count_tokens(text),
            },
        )
