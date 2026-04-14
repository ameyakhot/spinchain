"""Random selection baseline (floor)."""

from __future__ import annotations

import random

from benchmarks.datasets.base import Problem
from benchmarks.extractors import extract_answer
from benchmarks.methods.base import MethodResult, count_tokens, total_chain_tokens
from benchmarks.scoring import score


class RandomSelection:
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def run(self, chains: list[str], problem: Problem) -> MethodResult:
        rng = random.Random(hash(problem.id) ^ self.seed)
        chain = rng.choice(chains)
        predicted = extract_answer(chain, problem.dataset, problem.choices)
        return MethodResult(
            method=self.name,
            predicted_answer=predicted,
            correct=score(predicted, problem.ground_truth, problem.dataset) if predicted else False,
            metadata={
                "selected_chain_index": chains.index(chain),
                "input_tokens": total_chain_tokens(chains),
                "output_tokens": count_tokens(chain),
            },
        )
