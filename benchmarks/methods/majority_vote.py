"""Majority vote (self-consistency) baseline."""

from __future__ import annotations

from collections import Counter

from benchmarks.datasets.base import Problem
from benchmarks.extractors import extract_answer
from benchmarks.methods.base import MethodResult
from benchmarks.scoring import score


class MajorityVote:
    name = "majority_vote"

    def run(self, chains: list[str], problem: Problem) -> MethodResult:
        answers = []
        for chain in chains:
            ans = extract_answer(chain, problem.dataset, problem.choices)
            if ans is not None:
                answers.append(ans)

        if not answers:
            return MethodResult(
                method=self.name,
                predicted_answer=None,
                correct=False,
                metadata={"vote_counts": {}, "extractable": 0},
            )

        counter = Counter(answers)
        predicted = counter.most_common(1)[0][0]
        return MethodResult(
            method=self.name,
            predicted_answer=predicted,
            correct=score(predicted, problem.ground_truth, problem.dataset),
            metadata={
                "vote_counts": dict(counter),
                "extractable": len(answers),
                "total_chains": len(chains),
            },
        )
