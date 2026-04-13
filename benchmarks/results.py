"""Accumulate benchmark results and produce summary reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from benchmarks.methods.base import MethodResult


@dataclass
class ProblemRecord:
    problem_id: str
    ground_truth: str
    agreement: bool  # True if all chains produced the same answer
    results: dict[str, MethodResult] = field(default_factory=dict)


class ResultsAccumulator:
    def __init__(self) -> None:
        self.records: list[ProblemRecord] = []

    def add(self, record: ProblemRecord) -> None:
        self.records.append(record)

    def summary(self) -> dict:
        """Compute summary statistics."""
        if not self.records:
            return {"total": 0}

        methods = set()
        for r in self.records:
            methods.update(r.results.keys())

        agreement_count = sum(1 for r in self.records if r.agreement)
        disagreement_count = len(self.records) - agreement_count

        per_method = {}
        for method in sorted(methods):
            correct_all = sum(
                1 for r in self.records
                if method in r.results and r.results[method].correct
            )
            correct_disagree = sum(
                1 for r in self.records
                if not r.agreement and method in r.results and r.results[method].correct
            )
            no_answer = sum(
                1 for r in self.records
                if method in r.results and r.results[method].predicted_answer is None
            )
            per_method[method] = {
                "accuracy_all": correct_all / len(self.records) if self.records else 0,
                "correct_all": correct_all,
                "accuracy_disagree": (
                    correct_disagree / disagreement_count if disagreement_count > 0 else None
                ),
                "correct_disagree": correct_disagree,
                "no_answer": no_answer,
            }

        return {
            "total": len(self.records),
            "agreement": agreement_count,
            "disagreement": disagreement_count,
            "agreement_rate": agreement_count / len(self.records),
            "methods": per_method,
        }

    def print_summary(self) -> None:
        s = self.summary()
        total = s["total"]
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK RESULTS ({total} problems)")
        print(f"{'=' * 70}")
        if total == 0:
            print("No results to report.")
            return
        print(f"Agreement:    {s['agreement']:>4} / {total} ({s['agreement_rate']:.1%})")
        print(f"Disagreement: {s['disagreement']:>4} / {total}")
        print()

        header = f"{'Method':<20} {'Overall':>10} {'Disagree':>10} {'No Answer':>10}"
        print(header)
        print("-" * len(header))

        for method, stats in s.get("methods", {}).items():
            acc_all = f"{stats['accuracy_all']:.1%}"
            acc_dis = f"{stats['accuracy_disagree']:.1%}" if stats["accuracy_disagree"] is not None else "N/A"
            no_ans = str(stats["no_answer"])
            print(f"{method:<20} {acc_all:>10} {acc_dis:>10} {no_ans:>10}")

        print(f"{'=' * 70}\n")

    def save_json(self, path: str) -> None:
        data = {
            "summary": self.summary(),
            "records": [
                {
                    "problem_id": r.problem_id,
                    "ground_truth": r.ground_truth,
                    "agreement": r.agreement,
                    "results": {
                        name: {
                            "method": mr.method,
                            "predicted_answer": mr.predicted_answer,
                            "correct": mr.correct,
                            "metadata": mr.metadata,
                        }
                        for name, mr in r.results.items()
                    },
                }
                for r in self.records
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
