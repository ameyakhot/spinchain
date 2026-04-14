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

    def print_diagnostics(self) -> None:
        """Print QUBO coefficient diagnostics for SpinChain results."""
        has_diag = any(
            "spinchain" in r.results
            and "diagnostics" in r.results["spinchain"].metadata
            for r in self.records
        )
        if not has_diag:
            return

        print(f"{'=' * 90}")
        print("COEFFICIENT DIAGNOSTICS")
        print(f"{'=' * 90}")
        header = (
            f"{'Problem':<16} {'|linear| mean':>14} {'|quad| mean':>14} "
            f"{'Ratio':>8} {'Frags':>6} {'Co-occur%':>10} {'Sim mean':>10}"
        )
        print(header)
        print("-" * len(header))

        for r in self.records:
            if "spinchain" not in r.results:
                continue
            diag = r.results["spinchain"].metadata.get("diagnostics")
            if not diag or diag.get("skipped"):
                continue

            lin_mean = diag["linear_magnitude"]["mean"]
            quad_mean = diag["quadratic_magnitude"]["mean"]
            ratio = diag["linear_vs_quadratic_ratio"]
            frags = diag["num_fragments"]
            co_occ = diag["co_occurrence_density"]
            sim_mean = diag["similarity_stats"]["mean"]

            ratio_str = f"{ratio:.1f}x" if ratio != float("inf") else "inf"
            print(
                f"{r.problem_id:<16} {lin_mean:>14.6f} {quad_mean:>14.6f} "
                f"{ratio_str:>8} {frags:>6} {co_occ:>9.1%} {sim_mean:>10.4f}"
            )

        print(f"{'=' * 90}\n")

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
