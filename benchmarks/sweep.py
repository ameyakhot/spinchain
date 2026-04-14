"""Hyperparameter sweep over cached synthetic problems.

Runs the full SpinChain pipeline with varying hyperparameters, including
verification terms (phi, psi, omega) tested on arithmetic-error chains.

Usage:
    uv run python -c "from benchmarks.sweep import main; main()"
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

from spinchain.formulation.coefficient_builder import (
    CoefficientBuilder,
    _extract_numbers,
    _arithmetic_consistency,
    _verify_arithmetic,
)
from spinchain.formulation.fragment_extractor import FragmentExtractor
from spinchain.formulation.qubo_builder import QUBOBuilder
from spinchain.solvers.simulated_annealing import SimulatedAnnealingSolver

from benchmarks.cache import ChainCache
from benchmarks.datasets.gsm8k import GSM8KLoader
from benchmarks.extractors import extract_answer
from benchmarks.scoring import score

# Sweep grid — verification terms
PHI_VALUES = [0.0, 1.0, 2.0, 4.0]
PSI_VALUES = [0.0, 1.0, 2.0]
OMEGA_VALUES = [0.0, 1.0, 2.0]

# Fixed
FIXED_MU = 1.0
FIXED_ALPHA = 0.5
FIXED_BETA = 1.0
FIXED_LAMBDA_SIM = 0.3


# Arithmetic-error test chains for gsm8k_2
# Ground truth: 65000 (80000 + 50000 = 130000, 130000 * 1.5 = 195000, 195000 - 130000 = 65000)
ARITHMETIC_ERROR_CHAINS = {
    "gsm8k_0": [
        # Same as before — all correct, answer=18
        "Janet has 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. So she uses 7 eggs. 16 - 7 = 9. She sells the rest at $2 each. 9 * 2 = 18. The answer is 18.",
        "Janet gets 16 eggs daily. She eats 3 and uses 4 for muffins, that is 7 used. 16 - 7 = 9 eggs remaining. At $2 each, she makes 9 * 2 = 18 dollars. The answer is 18.",
        "Janet has 16 eggs. She eats 3 and bakes 4. 3 + 4 = 7. She has 16 - 7 = 9 left. She sells at $2 each: 9 * 2 = 18. The answer is 18.",
    ],
    "gsm8k_1": [
        # Same as before — 2 correct (answer=3), 1 wrong (answer=4)
        "A robe takes 2 bolts of blue fiber and half that much white fiber. Half of 2 is 1. So total is 2 + 1 = 3 bolts. The answer is 3.",
        "Two bolts of blue fiber and half as much white fiber. Half of 2 is 1. Total fiber needed: 2 + 1 = 3 bolts. The answer is 3.",
        "A robe needs 2 blue bolts and half that in white, which is 2 / 2 = 1. But wait, maybe half the robe is white? That would be 2 + 2 = 4. The answer is 4.",
    ],
    "gsm8k_2": [
        # WRONG — arithmetic error: 80000 + 50000 = 120000 (should be 130000)
        "Josh buys a house for 80000 and puts 50000 into repairs. Total cost: 80000 + 50000 = 120000. He sells at 150% of cost. 120000 * 1.5 = 180000. Profit = 180000 - 120000 = 60000. The answer is 60000.",
        # WRONG — same arithmetic error
        "Cost is 80000 + 50000 = 120000. At 150%, he sells for 120000 * 1.5 = 180000. Profit: 180000 - 120000 = 60000. The answer is 60000.",
        # CORRECT
        "80000 + 50000 = 130000. He sells for 130000 * 1.5 = 195000. Profit = 195000 - 130000 = 65000. The answer is 65000.",
    ],
}

GROUND_TRUTHS = {"gsm8k_0": "18", "gsm8k_1": "3", "gsm8k_2": "65000"}


def stability_ranking(
    sample_set, num_fragments: int,
    selection_threshold: float = 0.25,
    inclusion_threshold: float = 0.50,
) -> list[int]:
    """Select fragments via stability ranking over low-energy solutions."""
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

    selected = [i for i in range(num_fragments) if inclusion_freq[i] >= inclusion_threshold]
    selected.sort(key=lambda i: -inclusion_freq[i])
    return selected


def precompute_cluster_data(
    fragments: list[str],
    sources: list[set[int]],
    chains: list[str],
    dataset: str,
) -> dict:
    """Pre-compute cluster and verification arrays."""
    r = len(fragments)

    # Extract answer from each chain → build clusters
    chain_answers = {}
    for i, chain in enumerate(chains):
        ans = extract_answer(chain, dataset)
        if ans:
            chain_answers[i] = ans

    answer_clusters: dict[str, set[int]] = defaultdict(set)
    for idx, ans in chain_answers.items():
        answer_clusters[ans].add(idx)
    answer_clusters = dict(answer_clusters)

    # Verification scores per fragment
    verification_scores = np.array([_verify_arithmetic(frag) for frag in fragments])

    # Cluster integrity: any verified-wrong fragment makes integrity negative
    # A single arithmetic error taints the entire cluster
    cluster_integrity: dict[str, float] = {}
    for answer, chain_set in answer_clusters.items():
        correct_count = 0
        wrong_count = 0
        for i in range(r):
            if sources[i] & chain_set:
                if verification_scores[i] > 0:
                    correct_count += 1
                elif verification_scores[i] < 0:
                    wrong_count += 1
        if wrong_count > 0:
            cluster_integrity[answer] = -1.0  # tainted
        elif correct_count > 0:
            cluster_integrity[answer] = 1.0   # clean
        else:
            cluster_integrity[answer] = 0.0   # no evidence

    # Per-fragment cluster integrity
    cluster_integrity_per_fragment = np.zeros(r)
    for i in range(r):
        for answer, chain_set in answer_clusters.items():
            if sources[i] & chain_set:
                cluster_integrity_per_fragment[i] = cluster_integrity[answer]
                break

    # Verification agreement (quadratic)
    verification_agreement = np.zeros((r, r))
    for i in range(r):
        if verification_scores[i] == 0:
            continue
        sign_i = 1.0 if verification_scores[i] > 0 else -1.0
        for j in range(i + 1, r):
            if verification_scores[j] == 0:
                continue
            sign_j = 1.0 if verification_scores[j] > 0 else -1.0
            verification_agreement[i, j] = sign_i * sign_j
            verification_agreement[j, i] = sign_i * sign_j

    return {
        "answer_clusters": answer_clusters,
        "cluster_integrity": cluster_integrity,
        "verification_scores": verification_scores,
        "cluster_integrity_per_fragment": cluster_integrity_per_fragment,
        "verification_agreement": verification_agreement,
    }


def run_config(
    fragments: list[str],
    sources: list[set[int]],
    embeddings: np.ndarray,
    num_completions: int,
    cluster_data: dict,
    phi: float, psi: float, omega: float,
    ground_truth: str, dataset: str,
) -> dict:
    """Run one hyperparameter config through the full pipeline."""
    r = len(fragments)
    builder = CoefficientBuilder(
        mu=FIXED_MU, alpha=FIXED_ALPHA, beta=FIXED_BETA,
        lambda_sim=FIXED_LAMBDA_SIM,
    )
    linear_w = builder.compute_linear_weights(sources, num_completions)

    # Verification linear terms
    if phi > 0:
        linear_w = linear_w + (-phi * cluster_data["verification_scores"])
    if psi > 0:
        linear_w = linear_w + (-psi * cluster_data["cluster_integrity_per_fragment"])

    quadratic_w = builder.compute_quadratic_weights(sources, num_completions, embeddings)

    # Verification quadratic term
    if omega > 0:
        quadratic_w = quadratic_w + (-omega * cluster_data["verification_agreement"])

    qubo_builder = QUBOBuilder()
    bqm = qubo_builder.build(linear_w, quadratic_w)

    solver = SimulatedAnnealingSolver(num_reads=100, num_sweeps=1000)
    sample_set = solver.solve(bqm)

    selected_indices = stability_ranking(sample_set, r)
    selected_fragments = [fragments[i] for i in selected_indices]
    text = " ".join(selected_fragments)
    predicted = extract_answer(text, dataset)

    correct = score(predicted, ground_truth, dataset) if predicted else False
    min_energy = min(float(d.energy) for d in sample_set.data())

    return {
        "predicted": predicted,
        "correct": correct,
        "num_selected": len(selected_indices),
        "min_energy": min_energy,
    }


def main() -> None:
    # Load dataset for problem metadata (questions, ground truths)
    loader = GSM8KLoader()
    problems = loader.load(limit=3)

    # Use arithmetic-error chains (hardcoded, no cache needed)
    extractor = FragmentExtractor(similarity_threshold=0.90)
    problem_data = []

    for problem in problems:
        chains = ARITHMETIC_ERROR_CHAINS.get(problem.id)
        if chains is None:
            continue
        gt = GROUND_TRUTHS[problem.id]
        problem.ground_truth = gt  # override with our known ground truth

        fragments = extractor.extract_fragments(chains)
        cluster_data = precompute_cluster_data(fragments, extractor.fragment_sources, chains, problem.dataset)

        problem_data.append({
            "problem": problem,
            "chains": chains,
            "fragments": fragments,
            "sources": extractor.fragment_sources,
            "embeddings": extractor.fragment_embeddings.copy(),
            "num_completions": extractor.num_completions,
            "cluster_data": cluster_data,
        })

    # Print verification info
    print("ARITHMETIC-ERROR TEST CHAINS")
    print("=" * 80)
    for pd in problem_data:
        pid = pd["problem"].id
        cd = pd["cluster_data"]
        vs = cd["verification_scores"]
        verified_correct = int((vs > 0).sum())
        verified_wrong = int((vs < 0).sum())
        unverifiable = int((vs == 0).sum())
        clusters = cd["answer_clusters"]
        integrity = cd["cluster_integrity"]
        print(f"  {pid}: truth={pd['problem'].ground_truth}, "
              f"clusters={dict((k, len(v)) for k, v in clusters.items())}")
        print(f"    fragments={len(pd['fragments'])}, "
              f"verified_correct={verified_correct}, "
              f"verified_wrong={verified_wrong}, "
              f"unverifiable={unverifiable}")
        print(f"    cluster_integrity={integrity}")

    # Identify disagreement problems
    disagree_ids = set()
    for pd in problem_data:
        if len(pd["cluster_data"]["answer_clusters"]) > 1:
            disagree_ids.add(pd["problem"].id)

    # Generate config grid
    configs = list(itertools.product(PHI_VALUES, PSI_VALUES, OMEGA_VALUES))
    total_solves = len(configs) * len(problem_data)

    print(f"\nVERIFICATION SWEEP")
    print(f"{'=' * 80}")
    print(f"Problems: {len(problem_data)}, Configs: {len(configs)}, Total solves: {total_solves}")
    print(f"Disagreement problems: {disagree_ids}")
    print(f"Fixed: mu={FIXED_MU}, alpha={FIXED_ALPHA}, beta={FIXED_BETA}, lambda_sim={FIXED_LAMBDA_SIM}")
    print(f"Sweep: phi={PHI_VALUES}, psi={PSI_VALUES}, omega={OMEGA_VALUES}")
    print()

    # Run sweep
    results = {}
    for i, (phi, psi, omega) in enumerate(configs):
        config_key = (phi, psi, omega)
        results[config_key] = {}
        for pd in problem_data:
            r = run_config(
                pd["fragments"], pd["sources"], pd["embeddings"], pd["num_completions"],
                pd["cluster_data"],
                phi, psi, omega,
                pd["problem"].ground_truth, pd["problem"].dataset,
            )
            results[config_key][pd["problem"].id] = r

    print(f"  {len(configs)}/{len(configs)} configs done.\n")

    # Report: baseline (no verification)
    baseline_key = (0.0, 0.0, 0.0)
    print(f"BASELINE (phi=0, psi=0, omega=0 — no verification)")
    print("-" * 80)
    for pd in problem_data:
        pid = pd["problem"].id
        r = results[baseline_key][pid]
        gt = pd["problem"].ground_truth
        mark = "correct" if r["correct"] else "WRONG"
        disagree_mark = " [disagree]" if pid in disagree_ids else ""
        print(f"  {pid}: predicted={r['predicted']}, truth={gt}, {mark}{disagree_mark}")
    print()

    # Find majority-vote failures at baseline
    majority_vote_failures = set()
    for pd in problem_data:
        pid = pd["problem"].id
        if pid not in disagree_ids:
            continue
        r = results[baseline_key][pid]
        if not r["correct"]:
            majority_vote_failures.add(pid)

    # Report: configs that fix failures
    if majority_vote_failures:
        print(f"CONFIGS THAT FIX MAJORITY-VOTE FAILURES ({majority_vote_failures})")
        print("-" * 80)
        winners = []
        for config_key, problem_results in results.items():
            fixes_all = all(
                problem_results[pid]["correct"]
                for pid in majority_vote_failures
            )
            if fixes_all:
                breaks = [
                    pid for pid in disagree_ids - majority_vote_failures
                    if not problem_results[pid]["correct"]
                ]
                winners.append((config_key, breaks))

        if winners:
            clean_winners = [(k, b) for k, b in winners if not b]
            dirty_winners = [(k, b) for k, b in winners if b]

            if clean_winners:
                print(f"\n  CLEAN WINS (fix failure, break nothing): {len(clean_winners)}")
                for config_key, _ in clean_winners:
                    phi, psi, omega = config_key
                    print(f"    phi={phi}  psi={psi}  omega={omega}")

            if dirty_winners:
                print(f"\n  Dirty wins (fix failure, break something): {len(dirty_winners)}")
                for config_key, breaks in dirty_winners[:5]:
                    phi, psi, omega = config_key
                    print(f"    phi={phi}  psi={psi}  omega={omega}  (breaks: {breaks})")

            print(f"\n  Total: {len(winners)} / {len(configs)} configs fix the failure")
            print(f"  Clean: {len(clean_winners)}, Dirty: {len(dirty_winners)}")

            if clean_winners:
                print("\n  >>> VERIFICATION TERMS OUTPERFORM MAJORITY VOTE <<<")
        else:
            print("  NO CONFIG FIXES THE FAILURE.")
    else:
        print("Baseline already gets all disagreement cases correct.")

    print()

    # Top 10
    print("TOP 10 CONFIGS BY DISAGREEMENT ACCURACY")
    print("-" * 80)
    config_scores = []
    for config_key, problem_results in results.items():
        disagree_correct = sum(
            1 for pid in disagree_ids if problem_results[pid]["correct"]
        )
        all_correct = sum(1 for r in problem_results.values() if r["correct"])
        config_scores.append((config_key, disagree_correct, all_correct))

    config_scores.sort(key=lambda x: (-x[1], -x[2]))
    for config_key, dc, ac in config_scores[:10]:
        phi, psi, omega = config_key
        tag = " ← baseline" if config_key == baseline_key else ""
        print(f"  phi={phi:<4}  psi={psi:<4}  omega={omega:<4}  "
              f"disagree={dc}/{len(disagree_ids)}  overall={ac}/{len(problem_data)}{tag}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
