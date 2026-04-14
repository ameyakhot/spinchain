"""Hyperparameter sweep over cached synthetic problems.

Runs the full SpinChain pipeline with varying hyperparameters, including
the new cluster-aware terms (delta, epsilon, kappa).

Usage:
    uv run python -c "from benchmarks.sweep import main; main()"
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict

import numpy as np

from spinchain.formulation.coefficient_builder import CoefficientBuilder
from spinchain.formulation.fragment_extractor import FragmentExtractor
from spinchain.formulation.qubo_builder import QUBOBuilder
from spinchain.solvers.simulated_annealing import SimulatedAnnealingSolver

from benchmarks.cache import ChainCache
from benchmarks.datasets.gsm8k import GSM8KLoader
from benchmarks.extractors import extract_answer
from benchmarks.scoring import score

# Sweep grid — fix embedding-space terms, sweep cluster + consistency terms
MU_VALUES = [0.5, 1.0, 2.0]
DELTA_VALUES = [0.0, 1.0]
EPSILON_VALUES = [0.0, 1.0]
KAPPA_VALUES = [0.0, 2.0]
ETA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]

# Fixed (proven ineffective in isolation)
FIXED_ALPHA = 0.5
FIXED_BETA = 1.0
FIXED_LAMBDA_SIM = 0.3
FIXED_GAMMA = 0.0


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
    """Pre-compute cluster-aware arrays (independent of hyperparameters)."""
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
    num_clusters = len(answer_clusters)

    # d_i: cross-cluster agreement (fraction of clusters containing fragment)
    shared_scores = np.zeros(r)
    if num_clusters > 0:
        for i in range(r):
            clusters_containing = sum(
                1 for chains_in_cluster in answer_clusters.values()
                if sources[i] & chains_in_cluster
            )
            shared_scores[i] = clusters_containing / num_clusters

    # a_i: answer anchor flag
    anchor_flags = np.zeros(r)
    pattern = re.compile(r"(?:[Tt]he answer is|####)\s*\S", re.IGNORECASE)
    for i in range(r):
        if pattern.search(fragments[i]):
            anchor_flags[i] = 1.0

    # c_ij: cluster coherence (fraction of clusters where both fragments appear)
    cluster_coherence = np.zeros((r, r))
    if num_clusters > 0:
        for i in range(r):
            for j in range(i + 1, r):
                clusters_both = sum(
                    1 for chains_in_cluster in answer_clusters.values()
                    if (sources[i] & chains_in_cluster) and (sources[j] & chains_in_cluster)
                )
                c_ij = clusters_both / num_clusters
                cluster_coherence[i, j] = c_ij
                cluster_coherence[j, i] = c_ij

    # v_ij: numerical consistency (arithmetic relationships between numbers)
    from spinchain.formulation.coefficient_builder import _extract_numbers, _arithmetic_consistency
    fragment_numbers = [_extract_numbers(frag) for frag in fragments]
    numerical_consistency = np.zeros((r, r))
    for i in range(r):
        for j in range(i + 1, r):
            v_ij = _arithmetic_consistency(fragment_numbers[i], fragment_numbers[j])
            numerical_consistency[i, j] = v_ij
            numerical_consistency[j, i] = v_ij

    return {
        "answer_clusters": answer_clusters,
        "shared_scores": shared_scores,
        "anchor_flags": anchor_flags,
        "cluster_coherence": cluster_coherence,
        "numerical_consistency": numerical_consistency,
        "fragment_numbers": fragment_numbers,
    }


def run_config(
    fragments: list[str],
    sources: list[set[int]],
    embeddings: np.ndarray,
    num_completions: int,
    cluster_data: dict,
    mu: float, delta: float, epsilon: float, kappa: float, eta: float,
    ground_truth: str, dataset: str,
) -> dict:
    """Run one hyperparameter config through the full pipeline."""
    r = len(fragments)
    builder = CoefficientBuilder(
        mu=mu, alpha=FIXED_ALPHA, beta=FIXED_BETA,
        lambda_sim=FIXED_LAMBDA_SIM, gamma=FIXED_GAMMA,
    )
    linear_w = builder.compute_linear_weights(sources, num_completions)

    # Add cluster-aware linear terms
    if delta > 0:
        linear_w = linear_w + (-delta * cluster_data["shared_scores"])
    if kappa > 0:
        linear_w = linear_w + (-kappa * cluster_data["anchor_flags"])

    quadratic_w = builder.compute_quadratic_weights(sources, num_completions, embeddings)

    # Add cluster coherence quadratic term
    if epsilon > 0:
        quadratic_w = quadratic_w + (-epsilon * cluster_data["cluster_coherence"])

    # Add numerical consistency quadratic term
    if eta > 0:
        quadratic_w = quadratic_w + (-eta * cluster_data["numerical_consistency"])

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
    # Load dataset and cache
    loader = GSM8KLoader()
    problems = loader.load(limit=3)
    cache = ChainCache(
        cache_dir="benchmarks/.cache",
        dataset="gsm8k",
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        chains=3,
    )

    # Pre-extract fragments and cluster data for each problem
    problem_data = []
    extractor = FragmentExtractor(similarity_threshold=0.85)

    for problem in problems:
        chains = cache.get(problem.id)
        if chains is None:
            print(f"Skipping {problem.id} — not in cache")
            continue
        fragments = extractor.extract_fragments(chains)
        cluster_data = precompute_cluster_data(
            fragments, extractor.fragment_sources, chains, problem.dataset,
        )

        problem_data.append({
            "problem": problem,
            "chains": chains,
            "fragments": fragments,
            "sources": extractor.fragment_sources,
            "embeddings": extractor.fragment_embeddings.copy(),
            "num_completions": extractor.num_completions,
            "cluster_data": cluster_data,
        })

    if not problem_data:
        print("No cached problems found. Run the benchmark first to populate the cache.")
        return

    # Print cluster info
    for pd in problem_data:
        pid = pd["problem"].id
        clusters = pd["cluster_data"]["answer_clusters"]
        anchors = int(pd["cluster_data"]["anchor_flags"].sum())
        print(f"  {pid}: clusters={dict((k, len(v)) for k, v in clusters.items())}, "
              f"anchors={anchors}, fragments={len(pd['fragments'])}")

    # Identify disagreement problems
    disagree_ids = set()
    for pd in problem_data:
        if len(pd["cluster_data"]["answer_clusters"]) > 1:
            disagree_ids.add(pd["problem"].id)

    # Print numerical consistency stats
    for pd in problem_data:
        pid = pd["problem"].id
        nc = pd["cluster_data"]["numerical_consistency"]
        nonzero = np.count_nonzero(nc[np.triu_indices(len(pd["fragments"]), k=1)])
        total_pairs = len(pd["fragments"]) * (len(pd["fragments"]) - 1) // 2
        nums_per_frag = [len(n) for n in pd["cluster_data"]["fragment_numbers"]]
        print(f"  {pid}: numeric pairs={nonzero}/{total_pairs}, "
              f"numbers/fragment={nums_per_frag}")

    # Generate config grid
    configs = list(itertools.product(
        MU_VALUES, DELTA_VALUES, EPSILON_VALUES, KAPPA_VALUES, ETA_VALUES,
    ))
    total_solves = len(configs) * len(problem_data)

    print(f"\nCLUSTER + CONSISTENCY SWEEP")
    print(f"{'=' * 80}")
    print(f"Problems: {len(problem_data)}, Configs: {len(configs)}, Total solves: {total_solves}")
    print(f"Disagreement problems: {disagree_ids}")
    print(f"Fixed: alpha={FIXED_ALPHA}, beta={FIXED_BETA}, lambda_sim={FIXED_LAMBDA_SIM}, gamma={FIXED_GAMMA}")
    print(f"Sweep: mu={MU_VALUES}, delta={DELTA_VALUES}, epsilon={EPSILON_VALUES}, "
          f"kappa={KAPPA_VALUES}, eta={ETA_VALUES}")
    print()

    # Run sweep
    results = {}
    for i, (mu, delta, epsilon, kappa, eta) in enumerate(configs):
        config_key = (mu, delta, epsilon, kappa, eta)
        results[config_key] = {}
        for pd in problem_data:
            r = run_config(
                pd["fragments"], pd["sources"], pd["embeddings"], pd["num_completions"],
                pd["cluster_data"],
                mu, delta, epsilon, kappa, eta,
                pd["problem"].ground_truth, pd["problem"].dataset,
            )
            results[config_key][pd["problem"].id] = r
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(configs)} configs done...")

    print(f"  {len(configs)}/{len(configs)} configs done.\n")

    # Report: baseline (no cluster or consistency terms)
    baseline_key = (1.0, 0.0, 0.0, 0.0, 0.0)
    print(f"BASELINE (mu=1.0, no cluster/consistency terms)")
    print("-" * 80)
    for pd in problem_data:
        pid = pd["problem"].id
        r = results[baseline_key][pid]
        gt = pd["problem"].ground_truth
        mark = "correct" if r["correct"] else "WRONG"
        disagree_mark = " [disagree]" if pid in disagree_ids else ""
        print(f"  {pid}: predicted={r['predicted']}, truth={gt}, {mark}{disagree_mark}")
    print()

    # Find majority-vote failures
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
            # Group by whether they break other problems
            clean_winners = [(k, b) for k, b in winners if not b]
            dirty_winners = [(k, b) for k, b in winners if b]

            if clean_winners:
                print(f"\n  Clean wins (fix failure, break nothing): {len(clean_winners)}")
                for config_key, _ in clean_winners:
                    mu, delta, epsilon, kappa, eta = config_key
                    print(f"    mu={mu}  delta={delta}  epsilon={epsilon}  "
                          f"kappa={kappa}  eta={eta}")

            if dirty_winners:
                print(f"\n  Dirty wins (fix failure, break something else): {len(dirty_winners)}")
                for config_key, breaks in dirty_winners[:5]:
                    mu, delta, epsilon, kappa, eta = config_key
                    print(f"    mu={mu}  delta={delta}  epsilon={epsilon}  "
                          f"kappa={kappa}  eta={eta}  (breaks: {breaks})")

            print(f"\n  Total: {len(winners)} / {len(configs)} configs fix the failure")
        else:
            print("  NO CONFIG FIXES THE FAILURE.")
    else:
        print("Baseline already gets all disagreement cases correct.")

    print()

    # Top 10 by disagreement accuracy
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
        mu, delta, epsilon, kappa, eta = config_key
        tag = " ← baseline" if config_key == baseline_key else ""
        print(f"  mu={mu:<4}  delta={delta:<4}  epsilon={epsilon:<4}  kappa={kappa:<4}  "
              f"eta={eta:<4}  disagree={dc}/{len(disagree_ids)}  "
              f"overall={ac}/{len(problem_data)}{tag}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
