"""CLI orchestrator for SpinChain benchmarks."""

from __future__ import annotations

import argparse
import os
import sys

from tqdm import tqdm

from benchmarks.cache import ChainCache
from benchmarks.chain_generator import ChainGenerator
from benchmarks.config import BenchmarkConfig
from benchmarks.datasets import get_loader
from benchmarks.extractors import extract_answer
from benchmarks.methods import get_methods
from benchmarks.results import ProblemRecord, ResultsAccumulator


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="SpinChain benchmark harness")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "arc", "strategyqa"])
    parser.add_argument("--chains", type=int, default=7, help="Number of reasoning chains per problem")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N problems")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--methods", nargs="+", default=["spinchain", "majority_vote", "random", "union"])
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    parser.add_argument("--no-generate", action="store_true", help="Use cached chains only")
    parser.add_argument("--diagnostics", action="store_true", help="Dump QUBO coefficient diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = BenchmarkConfig(
        model=args.model,
        temperature=args.temperature,
        chains=args.chains,
        limit=args.limit,
        seed=args.seed,
        methods=args.methods,
        dataset=args.dataset,
        output=args.output,
        no_generate=args.no_generate,
    )
    config._diagnostics = args.diagnostics
    return config


def _classify_agreement(chains: list[str], dataset: str) -> bool:
    """Return True if all chains produce the same extracted answer."""
    answers = set()
    for chain in chains:
        ans = extract_answer(chain, dataset)
        if ans is not None:
            answers.add(ans)
    return len(answers) <= 1


def main() -> None:
    config = parse_args()

    # Load dataset
    loader = get_loader(config.dataset)
    print(f"Loading {config.dataset} dataset...")
    problems = loader.load(limit=config.limit)
    print(f"Loaded {len(problems)} problems")

    # Set up cache
    cache = ChainCache(
        cache_dir=config.cache_dir,
        dataset=config.dataset,
        model=config.model,
        temperature=config.temperature,
        chains=config.chains,
    )
    print(f"Cache: {len(cache)} problems cached at {cache.path}")

    # Set up chain generator
    generator = None
    if not config.no_generate:
        generator = ChainGenerator(model=config.model, temperature=config.temperature)

    # Set up methods
    diagnostics = getattr(config, "_diagnostics", False)
    methods = get_methods(config.methods, config, diagnostics=diagnostics)
    print(f"Methods: {[m.name for m in methods]}")
    print()

    # Main loop
    accumulator = ResultsAccumulator()
    skipped = 0

    for problem in tqdm(problems, desc="Evaluating"):
        # Get or generate chains
        chains = cache.get(problem.id)
        if chains is None:
            if config.no_generate:
                skipped += 1
                continue
            chains = generator.generate(problem.question, config.chains, config.dataset)
            cache.put(problem.id, chains)

        # Classify agreement
        agreement = _classify_agreement(chains, config.dataset)

        # Run each method
        record = ProblemRecord(
            problem_id=problem.id,
            ground_truth=problem.ground_truth,
            agreement=agreement,
        )
        for method in methods:
            result = method.run(chains, problem)
            record.results[method.name] = result

        accumulator.add(record)

    # Report
    if skipped:
        print(f"\nSkipped {skipped} problems (not in cache, --no-generate set)")
    accumulator.print_summary()

    if diagnostics:
        accumulator.print_diagnostics()

    if config.output:
        os.makedirs(os.path.dirname(config.output) or ".", exist_ok=True)
        accumulator.save_json(config.output)
        print(f"Results saved to {config.output}")
