# Benchmark Harness

## Overview

The benchmark harness evaluates SpinChain's fragment-level QUBO optimization against standard baselines on reasoning datasets with known ground truth. It answers one question: does SpinChain select better reasoning fragments than simpler methods?

The harness lives in `benchmarks/` at the project root. It is evaluation code, not part of the library — it depends on the Anthropic SDK and HuggingFace `datasets`, which are optional dependencies.

## Quick Start

```bash
# Install benchmark dependencies
uv sync --extra benchmark

# Run on 20 GSM8K problems, 7 chains each (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="sk-..."
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 20

# Rerun with cached chains (no API calls)
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 20 --no-generate

# Save results to JSON
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 100 --output results/gsm8k_k7.json
```

## Architecture

```
benchmarks/
  __main__.py              Entry point (python -m benchmarks)
  run.py                   CLI orchestrator: load → generate → evaluate → report
  config.py                Defaults: model, temperature, solver params, system prompts
  chain_generator.py       Anthropic SDK wrapper with exponential backoff
  cache.py                 JSONL chain cache (one file per model/temp/k config)
  scoring.py               Exact-match scoring per dataset type
  extractors/
    answer_extractor.py    Regex-based answer extraction (GSM8K, ARC, StrategyQA)
  datasets/
    base.py                Problem dataclass and DatasetLoader protocol
    gsm8k.py               HuggingFace openai/gsm8k loader
  methods/
    base.py                Method protocol and MethodResult dataclass
    majority_vote.py       Self-consistency baseline (Wang et al. 2022)
    random_selection.py    Random chain selection (floor baseline)
    spinchain_method.py    QUBO/SA optimization via optimize_reasoning()
    union.py               All deduplicated fragments without optimization
```

## Methods

### SpinChain (the method under test)

Passes K reasoning chains to `optimize_reasoning()`. The pipeline extracts sentence-level fragments, builds a QUBO from popularity, co-occurrence, and semantic similarity, solves with simulated annealing (100 reads, 1000 sweeps), and returns stability-ranked fragments. The final answer is extracted from the joined selected fragments via regex.

If SpinChain's stability ranking drops the fragment containing the final answer, the extraction returns None and the problem is scored as incorrect. This is tracked as a diagnostic metric — it reveals whether the QUBO formulation undervalues answer-bearing fragments.

### Majority Vote (primary baseline)

Extracts the final answer from each of K chains independently, then picks the most common answer. This is the standard self-consistency method (Wang et al. 2022) and the baseline SpinChain must beat to justify its complexity.

### Random Selection (floor baseline)

Picks one chain at random (seeded by problem ID for reproducibility) and extracts its answer. Any method that cannot beat random selection is broken.

### Union (ablation baseline)

Runs SpinChain's `FragmentExtractor` to deduplicate fragments across chains, but skips the QUBO optimization step entirely. Extracts the answer from all deduplicated fragments joined together. This isolates whether the optimization adds value beyond deduplication.

## Answer Extraction

All methods use the same regex-based extractor (`extractors/answer_extractor.py`). For GSM8K, it tries three patterns in order:

1. `#### [number]` (GSM8K canonical format)
2. `the answer is [number]`
3. Last bare integer in the text

This ensures fair comparison — all methods are scored using identical extraction logic.

## Chain Generation and Caching

Generating K chains per problem is the dominant cost ($5-15 for full GSM8K at K=7 with Sonnet). The cache eliminates this cost on reruns:

- Cache location: `benchmarks/.cache/`
- Filename format: `{dataset}_{model}_t{temperature}_k{chains}.jsonl`
- Each line: `{"problem_id": "...", "chains": [...], "timestamp": "..."}`
- `--no-generate` flag: fail on cache miss instead of calling the API

Different model/temperature/K configurations get separate cache files, so experiments don't collide.

## Metrics

**Primary:** Accuracy on disagreement cases — problems where the K chains produce different final answers. This is where SpinChain's optimization can differentiate itself from baselines.

**Secondary:**
- Overall accuracy (agreement + disagreement combined)
- Agreement rate (fraction of problems where all chains agree)

**Diagnostic (in per-problem JSON):**
- SpinChain: `min_energy`, `num_fragments`, `num_selected`, `fallback`
- Majority vote: `vote_counts`, `extractable`
- Random: `selected_chain_index`

## Validation Run

The harness was validated with 3 synthetic GSM8K problems (K=3 chains each), deliberately constructed to include both agreement and disagreement cases.

### Test Problems

| Problem | Ground Truth | Chain Answers | Agreement |
|---------|-------------|---------------|-----------|
| gsm8k_0 (Janet's eggs) | 18 | 18, 18, 18 | Yes |
| gsm8k_1 (robe fiber) | 3 | 3, 3, 4 | No |
| gsm8k_2 (house flip) | 70000 | 65000, 65000, 70000 | No |

### Results

```
======================================================================
BENCHMARK RESULTS (3 problems)
======================================================================
Agreement:       1 / 3 (33.3%)
Disagreement:    2 / 3

Method                  Overall   Disagree  No Answer
-----------------------------------------------------
majority_vote             66.7%      50.0%          0
random                    66.7%      50.0%          0
spinchain                 66.7%      50.0%          0
union                     66.7%      50.0%          0
======================================================================
```

### Per-Problem Analysis

**gsm8k_0 (agreement):** All chains said 18. All methods correct. No differentiation possible — this is the expected null case.

**gsm8k_1 (mild disagreement):** Two chains said 3 (correct), one said 4. Majority vote correctly picked 3. SpinChain also picked 3 — the QUBO formulation favored the more popular and mutually coherent fragments. Random unluckily picked the wrong chain (answer: 4). SpinChain metadata: 9 fragments extracted, 6 selected, min energy -7.17.

**gsm8k_2 (adversarial disagreement):** Two chains said 65000 (wrong), one said 70000 (correct). Majority vote picked 65000 — the popular-but-wrong answer. SpinChain also picked 65000 — the QUBO's popularity term (`-mu * p_i`) dominated, selecting fragments from the majority even though they were incorrect. Random got lucky and picked 70000. This is the exact failure mode predicted in the benchmarking strategy: when the majority is wrong, popularity-based optimization inherits the same bias as majority vote.

### Key Takeaway

On this small synthetic validation:
- The harness runs correctly end-to-end (dataset loading, caching, all 4 methods, scoring, reporting)
- Agreement/disagreement classification works
- SpinChain's current QUBO formulation tracks majority vote behavior — it favors popular fragments, which means it inherits majority vote's failure mode on problems where the majority is wrong
- A larger-scale run on real LLM-generated chains is needed to determine if the co-occurrence and similarity terms provide enough signal to differentiate from majority vote

## CLI Reference

```
uv run python -m benchmarks [OPTIONS]

Options:
  --dataset       Dataset name: gsm8k (default), arc, strategyqa
  --chains        Number of reasoning chains per problem (default: 7)
  --limit         Limit to first N problems (default: all)
  --model         Anthropic model ID (default: claude-sonnet-4-20250514)
  --temperature   Sampling temperature (default: 0.7)
  --methods       Space-separated method names (default: spinchain majority_vote random union)
  --output        Path to save results JSON
  --no-generate   Use cached chains only; fail on cache miss
  --seed          Random seed for reproducibility (default: 42)
```

## Next Steps

1. **Run at scale:** 100+ GSM8K problems with real LLM-generated chains to get statistically meaningful results on disagreement cases
2. **Add ARC and StrategyQA loaders** (extractors and scoring already support them)
3. **Hyperparameter sweep:** Vary SpinChain's mu, alpha, beta, lambda to see if the QUBO formulation can be tuned to outperform majority vote
4. **Answer fragment retention rate:** Track how often SpinChain's stability ranking drops the fragment containing the correct final answer
