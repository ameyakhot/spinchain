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

**gsm8k_2 (adversarial disagreement):** Two chains said 65000 (wrong), one said 70000 (correct). Majority vote picked 65000 — the popular-but-wrong answer. SpinChain also picked 65000. Random got lucky and picked 70000.

## Coefficient Diagnostics

### The `--diagnostics` flag

Running with `--diagnostics` dumps the QUBO coefficient magnitudes for each problem, showing how linear (1-body) and quadratic (2-body) terms compare in the energy landscape:

```bash
uv run python -m benchmarks --dataset gsm8k --chains 3 --limit 3 --no-generate --diagnostics
```

### Results

```
COEFFICIENT DIAGNOSTICS
Problem           |linear| mean    |quad| mean    Ratio  Frags  Co-occur%   Sim mean
------------------------------------------------------------------------------------
gsm8k_0                0.367521       0.796589     0.5x     13     53.8%     0.4030
gsm8k_1                0.345679       0.811971     0.4x      9     47.2%     0.3958
gsm8k_2                0.477778       0.766255     0.6x     10     64.4%     0.4460
```

### What this tells us

The initial hypothesis was that the linear popularity term (`-mu * p_i`) dominates the QUBO, causing SpinChain to behave like majority vote. **The diagnostics disproved this.** The actual magnitudes show:

- **Linear terms are smaller than quadratic terms** — the ratio is 0.4-0.6x, meaning quadratic weights are roughly **2x larger** than linear weights
- **Co-occurrence density is high** (47-64%) — most fragment pairs appear together in at least one chain, because with only 3 chains there are limited source combinations
- **Cosine similarity is moderate** (~0.40 mean) — fragments from the same math problem are semantically related, producing non-trivial similarity penalties

The quadratic formula is `w_ij = -beta * (z_corr_ij - lambda^2 * sim_ij)`. The similarity term (`lambda^2 * sim`) acts as repulsion between similar fragments. With moderate similarity across most pairs, this repulsion generates large positive quadratic weights that dominate the energy landscape. The QUBO is spending most of its energy budget **pushing apart similar fragments** rather than pulling together co-occurring ones.

### Revised understanding

SpinChain tracks majority vote not because popularity dominates, but because **similarity repulsion dominates the quadratic terms**. The solver preferentially selects diverse, non-redundant fragments — which happen to come from the popular chains because they contribute more fragments to the pool. The co-occurrence attraction signal (`z_corr_ij`) is too weak relative to the similarity penalty to steer selection toward correctness.

This pointed to a specific question: can rebalancing the existing hyperparameters fix the failure mode, or does the formulation need a fundamentally new term?

## Hyperparameter Sweep

### Setup

A sweep across 120 hyperparameter configurations tested whether any rebalancing of the existing QUBO terms could make SpinChain outperform majority vote on gsm8k_2 (where 2/3 chains are wrong):

```bash
uv run python -c "from benchmarks.sweep import main; main()"
```

**Grid:** `mu` × [0.5, 1.0, 2.0, 4.0] · `alpha` × [0.0, 0.5] · `beta` × [0.5, 1.0, 2.0] · `lambda_sim` × [0.0, 0.1, 0.2, 0.3, 0.5] = 120 configs, 360 total SA solves across 3 problems.

### Results

```
DEFAULT CONFIG (mu=1.0, alpha=0.5, beta=1.0, lambda_sim=0.3)
  gsm8k_0: predicted=18, truth=18, correct
  gsm8k_1: predicted=3, truth=3, correct [disagree]
  gsm8k_2: predicted=65000, truth=70000, WRONG [disagree]

CONFIGS THAT FIX MAJORITY-VOTE FAILURES ({'gsm8k_2'})
  NO CONFIG FIXES THE FAILURE.
  The formulation needs a new term, not just rebalancing.
```

**Zero out of 120 configurations** can make SpinChain get gsm8k_2 correct. Every configuration achieves at most 1/2 disagreement accuracy — identical to majority vote. The top 10 configs are all ties.

### Interpretation

This is a definitive result. The three signals currently available to the QUBO formulation are:

1. **Popularity** (`p_i`) — how many chains contain a fragment
2. **Co-occurrence** (`corr_ij`) — whether two fragments appear together in the same chains
3. **Semantic similarity** (`sim_ij`) — cosine similarity of fragment embeddings

None of these correlate with correctness when the majority is wrong. Popularity explicitly favors the majority. Co-occurrence reflects chain membership, not reasoning quality. Semantic similarity measures redundancy, not accuracy. No linear combination of these three signals can distinguish "correct but unpopular" from "wrong but popular."

**The formulation needs a new term that carries a correctness signal independent of popularity.**

## Question-Relevance Experiment

### Hypothesis

Fragments semantically closer to the original question may be more likely to contain correct reasoning. A new linear term `w_i += -gamma * cosine_sim(question, fragment_i)` was added to the QUBO, where `gamma` controls the strength of question-relevance reward.

This was implemented in `CoefficientBuilder.compute_relevance_weights()` and integrated into both the MCP server (optional `question` parameter on `optimize_reasoning`) and the benchmark harness.

### Sweep

The hyperparameter grid was expanded to 600 configs by adding `gamma` ∈ [0.0, 0.5, 1.0, 2.0, 4.0]:

```bash
uv run python -c "from benchmarks.sweep import main; main()"
```

### Result

```
CONFIGS THAT FIX MAJORITY-VOTE FAILURES ({'gsm8k_2'})
  NO CONFIG FIXES THE FAILURE (including with question-relevance).
```

**Zero out of 600 configs fix gsm8k_2**, across all gamma values. Question-relevance has no effect.

### Why it fails

All fragments — from both correct and incorrect chains — discuss the same math problem (house flipping profit). They have roughly equal cosine similarity to the question. Question-relevance can distinguish "on-topic vs. off-topic" but **cannot distinguish "correct math vs. incorrect math about the same topic."**

### Implication

The limitation is now well-characterized: **no embedding-space signal** — popularity, co-occurrence, semantic similarity, or question-relevance — **can distinguish correct from incorrect reasoning when all chains address the same problem.** The correctness signal lives in the logical structure of the computation, not in the semantic content of the embeddings.

Possible directions that go beyond embedding similarity:
- **Numerical consistency checking** — verify that intermediate numbers in a fragment's reasoning are arithmetically consistent with each other
- **Verification via re-computation** — use an LLM or symbolic tool to check whether a chain's intermediate steps lead to its stated answer
- **Cross-chain logical entailment** — rather than cosine similarity, check whether fragments from different chains logically support or contradict each other

These require moving beyond the pure QUBO embedding formulation into hybrid approaches.

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
  --diagnostics   Dump QUBO coefficient magnitude analysis per problem
  --seed          Random seed for reproducibility (default: 42)
```

## Next Steps

1. **New QUBO term research:** Investigate signals that correlate with correctness independent of popularity — candidates include question-relevance (embedding distance to question), internal consistency (logical coherence between fragments), and chain confidence scores
2. **Run at scale:** 100+ GSM8K problems with real LLM-generated chains to confirm whether the sweep result holds beyond synthetic data
3. **Add ARC and StrategyQA loaders** (extractors and scoring already support them)
4. **Answer fragment retention rate:** Track how often stability ranking drops the fragment containing the correct final answer
