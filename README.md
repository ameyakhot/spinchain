# SpinChain

Automatic error detection and correction for LLM reasoning, using quantum-inspired optimization to select verified-correct fragments from multiple reasoning attempts.

## What SpinChain Does

When an LLM reasons through a problem, it sometimes makes computational errors — wrong arithmetic, incorrect percentages, bad unit conversions. SpinChain catches these errors automatically and selects the correct reasoning.

```
Without SpinChain:

  User asks question → LLM thinks once → answer (might have errors)

With SpinChain:

  User asks question → LLM thinks K times → SpinChain verifies
                                                    │
                                              checks every computation
                                              detects errors automatically
                                              selects verified-correct fragments
                                              logs errors for self-improvement
                                                    │
                                                    ▼
                                              verified answer
```

SpinChain is the **quality assurance layer** for LLM reasoning. It doesn't fix errors or rewrite reasoning — it **selects around errors** by finding the correct reasoning that already exists in one of the K attempts.

### How it works

1. The LLM generates K diverse reasoning chains for the same question
2. SpinChain breaks them into sentence-level fragments
3. Every computation in every fragment is verified (`A + B = C` → is that actually true?)
4. Fragment selection is formulated as a QUBO (Quadratic Unconstrained Binary Optimization) problem
5. Simulated annealing finds the optimal subset — verified-correct, coherent, non-redundant
6. A single detected error taints the entire answer cluster, flipping selection to the correct minority

### What it catches (domain-agnostic)

SpinChain doesn't need to know the domain. It checks every computation the LLM writes in its reasoning:

| Pattern | Example | Detection |
|---------|---------|-----------|
| Arithmetic | "80000 + 50000 = 120000" | eval: 80000+50000 ≠ 120000 |
| Percentages | "25% of 80000 = 15000" | eval: 80000×0.25 ≠ 15000 |
| Multiplication | "24 * 0.8 = 18.2" | eval: 24×0.8 ≠ 18.2 |
| Any `A op B = C` | Works across finance, medicine, engineering, legal | Automatic |

Whether the LLM is calculating drug dosages, financial projections, engineering tolerances, or tax returns — if it writes explicit math and gets it wrong, SpinChain catches it.

### Self-improving

Every detected error is logged to `~/.spinchain/errors.jsonl` with structured details (expression, stated result, correct result). On subsequent calls, SpinChain reads the error history and boosts verification weights for frequently observed error types. No retraining — the Hamiltonian adapts its coefficients based on accumulated evidence.

### Why QUBO / simulated annealing

Fragment selection isn't independent — fragments depend on each other, contradict each other, or duplicate each other. The QUBO formulation encodes these interactions as pairwise energy terms and evaluates all 2^R combinations jointly. SA finds the lowest-energy (optimal) configuration. Greedy methods evaluate fragments one at a time and miss these interactions.

The QUBO formulation is hardware-portable: the same Hamiltonian runs on classical SA today and on D-Wave quantum annealers, optical Ising machines, or QAOA circuits tomorrow.

## Proven Results

**936 configurations tested across 9 signals** to understand what the QUBO can and cannot do:

- Embedding-space signals (popularity, co-occurrence, similarity, question-relevance) — cannot distinguish correct from incorrect reasoning when all chains are on-topic
- Structural signals (cluster coherence, cross-cluster agreement) — cannot overcome the numerical advantage of the wrong majority
- **Arithmetic verification — 24 out of 36 configs outperform majority vote with zero regressions**

The critical term is **cluster integrity (ψ)**: one detected arithmetic error taints the entire answer cluster, allowing the solver to select the correct minority even against a 2:1 popularity disadvantage.

[Live demo: Claude Code → SpinChain MCP → verified answer](docs/live-demo.md) | [Full Hamiltonian specification](docs/qubo-formulation.md) | [Benchmark results](docs/benchmark-harness.md)

## Where SpinChain Fits

| Layer | What it does | Product |
|-------|-------------|---------|
| LLM (Claude) | Generates reasoning | Claude Code |
| **Error detection + optimization** | **Catches errors, selects correct fragments** | **SpinChain** |
| Context selection | Picks which code to show the LLM | [Anneal](https://github.com/ameyakhot/anneal) |

SpinChain optimizes the **output** (what the user gets). Anneal optimizes the **input** (what the LLM sees). Both use the same QUBO/SA engine.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install skills (recommended)

Install SpinChain skills into Claude Code via the plugin marketplace:

```
/plugin marketplace add ameyakhot/spinchain
/plugin install spinchain-skills@spinchain
```

This gives you 6 skills: `spinchain-optimize`, `analyze-spinchain`, `test-spinchain`, `spinchain-status`, `spinchain-bench`, `spinchain-trace`.

### Clone and install the server

```bash
git clone https://github.com/ameyakhot/spinchain.git
cd spinchain
uv sync
```

To install with dev dependencies (pytest, ruff):

```bash
uv sync --extra dev
```

### Register as an MCP server in Claude Code

Add SpinChain to your Claude Code config (`~/.claude.json`):

```json
{
  "mcpServers": {
    "spinchain": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/spinchain",
        "python",
        "-m",
        "spinchain"
      ],
      "env": {
        "SPINCHAIN_TRACE_DIR": "/path/to/spinchain/traces"
      }
    }
  }
}
```

Replace `/path/to/spinchain` with the actual path to your cloned repo.

Restart Claude Code. The `optimize_reasoning` tool will be available as `mcp__spinchain__optimize_reasoning`.

## How it works

```
User asks a complex reasoning question
        │
        ▼
Claude generates N diverse reasoning chains
        │
        ▼
Claude calls optimize_reasoning(completions=[...])
        │
        ▼
┌─────────────────────────────────────────────┐
│  SpinChain Pipeline                         │
│                                             │
│  1. Fragment Extraction                     │
│     Split into sentences, embed with        │
│     sentence-transformers, deduplicate      │
│     by cosine similarity                    │
│                                             │
│  2. QUBO Formulation (QCR-LLM)             │
│     Linear: -μ·popularity + α·risk          │
│     Quadratic: -β·(correlation - λ²·sim)    │
│                                             │
│  3. Simulated Annealing (dwave-neal)        │
│     100 reads × 1000 sweeps                 │
│                                             │
│  4. Stability Ranking                       │
│     Select fragments appearing in ≥50%      │
│     of low-energy solutions                 │
└─────────────────────────────────────────────┘
        │
        ▼
Claude receives optimized fragments
and synthesizes final answer
```

## Benchmarking

SpinChain includes a benchmark harness that evaluates fragment-level QUBO optimization against standard baselines on reasoning datasets with known ground truth.

### Methods compared

| Method | Approach |
|--------|----------|
| **SpinChain** | QUBO/SA fragment selection — popularity, co-occurrence, and similarity encoded as an Ising model |
| **Majority vote** | Self-consistency (Wang et al. 2022) — pick the most common final answer across K chains |
| **Random** | Pick one chain at random — floor baseline |
| **Union** | All deduplicated fragments, no optimization — isolates whether the QUBO step adds value |

### Running benchmarks

```bash
# Install benchmark dependencies
uv sync --extra benchmark

# Run on GSM8K (requires ANTHROPIC_API_KEY for chain generation)
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 100

# Rerun with cached chains (no API calls, free)
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 100 --no-generate
```

Chains are cached to `benchmarks/.cache/` so generation cost is paid once. Different model/temperature/K configurations get separate cache files.

### Validation results

Validated on 3 GSM8K problems (K=3 chains each) with synthetic chains containing both agreement and disagreement cases:

```
BENCHMARK RESULTS (3 problems)
Agreement:       1 / 3 (33.3%)
Disagreement:    2 / 3

Method                  Overall   Disagree  No Answer
-----------------------------------------------------
majority_vote             66.7%      50.0%          0
random                    66.7%      50.0%          0
spinchain                 66.7%      50.0%          0
union                     66.7%      50.0%          0
```

### What the results show

SpinChain only adds value when reasoning chains **disagree**. On agreement cases (all chains produce the same answer), every method ties — there is nothing to optimize.

On disagreement cases, SpinChain currently tracks majority vote behavior. Coefficient diagnostics (`--diagnostics` flag) revealed why: **similarity repulsion dominates the QUBO energy landscape**, not the popularity term as initially hypothesized. The quadratic weights are ~2x larger than linear weights (ratio 0.4-0.6x). The similarity penalty (`lambda^2 * sim_ij`) generates large repulsive terms between semantically related fragments, and the solver spends most of its energy budget pushing apart similar fragments rather than pulling together co-occurring ones. This causes SpinChain to select diverse fragments that happen to come from popular chains.

A systematic sweep across **936 configurations and 9 distinct signals** — embedding-space (popularity, risk, co-occurrence, similarity, question-relevance), structural (cluster coherence, cross-cluster agreement, answer anchoring), and arithmetic (numerical consistency) — confirmed that **no combination of self-contained signals fixes the failure mode**. The error in the adversarial case is not in computation but in natural language interpretation ("150% of cost" vs. "150% increase in value"), which no chain-derived signal can detect without understanding the problem statement's semantics. See [docs/qubo-formulation.md](docs/qubo-formulation.md) for the full Hamiltonian and [docs/benchmark-harness.md](docs/benchmark-harness.md) for sweep results.

### Datasets supported

| Dataset | Format | Status |
|---------|--------|--------|
| **GSM8K** | 1,319 math problems, integer answers | Loader implemented |
| **ARC Challenge** | 1,172 science MC questions | Extractor ready, loader planned |
| **StrategyQA** | 2,290 yes/no multi-hop questions | Extractor ready, loader planned |

See [docs/benchmark-harness.md](docs/benchmark-harness.md) for architecture details, per-problem analysis, and CLI reference.

## Usage

### As an MCP server (primary use)

Once registered, Claude calls SpinChain automatically when using the `optimize_reasoning` tool. No manual invocation needed.

### Standalone example

Run the bundled example with synthetic reasoning chains (no API key needed):

```bash
uv run python examples/basic_usage.py
```

### Trace analysis

Every MCP call is logged to `SPINCHAIN_TRACE_DIR/spinchain_traces.jsonl`. Analyze with:

```bash
# Human-readable report
uv run python -m spinchain.analyze

# JSON output
uv run python -m spinchain.analyze --json

# Last 10 calls only
uv run python -m spinchain.analyze --last 10
```

### Run tests

```bash
uv run --extra dev python -m pytest tests/ -v
```

## Skills

Install via `/plugin marketplace add ameyakhot/spinchain` in Claude Code, or use as slash commands when working inside the project directory.

| Skill | Description |
|-------|-------------|
| `spinchain-optimize` | Core skill — teaches Claude how to generate diverse chains and call `optimize_reasoning` effectively |
| `analyze-spinchain` | Run trace analysis — usage stats, latency breakdown, anomalies |
| `test-spinchain` | Run the full test suite (72 tests) |
| `spinchain-status` | Health check — config, imports, trace summary |
| `spinchain-bench` | Run the example benchmark and inspect pipeline output |
| `spinchain-trace` | Inspect raw trace records — recent calls, specific trace IDs |

## Project structure

```
spinchain/
├── src/spinchain/
│   ├── server.py                  # MCP server — optimize_reasoning tool
│   ├── tracing.py                 # JSONL trace logger
│   ├── analyze.py                 # Trace analysis CLI
│   ├── formulation/
│   │   ├── fragment_extractor.py  # Sentence splitting, embedding, dedup
│   │   ├── coefficient_builder.py # QCR-LLM linear + quadratic weights
│   │   └── qubo_builder.py        # BQM construction + cardinality constraint
│   └── solvers/
│       ├── base.py                # Abstract solver interface
│       └── simulated_annealing.py # dwave-neal SA solver
├── tests/                         # 72 unit tests
├── benchmarks/                    # Benchmark harness (vs. majority vote, random, union)
│   ├── datasets/                  # Dataset loaders (GSM8K, ARC, StrategyQA)
│   ├── methods/                   # Baseline and SpinChain method implementations
│   └── extractors/                # Regex answer extraction per dataset type
├── examples/
│   └── basic_usage.py             # Synthetic demo (no API key needed)
├── .claude/commands/              # Claude Code slash commands
└── traces/                        # Trace logs (created at runtime)
```

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPINCHAIN_TRACE_DIR` | `~/.spinchain/traces/` | Directory for JSONL trace logs |

### optimize_reasoning parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `completions` | required | List of reasoning chain strings (min 2) |
| `num_reads` | 100 | SA samples per solve |
| `num_sweeps` | 1000 | MC sweeps per sample |
| `similarity_threshold` | 0.85 | Cosine similarity for fragment dedup |
| `selection_threshold` | 0.25 | Fraction of low-energy solutions for ranking |
| `inclusion_threshold` | 0.50 | Min frequency to include a fragment |
| `cardinality_k` | None | Target number of fragments to select |

### QUBO hyperparameters (in code)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu` | 1.0 | Fragment popularity weight |
| `alpha` | 0.5 | Risk/variance penalty |
| `beta` | 1.0 | Pairwise coherence weight |
| `lambda_sim` | 0.3 | Semantic similarity penalty factor |
| `penalty_strength` | 5.0 | Cardinality constraint weight |

## Solver roadmap

1. **Simulated Annealing** (current) — dwave-neal, pure Python
2. **COBI Ising chip** — millisecond solves, hardware accelerated
3. **Quanfluence CIM** — optical coherent Ising machine
4. **IBM Quantum QAOA** — gate-based quantum validation

All solvers share the same `BaseSolver` interface and QUBO formulation.

## License

MIT
