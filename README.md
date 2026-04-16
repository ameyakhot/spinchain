# SpinChain

A verification layer for LLM reasoning. Detects computational errors across multiple reasoning attempts and selects verified-correct fragments using QUBO optimization.

```bash
pip install spinchain
```

## The Problem

LLMs make computational errors — wrong arithmetic, bad percentages, incorrect unit conversions. The standard fix is majority vote: generate multiple attempts, pick the most common answer. This fails when the majority makes the same mistake.

We tested 936 hyperparameter configurations across 9 distinct signals — popularity, co-occurrence, semantic similarity, question-relevance, cluster coherence, and more. **No combination of embedding-space signals can distinguish correct from incorrect reasoning when all chains address the same problem.** Correctness is not a property of the embedding space. It lives in the computation itself.

## The Approach

SpinChain breaks multiple reasoning attempts into sentence-level fragments, verifies every computation it finds, and formulates fragment selection as a QUBO (Quadratic Unconstrained Binary Optimization) problem. Simulated annealing finds the optimal subset.

The key mechanism is **error propagation**: one detected arithmetic error in one fragment taints the entire answer cluster. The solver then selects from the clean cluster, even when it's outnumbered 2:1.

```
LLM generates K reasoning chains
        |
        v
Fragment Extraction — split, embed, deduplicate
        |
        v
Verification — parse and check every "A op B = C"
        |
        v
QUBO Formulation — encode quality + verification as energy
        |
        v
Simulated Annealing — find lowest-energy fragment set
        |
        v
Stability Ranking — keep fragments stable across solutions
```

## Results

**24 out of 36 verification configurations outperform majority vote. Zero regressions.**

| Scenario | Majority Vote | SpinChain |
|----------|--------------|-----------|
| All chains agree | Correct | Correct |
| Majority has arithmetic error | Follows majority (wrong) | Detects error, selects correct minority |
| Error is interpretive, not computational | Follows majority | Matches majority (cannot detect) |

The critical term is **cluster integrity (psi)**: every winning configuration has psi >= 1.0. One verified-wrong expression is sufficient to flip the energy landscape.

> [!NOTE]
> Arithmetic verification is the base study. The architecture supports any verification function — type checking, unit consistency, schema validation, reference lookup — as a drop-in. One function per error class, not one rule per error.

## What It Catches

SpinChain verifies every explicit computation in the reasoning, regardless of domain:

| Pattern | Example | Detection |
|---------|---------|-----------|
| Arithmetic | `80000 + 50000 = 120000` | eval: 80000+50000 != 120000 |
| Percentages | `25% of 80000 = 15000` | eval: 80000*0.25 != 15000 |
| Multiplication | `24 * 0.8 = 18.2` | eval: 24*0.8 != 18.2 |

Whether the LLM is computing drug dosages, financial projections, or engineering tolerances — if it writes explicit math and gets it wrong, SpinChain catches it.

## Why QUBO

Fragment selection has pairwise dependencies — fragments support, contradict, or duplicate each other. QUBO encodes these as energy terms and evaluates all combinations jointly. Greedy methods evaluate fragments independently and miss these interactions.

The formulation is hardware-portable. The same Hamiltonian runs on classical SA today and on D-Wave quantum annealers, optical Ising machines, or QAOA circuits without modification.

## Setup

### MCP Server (Claude Code)

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "spinchain": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/spinchain", "python", "-m", "spinchain"],
      "env": {
        "SPINCHAIN_TRACE_DIR": "/path/to/spinchain/traces"
      }
    }
  }
}
```

Restart Claude Code. The `optimize_reasoning` tool becomes available automatically.

### From Source

```bash
git clone https://github.com/ameyakhot/spinchain.git
cd spinchain
uv sync
```

### Skills (Claude Code Plugin)

```
/plugin marketplace add ameyakhot/spinchain
/plugin install spinchain-skills@spinchain
```

Adds 6 skills: `spinchain-optimize`, `analyze-spinchain`, `test-spinchain`, `spinchain-status`, `spinchain-bench`, `spinchain-trace`.

## Configuration

### optimize_reasoning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `completions` | required | List of reasoning chain strings (min 2) |
| `num_reads` | 100 | SA samples per solve |
| `num_sweeps` | 1000 | MC sweeps per sample |
| `similarity_threshold` | 0.85 | Cosine similarity for fragment dedup |
| `selection_threshold` | 0.25 | Fraction of low-energy solutions for ranking |
| `inclusion_threshold` | 0.50 | Min frequency to include a fragment |
| `cardinality_k` | None | Target number of fragments to select |

### QUBO Coefficients

| Parameter | Default | Role |
|-----------|---------|------|
| `mu` | 1.0 | Fragment popularity |
| `alpha` | 0.5 | Risk penalty |
| `beta` | 1.0 | Pairwise coherence |
| `lambda_sim` | 0.3 | Similarity penalty factor |
| `phi` | 1.0 | Per-fragment verification |
| `psi` | 1.0 | Cluster integrity |
| `omega` | 1.0 | Verification agreement |

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `SPINCHAIN_TRACE_DIR` | `~/.spinchain/traces/` | JSONL trace log directory |

## Benchmarking

```bash
# Install benchmark dependencies
uv sync --extra benchmark

# Run on GSM8K (requires ANTHROPIC_API_KEY)
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 100

# Rerun from cache (no API calls)
uv run python -m benchmarks --dataset gsm8k --chains 7 --limit 100 --no-generate
```

Methods compared: SpinChain (QUBO/SA), majority vote (self-consistency), random selection, union (dedup without optimization).

| Dataset | Problems | Status |
|---------|----------|--------|
| GSM8K | 1,319 math problems | Implemented |
| ARC Challenge | 1,172 science MC | Extractor ready |
| StrategyQA | 2,290 yes/no multi-hop | Extractor ready |

Full results and methodology: [docs/benchmark-harness.md](docs/benchmark-harness.md)

## Trace Analysis

Every call is logged to `SPINCHAIN_TRACE_DIR/spinchain_traces.jsonl`.

```bash
uv run python -m spinchain.analyze            # human-readable report
uv run python -m spinchain.analyze --json      # JSON output
uv run python -m spinchain.analyze --last 10   # last 10 calls
```

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -v    # 72 tests
```

## Project Structure

```
src/spinchain/
  server.py                  MCP server
  tracing.py                 JSONL trace logger
  analyze.py                 Trace analysis CLI
  formulation/
    fragment_extractor.py    Sentence splitting, embedding, dedup
    coefficient_builder.py   Linear + quadratic weights with verification
    qubo_builder.py          BQM construction + cardinality constraint
  solvers/
    base.py                  Abstract solver interface
    simulated_annealing.py   dwave-neal SA solver
benchmarks/                  Evaluation harness
tests/                       72 unit tests
```

## Related

**[Anneal](https://github.com/ameyakhot/anneal)** — Optimal context selection for AI coding assistants. Uses SpinChain's QUBO/SA engine to select which code an LLM should see for a given task.

SpinChain optimizes the **output** (reasoning quality). Anneal optimizes the **input** (code context). Same engine, different formulations.

## Documentation

- [Hamiltonian specification](docs/qubo-formulation.md) — Full 12-term formulation
- [Pipeline stages](docs/pipeline-stages.md) — Why each stage exists
- [Benchmark harness](docs/benchmark-harness.md) — 936-config sweep results
- [Signal inventory](docs/qubo-signal-inventory.md) — All available QUBO signals
- [Live demo](docs/live-demo.md) — End-to-end Claude Code example

## License

MIT
