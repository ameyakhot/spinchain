# SpinChain

Inference-time reasoning optimization for LLMs using QUBO/Ising formulations. Built as an MCP server for Claude Code.

SpinChain takes multiple diverse reasoning chains, extracts fragments, formulates their selection as a QUBO problem (following [QCR-LLM](https://arxiv.org/abs/2510.24509)), solves it with simulated annealing, and returns the most stable and coherent fragment subset.

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
| `test-spinchain` | Run the full test suite (71 tests) |
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
├── tests/                         # 71 tests
├── examples/
│   └── basic_usage.py             # Synthetic benchmark
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
