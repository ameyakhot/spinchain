# SpinChain MCP Server Refactor

## Context

SpinChain implements inference-time reasoning optimization based on the QCR-LLM paper (arXiv 2510.24509). It formulates LLM reasoning path selection as a QUBO problem solved by simulated annealing, with a hardware-agnostic solver interface for future Ising chip backends (COBI, Quanfluence, IBM Quantum).

The current implementation is a standalone pipeline that calls the Anthropic API directly to generate completions and final answers. This is architecturally wrong — SpinChain should be a tool that Claude calls during its own reasoning, not an external app that drives Claude. The refactor converts SpinChain from a standalone pipeline into an MCP server that Claude invokes mid-conversation.

## Design

### Architecture

```
User asks complex reasoning question
  ↓
Claude generates N diverse reasoning chains
  (via extended thinking or self-prompting)
  ↓
Claude calls MCP tool: optimize_reasoning(completions=[...])
  ↓
SpinChain MCP Server:
  FragmentExtractor → CoefficientBuilder → QUBOBuilder → SA Solver → Stability Ranking
  ↓
Returns: {selected_fragments, all_fragments, energies, solver, metadata}
  ↓
Claude synthesizes final answer using optimized fragments
```

### What Stays (unchanged)

These modules are the core math layer and remain as-is:

- `formulation/fragment_extractor.py` — sentence splitting, SentenceTransformer embeddings, cosine similarity deduplication
- `formulation/coefficient_builder.py` — linear weights (popularity + risk), quadratic weights (co-occurrence - similarity)
- `formulation/qubo_builder.py` — dimod BinaryQuadraticModel construction with optional cardinality constraint
- `solvers/base.py` — BaseSolver ABC
- `solvers/simulated_annealing.py` — dwave-neal SA solver

### What Gets Removed

- `pipeline/spinchain_pipeline.py` — the current end-to-end pipeline with Anthropic API calls. The orchestration logic (stability ranking, fragment assembly) moves into the MCP server's tool handler. The API-calling methods (`_generate_completions`, `_generate_final_answer`) are deleted entirely.

### What Gets Added

#### `server.py` — MCP Server

A single file implementing the MCP server using the `mcp` Python SDK with stdio transport.

**Tool exposed:** `optimize_reasoning`

**Input schema:**
```json
{
  "completions": ["chain 1 text", "chain 2 text", ...],
  "solver_config": {
    "num_reads": 100,
    "num_sweeps": 1000
  },
  "similarity_threshold": 0.85,
  "selection_threshold": 0.25,
  "inclusion_threshold": 0.50,
  "cardinality_k": null
}
```

Only `completions` is required. All other fields have sensible defaults.

**Output schema:**
```json
{
  "selected_fragments": ["fragment 1", "fragment 2", ...],
  "all_fragments": ["fragment 1", ..., "fragment N"],
  "selected_indices": [0, 3, 5],
  "num_completions": 5,
  "num_fragments": 12,
  "solver": "simulated_annealing",
  "min_energy": -4.23,
  "energies": [-4.23, -3.91, ...],
  "fallback": false
}
```

If fewer than 2 fragments are extracted, returns `fallback: true` with the first completion as the only selected fragment.

**Stability ranking logic** (moved from pipeline):
- Take bottom `selection_threshold` fraction of solutions by energy
- Compute inclusion frequency of each fragment across low-energy solutions
- Select fragments above `inclusion_threshold`, sorted by frequency descending

#### `__main__.py`

Entry point for `python -m spinchain.server` that starts the MCP server.

#### Updated `pyproject.toml`

- Add `mcp` SDK dependency
- Remove `anthropic` from required dependencies (move to optional if needed)
- Update entry point

### Installation

Add to Claude Code settings (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "spinchain": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/maverick/quantum/spinchain", "python", "-m", "spinchain.server"]
    }
  }
}
```

### File Changes Summary

| File | Action |
|------|--------|
| `src/spinchain/server.py` | **Create** — MCP server with optimize_reasoning tool |
| `src/spinchain/__main__.py` | **Create** — `python -m spinchain.server` entry point |
| `src/spinchain/pipeline/spinchain_pipeline.py` | **Delete** — replaced by server.py |
| `src/spinchain/pipeline/__init__.py` | **Update** — remove pipeline exports |
| `src/spinchain/__init__.py` | **Update** — export server, remove pipeline |
| `pyproject.toml` | **Update** — add mcp dep, remove anthropic from required |
| `examples/basic_usage.py` | **Rewrite** — show how to test the MCP tool directly |

### Testing

1. **Unit test (no API key needed):** Call optimize_reasoning logic directly with synthetic completions (e.g., 5 hand-written reasoning chains). Verify fragment extraction, QUBO formulation, SA solving, and stability ranking produce valid output.

2. **MCP integration test:** Start the server via stdio, send a JSON-RPC `tools/call` request, verify response schema.

3. **End-to-end in Claude Code:** Install the MCP server, ask Claude a reasoning question, observe whether Claude calls the tool and uses the optimized fragments.

### Dependencies

**Add:**
- `mcp >= 1.0.0` (MCP Python SDK)

**Remove from required (keep as optional):**
- `anthropic` (no longer called by SpinChain itself)

**Keep:**
- `dwave-neal`, `dimod`, `pyqubo` (solver layer)
- `sentence-transformers` (fragment extraction)
- `numpy`, `scipy` (numerical computation)
