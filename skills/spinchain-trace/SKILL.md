---
name: spinchain-trace
description: Inspect raw SpinChain trace records. View recent MCP calls, look up specific trace IDs, or dump the full trace log. Each trace contains timestamps, input parameters, per-stage timing, and solver output.
---

# Inspect SpinChain Traces

View raw JSONL trace records from SpinChain MCP calls.

## Steps

1. Determine the trace file location:

```bash
TRACE_DIR="${SPINCHAIN_TRACE_DIR:-$HOME/.spinchain/traces}"
echo "Trace file: $TRACE_DIR/spinchain_traces.jsonl"
```

2. Based on what the user asks:

### View last N traces (default 5)

```bash
uv run --directory <spinchain-project-dir> python -c "
from spinchain.tracing import TraceLogger
import json, os
tl = TraceLogger(trace_dir=os.environ.get('SPINCHAIN_TRACE_DIR'))
for t in tl.read_traces(last_n=5):
    print(json.dumps(t, indent=2))
    print('---')
"
```

### Show summary counts

```bash
uv run --directory <spinchain-project-dir> python -m spinchain.analyze --json | python3 -c "
import sys, json
r = json.load(sys.stdin)
u = r['usage']
print(f'Traces: {u[\"total_calls\"]}')
print(f'Range: {u[\"first_call\"]} -> {u[\"last_call\"]}')
print(f'Success: {u[\"success\"]}, Fallback: {u[\"fallback\"]}, Errors: {u[\"errors\"]}')
"
```

3. For each trace, highlight:
   - **trace_id** and **timestamp** — when was this call made
   - **input_params** — how many completions, solver config
   - **stages** — duration of each pipeline step
   - **output_summary** — fragments selected, energy, fallback status
   - **error** — any exceptions

## Trace schema

```json
{
  "trace_id": "a1b2c3d4e5f6",
  "timestamp": "2026-04-12T10:30:00+0000",
  "input_params": {
    "num_completions": 5,
    "num_reads": 100,
    "num_sweeps": 1000,
    "similarity_threshold": 0.85,
    "selection_threshold": 0.25,
    "inclusion_threshold": 0.50,
    "cardinality_k": null
  },
  "stages": [
    {"name": "fragment_extraction", "duration_ms": 450.2, "num_merged_fragments": 12},
    {"name": "qubo_formulation", "duration_ms": 35.1, "num_linear_terms": 12, "num_quadratic_terms": 66},
    {"name": "simulated_annealing", "duration_ms": 1200.5, "num_samples": 100, "min_energy": -4.23},
    {"name": "stability_ranking", "duration_ms": 2.3, "num_selected": 5}
  ],
  "output_summary": {
    "fallback": false,
    "num_fragments": 12,
    "num_selected": 5,
    "min_energy": -4.23
  },
  "total_duration_ms": 1692.4,
  "error": null
}
```
