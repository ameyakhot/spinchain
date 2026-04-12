---
allowed-tools: Bash(uv run:*), Bash(cat:*), Bash(tail:*), Bash(wc:*), Bash(jq:*), Read
description: Inspect raw SpinChain trace records — view recent calls, filter by trace ID, or dump full trace log
---

## Context

- Trace file: `SPINCHAIN_TRACE_DIR/spinchain_traces.jsonl` (default `~/.spinchain/traces/`)
- Format: one JSON object per line (JSONL)

## Your task

Inspect SpinChain trace records based on what the user asks.

### Steps

1. Determine the trace file location:

```bash
TRACE_DIR="${SPINCHAIN_TRACE_DIR:-$HOME/.spinchain/traces}"
TRACE_FILE="$TRACE_DIR/spinchain_traces.jsonl"
```

2. Based on user intent:

   **View last N traces** (default: last 5):
   ```bash
   uv run --directory /Users/maverick/quantum/spinchain python -c "
   from spinchain.tracing import TraceLogger
   import json, os
   tl = TraceLogger(trace_dir=os.environ.get('SPINCHAIN_TRACE_DIR'))
   for t in tl.read_traces(last_n=${ARGUMENTS:-5}):
       print(json.dumps(t, indent=2))
       print('---')
   "
   ```

   **Find a specific trace by ID**:
   ```bash
   grep '$ARGUMENTS' "$TRACE_FILE" | python3 -m json.tool
   ```

   **Show trace count and date range**:
   ```bash
   uv run --directory /Users/maverick/quantum/spinchain python -m spinchain.analyze --json | python3 -c "
   import sys, json
   r = json.load(sys.stdin)
   u = r['usage']
   print(f'Traces: {u[\"total_calls\"]}')
   print(f'Range: {u[\"first_call\"]} -> {u[\"last_call\"]}')
   print(f'Success: {u[\"success\"]}, Fallback: {u[\"fallback\"]}, Errors: {u[\"errors\"]}')
   "
   ```

3. Present the trace data formatted and readable, highlighting:
   - Timestamp and trace ID
   - Input parameters (num_completions, solver config)
   - Per-stage durations
   - Output: fragments selected, energy, fallback status
   - Any errors
