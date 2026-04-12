---
allowed-tools: Bash(uv run:*), Bash(wc:*), Bash(cat:*), Read
description: Analyze SpinChain MCP trace logs — usage stats, latency breakdown, energy distribution, anomalies
---

## Context

- Trace directory: `$SPINCHAIN_TRACE_DIR` or `~/.spinchain/traces/`
- Trace file: `spinchain_traces.jsonl`

## Your task

Run the SpinChain trace analyzer and present the results.

### Steps

1. First check if any trace data exists:

```bash
uv run --directory /Users/maverick/quantum/spinchain python -m spinchain.analyze
```

2. If traces exist, also run the JSON output for detailed inspection:

```bash
uv run --directory /Users/maverick/quantum/spinchain python -m spinchain.analyze --json
```

3. Present the findings to the user in a clear summary:
   - How many times has `optimize_reasoning` been called?
   - What's the typical latency? Which stage is the bottleneck?
   - Are there any anomalies (errors, slow calls, empty selections)?
   - What's the energy distribution trend?

4. If the user passed an argument like `--last 10`, forward it:

```bash
uv run --directory /Users/maverick/quantum/spinchain python -m spinchain.analyze --last $ARGUMENTS
```

5. If no traces exist yet, tell the user they need to use SpinChain in a conversation first, or run the example to seed trace data:

```bash
uv run --directory /Users/maverick/quantum/spinchain python examples/basic_usage.py
```
