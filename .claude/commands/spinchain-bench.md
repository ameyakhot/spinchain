---
allowed-tools: Bash(uv run:*), Read
description: Run the SpinChain example benchmark with synthetic reasoning chains to validate the full pipeline
---

## Context

- Example script: `/Users/maverick/quantum/spinchain/examples/basic_usage.py`
- Tests the full pipeline: fragment extraction → QUBO → SA → stability ranking
- Uses hardcoded synthetic completions (no API key needed)

## Your task

Run the SpinChain benchmark example and analyze the results.

### Steps

1. Run the basic usage example:

```bash
uv run --directory /Users/maverick/quantum/spinchain python examples/basic_usage.py
```

2. Parse the JSON output and present:
   - How many completions were provided?
   - How many fragments were extracted vs. selected?
   - What was the minimum energy found?
   - Which fragments were selected and why they make sense
   - Was there a fallback?

3. If the trace logger is configured, check the trace that was just written:

```bash
uv run --directory /Users/maverick/quantum/spinchain python -m spinchain.analyze --last 1
```

4. Report the per-stage latency breakdown from this run:
   - Fragment extraction time (model loading + embedding + dedup)
   - QUBO formulation time
   - SA solve time
   - Stability ranking time
   - Total wall clock
