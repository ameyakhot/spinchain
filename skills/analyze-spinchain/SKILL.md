---
name: analyze-spinchain
description: Analyze SpinChain MCP trace logs. Shows usage stats, per-stage latency breakdown, energy distribution, and anomaly detection. Use when you want to understand how SpinChain is being used or diagnose performance issues.
---

# Analyze SpinChain Traces

Run the SpinChain trace analyzer to inspect usage patterns, performance, and anomalies.

## Steps

1. Run the trace analyzer:

```bash
uv run --directory <spinchain-project-dir> python -m spinchain.analyze
```

2. If traces exist, also get the JSON output for detailed data:

```bash
uv run --directory <spinchain-project-dir> python -m spinchain.analyze --json
```

3. Present findings clearly:
   - **Usage**: total calls, success vs fallback vs error rates
   - **Latency**: which pipeline stage is the bottleneck (extraction, formulation, SA, ranking)
   - **Energy**: min energy trend across calls (lower = stronger consensus)
   - **Anomalies**: errors, slow calls (>10s), slow stages (>5s), empty selections

4. To analyze only recent calls, use `--last N`:

```bash
uv run --directory <spinchain-project-dir> python -m spinchain.analyze --last 10
```

5. If no traces exist yet, inform the user they need to use the `optimize_reasoning` MCP tool first, or run the example:

```bash
uv run --directory <spinchain-project-dir> python examples/basic_usage.py
```

## What the report tells you

| Section | Insight |
|---------|---------|
| **Calls** | How often SpinChain is being invoked |
| **Latency (fragment_extraction)** | Model loading + embedding + dedup time |
| **Latency (qubo_formulation)** | Coefficient computation — should be fast (<100ms) |
| **Latency (simulated_annealing)** | SA solve time — scales with num_reads x num_sweeps |
| **Latency (stability_ranking)** | Fragment selection — should be negligible |
| **Anomalies** | Errors, timeouts, or empty results to investigate |
