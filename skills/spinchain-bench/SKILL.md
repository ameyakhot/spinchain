---
name: spinchain-bench
description: Run the SpinChain benchmark with synthetic reasoning chains. Tests the full pipeline end-to-end (fragment extraction, QUBO formulation, simulated annealing, stability ranking) without needing an API key.
---

# SpinChain Benchmark

Run the bundled example to validate the full pipeline with synthetic reasoning chains.

## Steps

1. Run the basic usage example:

```bash
uv run --directory <spinchain-project-dir> python examples/basic_usage.py
```

2. Parse the JSON output and present:
   - How many completions were provided
   - How many fragments extracted vs. selected
   - Minimum energy found by SA
   - Which fragments were selected and why they make sense
   - Whether a fallback occurred

3. Check the trace that was just written:

```bash
uv run --directory <spinchain-project-dir> python -m spinchain.analyze --last 1
```

4. Report per-stage latency:

| Stage | What it measures |
|-------|-----------------|
| fragment_extraction | Model loading + sentence embedding + deduplication |
| qubo_formulation | Coefficient computation (linear + quadratic weights) |
| simulated_annealing | SA solve (num_reads x num_sweeps) |
| stability_ranking | Fragment selection from low-energy solutions |
| total | End-to-end wall clock |

## Notes

- First run will be slow (~30-60s) due to sentence-transformers model download
- Subsequent runs are faster (~5-15s) since the model is cached
- The example uses hardcoded completions about a classic river-crossing puzzle
