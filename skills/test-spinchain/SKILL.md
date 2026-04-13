---
name: test-spinchain
description: Run the SpinChain test suite. 71 tests covering QUBO coefficient math, BQM construction, cardinality constraints, stability ranking, trace logging, and analysis pipeline. Use to validate the system after changes.
---

# Run SpinChain Tests

Execute the full test suite and report results.

## Steps

1. Run all tests with verbose output:

```bash
uv run --directory <spinchain-project-dir> --extra dev python -m pytest tests/ -v
```

2. Summarize results by module:
   - `test_coefficient_builder` — QCR-LLM linear and quadratic weight math
   - `test_qubo_builder` — BQM construction, cardinality constraints, brute-force ground state
   - `test_stability_ranking` — fragment selection from SA solutions
   - `test_tracing` — JSONL trace lifecycle, stages, multi-trace accumulation
   - `test_analyze` — usage stats, latency breakdown, energy stats, anomaly detection

3. If any tests fail:
   - Read the failing test and the source module it covers
   - Diagnose whether the issue is in the test or the implementation
   - Report the root cause

4. To run a subset, pass pytest arguments:

```bash
# Run only coefficient tests
uv run --directory <spinchain-project-dir> --extra dev python -m pytest tests/test_coefficient_builder.py -v

# Run tests matching a keyword
uv run --directory <spinchain-project-dir> --extra dev python -m pytest tests/ -k "cardinality" -v
```
