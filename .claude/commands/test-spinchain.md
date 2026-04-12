---
allowed-tools: Bash(uv run:*), Bash(cat:*)
description: Run the SpinChain test suite (71 tests covering QUBO math, stability ranking, tracing, and analysis)
---

## Context

- Test directory: `/Users/maverick/quantum/spinchain/tests/`
- Test framework: pytest (installed via `[dev]` extras)

## Your task

Run the full SpinChain test suite and report results.

### Steps

1. Run all tests with verbose output:

```bash
uv run --directory /Users/maverick/quantum/spinchain --extra dev python -m pytest tests/ -v
```

2. Summarize the results:
   - Total passed / failed / errors
   - If any tests failed, read the failing test file and the source it tests to diagnose the issue
   - Group results by module: coefficient_builder, qubo_builder, stability_ranking, tracing, analyze

3. If the user passed arguments (e.g., `-k test_cardinality`, `--tb=short`), forward them to pytest:

```bash
uv run --directory /Users/maverick/quantum/spinchain --extra dev python -m pytest tests/ $ARGUMENTS
```
