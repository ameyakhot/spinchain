---
name: spinchain-optimize
description: Use SpinChain to optimize complex reasoning. When facing a hard problem, generate multiple diverse reasoning chains and pass them to optimize_reasoning to extract the best fragments via QUBO optimization and simulated annealing.
---

# SpinChain Reasoning Optimizer

Use the SpinChain MCP server to improve reasoning quality on complex problems. SpinChain formulates reasoning path selection as a QUBO problem, solves it with simulated annealing, and returns the most stable and coherent fragments across multiple reasoning attempts.

## When to use

- Multi-step reasoning problems where different approaches may yield different insights
- Math, logic, or analysis tasks where correctness matters
- Any question where you want to reduce hallucination risk by cross-validating reasoning paths

## How to use

### Step 1: Generate diverse reasoning chains

Think through the problem 3-5 times using different approaches, angles, or starting points. Each chain should be a complete reasoning attempt. Store them as separate strings.

### Step 2: Call optimize_reasoning

```
mcp__spinchain__optimize_reasoning(
    completions=[chain1, chain2, chain3, ...],
    num_reads=100,
    num_sweeps=1000
)
```

### Step 3: Synthesize from optimized fragments

Use the `selected_fragments` from the response to construct your final answer. These fragments were selected for:
- **Popularity**: appearing across multiple chains (consensus)
- **Stability**: consistently selected across low-energy solutions
- **Coherence**: co-occurring with other selected fragments
- **Diversity**: not redundant with each other

## Parameters

| Parameter | Default | When to change |
|-----------|---------|---------------|
| `completions` | required | Always provide 2+ chains. 5-10 is ideal. |
| `num_reads` | 100 | Increase to 200-500 for critical reasoning tasks |
| `num_sweeps` | 1000 | Increase to 5000 for very large fragment pools |
| `similarity_threshold` | 0.85 | Lower to 0.7 if chains are very diverse |
| `cardinality_k` | None | Set to constrain output to exactly K fragments |

## Interpreting results

- `fallback: true` means SpinChain couldn't optimize (too few completions or fragments)
- `min_energy` closer to 0 means weak signal; very negative means strong consensus
- `selected_fragments` are ordered by stability (most stable first)
- Check `num_fragments` vs `num_selected` — a low ratio means high selectivity
