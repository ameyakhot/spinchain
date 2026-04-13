# Testing & Benchmarking

## Part 1: Micro-Testing

72 unit tests verify the correctness of each pipeline stage. This section describes what they prove and, critically, what they do not.

### Coefficient Builder (15 tests)

**File:** `tests/test_coefficient_builder.py`

Tests the QCR-LLM coefficient formulas that define the QUBO energy landscape.

**Proven:**

- Linear weight formula `w_i = -mu * p_i + alpha * p_i(1 - p_i)` produces correct values at boundary popularities (p=1.0, 0.5, 0.2) and across multi-fragment inputs
- `mu` scales the popularity term linearly; `alpha=0` eliminates the risk term
- More popular fragments receive more negative weights (lower energy = preferred by solver)
- Quadratic weight formula `w_ij = -beta * (z_corr_ij - lambda^2 * sim_ij)` produces symmetric matrices with zero diagonals
- Co-occurring fragment pairs attract (negative quadratic weight)
- Semantically similar fragments repel (positive quadratic weight via cosine similarity)
- `beta=0` kills all quadratic terms
- Z-score clipping: fragment pairs with non-positive raw correlation cannot receive positive z-scores, preventing the solver from rewarding diverse noise over genuine co-occurrence
- Single-pair edge case (2 fragments) skips z-score normalization gracefully

**Not proven:**

- Whether the default hyperparameters (mu=1.0, alpha=0.5, beta=1.0, lambda=0.3) are good choices for real LLM reasoning output. This requires benchmarking against ground truth.
- Whether z-score normalization is the right standardization strategy vs. alternatives (min-max scaling, raw values).

### QUBO Builder (11 tests)

**File:** `tests/test_qubo_builder.py`

Tests the construction of binary quadratic models (BQMs) from coefficient arrays.

**Proven:**

- BQM correctly assembled from linear and quadratic weight arrays (correct variable count, correct bias values, BINARY vartype)
- Small quadratic terms (`|w| < 1e-10`) are pruned to reduce solver noise
- Negative quadratic weights (attraction between fragments) are preserved
- Cardinality constraint `penalty * (sum(x_i) - K)^2` expands correctly: adds `penalty*(1-2K)` to linear terms and `2*penalty` to quadratic terms
- Cardinality penalty stacks additively with existing QUBO weights
- Brute-force verification: with uniform weights and K=2 cardinality on 4 variables, the ground state selects exactly 2 fragments
- BQM-to-QUBO dictionary conversion round-trips without loss

**Not proven:**

- Behavior at scale. Tests use up to 6 fragments; real problems produce 10-50 fragments with denser quadratic coupling.
- Whether `penalty_strength=5.0` (the default) is sufficient to enforce cardinality in practice. Only tested with `penalty=10` in the brute-force test.

### Stability Ranking (6 tests)

**File:** `tests/test_stability_ranking.py`

Tests the post-SA filter that selects fragments based on their frequency in low-energy solutions.

**Proven:**

- Fragments present in all low-energy solutions are selected
- Fragments absent from low-energy solutions are excluded
- Boundary: fragments at exactly the `inclusion_threshold` frequency are included
- Results sorted descending by inclusion frequency
- Empty list returned when no fragment meets the threshold
- Single-sample edge case works correctly

**Not proven:**

- Interaction with actual SA solver output. All tests use hand-crafted `dimod.SampleSet` objects, not real solver runs.
- Whether the default thresholds (`selection_threshold=0.25`, `inclusion_threshold=0.50`) produce good results on real problems.

### Tracing (11 tests)

**File:** `tests/test_tracing.py`

Tests the JSONL-based observability layer that logs every `optimize_reasoning()` call.

**Proven:**

- Full trace lifecycle: start -> add stage -> finish -> write to disk
- Timestamps and trace IDs are generated correctly
- Total duration tracked via `perf_counter` (millisecond precision)
- Errors captured in trace records; null on success
- Per-stage timing and custom metadata (e.g., `num_raw_fragments`) recorded
- Multiple traces append to the same file; `read_last_n` filters correctly
- Each line is independently parseable JSON
- Unknown trace ID lookup returns None

**Not proven:**

- Thread safety under concurrent MCP calls
- Disk-full or permission-error handling

### Analysis (25 tests)

**File:** `tests/test_analyze.py`

Tests the trace reporting module that computes aggregate statistics.

**Proven:**

- Percentile helper handles odd/even-length lists and empty input
- Usage summary correctly counts total, success, fallback, and error calls; tracks time range
- Per-stage latency breakdown: min, max, mean, p50, p95 computed correctly
- Energy statistics: min, max, mean across traces
- Anomaly detection flags: errors, slow total durations, slow individual stages, empty fragment selections
- Normal traces produce zero anomaly flags
- Full report is JSON-serializable
- `print_report` produces human-readable output without errors
- File loading from JSONL with `last_n` support

**Not proven:**

- Whether anomaly thresholds reflect meaningful operational boundaries (what constitutes "slow" for real workloads)

### Coverage Gaps

These pipeline components have **zero automated test coverage:**

| Component | What's untested | Risk |
|-----------|----------------|------|
| **FragmentExtractor** | Sentence splitting, SentenceTransformer embedding, cosine-similarity deduplication | High. This is the pipeline's entry point and most likely source of surprising behavior with real LLM output. |
| **End-to-end pipeline** | No automated test runs `optimize_reasoning()` with realistic chains | High. Stage interactions (extraction output feeding coefficients) are unverified. |
| **SA solver** | Thin `dwave-neal` wrapper; no test verifies it finds good solutions for known problems | Medium. We trust the upstream library but don't verify integration. |
| **MCP server transport** | No test for stdio message framing or error responses | Low. Covered by the MCP SDK. |

---

## Part 2: Benchmarking Strategy

### The Core Problem

SpinChain selects optimal fragments from multiple reasoning chains. When all chains agree on the answer, every selection method -- SpinChain, majority vote, random pick -- produces the same result.

The fencing problem demo illustrates this: 7 chains, all producing "640 meters," 26 fragments extracted, 11 selected, energy -43.67. The pipeline ran correctly. But any baseline would also return 640 meters. **SpinChain only adds value in the disagreement regime**, when chains reach different conclusions and the correct reasoning is scattered across multiple flawed responses.

A proper benchmark must specifically measure performance on disagreement cases.

### Baselines

| Method | How it works | Role |
|--------|-------------|------|
| **Random selection** | Pick one chain's answer uniformly at random | Floor. SpinChain must beat this trivially. |
| **Majority vote** (self-consistency) | Take the most common final answer across K chains | The standard baseline (Wang et al. 2022). This is the method to beat. |
| **Best-of-N** | Select the chain with the highest log-probability or confidence score | Tests whether fragment-level selection outperforms chain-level selection. Requires model logprobs. |
| **Union** | Concatenate all unique fragments after deduplication, no QUBO optimization | Tests whether the optimization step adds value beyond deduplication alone. |

### Datasets

| Dataset | Size | Format | Why |
|---------|------|--------|-----|
| **GSM8K** | 1,319 test problems | Numeric answer, exact match | Math reasoning with known ground truth. High disagreement rate at elevated temperatures. |
| **ARC Challenge** | 1,172 questions | Multiple choice (A-D) | Science reasoning requiring conceptual understanding, not just arithmetic. |
| **StrategyQA** | 2,290 questions | Yes/No | Multi-hop reasoning with high disagreement rate. Tests whether fragment-level optimization helps when reasoning paths diverge significantly. |

### Experiment Protocol

For each problem in the dataset:

1. **Generate** K reasoning chains (K = 7, 11, 15) using the target LLM at temperature > 0
2. **Classify** as *agreement* (all chains produce the same answer) or *disagreement* (chains differ)
3. **Run each method** on disagreement cases:
   - SpinChain: pass chains to `optimize_reasoning()`, synthesize answer from selected fragments
   - Majority vote: count final answers, pick the mode
   - Best-of-N: pick the chain with highest log-probability
   - Random: pick one chain uniformly at random
   - Union: deduplicate fragments, present all without optimization
4. **Score** each method's answer against ground truth

### Metrics

**Primary:** Accuracy on disagreement cases. This is where SpinChain must demonstrate value.

**Secondary:**
- Overall accuracy (agreement + disagreement combined)
- Agreement rate per dataset (what fraction of problems have full chain consensus)

**Diagnostic:**
- Energy gap: difference between minimum and mean SA energy (measures optimization signal strength)
- Fragment selection ratio: how many fragments survive vs. total extracted
- Per-method latency breakdown

### Success and Failure Criteria

**Success:** SpinChain accuracy on disagreement cases exceeds majority vote by a statistically significant margin (>2 percentage points on 200+ disagreement cases).

**Null result:** SpinChain matches majority vote accuracy. This would indicate that the QUBO formulation captures popularity (which majority vote already exploits) but not reasoning quality.

**Failure modes:**

- *Low disagreement rate* (>90% agreement): The task is too easy or the temperature is too low to produce meaningful diversity. Fix: increase temperature, use harder problem subsets.
- *Fragments don't compose*: SpinChain selects good individual fragments but they don't assemble into a coherent answer. This would indicate that sentence-level granularity is wrong, or that a recomposition step is needed.
- *Embedding quality*: `all-MiniLM-L6-v2` may not capture reasoning-relevant similarity. Domain-specific or larger embedding models may be needed.

### What Is Not Benchmarked

- **Latency**: SA adds 1-2 seconds per call. This is negligible compared to LLM chain generation time (which is identical across all methods). Not a differentiator.
- **Cost**: SpinChain runs locally with zero additional API cost. The dominant cost is generating K chains, which is constant across methods.
- **Hardware solvers**: D-Wave quantum annealers, optical Ising machines (COBI/Quanfluence), and QAOA on gate-based quantum processors are future solver backends. Benchmarking against SA is deferred until those integrations exist.
