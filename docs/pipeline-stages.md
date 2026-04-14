# Pipeline Stages: Why Each Stage Exists

This document explains each stage of the SpinChain pipeline — what it does, why it's necessary, and what research motivates its design.

## Stage 1: Fragment Extraction

### What it does

Takes K reasoning chains (complete LLM responses) and produces R deduplicated sentence-level fragments with embeddings and source tracking.

1. Split each chain into sentences using period/newline boundaries
2. Encode all sentences with SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
3. Compute pairwise cosine similarity across all sentences
4. Merge near-duplicates (similarity ≥ 0.85) into single fragments, tracking which chains contributed each fragment

### Why we do it

**The fundamental premise of the entire system.** Standard methods (majority vote, best-of-N) treat LLM responses as atomic units — you accept or reject an entire response. But real reasoning is compositional. A response that gets the final answer wrong may contain correct intermediate steps. A response that gets the right answer may contain flawed reasoning that happens to cancel out.

QCR-LLM (Flores-Garrigos et al., 2025, arXiv:2510.24509) formalized this insight: "each completion produces a structured reasoning trace composed of short, self-contained fragments that we denote as reasons. Using sentence embeddings, these fragments are extracted, cleaned, and semantically normalized to remove redundancies or stylistic noise" (Section II).

The deduplication step is critical. Without it, fragments repeated verbatim across chains would appear as separate variables in the QUBO, inflating the problem size without adding information. QCR-LLM: "we compute pairwise cosine similarities between all fragments and merge those whose semantic distance falls below a predefined threshold, resulting in a consolidated set of R distinct reasoning fragments" (Section II).

**Source tracking** (which chains contain each fragment) is what enables the QUBO formulation. A fragment's popularity (p_i = |sources_i| / K) and co-occurrence with other fragments (which pairs appear in the same chains) are the raw inputs to the Hamiltonian's coefficients.

### Research foundation

- **Chain-of-Thought prompting** (Wei et al., 2022) established that LLMs produce better answers when reasoning step-by-step. This means the intermediate steps contain signal, not just the final answer.
- **Self-Consistency** (Wang et al., 2023, arXiv:2203.11171) showed that sampling multiple CoT chains and aggregating improves accuracy. But it aggregates at the answer level (majority vote), discarding intermediate reasoning.
- **QCR-LLM** (Flores-Garrigos et al., 2025) made the key leap: aggregate at the fragment level, not the answer level. This is what enables combinatorial optimization over reasoning steps.

---

## Stage 2: Error Classification

### What it does

For each fragment, determine whether it contains a verifiable arithmetic expression, and if so, whether that expression is correct.

1. Parse fragment text for arithmetic patterns: "A + B = C", "A × B = C", "A - B = C", "A / B = C"
2. Evaluate the left-hand side programmatically
3. Compare to the stated right-hand side
4. Tag each fragment: VERIFIED_CORRECT (+1.0), VERIFIED_WRONG (-1.0), or UNVERIFIABLE (0.0)

### Why we do it

**This is the stage that the original QCR-LLM formulation is missing, and the stage that our benchmarking proved is necessary.**

QCR-LLM's Hamiltonian uses only descriptive signals: popularity (p_i), co-occurrence correlation (c_ij), and semantic similarity (sim(i,j)). Our sweep across 936 configurations proved that no combination of these signals can distinguish correct from incorrect fragments when the majority is wrong (see docs/qubo-formulation.md).

The reason: all descriptive signals measure properties of the fragment *relative to other fragments* (how popular, how similar, how often co-occurring). None measure properties of the fragment *relative to ground truth*. Verification is the first prescriptive signal — it evaluates a fragment's content against an external standard (arithmetic correctness) rather than against other fragments.

Without this stage, the Hamiltonian has one energy well centered on the popular consensus. With this stage, the Hamiltonian can have two competing wells — one around the popular cluster, one around the verified-correct cluster. When these differ (majority is wrong but its arithmetic is detectably wrong), the verification term can flip the ground state.

### Why arithmetic specifically

Of the error types LLMs produce, arithmetic errors are uniquely verifiable without external knowledge:

| Error type | External knowledge needed? | Verifiable locally? |
|-----------|---------------------------|-------------------|
| Arithmetic (80+50=120) | No — just math | Yes |
| Wrong number propagation | No — track within chain | Yes |
| Interpretation ("150% of what?") | Yes — problem semantics | No |
| Hallucinated fact | Yes — world knowledge | No |
| Logic non-sequitur | Partially — formal logic | Partially |

Arithmetic verification is the lowest-hanging fruit: high-value signal, zero external dependencies, computable by regex + eval.

### Research foundation

- **Tool-augmented reasoning** (Schick et al., 2023, "Toolformer") showed that LLMs benefit from external tool calls for arithmetic. We apply the same principle to the optimization layer: the Hamiltonian calls a symbolic evaluator to score fragments.
- **Process reward models** (Lightman et al., 2023, "Let's Verify Step by Step") showed that per-step verification outperforms outcome-only evaluation. Our φ term is a lightweight, symbolic version of a process reward model — it scores individual reasoning steps, not just final answers.
- **Our empirical finding** (docs/qubo-formulation.md): 936 configs across 9 descriptive signals all fail. The Hamiltonian needs a prescriptive signal. Arithmetic verification is the simplest prescriptive signal that doesn't require an oracle.

---

## Stage 3: Answer Cluster Integrity

### What it does

Groups chains by their final answer, then scores each cluster's integrity based on how many of its fragments pass arithmetic verification.

1. Extract final answer from each chain (regex: "the answer is [X]", "#### [X]")
2. Group chains by answer → answer clusters (e.g., {65000: [chain0, chain1], 70000: [chain2]})
3. For each cluster, count verified-correct and verified-wrong fragments
4. Compute integrity = (correct_count - wrong_count) / total_verifiable
5. Assign each fragment the integrity score of its cluster

### Why we do it

**Fragment-level verification (Stage 2) has a coverage problem.** Not every fragment contains a verifiable expression. Many fragments are pure text reasoning ("Let me think about this step by step"). These unverifiable fragments can't receive a φ score directly.

Cluster integrity solves this by **propagating verification from verifiable to unverifiable fragments within the same answer cluster.** If a cluster's verifiable fragments are mostly correct, its unverifiable fragments are more likely correct too (they come from the same chains). The ψ term gives unverifiable fragments a score inherited from their cluster.

This is analogous to **belief propagation** in graphical models: local evidence (arithmetic verification on a few fragments) propagates through the graph structure (cluster membership) to inform beliefs about unobserved nodes (unverifiable fragments).

### Why clusters specifically

QCR-LLM does not use answer clusters. Its formulation treats all fragments as coming from a single pool, weighted only by popularity and co-occurrence. This means fragments from chains that reach different conclusions are mixed together in the QUBO — the solver doesn't know that selecting a fragment "the answer is 65000" is incompatible with selecting a fragment "the answer is 70000."

Answer clusters make the structure explicit. The QUBO can now reason about *which conclusion to support* rather than just *which fragments are popular*. The cluster coherence term (ε) creates ferromagnetic coupling within clusters (same-answer fragments attract), while the integrity term (ψ) weights clusters by their verified correctness.

### Research foundation

- **Self-Consistency** (Wang et al., 2023) groups chains by final answer and picks the majority. We extend this: instead of just counting chains per answer, we evaluate each cluster's internal reasoning quality via verification scores.
- **Universal Self-Consistency** (Chen et al., 2023, arXiv:2311.17311) recognized that answer extraction and comparison are themselves reasoning tasks. Our cluster integrity goes further — it's not just comparing answers but evaluating the reasoning quality within each answer cluster.
- **Condorcet's Jury Theorem** (1785) proves majority vote is optimal when each voter has >50% individual accuracy. When this assumption fails, cluster integrity provides a correction mechanism: instead of "which answer has more votes," it asks "which answer has better-verified reasoning."

---

## Stage 4: Hamiltonian Construction

### What it does

Combines all signals — descriptive (from Stage 1) and prescriptive (from Stages 2-3) — into a single QUBO energy function.

```
H(x) = Σ_i h_i · x_i  +  Σ_{i<j} J_ij · x_i · x_j
```

Linear coefficients (h_i) encode per-fragment properties:
- Popularity (μ), risk (α): from QCR-LLM
- Verification (φ): from Stage 2
- Cluster integrity (ψ): from Stage 3
- Answer anchoring (κ): structural role detection

Quadratic coefficients (J_ij) encode pairwise relationships:
- Co-occurrence (β) and similarity (λ): from QCR-LLM
- Cluster coherence (ε): same-answer attraction
- Numerical consistency (η): arithmetic chaining
- Verification agreement (ω): correct/incorrect repulsion from Stage 2

### Why we do it

**The QUBO formulation is the mathematical bridge between reasoning evaluation and quantum optimization.**

Each binary variable x_i ∈ {0, 1} represents a fragment selection decision. The Hamiltonian assigns an energy to every possible combination of selected fragments (2^R configurations). The solver's job is to find the lowest-energy configuration — the optimal fragment subset.

QCR-LLM (Flores-Garrigos et al., 2025) established the general form: "each reasoning fragment r_i is assigned a binary decision variable x_i ∈ {0, 1} [...] The objective of our framework is to find the configuration x that minimizes an energy function that represents the global coherence, diversity and statistical relevance of the selected fragments" (Section II).

The linear terms (h_i) define the **external field** — each fragment's intrinsic preference for being selected or not, independent of other fragments. In Ising model physics, this is analogous to an applied magnetic field biasing individual spins.

The quadratic terms (J_ij) define the **spin-spin couplings** — how pairs of fragments interact. Negative J_ij (ferromagnetic) means the two fragments prefer to be selected together. Positive J_ij (antiferromagnetic) means they repel — the solver tends to select one or the other but not both. QCR-LLM: "pairwise relations between fragments are encoded through the quadratic (2-body) coefficients [...] w_ij = -β(c̃_ij - λ²_sim · sim(i,j))" (Section II, Eq. 4).

### Why QUBO specifically

The QUBO formulation has a unique property: it is the **native input format for quantum optimization hardware.** D-Wave quantum annealers, optical Ising machines (COBI), and QAOA circuits on gate-based quantum processors all accept QUBO or its equivalent Ising form. By formulating fragment selection as a QUBO, the same problem description runs on any of these solvers without reformulation.

QCR-LLM: "we tackle the resulting high-order optimization problem both classically, via simulated annealing, and quantumly through the bias-field digitized counterdiabatic quantum optimizer (BF-DCQO) executed on IBM's superconducting digital quantum processors" (Abstract).

The Hamiltonian is also naturally extensible. Adding a new signal (like verification) means adding new coefficients to h_i or J_ij. The solver doesn't change. The formulation is a modular interface between "what we know about fragment quality" and "how we search for the optimal subset."

### Research foundation

- **QCR-LLM** (Flores-Garrigos et al., 2025): Defined the original linear (Eq. 2) and quadratic (Eq. 4) coefficient design. Their formulation uses popularity, risk, co-occurrence, and similarity. SpinChain implements this exactly and extends it with verification terms.
- **QUBO/Ising universality**: Any combinatorial optimization problem can be mapped to QUBO form (Lucas, 2014, "Ising formulations of many NP problems"). This means fragment selection is one instance of a universal optimization framework.
- **Esencan et al., 2024** (arXiv:2307.00071, "Combinatorial reasoning: Selecting reasons in generative AI pipelines via combinatorial optimization"): Earlier work formulating reason selection as QUBO, establishing the fragment-as-binary-variable paradigm that QCR-LLM extended.

---

## Stage 5: Solve

### What it does

Takes the QUBO/BQM and finds low-energy configurations using a sampler.

1. Pass BQM to the solver (SimulatedAnnealingSolver wrapping dwave-neal)
2. Run num_reads=100 independent annealing runs, each with num_sweeps=1000 Monte Carlo sweeps
3. Return a SampleSet containing 100 binary solutions and their energies

### Why we do it

**This is the optimization core.** The Hamiltonian defines the energy landscape; the solver explores it. The solver doesn't know what fragments are or what reasoning means — it only sees binary variables and energy values. Its job is to find the lowest-energy configuration.

Simulated annealing works by starting at high "temperature" (accepting random moves) and gradually cooling (only accepting moves that lower energy). This mimics the physical process of annealing a metal — slow cooling produces a more ordered (lower-energy) crystal structure.

QCR-LLM uses two solvers: classical SA and the BF-DCQO quantum algorithm on IBM hardware. They report: "the BF-DCQO solver yields an interpretable energy landscape whose low-energy configurations define the stable reasoning subset" (Section III). The key finding: "the solver recovers the optimal high-level structure of the reasoning Hamiltonian" (Section III) — meaning the solver successfully finds fragment subsets that correspond to coherent reasoning.

### Why multiple reads

A single SA run might get stuck in a local minimum. 100 independent runs provide a distribution over the energy landscape. This distribution is the input to Stage 6 (stability ranking). Fragments that appear in many low-energy solutions are "stable" — they're selected regardless of the solver's random starting point.

QCR-LLM: "running multiple annealing trajectories, we obtain not only a candidate ground state but also a distribution of near-optimal solutions that collectively describe the low-energy manifold of the reasoning problem" (Section III).

### Why SA specifically (and why quantum later)

SA is the baseline classical solver. It works well for small problems (R < 50 fragments) where the energy landscape is relatively smooth. For larger problems with rugged landscapes (many local minima), quantum solvers offer potential advantages:

- **D-Wave quantum annealing**: Quantum tunneling can traverse energy barriers that SA must thermally climb over. QCR-LLM demonstrated this on IBM hardware using BF-DCQO.
- **Optical Ising machines**: Solve in nanoseconds by mapping the QUBO to optical pulses. Relevant when latency matters.
- **QAOA on gate-based quantum processors**: Variational approach that can handle higher-order (3-body+) interactions natively, without the quadratic reduction that QCR-LLM requires.

QCR-LLM: "for lower-order or sparse Hamiltonians (K ≤ 5), classical methods remain a practical option, while for denser and higher-order configurations, quantum solvers have the potential to provide computational advantage" (Section III).

The verification terms (φ, ψ, ω) proposed in our extension create a more rugged landscape (two competing wells instead of one) — precisely the regime where quantum advantage is expected.

### Research foundation

- **Simulated Annealing** (Kirkpatrick et al., 1983, Science 220, 671): The foundational algorithm for combinatorial optimization via thermal simulation. QCR-LLM cites this as their classical baseline.
- **BF-DCQO** (Cadavid et al., 2025, Phys. Rev. 7, L022010): The quantum algorithm QCR-LLM uses on IBM hardware. Digitized counterdiabatic quantum optimization with bias fields.
- **QCR-LLM** (Flores-Garrigos et al., 2025): Reports that BF-DCQO "slightly improves over the classical SA baseline +1.0 pp in Causal, and +0.5 pp in NYCC" on real quantum hardware (Table II), demonstrating that the solver can access the optimal structure of the reasoning Hamiltonian.

---

## Stage 6: Stability Ranking + Output

### What it does

Post-processes the 100 solver solutions to select the most stable fragments.

1. Sort all 100 solutions by energy (ascending)
2. Take the bottom 25% (25 lowest-energy solutions)
3. For each fragment, count how often it appears in these 25 solutions (inclusion frequency)
4. Select fragments with inclusion frequency ≥ 50%
5. Sort selected fragments by frequency (descending)
6. Return as the optimized fragment subset

### Why we do it

**The ground state alone is fragile.** A single lowest-energy solution might include fragments that are there by chance — the solver happened to flip that bit in its best run. Stability ranking asks a stronger question: "which fragments consistently appear in *good* solutions, not just the *best* solution?"

This is the statistical physics concept of **thermal stability**. In a physical system, the ground state configuration is the T=0 answer. But at finite temperature, some variables fluctuate while others are "frozen" — locked into their ground-state value across the entire low-energy manifold. Frozen variables are the stable ones.

QCR-LLM: "fragments that consistently appear across these near-optimal configurations are considered highly stable and form the backbone of the aggregated reasoning chain, while those with fluctuating or marginal inclusion are treated as context-dependent or peripheral. This frequency-based ranking naturally differentiates essential reasoning steps from optional or redundant ones" (Section III).

### Why thresholds specifically

The two thresholds (selection_threshold=0.25 and inclusion_threshold=0.50) control the tradeoff between precision and recall:

- **selection_threshold = 0.25**: Only consider the best 25% of solutions. This filters out high-energy solutions where the solver was far from the ground state. Lower threshold = stricter (fewer solutions considered, more precise). Higher = more inclusive.
- **inclusion_threshold = 0.50**: A fragment must appear in at least 50% of the selected solutions to be included. This is the "majority of good solutions" criterion. Lower threshold = include more fragments (higher recall). Higher = include fewer (higher precision).

QCR-LLM: "this threshold, however, can be adjusted to control the trade-off between information richness and prompt compactness; a lower threshold includes more reasoning [...] while a higher threshold yields a more concise yet semantically diverse reasoning chain" (Section III).

### Research foundation

- **Stability analysis in spin glasses** (Mezard et al., 1987, "Spin Glass Theory and Beyond"): The concept of "frozen" variables in disordered systems. Variables that take the same value across many near-ground-state configurations are structurally stable.
- **QCR-LLM** (Flores-Garrigos et al., 2025): Implemented this as "expected inclusion frequencies" computed from the low-energy subset, with an adjustable threshold (Figure 1b shows fragments sorted by inclusion probability with the 50% threshold line).
- **Ensemble methods in ML** (Breiman, 1996, "Bagging Predictors"): The same principle — aggregate predictions from multiple models (here, multiple solver runs) to reduce variance and increase robustness.
