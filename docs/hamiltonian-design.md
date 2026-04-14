# Hamiltonian Design: From Descriptive to Prescriptive

## The Problem with the Current Hamiltonian

The current Hamiltonian has 9 signals across 10 hyperparameters. All 9 signals are **descriptive** — they describe properties of fragments (what they're about, how popular they are, who they co-occur with). None are **prescriptive** — none judge whether a fragment's reasoning is correct.

936 configurations were tested. Zero outperform majority vote when the majority is wrong. The ground state of the current Hamiltonian is the consensus answer, not the correct answer. These are different configurations when the majority makes an error.

## Why This Happens

The Hamiltonian can only encode information present in its terms. The current terms are computed from:

- Fragment embeddings → semantic content (what it's about)
- Fragment sources → structural membership (which chain it came from)
- Fragment numbers → arithmetic relationships (what it computes)

None of these carry a label that says "this fragment is correct." The optimization is unsupervised — it finds the most popular/diverse/coherent subset, which is not necessarily the most correct subset.

## The Missing Stage: Verification

The corrected pipeline inserts **Error Classification** and **Cluster Integrity Analysis** between fragment extraction and Hamiltonian construction. These stages produce a verification signal that the Hamiltonian currently lacks.

## Pipeline Flowchart

```
┌─────────────────────────────────────────────┐
│  INPUT: K reasoning chains + question text   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: Fragment Extraction                │
│                                              │
│  Split chains into sentences                 │
│  Embed with SentenceTransformer              │
│  Deduplicate by cosine similarity            │
│                                              │
│  Output: R fragments, embeddings, sources    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: Error Classification               │
│                                              │
│  For each fragment:                          │
│    ├─ Contains arithmetic expression?        │
│    │    Parse: "80000 + 50000 = 130000"      │
│    │    Verify: eval(80000 + 50000) == 130000│
│    │    Tag: VERIFIED_CORRECT or             │
│    │         VERIFIED_WRONG                  │
│    │                                         │
│    ├─ Contains numbers but no equation?      │
│    │    Tag: UNVERIFIABLE_NUMERIC            │
│    │                                         │
│    └─ Pure text reasoning?                   │
│         Tag: UNVERIFIABLE_TEXT               │
│                                              │
│  Output: verification_score per fragment     │
│    +1.0 = verified correct arithmetic        │
│     0.0 = unverifiable                       │
│    -1.0 = verified wrong arithmetic          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 3: Answer Cluster Integrity           │
│                                              │
│  Extract final answer from each chain        │
│  Group chains by answer → clusters           │
│  For each cluster:                           │
│    Count verified_correct fragments          │
│    Count verified_wrong fragments            │
│    Compute cluster_integrity =               │
│      (correct - wrong) / total_verifiable    │
│                                              │
│  Output: per-cluster integrity score         │
│          per-fragment cluster membership      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 4: Hamiltonian Construction           │
│                                              │
│  Combine descriptive terms (existing)        │
│  with prescriptive terms (new)               │
│  into a single QUBO energy function          │
│                                              │
│  See: Term Specification below               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 5: Solve                              │
│                                              │
│  Simulated Annealing (classical, current)    │
│  D-Wave QPU (quantum annealing, future)      │
│  Optical Ising machine (COBI, future)        │
│  QAOA circuit (gate-based quantum, future)   │
│                                              │
│  All solvers take the same QUBO/BQM input    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 6: Stability Ranking + Output         │
│                                              │
│  Take bottom 25% of solutions by energy      │
│  Select fragments appearing in ≥50% of them  │
│  Return optimized fragment subset            │
└─────────────────────────────────────────────┘
```

## Term Specification

### Linear terms (external field, per fragment)

```
h_i = -μ · popularity_i           (existing: prefer common fragments)
    + α · risk_i                   (existing: penalize uncertain fragments)
    - φ · verification_i           (NEW: reward verified-correct, penalize verified-wrong)
    - κ · anchor_i                 (existing: pin conclusion fragments)
    - ψ · cluster_integrity_i      (NEW: reward fragments from high-integrity clusters)
```

**φ · verification_i** — the fragment-level verification term:

| Fragment content | verification_i | Effect |
|-----------------|---------------|--------|
| Contains verified-correct arithmetic (e.g., "80000 + 50000 = 130000" and 80000+50000 does equal 130000) | +1.0 | Energy lowered by φ (selected) |
| Contains verified-wrong arithmetic (e.g., "80000 + 50000 = 120000" and 80000+50000 does not equal 120000) | -1.0 | Energy raised by φ (rejected) |
| No verifiable arithmetic | 0.0 | No effect (judged by other terms) |

**ψ · cluster_integrity_i** — the cluster-level integrity term:

| Fragment's cluster | cluster_integrity | Effect |
|-------------------|------------------|--------|
| Cluster has mostly verified-correct fragments | High (+) | Energy lowered (cluster favored) |
| Cluster has mostly verified-wrong fragments | Low (-) | Energy raised (cluster disfavored) |
| Cluster has no verifiable fragments | 0.0 | No effect |

This propagates fragment-level verification to the cluster level. Even unverifiable fragments get a boost if they belong to a cluster whose verifiable fragments check out.

### Quadratic terms (spin-spin coupling, per fragment pair)

```
J_ij = -β · co_occurrence_ij       (existing: attract co-occurring fragments)
     + β·λ² · similarity_ij        (existing: repel similar fragments)
     - ε · cluster_coherence_ij    (existing: attract same-cluster fragments)
     - η · numerical_consistency_ij (existing: attract arithmetically related fragments)
     - ω · verification_agreement_ij (NEW: attract same-verification fragments)
```

**ω · verification_agreement_ij** — verification-based coupling:

| Fragment i | Fragment j | verification_agreement_ij | Effect |
|-----------|-----------|--------------------------|--------|
| Verified correct | Verified correct | +1.0 | Strong attraction (ferromagnetic) |
| Verified correct | Verified wrong | -1.0 | Strong repulsion (antiferromagnetic) |
| Verified wrong | Verified wrong | +1.0 | Attraction (cluster together for rejection) |
| Either unverifiable | Any | 0.0 | No coupling |

This creates **repulsion between correct and incorrect fragments**. The solver cannot select both — it must pick a side. Combined with φ (which makes the correct side lower energy), the solver is pushed toward the verified-correct cluster.

## Energy Landscape Transformation

### Current landscape (without verification)

```
Energy
  ▲
  │
  │   ╭────────╮
  │   │        │  ← one wide well around popular fragments
  │   │        │     (wrong and correct mixed together)
  │   │        │     ground state = consensus = majority vote
  │───╯        ╰──────────────────────
  └───────────────────────────────────► fragment configurations
```

The solver finds the consensus. When the majority is wrong, the consensus is wrong.

### Proposed landscape (with verification)

```
Energy
  ▲
  │            ╭────╮
  │   ╭────╮   │    │
  │   │    │   │    │  ← two wells, separated by verification
  │   │    │   │    │
  │   │    │   │    │
  │───╯    ╰───╯    ╰────────────────
  └───────────────────────────────────► fragment configurations
      ▲              ▲
      │              │
   verified       popular but
   correct        has errors
   (deeper        (shallower
    well)          well)
```

The φ and ω terms make the verified-correct well deeper. Even if the popular cluster has more fragments (wider well), the verified-correct cluster has lower energy per fragment (deeper well). The solver selects the correct minority when the verification signal is strong enough.

## Error Type Coverage

| Error type | Verifiable? | Mechanism | Term |
|-----------|-------------|-----------|------|
| Arithmetic error (80+50=120) | Yes | Parse and eval expression | φ (linear) |
| Wrong intermediate number carried forward | Yes | Track number provenance across fragments | η (quadratic) |
| Interpretation error (150% of what?) | No | Requires understanding problem semantics | Not addressable |
| Logic non-sequitur | Partially | Check if conclusion follows from premises | Future work |
| Hallucinated fact | No | Requires external knowledge | Not addressable |

The verification terms (φ, ψ, ω) can catch arithmetic errors and wrong-number-propagation errors. They cannot catch interpretation errors or hallucinations. This is a known boundary — the Hamiltonian is effective on the subset of errors that are mechanically verifiable.

## Implementation Requirements

### Stage 2 (Error Classification) needs:

1. **Arithmetic expression parser** — extract patterns like "A + B = C", "A × B = C", "A - B = C", "A / B = C" from natural language text
2. **Expression evaluator** — compute the left side and compare to the right side
3. **Tolerance handling** — approximate equality for floating point and rounding

This is a regex + eval problem. No ML models, no API calls, no external dependencies beyond Python's standard library.

### Stage 3 (Cluster Integrity) needs:

1. **Answer extraction** — already implemented in `benchmarks/extractors/answer_extractor.py`
2. **Cluster grouping** — already implemented in `benchmarks/sweep.py`
3. **Integrity score computation** — count verified-correct and verified-wrong per cluster, compute ratio

### Stage 4 (Hamiltonian Construction) needs:

1. Three new methods in `CoefficientBuilder`:
   - `compute_verification_weights(verification_scores)` → linear term (φ)
   - `compute_cluster_integrity_weights(fragment_sources, cluster_integrity_scores)` → linear term (ψ)
   - `compute_verification_agreement(verification_scores)` → quadratic term (ω)

## Test Case for Validation

The correct test case has chains with **arithmetic errors** (not interpretation errors):

```
Chain 1 (wrong — arithmetic error):
  "80000 + 50000 = 120000. Sells for 150% = 180000. Profit = 60000."
                    ▲
            80000 + 50000 ≠ 120000  ← detectable by Stage 2

Chain 2 (wrong — same arithmetic error):
  "Cost: 80000 + 50000 = 120000. At 150%, sells for 180000. Profit: 60000."

Chain 3 (correct):
  "80000 + 50000 = 130000. Sells for 150% = 195000. Profit = 65000."
```

Stage 2 would tag Chain 1 and 2 fragments as VERIFIED_WRONG (80000+50000≠120000). Chain 3 fragments would be VERIFIED_CORRECT. The φ term raises the energy of wrong-chain fragments, making the correct minority cluster the ground state.
