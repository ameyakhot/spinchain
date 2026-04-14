# QUBO Signal Inventory

## Purpose

This document catalogs every signal available in the SpinChain environment that could be encoded as a term in the QUBO Hamiltonian. Signals are grouped by type and marked by their current usage status.

## Signals Currently in the QUBO

### 1. Fragment Popularity (linear)

- **Source:** `FragmentExtractor.fragment_sources`
- **Formula:** `p_i = len(sources[i]) / num_completions`
- **QUBO term:** `w_i = -mu * p_i` (lower energy for popular fragments)
- **What it captures:** How many chains contain this fragment
- **Limitation:** Explicitly favors the majority. When the majority is wrong, this term actively selects wrong fragments.

### 2. Fragment Risk / Variance (linear)

- **Source:** Derived from popularity
- **Formula:** `risk_i = p_i * (1 - p_i)`
- **QUBO term:** `w_i += alpha * risk_i` (penalizes fragments with high variance)
- **What it captures:** Fragments that appear in roughly half the chains are "risky" — uncertain signal
- **Limitation:** Peaks at p=0.5, zero at p=0 and p=1. Penalizes contested fragments regardless of correctness.

### 3. Co-occurrence Correlation (quadratic)

- **Source:** `FragmentExtractor.fragment_sources` (set intersection)
- **Formula:** `corr_ij = (n_ij / n) - p_i * p_j`, then z-score normalized
- **QUBO term:** `w_ij = -beta * z_corr_ij` (attracts co-occurring fragments)
- **What it captures:** Whether two fragments appear together more often than chance predicts
- **Limitation:** Reflects chain membership, not reasoning quality. Fragments co-occur because they came from the same chain, not because they're logically connected.

### 4. Semantic Similarity (quadratic)

- **Source:** `FragmentExtractor.fragment_embeddings` (cosine similarity)
- **Formula:** `sim_ij = cosine(embedding_i, embedding_j)`
- **QUBO term:** `w_ij += beta * lambda_sim^2 * sim_ij` (repels similar fragments)
- **What it captures:** Semantic redundancy between fragments
- **Limitation:** Dominates the quadratic terms (~2x larger than linear). Drives selection toward diversity, which is orthogonal to correctness.

### 5. Question Relevance (linear, recently added)

- **Source:** `SentenceTransformer.encode(question)` vs. fragment embeddings
- **Formula:** `relevance_i = cosine(question_embedding, fragment_embedding_i)`
- **QUBO term:** `w_i += -gamma * relevance_i`
- **What it captures:** How semantically close a fragment is to the original question
- **Limitation:** All fragments about the same math problem score similarly. Cannot distinguish correct vs. incorrect reasoning about the same topic.

## Signals Available but Unused

### 6. Fragment Position Within Chain

- **Source:** Fragment index within the chain's sentence list (available during extraction in `FragmentExtractor._split_into_sentences`)
- **Type:** Linear
- **What it could capture:** Early fragments tend to be problem setup (restating the question), middle fragments contain reasoning steps, late fragments contain conclusions/answers. Position could weight conclusion fragments differently from setup fragments.
- **Encoding:** `position_i = fragment_index / total_fragments_in_chain` (0.0 = start, 1.0 = end)

### 7. Fragment Length

- **Source:** `len(fragment.split())` or `len(fragment)`
- **Type:** Linear
- **What it could capture:** Very short fragments may be filler ("Let me think about this"). Very long fragments may contain more substantive reasoning. Length is a weak proxy for information density.
- **Encoding:** Word count or character count, normalized

### 8. Sentence Type / Content Markers

- **Source:** Regex on fragment text
- **Type:** Linear
- **What it could capture:** Whether a fragment contains:
  - Numbers or equations (computational step)
  - Conclusion markers ("therefore", "so", "the answer is")
  - Hedging language ("wait", "actually", "let me re-read")
  - Mathematical operators (+, -, ×, ÷, =)
- **Encoding:** Binary or count features per fragment

### 9. Numerical Content

- **Source:** Regex extraction of numbers from fragment text
- **Type:** Linear and quadratic
- **What it could capture:**
  - Which numbers appear in each fragment
  - Whether a fragment's output number matches another fragment's input (arithmetic chaining)
  - Whether intermediate values are arithmetically consistent
- **Encoding:** Extract numbers, compare across fragments
- **Potential:** High for math problems. A fragment claiming "17 - 9 = 8" is verifiable without knowing the answer. This is a **logical** signal, not a semantic one.

### 10. Answer Cluster Membership

- **Source:** Extract final answer from each chain, group chains by answer
- **Type:** Linear (per-fragment label)
- **What it could capture:** Which "answer camp" a fragment belongs to. Fragments from chains that produce answer A vs. answer B can be tagged. This doesn't tell you which camp is correct, but it enables cluster-aware selection strategies.
- **Encoding:** `cluster_i = answer_of_chain_containing_fragment_i`
- **Potential:** Combined with other signals, could enable "select the most internally consistent cluster" rather than "select the most popular fragments."

### 11. Cross-Cluster Fragment Sharing

- **Source:** Fragment sources + answer cluster labels
- **Type:** Linear
- **What it could capture:** Fragments shared across answer clusters may represent agreed-upon setup or universally accepted reasoning steps. Fragments unique to one cluster represent the divergence point.
- **Encoding:** `shared_i = 1 if fragment appears in chains with different final answers, else 0`
- **Potential:** Shared fragments are likely correct (both camps agree). Unique fragments are where the error lives.

### 12. Internal Arithmetic Consistency

- **Source:** Number extraction + symbolic evaluation
- **Type:** Linear or quadratic
- **What it could capture:** Within a chain's fragments, do the numbers form a consistent arithmetic chain? E.g., if fragment A says "cost is 130000" and fragment B says "150% of cost is 195000", check: 130000 × 1.5 = 195000? If yes, these fragments are arithmetically consistent.
- **Encoding:** For each fragment pair from the same chain, compute consistency score. Aggregate per fragment.
- **Potential:** This is the only signal that can detect *incorrect computation* without knowing the answer. A chain that says "130000 × 1.5 = 195000" is internally consistent. A chain that says "80000 × 2.5 = 200000" is also internally consistent. But a chain that says "80000 + 50000 = 120000" would be flagged as inconsistent.
- **Complexity:** Requires parsing arithmetic expressions, which is fragile on natural language.

### 13. Fragment-to-Chain-Answer Alignment

- **Source:** Fragment text + chain's extracted final answer
- **Type:** Linear
- **What it could capture:** Does this fragment's content (especially numbers) align with the chain's final answer? A fragment saying "the farmer has 9 sheep" in a chain whose final answer is "9" is aligned. This measures whether intermediate reasoning supports the conclusion.
- **Encoding:** Check if the chain's final answer number appears in the fragment or is derivable from it.

### 14. Embedding Distance to Answer Fragments

- **Source:** Fragment embeddings, filtered to fragments containing "the answer is"
- **Type:** Quadratic
- **What it could capture:** How semantically close a reasoning fragment is to the conclusion fragments. Reasoning steps that lead toward the conclusion may cluster near it in embedding space.
- **Limitation:** Same embedding-space limitation as question-relevance — proximity doesn't imply correctness.

## External Resources (available but expensive)

### 15. LLM-as-Judge

- **Source:** Anthropic API call
- **What it could capture:** Ask a model to rate each fragment's reasoning quality
- **Cost:** One API call per fragment, defeats the purpose of local optimization
- **Potential:** Could be used as a one-time labeling step to train a lightweight classifier, then encode classifier scores as QUBO weights

### 16. Symbolic Math Evaluator

- **Source:** Python `eval()`, `sympy`, or similar
- **What it could capture:** Parse and evaluate arithmetic expressions in fragments
- **Potential:** High for math domains. Could verify "17 - 9 = 8" programmatically.
- **Limitation:** Requires robust natural language → expression parsing

## Signal Classification

| # | Signal | Type | Domain | Status | Correctness Signal? |
|---|--------|------|--------|--------|-------------------|
| 1 | Popularity | Linear | General | In QUBO | No — favors majority |
| 2 | Risk/Variance | Linear | General | In QUBO | No — penalizes uncertainty |
| 3 | Co-occurrence | Quadratic | General | In QUBO | No — reflects chain membership |
| 4 | Semantic similarity | Quadratic | General | In QUBO | No — measures redundancy |
| 5 | Question relevance | Linear | General | In QUBO | No — all on-topic fragments score equally |
| 6 | Position in chain | Linear | General | Unused | Weak — positional bias |
| 7 | Fragment length | Linear | General | Unused | Weak — length ≠ quality |
| 8 | Content markers | Linear | General | Unused | Weak — surface features |
| 9 | Numerical content | Linear+Quad | Math | Unused | Possible — numbers are verifiable |
| 10 | Answer cluster | Linear | General | Unused | Indirect — enables cluster-aware strategies |
| 11 | Cross-cluster sharing | Linear | General | Unused | Moderate — shared = agreed upon |
| 12 | Arithmetic consistency | Linear+Quad | Math | Unused | **Strong** — detects computation errors |
| 13 | Answer alignment | Linear | General | Unused | Moderate — checks internal coherence |
| 14 | Distance to answer frags | Quadratic | General | Unused | No — same embedding limitation |
| 15 | LLM-as-judge | Linear | General | Unused | Strong — but expensive |
| 16 | Symbolic evaluator | Linear | Math | Unused | **Strong** — verifies arithmetic |

## Key Insight

Signals 1-5 and 14 operate in **embedding space** — they measure what fragments are *about*. The sweep across 600 configurations proved that no combination of embedding-space signals can distinguish correct from incorrect reasoning about the same topic.

Signals 9, 12, 13, and 16 operate on **logical/arithmetic content** — they measure what fragments *compute*. These are the candidates most likely to carry a genuine correctness signal, but they are domain-specific (math) and require natural language parsing.

Signals 10-11 operate on **structural properties** of the chain-fragment relationship — they don't directly detect correctness but could enable strategies that select the most internally consistent answer cluster rather than the most popular one.
