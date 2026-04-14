"""Build HUBO/QUBO coefficients from fragment statistics.

Implements the coefficient design from QCR-LLM (arXiv 2510.24509) plus
cluster-aware extensions:
- Linear: w_i = -mu * p_i + alpha * risk_i - delta * d_i - kappa * a_i
- Quadratic: w_ij = -beta * (corr_ij - lambda^2 * sim(i,j)) - epsilon * c_ij
"""

from __future__ import annotations

import re

import numpy as np


class CoefficientBuilder:
    """Computes linear and quadratic coefficients for the QUBO formulation.

    Args:
        mu: Weight for fragment popularity (higher = prefer common fragments).
        alpha: Weight for fragment stability/risk penalty.
        beta: Weight for pairwise coherence terms.
        lambda_sim: Semantic similarity penalty factor.
        gamma: Weight for question-relevance term.
        delta: Weight for cross-cluster agreement (reward shared fragments).
        epsilon: Weight for answer cluster coherence (attract same-conclusion fragments).
        kappa: Weight for answer fragment anchoring (pin conclusions).
        regularization: Small constant for z-score stability.
    """

    def __init__(
        self,
        mu: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.0,
        lambda_sim: float = 0.3,
        gamma: float = 0.0,
        delta: float = 0.0,
        epsilon: float = 0.0,
        kappa: float = 0.0,
        eta: float = 0.0,
        phi: float = 0.0,
        psi: float = 0.0,
        omega: float = 0.0,
        regularization: float = 1e-6,
    ):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lambda_sim = lambda_sim
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.kappa = kappa
        self.eta = eta
        self.phi = phi
        self.psi = psi
        self.omega = omega
        self.regularization = regularization

    def compute_linear_weights(
        self,
        fragment_sources: list[set[int]],
        num_completions: int,
    ) -> np.ndarray:
        """Compute linear (1-body) weights for each fragment.

        w_i = -mu * p_i + alpha * risk_i
        where p_i = popularity, risk_i = p_i * (1 - p_i)
        """
        r = len(fragment_sources)
        weights = np.zeros(r)

        for i in range(r):
            p_i = len(fragment_sources[i]) / num_completions
            risk_i = p_i * (1.0 - p_i)
            weights[i] = -self.mu * p_i + self.alpha * risk_i

        return weights

    def compute_relevance_weights(
        self,
        question_embedding: np.ndarray,
        fragment_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute question-relevance weights for each fragment.

        w_i = -gamma * cosine_sim(question, fragment_i)

        Fragments semantically closer to the question receive lower energy
        (more likely to be selected). This signal is independent of popularity.
        """
        q_norm = np.linalg.norm(question_embedding)
        if q_norm < 1e-10:
            return np.zeros(len(fragment_embeddings))

        q_normed = question_embedding / q_norm
        f_norms = np.linalg.norm(fragment_embeddings, axis=1, keepdims=True)
        f_norms = np.maximum(f_norms, 1e-10)
        f_normed = fragment_embeddings / f_norms

        cosine_sim = f_normed @ q_normed
        return -self.gamma * cosine_sim

    def compute_shared_weights(
        self,
        fragment_sources: list[set[int]],
        answer_clusters: dict[str, set[int]],
    ) -> np.ndarray:
        """Compute cross-cluster agreement weights.

        w_i = -delta * d_i
        where d_i = fraction of answer clusters containing fragment i.

        Fragments appearing across all answer clusters are universally agreed
        upon and receive lower energy (more likely to be selected).
        """
        r = len(fragment_sources)
        num_clusters = len(answer_clusters)
        if num_clusters == 0:
            return np.zeros(r)

        weights = np.zeros(r)
        for i in range(r):
            clusters_containing = sum(
                1 for chains in answer_clusters.values()
                if fragment_sources[i] & chains
            )
            d_i = clusters_containing / num_clusters
            weights[i] = -self.delta * d_i

        return weights

    def compute_anchor_weights(
        self,
        fragments: list[str],
    ) -> np.ndarray:
        """Compute answer-anchoring weights.

        w_i = -kappa * a_i
        where a_i = 1 if fragment contains a final answer pattern.

        Pins conclusion fragments so stability ranking doesn't drop them.
        """
        r = len(fragments)
        weights = np.zeros(r)
        pattern = re.compile(
            r"(?:[Tt]he answer is|####)\s*\S", re.IGNORECASE,
        )
        for i in range(r):
            if pattern.search(fragments[i]):
                weights[i] = -self.kappa

        return weights

    def compute_cluster_coherence(
        self,
        fragment_sources: list[set[int]],
        answer_clusters: dict[str, set[int]],
    ) -> np.ndarray:
        """Compute answer cluster coherence weights (quadratic).

        w_ij = -epsilon * c_ij
        where c_ij = fraction of answer clusters where both fragments appear.

        Fragments supporting the same conclusion attract each other,
        creating competing energy wells per answer cluster.
        """
        r = len(fragment_sources)
        num_clusters = len(answer_clusters)
        weights = np.zeros((r, r))

        if num_clusters == 0:
            return weights

        for i in range(r):
            for j in range(i + 1, r):
                clusters_both = sum(
                    1 for chains in answer_clusters.values()
                    if (fragment_sources[i] & chains) and (fragment_sources[j] & chains)
                )
                c_ij = clusters_both / num_clusters
                weights[i, j] = -self.epsilon * c_ij
                weights[j, i] = weights[i, j]

        return weights

    def compute_verification_weights(
        self,
        fragments: list[str],
    ) -> np.ndarray:
        """Compute arithmetic verification weights (linear).

        w_i = -phi * v_i
        where v_i = +1 if fragment has verified-correct arithmetic,
                   -1 if fragment has verified-wrong arithmetic,
                    0 if no verifiable expressions.
        """
        r = len(fragments)
        weights = np.zeros(r)
        for i in range(r):
            weights[i] = -self.phi * _verify_arithmetic(fragments[i])
        return weights

    def compute_cluster_integrity_weights(
        self,
        fragment_sources: list[set[int]],
        answer_clusters: dict[str, set[int]],
        verification_scores: np.ndarray,
    ) -> np.ndarray:
        """Compute cluster integrity weights (linear).

        w_i = -psi * integrity(cluster_of_i)
        where integrity = mean(verification_scores) of verifiable fragments
        in the cluster.

        Propagates verification signal from verifiable to unverifiable
        fragments within the same answer cluster.
        """
        r = len(fragment_sources)
        weights = np.zeros(r)

        # Compute integrity per cluster
        cluster_integrity: dict[str, float] = {}
        for answer, chain_set in answer_clusters.items():
            scores = []
            for i in range(r):
                if fragment_sources[i] & chain_set and verification_scores[i] != 0:
                    scores.append(verification_scores[i])
            cluster_integrity[answer] = (
                sum(scores) / len(scores) if scores else 0.0
            )

        # Assign each fragment its cluster's integrity
        for i in range(r):
            for answer, chain_set in answer_clusters.items():
                if fragment_sources[i] & chain_set:
                    weights[i] = -self.psi * cluster_integrity[answer]
                    break

        return weights

    def compute_verification_agreement(
        self,
        verification_scores: np.ndarray,
    ) -> np.ndarray:
        """Compute verification agreement weights (quadratic).

        w_ij = -omega * sign(v_i) * sign(v_j)
        where v_i, v_j are verification scores.

        Creates ferromagnetic coupling between same-verification fragments
        and antiferromagnetic coupling between correct/wrong fragment pairs.
        """
        r = len(verification_scores)
        weights = np.zeros((r, r))

        for i in range(r):
            if verification_scores[i] == 0:
                continue
            for j in range(i + 1, r):
                if verification_scores[j] == 0:
                    continue
                sign_i = 1.0 if verification_scores[i] > 0 else -1.0
                sign_j = 1.0 if verification_scores[j] > 0 else -1.0
                agreement = sign_i * sign_j
                weights[i, j] = -self.omega * agreement
                weights[j, i] = weights[i, j]

        return weights

    def compute_numerical_consistency(
        self,
        fragments: list[str],
    ) -> np.ndarray:
        """Compute numerical consistency weights (quadratic).

        w_ij = -eta * v_ij
        where v_ij measures arithmetic relationships between numbers in
        fragments i and j.

        Fragments whose numbers are connected by basic arithmetic
        (+, -, ×, ÷) receive ferromagnetic coupling — they form a
        consistent computational chain.
        """
        r = len(fragments)
        weights = np.zeros((r, r))

        if self.eta == 0:
            return weights

        # Extract numbers from each fragment
        fragment_numbers = []
        for frag in fragments:
            nums = _extract_numbers(frag)
            fragment_numbers.append(nums)

        # Compute pairwise consistency
        for i in range(r):
            for j in range(i + 1, r):
                v_ij = _arithmetic_consistency(
                    fragment_numbers[i], fragment_numbers[j],
                )
                if v_ij > 0:
                    weights[i, j] = -self.eta * v_ij
                    weights[j, i] = weights[i, j]

        return weights

    def compute_quadratic_weights(
        self,
        fragment_sources: list[set[int]],
        num_completions: int,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute quadratic (2-body) weights for fragment pairs.

        w_ij = -beta * (z_score(corr_ij) - lambda^2 * sim(i,j))
        where corr_ij = co-occurrence correlation, sim = cosine similarity
        """
        r = len(fragment_sources)
        n = num_completions

        # Popularity
        p = np.array([len(s) / n for s in fragment_sources])

        # Co-occurrence matrix
        co_occurrence = np.zeros((r, r))
        for i in range(r):
            for j in range(i + 1, r):
                n_ij = len(fragment_sources[i] & fragment_sources[j])
                co_occurrence[i, j] = n_ij / n - p[i] * p[j]
                co_occurrence[j, i] = co_occurrence[i, j]

        # Standardize correlations (z-scores)
        upper_tri = co_occurrence[np.triu_indices(r, k=1)]
        if len(upper_tri) > 1 and np.std(upper_tri) > self.regularization:
            mean_c = np.mean(upper_tri)
            std_c = np.std(upper_tri)
            z_corr = (co_occurrence - mean_c) / (std_c + self.regularization)
        else:
            z_corr = co_occurrence

        # Clip: pairs with non-positive raw correlation must not get positive
        # z-scores. Without this, the negative mean (most pairs don't co-occur)
        # inflates uncorrelated pairs, rewarding diverse noise selection.
        z_corr = np.where(co_occurrence <= 0, np.minimum(z_corr, 0.0), z_corr)

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Quadratic weights
        weights = np.zeros((r, r))
        for i in range(r):
            for j in range(i + 1, r):
                weights[i, j] = -self.beta * (
                    z_corr[i, j] - self.lambda_sim**2 * sim_matrix[i, j]
                )
                weights[j, i] = weights[i, j]

        return weights


def _extract_numbers(text: str) -> set[float]:
    """Extract all numbers from a text fragment."""
    matches = re.findall(r"(?<!\w)([\d,]+(?:\.\d+)?)(?!\w)", text)
    numbers = set()
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            if val != 0:
                numbers.add(val)
        except ValueError:
            continue
    return numbers


def _arithmetic_consistency(nums_a: set[float], nums_b: set[float]) -> float:
    """Score arithmetic consistency between two sets of numbers.

    Checks if any number in one set can be derived from numbers in the
    other via +, -, x, /. Returns fraction of derivable relationships.
    """
    if not nums_a or not nums_b:
        return 0.0

    shared = nums_a & nums_b
    derivable = 0
    total_checks = 0

    for target_set, source_set in [(nums_b, nums_a), (nums_a, nums_b)]:
        source_list = sorted(source_set)
        for target in target_set:
            total_checks += 1
            if target in shared:
                derivable += 1
                continue
            found = False
            for k, a in enumerate(source_list):
                for b in source_list[k:]:
                    if _approx_eq(a + b, target):
                        found = True
                    elif _approx_eq(a * b, target):
                        found = True
                    elif _approx_eq(abs(a - b), target):
                        found = True
                    elif b != 0 and _approx_eq(a / b, target):
                        found = True
                    elif a != 0 and _approx_eq(b / a, target):
                        found = True
                    if found:
                        break
                if found:
                    break
            if found:
                derivable += 1

    return derivable / total_checks if total_checks > 0 else 0.0


def _extract_error_details(text: str) -> dict:
    """Extract structured details about arithmetic errors in text."""
    matches = _EXPR_PATTERN.findall(text)
    errors = []
    for lhs_a, op, lhs_b, rhs in matches:
        a = float(lhs_a.replace(",", ""))
        b = float(lhs_b.replace(",", ""))
        expected = float(rhs.replace(",", ""))
        if op == "+":
            actual = a + b
        elif op == "-":
            actual = a - b
        elif op in ("*", "×"):
            actual = a * b
        elif op in ("/", "÷"):
            actual = a / b if b != 0 else float("inf")
        else:
            continue
        if not _approx_eq(actual, expected):
            errors.append({
                "expression": f"{lhs_a} {op} {lhs_b}",
                "stated_result": expected,
                "correct_result": actual,
            })
    return {"arithmetic_errors": errors} if errors else {}


def _approx_eq(a: float, b: float, rtol: float = 1e-6) -> bool:
    """Check approximate equality of two floats."""
    if b == 0:
        return abs(a) < rtol
    return abs(a - b) / max(abs(a), abs(b)) < rtol


# Pattern: "number op number = number" where op is +, -, *, ×, /
_EXPR_PATTERN = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*([+\-*/×÷])\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)"
)


def _verify_arithmetic(text: str) -> float:
    """Parse arithmetic expressions in text and verify them.

    Returns:
        +1.0 if all found expressions are correct
        -1.0 if any expression is incorrect
         0.0 if no expressions found
    """
    matches = _EXPR_PATTERN.findall(text)
    if not matches:
        return 0.0

    for lhs_a, op, lhs_b, rhs in matches:
        a = float(lhs_a.replace(",", ""))
        b = float(lhs_b.replace(",", ""))
        expected = float(rhs.replace(",", ""))

        if op == "+":
            actual = a + b
        elif op == "-":
            actual = a - b
        elif op in ("*", "×"):
            actual = a * b
        elif op in ("/", "÷"):
            if b == 0:
                continue
            actual = a / b
        else:
            continue

        if not _approx_eq(actual, expected):
            return -1.0

    return 1.0
