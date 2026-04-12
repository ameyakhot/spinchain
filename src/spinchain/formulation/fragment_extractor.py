"""Extract and deduplicate reasoning fragments from multiple LLM completions."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class FragmentExtractor:
    """Extracts distinct reasoning fragments from N CoT completions.

    Following QCR-LLM: splits each completion into sentence-level fragments,
    computes embeddings, merges near-duplicates via cosine similarity threshold,
    and returns a normalized fragment pool.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def extract_fragments(self, completions: list[str]) -> list[str]:
        """Extract and deduplicate fragments from multiple completions.

        Args:
            completions: List of N CoT completion strings from the LLM.

        Returns:
            List of R distinct reasoning fragments.
        """
        raw_fragments = []
        fragment_sources: list[set[int]] = []

        for comp_idx, completion in enumerate(completions):
            sentences = self._split_into_sentences(completion)
            for sentence in sentences:
                cleaned = sentence.strip()
                if len(cleaned) < 10:
                    continue
                raw_fragments.append(cleaned)
                fragment_sources.append({comp_idx})

        if not raw_fragments:
            return []

        embeddings = self.model.encode(raw_fragments, convert_to_numpy=True)
        merged_fragments, merged_sources, merged_embeddings = self._merge_similar(
            raw_fragments, fragment_sources, embeddings
        )

        self._last_fragments = merged_fragments
        self._last_sources = merged_sources
        self._last_embeddings = merged_embeddings
        self._num_completions = len(completions)

        return merged_fragments

    @property
    def fragment_embeddings(self) -> np.ndarray:
        return self._last_embeddings

    @property
    def fragment_sources(self) -> list[set[int]]:
        return self._last_sources

    @property
    def num_completions(self) -> int:
        return self._num_completions

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into reasoning sentences. Simple split on period/newline."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [s for s in sentences if s.strip()]

    def _merge_similar(
        self,
        fragments: list[str],
        sources: list[set[int]],
        embeddings: np.ndarray,
    ) -> tuple[list[str], list[set[int]], np.ndarray]:
        """Merge fragments whose cosine similarity exceeds threshold."""
        n = len(fragments)
        if n == 0:
            return [], [], np.array([])

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        merged = [False] * n
        result_fragments = []
        result_sources = []
        result_embeddings = []

        for i in range(n):
            if merged[i]:
                continue
            group_sources = set(sources[i])
            for j in range(i + 1, n):
                if merged[j]:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    group_sources |= sources[j]
                    merged[j] = True
            result_fragments.append(fragments[i])
            result_sources.append(group_sources)
            result_embeddings.append(embeddings[i])

        return result_fragments, result_sources, np.array(result_embeddings)
