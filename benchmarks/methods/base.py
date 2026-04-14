"""Base types for benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from benchmarks.datasets.base import Problem

_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    """Approximate token count."""
    return len(text) // _CHARS_PER_TOKEN


def total_chain_tokens(chains: list[str]) -> int:
    """Total tokens across all chains."""
    return sum(count_tokens(c) for c in chains)


@dataclass
class MethodResult:
    method: str
    predicted_answer: str | None
    correct: bool
    metadata: dict = field(default_factory=dict)


class Method(Protocol):
    name: str

    def run(self, chains: list[str], problem: Problem) -> MethodResult: ...
