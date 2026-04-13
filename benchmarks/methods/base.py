"""Base types for benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from benchmarks.datasets.base import Problem


@dataclass
class MethodResult:
    method: str
    predicted_answer: str | None
    correct: bool
    metadata: dict = field(default_factory=dict)


class Method(Protocol):
    name: str

    def run(self, chains: list[str], problem: Problem) -> MethodResult: ...
