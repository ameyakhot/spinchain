"""Base types for benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Problem:
    id: str
    question: str
    ground_truth: str
    dataset: str
    choices: list[str] | None = None


class DatasetLoader(Protocol):
    name: str

    def load(self, limit: int | None = None) -> list[Problem]: ...
