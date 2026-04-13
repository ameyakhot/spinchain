"""Dataset loader registry."""

from __future__ import annotations

from benchmarks.datasets.base import DatasetLoader, Problem
from benchmarks.datasets.gsm8k import GSM8KLoader

LOADERS: dict[str, DatasetLoader] = {
    "gsm8k": GSM8KLoader(),
}


def get_loader(name: str) -> DatasetLoader:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADERS.keys())}")
    return LOADERS[name]


__all__ = ["Problem", "DatasetLoader", "get_loader"]
