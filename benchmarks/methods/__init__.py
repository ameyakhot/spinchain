"""Method registry."""

from __future__ import annotations

from benchmarks.config import BenchmarkConfig
from benchmarks.methods.base import Method, MethodResult
from benchmarks.methods.majority_vote import MajorityVote
from benchmarks.methods.random_selection import RandomSelection
from benchmarks.methods.spinchain_method import SpinChainMethod
from benchmarks.methods.union import UnionMethod


def get_methods(names: list[str], config: BenchmarkConfig) -> list[Method]:
    registry: dict[str, Method] = {
        "majority_vote": MajorityVote(),
        "random": RandomSelection(seed=config.seed),
        "spinchain": SpinChainMethod(config=config),
        "union": UnionMethod(similarity_threshold=config.similarity_threshold),
    }
    methods = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown method: {name}. Available: {list(registry.keys())}")
        methods.append(registry[name])
    return methods


__all__ = ["Method", "MethodResult", "get_methods"]
