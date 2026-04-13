"""Benchmark configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    chains: int = 7
    limit: int | None = None
    seed: int = 42
    methods: list[str] = field(default_factory=lambda: [
        "spinchain", "majority_vote", "random", "union",
    ])
    dataset: str = "gsm8k"
    output: str | None = None
    no_generate: bool = False

    # SpinChain solver params
    num_reads: int = 100
    num_sweeps: int = 1000
    similarity_threshold: float = 0.85
    selection_threshold: float = 0.25
    inclusion_threshold: float = 0.50

    # Cache
    cache_dir: str = "benchmarks/.cache"


SYSTEM_PROMPTS = {
    "gsm8k": (
        "Solve this math problem step by step. Show your reasoning clearly. "
        "End your response with 'The answer is [number].' where [number] is "
        "a single integer."
    ),
    "arc": (
        "Answer this multiple choice science question. Think step by step. "
        "End your response with 'The answer is [letter].' where [letter] is "
        "A, B, C, D, or E."
    ),
    "strategyqa": (
        "Answer this yes/no question. Think step by step. "
        "End your response with 'The answer is yes.' or 'The answer is no.'"
    ),
}
