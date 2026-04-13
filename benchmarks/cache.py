"""JSONL-based chain cache to avoid regenerating LLM completions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class ChainCache:
    """Reads and writes cached reasoning chains as JSONL files.

    Filename encodes the generation config so different configurations
    don't collide: {dataset}_{model}_{temperature}_k{chains}.jsonl
    """

    def __init__(self, cache_dir: str, dataset: str, model: str, temperature: float, chains: int):
        os.makedirs(cache_dir, exist_ok=True)
        safe_model = model.replace("/", "_")
        filename = f"{dataset}_{safe_model}_t{temperature}_k{chains}.jsonl"
        self.path = os.path.join(cache_dir, filename)
        self._cache: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._cache[entry["problem_id"]] = entry["chains"]

    def get(self, problem_id: str) -> list[str] | None:
        return self._cache.get(problem_id)

    def put(self, problem_id: str, chains: list[str]) -> None:
        self._cache[problem_id] = chains
        entry = {
            "problem_id": problem_id,
            "chains": chains,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def __contains__(self, problem_id: str) -> bool:
        return problem_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)
