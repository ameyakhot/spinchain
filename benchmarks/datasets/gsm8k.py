"""GSM8K dataset loader."""

from __future__ import annotations

import re

from datasets import load_dataset

from benchmarks.datasets.base import Problem


class GSM8KLoader:
    name = "gsm8k"

    def load(self, limit: int | None = None) -> list[Problem]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = []
        for i, example in enumerate(ds):
            if limit is not None and i >= limit:
                break
            answer = _extract_answer(example["answer"])
            if answer is None:
                continue
            problems.append(Problem(
                id=f"gsm8k_{i}",
                question=example["question"],
                ground_truth=answer,
                dataset="gsm8k",
            ))
        return problems


def _extract_answer(answer_text: str) -> str | None:
    """Extract the integer answer after #### in GSM8K format."""
    match = re.search(r"####\s*([\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None
