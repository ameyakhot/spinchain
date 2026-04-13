"""Generate reasoning chains via the Anthropic API."""

from __future__ import annotations

import time

import anthropic

from benchmarks.config import SYSTEM_PROMPTS


class ChainGenerator:
    """Generates K diverse reasoning chains for a given question."""

    def __init__(self, model: str, temperature: float, max_retries: int = 5):
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def generate(self, question: str, k: int, dataset: str) -> list[str]:
        """Generate K reasoning chains for a question.

        Uses exponential backoff on rate limit errors.
        """
        system_prompt = SYSTEM_PROMPTS.get(dataset, SYSTEM_PROMPTS["gsm8k"])
        chains = []

        for _ in range(k):
            chain = self._generate_one(question, system_prompt)
            chains.append(chain)

        return chains

    def _generate_one(self, question: str, system_prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": question}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed after {self.max_retries} retries")
