"""Regex-based answer extraction from reasoning text, per dataset type."""

from __future__ import annotations

import re


def extract_answer(text: str, dataset: str, choices: list[str] | None = None) -> str | None:
    """Extract a final answer from reasoning text.

    Returns the normalized answer string, or None if no answer found.
    """
    if dataset == "gsm8k":
        return _extract_gsm8k(text)
    elif dataset == "arc":
        return _extract_arc(text)
    elif dataset == "strategyqa":
        return _extract_strategyqa(text)
    return None


def _extract_gsm8k(text: str) -> str | None:
    """Extract numeric answer from GSM8K-style reasoning.

    Tries in order:
    1. "#### [number]" (GSM8K canonical format)
    2. "the answer is [number]"
    3. Last bare number in the text
    """
    # Pattern 1: GSM8K canonical
    match = re.search(r"####\s*([\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 2: "the answer is [number]"
    match = re.search(r"[Tt]he answer is\s*\$?\s*([\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 3: last number in text (integers and decimals)
    numbers = re.findall(r"(?<!\w)([\d,]+)(?!\w)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def _extract_arc(text: str) -> str | None:
    """Extract letter answer (A-E) from ARC-style reasoning."""
    # "the answer is (B)" or "the answer is B"
    match = re.search(r"[Tt]he answer is\s*\(?([A-Ea-e])\)?", text)
    if match:
        return match.group(1).upper()

    # Last standalone letter A-E
    matches = re.findall(r"\b([A-Ea-e])\b", text)
    if matches:
        return matches[-1].upper()

    return None


def _extract_strategyqa(text: str) -> str | None:
    """Extract yes/no answer from StrategyQA-style reasoning."""
    # "the answer is yes/no"
    match = re.search(r"[Tt]he answer is\s*(yes|no)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Last yes/no in text
    matches = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()

    return None
