"""Score predicted answers against ground truth."""

from __future__ import annotations


def score(predicted: str | None, ground_truth: str, dataset: str) -> bool:
    """Return True if predicted matches ground truth for the given dataset."""
    if predicted is None:
        return False

    if dataset == "gsm8k":
        return _normalize_number(predicted) == _normalize_number(ground_truth)
    elif dataset == "arc":
        return predicted.strip().upper() == ground_truth.strip().upper()
    elif dataset == "strategyqa":
        return predicted.strip().lower() == ground_truth.strip().lower()

    return predicted.strip() == ground_truth.strip()


def _normalize_number(s: str) -> str:
    """Normalize a number string: strip whitespace, commas, leading zeros."""
    s = s.strip().replace(",", "")
    try:
        return str(int(s))
    except ValueError:
        return s
