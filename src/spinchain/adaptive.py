"""Adaptive coefficient adjustment based on error history.

Reads the error log and returns QUBO hyperparameters tuned to the
observed error patterns. More frequent error types get stronger
verification weights.
"""

from __future__ import annotations

from spinchain.error_logger import get_error_logger

# Base coefficients (proven in sweep: psi >= 1.0 is necessary and sufficient)
_BASE = {
    "mu": 1.0,
    "alpha": 0.5,
    "beta": 1.0,
    "lambda_sim": 0.3,
    "gamma": 0.0,
    "delta": 0.0,
    "epsilon": 0.0,
    "kappa": 2.0,
    "phi": 1.0,
    "psi": 2.0,
    "omega": 1.0,
    "eta": 0.0,
}

# Thresholds for boosting coefficients
_BOOST_THRESHOLD = 5      # errors before boosting starts
_MAX_BOOST_MULTIPLIER = 4.0  # cap on coefficient scaling


class AdaptiveCoefficients:
    """Adjusts QUBO hyperparameters based on observed error patterns."""

    def __init__(self) -> None:
        self._logger = get_error_logger()

    def get_coefficients(self, has_clusters: bool = False, has_question: bool = False) -> dict:
        """Return coefficient dict adjusted for error history and context."""
        patterns = self._logger.get_error_patterns()
        coeffs = dict(_BASE)

        # Enable cluster terms when clusters are available
        if has_clusters:
            coeffs["delta"] = 1.0
            coeffs["epsilon"] = 1.0

        # Enable question relevance when question is available
        if has_question:
            coeffs["gamma"] = 0.5

        # Boost verification weights based on observed error frequency
        arithmetic_errors = patterns.get("arithmetic", 0)
        if arithmetic_errors >= _BOOST_THRESHOLD:
            boost = min(_MAX_BOOST_MULTIPLIER, 1.0 + arithmetic_errors * 0.1)
            coeffs["phi"] = round(coeffs["phi"] * boost, 2)
            coeffs["psi"] = round(coeffs["psi"] * boost, 2)

        return coeffs

    def get_error_summary(self) -> dict:
        """Return a summary for logging/debugging."""
        patterns = self._logger.get_error_patterns()
        total = sum(patterns.values())
        return {
            "total_errors_logged": total,
            "patterns": patterns,
            "coefficients_boosted": total >= _BOOST_THRESHOLD,
        }
