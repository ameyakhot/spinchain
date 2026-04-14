"""Persistent error logger for detected reasoning errors.

Logs every verified-wrong fragment to ~/.spinchain/errors.jsonl.
Provides aggregated error patterns for adaptive coefficient adjustment.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ERROR_DIR = os.path.join(os.path.expanduser("~"), ".spinchain")
_ERROR_FILE = "spinchain_errors.jsonl"

_instance: ErrorLogger | None = None


class ErrorLogger:
    def __init__(self, error_dir: str | None = None):
        self._dir = Path(error_dir or os.environ.get("SPINCHAIN_TRACE_DIR", DEFAULT_ERROR_DIR))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _ERROR_FILE

    def log_error(
        self,
        trace_id: str,
        error_type: str,
        fragment_text: str,
        details: dict | None = None,
    ) -> None:
        record = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "fragment": fragment_text,
            **(details or {}),
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_errors(self, last_n: int | None = None) -> list[dict]:
        if not self._path.exists():
            return []
        records = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if last_n is not None:
            records = records[-last_n:]
        return records

    def get_error_patterns(self) -> dict[str, int]:
        """Return error type → count mapping from the full history."""
        errors = self.read_errors()
        counter = Counter(e.get("error_type", "unknown") for e in errors)
        return dict(counter)

    @property
    def path(self) -> Path:
        return self._path


def get_error_logger() -> ErrorLogger:
    global _instance
    if _instance is None:
        _instance = ErrorLogger()
    return _instance
