"""Trace logger for SpinChain MCP calls.

Logs every optimize_reasoning invocation to a JSONL file with:
- Timestamp, request ID
- Input parameters (completions count, solver config)
- Pipeline stages with timing (extraction, formulation, solve, ranking)
- Output summary (fragments selected, energies, fallback status)

The trace log enables:
- Usage monitoring: when/how often the MCP is called
- Performance profiling: which stage is the bottleneck
- Debugging: reproduce issues from logged parameters
- Traceability pipeline: full audit trail of reasoning optimization
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("spinchain.tracing")

# Default trace directory: ~/.spinchain/traces/
DEFAULT_TRACE_DIR = Path.home() / ".spinchain" / "traces"


@dataclass
class StageTrace:
    """Timing and metadata for a single pipeline stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    """Complete trace of one optimize_reasoning call."""

    trace_id: str
    timestamp: str
    input_params: dict[str, Any]
    stages: list[dict[str, Any]] = field(default_factory=list)
    output_summary: dict[str, Any] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    error: str | None = None


class TraceLogger:
    """Writes JSONL trace records for every MCP tool invocation.

    Usage:
        tracer = TraceLogger()
        trace = tracer.start_trace(params)
        with tracer.stage(trace, "extraction") as stage:
            # do work
            stage.metadata["num_raw_fragments"] = 42
        tracer.finish_trace(trace, output_summary)
    """

    def __init__(self, trace_dir: str | Path | None = None):
        self.trace_dir = Path(trace_dir) if trace_dir else DEFAULT_TRACE_DIR
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_file = self.trace_dir / "spinchain_traces.jsonl"
        self._active_traces: dict[str, TraceRecord] = {}

    def start_trace(self, input_params: dict[str, Any]) -> str:
        """Begin tracing a new optimize_reasoning call. Returns trace_id."""
        trace_id = uuid.uuid4().hex[:12]
        record = TraceRecord(
            trace_id=trace_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            input_params=input_params,
        )
        self._active_traces[trace_id] = record
        record._start = time.perf_counter()
        return trace_id

    def start_stage(self, trace_id: str, stage_name: str) -> StageTrace:
        """Begin timing a pipeline stage."""
        stage = StageTrace(name=stage_name, start_time=time.perf_counter())
        return stage

    def end_stage(self, trace_id: str, stage: StageTrace) -> None:
        """Finish timing a stage and attach it to the trace."""
        stage.end_time = time.perf_counter()
        stage.duration_ms = (stage.end_time - stage.start_time) * 1000
        record = self._active_traces.get(trace_id)
        if record:
            record.stages.append({
                "name": stage.name,
                "duration_ms": round(stage.duration_ms, 2),
                **stage.metadata,
            })

    def finish_trace(
        self,
        trace_id: str,
        output_summary: dict[str, Any],
        error: str | None = None,
    ) -> TraceRecord | None:
        """Complete the trace and write it to disk."""
        record = self._active_traces.pop(trace_id, None)
        if not record:
            logger.warning("Trace %s not found — skipping write", trace_id)
            return None

        record.total_duration_ms = round(
            (time.perf_counter() - record._start) * 1000, 2
        )
        record.output_summary = output_summary
        record.error = error

        self._write_record(record)
        return record

    def _write_record(self, record: TraceRecord) -> None:
        """Append a trace record as one JSON line."""
        entry = {
            "trace_id": record.trace_id,
            "timestamp": record.timestamp,
            "input_params": record.input_params,
            "stages": record.stages,
            "output_summary": record.output_summary,
            "total_duration_ms": record.total_duration_ms,
            "error": record.error,
        }
        try:
            with open(self._trace_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.error("Failed to write trace: %s", e)

    @property
    def trace_file(self) -> Path:
        """Path to the JSONL trace file."""
        return self._trace_file

    def read_traces(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Read trace records from disk. Useful for analysis/debugging."""
        if not self._trace_file.exists():
            return []
        traces = []
        with open(self._trace_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        if last_n is not None:
            traces = traces[-last_n:]
        return traces


# Module-level singleton — initialized lazily on first use
_tracer: TraceLogger | None = None


def get_tracer() -> TraceLogger:
    """Get or create the module-level TraceLogger singleton."""
    global _tracer
    if _tracer is None:
        trace_dir = os.environ.get("SPINCHAIN_TRACE_DIR")
        _tracer = TraceLogger(trace_dir=trace_dir)
    return _tracer
