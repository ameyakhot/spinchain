"""Tests for SpinChain trace logging system."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from spinchain.tracing import TraceLogger


@pytest.fixture
def tmp_tracer(tmp_path: Path) -> TraceLogger:
    """Create a TraceLogger writing to a temp directory."""
    return TraceLogger(trace_dir=tmp_path)


class TestTraceLifecycle:
    """Test start → stage → finish → write cycle."""

    def test_basic_trace_written(self, tmp_tracer: TraceLogger):
        """A completed trace should produce one JSONL line."""
        trace_id = tmp_tracer.start_trace({"num_completions": 3})
        tmp_tracer.finish_trace(trace_id, {"fallback": False})

        traces = tmp_tracer.read_traces()
        assert len(traces) == 1
        assert traces[0]["trace_id"] == trace_id
        assert traces[0]["input_params"]["num_completions"] == 3
        assert traces[0]["output_summary"]["fallback"] is False

    def test_trace_has_timestamp(self, tmp_tracer: TraceLogger):
        """Trace record should include an ISO timestamp."""
        trace_id = tmp_tracer.start_trace({})
        tmp_tracer.finish_trace(trace_id, {})

        traces = tmp_tracer.read_traces()
        assert "timestamp" in traces[0]
        # Should be parseable as a date string (YYYY-MM-DD...)
        assert traces[0]["timestamp"][:4].isdigit()

    def test_total_duration_tracked(self, tmp_tracer: TraceLogger):
        """Total duration should be > 0 for any trace."""
        trace_id = tmp_tracer.start_trace({})
        time.sleep(0.01)  # small delay to ensure measurable duration
        tmp_tracer.finish_trace(trace_id, {})

        traces = tmp_tracer.read_traces()
        assert traces[0]["total_duration_ms"] > 0

    def test_error_recorded(self, tmp_tracer: TraceLogger):
        """Errors should be captured in the trace."""
        trace_id = tmp_tracer.start_trace({})
        tmp_tracer.finish_trace(trace_id, {}, error="QUBO build failed")

        traces = tmp_tracer.read_traces()
        assert traces[0]["error"] == "QUBO build failed"

    def test_no_error_is_null(self, tmp_tracer: TraceLogger):
        """Successful traces should have null error."""
        trace_id = tmp_tracer.start_trace({})
        tmp_tracer.finish_trace(trace_id, {})

        traces = tmp_tracer.read_traces()
        assert traces[0]["error"] is None


class TestStageTracing:
    """Test per-stage timing and metadata."""

    def test_stage_duration(self, tmp_tracer: TraceLogger):
        """Stage duration should be recorded in ms."""
        trace_id = tmp_tracer.start_trace({})
        stage = tmp_tracer.start_stage(trace_id, "extraction")
        time.sleep(0.01)
        tmp_tracer.end_stage(trace_id, stage)
        tmp_tracer.finish_trace(trace_id, {})

        traces = tmp_tracer.read_traces()
        assert len(traces[0]["stages"]) == 1
        assert traces[0]["stages"][0]["name"] == "extraction"
        assert traces[0]["stages"][0]["duration_ms"] > 0

    def test_multiple_stages(self, tmp_tracer: TraceLogger):
        """Multiple stages should be recorded in order."""
        trace_id = tmp_tracer.start_trace({})
        for name in ["extraction", "formulation", "solve", "ranking"]:
            stage = tmp_tracer.start_stage(trace_id, name)
            tmp_tracer.end_stage(trace_id, stage)
        tmp_tracer.finish_trace(trace_id, {})

        traces = tmp_tracer.read_traces()
        stage_names = [s["name"] for s in traces[0]["stages"]]
        assert stage_names == ["extraction", "formulation", "solve", "ranking"]

    def test_stage_metadata(self, tmp_tracer: TraceLogger):
        """Custom metadata attached to a stage should be written."""
        trace_id = tmp_tracer.start_trace({})
        stage = tmp_tracer.start_stage(trace_id, "extraction")
        stage.metadata["num_raw_fragments"] = 42
        stage.metadata["num_merged_fragments"] = 15
        tmp_tracer.end_stage(trace_id, stage)
        tmp_tracer.finish_trace(trace_id, {})

        traces = tmp_tracer.read_traces()
        assert traces[0]["stages"][0]["num_raw_fragments"] == 42
        assert traces[0]["stages"][0]["num_merged_fragments"] == 15


class TestMultipleTraces:
    """Test accumulation and reading of multiple trace records."""

    def test_multiple_traces_appended(self, tmp_tracer: TraceLogger):
        """Each call should append a new line to the JSONL file."""
        for i in range(5):
            tid = tmp_tracer.start_trace({"call": i})
            tmp_tracer.finish_trace(tid, {})

        traces = tmp_tracer.read_traces()
        assert len(traces) == 5
        assert traces[0]["input_params"]["call"] == 0
        assert traces[4]["input_params"]["call"] == 4

    def test_read_last_n(self, tmp_tracer: TraceLogger):
        """read_traces(last_n=2) should return the last 2 entries."""
        for i in range(5):
            tid = tmp_tracer.start_trace({"call": i})
            tmp_tracer.finish_trace(tid, {})

        traces = tmp_tracer.read_traces(last_n=2)
        assert len(traces) == 2
        assert traces[0]["input_params"]["call"] == 3
        assert traces[1]["input_params"]["call"] == 4

    def test_read_empty_file(self, tmp_tracer: TraceLogger):
        """Reading before any traces are written should return empty list."""
        traces = tmp_tracer.read_traces()
        assert traces == []


class TestTraceFileFormat:
    """Verify the JSONL output format is correct and parseable."""

    def test_each_line_is_valid_json(self, tmp_tracer: TraceLogger):
        """Every line in the trace file should be valid JSON."""
        for i in range(3):
            tid = tmp_tracer.start_trace({"i": i})
            tmp_tracer.finish_trace(tid, {"ok": True})

        with open(tmp_tracer.trace_file) as f:
            for line in f:
                parsed = json.loads(line.strip())
                assert "trace_id" in parsed

    def test_trace_file_path(self, tmp_tracer: TraceLogger):
        """Trace file should be at <trace_dir>/spinchain_traces.jsonl."""
        assert tmp_tracer.trace_file.name == "spinchain_traces.jsonl"

    def test_unknown_trace_id_returns_none(self, tmp_tracer: TraceLogger):
        """Finishing a nonexistent trace returns None."""
        result = tmp_tracer.finish_trace("nonexistent", {})
        assert result is None
