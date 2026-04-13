"""Tests for SpinChain trace analysis pipeline."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path


from spinchain.analyze import TraceAnalyzer, _percentile


def _make_trace(
    trace_id: str = "abc123",
    timestamp: str = "2026-04-12T10:00:00",
    num_completions: int = 5,
    stages: list | None = None,
    fallback: bool = False,
    error: str | None = None,
    min_energy: float | None = -3.5,
    num_selected: int = 4,
    total_duration_ms: float = 2000.0,
) -> dict:
    """Build a synthetic trace record."""
    if stages is None:
        stages = [
            {"name": "fragment_extraction", "duration_ms": 500.0, "num_merged_fragments": 10},
            {"name": "qubo_formulation", "duration_ms": 50.0, "num_linear_terms": 10},
            {"name": "simulated_annealing", "duration_ms": 1200.0, "num_samples": 100},
            {"name": "stability_ranking", "duration_ms": 5.0, "num_selected": num_selected},
        ]
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "input_params": {"num_completions": num_completions},
        "stages": stages if not fallback and not error else [],
        "output_summary": {
            "fallback": fallback,
            "min_energy": min_energy if not fallback else None,
            "num_selected": num_selected if not fallback else None,
            "num_fragments": 10 if not fallback else num_completions,
        },
        "total_duration_ms": total_duration_ms,
        "error": error,
    }


class TestPercentile:
    def test_median_odd(self):
        assert _percentile([1, 2, 3], 50) == 2.0

    def test_median_even(self):
        assert _percentile([1, 2, 3, 4], 50) == 2.5

    def test_p0(self):
        assert _percentile([10, 20, 30], 0) == 10.0

    def test_p100(self):
        assert _percentile([10, 20, 30], 100) == 30.0

    def test_empty(self):
        assert _percentile([], 50) == 0.0


class TestUsageSummary:
    def test_empty(self):
        analyzer = TraceAnalyzer([])
        s = analyzer.usage_summary()
        assert s.total_calls == 0
        assert s.first_call is None

    def test_counts(self):
        traces = [
            _make_trace(trace_id="1"),
            _make_trace(trace_id="2"),
            _make_trace(trace_id="3", fallback=True, min_energy=None),
            _make_trace(trace_id="4", error="boom"),
        ]
        analyzer = TraceAnalyzer(traces)
        s = analyzer.usage_summary()
        assert s.total_calls == 4
        assert s.success_count == 2
        assert s.fallback_count == 1
        assert s.error_count == 1

    def test_time_range(self):
        traces = [
            _make_trace(timestamp="2026-04-12T08:00:00"),
            _make_trace(timestamp="2026-04-12T12:00:00"),
        ]
        analyzer = TraceAnalyzer(traces)
        s = analyzer.usage_summary()
        assert s.first_call == "2026-04-12T08:00:00"
        assert s.last_call == "2026-04-12T12:00:00"


class TestLatencyBreakdown:
    def test_stage_names(self):
        analyzer = TraceAnalyzer([_make_trace()])
        latency = analyzer.latency_breakdown()
        names = [s.stage for s in latency]
        assert "fragment_extraction" in names
        assert "simulated_annealing" in names
        assert "total" in names

    def test_single_trace_stats(self):
        analyzer = TraceAnalyzer([_make_trace()])
        latency = {s.stage: s for s in analyzer.latency_breakdown()}

        sa = latency["simulated_annealing"]
        assert sa.count == 1
        assert sa.min_ms == 1200.0
        assert sa.max_ms == 1200.0
        assert sa.mean_ms == 1200.0

    def test_multiple_traces_aggregated(self):
        traces = [
            _make_trace(stages=[{"name": "simulated_annealing", "duration_ms": 1000.0}], total_duration_ms=1000),
            _make_trace(stages=[{"name": "simulated_annealing", "duration_ms": 2000.0}], total_duration_ms=2000),
            _make_trace(stages=[{"name": "simulated_annealing", "duration_ms": 3000.0}], total_duration_ms=3000),
        ]
        analyzer = TraceAnalyzer(traces)
        latency = {s.stage: s for s in analyzer.latency_breakdown()}

        sa = latency["simulated_annealing"]
        assert sa.count == 3
        assert sa.min_ms == 1000.0
        assert sa.max_ms == 3000.0
        assert sa.mean_ms == 2000.0

    def test_no_stages_gives_empty(self):
        traces = [_make_trace(fallback=True, min_energy=None)]
        analyzer = TraceAnalyzer(traces)
        latency = analyzer.latency_breakdown()
        # Only "total" should exist (no pipeline stages on fallback)
        names = [s.stage for s in latency]
        assert "fragment_extraction" not in names


class TestEnergyStats:
    def test_single_trace(self):
        analyzer = TraceAnalyzer([_make_trace(min_energy=-5.0)])
        e = analyzer.energy_stats()
        assert e.count == 1
        assert e.min_energy == -5.0
        assert e.mean_energy == -5.0

    def test_multiple_traces(self):
        traces = [
            _make_trace(min_energy=-2.0),
            _make_trace(min_energy=-4.0),
            _make_trace(min_energy=-6.0),
        ]
        analyzer = TraceAnalyzer(traces)
        e = analyzer.energy_stats()
        assert e.count == 3
        assert e.min_energy == -6.0
        assert e.mean_energy == -4.0
        assert e.max_energy == -2.0

    def test_no_energy_data(self):
        traces = [_make_trace(fallback=True, min_energy=None)]
        analyzer = TraceAnalyzer(traces)
        e = analyzer.energy_stats()
        assert e.count == 0
        assert e.min_energy is None


class TestAnomalyDetection:
    def test_error_flagged(self):
        traces = [_make_trace(error="QUBO failed")]
        anomalies = TraceAnalyzer(traces).detect_anomalies()
        assert len(anomalies) == 1
        assert anomalies[0].reason == "error"

    def test_slow_total_flagged(self):
        traces = [_make_trace(total_duration_ms=15000.0)]
        anomalies = TraceAnalyzer(traces).detect_anomalies(slow_threshold_ms=10000)
        reasons = [a.reason for a in anomalies]
        assert "slow_total" in reasons

    def test_slow_stage_flagged(self):
        traces = [_make_trace(stages=[
            {"name": "simulated_annealing", "duration_ms": 8000.0},
        ])]
        anomalies = TraceAnalyzer(traces).detect_anomalies(slow_stage_threshold_ms=5000)
        assert any(a.reason == "slow_stage" for a in anomalies)

    def test_empty_selection_flagged(self):
        traces = [_make_trace(num_selected=0)]
        # Need to fix output_summary to match
        traces[0]["output_summary"]["num_selected"] = 0
        traces[0]["output_summary"]["fallback"] = False
        anomalies = TraceAnalyzer(traces).detect_anomalies()
        assert any(a.reason == "empty_selection" for a in anomalies)

    def test_normal_trace_no_anomalies(self):
        traces = [_make_trace()]
        anomalies = TraceAnalyzer(traces).detect_anomalies()
        assert len(anomalies) == 0


class TestFullReport:
    def test_report_structure(self):
        traces = [_make_trace(), _make_trace(trace_id="def456")]
        report = TraceAnalyzer(traces).full_report()

        assert "usage" in report
        assert "latency" in report
        assert "energy" in report
        assert "anomalies" in report
        assert report["usage"]["total_calls"] == 2

    def test_json_serializable(self):
        traces = [_make_trace()]
        report = TraceAnalyzer(traces).full_report()
        # Should not raise
        json.dumps(report)

    def test_print_report_runs(self):
        traces = [_make_trace(), _make_trace(error="oops")]
        analyzer = TraceAnalyzer(traces)
        buf = StringIO()
        analyzer.print_report(file=buf)
        output = buf.getvalue()
        assert "SpinChain Trace Analysis" in output
        assert "Calls: 2" in output


class TestFromFile:
    def test_load_from_jsonl(self, tmp_path: Path):
        """Write a JSONL file, then load it via from_file."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        trace_file = trace_dir / "spinchain_traces.jsonl"

        traces = [_make_trace(trace_id="t1"), _make_trace(trace_id="t2")]
        with open(trace_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

        analyzer = TraceAnalyzer.from_file(trace_dir=trace_dir)
        assert analyzer.usage_summary().total_calls == 2

    def test_load_last_n(self, tmp_path: Path):
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        trace_file = trace_dir / "spinchain_traces.jsonl"

        with open(trace_file, "w") as f:
            for i in range(10):
                f.write(json.dumps(_make_trace(trace_id=f"t{i}")) + "\n")

        analyzer = TraceAnalyzer.from_file(trace_dir=trace_dir, last_n=3)
        assert analyzer.usage_summary().total_calls == 3
