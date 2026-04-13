"""Trace analysis pipeline for SpinChain MCP usage.

Reads JSONL traces and produces:
- Usage summary: call count, time range, success/fallback rates
- Latency breakdown: per-stage timing percentiles
- Energy distribution: min/mean/spread across calls
- Anomaly flags: unusually slow stages, empty selections, errors

Run as: uv run python -m spinchain.analyze [--last N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinchain.tracing import TraceLogger


@dataclass
class UsageSummary:
    """High-level usage statistics."""

    total_calls: int = 0
    first_call: str | None = None
    last_call: str | None = None
    success_count: int = 0
    fallback_count: int = 0
    error_count: int = 0


@dataclass
class LatencyStats:
    """Latency percentiles for a pipeline stage (in ms)."""

    stage: str
    count: int = 0
    min_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0


@dataclass
class EnergyStats:
    """Energy distribution across SA solutions."""

    count: int = 0
    min_energy: float | None = None
    mean_energy: float | None = None
    max_energy: float | None = None


@dataclass
class Anomaly:
    """A flagged trace with reason."""

    trace_id: str
    timestamp: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def _percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) from sorted values."""
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(values):
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


class TraceAnalyzer:
    """Analyzes SpinChain trace records from JSONL files."""

    def __init__(self, traces: list[dict[str, Any]]):
        self.traces = traces

    @classmethod
    def from_file(cls, trace_dir: str | Path | None = None, last_n: int | None = None) -> "TraceAnalyzer":
        """Load traces from the JSONL file."""
        tracer = TraceLogger(trace_dir=trace_dir)
        traces = tracer.read_traces(last_n=last_n)
        return cls(traces)

    def usage_summary(self) -> UsageSummary:
        """Compute usage summary stats."""
        summary = UsageSummary(total_calls=len(self.traces))
        if not self.traces:
            return summary

        summary.first_call = self.traces[0].get("timestamp")
        summary.last_call = self.traces[-1].get("timestamp")

        for t in self.traces:
            if t.get("error"):
                summary.error_count += 1
            elif t.get("output_summary", {}).get("fallback"):
                summary.fallback_count += 1
            else:
                summary.success_count += 1

        return summary

    def latency_breakdown(self) -> list[LatencyStats]:
        """Compute per-stage latency percentiles."""
        stage_times: dict[str, list[float]] = {}

        for t in self.traces:
            for stage in t.get("stages", []):
                name = stage.get("name", "unknown")
                dur = stage.get("duration_ms", 0.0)
                stage_times.setdefault(name, []).append(dur)

        # Also compute total
        total_times = [
            t.get("total_duration_ms", 0.0)
            for t in self.traces
            if t.get("total_duration_ms")
        ]
        if total_times:
            stage_times["total"] = total_times

        results = []
        for stage_name in ["fragment_extraction", "qubo_formulation", "simulated_annealing", "stability_ranking", "total"]:
            times = sorted(stage_times.get(stage_name, []))
            if not times:
                continue
            results.append(LatencyStats(
                stage=stage_name,
                count=len(times),
                min_ms=round(times[0], 2),
                median_ms=round(_percentile(times, 50), 2),
                p95_ms=round(_percentile(times, 95), 2),
                max_ms=round(times[-1], 2),
                mean_ms=round(sum(times) / len(times), 2),
            ))

        return results

    def energy_stats(self) -> EnergyStats:
        """Compute energy distribution across successful calls."""
        min_energies = []
        for t in self.traces:
            out = t.get("output_summary", {})
            me = out.get("min_energy")
            if me is not None:
                min_energies.append(me)

        if not min_energies:
            return EnergyStats()

        return EnergyStats(
            count=len(min_energies),
            min_energy=round(min(min_energies), 4),
            mean_energy=round(sum(min_energies) / len(min_energies), 4),
            max_energy=round(max(min_energies), 4),
        )

    def detect_anomalies(
        self,
        slow_threshold_ms: float = 10000.0,
        slow_stage_threshold_ms: float = 5000.0,
    ) -> list[Anomaly]:
        """Flag traces with potential issues."""
        anomalies = []

        for t in self.traces:
            tid = t.get("trace_id", "?")
            ts = t.get("timestamp", "?")

            # Error
            if t.get("error"):
                anomalies.append(Anomaly(
                    trace_id=tid, timestamp=ts,
                    reason="error",
                    details={"error": t["error"]},
                ))

            # Slow total
            total_ms = t.get("total_duration_ms", 0.0)
            if total_ms > slow_threshold_ms:
                anomalies.append(Anomaly(
                    trace_id=tid, timestamp=ts,
                    reason="slow_total",
                    details={"total_duration_ms": total_ms},
                ))

            # Slow individual stage
            for stage in t.get("stages", []):
                if stage.get("duration_ms", 0) > slow_stage_threshold_ms:
                    anomalies.append(Anomaly(
                        trace_id=tid, timestamp=ts,
                        reason="slow_stage",
                        details={
                            "stage": stage["name"],
                            "duration_ms": stage["duration_ms"],
                        },
                    ))

            # Empty selection (non-fallback call selected 0 fragments)
            out = t.get("output_summary", {})
            if not out.get("fallback") and out.get("num_selected", 1) == 0:
                anomalies.append(Anomaly(
                    trace_id=tid, timestamp=ts,
                    reason="empty_selection",
                    details={"num_fragments": out.get("num_fragments")},
                ))

        return anomalies

    def full_report(self) -> dict[str, Any]:
        """Generate complete analysis report as a dict."""
        usage = self.usage_summary()
        latency = self.latency_breakdown()
        energy = self.energy_stats()
        anomalies = self.detect_anomalies()

        return {
            "usage": {
                "total_calls": usage.total_calls,
                "first_call": usage.first_call,
                "last_call": usage.last_call,
                "success": usage.success_count,
                "fallback": usage.fallback_count,
                "errors": usage.error_count,
            },
            "latency": [
                {
                    "stage": s.stage,
                    "count": s.count,
                    "min_ms": s.min_ms,
                    "median_ms": s.median_ms,
                    "p95_ms": s.p95_ms,
                    "max_ms": s.max_ms,
                    "mean_ms": s.mean_ms,
                }
                for s in latency
            ],
            "energy": {
                "count": energy.count,
                "min": energy.min_energy,
                "mean": energy.mean_energy,
                "max": energy.max_energy,
            },
            "anomalies": [
                {
                    "trace_id": a.trace_id,
                    "timestamp": a.timestamp,
                    "reason": a.reason,
                    **a.details,
                }
                for a in anomalies
            ],
        }

    def print_report(self, file=sys.stdout) -> None:
        """Print a human-readable report."""
        usage = self.usage_summary()

        print("=" * 60, file=file)
        print("  SpinChain Trace Analysis", file=file)
        print("=" * 60, file=file)

        print(f"\n  Calls: {usage.total_calls}  "
              f"(success: {usage.success_count}, "
              f"fallback: {usage.fallback_count}, "
              f"errors: {usage.error_count})", file=file)
        if usage.first_call:
            print(f"  Range: {usage.first_call} -> {usage.last_call}", file=file)

        latency = self.latency_breakdown()
        if latency:
            print(f"\n  {'Stage':<25} {'Count':>5} {'Min':>8} {'Median':>8} {'P95':>8} {'Max':>8} {'Mean':>8}", file=file)
            print(f"  {'-'*23}  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", file=file)
            for s in latency:
                print(f"  {s.stage:<25} {s.count:>5} {s.min_ms:>7.1f}ms {s.median_ms:>7.1f}ms "
                      f"{s.p95_ms:>7.1f}ms {s.max_ms:>7.1f}ms {s.mean_ms:>7.1f}ms", file=file)

        energy = self.energy_stats()
        if energy.count:
            print(f"\n  Energy: min={energy.min_energy}, mean={energy.mean_energy}, "
                  f"max={energy.max_energy} (n={energy.count})", file=file)

        anomalies = self.detect_anomalies()
        if anomalies:
            print(f"\n  Anomalies ({len(anomalies)}):", file=file)
            for a in anomalies:
                print(f"    [{a.reason}] {a.trace_id} @ {a.timestamp} — {a.details}", file=file)
        else:
            print("\n  No anomalies detected.", file=file)

        print("", file=file)


def main():
    """CLI entry point for trace analysis."""
    import os

    parser = argparse.ArgumentParser(description="Analyze SpinChain MCP traces")
    parser.add_argument("--trace-dir", default=os.environ.get("SPINCHAIN_TRACE_DIR"),
                        help="Path to trace directory (default: SPINCHAIN_TRACE_DIR or ~/.spinchain/traces)")
    parser.add_argument("--last", type=int, default=None,
                        help="Only analyze last N traces")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable")
    args = parser.parse_args()

    analyzer = TraceAnalyzer.from_file(trace_dir=args.trace_dir, last_n=args.last)

    if args.json:
        print(json.dumps(analyzer.full_report(), indent=2))
    else:
        analyzer.print_report()


if __name__ == "__main__":
    main()
