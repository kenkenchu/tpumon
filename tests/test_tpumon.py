"""Unit tests for tpumon -- no TPU hardware required."""

import time

import pytest

from tpumon import (
    BenchmarkStats,
    SysfsPoller,
    TpuMetrics,
    activity_percent,
    bar_gauge,
    sparkline,
)


# ---------------------------------------------------------------------------
# SysfsPoller parsing
# ---------------------------------------------------------------------------


class TestParseInterruptCounts:
    def test_basic(self):
        text = "0x00: 12345\n0x01: 678\n0x02: 0\n0x03: 99"
        result = SysfsPoller.parse_interrupt_counts(text)
        assert result == [12345, 678, 0, 99]

    def test_empty(self):
        assert SysfsPoller.parse_interrupt_counts("") == []

    def test_whitespace_only(self):
        assert SysfsPoller.parse_interrupt_counts("   \n  ") == []

    def test_full_13_vectors(self):
        lines = [f"0x{i:02x}: {i * 1000}" for i in range(13)]
        text = "\n".join(lines)
        result = SysfsPoller.parse_interrupt_counts(text)
        assert len(result) == 13
        assert result[0] == 0
        assert result[12] == 12000

    def test_large_counts(self):
        text = "0x00: 82373343\n0x01: 82373335"
        result = SysfsPoller.parse_interrupt_counts(text)
        assert result == [82373343, 82373335]


class TestParseIrqCount:
    def test_basic(self):
        text = (
            "           CPU0       CPU1       CPU2       CPU3\n"
            " 41:       1000       1200        800       1427   PCI-MSI  apex\n"
            " 42:          0          0          0          0   PCI-MSI  other\n"
        )
        assert SysfsPoller.parse_irq_count(text) == 1000 + 1200 + 800 + 1427

    def test_no_apex(self):
        text = (
            "           CPU0       CPU1\n"
            " 41:       1000       1200   PCI-MSI  something\n"
        )
        assert SysfsPoller.parse_irq_count(text) == 0

    def test_empty(self):
        assert SysfsPoller.parse_irq_count("") == 0

    def test_single_cpu(self):
        text = " 41:       5000   PCI-MSI  apex\n"
        assert SysfsPoller.parse_irq_count(text) == 5000


class TestParseAerTotal:
    def test_basic(self):
        text = (
            "RxErr 0\n"
            "BadTLP 0\n"
            "BadDLLP 0\n"
            "Rollover 0\n"
            "TOTAL_ERR_COR 0\n"
        )
        assert SysfsPoller.parse_aer_total(text) == 0

    def test_nonzero(self):
        text = (
            "RxErr 3\n"
            "BadTLP 2\n"
            "TOTAL_ERR_COR 5\n"
        )
        assert SysfsPoller.parse_aer_total(text) == 5

    def test_empty(self):
        assert SysfsPoller.parse_aer_total("") == 0


class TestReadHelpers:
    def test_read_missing_file(self):
        from pathlib import Path

        result = SysfsPoller._read_file(Path("/nonexistent/path/file"))
        assert result == ""

    def test_read_int_missing_file(self):
        from pathlib import Path

        result = SysfsPoller._read_int(Path("/nonexistent/path/file"))
        assert result == 0

    def test_read_int_default(self):
        from pathlib import Path

        result = SysfsPoller._read_int(Path("/nonexistent/path"), default=42)
        assert result == 42

    def test_read_file_default(self):
        from pathlib import Path

        result = SysfsPoller._read_file(
            Path("/nonexistent/path"), default="fallback"
        )
        assert result == "fallback"


# ---------------------------------------------------------------------------
# BenchmarkStats
# ---------------------------------------------------------------------------


class TestBenchmarkStats:
    def test_empty_stats(self):
        stats = BenchmarkStats()
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.avg_ms == 0.0
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0
        assert stats.throughput == 0.0

    def test_stats_with_data(self):
        stats = BenchmarkStats()
        stats.start_time = time.monotonic()
        # Record known values: 1.0 through 10.0
        for v in [float(i) for i in range(1, 11)]:
            stats.record(v)

        assert stats.min_ms == 1.0
        assert stats.max_ms == 10.0
        assert stats.avg_ms == pytest.approx(5.5)
        assert stats.p50_ms == pytest.approx(5.5, abs=0.5)
        assert stats.total_inferences == 10

    def test_p95_p99(self):
        stats = BenchmarkStats()
        # 100 values from 1..100
        for i in range(1, 101):
            stats.record(float(i))

        assert stats.p50_ms == pytest.approx(50.5, abs=1.0)
        assert stats.p95_ms == pytest.approx(95.05, abs=1.0)
        assert stats.p99_ms == pytest.approx(99.01, abs=1.0)

    def test_throughput(self):
        stats = BenchmarkStats()
        stats.start_time = time.monotonic() - 2.0  # 2 seconds ago
        for _ in range(100):
            stats.record(1.0)

        assert stats.throughput == pytest.approx(50.0, rel=0.1)

    def test_deque_maxlen(self):
        stats = BenchmarkStats()
        for i in range(1500):
            stats.record(float(i))

        assert len(stats.latencies_ms) == BenchmarkStats.MAXLEN
        assert stats.total_inferences == 1500
        # Oldest values dropped -- deque should start around 500
        assert stats.latencies_ms[0] == 500.0

    def test_reset(self):
        stats = BenchmarkStats()
        for i in range(10):
            stats.record(float(i))
        stats.reset()
        assert stats.total_inferences == 0
        assert len(stats.latencies_ms) == 0
        assert stats.start_time > 0


# ---------------------------------------------------------------------------
# TpuMetrics
# ---------------------------------------------------------------------------


class TestTpuMetrics:
    def test_temp_conversion(self):
        m = TpuMetrics(tpu_temp_millic=46100, cpu_temp_millic=49050)
        assert m.tpu_temp_c == pytest.approx(46.1)
        assert m.cpu_temp_c == pytest.approx(49.05)

    def test_temp_zero(self):
        m = TpuMetrics()
        assert m.tpu_temp_c == 0.0
        assert m.cpu_temp_c == 0.0

    def test_total_interrupts(self):
        m = TpuMetrics(interrupt_vectors=[100, 200, 0, 50])
        assert m.total_interrupts == 350

    def test_total_interrupts_empty(self):
        m = TpuMetrics()
        assert m.total_interrupts == 0


# ---------------------------------------------------------------------------
# Activity gauge logic
# ---------------------------------------------------------------------------


class TestActivityGauge:
    def test_activity_idle(self):
        assert activity_percent(0, 10000) == 0.0

    def test_activity_active(self):
        pct = activity_percent(5000, 10000)
        assert pct == pytest.approx(50.0)

    def test_activity_saturated(self):
        pct = activity_percent(15000, 10000)
        assert pct == 100.0  # Clamped

    def test_activity_zero_max(self):
        assert activity_percent(100, 0) == 0.0

    def test_activity_small(self):
        pct = activity_percent(100, 10000)
        assert pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_constant(self):
        result = sparkline([5.0, 5.0, 5.0])
        assert len(result) == 3
        # All same value -- all should be the same character
        assert len(set(result)) == 1

    def test_ascending(self):
        result = sparkline([1.0, 2.0, 3.0, 4.0])
        assert len(result) == 4
        # First char should be lowest, last should be highest
        assert result[0] == "▁"
        assert result[-1] == "█"

    def test_width_limit(self):
        values = list(range(100))
        result = sparkline(values, width=20)
        assert len(result) == 20


class TestBarGauge:
    def test_full(self):
        result = bar_gauge(10.0, 10.0, 5)
        assert result == "█████"

    def test_empty(self):
        result = bar_gauge(0.0, 10.0, 5)
        assert result == "░░░░░"

    def test_half(self):
        result = bar_gauge(5.0, 10.0, 10)
        assert result == "█████░░░░░"

    def test_zero_max(self):
        result = bar_gauge(5.0, 0.0, 5)
        assert result == "░░░░░"

    def test_over_max(self):
        result = bar_gauge(20.0, 10.0, 5)
        assert result == "█████"  # Clamped to full
