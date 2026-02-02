"""tpumon -- Coral Edge TPU Monitor TUI.

A live terminal dashboard for the Coral Edge TPU, similar to btop/nvtop.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.worker import get_current_worker
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header, Static

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APEX_SYSFS = Path("/sys/class/apex/apex_0")
PCI_SYSFS = Path("/sys/bus/pci/devices/0001:06:00.0")
THERMAL_ZONE = Path("/sys/devices/virtual/thermal/thermal_zone0/temp")
COOLING_DEVICE = Path("/sys/class/thermal/cooling_device0/cur_state")
APEX_MODULE = Path("/sys/module/apex/parameters")
PROC_INTERRUPTS = Path("/proc/interrupts")
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = MODELS_DIR / "regime_model_quant_edgetpu.tflite"

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
ACTIVITY_MAX_FLOOR = 50.0  # Minimum scale for activity gauge auto-calibration

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TpuMetrics:
    """Snapshot of all sysfs readings."""

    status: str = "UNKNOWN"
    tpu_temp_millic: int = 0
    cpu_temp_millic: int = 0
    fan_state: int = 0
    fan_max: int = 4
    trip_points: list[float] = field(default_factory=list)
    interrupt_vectors: list[int] = field(default_factory=list)
    irq_count: int = 0
    reset_count: int = 0
    device_owner: int = 0
    mapped_pages: int = 0
    page_table_entries: int = 0
    driver_version: str = ""
    framework_version: str = ""
    pci_device_name: str = ""
    pcie_link_speed: str = ""
    pcie_link_width: str = ""
    power_state: str = ""
    aer_correctable: int = 0
    aer_nonfatal: int = 0
    aer_fatal: int = 0
    hw_clock_gating: bool = False
    sw_clock_gating: bool = False
    power_save: bool = False
    # Hardware temperature warnings
    hw_temp_warn1: int = 0
    hw_temp_warn1_en: bool = False
    hw_temp_warn2: int = 0
    hw_temp_warn2_en: bool = False
    # Detailed AER error breakdown (non-zero types only)
    aer_correctable_detail: dict[str, int] = field(default_factory=dict)
    aer_nonfatal_detail: dict[str, int] = field(default_factory=dict)
    aer_fatal_detail: dict[str, int] = field(default_factory=dict)
    # PCIe max link capabilities
    pcie_max_link_speed: str = ""
    pcie_max_link_width: str = ""
    # Per-CPU interrupt counts and polling CPU
    irq_per_cpu: list[int] = field(default_factory=list)
    irq_polling_cpu: int = -1
    # Device ownership detail
    write_open_count: int = 0
    is_device_owned: bool = False
    # Driver temp polling interval (ms)
    temp_poll_interval: int = 0
    timestamp: float = 0.0

    @property
    def tpu_temp_c(self) -> float:
        return self.tpu_temp_millic / 1000.0

    @property
    def cpu_temp_c(self) -> float:
        return self.cpu_temp_millic / 1000.0

    @property
    def total_interrupts(self) -> int:
        return sum(self.interrupt_vectors)


@dataclass
class BenchmarkStats:
    """Accumulates inference latency measurements."""

    MAXLEN: ClassVar[int] = 1000

    latencies_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=BenchmarkStats.MAXLEN)
    )
    total_inferences: int = 0
    start_time: float = 0.0
    model_name: str = ""
    model_size_kb: float = 0.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def throughput(self) -> float:
        elapsed = time.monotonic() - self.start_time if self.start_time else 0.0
        return self.total_inferences / elapsed if elapsed > 0 else 0.0

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.total_inferences += 1

    def reset(self) -> None:
        self.latencies_ms.clear()
        self.total_inferences = 0
        self.start_time = time.monotonic()

    def _percentile(self, p: int) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        k = (p / 100.0) * (len(sorted_lat) - 1)
        f = int(k)
        c = f + 1
        if c >= len(sorted_lat):
            return sorted_lat[f]
        return sorted_lat[f] + (k - f) * (sorted_lat[c] - sorted_lat[f])


# ---------------------------------------------------------------------------
# Sysfs poller
# ---------------------------------------------------------------------------


class SysfsPoller:
    """Reads all sysfs files and returns a TpuMetrics snapshot."""

    @staticmethod
    def _read_file(path: Path, default: str = "") -> str:
        try:
            return path.read_text().strip()
        except OSError:
            return default

    @staticmethod
    def _read_int(path: Path, default: int = 0) -> int:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return default

    @staticmethod
    def parse_interrupt_counts(text: str) -> list[int]:
        """Parse interrupt_counts sysfs file.

        Format: "0x00: 12345\\n0x01: 678\\n..."
        """
        if not text.strip():
            return []
        counts = []
        for line in text.strip().splitlines():
            match = re.match(r"0x[0-9a-fA-F]+:\s*(\d+)", line.strip())
            if match:
                counts.append(int(match.group(1)))
        return counts

    @staticmethod
    def parse_irq_count(text: str) -> int:
        """Parse /proc/interrupts for apex IRQ total.

        Sums all per-CPU columns for the apex line.
        """
        for line in text.splitlines():
            if "apex" not in line:
                continue
            parts = line.split()
            total = 0
            for part in parts[1:]:
                try:
                    total += int(part)
                except ValueError:
                    break
            return total
        return 0

    @staticmethod
    def parse_aer_total(text: str) -> int:
        """Parse AER error sysfs file for TOTAL line."""
        for line in text.splitlines():
            if "TOTAL" in line.upper():
                match = re.search(r"(\d+)", line)
                if match:
                    return int(match.group(1))
        return 0

    @staticmethod
    def parse_aer_detail(text: str) -> dict[str, int]:
        """Parse AER error sysfs file into {error_type: count} for non-zero entries."""
        detail: dict[str, int] = {}
        for line in text.strip().splitlines():
            match = re.match(r"(\S+)\s+(\d+)", line.strip())
            if match:
                name, count = match.group(1), int(match.group(2))
                if count > 0 and "TOTAL" not in name.upper():
                    detail[name] = count
        return detail

    @staticmethod
    def parse_irq_per_cpu(text: str) -> tuple[list[int], int]:
        """Parse /proc/interrupts for per-CPU apex IRQ counts.

        Returns (per_cpu_counts, polling_cpu_index).
        The polling CPU is the one handling all hrtimer-polled interrupts.
        """
        for line in text.splitlines():
            if "apex" not in line:
                continue
            parts = line.split()
            counts: list[int] = []
            for part in parts[1:]:
                try:
                    counts.append(int(part))
                except ValueError:
                    break
            # Identify the polling CPU (the one with all the counts)
            polling_cpu = -1
            if counts:
                max_count = max(counts)
                if max_count > 0:
                    polling_cpu = counts.index(max_count)
            return counts, polling_cpu
        return [], -1

    # PCI vendor:device -> human-readable name
    _PCI_NAMES: ClassVar[dict[tuple[str, str], str]] = {
        ("0x1ac1", "0x089a"): "Global Unichip Corp. Coral Edge TPU",
    }

    def resolve_pci_device_name(self, pci_path: Path) -> str:
        """Resolve PCI device name from vendor/device IDs."""
        vendor = self._read_file(pci_path / "vendor")
        device = self._read_file(pci_path / "device")
        if not vendor or not device:
            return ""
        name = self._PCI_NAMES.get((vendor, device))
        if name:
            return name
        return f"PCI {vendor}:{device}"

    def poll(self) -> TpuMetrics:
        """Read all sysfs files and return a metrics snapshot."""
        trip_points = []
        for i in range(3):
            tp = self._read_int(APEX_SYSFS / f"trip_point{i}_temp", -1)
            if tp >= 0:
                trip_points.append(tp / 1000.0)

        int_text = self._read_file(APEX_SYSFS / "interrupt_counts")
        proc_int_text = self._read_file(PROC_INTERRUPTS)
        aer_cor_text = self._read_file(PCI_SYSFS / "aer_dev_correctable")
        aer_nf_text = self._read_file(PCI_SYSFS / "aer_dev_nonfatal")
        aer_fat_text = self._read_file(PCI_SYSFS / "aer_dev_fatal")
        irq_per_cpu, irq_polling_cpu = self.parse_irq_per_cpu(proc_int_text)

        return TpuMetrics(
            status=self._read_file(APEX_SYSFS / "status", "UNKNOWN"),
            tpu_temp_millic=self._read_int(APEX_SYSFS / "temp"),
            cpu_temp_millic=self._read_int(THERMAL_ZONE),
            fan_state=self._read_int(COOLING_DEVICE),
            trip_points=trip_points,
            interrupt_vectors=self.parse_interrupt_counts(int_text),
            irq_count=self.parse_irq_count(proc_int_text),
            reset_count=self._read_int(APEX_SYSFS / "reset_count"),
            device_owner=self._read_int(APEX_SYSFS / "device_owner"),
            mapped_pages=self._read_int(
                APEX_SYSFS / "node_0_num_mapped_pages"
            ),
            page_table_entries=self._read_int(
                APEX_SYSFS / "node_0_page_table_entries"
            ),
            driver_version=self._read_file(APEX_SYSFS / "driver_version"),
            framework_version=self._read_file(
                APEX_SYSFS / "framework_version"
            ),
            pci_device_name=self.resolve_pci_device_name(PCI_SYSFS),
            pcie_link_speed=self._read_file(
                PCI_SYSFS / "current_link_speed"
            ),
            pcie_link_width=self._read_file(
                PCI_SYSFS / "current_link_width"
            ),
            power_state=self._read_file(PCI_SYSFS / "power_state"),
            aer_correctable=self.parse_aer_total(aer_cor_text),
            aer_nonfatal=self.parse_aer_total(aer_nf_text),
            aer_fatal=self.parse_aer_total(aer_fat_text),
            hw_clock_gating=self._read_file(
                APEX_MODULE / "allow_hw_clock_gating"
            )
            == "1",
            sw_clock_gating=self._read_file(
                APEX_MODULE / "allow_sw_clock_gating"
            )
            == "1",
            power_save=self._read_file(APEX_MODULE / "allow_power_save")
            == "1",
            hw_temp_warn1=self._read_int(APEX_SYSFS / "hw_temp_warn1"),
            hw_temp_warn1_en=self._read_file(
                APEX_SYSFS / "hw_temp_warn1_en"
            )
            == "1",
            hw_temp_warn2=self._read_int(APEX_SYSFS / "hw_temp_warn2"),
            hw_temp_warn2_en=self._read_file(
                APEX_SYSFS / "hw_temp_warn2_en"
            )
            == "1",
            aer_correctable_detail=self.parse_aer_detail(aer_cor_text),
            aer_nonfatal_detail=self.parse_aer_detail(aer_nf_text),
            aer_fatal_detail=self.parse_aer_detail(aer_fat_text),
            pcie_max_link_speed=self._read_file(
                PCI_SYSFS / "max_link_speed"
            ),
            pcie_max_link_width=self._read_file(
                PCI_SYSFS / "max_link_width"
            ),
            irq_per_cpu=irq_per_cpu,
            irq_polling_cpu=irq_polling_cpu,
            write_open_count=self._read_int(
                APEX_SYSFS / "write_open_count"
            ),
            is_device_owned=self._read_file(
                APEX_SYSFS / "is_device_owned"
            )
            == "1",
            temp_poll_interval=self._read_int(
                APEX_SYSFS / "temp_poll_interval"
            ),
            timestamp=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def sparkline(values: list[float], width: int = 40) -> str:
    """Render a sparkline string from a list of values."""
    if not values:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0
    return "".join(
        SPARKLINE_CHARS[min(int((v - lo) / span * (len(SPARKLINE_CHARS) - 1)), len(SPARKLINE_CHARS) - 1)]
        for v in recent
    )


def bar_gauge(value: float, maximum: float, width: int = 10) -> str:
    """Render a block-character bar gauge."""
    if maximum <= 0:
        return "░" * width
    ratio = max(0.0, min(1.0, value / maximum))
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def temp_color(temp_c: float) -> str:
    """Return a rich color name based on temperature."""
    if temp_c < 50:
        return "green"
    if temp_c < 70:
        return "yellow"
    return "red"


def activity_percent(delta_per_sec: float, max_rate: float) -> float:
    """Convert interrupt rate to 0-100 activity percentage."""
    if max_rate <= 0:
        return 0.0
    return max(0.0, min(100.0, (delta_per_sec / max_rate) * 100.0))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class TpuStatusPanel(Widget):
    """TPU status, temperature, fan, and activity."""

    DEFAULT_CSS = """
    TpuStatusPanel {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="tpu-status-content")

    def update_content(
        self,
        metrics: TpuMetrics,
        activity_pct: float,
    ) -> None:
        status_color = "green" if metrics.status == "ALIVE" else "red"
        tpu_tc = temp_color(metrics.tpu_temp_c)
        cpu_tc = temp_color(metrics.cpu_temp_c)

        activity_bar = bar_gauge(activity_pct, 100.0, 8)
        if activity_pct < 5:
            act_color = "dim"
        elif activity_pct < 80:
            act_color = "green"
        else:
            act_color = "yellow"

        tpu_bar = bar_gauge(metrics.tpu_temp_c, 95.0, 6)
        cpu_bar = bar_gauge(metrics.cpu_temp_c, 95.0, 6)
        fan_bar = bar_gauge(metrics.fan_state, metrics.fan_max, 4)

        trips = "/".join(f"{t:.1f}" for t in metrics.trip_points) if metrics.trip_points else "n/a"

        # Hardware temperature warnings
        warns = []
        if metrics.hw_temp_warn1 > 0:
            w1_c = metrics.hw_temp_warn1 / 1000.0
            en1 = "[green]on[/]" if metrics.hw_temp_warn1_en else "[dim]off[/]"
            warns.append(f"W1={w1_c:.0f}°C({en1})")
        if metrics.hw_temp_warn2 > 0:
            w2_c = metrics.hw_temp_warn2 / 1000.0
            en2 = "[green]on[/]" if metrics.hw_temp_warn2_en else "[dim]off[/]"
            warns.append(f"W2={w2_c:.0f}°C({en2})")
        warn_str = " ".join(warns) if warns else "n/a"

        gates = []
        if metrics.hw_clock_gating:
            gates.append("HW-gate")
        if metrics.sw_clock_gating:
            gates.append("SW-gate")
        if metrics.power_save:
            gates.append("PwrSave")
        gate_str = " ".join(gates) if gates else "none"

        poll_str = f"  TempPoll: {metrics.temp_poll_interval}ms\n" if metrics.temp_poll_interval > 0 else ""

        content = (
            f"[b]TPU Status[/b]\n"
            f"  Status:   [{status_color}]{metrics.status}[/]\n"
            f"  Activity: [{act_color}]{activity_bar}[/] {activity_pct:.0f}%\n"
            f"  TPU  [{tpu_tc}]{metrics.tpu_temp_c:5.1f}°C {tpu_bar}[/]\n"
            f"  CPU  [{cpu_tc}]{metrics.cpu_temp_c:5.1f}°C {cpu_bar}[/]\n"
            f"  Fan  {metrics.fan_state}/{metrics.fan_max}  {fan_bar}\n"
            f"  Trips: {trips}\n"
            f"  HW Warn: {warn_str}\n"
            f"{poll_str}"
            f"  Power: {metrics.power_state}  {gate_str}"
        )
        self.query_one("#tpu-status-content", Static).update(content)


class PcieDriverPanel(Widget):
    """PCIe link info and driver versions."""

    DEFAULT_CSS = """
    PcieDriverPanel {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="pcie-content")

    def update_content(self, metrics: TpuMetrics) -> None:
        # Device ownership
        if metrics.device_owner:
            owner = f"PID {metrics.device_owner}"
        elif metrics.is_device_owned:
            owner = "owned"
        else:
            owner = "none"
        if metrics.write_open_count > 0:
            owner += f" ({metrics.write_open_count} fd)"

        # PCIe link: current vs max
        cur_speed = metrics.pcie_link_speed
        cur_width = f"x{metrics.pcie_link_width}"
        if metrics.pcie_max_link_speed and metrics.pcie_max_link_width:
            max_speed = metrics.pcie_max_link_speed
            max_width = f"x{metrics.pcie_max_link_width}"
            speed_ok = cur_speed == max_speed
            width_ok = cur_width == max_width
            speed_color = "green" if speed_ok else "red"
            width_color = "green" if width_ok else "red"
            link_str = (
                f"[{speed_color}]{cur_speed}[/]/{max_speed} "
                f"[{width_color}]{cur_width}[/]/{max_width}"
            )
        else:
            link_str = f"{cur_speed} {cur_width}"

        # AER summary + non-zero detail
        aer_line = f"  AER: cor={metrics.aer_correctable} nf={metrics.aer_nonfatal} fat={metrics.aer_fatal}"
        all_detail = {}
        all_detail.update(metrics.aer_correctable_detail)
        all_detail.update(metrics.aer_nonfatal_detail)
        all_detail.update(metrics.aer_fatal_detail)
        if all_detail:
            detail_parts = [f"{k}={v}" for k, v in all_detail.items()]
            aer_line += f"\n  [yellow]  {' '.join(detail_parts)}[/]"

        content = (
            f"[b]PCIe & Driver[/b]\n"
            f"  {metrics.pci_device_name}\n"
            f"  Link: {link_str}\n"
            f"{aer_line}\n"
            f"  Driver: {metrics.driver_version}  Fw: {metrics.framework_version}\n"
            f"  Owner: {owner}  Resets: {metrics.reset_count}"
        )
        self.query_one("#pcie-content", Static).update(content)


class TempHistoryPanel(Widget):
    """Temperature sparkline history."""

    DEFAULT_CSS = """
    TempHistoryPanel {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="temp-history-content")

    def update_content(
        self,
        tpu_history: list[float],
        cpu_history: list[float],
        tpu_temp: float,
        cpu_temp: float,
    ) -> None:
        tpu_spark = sparkline(tpu_history)
        cpu_spark = sparkline(cpu_history)
        content = (
            f"[b]Temperature History (60s)[/b]\n"
            f"  {tpu_spark}  TPU [{temp_color(tpu_temp)}]{tpu_temp:.1f}°C[/]\n"
            f"  {cpu_spark}  CPU [{temp_color(cpu_temp)}]{cpu_temp:.1f}°C[/]"
        )
        self.query_one("#temp-history-content", Static).update(content)


class ActivityHistoryPanel(Widget):
    """Activity sparkline history."""

    DEFAULT_CSS = """
    ActivityHistoryPanel {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="activity-history-content")

    def update_content(
        self,
        activity_history: list[float],
        activity_pct: float,
    ) -> None:
        spark = sparkline(activity_history)
        if activity_pct < 5:
            color = "dim"
        elif activity_pct < 80:
            color = "green"
        else:
            color = "yellow"
        content = (
            f"[b]Activity History (60s)[/b]\n"
            f"  {spark}  [{color}]{activity_pct:.0f}%[/]"
        )
        self.query_one("#activity-history-content", Static).update(content)


class BenchmarkPanel(Widget):
    """Inference benchmark stats and latency sparkline."""

    DEFAULT_CSS = """
    BenchmarkPanel {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="bench-content")

    def update_content(self, stats: BenchmarkStats) -> None:
        if not stats.latencies_ms:
            content = (
                f"[b]Benchmark[/b]\n"
                f"  Model: {stats.model_name}\n"
                f"  Waiting for data..."
            )
        else:
            lat_spark = sparkline(list(stats.latencies_ms)[-40:])
            content = (
                f"[b]Benchmark[/b]\n"
                f"  Model: {stats.model_name} ({stats.model_size_kb:.0f} KB)\n"
                f"  cur {stats.latencies_ms[-1]:.2f}ms  "
                f"min {stats.min_ms:.2f}ms  "
                f"avg {stats.avg_ms:.2f}ms  "
                f"  {stats.throughput:.0f} inf/s\n"
                f"  p50 {stats.p50_ms:.2f}ms  "
                f"p95 {stats.p95_ms:.2f}ms  "
                f"p99 {stats.p99_ms:.2f}ms  "
                f"  n={stats.total_inferences}\n"
                f"  {lat_spark}"
            )
        self.query_one("#bench-content", Static).update(content)


class InterruptsPanel(Widget):
    """Interrupt counts and memory mappings."""

    DEFAULT_CSS = """
    InterruptsPanel {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="irq-content")

    def update_content(
        self,
        metrics: TpuMetrics,
        irq_rate: float,
    ) -> None:
        vec_strs = []
        for i, count in enumerate(metrics.interrupt_vectors[:4]):
            vec_strs.append(f"{i:02d}:{count}")
        vectors_line = "  ".join(vec_strs)

        # Polling CPU (hrtimer is pinned to one core)
        if metrics.irq_polling_cpu >= 0:
            poll_cpu_str = f"  Poll CPU: {metrics.irq_polling_cpu} (hrtimer)"
        else:
            poll_cpu_str = "  Poll CPU: n/a"

        content = (
            f"[b]Interrupts & Memory[/b]\n"
            f"  IRQ: {metrics.irq_count:,} (+{irq_rate:.0f}/s)\n"
            f"  Vectors: {vectors_line}\n"
            f"{poll_cpu_str}\n"
            f"  Pages: {metrics.mapped_pages}/{metrics.page_table_entries or 8192} mapped"
        )
        self.query_one("#irq-content", Static).update(content)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class MetricsUpdated(Message):
    def __init__(self, metrics: TpuMetrics) -> None:
        super().__init__()
        self.metrics = metrics


class BenchmarkResult(Message):
    def __init__(self, stats: BenchmarkStats) -> None:
        super().__init__()
        self.stats = stats


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class TpuMonApp(App):
    """Coral Edge TPU Monitor TUI."""

    TITLE = "tpumon - Coral Edge TPU Monitor"

    CSS = """
    Screen {
        layout: vertical;
    }
    #top-row {
        height: auto;
        layout: horizontal;
    }
    #top-row > * {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "toggle_bench", "Bench"),
        Binding("r", "reset_bench", "Reset"),
        Binding("plus,equal", "faster", "+Rate"),
        Binding("minus", "slower", "-Rate"),
    ]

    poll_interval: reactive[float] = reactive(1.0)
    bench_running: reactive[bool] = reactive(False)

    def __init__(
        self,
        start_bench: bool = False,
        model_path: Path | None = None,
        interval: float = 1.0,
    ) -> None:
        super().__init__()
        self._start_bench = start_bench
        self._model_path = model_path or DEFAULT_MODEL
        self._poll_timer = None
        self._poller = SysfsPoller()
        self.poll_interval = interval
        self._bench_stats = BenchmarkStats()
        self._tpu_temp_history: list[float] = []
        self._cpu_temp_history: list[float] = []
        self._activity_history: list[float] = []
        self._prev_vectors: list[int] | None = None
        self._prev_irq_count: int | None = None
        self._prev_timestamp: float | None = None
        self._activity_max = ACTIVITY_MAX_FLOOR
        self._excess_history: deque = deque(maxlen=60)
        self._irq_rate: float = 0.0
        self._activity_pct: float = 0.0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-row"):
            yield TpuStatusPanel()
            yield PcieDriverPanel()
        yield TempHistoryPanel()
        yield ActivityHistoryPanel()
        yield BenchmarkPanel()
        yield InterruptsPanel()
        yield Footer()

    def on_mount(self) -> None:
        self._poll_timer = self.set_interval(
            self.poll_interval, self._poll_sysfs
        )
        if self._start_bench:
            self.bench_running = True

    def watch_bench_running(self, running: bool) -> None:
        panel = self.query_one(BenchmarkPanel)
        panel.styles.display = "block" if running else "none"
        if running:
            self._bench_stats.reset()
            self._bench_stats.model_name = self._model_path.stem
            try:
                self._bench_stats.model_size_kb = (
                    self._model_path.stat().st_size / 1024.0
                )
            except OSError:
                self._bench_stats.model_size_kb = 0
            self._run_benchmark()

    def watch_poll_interval(self, interval: float) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = self.set_interval(
                interval, self._poll_sysfs
            )

    def _poll_sysfs(self) -> None:
        metrics = self._poller.poll()

        # Compute activity from per-vector interrupt deltas.
        # The hrtimer polls all vectors uniformly (~4000/s each), so the
        # raw total always looks busy.  Real TPU work causes specific
        # vectors (doorbell / completion) to increment *faster* than the
        # baseline.  We use the median per-vector delta as the baseline
        # and sum the excess above it as the activity signal.
        now = metrics.timestamp
        vectors = metrics.interrupt_vectors
        excess = 0.0
        if self._prev_vectors is not None and self._prev_timestamp is not None:
            dt = now - self._prev_timestamp
            if dt > 0 and len(vectors) == len(self._prev_vectors):
                deltas = [
                    (cur - prev) / dt
                    for cur, prev in zip(vectors, self._prev_vectors)
                ]
                baseline = statistics.median(deltas) if deltas else 0.0
                excess = sum(max(0.0, d - baseline) for d in deltas)

        # Auto-scale activity max from recent excess rates.
        # Use the peak excess from the last 60 polls to set the 100%
        # mark.  This adapts to the current workload: single inferences
        # register clearly, while sustained bursts scale the gauge up.
        self._excess_history.append(excess)
        observed_max = max(self._excess_history)
        self._activity_max = max(observed_max * 1.25, ACTIVITY_MAX_FLOOR)
        self._activity_pct = activity_percent(excess, self._activity_max)

        self._prev_vectors = vectors
        self._prev_timestamp = now

        if self._prev_irq_count is not None:
            irq_delta = metrics.irq_count - self._prev_irq_count
            self._irq_rate = irq_delta / self.poll_interval if self.poll_interval > 0 else 0
        self._prev_irq_count = metrics.irq_count

        # History (keep 60 samples each)
        self._tpu_temp_history.append(metrics.tpu_temp_c)
        self._cpu_temp_history.append(metrics.cpu_temp_c)
        self._activity_history.append(self._activity_pct)
        if len(self._tpu_temp_history) > 60:
            self._tpu_temp_history = self._tpu_temp_history[-60:]
        if len(self._cpu_temp_history) > 60:
            self._cpu_temp_history = self._cpu_temp_history[-60:]
        if len(self._activity_history) > 60:
            self._activity_history = self._activity_history[-60:]

        self.post_message(MetricsUpdated(metrics))

    def on_metrics_updated(self, message: MetricsUpdated) -> None:
        m = message.metrics
        self.query_one(TpuStatusPanel).update_content(m, self._activity_pct)
        self.query_one(PcieDriverPanel).update_content(m)
        self.query_one(TempHistoryPanel).update_content(
            self._tpu_temp_history,
            self._cpu_temp_history,
            m.tpu_temp_c,
            m.cpu_temp_c,
        )
        self.query_one(ActivityHistoryPanel).update_content(
            self._activity_history, self._activity_pct
        )
        self.query_one(InterruptsPanel).update_content(m, self._irq_rate)
        if self.bench_running:
            self.query_one(BenchmarkPanel).update_content(self._bench_stats)

    @work(thread=True)
    def _run_benchmark(self) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            self._bench_stats.model_name += " [tflite not available]"
            return

        import numpy as np

        model_path = str(self._model_path)
        try:
            delegate = load_delegate("libedgetpu.so.1")
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[delegate],
            )
        except (ValueError, OSError):
            try:
                # Fall back to CPU
                cpu_path = model_path.replace("_edgetpu.tflite", ".tflite")
                interpreter = Interpreter(model_path=cpu_path)
                self._bench_stats.model_name += " [CPU]"
            except Exception:
                self._bench_stats.model_name += " [load failed]"
                return

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Allocate dummy input
        input_shape = input_details["shape"]
        input_dtype = input_details["dtype"]
        dummy_input = np.zeros(input_shape, dtype=input_dtype)

        self._bench_stats.start_time = time.monotonic()
        worker = get_current_worker()

        while self.bench_running:
            if worker.is_cancelled:
                break
            interpreter.set_tensor(input_details["index"], dummy_input)
            t0 = time.perf_counter()
            interpreter.invoke()
            latency = (time.perf_counter() - t0) * 1000.0
            self._bench_stats.record(latency)

            # Calibrate activity max from benchmark excess rate.
            # After 100 inferences the excess interrupt rate reflects
            # real TPU workload; treat it as the ~80% mark.
            if self._bench_stats.total_inferences == 100:
                if self._activity_pct > 0:
                    current_excess = self._activity_pct / 100.0 * self._activity_max
                    self._activity_max = max(current_excess * 1.25, ACTIVITY_MAX_FLOOR)

    def action_toggle_bench(self) -> None:
        self.bench_running = not self.bench_running

    def action_reset_bench(self) -> None:
        if self.bench_running:
            self._bench_stats.reset()

    def action_faster(self) -> None:
        self.poll_interval = max(0.25, self.poll_interval - 0.25)

    def action_slower(self) -> None:
        self.poll_interval = min(5.0, self.poll_interval + 0.25)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tpumon -- Coral Edge TPU Monitor TUI"
    )
    parser.add_argument(
        "-b",
        "--bench",
        action="store_true",
        help="Start with benchmark running",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to Edge TPU compiled model for benchmark",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    app = TpuMonApp(
        start_bench=args.bench,
        model_path=args.model,
        interval=args.interval,
    )
    app.run()


if __name__ == "__main__":
    main()
