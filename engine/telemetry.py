"""
GeneWeaver - Hardware Telemetry & Profiling Module
Tracks real-time GPU VRAM consumption, host memory allocation,
and memory transfer bandwidth between Host RAM and Device VRAM.
"""

from dataclasses import dataclass
import os
import time
from typing import Dict, Optional, Tuple
import psutil
from numba import cuda


@dataclass
class HardwareSnapshot:
    """Represents a point-in-time telemetry snapshot of host and GPU hardware."""
    timestamp: float
    gpu_available: bool
    gpu_name: str
    compute_capability: str
    vram_total_mb: float
    vram_free_mb: float
    vram_used_mb: float
    vram_percent: float
    host_ram_total_mb: float
    host_ram_used_mb: float
    host_ram_percent: float
    cpu_percent: float


class HardwareTelemetry:
    """Monitors hardware health, VRAM paging headroom, and host RAM utilization."""

    def __init__(self):
        self.gpu_available = cuda.is_available()
        self.gpu_name = "N/A"
        self.compute_cap = "N/A"

        if self.gpu_available:
            try:
                device = cuda.get_current_device()
                self.gpu_name = device.name.decode("utf-8") if isinstance(device.name, bytes) else str(device.name)
                self.compute_cap = f"{device.compute_capability[0]}.{device.compute_capability[1]}"
            except Exception:
                self.gpu_available = False

    def capture_snapshot(self) -> HardwareSnapshot:
        """Capture current memory and utilization telemetry."""
        now = time.time()

        # Host RAM
        mem = psutil.virtual_memory()
        host_total_mb = mem.total / (1024 * 1024)
        host_used_mb = mem.used / (1024 * 1024)
        host_pct = mem.percent
        cpu_pct = psutil.cpu_percent(interval=None)

        # GPU VRAM
        vram_total_mb = 0.0
        vram_free_mb = 0.0
        vram_used_mb = 0.0
        vram_pct = 0.0

        if self.gpu_available:
            try:
                ctx_mem = cuda.current_context().get_memory_info()
                vram_free_mb = ctx_mem.free / (1024 * 1024)
                vram_total_mb = ctx_mem.total / (1024 * 1024)
                vram_used_mb = max(0.0, vram_total_mb - vram_free_mb)
                vram_pct = (vram_used_mb / vram_total_mb * 100.0) if vram_total_mb > 0 else 0.0
            except Exception:
                pass

        return HardwareSnapshot(
            timestamp=now,
            gpu_available=self.gpu_available,
            gpu_name=self.gpu_name,
            compute_capability=self.compute_cap,
            vram_total_mb=vram_total_mb,
            vram_free_mb=vram_free_mb,
            vram_used_mb=vram_used_mb,
            vram_percent=vram_pct,
            host_ram_total_mb=host_total_mb,
            host_ram_used_mb=host_used_mb,
            host_ram_percent=host_pct,
            cpu_percent=cpu_pct,
        )

    def calculate_transfer_bandwidth_gbps(self, byte_count: int, elapsed_seconds: float) -> float:
        """Calculate PCI-e Host <-> Device transfer bandwidth in Gigabytes/second (GB/s)."""
        if elapsed_seconds <= 0:
            return 0.0
        gigabytes = byte_count / (1024 ** 3)
        return gigabytes / elapsed_seconds
