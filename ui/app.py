"""
GeneWeaver - Textual Terminal User Interface (Week 2)
Real-time dashboard displaying live GPU memory consumption,
core utilization, genomic chunking progress, and sequence alignment results.
"""

import asyncio
import os
import sys
import time
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Label, Button
from textual.worker import Worker, WorkerState

from engine.parser import FastAStreamParser
from engine.cuda_kernels import CUDAAlignmentEngine
from engine.cpu_baseline import CPUBaselineAligner
from ui.widgets import GPUMonitorWidget, ChunkProgressWidget, HitsTableWidget


TUI_CSS = """
Screen {
    background: #0f141c;
    color: #e0e6ed;
}

Header {
    background: #1a2234;
    color: #4da6ff;
    dock: top;
    text-style: bold;
}

Footer {
    background: #1a2234;
    color: #8fa0b5;
    dock: bottom;
}

.box {
    background: #161f30;
    border: solid #2c3e5a;
    padding: 1 2;
    margin: 1;
}

.gpu-box {
    background: #161f30;
    border: solid #00b4d8;
    padding: 1;
    margin-bottom: 1;
}

.gpu-header {
    text-style: bold;
    color: #00b4d8;
    margin-bottom: 1;
}

.chunk-box {
    background: #161f30;
    border: solid #06d6a0;
    padding: 1;
    margin-bottom: 1;
}

.chunk-header {
    text-style: bold;
    color: #06d6a0;
    margin-bottom: 1;
}

#left-panel {
    width: 42%;
    height: 100%;
    padding: 1;
}

#right-panel {
    width: 58%;
    height: 100%;
    padding: 1;
}

#hits-box {
    height: 100%;
    border: solid #8338ec;
    background: #161f30;
    padding: 1;
}

#hits-header {
    text-style: bold;
    color: #ff007f;
    margin-bottom: 1;
}

DataTable {
    height: 100%;
    background: #121927;
}

ProgressBar {
    margin: 1 0;
}

#status-bar {
    background: #1a2234;
    color: #ffd166;
    padding: 0 1;
    height: 1;
}
"""


class GeneWeaverApp(App):
    """GeneWeaver Interactive Terminal Dashboard."""

    CSS = TUI_CSS
    TITLE = "GeneWeaver"
    SUB_TITLE = "GPU-Accelerated CRISPR Alignment Engine"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "start_scan", "Start/Restart Scan"),
    ]

    def __init__(
        self,
        fasta_path: str,
        guide_seq: str = "GAGTCCGAGCAGAAGAAGAA",
        max_mismatches: int = 3,
        allow_nag: bool = True,
        chunk_size: int = 100_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fasta_path = fasta_path
        self.guide_seq = guide_seq
        self.max_mismatches = max_mismatches
        self.allow_nag = allow_nag
        self.chunk_size = chunk_size

        self.engine: Optional[CUDAAlignmentEngine] = None
        self.is_scanning = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="left-panel"):
                yield GPUMonitorWidget(
                    device_name="NVIDIA RTX 3050",
                    compute_cap="8.6",
                    id="gpu-widget",
                )
                yield ChunkProgressWidget(id="chunk-widget")

                with Vertical(classes="box"):
                    yield Label("[bold white]🎯 CRISPR Configuration[/bold white]")
                    yield Label(f"Guide RNA: [yellow]{self.guide_seq}[/yellow]")
                    yield Label(f"Max Mismatches: [cyan]{self.max_mismatches}[/cyan]")
                    yield Label(f"PAM: [green]NGG{' & NAG' if self.allow_nag else ''}[/green]")
                    yield Label(f"Dataset: [dim]{os.path.basename(self.fasta_path)}[/dim]")

            with Vertical(id="right-panel"):
                with Vertical(id="hits-box"):
                    yield Label("🔬 Detected CRISPR Off-Target & On-Target Sites", id="hits-header")
                    yield HitsTableWidget(id="hits-widget")

        yield Label("Status: Ready. Press [s] to initiate GPU scan, [q] to exit.", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize CUDA engine and GPU monitoring upon mounting."""
        try:
            self.engine = CUDAAlignmentEngine(
                guide_sequence=self.guide_seq,
                max_mismatches=self.max_mismatches,
                allow_nag_pam=self.allow_nag,
            )
            gpu_widget = self.query_one("#gpu-widget", GPUMonitorWidget)
            gpu_widget.device_name = self.engine.device_name
            gpu_widget.compute_cap = f"{self.engine.compute_cap[0]}.{self.engine.compute_cap[1]}"

            free_b, tot_b = self.engine.get_vram_info()
            gpu_widget.update_vram(free_b, tot_b)
            self.query_one("#status-bar", Label).update("Status: [bold green]RTX 3050 GPU Initialized.[/bold green] Press [s] to start scan.")
        except Exception as e:
            self.query_one("#status-bar", Label).update(f"Status: [bold red]CUDA Init Error: {e}[/bold red]")

        # Set periodic timer to refresh GPU VRAM
        self.set_interval(1.0, self.refresh_vram)

    def refresh_vram(self) -> None:
        """Periodic background refresh of GPU VRAM gauge."""
        if self.engine:
            try:
                free_b, tot_b = self.engine.get_vram_info()
                self.query_one("#gpu-widget", GPUMonitorWidget).update_vram(free_b, tot_b)
            except Exception:
                pass

    def action_start_scan(self) -> None:
        """Trigger background GPU alignment scan."""
        if self.is_scanning:
            return
        self.is_scanning = True
        self.query_one("#status-bar", Label).update("Status: [bold cyan]Scanning genomic chunks across GPU cores...[/bold cyan]")
        self.run_worker(self._async_alignment_worker, exclusive=True, thread=True)

    def _async_alignment_worker(self) -> None:
        """Worker thread executing the sliding-window CUDA alignment pipeline."""
        if not self.engine or not os.path.exists(self.fasta_path):
            return

        parser = FastAStreamParser(self.fasta_path, chunk_size=self.chunk_size, overlap_size=22)
        summary = parser.get_summary()
        total_bp = summary["total_base_pairs"]
        chunks = list(parser.stream_chunks())
        total_chunks = len(chunks)

        chunk_widget = self.query_one("#chunk-widget", ChunkProgressWidget)
        hits_widget = self.query_one("#hits-widget", HitsTableWidget)

        scanned_bp = 0
        start_time = time.perf_counter()

        for idx, chunk in enumerate(chunks):
            # Align on GPU
            chunk_hits = self.engine.align_chunk(chunk)

            # Update progress metrics
            scanned_bp += (chunk.length - chunk.overlap_size) if scanned_bp > 0 else chunk.length
            elapsed = time.perf_counter() - start_time
            throughput = (scanned_bp / 1_000_000) / elapsed if elapsed > 0 else 0.0

            # Post updates to UI
            self.call_from_thread(
                chunk_widget.update_progress,
                scanned_bp,
                total_bp,
                idx,
                total_chunks,
                throughput,
            )

            for hit in chunk_hits:
                self.call_from_thread(hits_widget.add_hit, hit)

            # Refresh VRAM after chunk processing
            free_b, tot_b = self.engine.get_vram_info()
            self.call_from_thread(self.query_one("#gpu-widget", GPUMonitorWidget).update_vram, free_b, tot_b)

            time.sleep(0.01)  # Brief yield for UI fluidity

        self.is_scanning = False
        self.call_from_thread(
            self.query_one("#status-bar", Label).update,
            f"Status: [bold green]Scan Completed Successfully![/bold green] Throughput: {throughput:.2f} Mbp/s.",
        )


def launch_tui(fasta_path: str, guide_seq: str = "GAGTCCGAGCAGAAGAAGAA"):
    """Launch the GeneWeaver Textual application."""
    app = GeneWeaverApp(fasta_path=fasta_path, guide_seq=guide_seq)
    app.run()


if __name__ == "__main__":
    test_fasta = os.path.join(os.path.dirname(__file__), "..", "data", "mock_chr1.fasta")
    launch_tui(test_fasta)
