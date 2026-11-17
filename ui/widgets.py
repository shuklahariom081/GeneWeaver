"""
GeneWeaver - Textual UI Custom Widgets (Week 2)
Provides real-time GPU telemetry, chunking progress bars, and genomic hit tables.
"""

from typing import Optional, Tuple
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, ProgressBar, DataTable, Label
from textual.reactive import reactive


class GPUMonitorWidget(Static):
    """Displays real-time NVIDIA GPU VRAM allocation and device status."""

    free_mb = reactive(0.0)
    total_mb = reactive(0.0)
    used_mb = reactive(0.0)
    util_percent = reactive(0.0)

    def __init__(self, device_name: str = "NVIDIA GPU", compute_cap: str = "8.6", **kwargs):
        super().__init__(**kwargs)
        self.device_name = device_name
        self.compute_cap = compute_cap

    def compose(self) -> ComposeResult:
        with Vertical(classes="gpu-box"):
            yield Label(f"🎮 GPU Hardware: [bold cyan]{self.device_name}[/bold cyan] (sm_{self.compute_cap})", classes="gpu-header")
            yield Label("VRAM Allocation: Initializing...", id="vram-label")
            yield ProgressBar(total=100, show_eta=False, id="vram-bar")

    def update_vram(self, free_bytes: int, total_bytes: int):
        """Update VRAM gauge with live device memory metrics."""
        self.free_mb = free_bytes / (1024 * 1024)
        self.total_mb = total_bytes / (1024 * 1024)
        self.used_mb = max(0.0, self.total_mb - self.free_mb)
        self.util_percent = (self.used_mb / self.total_mb * 100.0) if self.total_mb > 0 else 0.0

        vram_label = self.query_one("#vram-label", Label)
        vram_bar = self.query_one("#vram-bar", ProgressBar)

        vram_label.update(
            f"VRAM: [bold yellow]{self.used_mb:.1f} MB[/bold yellow] / "
            f"[dim]{self.total_mb:.1f} MB[/dim] ({self.util_percent:.1f}% used) | "
            f"[green]{self.free_mb:.1f} MB Free[/green]"
        )
        vram_bar.progress = self.util_percent


class ChunkProgressWidget(Static):
    """Tracks the genomic ingestion pipeline, chunk boundaries, and alignment throughput."""

    total_bp = reactive(0)
    scanned_bp = reactive(0)
    throughput_mbps = reactive(0.0)

    def compose(self) -> ComposeResult:
        with Vertical(classes="chunk-box"):
            yield Label("🧬 Genomic Pipeline: [bold green]Active[/bold green]", classes="chunk-header")
            yield Label("Chunks: Initializing...", id="chunk-label")
            yield ProgressBar(total=100, id="chunk-bar")
            yield Label("Throughput: 0.00 Mbp/s", id="throughput-label")

    def update_progress(self, scanned_bp: int, total_bp: int, current_chunk: int, total_chunks: int, throughput: float):
        """Update progress bar and throughput meter."""
        self.scanned_bp = scanned_bp
        self.total_bp = total_bp
        self.throughput_mbps = throughput

        pct = (scanned_bp / total_bp * 100.0) if total_bp > 0 else 0.0

        chunk_label = self.query_one("#chunk-label", Label)
        chunk_bar = self.query_one("#chunk-bar", ProgressBar)
        tp_label = self.query_one("#throughput-label", Label)

        chunk_label.update(
            f"Chunk [bold cyan]{current_chunk + 1}[/bold cyan] of [dim]{total_chunks}[/dim] | "
            f"Processed: [yellow]{scanned_bp:,}[/yellow] / [dim]{total_bp:,} bp[/dim] ({pct:.1f}%)"
        )
        chunk_bar.progress = pct
        tp_label.update(f"⚡ GPU Throughput: [bold green]{throughput:.2f} Mbp/s[/bold green]")


class HitsTableWidget(Static):
    """DataTable rendering discovered CRISPR target and off-target candidate sites."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="hits-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "Coordinate",
            "Strand",
            "Target Sequence (20bp spacer + PAM)",
            "PAM",
            "Mismatches",
            "Severity",
        )

    def add_hit(self, hit):
        """Append an alignment hit to the interactive table."""
        table = self.query_one(DataTable)

        if hit.mismatches == 0:
            sev = Text("ON-TARGET", style="bold green")
        elif hit.mismatches == 1:
            sev = Text("HIGH RISK (1 mut)", style="bold yellow")
        elif hit.mismatches == 2:
            sev = Text("MED RISK (2 mut)", style="dark_orange")
        else:
            sev = Text(f"LOW RISK ({hit.mismatches} mut)", style="dim red")

        # Visual formatting for spacer and PAM
        seq_text = Text()
        seq_text.append(hit.spacer_sequence, style="bold white")
        seq_text.append(" ")
        seq_text.append(hit.pam, style="bold green")

        table.add_row(
            f"{hit.chromosome}:{hit.position:,}",
            hit.strand,
            seq_text,
            hit.pam,
            str(hit.mismatches),
            sev,
        )
