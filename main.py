"""
GeneWeaver - CLI Driver & Week 1 Runner
GPU-Accelerated CRISPR Alignment Engine
"""

import argparse
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from data.mock_genome_generator import generate_mock_genome
from engine.parser import FastAStreamParser
from engine.cpu_baseline import CPUBaselineAligner

console = Console()

DEFAULT_GUIDE = "GAGTCCGAGCAGAAGAAGAA"  # Human EMX1 locus


def print_banner():
    banner = """[bold cyan]
  ____                  __        __                            
 / ___| ___ _ __   ___  \\ \\      / /__  __ ___   _____ _ __     
| |  _ / _ \\ '_ \\ / _ \\  \\ \\ /\\ / / _ \\/ _` \\ \\ / / _ \\ '__|    
| |_| |  __/ | | |  __/   \\ V  V /  __/ (_| |\\ V /  __/ |       
 \\____|\\___|_| |_|\\___|    \\_/\\_/ \\___|\\__,_| \\_/ \\___|_|       
[/bold cyan]
[bold white]GPU-Accelerated CRISPR Alignment Engine[/bold white]
[dim]Bioinformatics & High-Performance Computing (HPC)[/dim]
"""
    console.print(Panel(banner, border_style="cyan"))


def run_week1_pipeline(
    fasta_path: str,
    guide_seq: str,
    max_mismatches: int = 3,
    allow_nag: bool = True,
    chunk_size: int = 100_000,
):
    print_banner()

    console.print(f"[bold green]▶ Target Guide RNA:[/bold green] [yellow]{guide_seq}[/yellow] (20 bp)")
    console.print(f"[bold green]▶ Max Mismatches:[/bold green]  [yellow]{max_mismatches}[/yellow]")
    console.print(f"[bold green]▶ PAM Motif:[/bold green]       [yellow]NGG{' & NAG' if allow_nag else ''}[/yellow]")
    console.print(f"[bold green]▶ Target FASTA:[/bold green]     [cyan]{fasta_path}[/cyan]\n")

    parser = FastAStreamParser(fasta_path, chunk_size=chunk_size, overlap_size=22)
    summary = parser.get_summary()

    console.print(f"[bold blue]ℹ Genomic Summary:[/bold blue] {len(summary['chromosomes'])} chromosome(s), {summary['total_base_pairs']:,} base pairs total.")

    aligner = CPUBaselineAligner(
        guide_sequence=guide_seq,
        max_mismatches=max_mismatches,
        allow_nag_pam=allow_nag,
    )

    console.print("\n[bold yellow]⚡ Running Pure Python CPU Baseline Aligner...[/bold yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Scanning DNA chunks...", total=summary["total_base_pairs"])

        all_hits = []
        total_bp = 0
        import time
        start_t = time.perf_counter()

        for chunk in parser.stream_chunks():
            chunk_hits = aligner.align_chunk(chunk)
            all_hits.extend(chunk_hits)
            scanned = chunk.length - chunk.overlap_size if total_bp > 0 else chunk.length
            total_bp += scanned
            progress.update(task, completed=min(total_bp, summary["total_base_pairs"]))

        elapsed = time.perf_counter() - start_t

    throughput = (total_bp / 1_000_000) / elapsed if elapsed > 0 else 0.0

    # Display Results Table
    console.print("\n[bold green]✔ Alignment Scan Complete![/bold green]\n")

    perf_table = Table(title="CPU Baseline Benchmark Results", border_style="cyan")
    perf_table.add_column("Metric", style="bold white")
    perf_table.add_column("Value", style="bold yellow")

    perf_table.add_row("Base Pairs Scanned", f"{total_bp:,} bp ({total_bp / 1e6:.2f} Mbp)")
    perf_table.add_row("Execution Time", f"{elapsed:.4f} s")
    perf_table.add_row("Throughput", f"{throughput:.3f} Mbp/s")
    perf_table.add_row("Candidates Found", str(len(all_hits)))
    console.print(perf_table)

    # Display Hits Table (Top 15)
    if all_hits:
        hits_table = Table(title=f"Detected CRISPR Target Sites (Showing Top {min(15, len(all_hits))})", border_style="magenta")
        hits_table.add_column("Coord", style="dim")
        hits_table.add_column("Strand", style="cyan")
        hits_table.add_column("Target Sequence (20bp spacer + 3bp PAM)", style="bold")
        hits_table.add_column("PAM", style="green")
        hits_table.add_column("Mismatches", style="red")
        hits_table.add_column("Type", style="italic")

        # Sort hits by mismatches (on-target first)
        sorted_hits = sorted(all_hits, key=lambda h: (h.mismatches, h.position))

        for hit in sorted_hits[:15]:
            site_type = "[bold green]ON-TARGET[/bold green]" if hit.mismatches == 0 else f"[yellow]OFF-TARGET ({hit.mismatches} mut)[/yellow]"
            
            # Format sequence with highlight
            spacer_str = hit.spacer_sequence
            pam_str = f"[bold green]{hit.pam}[/bold green]"
            seq_display = f"{spacer_str} {pam_str}"

            hits_table.add_row(
                f"{hit.chromosome}:{hit.position:,}",
                hit.strand,
                seq_display,
                hit.pam,
                str(hit.mismatches),
                site_type,
            )

        console.print(hits_table)
    else:
        console.print("[yellow]No target sites discovered within the mismatch threshold.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="GeneWeaver: GPU-Accelerated CRISPR Alignment Engine")
    parser.add_argument("--fasta", type=str, default=None, help="Path to input .fasta file")
    parser.add_argument("--guide", type=str, default=DEFAULT_GUIDE, help="20bp CRISPR guide RNA sequence")
    parser.add_argument("--max-mismatches", type=int, default=3, help="Max allowable mismatches in spacer (default: 3)")
    parser.add_argument("--no-nag", action="store_true", help="Disallow non-canonical NAG PAMs")
    parser.add_argument("--gen-mock", action="store_true", help="Generate a mock genome before running")
    parser.add_argument("--mock-bp", type=int, default=1_000_000, help="Number of base pairs for mock genome (default: 1,000,000)")

    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    default_fasta = os.path.join(data_dir, "mock_chr1.fasta")
    default_gt = os.path.join(data_dir, "ground_truth.json")

    if args.gen_mock or not args.fasta or not os.path.exists(args.fasta or default_fasta):
        console.print(f"[bold cyan]Generating synthetic mock genome ({args.mock_bp:,} bp)...[/bold cyan]")
        generate_mock_genome(
            output_fasta=default_fasta,
            output_ground_truth=default_gt,
            total_base_pairs=args.mock_bp,
            guide_sequences=[args.guide],
            num_off_targets_per_guide=15,
            max_mismatches=args.max_mismatches,
        )
        target_fasta = default_fasta
    else:
        target_fasta = args.fasta

    run_week1_pipeline(
        fasta_path=target_fasta,
        guide_seq=args.guide,
        max_mismatches=args.max_mismatches,
        allow_nag=not args.no_nag,
    )


if __name__ == "__main__":
    main()
