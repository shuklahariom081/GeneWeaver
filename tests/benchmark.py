"""
GeneWeaver - Performance Audit & Benchmark Runner (Mid-Project Review)
Benchmarks the GPU implementation against the CPU baseline,
auditing the speedup multiplier and VRAM memory behavior across dataset scales.
"""

import argparse
import os
import sys
import time
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")


from data.mock_genome_generator import generate_mock_genome
from engine.parser import FastAStreamParser
from engine.cpu_baseline import CPUBaselineAligner
from engine.cuda_kernels import CUDAAlignmentEngine

console = Console()


def run_benchmark_audit(scales=(200_000, 1_000_000, 2_500_000), guide_seq="GAGTCCGAGCAGAAGAAGAA"):
    console.print(Panel(
        "[bold cyan]GeneWeaver Mid-Project Review: Performance & Memory Audit[/bold cyan]\n"
        "[dim]Auditing CPU Baseline vs Numba CUDA Engine on NVIDIA RTX 3050[/dim]",
        border_style="cyan"
    ))

    data_dir = os.path.join(os.path.dirname(__file__), "benchmark_data")
    os.makedirs(data_dir, exist_ok=True)

    results_table = Table(title="Performance Audit: CPU Baseline vs Numba CUDA JIT", border_style="green")
    results_table.add_column("Dataset Size (bp)", style="bold cyan")
    results_table.add_column("CPU Time (s)", style="bold yellow")
    results_table.add_column("CPU Mbp/s", style="yellow")
    results_table.add_column("GPU Time (s)", style="bold green")
    results_table.add_column("GPU Mbp/s", style="green")
    results_table.add_column("Speedup Multiplier", style="bold magenta")
    results_table.add_column("Hit Parity", style="white")

    for bp in scales:
        fasta_path = os.path.join(data_dir, f"bench_{bp}.fasta")
        gt_path = os.path.join(data_dir, f"bench_{bp}_gt.json")

        console.print(f"[bold blue]Generating {bp:,} bp dataset...[/bold blue]")
        generate_mock_genome(
            output_fasta=fasta_path,
            output_ground_truth=gt_path,
            total_base_pairs=bp,
            chromosome_name=f"chrBench_{bp}",
            guide_sequences=[guide_seq],
            num_off_targets_per_guide=15,
            max_mismatches=3,
            random_seed=789,
        )

        chunk_size = 250_000 if bp >= 500_000 else 50_000
        parser_cpu = FastAStreamParser(fasta_path, chunk_size=chunk_size, overlap_size=22)
        parser_gpu = FastAStreamParser(fasta_path, chunk_size=chunk_size, overlap_size=22)

        cpu_engine = CPUBaselineAligner(guide_seq, max_mismatches=3, allow_nag_pam=True, search_both_strands=False)
        cuda_engine = CUDAAlignmentEngine(guide_seq, max_mismatches=3, allow_nag_pam=True, search_both_strands=False)

        # Run CPU
        console.print(f"  ▶ Benchmarking CPU Baseline on {bp:,} bp...")
        cpu_res = cpu_engine.align_fasta(parser_cpu)

        # Run GPU
        console.print(f"  ▶ Benchmarking CUDA GPU Engine on {bp:,} bp...")
        gpu_res = cuda_engine.align_fasta(parser_gpu)

        speedup = cpu_res.elapsed_seconds / gpu_res.elapsed_seconds if gpu_res.elapsed_seconds > 0 else 0.0
        parity = "✔ Match" if cpu_res.hits_found == gpu_res.hits_found else "✘ Mismatch"

        results_table.add_row(
            f"{bp:,} bp ({bp / 1e6:.1f} Mbp)",
            f"{cpu_res.elapsed_seconds:.4f} s",
            f"{cpu_res.throughput_mbps:.2f}",
            f"{gpu_res.elapsed_seconds:.4f} s",
            f"{gpu_res.throughput_mbps:.2f}",
            f"[bold magenta]{speedup:.1f}x[/bold magenta]",
            f"[bold green]{parity}[/bold green] ({gpu_res.hits_found} sites)",
        )

    console.print("\n")
    console.print(results_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick 2-scale audit")
    args = parser.parse_args()

    if args.quick:
        run_benchmark_audit(scales=(250_000, 1_000_000))
    else:
        run_benchmark_audit()
