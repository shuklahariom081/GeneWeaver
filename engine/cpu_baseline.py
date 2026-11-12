"""
GeneWeaver - CPU Baseline CRISPR Alignment Engine
Pure Python sliding-window sequence alignment algorithm.
Establishes the performance baseline (time taken, throughput in Mbp/s)
to benchmark future GPU-accelerated CUDA kernels.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional
import numpy as np

from engine.parser import encode_sequence, decode_sequence, GenomicChunk, FastAStreamParser


@dataclass
class AlignmentHit:
    """Represents a detected CRISPR off-target or on-target candidate."""
    chromosome: str
    position: int  # 0-based global genomic coordinate
    strand: str    # '+' for forward, '-' for reverse complement
    target_sequence: str
    spacer_sequence: str
    pam: str
    mismatches: int
    mutated_indices: List[int] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Stores performance metrics from an alignment run."""
    engine_name: str
    total_base_pairs: int
    elapsed_seconds: float
    throughput_mbps: float  # Megabases per second
    hits_found: int
    hits_by_mismatch: Dict[int, int]
    hits: List[AlignmentHit]

    def summary(self) -> str:
        lines = [
            f"=== {self.engine_name} Performance Report ===",
            f"Base Pairs Scanned: {self.total_base_pairs:,} bp ({self.total_base_pairs / 1e6:.2f} Mbp)",
            f"Time Elapsed:       {self.elapsed_seconds:.4f} seconds",
            f"Throughput:         {self.throughput_mbps:.3f} Mbp/s",
            f"Total Candidates:   {self.hits_found}",
            f"Mismatch Breakdown: {dict(sorted(self.hits_by_mismatch.items()))}",
        ]
        return "\n".join(lines)


class CPUBaselineAligner:
    """
    Pure Python sliding-window aligner for CRISPR-Cas9 sequence search.
    Operates without JIT or low-level acceleration to provide a rigorous baseline.
    """

    def __init__(
        self,
        guide_sequence: str,
        max_mismatches: int = 3,
        allow_nag_pam: bool = False,
        search_both_strands: bool = True,
    ):
        guide_clean = guide_sequence.strip().upper()
        if len(guide_clean) != 20:
            raise ValueError(f"CRISPR spacer must be 20 bp, got {len(guide_clean)} bp")

        self.guide_str = guide_clean
        self.guide_encoded = encode_sequence(guide_clean)  # shape (20,) uint8
        self.max_mismatches = max_mismatches
        self.allow_nag_pam = allow_nag_pam
        self.search_both_strands = search_both_strands

        # Precompute reverse complement guide spacer
        # Complement: A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0)
        complement_map = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
        rc_encoded = np.array([complement_map[x] for x in self.guide_encoded[::-1]], dtype=np.uint8)
        self.guide_rc_str = decode_sequence(rc_encoded)

    def _align_window_forward(
        self,
        seq_arr: np.ndarray,
        window_start: int,
        global_offset: int,
        chromosome: str,
    ) -> Optional[AlignmentHit]:
        """
        Check forward strand window [start : start + 23].
        Window format: 20bp spacer + 3bp PAM (pos 20: N, pos 21: G, pos 22: G)
        """
        # 1. Validate PAM (NGG or NAG)
        pam_base1 = seq_arr[window_start + 21]  # G = 2
        pam_base2 = seq_arr[window_start + 22]  # G = 2

        if pam_base2 != 2:
            return None  # Last base must be G

        if pam_base1 != 2:
            if not (self.allow_nag_pam and pam_base1 == 0):  # A = 0
                return None

        # 2. Count mismatches against the 20bp guide spacer
        mismatches = 0
        mutated_indices = []
        for i in range(20):
            if seq_arr[window_start + i] != self.guide_encoded[i]:
                mismatches += 1
                mutated_indices.append(i)
                if mismatches > self.max_mismatches:
                    return None

        full_window = seq_arr[window_start : window_start + 23]
        target_seq = decode_sequence(full_window)

        return AlignmentHit(
            chromosome=chromosome,
            position=global_offset + window_start,
            strand="+",
            target_sequence=target_seq,
            spacer_sequence=target_seq[:20],
            pam=target_seq[20:],
            mismatches=mismatches,
            mutated_indices=mutated_indices,
        )

    def align_chunk(self, chunk: GenomicChunk) -> List[AlignmentHit]:
        """Align guide against a single genomic chunk."""
        hits = []
        seq = chunk.data
        seq_len = chunk.length
        chrom = chunk.chromosome
        offset = chunk.global_start_offset

        # Sliding window across all valid 23 bp spans
        for w_start in range(seq_len - 22):
            hit = self._align_window_forward(seq, w_start, offset, chrom)
            if hit is not None:
                hits.append(hit)

        return hits

    def align_fasta(self, parser: FastAStreamParser) -> BenchmarkResult:
        """
        Stream entire FASTA file, execute alignment, and measure precise performance.
        """
        all_hits: List[AlignmentHit] = []
        total_bp = 0

        start_time = time.perf_counter()

        for chunk in parser.stream_chunks():
            total_bp += (chunk.length - chunk.overlap_size) if total_bp > 0 else chunk.length
            chunk_hits = self.align_chunk(chunk)
            all_hits.extend(chunk_hits)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        throughput = (total_bp / 1_000_000) / elapsed if elapsed > 0 else 0.0

        # Tally breakdown
        breakdown: Dict[int, int] = {}
        for h in all_hits:
            breakdown[h.mismatches] = breakdown.get(h.mismatches, 0) + 1

        return BenchmarkResult(
            engine_name="Pure Python CPU Baseline",
            total_base_pairs=total_bp,
            elapsed_seconds=elapsed,
            throughput_mbps=throughput,
            hits_found=len(all_hits),
            hits_by_mismatch=breakdown,
            hits=all_hits,
        )
