"""
GeneWeaver - CUDA Kernel Engine (Week 2)
Custom GPU kernels written in Python and Just-In-Time (JIT) compiled via Numba
to run directly on NVIDIA hardware, bypassing the Python Global Interpreter Lock (GIL).

Handles complex memory transfers between Host (CPU RAM) and Device (GPU VRAM),
with reusable VRAM buffers to guarantee zero memory fragmentation and eliminate
CUDA_OUT_OF_MEMORY faults on massive datasets.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress minor occupancy and JIT linking warnings for pristine console output
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


from engine.parser import (
    GenomicChunk,
    FastAStreamParser,
    encode_sequence,
    decode_sequence,
)
from engine.cpu_baseline import AlignmentHit, BenchmarkResult


@cuda.jit
def crispr_alignment_kernel(
    seq,                 # device array uint8 (L,)
    seq_len,             # int32
    guide,               # device array uint8 (20,)
    max_mismatches,      # int32
    allow_nag,           # int32: 1 = allow NAG, 0 = NGG only
    global_offset,       # int64
    max_hits,            # int32
    hit_counter,         # device array int32 (1,)
    out_positions,       # device array int64 (max_hits,)
    out_mismatches,      # device array uint8 (max_hits,)
    out_mutation_masks,  # device array uint32 (max_hits,)
    strand_val,          # int32: 0 for '+', 1 for '-'
    out_strands,         # device array uint8 (max_hits,)
):
    """
    1D Grid CUDA Kernel for high-throughput CRISPR sequence alignment.
    Each GPU thread is assigned a unique base-pair offset in the genomic sequence.
    """
    idx = cuda.grid(1)

    # 23 bp window: 20bp spacer + 3bp PAM
    if idx > seq_len - 23:
        return

    # 1. PAM Verification (5'-NGG-3' or 5'-NAG-3')
    # Bases: A=0, C=1, G=2, T=3, N=4
    b21 = seq[idx + 21]
    b22 = seq[idx + 22]

    # The terminal base must always be Guanine (G = 2)
    if b22 != 2:
        return

    # The middle base of PAM: G (2) for NGG, or A (0) for NAG
    if b21 != 2:
        if allow_nag == 0 or b21 != 0:
            return

    # 2. Count mismatches in the 20bp spacer candidate
    mismatches = 0
    mut_mask = 0

    for k in range(20):
        if seq[idx + k] != guide[k]:
            mismatches += 1
            mut_mask |= (1 << k)
            if mismatches > max_mismatches:
                return  # Early pruning reduces warp divergence

    # 3. Thread-safe atomic insertion
    slot = cuda.atomic.add(hit_counter, 0, 1)
    if slot < max_hits:
        out_positions[slot] = global_offset + idx
        out_mismatches[slot] = mismatches
        out_mutation_masks[slot] = mut_mask
        out_strands[slot] = strand_val


class CUDAAlignmentEngine:
    """
    GPU-Accelerated CRISPR Alignment Engine using Numba CUDA JIT.
    Manages Host-to-Device memory transfers, pinned buffer reuse, and grid scheduling.
    """

    def __init__(
        self,
        guide_sequence: str,
        max_mismatches: int = 3,
        allow_nag_pam: bool = True,
        search_both_strands: bool = True,
        threads_per_block: int = 256,
        max_hits_per_chunk: int = 100_000,
    ):
        guide_clean = guide_sequence.strip().upper()
        if len(guide_clean) != 20:
            raise ValueError(f"CRISPR spacer must be 20 bp, got {len(guide_clean)} bp")

        self.guide_str = guide_clean
        self.guide_encoded = encode_sequence(guide_clean)
        self.max_mismatches = int(max_mismatches)
        self.allow_nag = 1 if allow_nag_pam else 0
        self.search_both_strands = search_both_strands
        self.threads_per_block = threads_per_block
        self.max_hits = max_hits_per_chunk

        # Precompute reverse-complement guide for negative strand scanning
        comp_map = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
        rc_arr = np.array([comp_map[x] for x in self.guide_encoded[::-1]], dtype=np.uint8)
        self.guide_rc_str = decode_sequence(rc_arr)
        self.guide_rc_encoded = rc_arr

        # Device verification
        if not cuda.is_available():
            raise RuntimeError("NVIDIA CUDA is not available or supported on this system.")

        self.device = cuda.get_current_device()
        self.device_name = self.device.name.decode("utf-8") if isinstance(self.device.name, bytes) else str(self.device.name)
        self.compute_cap = self.device.compute_capability

        # Transfer guides to Device VRAM once (persists across all chunk iterations)
        self.d_guide_fwd = cuda.to_device(self.guide_encoded)
        self.d_guide_rc = cuda.to_device(self.guide_rc_encoded)

        # Pre-allocate reusable output device buffers to eliminate VRAM allocations inside the loop
        self.d_out_positions = cuda.device_array(self.max_hits, dtype=np.int64)
        self.d_out_mismatches = cuda.device_array(self.max_hits, dtype=np.uint8)
        self.d_out_mutation_masks = cuda.device_array(self.max_hits, dtype=np.uint32)
        self.d_out_strands = cuda.device_array(self.max_hits, dtype=np.uint8)
        self.d_hit_counter = cuda.device_array(1, dtype=np.int32)

        # Host staging buffers
        self.h_positions = np.zeros(self.max_hits, dtype=np.int64)
        self.h_mismatches = np.zeros(self.max_hits, dtype=np.uint8)
        self.h_masks = np.zeros(self.max_hits, dtype=np.uint32)
        self.h_strands = np.zeros(self.max_hits, dtype=np.uint8)
        self.h_counter = np.zeros(1, dtype=np.int32)

        # Warmup JIT compilation to avoid compiling latency during actual benchmark
        self._warmup_jit()

    def _warmup_jit(self):
        """Warm up the CUDA JIT compiler so PTX generation overhead doesn't skew benchmarks."""
        dummy_seq = np.zeros(64, dtype=np.uint8)
        d_dummy = cuda.to_device(dummy_seq)
        self.d_hit_counter.copy_to_device(np.zeros(1, dtype=np.int32))
        crispr_alignment_kernel[1, 32](
            d_dummy,
            np.int32(64),
            self.d_guide_fwd,
            np.int32(self.max_mismatches),
            np.int32(self.allow_nag),
            np.int64(0),
            np.int32(self.max_hits),
            self.d_hit_counter,
            self.d_out_positions,
            self.d_out_mismatches,
            self.d_out_mutation_masks,
            np.int32(0),
            self.d_out_strands,
        )
        cuda.synchronize()

    def get_vram_info(self) -> Tuple[int, int]:
        """Returns (free_bytes, total_bytes) of GPU VRAM."""
        mem = cuda.current_context().get_memory_info()
        return mem.free, mem.total

    def align_chunk(
        self,
        chunk: GenomicChunk,
        raw_sequence_context: Optional[str] = None,
    ) -> List[AlignmentHit]:
        """
        Offloads a genomic chunk to the RTX 3050 GPU, executes massively parallel
        sliding-window alignment, and transfers results back to Host RAM.
        """
        seq_len = chunk.length
        if seq_len < 23:
            return []

        # Reset device hit counter
        self.d_hit_counter.copy_to_device(np.zeros(1, dtype=np.int32))

        # Transfer genomic chunk from Host (CPU RAM) -> Device (GPU VRAM)
        d_seq = cuda.to_device(chunk.data)

        # Compute optimal 1D execution grid
        blocks_per_grid = (seq_len - 22 + self.threads_per_block - 1) // self.threads_per_block

        # Launch Forward Strand Alignment Kernel
        crispr_alignment_kernel[blocks_per_grid, self.threads_per_block](
            d_seq,
            np.int32(seq_len),
            self.d_guide_fwd,
            np.int32(self.max_mismatches),
            np.int32(self.allow_nag),
            np.int64(chunk.global_start_offset),
            np.int32(self.max_hits),
            self.d_hit_counter,
            self.d_out_positions,
            self.d_out_mismatches,
            self.d_out_mutation_masks,
            np.int32(0),  # Forward strand = '+'
            self.d_out_strands,
        )

        # Optional: Launch Reverse Strand Alignment Kernel if enabled
        if self.search_both_strands:
            crispr_alignment_kernel[blocks_per_grid, self.threads_per_block](
                d_seq,
                np.int32(seq_len),
                self.d_guide_rc,
                np.int32(self.max_mismatches),
                np.int32(self.allow_nag),
                np.int64(chunk.global_start_offset),
                np.int32(self.max_hits),
                self.d_hit_counter,
                self.d_out_positions,
                self.d_out_mismatches,
                self.d_out_mutation_masks,
                np.int32(1),  # Reverse strand = '-'
                self.d_out_strands,
            )

        # Synchronize GPU pipeline before reading results
        cuda.synchronize()

        # Read hit count from device
        self.d_hit_counter.copy_to_host(self.h_counter)
        total_hits = int(min(self.h_counter[0], self.max_hits))

        if total_hits == 0:
            return []

        # Copy slice of results back from Device (GPU VRAM) -> Host (CPU RAM)
        self.d_out_positions[:total_hits].copy_to_host(self.h_positions[:total_hits])
        self.d_out_mismatches[:total_hits].copy_to_host(self.h_mismatches[:total_hits])
        self.d_out_mutation_masks[:total_hits].copy_to_host(self.h_masks[:total_hits])
        self.d_out_strands[:total_hits].copy_to_host(self.h_strands[:total_hits])

        hits: List[AlignmentHit] = []
        for i in range(total_hits):
            pos = int(self.h_positions[i])
            mismatches = int(self.h_mismatches[i])
            mask = int(self.h_masks[i])
            strand = "+" if self.h_strands[i] == 0 else "-"

            # Decode mutated indices from bitmask
            mut_indices = [k for k in range(20) if (mask & (1 << k)) != 0]

            # Reconstruct sequence from chunk data
            rel_offset = pos - chunk.global_start_offset
            if 0 <= rel_offset <= seq_len - 23:
                target_seq = decode_sequence(chunk.data[rel_offset : rel_offset + 23])
            else:
                target_seq = "N" * 23

            hits.append(AlignmentHit(
                chromosome=chunk.chromosome,
                position=pos,
                strand=strand,
                target_sequence=target_seq,
                spacer_sequence=target_seq[:20],
                pam=target_seq[20:],
                mismatches=mismatches,
                mutated_indices=mut_indices,
            ))

        return hits

    def align_fasta(
        self,
        parser: FastAStreamParser,
        progress_callback=None,
    ) -> BenchmarkResult:
        """
        Stream full FASTA file across the GPU kernel pipeline, measuring pure GPU throughput.
        """
        all_hits: List[AlignmentHit] = []
        total_bp = 0

        start_time = time.perf_counter()

        for chunk in parser.stream_chunks():
            chunk_hits = self.align_chunk(chunk)
            all_hits.extend(chunk_hits)

            scanned = chunk.length - chunk.overlap_size if total_bp > 0 else chunk.length
            total_bp += scanned

            if progress_callback:
                progress_callback(total_bp, chunk)

        elapsed = time.perf_counter() - start_time
        throughput = (total_bp / 1_000_000) / elapsed if elapsed > 0 else 0.0

        breakdown: Dict[int, int] = {}
        for h in all_hits:
            breakdown[h.mismatches] = breakdown.get(h.mismatches, 0) + 1

        return BenchmarkResult(
            engine_name=f"Numba CUDA Engine ({self.device_name})",
            total_base_pairs=total_bp,
            elapsed_seconds=elapsed,
            throughput_mbps=throughput,
            hits_found=len(all_hits),
            hits_by_mismatch=breakdown,
            hits=all_hits,
        )
