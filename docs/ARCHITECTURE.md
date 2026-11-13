# GeneWeaver High-Performance Computing Architecture

## 1. System Pipeline Overview

```
                      +-----------------------------+
                      |   Human Genome FASTA File   |
                      |   (3.2 Billion Base Pairs)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  BioPython Streaming Parser |
                      |    (FastAStreamParser)      |
                      +--------------+--------------+
                                     |
                         [Chunks of 100k - 1M bp]
                       (with 22bp boundary overlap)
                                     |
                                     v
                      +-----------------------------+
                      |   uint8 Vectorized DNA LUT  |
                      |   A:0, C:1, G:2, T:3, N:4   |
                      +--------------+--------------+
                                     |
                     Host-to-Device Memory Transfer
                                     |
                                     v
                +-----------------------------------------+
                |     NVIDIA RTX 3050 CUDA 1D Kernel      |
                |   - 2048 Cores across 16 Ampere SMs     |
                |   - 1 Thread per Base Pair Offset       |
                |   - PAM Fast Pruning & Spacer Matching  |
                |   - Atomic Output Slot Reservation      |
                +--------------------+--------------------+
                                     |
                     Device-to-Host Result Transfer
                                     |
                                     v
                +-----------------------------------------+
                |        Textual TUI Dashboard            |
                |   - Real-time VRAM Allocation Gauge     |
                |   - Live Genomic Progress & Throughput  |
                |   - Mismatch Highlighter & Inspector    |
                +-----------------------------------------+
```

## 2. Memory Ingestion & Chunk Boundary Preservation

When chunking genomic sequences for a $K$-mer window of length $W$ (where $W = 23$ for Cas9), slicing a sequence naively at index $C$ causes any CRISPR target site that straddles the boundary $C - k$ to be truncated and missed.

GeneWeaver implements an overlap buffer $\Delta = W - 1 = 22 \text{ bp}$.

For chunk $i$ spanning $[S_i, E_i]$, chunk $i + 1$ begins at $S_{i+1} = E_i - \Delta$. This mathematical formulation guarantees zero boundary false negatives while maintaining linear time complexity.
