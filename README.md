# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

> **Domain:** Bioinformatics & High-Performance Computing (HPC)  
> **Target:** NVIDIA CUDA Acceleration via Numba, BioPython, Dask, Textual TUI

---

## 🧬 Overview

**GeneWeaver** is a high-performance genomic search engine designed to scan massive DNA datasets (up to human genome scale, ~3.2 billion base pairs) for unintended CRISPR-Cas9 "off-target" mutations in minutes rather than days.

By replacing standard single-threaded Python string matching loops with Just-In-Time (JIT) compiled CUDA kernels running on NVIDIA GPU hardware, GeneWeaver achieves 100x+ alignment throughput while streaming data to prevent `CUDA_OUT_OF_MEMORY` faults.

---

## ⚡ Key Modules

* **Genomic Parser (`engine/parser.py`)**: Streams and cleanses large `.fasta` files using BioPython and converts sequences into compact numeric arrays (`uint8` / 2-bit encoding).
* **CPU Baseline Engine (`engine/cpu_baseline.py`)**: Reference sliding-window string alignment algorithm in pure Python to measure baseline throughput (Mbp/s).
* **CUDA Kernel Engine (`engine/cuda_kernels.py`)**: Custom Numba `@cuda.jit` kernels executing massively parallel base-pair comparisons across GPU threads and blocks.
* **CUDA Shared Memory Optimizer (`engine/cuda_shared_mem.py`)**: Cooperative tile loading into on-chip shared memory cache to minimize global VRAM latency.
* **Biological Scoring Matrix (`engine/scoring.py`)**: Distance-weighted mismatch penalties (seed region vs. distal region) and PAM motif recognition (`NGG` / `NAG`).
* **Distributed Scheduler (`engine/distributed.py`)**: Dask orchestration for chunk partitioning across CPU/GPU workers.
* **TUI Dashboard (`ui/app.py`)**: Terminal User Interface built with Textual displaying live VRAM allocation, GPU core utilization, throughput, and sequence mismatch viewers.

---

## 🛠️ System Requirements

* **OS:** Windows 10/11 or Linux
* **GPU:** NVIDIA GPU with Compute Capability $\ge$ 6.0 (Ampere RTX 3050+ recommended)
* **Python:** 3.11+
* **Dependencies:** `numba`, `numba-cuda`, `biopython`, `textual`, `dask`, `rich`, `numpy`

---

## 🚀 Quickstart

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run GeneWeaver
python main.py
```
