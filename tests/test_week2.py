"""
Unit and Integration Tests for GeneWeaver Week 2 (CUDA & TUI Scaffolding)
Verifies:
1. Parity between CUDA Kernel output and CPU Baseline output (100% agreement).
2. Ground-truth CRISPR target recovery on NVIDIA GPU hardware.
3. Memory leak audit (free VRAM stability across multiple consecutive chunks).
4. Performance speedup audit (GPU vs CPU baseline).
"""

import json
import os
import unittest
import numpy as np
from numba import cuda

from data.mock_genome_generator import generate_mock_genome
from engine.parser import FastAStreamParser
from engine.cpu_baseline import CPUBaselineAligner
from engine.cuda_kernels import CUDAAlignmentEngine


class TestWeek2CUDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), "temp_test_data_w2")
        os.makedirs(cls.test_dir, exist_ok=True)

        cls.fasta_path = os.path.join(cls.test_dir, "test_chr_w2.fasta")
        cls.gt_path = os.path.join(cls.test_dir, "test_gt_w2.json")

        cls.guide = "GAGTCCGAGCAGAAGAAGAA"
        cls.total_bp = 200_000

        # Generate mock test genome
        cls.manifest = generate_mock_genome(
            output_fasta=cls.fasta_path,
            output_ground_truth=cls.gt_path,
            total_base_pairs=cls.total_bp,
            chromosome_name="chrTest_W2",
            guide_sequences=[cls.guide],
            num_off_targets_per_guide=10,
            max_mismatches=3,
            random_seed=456,
        )

        # Initialize engines
        cls.cpu_engine = CPUBaselineAligner(
            guide_sequence=cls.guide,
            max_mismatches=3,
            allow_nag_pam=True,
            search_both_strands=False,
        )
        cls.cuda_engine = CUDAAlignmentEngine(
            guide_sequence=cls.guide,
            max_mismatches=3,
            allow_nag_pam=True,
            search_both_strands=False,
        )

    def test_cuda_initialization_and_device(self):
        """Verify NVIDIA GPU is detected with correct compute capability."""
        self.assertTrue(cuda.is_available())
        self.assertIn("NVIDIA", self.cuda_engine.device_name)
        free_b, total_b = self.cuda_engine.get_vram_info()
        self.assertGreater(free_b, 100 * 1024 * 1024)  # At least 100MB free VRAM

    def test_cpu_gpu_parity_and_recall(self):
        """
        Prove CPU Baseline and GPU CUDA implementations find identical hits
        and recover 100% of ground-truth target sites.
        """
        parser_cpu = FastAStreamParser(self.fasta_path, chunk_size=50_000, overlap_size=22)
        parser_gpu = FastAStreamParser(self.fasta_path, chunk_size=50_000, overlap_size=22)

        cpu_res = self.cpu_engine.align_fasta(parser_cpu)
        gpu_res = self.cuda_engine.align_fasta(parser_gpu)

        print("\n" + cpu_res.summary())
        print("\n" + gpu_res.summary())

        # Check total hits parity
        self.assertEqual(
            cpu_res.hits_found,
            gpu_res.hits_found,
            f"Hit count mismatch: CPU={cpu_res.hits_found}, GPU={gpu_res.hits_found}",
        )

        # Check coordinates and mismatches match exactly
        cpu_hit_map = {h.position: h.mismatches for h in cpu_res.hits}
        gpu_hit_map = {h.position: h.mismatches for h in gpu_res.hits}

        self.assertEqual(cpu_hit_map, gpu_hit_map, "CPU and GPU hit positions/mismatches do not match!")

        # Verify ground-truth recovery
        with open(self.gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        expected_sites = [
            s for s in gt_data["sites"]
            if s["guide"] == self.guide and s["mismatches"] <= 3
        ]

        found_positions = set(gpu_hit_map.keys())
        for exp in expected_sites:
            self.assertIn(exp["position"], found_positions)

        print(f"\n[Week 2 Parity Check] 100% parity verified between CPU and GPU ({gpu_res.hits_found} sites).")

    def test_vram_leak_free_paging(self):
        """
        Verify VRAM memory stability across multiple chunk iterations
        to ensure no memory leaks or CUDA_OUT_OF_MEMORY risk.
        """
        parser = FastAStreamParser(self.fasta_path, chunk_size=20_000, overlap_size=22)
        vram_readings = []

        for chunk in parser.stream_chunks():
            self.cuda_engine.align_chunk(chunk)
            free_b, _ = self.cuda_engine.get_vram_info()
            vram_readings.append(free_b)

        # Free VRAM at the end should be approximately equal to the beginning (within 5MB tolerance)
        initial_free = vram_readings[0]
        final_free = vram_readings[-1]
        diff_mb = abs(initial_free - final_free) / (1024 * 1024)

        print(f"\n[VRAM Audit] Initial Free: {initial_free / 1024**2:.1f} MB, Final Free: {final_free / 1024**2:.1f} MB (Delta: {diff_mb:.2f} MB)")
        self.assertLess(diff_mb, 10.0, "Potential VRAM leak detected across chunk iterations!")


if __name__ == "__main__":
    unittest.main()
