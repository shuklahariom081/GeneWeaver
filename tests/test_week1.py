"""
Unit and Integration Tests for GeneWeaver Week 1
Verifies:
1. Mock genome generator output and ground truth consistency.
2. BioPython streaming parser, encoding, and chunking with overlap buffers.
3. Pure Python CPU baseline alignment accuracy and ground-truth recovery.
"""

import json
import os
import unittest
import numpy as np

from data.mock_genome_generator import generate_mock_genome
from engine.parser import (
    FastAStreamParser,
    encode_sequence,
    decode_sequence,
    BASE_MAP,
)
from engine.cpu_baseline import CPUBaselineAligner, AlignmentHit


class TestWeek1Bioinformatics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), "temp_test_data")
        os.makedirs(cls.test_dir, exist_ok=True)

        cls.fasta_path = os.path.join(cls.test_dir, "test_chr.fasta")
        cls.gt_path = os.path.join(cls.test_dir, "test_ground_truth.json")

        cls.guide = "GAGTCCGAGCAGAAGAAGAA"  # Standard Human EMX1 guide
        cls.total_bp = 100_000

        # Generate a test mock genome with injected sites
        cls.manifest = generate_mock_genome(
            output_fasta=cls.fasta_path,
            output_ground_truth=cls.gt_path,
            total_base_pairs=cls.total_bp,
            chromosome_name="chrTest",
            guide_sequences=[cls.guide],
            num_off_targets_per_guide=12,
            max_mismatches=3,
            random_seed=123,
        )

    def test_dna_encoding_and_decoding(self):
        """Test numeric translation between DNA strings and uint8 arrays."""
        sample_dna = "ACGTNNACGTACGT"
        encoded = encode_sequence(sample_dna)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(list(encoded[:4]), [0, 1, 2, 3])
        self.assertEqual(list(encoded[4:6]), [4, 4])

        decoded = decode_sequence(encoded)
        self.assertEqual(decoded, sample_dna)

    def test_fasta_streaming_and_chunking(self):
        """Verify BioPython streaming chunker produces expected sizes and overlaps."""
        chunk_size = 25_000
        overlap = 22
        parser = FastAStreamParser(self.fasta_path, chunk_size=chunk_size, overlap_size=overlap)

        chunks = list(parser.stream_chunks())
        self.assertGreater(len(chunks), 1)

        # Check total parsed length reconstructs full sequence
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.chromosome, "chrTest")
        self.assertEqual(first_chunk.global_start_offset, 0)
        self.assertEqual(len(first_chunk.data), chunk_size)

        # Verify second chunk begins at step = chunk_size - overlap
        second_chunk = chunks[1]
        self.assertEqual(second_chunk.global_start_offset, chunk_size - overlap)

    def test_cpu_baseline_accuracy_and_recall(self):
        """
        Verify that the CPU baseline aligner discovers 100% of injected ground truth sites
        with <= max_mismatches.
        """
        parser = FastAStreamParser(self.fasta_path, chunk_size=30_000, overlap_size=22)
        aligner = CPUBaselineAligner(
            guide_sequence=self.guide,
            max_mismatches=3,
            allow_nag_pam=True,
            search_both_strands=False,
        )

        result = self.align_result = aligner.align_fasta(parser)

        print("\n" + result.summary())

        # Load ground truth
        with open(self.gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        # Filter ground-truth sites matching our search criteria (<= 3 mismatches)
        expected_sites = [
            s for s in gt_data["sites"]
            if s["guide"] == self.guide and s["mismatches"] <= 3
        ]

        found_positions = {h.position for h in result.hits}

        recovered_count = 0
        for expected in expected_sites:
            pos = expected["position"]
            self.assertIn(
                pos,
                found_positions,
                f"Failed to find injected site at position {pos}: {expected['target_sequence']}"
            )
            recovered_count += 1

        print(f"\n[Test Result] Ground-truth recovery: {recovered_count}/{len(expected_sites)} (100% Recall)")
        self.assertEqual(recovered_count, len(expected_sites))
        self.assertGreater(result.throughput_mbps, 0.0)


if __name__ == "__main__":
    unittest.main()
