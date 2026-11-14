"""
Unit tests for GeneWeaver Sequence & Input Validation Module
"""

import unittest
from engine.validators import (
    validate_guide_sequence,
    validate_pam_motif,
    sanitize_genomic_header,
    ValidationError,
)


class TestValidators(unittest.TestCase):

    def test_valid_guide_rna(self):
        """Test standard 20bp guide validation and RNA U->T conversion."""
        raw_guide = "GAGTCCGAGCAGAAGAAGAA"
        cleaned = validate_guide_sequence(raw_guide)
        self.assertEqual(cleaned, raw_guide)

        # Test RNA conversion (U -> T)
        rna_guide = "GAGUCCGAGCAGAAGAAGAA"
        cleaned_rna = validate_guide_sequence(rna_guide)
        self.assertEqual(cleaned_rna, raw_guide)

    def test_invalid_guide_lengths(self):
        """Test rejection of truncated or overly long spacer sequences."""
        with self.assertRaises(ValidationError):
            validate_guide_sequence("GAGTCC")  # Too short

        with self.assertRaises(ValidationError):
            validate_guide_sequence("GAGTCCGAGCAGAAGAAGAAGAA")  # Too long (23 bp)

    def test_illegal_characters(self):
        """Test rejection of non-nucleotide characters."""
        with self.assertRaises(ValidationError):
            validate_guide_sequence("GAGTCCGAGCAGAAGAAG12")

        with self.assertRaises(ValidationError):
            validate_guide_sequence("GAGTCCGAGCAG-AGAAGAA")

    def test_pam_motif_matching(self):
        """Test validation of canonical and non-canonical Cas9 PAM sequences."""
        self.assertTrue(validate_pam_motif("AGG"))
        self.assertTrue(validate_pam_motif("CGG"))
        self.assertTrue(validate_pam_motif("GGG"))
        self.assertTrue(validate_pam_motif("TGG"))
        self.assertTrue(validate_pam_motif("AAG"))  # NAG
        self.assertTrue(validate_pam_motif("CAG"))  # NAG

        # Negative tests
        self.assertFalse(validate_pam_motif("TGA"))
        self.assertFalse(validate_pam_motif("ATC"))
        self.assertFalse(validate_pam_motif("GG"))

    def test_header_sanitization(self):
        """Test FASTA header cleaning."""
        raw_header = ">chr1_GRCh38.p14 primary assembly"
        clean = sanitize_genomic_header(raw_header)
        self.assertEqual(clean, "chr1_GRCh38.p14")


if __name__ == "__main__":
    unittest.main()
