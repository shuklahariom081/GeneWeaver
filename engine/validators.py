"""
GeneWeaver - Sequence & Input Validation Module
Validates genomic inputs, guide RNA lengths, IUPAC nucleotide compliance,
and FASTA structural integrity.
"""

import re
from typing import Tuple

VALID_DNA_PATTERN = re.compile(r"^[ACGTNacgtn]+$")
IUPAC_EXTENDED_PATTERN = re.compile(r"^[ACGTNRYKMSWBDHV]+$", re.IGNORECASE)


class ValidationError(ValueError):
    """Raised when sequence data or CRISPR parameters fail biological validation."""
    pass


def validate_guide_sequence(guide: str, expected_length: int = 20) -> str:
    """
    Validate and cleanse a CRISPR guide RNA spacer sequence.
    
    Args:
        guide: Raw guide sequence string (e.g. 'GAGTCCGAGCAGAAGAAGAA').
        expected_length: Expected spacer length (default 20 bp for SpCas9).
        
    Returns:
        Cleaned, uppercase guide sequence string.
        
    Raises:
        ValidationError: If length is invalid or contains non-nucleotide characters.
    """
    if not guide or not isinstance(guide, str):
        raise ValidationError("Guide sequence must be a non-empty string.")

    cleaned = guide.strip().upper().replace("U", "T")  # Convert RNA uracil to DNA thymine

    if len(cleaned) != expected_length:
        raise ValidationError(
            f"Invalid guide length: expected {expected_length} bp, received {len(cleaned)} bp ('{cleaned}')"
        )

    if not VALID_DNA_PATTERN.match(cleaned):
        invalid_chars = set(cleaned) - set("ACGTN")
        raise ValidationError(f"Guide contains illegal nucleotide characters: {invalid_chars}")

    return cleaned


def validate_pam_motif(pam: str, allowed_motifs: Tuple[str, ...] = ("NGG", "NAG")) -> bool:
    """
    Check if a 3 bp PAM sequence conforms to target Cas9 motifs.
    """
    pam = pam.strip().upper()
    if len(pam) != 3:
        return False

    for motif in allowed_motifs:
        match = True
        for p_char, m_char in zip(pam, motif):
            if m_char == "N":
                continue
            if p_char != m_char:
                match = False
                break
        if match:
            return True

    return False


def sanitize_genomic_header(header: str) -> str:
    """Cleanse FASTA header strings, extracting clean chromosome/contig identifiers."""
    cleaned = header.strip().lstrip(">").split()[0]
    # Remove filesystem-unsafe characters
    return re.sub(r"[^\w\-.]", "_", cleaned)
