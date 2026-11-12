"""
GeneWeaver - Genomic Parser & Chunker
Streams and cleanses massive .fasta files using BioPython,
encoding nucleobases into compact uint8 arrays with window overlap preservation.
"""

from dataclasses import dataclass
import os
from typing import Generator, Tuple
from Bio import SeqIO
import numpy as np

# Numeric base mapping for GPU and CPU memory efficiency
# A: 0, C: 1, G: 2, T: 3, N/Unknown: 4
BASE_MAP = {
    ord("A"): 0, ord("a"): 0,
    ord("C"): 1, ord("c"): 1,
    ord("G"): 2, ord("g"): 2,
    ord("T"): 3, ord("t"): 3,
    ord("N"): 4, ord("n"): 4,
}

# Lookup table (256-byte ASCII array for blazingly fast vectorized translation)
ENCODE_LUT = np.full(256, fill_value=4, dtype=np.uint8)
for char_code, mapped_val in BASE_MAP.items():
    ENCODE_LUT[char_code] = mapped_val

DECODE_MAP = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


def encode_sequence(seq_str: str) -> np.ndarray:
    """
    Vectorized encoding of a DNA string into a compact uint8 NumPy array.
    'A' -> 0, 'C' -> 1, 'G' -> 2, 'T' -> 3, Ambiguous/N -> 4
    """
    ascii_bytes = np.frombuffer(seq_str.encode("ascii", errors="replace"), dtype=np.uint8)
    return ENCODE_LUT[ascii_bytes]


def decode_sequence(encoded_arr: np.ndarray) -> str:
    """Decode a uint8 encoded NumPy array back into DNA characters."""
    return "".join(DECODE_MAP.get(int(x), "N") for x in encoded_arr)


@dataclass
class GenomicChunk:
    """Represents a chunk of genomic sequence ready for CPU or GPU processing."""
    chromosome: str
    chunk_index: int
    global_start_offset: int
    data: np.ndarray  # uint8 array
    length: int
    overlap_size: int


class FastAStreamParser:
    """
    Streams large FASTA files using BioPython SeqIO, generating memory-efficient
    genomic chunks with overlap buffers to prevent target site truncation.
    """

    def __init__(
        self,
        fasta_path: str,
        chunk_size: int = 100_000,
        overlap_size: int = 22,  # Window size (23) - 1
    ):
        if not os.path.isfile(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        self.fasta_path = fasta_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def stream_chunks(self) -> Generator[GenomicChunk, None, None]:
        """
        Stream chromosomes and slice them into overlapping uint8 encoded chunks.
        Never loads entire gigabyte chromosomes into Python memory at once.
        """
        with open(self.fasta_path, "r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                chrom_id = record.id
                raw_seq = str(record.seq).upper()
                total_len = len(raw_seq)

                chunk_idx = 0
                step = self.chunk_size - self.overlap_size

                for start in range(0, total_len, step):
                    end = min(start + self.chunk_size, total_len)
                    subseq = raw_seq[start:end]
                    encoded_chunk = encode_sequence(subseq)

                    yield GenomicChunk(
                        chromosome=chrom_id,
                        chunk_index=chunk_idx,
                        global_start_offset=start,
                        data=encoded_chunk,
                        length=len(encoded_chunk),
                        overlap_size=self.overlap_size,
                    )

                    chunk_idx += 1
                    if end == total_len:
                        break

    def get_summary(self) -> dict:
        """Read headers and sequence lengths without parsing full contents."""
        summary = {"chromosomes": []}
        total_bp = 0
        with open(self.fasta_path, "r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                chrom_len = len(record.seq)
                total_bp += chrom_len
                summary["chromosomes"].append({"id": record.id, "length": chrom_len})
        summary["total_base_pairs"] = total_bp
        return summary
