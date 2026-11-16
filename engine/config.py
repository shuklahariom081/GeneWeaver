"""
GeneWeaver - Centralized System Configuration Module
Configures default biological constants, GPU kernel block sizing,
memory safety margins, and file chunking parameters.
"""

from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, Any


@dataclass
class EngineConfig:
    # Biological Constants
    spacer_length: int = 20
    pam_length: int = 3
    window_length: int = 23
    canonical_pam: str = "NGG"
    non_canonical_pam: str = "NAG"
    default_max_mismatches: int = 3

    # HPC & GPU Kernel Parameters
    threads_per_block: int = 256
    max_hits_per_chunk: int = 100_000
    vram_safety_margin_mb: int = 512
    default_chunk_size: int = 250_000
    chunk_overlap_size: int = 22

    # Scoring Matrix Presets (CFD / MIT foundations)
    seed_region_start: int = 8    # 1-based offset from 5' end (bases 9-20 proximal to PAM)
    seed_region_end: int = 20
    seed_mismatch_penalty_multiplier: float = 2.5
    distal_mismatch_penalty_multiplier: float = 1.0
    nag_pam_penalty_factor: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        """Save configuration preset to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "EngineConfig":
        """Load configuration preset from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# Default global instance
DEFAULT_CONFIG = EngineConfig()
