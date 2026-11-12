"""
GeneWeaver - Mock Genome Generator
Generates synthetic .fasta files mimicking human chromosomal data
with embedded CRISPR-Cas9 on-target and off-target sites.
"""

import json
import os
import random
from typing import Dict, List, Tuple

# Standard Cas9 PAM is 5'-NGG-3'
BASES = ["A", "C", "G", "T"]
BASE_WEIGHTS = [0.295, 0.205, 0.205, 0.295]  # Realistic human AT/GC ratio (~41% GC)


def generate_random_sequence(length: int) -> str:
    """Generate a random DNA sequence of specified length."""
    return "".join(random.choices(BASES, weights=BASE_WEIGHTS, k=length))


def mutate_spacer(spacer: str, num_mutations: int) -> Tuple[str, List[int]]:
    """
    Introduce exact number of point mutations into a 20bp spacer.
    Returns the mutated spacer and the mutated 0-based indices.
    """
    spacer_list = list(spacer)
    indices = sorted(random.sample(range(len(spacer)), k=num_mutations))
    for idx in indices:
        original = spacer_list[idx]
        alternatives = [b for b in BASES if b != original]
        spacer_list[idx] = random.choice(alternatives)
    return "".join(spacer_list), indices


def generate_mock_genome(
    output_fasta: str,
    output_ground_truth: str,
    total_base_pairs: int = 1_000_000,
    chromosome_name: str = "chr1_mock",
    guide_sequences: List[str] = None,
    num_off_targets_per_guide: int = 10,
    max_mismatches: int = 4,
    random_seed: int = 42,
) -> Dict:
    """
    Generate a mock FASTA file containing injected CRISPR target and off-target sites.
    
    Args:
        output_fasta: File path for the generated .fasta file.
        output_ground_truth: JSON file path storing positions of injected off-targets.
        total_base_pairs: Total length of chromosome sequence to generate.
        chromosome_name: Header label for FASTA record.
        guide_sequences: List of 20bp guide RNA target sequences (default: EMX1 & VEGF-A guides).
        num_off_targets_per_guide: How many off-targets to inject per guide.
        max_mismatches: Maximum allowable mutations in spacer region (1 to 4).
        random_seed: Seed for reproducible synthetic datasets.
    """
    random.seed(random_seed)

    # Standard well-characterized CRISPR guides (EMX1 & HEK293 site 4)
    if not guide_sequences:
        guide_sequences = [
            "GAGTCCGAGCAGAAGAAGAA",  # Human EMX1 guide
            "GGCACTGCGGCGGAGGTGGG",  # Human VEGFA site 1
        ]

    # Generate baseline random genomic sequence
    raw_seq = list(generate_random_sequence(total_base_pairs))

    # Space injection sites across the chromosome safely
    ground_truth = []
    min_spacing = 2000
    available_positions = list(range(1000, total_base_pairs - 1000, min_spacing))
    random.shuffle(available_positions)

    for guide in guide_sequences:
        guide = guide.upper()
        assert len(guide) == 20, "Guide spacer must be exactly 20 bp"

        # 1. Inject exact On-Target site (0 mismatches + canonical NGG PAM)
        on_target_pos = available_positions.pop()
        pam = random.choice(["AGG", "CGG", "GGG", "TGG"])
        full_on_target = guide + pam
        raw_seq[on_target_pos : on_target_pos + 23] = list(full_on_target)

        ground_truth.append({
            "chromosome": chromosome_name,
            "position": on_target_pos,
            "guide": guide,
            "target_sequence": full_on_target,
            "spacer_sequence": guide,
            "pam": pam,
            "mismatches": 0,
            "mutated_indices": [],
            "site_type": "on_target",
        })

        # 2. Inject Off-Target sites with 1, 2, 3, or 4 mutations
        for _ in range(num_off_targets_per_guide):
            if not available_positions:
                break
            pos = available_positions.pop()
            mismatch_count = random.randint(1, max_mismatches)
            mutated_spacer, mut_indices = mutate_spacer(guide, mismatch_count)

            # 85% canonical NGG, 15% non-canonical NAG
            if random.random() < 0.85:
                pam = random.choice(["AGG", "CGG", "GGG", "TGG"])
            else:
                pam = random.choice(["AAG", "CAG", "GAG", "TAG"])

            full_off_target = mutated_spacer + pam
            raw_seq[pos : pos + 23] = list(full_off_target)

            ground_truth.append({
                "chromosome": chromosome_name,
                "position": pos,
                "guide": guide,
                "target_sequence": full_off_target,
                "spacer_sequence": mutated_spacer,
                "pam": pam,
                "mismatches": mismatch_count,
                "mutated_indices": mut_indices,
                "site_type": "off_target",
            })

    final_sequence = "".join(raw_seq)

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_fasta)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_ground_truth)), exist_ok=True)

    # Write FASTA formatted file (wrapped at 80 chars per standard)
    with open(output_fasta, "w", encoding="utf-8") as f:
        f.write(f">{chromosome_name} mock_human_genome length={total_base_pairs}bp\n")
        line_length = 80
        for i in range(0, len(final_sequence), line_length):
            f.write(final_sequence[i : i + line_length] + "\n")

    # Write ground-truth JSON
    manifest = {
        "metadata": {
            "chromosome": chromosome_name,
            "total_base_pairs": total_base_pairs,
            "num_injected_sites": len(ground_truth),
            "guides_tested": guide_sequences,
        },
        "sites": ground_truth,
    }
    with open(output_ground_truth, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    fasta_path = os.path.join(os.path.dirname(__file__), "mock_chr1.fasta")
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    print(f"Generating 1,000,000 bp mock genome at: {fasta_path}")
    manifest = generate_mock_genome(fasta_path, gt_path, total_base_pairs=1_000_000)
    print(f"Generated successfully with {manifest['metadata']['num_injected_sites']} ground-truth CRISPR sites.")
