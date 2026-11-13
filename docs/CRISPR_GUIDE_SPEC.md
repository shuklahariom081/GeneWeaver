# CRISPR-Cas9 Target & Off-Target Biological Specification

## 1. Overview
The *Streptococcus pyogenes* Cas9 (SpCas9) endonuclease is guided by a chimeric single guide RNA (sgRNA) to introduce double-stranded breaks (DSBs) at designated genomic loci. The target site recognition requires both RNA-DNA sequence complementarity and the presence of a conserved Protospacer Adjacent Motif (PAM).

---

## 2. Target Site Geometry (23 bp)

```
5' - [ N1  N2  N3  ...  N19 N20 ] [ N21 G22 G23 ] - 3'
     |------ 20 bp Spacer ------| |-- 3 bp PAM -|
     Distal Region (13-20)  Seed Region (1-12)
```

* **Spacer Region (20 bp):** Complementary to the 5' segment of the guide RNA.
* **PAM Motif (3 bp):** Located directly 3' of the target protospacer on the non-target DNA strand.
  * **Canonical PAM:** `5'-NGG-3'` (where N is any nucleotide: A, C, G, T). Responsible for >90% of cleavage events.
  * **Non-Canonical PAM:** `5'-NAG-3'` (and rarely `5'-NGA-3'`). Displays approximately 10-20% relative cleavage efficiency compared to `NGG`.

---

## 3. Seed vs. Distal Region Biology

Empirical biochemistry demonstrates that mutations within the target site are not penalized equally:

1. **Seed Region (Bases 1 to 12 from PAM):**
   * Proximal to the RuvC and HNH nuclease cleavage domains.
   * Single mismatches in this region disrupt Cas9 activation kinetics and severely reduce cutting efficiency.
   * Off-target candidates with seed mismatches are classified with lower cleavage probability.

2. **Distal / Non-Seed Region (Bases 13 to 20 from PAM):**
   * Distal from the PAM interaction groove.
   * Tolerates single and multiple mismatches while retaining considerable cleavage activity.
   * Off-target candidates with exclusively distal mismatches represent high-risk mutagenic hazards.

---

## 4. Genomic Strand Coordinate Representation

* **Forward Strand (+):**
  * Target site read directly $5' \to 3'$ along the reference chromosome sequence.
  * Start coordinate represents the first base of the 20 bp spacer.
* **Reverse Complement Strand (-):**
  * Target site located on the anti-sense strand.
  * PAM sequence occurs on the opposite strand ($5'-CCN-3'$ on reference).
