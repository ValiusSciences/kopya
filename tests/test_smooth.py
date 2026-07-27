"""Unit tests for smooth.py.

Synthetic fixtures plant a clean chr2 amplification across "tumor" cells with
"normal" cells uniformly diploid; centering must zero out normal cells (modulo
noise) and elevate tumor cells on the amplified chromosome.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from kopya.smooth import (
    DEFAULT_SMOOTH_WINDOW,
    center_against_baseline,
    smooth_along_chromosomes,
)


def _three_chr_adata(
    n_normal=30,
    n_tumor=30,
    genes_per_chr=(150, 150, 100),
    amp_chr=1,            # which chromosome (index into genes_per_chr) is amplified
    amp_strength=0.6,     # log1p-space additive shift on the amped chromosome
    noise=0.05,
    seed=0,
):
    """Build a small AnnData with 3 synthetic chromosomes and a planted amp.

    Genes are ordered chr1 (150) | chr2 (150) | chr3 (100). All cells get
    baseline log1p expression around 1.0 with small Gaussian noise. The first
    n_normal cells are diploid everywhere; the next n_tumor cells get +amp_strength
    added across all genes on the amped chromosome — a clean broad-arm gain
    that segmentation should recover as a single segment.

    Args:
        n_normal / n_tumor: cell counts per planted group.
        genes_per_chr: tuple of three ints; chromosome layout.
        amp_chr: index into genes_per_chr (0/1/2) for the planted gain.
        amp_strength: additive shift applied to tumor cells on amp_chr.
        noise: per-entry Gaussian noise stddev.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple (adata, normal_mask, amp_chr_label, chr_starts).
        chr_starts maps chr label -> gene-axis start index.
    """
    rng = np.random.default_rng(seed)

    n_cells = n_normal + n_tumor
    n_genes = sum(genes_per_chr)

    # Gene order: synthetic "chr_synth_1/2/3" labels in adata.var.chr.
    chr_labels = []
    starts = {}
    offset = 0
    for i, n in enumerate(genes_per_chr):
        label = f"chr_synth_{i + 1}"
        starts[label] = offset
        chr_labels += [label] * n
        offset += n

    # Baseline: every cell at ~1.0 across every gene; small Gaussian noise.
    X = rng.normal(loc=1.0, scale=noise, size=(n_cells, n_genes)).astype(np.float64)

    # Plant the amp: tumor cells get +amp_strength across the amped chromosome.
    amp_label = f"chr_synth_{amp_chr + 1}"
    amp_start = starts[amp_label]
    amp_end = amp_start + genes_per_chr[amp_chr]
    X[n_normal:, amp_start:amp_end] += amp_strength

    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(
        {"chr": chr_labels},
        index=[f"GENE{i:05d}" for i in range(n_genes)],
    )

    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    normal_mask = np.array([True] * n_normal + [False] * n_tumor)
    return adata, normal_mask, amp_label, starts


def test_centering_zeros_normal_pool():
    """Normal cells, post-centering, average to ~0 (the baseline median by construction)."""
    adata, normal_mask, amp_label, _ = _three_chr_adata(amp_strength=0.6)

    centered = center_against_baseline(adata, normal_mask)

    # Median over normals is the per-gene baseline, so centered normals must
    # have a per-gene median of 0 by construction. Cohort median (over cells)
    # is the right summary — robust to the Gaussian noise per entry.
    normal_block = centered[normal_mask, :]
    per_gene_median = np.median(normal_block, axis=0)
    np.testing.assert_allclose(per_gene_median, 0.0, atol=1e-5)


def test_centering_elevates_tumor_on_amped_chr():
    """Tumor cells, post-centering, are clearly above 0 on the amped chromosome."""
    adata, normal_mask, amp_label, starts = _three_chr_adata(amp_strength=0.6)

    centered = center_against_baseline(adata, normal_mask)

    # Slice the amped chromosome columns.
    amp_idx = adata.var["chr"] == amp_label
    tumor_amp_block = centered[~normal_mask, :][:, amp_idx.values]

    # Tumor cells should sit close to the planted +0.6 shift on the amped chr.
    # Tolerance accounts for the small Gaussian noise per entry.
    mean_amp = tumor_amp_block.mean()
    assert 0.5 < mean_amp < 0.7, f"unexpected amp magnitude: {mean_amp:.3f}"

    # And the un-amped chromosomes should remain near 0 for tumor cells.
    non_amp_idx = ~amp_idx
    tumor_non_amp = centered[~normal_mask, :][:, non_amp_idx.values]
    mean_non_amp = tumor_non_amp.mean()
    assert abs(mean_non_amp) < 0.05, f"unexpected non-amp drift: {mean_non_amp:.3f}"


def test_centering_handles_empty_normal_mask():
    """An empty normal_mask falls back to cohort-median centering without crashing."""
    adata, _, _, _ = _three_chr_adata()
    empty_mask = np.zeros(adata.n_obs, dtype=bool)

    centered = center_against_baseline(adata, empty_mask)

    # No NaNs; shape preserved. The exact values are not meaningful here,
    # just that the fallback path executes.
    assert centered.shape == adata.shape
    assert not np.any(np.isnan(centered))


def test_smoothing_preserves_shape_and_dtype():
    """smooth_along_chromosomes returns same shape + dtype as input."""
    adata, normal_mask, _, _ = _three_chr_adata()
    centered = center_against_baseline(adata, normal_mask)

    smoothed = smooth_along_chromosomes(
        centered,
        chr_labels=adata.var["chr"].to_numpy(),
        window=20,
    )

    assert smoothed.shape == centered.shape
    assert smoothed.dtype == centered.dtype


def test_smoothing_no_cross_chromosome_leak():
    """Smoothing windows are clipped at chromosome boundaries.

    Plant a strong amp on chr_synth_2 only; the first few genes of chr_synth_3
    must not pick up the amp signal via window bleed-through (we use
    mode='nearest' specifically to prevent this).
    """
    adata, normal_mask, amp_label, starts = _three_chr_adata(amp_strength=1.0, noise=0.0)
    centered = center_against_baseline(adata, normal_mask)
    smoothed = smooth_along_chromosomes(
        centered,
        chr_labels=adata.var["chr"].to_numpy(),
        window=20,
    )

    # The first 5 genes of the chromosome AFTER the amped one (chr_synth_3
    # follows chr_synth_2 in the fixture's gene order). Compute tumor-cell
    # mean there and verify it is near zero.
    next_chr_start = starts["chr_synth_3"]
    leakage_block = smoothed[~normal_mask, next_chr_start : next_chr_start + 5]
    leakage_mean = leakage_block.mean()

    # Without nearest-mode clamping the boundary genes would pull from the
    # amped chr2 signal and read ~+0.5; clipped, they read ~0.
    assert abs(leakage_mean) < 0.1, f"cross-chr leak detected: {leakage_mean:.3f}"


def test_smoothing_recovers_planted_amp():
    """The smoothed signal across the amped chromosome should track the +amp shift."""
    adata, normal_mask, amp_label, starts = _three_chr_adata(amp_strength=0.6, noise=0.05)
    centered = center_against_baseline(adata, normal_mask)
    smoothed = smooth_along_chromosomes(
        centered,
        chr_labels=adata.var["chr"].to_numpy(),
        window=DEFAULT_SMOOTH_WINDOW,
    )

    # Tumor-mean smoothed signal on the amped chromosome should sit near +0.6,
    # well separated from the ~0 baseline on un-amped chromosomes.
    amp_idx = (adata.var["chr"] == amp_label).to_numpy()
    tumor_amp_smoothed = smoothed[~normal_mask][:, amp_idx]

    mean_tumor_amp = tumor_amp_smoothed.mean()
    assert 0.5 < mean_tumor_amp < 0.7
