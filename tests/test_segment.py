"""Unit tests for segment.py.

Reuses the planted-amp synthetic from test_smooth.py to verify:
- PELT finds at least one segment with elevated tumor_mean on the amped chr.
- Per-cell CN matrix cleanly separates tumor and normal cells on that segment.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from kopya.segment import (
    DEFAULT_MIN_SEG_GENES,
    DEFAULT_PENALTY_COEF,
    detect_segments,
    per_cell_segment_cn,
    per_segment_normal_baseline,
)
from kopya.smooth import (
    DEFAULT_SMOOTH_WINDOW,
    center_against_baseline,
    smooth_along_chromosomes,
)


# Reuse the planted-amp fixture from test_smooth. Importing test code is
# normally a smell, but the fixture is small, deterministic, and central to
# both modules' behavior — duplicating it would risk drift between tests.
from tests.test_smooth import _three_chr_adata


def _build_smoothed_fixture(amp_strength=0.6, **kwargs):
    """Run the M1-M3 prelude on a planted-amp synthetic.

    Returns the smoothed matrix + normal mask + chr labels + the amped
    chromosome label so individual segment tests can dispatch from one
    helper rather than re-running the full upstream pipeline.

    Args:
        amp_strength: forwarded to _three_chr_adata.
        **kwargs: forwarded to _three_chr_adata.

    Returns:
        Tuple of (smoothed, normal_mask, chr_labels, amp_label, starts).
    """
    adata, normal_mask, amp_label, starts = _three_chr_adata(
        amp_strength=amp_strength, **kwargs
    )
    centered = center_against_baseline(adata, normal_mask)
    smoothed = smooth_along_chromosomes(
        centered,
        chr_labels=adata.var["chr"].to_numpy(),
        window=DEFAULT_SMOOTH_WINDOW,
    )
    return smoothed, normal_mask, adata.var["chr"].to_numpy(), amp_label, starts


def test_detect_segments_finds_amp_on_correct_chromosome():
    """A planted amp on chr_synth_2 produces a segment with tumor_mean > 0 there."""
    smoothed, normal_mask, chr_labels, amp_label, _ = _build_smoothed_fixture(
        amp_strength=0.6
    )

    segments = detect_segments(smoothed, normal_mask, chr_labels)

    # At least one segment on the amped chromosome with clearly elevated mean.
    amp_segs = segments[segments["chr"] == amp_label]
    assert len(amp_segs) >= 1
    assert amp_segs["tumor_mean"].max() > 0.4, (
        f"expected tumor_mean > 0.4 on amped chr, got {amp_segs['tumor_mean'].max():.3f}"
    )


def test_detect_segments_un_amped_chr_is_near_zero():
    """Segments on unamplified chromosomes have tumor_mean near 0."""
    smoothed, normal_mask, chr_labels, amp_label, _ = _build_smoothed_fixture(
        amp_strength=0.6
    )

    segments = detect_segments(smoothed, normal_mask, chr_labels)

    non_amp = segments[segments["chr"] != amp_label]
    # All non-amp segments centered on 0 within ~0.1 noise.
    assert non_amp["tumor_mean"].abs().max() < 0.1, (
        f"unexpected non-amp segment mean: {non_amp['tumor_mean'].abs().max():.3f}"
    )


def test_detect_segments_table_schema():
    """Segments DataFrame has the documented columns and integer indices."""
    smoothed, normal_mask, chr_labels, _, _ = _build_smoothed_fixture()
    segments = detect_segments(smoothed, normal_mask, chr_labels)

    assert list(segments.columns) == ["chr", "start_idx", "end_idx", "n_genes", "tumor_mean"]
    # start_idx and end_idx must be integer types for downstream slicing.
    assert pd.api.types.is_integer_dtype(segments["start_idx"])
    assert pd.api.types.is_integer_dtype(segments["end_idx"])
    # n_genes equals end_idx - start_idx for every row.
    derived = segments["end_idx"] - segments["start_idx"]
    pd.testing.assert_series_equal(
        derived.astype("int64"),
        segments["n_genes"].astype("int64"),
        check_names=False,
    )


def test_detect_segments_min_seg_genes_floor():
    """No emitted segment is shorter than min_seg_genes."""
    smoothed, normal_mask, chr_labels, _, _ = _build_smoothed_fixture()
    segments = detect_segments(smoothed, normal_mask, chr_labels, min_seg_genes=30)

    assert (segments["n_genes"] >= 30).all()


def test_per_cell_segment_cn_separates_tumor_from_normal():
    """Per-cell CN on the amped segment is clearly higher for tumor than normal cells."""
    smoothed, normal_mask, chr_labels, amp_label, _ = _build_smoothed_fixture(
        amp_strength=0.6
    )

    segments = detect_segments(smoothed, normal_mask, chr_labels)
    cn_matrix = per_cell_segment_cn(smoothed, segments)

    # Find the amp-chr segment with the highest tumor_mean — the planted event.
    amp_segs = segments[segments["chr"] == amp_label]
    target_idx = amp_segs["tumor_mean"].idxmax()
    target_col = segments.index.get_loc(target_idx)

    tumor_values = cn_matrix[~normal_mask, target_col]
    normal_values = cn_matrix[normal_mask, target_col]

    # Distributions must be clearly separated: tumor mean ~0.6, normal mean ~0.
    assert tumor_values.mean() > normal_values.mean() + 0.3, (
        f"tumor/normal not separated on amped segment: "
        f"tumor={tumor_values.mean():.3f}, normal={normal_values.mean():.3f}"
    )


def test_per_cell_segment_cn_shape():
    """CN matrix shape is (n_cells, n_segments)."""
    smoothed, normal_mask, chr_labels, _, _ = _build_smoothed_fixture()
    segments = detect_segments(smoothed, normal_mask, chr_labels)
    cn_matrix = per_cell_segment_cn(smoothed, segments)

    assert cn_matrix.shape == (smoothed.shape[0], len(segments))


# =============================================================================
# Centering pedestal / absolute-output interpretability
# =============================================================================


def _pedestal_adata(
    seed=0,
    n_normal=80,
    n_tumor=80,
    genes_per_chr=(150, 150, 120),
    loss_chr=1,
    detect_p=0.45,
    expr=1.2,
    loss_frac=0.35,
):
    """Zero-inflated synthetic that reproduces the centering pedestal.

    Unlike ``test_smooth._three_chr_adata`` (dense ~1.0 baseline, no zero
    inflation), here every gene is detected in only ``detect_p`` of cells and is
    exactly 0 elsewhere. That is the pedestal regime: a gene detected
    in <=50% of normals has a per-gene normal median of 0, so
    ``center_against_baseline`` leaves it uncentered and stamps a positive log
    pedestal on every cell. Tumor cells additionally carry a broad LOSS on
    ``loss_chr`` (expression detected ``loss_frac``x as often), so a correct,
    de-pedestalled ``tumor_mean`` must be negative there.

    Returns:
        (adata, normal_mask, loss_label).
    """
    rng = np.random.default_rng(seed)
    n_cells = n_normal + n_tumor
    n_genes = sum(genes_per_chr)

    chr_labels, starts, offset = [], {}, 0
    for i, n in enumerate(genes_per_chr):
        label = f"chr_synth_{i + 1}"
        starts[label] = offset
        chr_labels += [label] * n
        offset += n

    # Zero-inflated log1p-like values: detected (~expr) with prob detect_p, else 0.
    detected = rng.random((n_cells, n_genes)) < detect_p
    X = np.where(detected, rng.normal(expr, 0.1, size=(n_cells, n_genes)), 0.0)

    # Plant the loss: tumor cells detect the loss chromosome far less often.
    loss_label = f"chr_synth_{loss_chr + 1}"
    ls = starts[loss_label]
    le = ls + genes_per_chr[loss_chr]
    tumor_detected = rng.random((n_tumor, le - ls)) < (detect_p * loss_frac)
    X[n_normal:, ls:le] = np.where(
        tumor_detected, rng.normal(expr, 0.1, size=(n_tumor, le - ls)), 0.0
    )
    X = np.clip(X, 0.0, None)

    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame({"chr": chr_labels}, index=[f"GENE{i:05d}" for i in range(n_genes)])
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    normal_mask = np.array([True] * n_normal + [False] * n_tumor)
    return adata, normal_mask, loss_label


def _pedestal_smoothed(**kwargs):
    """Run M3 prelude (center + smooth) on the pedestal synthetic.

    Returns (smoothed, normal_mask, chr_labels, loss_label).
    """
    adata, normal_mask, loss_label = _pedestal_adata(**kwargs)
    centered = center_against_baseline(adata, normal_mask)
    smoothed = smooth_along_chromosomes(
        centered, chr_labels=adata.var["chr"].to_numpy(), window=DEFAULT_SMOOTH_WINDOW
    )
    return smoothed, normal_mask, adata.var["chr"].to_numpy(), loss_label


def test_pedestal_present_in_raw_cn():
    """Sanity: the zero-inflated synthetic really does carry a positive pedestal.

    The raw per-cell CN of the normal pool medians well above 0 per segment —
    the constant additive offset described above. This guards the fix tests
    below: if the fixture ever stopped producing a pedestal they would pass
    vacuously.
    """
    smoothed, normal_mask, chr_labels, _ = _pedestal_smoothed()
    segments = detect_segments(smoothed, normal_mask, chr_labels)
    cn = per_cell_segment_cn(smoothed, segments)

    normal_seg_median = np.median(cn[normal_mask], axis=0)
    # Every segment's normal-pool median is a sizeable positive pedestal, not ~0.
    assert normal_seg_median.min() > 0.15, (
        f"expected a positive pedestal on every segment, got min "
        f"{normal_seg_median.min():.3f}"
    )


def test_tumor_mean_is_centered_diploid_zero_loss_negative():
    """detect_segments centers tumor_mean: diploid ~0, a planted loss clearly < 0.

    This is the core contract for segments.parquet. Before the fix the
    pedestal made every segment's tumor_mean positive, even on the lost
    chromosome; after it, diploid segments sit near 0 and the loss is negative.
    """
    smoothed, normal_mask, chr_labels, loss_label = _pedestal_smoothed()
    segments = detect_segments(smoothed, normal_mask, chr_labels)

    loss_tm = segments.loc[segments["chr"] == loss_label, "tumor_mean"]
    diploid_tm = segments.loc[segments["chr"] != loss_label, "tumor_mean"]

    # Every segment on the lost chromosome reads as a loss (< 0), clearly so.
    assert len(loss_tm) >= 1
    assert loss_tm.max() < -0.15, (
        f"planted loss not negative: max tumor_mean on loss chr = {loss_tm.max():.3f}"
    )
    # Diploid segments sit near 0 — the pedestal is gone (was ~+0.35 raw).
    assert diploid_tm.abs().max() < 0.2, (
        f"diploid tumor_mean not centered on 0: absmax = {diploid_tm.abs().max():.3f}"
    )


def test_per_segment_normal_baseline_matches_cn_median_and_falls_back():
    """per_segment_normal_baseline = per-segment normal-pool median, cohort fallback.

    It is the single diploid reference the absolute outputs subtract; assert it
    equals median(cn[normal]) and that an empty pool falls back to the cohort
    median without crashing (mirrors classify._normal_pool_baseline)."""
    smoothed, normal_mask, chr_labels, _ = _pedestal_smoothed()
    segments = detect_segments(smoothed, normal_mask, chr_labels)
    cn = per_cell_segment_cn(smoothed, segments)

    base = per_segment_normal_baseline(cn, normal_mask)
    assert base.shape == (cn.shape[1],)
    np.testing.assert_allclose(base, np.median(cn[normal_mask], axis=0))

    empty = np.zeros(cn.shape[0], dtype=bool)
    fb = per_segment_normal_baseline(cn, empty)
    np.testing.assert_allclose(fb, np.median(cn, axis=0))


def test_detect_segments_all_normal_pool_falls_back():
    """With all cells in the normal pool, segmentation runs on the cohort mean.

    Validates the fallback path (pooled signal computed across the full
    cohort) executes without crashing and produces a usable segment table.
    Does not assert near-zero means because the fixture still contains
    planted tumor cells — the cohort mean naturally absorbs half their amp.
    """
    smoothed, _, chr_labels, _, _ = _build_smoothed_fixture()
    all_normal = np.ones(smoothed.shape[0], dtype=bool)

    segments = detect_segments(smoothed, all_normal, chr_labels)

    # Schema sanity: non-empty table with the documented columns.
    assert len(segments) >= 1
    assert {"chr", "start_idx", "end_idx", "n_genes", "tumor_mean"} <= set(segments.columns)
    # Segment indices remain in valid range — no negative or out-of-bounds slicing.
    assert (segments["start_idx"] >= 0).all()
    assert (segments["end_idx"] <= smoothed.shape[1]).all()
