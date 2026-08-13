"""Unit tests for classify.py.

Synthetic CN matrix with a planted bimodal score distribution; tumor cells
must land in the tumor cluster, normal cells in the normal cluster, and
subclone discovery must respect max_subclones.
"""

import numpy as np
import pandas as pd
import pytest

from kopya.classify import (
    DEFAULT_CALL_CONFIDENCE,
    DEFAULT_COHERENCE_GATE,
    DEFAULT_MAX_SUBCLONES,
    _coherent_fraction,
    _low_complexity_mask,
    _normal_pool_baseline,
    classify_cells,
    compute_tumor_scores,
    discover_subclones,
    gmm_classify,
)


def _bimodal_cn_matrix(n_normal=50, n_tumor=50, n_segments=20, seed=0):
    """Build a (n_cells × n_segments) CN matrix with clear tumor/normal split.

    Normal cells: ~zero across all segments (diploid baseline already centered).
    Tumor cells: +0.5 on the first half of segments, -0.5 on the second half
    (one big gain block, one big loss block — a realistic per-clone profile).

    Args:
        n_normal: Number of planted-normal cells (rows 0..n_normal-1).
        n_tumor: Number of planted-tumor cells.
        n_segments: Total segments (split half/half into gain/loss).
        seed: RNG seed for noise.

    Returns:
        Tuple of (cn_matrix, normal_mask, tumor_mask).
    """
    rng = np.random.default_rng(seed)
    n_cells = n_normal + n_tumor

    # Per-entry Gaussian noise so cells aren't pathologically identical.
    cn = rng.normal(loc=0.0, scale=0.05, size=(n_cells, n_segments)).astype(np.float32)

    # Plant a gain (+0.5) on the first half of segments, a loss (-0.5) on the
    # second, applied only to tumor rows.
    half = n_segments // 2
    cn[n_normal:, :half] += 0.5
    cn[n_normal:, half:] -= 0.5

    normal_mask = np.array([True] * n_normal + [False] * n_tumor)
    tumor_mask = ~normal_mask
    return cn, normal_mask, tumor_mask


def test_compute_tumor_scores_normal_lower_than_tumor():
    """Tumor cells score higher than normal cells (by construction)."""
    cn, normal_mask, _ = _bimodal_cn_matrix()

    scores = compute_tumor_scores(cn, normal_mask)

    # Mean tumor score should clearly exceed mean normal score.
    normal_mean = scores[normal_mask].mean()
    tumor_mean = scores[~normal_mask].mean()
    assert tumor_mean > normal_mean * 3, (
        f"insufficient separation: normal={normal_mean:.3f}, tumor={tumor_mean:.3f}"
    )


def test_gmm_classify_recovers_tumor_normal_split():
    """GMM call correctly assigns the planted tumor/normal cells."""
    cn, normal_mask, tumor_mask = _bimodal_cn_matrix()
    scores = compute_tumor_scores(cn, normal_mask)

    df = gmm_classify(scores, normal_mask=normal_mask)

    # At least 90% of planted-tumor cells called 'tumor', and >= 90% of
    # planted-normal cells called 'normal' (uncertainty band gets the rest).
    tumor_calls = (df["class"].to_numpy() == "tumor")
    normal_calls = (df["class"].to_numpy() == "normal")
    tumor_recall = tumor_calls[tumor_mask].mean()
    normal_recall = normal_calls[normal_mask].mean()
    assert tumor_recall >= 0.9, f"tumor recall={tumor_recall:.2f}"
    assert normal_recall >= 0.9, f"normal recall={normal_recall:.2f}"


def test_gmm_classify_never_overrides_upstream_normal():
    """A cell flagged confident-normal upstream is never re-labeled 'tumor'."""
    cn, normal_mask, _ = _bimodal_cn_matrix()
    # Synthesize a malicious case: corrupt one "normal" cell's data to look
    # like a tumor cell, but keep its upstream normal_mask flag = True.
    # gmm_classify must still keep it labeled 'normal' (or 'uncertain'),
    # never 'tumor'.
    cn[0, :] = 5.0  # extreme deviation
    scores = compute_tumor_scores(cn, normal_mask)

    df = gmm_classify(scores, normal_mask=normal_mask)

    # Cell 0 was flagged upstream normal — must NOT be called 'tumor'.
    assert df.iloc[0]["class"] != "tumor"


def test_gmm_classify_uncertainty_label():
    """Cells with low GMM posterior become 'uncertain'."""
    # Construct a unimodal score distribution — GMM will be confused and
    # the posterior split will be near 0.5 for most cells.
    rng = np.random.default_rng(1)
    scores = rng.normal(loc=1.0, scale=0.1, size=100).astype(np.float64)

    df = gmm_classify(scores, confidence_threshold=0.95)

    # With a tight unimodal distribution and a near-1.0 threshold, most cells
    # should land in 'uncertain'.
    assert (df["class"] == "uncertain").mean() > 0.5


def test_discover_subclones_respects_max_subclones():
    """Subclone discovery never returns more than max_subclones distinct labels."""
    cn, _, tumor_mask = _bimodal_cn_matrix(n_normal=20, n_tumor=80)

    subclones = discover_subclones(cn, tumor_mask, max_subclones=3)

    # Non-tumor cells get "" (empty); tumor cells get "subclone_N".
    distinct_clones = {s for s in subclones[tumor_mask] if s}
    assert len(distinct_clones) <= 3
    assert all(s.startswith("subclone_") for s in distinct_clones)
    # Non-tumor cells must have empty subclone labels.
    assert all(s == "" for s in subclones[~tumor_mask])


def test_discover_subclones_below_floor_returns_one_clone():
    """With fewer than MIN_TUMOR_CELLS_FOR_SUBCLONES tumor cells, we collapse to k=1."""
    cn, _, tumor_mask = _bimodal_cn_matrix(n_normal=80, n_tumor=10)

    subclones = discover_subclones(cn, tumor_mask)

    tumor_clones = {s for s in subclones[tumor_mask] if s}
    assert tumor_clones == {"subclone_1"}


def test_classify_cells_end_to_end():
    """classify_cells composes scores + GMM + subclones into one DataFrame."""
    cn, normal_mask, tumor_mask = _bimodal_cn_matrix(n_normal=30, n_tumor=70)
    # Provide a minimal segments DataFrame (only n_genes_altered uses it
    # transitively, via the dev tolerance heuristic).
    segments = pd.DataFrame({
        "chr": ["chr1"] * cn.shape[1],
        "start_idx": list(range(cn.shape[1])),
        "end_idx": list(range(1, cn.shape[1] + 1)),
        "n_genes": [1] * cn.shape[1],
        "tumor_mean": [0.0] * cn.shape[1],
    })
    barcodes = [f"cell_{i:04d}" for i in range(cn.shape[0])]

    df = classify_cells(
        cn_matrix=cn,
        segments=segments,
        normal_mask=normal_mask,
        barcodes=barcodes,
    )

    # Schema sanity.
    assert df.index.name == "barcode"
    assert list(df.columns) == ["class", "confidence", "tumor_score", "subclone", "n_segments_altered", "low_complexity"]

    # Tumor cells were planted to alter ALL segments; n_segments_altered for
    # tumor cells should equal the total segment count (modulo noise floor).
    tumor_df = df[df["class"] == "tumor"]
    assert (tumor_df["n_segments_altered"] >= cn.shape[1] - 2).all()

    # Tumor cells should carry a subclone label.
    assert (tumor_df["subclone"].str.startswith("subclone_")).all()
    # The gain-half/loss-half profile sits on one chromosome but is two contiguous
    # runs, so the (moving-average) coherence gate must NOT wipe out the tumor
    # calls — a real tumor set survives.
    assert len(tumor_df) >= 0.8 * int(tumor_mask.sum())


def _two_chrom_segments(n_segments):
    """Segments split evenly across chr1/chr2 (for the gate tests)."""
    half = n_segments // 2
    return pd.DataFrame({
        "chr": ["chr1"] * half + ["chr2"] * (n_segments - half),
        "start_idx": list(range(n_segments)),
        "end_idx": list(range(1, n_segments + 1)),
        "n_genes": [50] * n_segments,
        "tumor_mean": [0.0] * n_segments,
    })


def test_low_complexity_gate_downgrades_tumor_to_uncertain():
    """A tumor-scoring cell flagged low-complexity is downgraded to uncertain."""
    cn, normal_mask, tumor_mask = _bimodal_cn_matrix(n_normal=40, n_tumor=40, n_segments=20)
    segments = _two_chrom_segments(cn.shape[1])
    barcodes = [f"cell_{i}" for i in range(cn.shape[0])]

    # All cells high-complexity except one tumor cell, which is ambient-like.
    complexity = np.full(cn.shape[0], 2000.0)
    first_tumor = int(np.where(tumor_mask)[0][0])
    complexity[first_tumor] = 50.0  # far below 0.5 * median

    df = classify_cells(cn, segments, normal_mask, barcodes, complexity=complexity)

    assert bool(df.iloc[first_tumor]["low_complexity"]) is True
    assert df.iloc[first_tumor]["class"] == "uncertain"
    # Invariant: a gate-downgraded cell must not keep a confident (>= threshold)
    # score — otherwise a downstream `confidence >= threshold` filter would treat
    # this untrustworthy cell as a confident call.
    assert df.iloc[first_tumor]["confidence"] < DEFAULT_CALL_CONFIDENCE
    # Other tumor cells (normal complexity, coherent signal) stay tumor.
    others = np.where(tumor_mask)[0][1:]
    assert (df.iloc[others]["class"] == "tumor").mean() > 0.8


def test_coherent_fraction_gates_edge_isolated_spike():
    """An isolated spike in the FIRST/LAST segment of a chromosome must score below
    the gate (regression: mode='nearest' padding gave it a spurious ~0.6)."""
    segments = _two_chrom_segments(12)
    # _coherent_fraction takes the reference-relative deviation directly, so these
    # arrays ARE the deviation (a zero baseline, per-cell centering already done).
    edge = np.zeros((1, 12), dtype=np.float32); edge[0, 0] = 0.8          # first segment of chr1
    run = np.zeros((1, 12), dtype=np.float32); run[0, 1:5] = 0.5          # contiguous run, interior
    assert _coherent_fraction(edge, segments)[0] < DEFAULT_COHERENCE_GATE
    assert _coherent_fraction(run, segments)[0] > DEFAULT_COHERENCE_GATE


def test_coherence_gate_downgrades_scattered_cell():
    """A sign-alternating (scattered) cell that scores in the tumor range is
    downgraded to uncertain, while coherent tumor cells are unaffected."""
    cn, normal_mask, _ = _bimodal_cn_matrix(n_normal=50, n_tumor=49, n_segments=20)
    # One cell with the same magnitude as tumor cells (so the GMM puts it in the
    # tumor component) but an alternating, non-contiguous profile — no real CNV.
    scattered = np.zeros((1, cn.shape[1]), dtype=np.float32)
    scattered[0, ::2] = 0.5
    scattered[0, 1::2] = -0.5
    cn2 = np.vstack([cn, scattered])
    normal_mask2 = np.append(normal_mask, False)
    segments = _two_chrom_segments(cn2.shape[1])
    barcodes = [f"cell_{i}" for i in range(cn2.shape[0])]

    # With the gate off, the scattered cell is called tumor (magnitude alone).
    off = classify_cells(cn2, segments, normal_mask2, barcodes, coherence_gate=0.0)
    assert off.iloc[-1]["class"] == "tumor"
    # With the gate on, it is downgraded to uncertain; coherent tumors stay tumor.
    on = classify_cells(cn2, segments, normal_mask2, barcodes)
    assert on.iloc[-1]["class"] == "uncertain"
    coherent_tumor = on.iloc[50:-1]
    assert (coherent_tumor["class"] == "tumor").mean() > 0.8


def test_coherence_gate_downgrades_same_sign_scatter():
    """A SAME-SIGN but non-contiguous (isolated-spike) cell is downgraded too —
    contiguity, not just sign-cancellation, must gate it. Regression guard: the
    earlier L1-conserving moving-average metric let this class through (coh~1)."""
    cn, normal_mask, _ = _bimodal_cn_matrix(n_normal=50, n_tumor=49, n_segments=20)
    # Isolated same-sign spikes (every third segment, separated by neutral ones),
    # summing to the tumor score range so the GMM puts it in the tumor component —
    # but with no contiguous run the local neighbourhood does not corroborate them.
    scatter = np.zeros((1, cn.shape[1]), dtype=np.float32)
    scatter[0, ::3] = 1.5
    cn2 = np.vstack([cn, scatter])
    normal_mask2 = np.append(normal_mask, False)
    segments = _two_chrom_segments(cn2.shape[1])
    barcodes = [f"cell_{i}" for i in range(cn2.shape[0])]

    off = classify_cells(cn2, segments, normal_mask2, barcodes, coherence_gate=0.0)
    assert off.iloc[-1]["class"] == "tumor"  # magnitude alone calls it tumor
    on = classify_cells(cn2, segments, normal_mask2, barcodes)
    assert on.iloc[-1]["class"] == "uncertain"  # contiguity gate catches it
    assert (on.iloc[50:-1]["class"] == "tumor").mean() > 0.8  # real tumors unaffected


def test_low_complexity_mask_contract():
    """_low_complexity_mask flags below frac*median and no-ops on the edge cases."""
    # None complexity -> all-False mask of the requested length.
    m = _low_complexity_mask(None, 0.5, 6)
    assert m.shape == (6,) and not m.any()

    # Normal case: flag cells below 0.5 * median. median([100,100,100,100,10,10])=100,
    # threshold 50 -> only the two 10s flagged.
    comp = np.array([100.0, 100.0, 100.0, 100.0, 10.0, 10.0])
    m = _low_complexity_mask(comp, 0.5, len(comp))
    assert m.tolist() == [False, False, False, False, True, True]

    # Degenerate median (all zeros) -> nothing to compare against -> all-False,
    # never divides or flags spuriously (guards the ambient-dominated edge).
    m = _low_complexity_mask(np.zeros(5), 0.5, 5)
    assert not m.any()

    # All-equal positive complexity -> none below frac*median (frac < 1).
    m = _low_complexity_mask(np.full(5, 3000.0), 0.5, 5)
    assert not m.any()


def test_normal_pool_baseline_fallback_and_reuse():
    """_normal_pool_baseline = per-segment normal median, with a cohort-median
    fallback when the pool is empty; it is exactly the baseline the tumor score uses."""
    cn, normal_mask, _ = _bimodal_cn_matrix(n_normal=40, n_tumor=40, n_segments=12)
    base = _normal_pool_baseline(cn, normal_mask)
    assert base.shape == (cn.shape[1],)
    np.testing.assert_allclose(base, np.median(cn[normal_mask], axis=0))
    # Passing this baseline into compute_tumor_scores must match computing it inline.
    np.testing.assert_allclose(
        compute_tumor_scores(cn, normal_mask, baseline=base),
        compute_tumor_scores(cn, normal_mask),
    )
    # Empty pool -> cohort-median fallback (no crash, no NaN).
    empty = np.zeros(cn.shape[0], dtype=bool)
    fb = _normal_pool_baseline(cn, empty)
    np.testing.assert_allclose(fb, np.median(cn, axis=0))
