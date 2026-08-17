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
    DEFAULT_OUTLIER_FENCE_MULT,
    MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT,
    _coherent_fraction,
    _low_complexity_mask,
    _normal_pool_baseline,
    classify_cells,
    compute_tumor_scores,
    consensus_template,
    discover_subclones,
    gmm_classify,
    reference_relative_deviation,
    segment_burden,
    template_projection,
    weighted_median_rows,
)


def _bimodal_cn_matrix(n_normal=50, n_tumor=50, n_segments=20, seed=0, n_diploid_padding=0):
    """Build a (n_cells × (n_segments + n_diploid_padding)) CN matrix with a
    clear tumor/normal split.

    Normal cells: ~zero across all segments (diploid baseline already centered).
    Tumor cells: +0.5 on the first half of `n_segments`, -0.5 on the second
    half (one gain block, one loss block — a realistic per-clone profile),
    followed by `n_diploid_padding` additional segments left at the diploid
    baseline (~0, same background noise as everywhere else).

    n_diploid_padding=0 (the default, used by most tests here) means the
    gain+loss blocks span the ENTIRE matrix — fine for tests that only need a
    clear aggregate tumor/normal score gap, but NOT a realistic input to
    reference_relative_deviation's per-cell centering step: that step finds
    each cell's own weighted-median deviation and treats it as a depth
    pedestal to subtract, which is only correct when the truly-altered
    segments are a MINORITY of the cell's genome (see that function's
    docstring: "a real gain on 30% of genome keeps that gain in residual").
    With no padding there is no genuine diploid majority for the median to
    find, so whichever block covers the larger share (both halves, at an
    exact tie) gets absorbed as if it were the pedestal — silently erasing
    that block's real signal. This isn't specific to an exact 50/50 split; it
    happens for any split once the whole genome is "altered." Tests that
    reason about n_segments_altered in the per-cell-centered frame should
    pass n_diploid_padding large enough that n_segments is a clear minority
    of the total, giving a genuine diploid majority for the median to find
    while keeping the gain/loss blocks themselves the same absolute length
    (and hence the same behavior under the coherence gate's fixed window) as
    the n_diploid_padding=0 case; see test_classify_cells_end_to_end.

    Args:
        n_normal: Number of planted-normal cells (rows 0..n_normal-1).
        n_tumor: Number of planted-tumor cells.
        n_segments: Segments covered by the gain+loss blocks (split evenly).
        seed: RNG seed for noise.
        n_diploid_padding: Additional always-diploid segments appended after
            the gain+loss blocks, so the total segment count is
            n_segments + n_diploid_padding.

    Returns:
        Tuple of (cn_matrix, normal_mask, tumor_mask).
    """
    rng = np.random.default_rng(seed)
    n_cells = n_normal + n_tumor
    total_segments = n_segments + n_diploid_padding

    # Per-entry Gaussian noise so cells aren't pathologically identical.
    cn = rng.normal(loc=0.0, scale=0.05, size=(n_cells, total_segments)).astype(np.float32)

    # Plant a gain (+0.5) on the first half of n_segments, a loss (-0.5) on
    # the second half, applied only to tumor rows. Columns n_segments:total
    # are left at the diploid background noise from above (the padding).
    half = n_segments // 2
    cn[n_normal:, :half] += 0.5
    cn[n_normal:, half:n_segments] -= 0.5

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
    # n_diploid_padding=30: a realistic clone where the 20-segment gain+loss
    # block is a clear minority (20/50 = 40%) against a genuine diploid
    # majority elsewhere — NOT n_diploid_padding=0 (the default), which plants
    # alteration across the whole genome and leaves no genuine diploid
    # majority for the per-cell centering step's median to find (see
    # _bimodal_cn_matrix's docstring). The gain/loss blocks keep the same
    # absolute length (10 segments each) either way, so the coherence gate's
    # behavior on them is unaffected by this change.
    cn, normal_mask, tumor_mask = _bimodal_cn_matrix(n_normal=30, n_tumor=70, n_diploid_padding=30)
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
    assert list(df.columns) == [
        "class", "confidence", "tumor_score", "cn_burden", "subclone",
        "n_segments_altered", "low_complexity",
    ]
    # tumor_score is the signed projection; cn_burden is its non-negative
    # magnitude companion. Consumers picking a diploid baseline rank on the
    # latter, so it must actually be non-negative.
    assert (df["cn_burden"].to_numpy() >= 0).all()

    # Tumor cells were planted to alter 20 of 50 segments (a minority, with a
    # genuine diploid majority in the padding); n_segments_altered for tumor
    # cells should land on that count (modulo noise floor) — the per-cell
    # centering step correctly leaves a minority-altered clone's signal
    # intact rather than absorbing it as a depth pedestal.
    tumor_df = df[df["class"] == "tumor"]
    assert (tumor_df["n_segments_altered"] >= 18).all()

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
    """A cell whose deviation ALIGNS with the cohort's consensus CN pattern in
    aggregate (enough to score in the tumor range via template_projection) but
    is NOT locally contiguous is still downgraded to uncertain; coherent tumor
    cells are unaffected.

    This is not a pure-noise alternating pattern (e.g. +0.5/-0.5 every other
    segment): under template_projection, a pattern whose positive and negative
    parts don't track the template's own sign structure nets to ~0 in the
    projection and is already correctly called 'normal' with NO gate involved
    at all -- template_projection's directional scoring already subsumes that
    failure mode, which the older, magnitude-only compute_tumor_scores did not.
    The remaining, still-real job for the coherence gate under the new scoring
    is a cell whose deviation matches the template's sign PER HALF (positive
    where the template gains, negative where it loses) but arrives as isolated
    spikes rather than a contiguous run -- template_projection alone cannot
    tell that apart from a real clonal block, because it only reasons about
    aggregate alignment, not spatial layout.
    """
    # n_diploid_padding=40 so the planted clone is a genuine minority of the
    # genome. Without it the per-cell centering absorbs the tumor cells' loss
    # block (see test_reference_relative_deviation_exact_tie_is_a_known_
    # limitation), the template's second half is noise rather than a loss, and
    # the "negative where they lose" premise below does not actually hold.
    cn, normal_mask, _ = _bimodal_cn_matrix(
        n_normal=50, n_tumor=49, n_segments=20, n_diploid_padding=40,
    )
    # Isolated spikes (every 3rd segment within each half -- the same spacing
    # test_coherence_gate_downgrades_same_sign_scatter already validates as
    # "not locally coherent"), signed to match the template: positive in the
    # first half (where the cohort's real tumor cells gain), negative in the
    # second (where they lose).
    #
    # Amplitude 1.5 against the clone's 0.5 is deliberate and is NOT "the same
    # magnitude range" (the claim this fixture used to make): covering a third of
    # the segments a real clonal block covers, a cell needs ~3x the per-segment
    # amplitude to reach the same aggregate projection, which is precisely what
    # makes it a test of spatial layout at matched score. It is low enough that
    # the cell is one ordinary seed among ~25 rather than a template-setter --
    # consensus_template's unit rescaling is what keeps that true, and
    # test_consensus_template_survives_high_amplitude_contamination is what pins
    # it. The old fixture used 2.5 with no padding, so it was measuring the cell's
    # contamination of the template it was then scored against.
    scattered = np.zeros((1, cn.shape[1]), dtype=np.float32)
    scattered[0, 0:10:3] = 1.5
    scattered[0, 10:20:3] = -1.5
    cn2 = np.vstack([cn, scattered])
    normal_mask2 = np.append(normal_mask, False)
    segments = _two_chrom_segments(cn2.shape[1])
    barcodes = [f"cell_{i}" for i in range(cn2.shape[0])]

    # With the gate off, the spiky cell is called tumor (aggregate alignment alone).
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


def test_reference_relative_deviation_exact_tie_is_a_known_limitation():
    """Known, documented limitation of the per-cell centering step: it treats
    whatever covers a MAJORITY (by weight) of a cell's segments as that cell's
    depth pedestal and subtracts it — correct when the truly-altered segments
    are a minority (the documented, realistic case), but not when "altered"
    covers half or more of the genome. In that case whichever block is larger
    (or, at an exact tie, whichever the median's tie-break happens to favor)
    gets absorbed as if it were the pedestal, silently erasing that block's
    real signal rather than a depth artifact.

    This test uses the sharpest instance (an exact 50/50 split, so there is no
    majority at all) because it is the cleanest illustration, not because
    only exact ties are affected — see _bimodal_cn_matrix's docstring, which
    documents this as a general property (the gain+loss blocks must be a
    minority of the total, via n_diploid_padding), and why
    test_classify_cells_end_to_end deliberately pads its matrix with extra
    diploid segments rather than relying on some other split ratio.
    """
    # One cell, 10 equal-weight segments: +0.5 on the first 5 (a real gain),
    # -0.5 on the last 5 (a real loss) — an exact 50/50 weighted tie.
    dev = np.array([[0.5] * 5 + [-0.5] * 5], dtype=np.float32)
    seg_weights = np.ones(10)

    centered = reference_relative_deviation(
        dev, normal_mask=np.array([False]), baseline=np.zeros(10), seg_weights=seg_weights,
    )

    # The tie collapses the symmetric +0.5/-0.5 signal into a single asymmetric
    # plateau (here: +1.0 on the gain block, 0.0 on the loss block) rather than
    # preserving both halves — half the segments read as "no deviation" even
    # though they carry real, planted CN.
    magnitudes = sorted(float(v) for v in np.unique(centered))
    assert magnitudes == [0.0, 1.0]


def test_reference_relative_deviation_keeps_float32():
    """The deviation must not silently promote to float64.

    cn_matrix is float32 but the normal-pool baseline is a float64 median, so a plain
    subtraction doubles the size of an array classify_cells then holds for its whole
    run — 3.2 GB instead of 1.6 GB at the documented 800k x 500, on the same path
    where weighted_median_rows blocks its own intermediates to 64 MB.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(n_segments=20, n_diploid_padding=40)
    cn = cn.astype(np.float32)
    w = _segments_for(cn.shape[1])["n_genes"].to_numpy()
    assert reference_relative_deviation(cn, normal_mask, seg_weights=w).dtype == np.float32
    # A float64 input is still respected rather than downcast.
    assert reference_relative_deviation(
        cn.astype(np.float64), normal_mask, seg_weights=w
    ).dtype == np.float64


def test_weighted_median_rows_guards_zero_total_weight():
    """Zero total weight must not return the row MINIMUM.

    half = 0.5 * 0 = 0 makes ``cumw >= half`` true at position 0, so argmax returns
    the smallest value in the row. Subtracting that as the per-cell center would make
    every deviation non-negative and erase all loss signal. segment_burden already
    guards its total the same way.
    """
    mat = np.array([[3.0, 1.0, 2.0], [9.0, -4.0, 0.0]])
    assert np.array_equal(weighted_median_rows(mat, np.zeros(3)), np.zeros(2))


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


# ── regression tests for the PR#2 review findings ─────────────────────────────

def _segments_for(total, per_chr=10):
    """Minimal segment table: `per_chr` segments per chromosome, equal gene counts."""
    return pd.DataFrame({
        "chr": [f"chr{i // per_chr + 1}" for i in range(total)],
        "n_genes": [100] * total,
    })


def _cohort_with_artifacts(n_art, amp, n_seg=20, n_pad=40, seed=0):
    """Bimodal cohort plus `n_art` anti-aligned cells at amplitude `amp`.

    The artifacts carry the clone's profile with the sign flipped and a much
    larger magnitude — the transcriptome-extreme cell (erythrocyte, platelet)
    whose deviation is big but directionally unrelated to the tumor's CN.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(
        n_segments=n_seg, n_diploid_padding=n_pad, seed=seed,
    )
    rng = np.random.default_rng(seed + 7)
    half = n_seg // 2
    art = rng.normal(0.0, 0.05, size=(n_art, cn.shape[1]))
    art[:, :half] -= amp
    art[:, half:n_seg] += amp
    cn = np.vstack([cn, art])
    normal_mask = np.concatenate([normal_mask, np.zeros(n_art, dtype=bool)])
    return cn, normal_mask


@pytest.mark.parametrize("n_art,amp", [(1, 12.0), (3, 6.0), (10, 6.0)])
def test_consensus_template_survives_high_amplitude_contamination(n_art, amp):
    """A handful of transcriptome-extreme cells must not set the template's sign.

    Seeds are chosen by burden, which selects FOR amplitude, so under a plain
    mean over raw deviations the cells most able to flip the template are exactly
    the ones the classifier exists to reject. Three anti-aligned cells at 12x the
    clonal amplitude — 2.9% of the cohort — used to inverted the whole call:
    every true tumor cell "normal", every artifact "tumor". Rescaling each seed to
    unit norm makes it a vote rather than a veto.
    """
    cn, normal_mask = _cohort_with_artifacts(n_art, amp)
    segments = _segments_for(cn.shape[1])
    barcodes = [f"cell{i}" for i in range(cn.shape[0])]

    df = classify_cells(
        cn_matrix=cn, segments=segments, normal_mask=normal_mask,
        barcodes=barcodes, discover_subclones_enabled=False,
    )
    cls = df["class"].to_numpy()

    # Planted tumor cells occupy rows 50..99; artifacts are appended after them.
    assert (cls[50:100] == "tumor").sum() >= 45, (
        f"{n_art} artifacts at amplitude {amp} hijacked the template: only "
        f"{(cls[50:100] == 'tumor').sum()}/50 true tumor cells called tumor"
    )
    # The artifacts themselves anti-align, so they must not be called tumor.
    assert (cls[100:] == "tumor").sum() == 0


def test_consensus_template_is_scale_invariant_downstream():
    """Template scale cancels in template_projection; only its shape matters."""
    cn, normal_mask, _ = _bimodal_cn_matrix(n_segments=20, n_diploid_padding=40)
    w = _segments_for(cn.shape[1])["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)
    tpl = consensus_template(dev, w)

    np.testing.assert_allclose(
        template_projection(dev, tpl, w),
        template_projection(dev, tpl * 37.5, w),
        rtol=1e-10,
    )


def test_consensus_template_excludes_low_complexity_seeds():
    """Ambient cells are barred from seeding, not merely gated after the fact.

    The low-complexity gate only downgrades a flagged cell's OWN call, which is
    too late: burden selection concentrates noise-dominated ambient cells in the
    seed set, where they steer everyone else's score.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(n_segments=20, n_diploid_padding=40)
    w = _segments_for(cn.shape[1])["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)

    # Three ambient-like rows with huge anti-aligned noise.
    dev = dev.copy()
    dev[:3] = -8.0 * dev[75]
    exclude = np.zeros(dev.shape[0], dtype=bool)
    exclude[:3] = True

    clean = consensus_template(dev, w, exclude=exclude)
    contaminated = consensus_template(dev, w)

    # Excluded, the template aligns with the clone; included, it does not.
    assert float(np.dot(clean, dev[75])) > 0
    assert float(np.dot(clean, dev[75])) > float(np.dot(contaminated, dev[75]))


def _opposing_subclone_cohort(seed=1, n_norm=50, n_a=50, n_b=50, n_seg=20, n_pad=40):
    """50 reference cells plus two equal, exactly-opposing high-burden clones."""
    rng = np.random.default_rng(seed)
    total = n_seg + n_pad
    half = n_seg // 2
    cn = rng.normal(0.0, 0.05, size=(n_norm + n_a + n_b, total))
    cn[n_norm:n_norm + n_a, :half] += 0.6
    cn[n_norm:n_norm + n_a, half:n_seg] -= 0.6
    cn[n_norm + n_a:, :half] -= 0.6
    cn[n_norm + n_a:, half:n_seg] += 0.6
    normal_mask = np.array([True] * n_norm + [False] * (n_a + n_b))
    return cn, normal_mask, _segments_for(total)


def test_consensus_template_opposing_subclones_surface_as_uncertain():
    """Two equal, exactly-opposing clones: one is called tumor, the other uncertain.

    consensus_template estimates ONE direction. With two opposing high-burden
    subclones the seed set holds both, and per-cell noise — not population size —
    breaks the tie: the template locks onto whichever clone's direction wins and
    that clone scores positive. The other scores symmetrically NEGATIVE, which under
    a signed score is the same region of the range as a confident diploid cell.

    Scoring that clone correctly needs multiple templates with best-aligned scoring
    — a change to the scoring contract, deliberately out of scope here. What IS in
    scope is that the failure not be silent: the anti-alignment gate in
    classify_cells recognises a large, coherent deviation pointing against the
    consensus and downgrades those cells to "uncertain" rather than asserting them
    normal. The clone is still not recovered; it is no longer hidden.

    NOTE on this test's history. It previously asserted that NEITHER clone is
    recovered, and passed, because the outlier fence's activation guard was
    comparing against median(scores[~normal_mask]) and firing spuriously, locking
    both clones to "normal". The symmetric outcome was an artifact of that bug;
    fixing the guard (see test_outlier_fence_holds_when_reference_pool_is_a_subset)
    exposed the asymmetric behaviour this now pins.
    """
    cn, normal_mask, segments = _opposing_subclone_cohort()
    n_norm = n_a = 50

    df = classify_cells(
        cn_matrix=cn, segments=segments, normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    cls = df["class"].to_numpy()
    a, b = cls[n_norm:n_norm + n_a], cls[n_norm + n_a:]

    tumor_counts = sorted((int((a == "tumor").sum()), int((b == "tumor").sum())))
    assert tumor_counts[1] >= 45, (
        f"expected one clone recovered as tumor, got {tumor_counts}"
    )
    # The other clone is NOT recovered — that is the standing limitation — but every
    # one of its cells is flagged rather than asserted normal. When the
    # multi-template change lands, both counts above should be >= 45 and this
    # assertion is what has to flip.
    erased = a if (a == "tumor").sum() < (b == "tumor").sum() else b
    assert int((erased == "uncertain").sum()) >= 45, (
        f"opposing clone silently called normal: {dict(zip(*np.unique(erased, return_counts=True)))} "
        "— the anti-alignment gate did not fire"
    )
    assert df.attrs["n_anti_aligned"] >= 45

    # The reference pool is untouched: the gate never fires on cells the caller
    # asserted are normal, and never on cells that are genuinely near diploid.
    assert (cls[:n_norm] == "normal").all()

    # A downgraded cell must not keep a high posterior, or a downstream
    # `confidence >= threshold` filter would sail straight past the flag.
    assert (df["confidence"].to_numpy()[cls == "uncertain"] < DEFAULT_CALL_CONFIDENCE).all()


def test_anti_alignment_gate_ignores_ordinary_anti_aligned_artifacts():
    """The gate must not fire on the negative tail of an ordinary sample.

    A signed score puts transcriptionally extreme normal cells (erythrocytes,
    platelets) at the bottom of its range roughly half the time, and there are far
    more of those than there are opposing subclones. Measured across this repo's 10
    annotated benchmark patients, 0.4% of the most anti-aligned 2% of cells are
    ground-truth tumor and their cn_burden sits at or below the cohort median on
    10 of 10 — they carry LESS copy number than an average cell. The gate must
    therefore key on burden and coherence, not on the sign alone.

    Here: a normal sample with anti-aligned scatter — cells whose deviation opposes
    the clone but is spread over isolated single segments rather than blocks. They
    score negative; none of them should be flagged.
    """
    rng = np.random.default_rng(7)
    n_norm, n_tumor, n_art = 60, 50, 15
    n_seg, n_pad = 20, 40
    total = n_seg + n_pad
    half = n_seg // 2
    cn = rng.normal(0.0, 0.05, size=(n_norm + n_tumor + n_art, total))
    cn[n_norm:n_norm + n_tumor, :half] += 0.5
    cn[n_norm:n_norm + n_tumor, half:n_seg] -= 0.5
    # Anti-aligned but SCATTERED: every third segment of the clone's footprint,
    # flipped. Same sign pattern as the erased clone, none of the contiguity.
    art = slice(n_norm + n_tumor, None)
    cn[art, 0:half:3] -= 0.5
    cn[art, half:n_seg:3] += 0.5
    normal_mask = np.array([True] * n_norm + [False] * (n_tumor + n_art))

    df = classify_cells(
        cn_matrix=cn, segments=_segments_for(total), normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    scores = df["tumor_score"].to_numpy()
    # Precondition: these cells really do land in the negative tail, so the test is
    # exercising the gate's specificity rather than a sample that never reaches it.
    assert scores[art].max() < 0, "fixture no longer produces anti-aligned cells"
    assert df.attrs["n_anti_aligned"] == 0, (
        "the anti-alignment gate fired on scattered transcriptional artifacts; it "
        "must key on coherent, high-burden deviation, not on negative sign alone"
    )
    # And the real clone is unaffected by the gate's presence.
    assert int((df["class"].to_numpy()[n_norm:n_norm + n_tumor] == "tumor").sum()) >= 45


@pytest.mark.parametrize("amp", [1.0, 2.0, 3.0])
def test_coherence_gate_zero_never_loosens_the_anti_alignment_gate(amp):
    """``coherence_gate=0`` is documented as disabling a downgrade. It must not
    ENABLE one somewhere else.

    Three places consult the coherent fraction and the knob cannot mean the same
    thing in all of them. In the tumor-side gate coherence IS the decision, so 0
    turns it off. In the anti-alignment gate it is a RESTRICTING condition — one of
    four that all must hold — so wiring it to the raw knob meant 0 dropped the
    condition and left the gate firing on three, downgrading exactly the scattered
    anti-aligned artifacts that test_anti_alignment_gate_ignores_ordinary_anti_
    aligned_artifacts pins as must-not-fire. Measured on this fixture at amp 1.0:
    n_anti_aligned 0 with the gate at its default, 15 with it at 0.

    The invariant: switching the tumor-side gate off may not increase the number of
    cells any other gate touches.
    """
    rng = np.random.default_rng(7)
    n_norm, n_tumor, n_art = 60, 50, 15
    n_seg, n_pad = 20, 40
    total = n_seg + n_pad
    half = n_seg // 2
    cn = rng.normal(0.0, 0.05, size=(n_norm + n_tumor + n_art, total))
    cn[n_norm:n_norm + n_tumor, :half] += 0.5
    cn[n_norm:n_norm + n_tumor, half:n_seg] -= 0.5
    # Anti-aligned and scattered, at a range of amplitudes.
    art = slice(n_norm + n_tumor, None)
    cn[art, 0:half:3] -= amp
    cn[art, half:n_seg:3] += amp
    normal_mask = np.array([True] * n_norm + [False] * (n_tumor + n_art))
    kw = dict(
        cn_matrix=cn, segments=_segments_for(total), normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )

    on = classify_cells(**kw)
    off = classify_cells(coherence_gate=0.0, **kw)
    assert off.attrs["n_anti_aligned"] <= on.attrs["n_anti_aligned"], (
        f"amp {amp}: disabling the tumor-side coherence gate made the anti-alignment "
        f"gate fire on more cells ({on.attrs['n_anti_aligned']} -> "
        f"{off.attrs['n_anti_aligned']})"
    )
    # Same invariant for the fence's HIGH side, the other place a coherent fraction
    # licenses an action: with no floor left it goes inert rather than unconditional.
    # The LOW side is deliberately unaffected — it needs no coherence, because under a
    # signed score no clone projects negatively (see the fence), so there is nothing
    # for a coherent fraction to protect down there. So every exclusion at
    # coherence_gate=0 must be a low-side one.
    segs = _segments_for(total)
    w = segs["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)
    scores = template_projection(dev, consensus_template(dev, w), w)
    ref = scores[normal_mask]
    q75, q25 = np.percentile(ref, 75), np.percentile(ref, 25)
    n_low_side = int((scores < q25 - DEFAULT_OUTLIER_FENCE_MULT * (q75 - q25)).sum())
    assert off.attrs["n_outlier_fenced"] == n_low_side, (
        f"amp {amp}: at coherence_gate=0 the fence excluded "
        f"{off.attrs['n_outlier_fenced']} cells but only {n_low_side} are on its low "
        "side — the high side did not go inert"
    )
    # And the label-level invariant: disabling a gate cannot leave MORE cells flagged.
    assert (off["class"].to_numpy() == "uncertain").sum() <= (
        on["class"].to_numpy() == "uncertain"
    ).sum()


def test_low_purity_sample_still_recovers_tumor_cells():
    """A rare tumor population must survive both the template and the fence.

    Two ways to get this wrong, both measured during review: a per-segment MEDIAN
    consensus needs the seed set to be majority-malignant and returns ~0 here; and
    a fence measured on the GLOBAL IQR collapses when normals dominate and cuts
    straight through the tumor mode (recall 1.00 -> 0.50 at 2.9% tumor fraction).
    """
    for n_tumor in (25, 6):
        cn, normal_mask, _ = _bimodal_cn_matrix(
            n_normal=200, n_tumor=n_tumor, n_segments=20, n_diploid_padding=40,
        )
        df = classify_cells(
            cn_matrix=cn, segments=_segments_for(cn.shape[1]),
            normal_mask=normal_mask,
            barcodes=[f"cell{i}" for i in range(cn.shape[0])],
            discover_subclones_enabled=False,
        )
        recall = (df["class"].to_numpy()[200:] == "tumor").mean()
        assert recall >= 0.9, f"tumor fraction {n_tumor}/{200 + n_tumor}: recall {recall}"


def test_outlier_fence_does_not_cut_into_the_tumor_mode():
    """The fence must not take a cleanly-separating sample's malignant population.

    Every tumor cell is "far above where the reference pool sits", so the fence
    reaches all of them — measured on this fixture, 49/49 clear it — and what holds
    them is the coherent fraction, not the fence's height. The fence therefore has
    to be handed a way to ask about coherence (``coherence_of``); with none, the
    high-scorers are indistinguishable from artifacts and it must stay inert rather
    than guess. Both halves are asserted here.

    History: three earlier versions of this second condition were phrased as
    activation guards on the score distribution (against clip_ceiling, against
    median(scores[~nm]), against a provisional GMM's high component) and each
    either never fired or fenced the tumor population outright. See the fence.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(n_segments=20, n_diploid_padding=40)
    segments = _segments_for(cn.shape[1])
    w = segments["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)
    scores = template_projection(dev, consensus_template(dev, w), w)

    # With coherence available, the tumor cells are recognised as clonal and kept.
    df = gmm_classify(
        scores, normal_mask=normal_mask,
        coherence_of=lambda idx: _coherent_fraction(dev[idx], segments),
    )
    assert (df["class"].to_numpy()[50:] == "tumor").sum() >= 45
    assert df.attrs["n_outlier_fenced"] == 0
    # Precondition for the above meaning anything: they really are fence candidates.
    b = scores[normal_mask]
    fence = np.percentile(b, 75) + 12.0 * (np.percentile(b, 75) - np.percentile(b, 25))
    assert (scores[50:] > fence).all(), "fixture no longer exercises the fence"

    # Without coherence there is nothing to license an exclusion, so none happens.
    bare = gmm_classify(scores, normal_mask=normal_mask)
    assert bare.attrs["n_outlier_fenced"] == 0
    assert (bare["class"].to_numpy()[50:] == "tumor").sum() >= 45


@pytest.mark.parametrize("n_tumor,n_normal", [(8, 2000), (15, 2015)])
def test_outlier_fence_spares_a_sub_one_percent_clone(n_tumor, n_normal):
    """A clone below ~1% of the sample must survive the fence.

    Regression for a frame mismatch: the fence's activation guard located the tumor
    mode with a GMM fitted on the WINSORIZED scores, while the fence itself and the
    exclusion test ran on the RAW ones. Below ~1% tumor fraction the malignant
    population sits above the global P99 clip ceiling, so the clipped vector holds no
    tumor mode at all, the fitted mode collapsed onto the normal mode, the guard
    passed — and the fence then selected exactly the tumor cells. Measured at 8/2008:
    ceiling 0.0222, tumor median 0.1763, fitted mode 0.006, fence 0.1368, and all 8
    malignant cells fenced for a recall of 0.00. The sibling tests all sit at >= 2.9%
    purity, where the clip ceiling still lands above the tumor mode and the frame
    mismatch is invisible.

    No statistic of the score vector can fix this: a handful of cells far above the
    normal mode is the same 1-D distribution whether they are a rare clone or a
    cluster of transcriptome artifacts. The fence separates them spatially instead
    (see test_outlier_fence_holds_out_incoherent_high_scorers for the other side).
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(
        n_normal=n_normal, n_tumor=n_tumor, n_segments=20, n_diploid_padding=40,
    )
    df = classify_cells(
        cn_matrix=cn, segments=_segments_for(cn.shape[1]), normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    recall = (df["class"].to_numpy()[n_normal:] == "tumor").mean()
    assert recall >= 0.9, (
        f"{n_tumor}/{n_normal + n_tumor} tumor fraction: recall {recall}, "
        f"{df.attrs['n_outlier_fenced']} cells fenced"
    )
    assert df.attrs["n_outlier_fenced"] == 0


def test_outlier_fence_holds_out_incoherent_high_scorers():
    """The other side: high-scoring SCATTERED cells are held out of the GMM fit.

    This is the case the fence exists for, and no version of it that guarded on the
    score distribution could ever reach: with no real tumor present the artifacts
    themselves are the high component, so a guard comparing the fence to a fitted
    tumor mode is structurally unable to fire. Measured before the fix — 980 cells at
    ~N(0, 0.01) plus 20 at 5.0, 100 of them labelled reference — the fence sat at
    0.170 against a fitted mode of ~5, nothing was fenced, and all 20 artifacts were
    called tumor.

    Note what is and is not asserted. The fence decides what the mixture is FITTED
    on; it does not label anything (see the note at gmm_classify's labelling step for
    why it stopped forcing "normal"). So the contract is that these cells are held
    out and end up not asserted malignant — the coherence gate downgrades whichever
    of them the mixture still puts in the tumor component.
    """
    rng = np.random.default_rng(0)
    n_norm, n_art, n_seg, n_pad = 500, 20, 20, 40
    total = n_seg + n_pad
    cn = rng.normal(0.0, 0.05, size=(n_norm + n_art, total))
    # Transcriptome-extreme, not CNV: large deviation on scattered single segments
    # with no contiguous run anywhere.
    for r in range(n_norm, n_norm + n_art):
        cols = rng.choice(total, size=12, replace=False)
        cn[r, cols] += rng.choice([-1, 1], size=12) * 3.0
    normal_mask = np.array([True] * n_norm + [False] * n_art)
    segments = _segments_for(total)

    df = classify_cells(
        cn_matrix=cn, segments=segments, normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    # Precondition: they are scattered by the same measure the fence uses.
    w = segments["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)
    assert np.median(_coherent_fraction(dev[n_norm:], segments)) < DEFAULT_COHERENCE_GATE

    assert df.attrs["n_outlier_fenced"] > 0, (
        "the fence did not fire on scattered transcriptome-extreme cells — the one "
        "case it exists for"
    )
    assert (df["class"].to_numpy()[n_norm:] == "tumor").sum() == 0
    # And the reference pool's LABELS are untouched. (The fence's high side only
    # considers ~normal_mask; its low side considers everyone, because holding a cell
    # out of the fit is not overriding its label.)
    assert (df["class"].to_numpy()[:n_norm] == "normal").all()


def _anti_aligned_cohort(n_norm=200, n_tumor=50, n_art=20, art_amp=1.2,
                         n_seg=20, n_pad=40, seed=3):
    """Normals + a coherent clone + a coherent ANTI-ALIGNED normal population.

    The artifacts here are true normals carrying a large contiguous deviation that
    opposes the clone — erythrocyte/platelet-like cells under a signed score. Sized at
    20/270 = 7.4% of the sample, deliberately above the 1% the winsorization can bound.

    Returns (cn, truth_normal, all_normals_pool, n_norm, n_tumor, n_art) where
    ``truth_normal`` marks every genuinely diploid cell (plain normals AND artifacts)
    and ``all_normals_pool`` is the "every normal labelled" reference mask.
    """
    rng = np.random.default_rng(seed)
    total = n_seg + n_pad
    half = n_seg // 2
    n = n_norm + n_tumor + n_art
    cn = rng.normal(0.0, 0.05, size=(n, total))
    cn[n_norm:n_norm + n_tumor, :half] += 0.5
    cn[n_norm:n_norm + n_tumor, half:n_seg] -= 0.5
    cn[n_norm + n_tumor:, :half] -= art_amp
    cn[n_norm + n_tumor:, half:n_seg] += art_amp
    truth_normal = np.zeros(n, dtype=bool)
    truth_normal[:n_norm] = True
    truth_normal[n_norm + n_tumor:] = True
    return cn, truth_normal, truth_normal.copy(), n_norm, n_tumor, n_art


def _heldout_pool(truth_normal, frac, seed=0):
    """Label only ``frac`` of the true normals — what a real run supplies.

    ``normal_mask`` is never the full normal population: pick_baseline's tiers each
    return a subset and a supervised run gets whatever barcodes the user labelled. A
    fixture that supplies all of them lets the mask's own override produce the right
    answer no matter what the classifier did, which is how several defects here stayed
    invisible. Cells outside the returned pool are the ones whose labels the
    classifier actually has to earn.
    """
    rng = np.random.default_rng(seed)
    pool = truth_normal.copy()
    idx = np.where(truth_normal)[0]
    drop = rng.permutation(idx)[int(len(idx) * frac):]
    pool[drop] = False
    return pool


@pytest.mark.parametrize("frac", [1.0, 0.5, 0.25])
def test_heldout_pool_does_not_call_unlabelled_normals_tumor(frac):
    """A large anti-aligned population must not collapse the tumor/normal split.

    Winsorization clips a FIXED FRACTION (1% per tail), so it answers the "one extreme
    cell" case and nothing larger. With the anti-aligned population at 7.4% of the
    sample the P1 floor landed inside it, most of it survived unclipped, and the
    2-component mixture spent one component describing IT — leaving all 200 plain
    normals in the same component as all 50 tumor cells (measured means -1.144 and
    +0.102). The mixture had stopped separating tumor from normal altogether.

    That was invisible at frac=1.0: every plain normal was in ``normal_mask`` and got
    overridden to "normal", so the emitted labels were exactly right while the
    mechanism underneath was inverted. Take the pool down to a subset and 90 of the
    unlabelled normals come out as "tumor". This test asserts on the UNLABELLED
    normals for that reason — they are the only cells whose label the classifier had
    to earn.
    """
    cn, truth_normal, _, n_norm, n_tumor, n_art = _anti_aligned_cohort()
    pool = _heldout_pool(truth_normal, frac)
    df = classify_cells(
        cn_matrix=cn, segments=_segments_for(cn.shape[1]), normal_mask=pool,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    cls = df["class"].to_numpy()
    held_out = truth_normal & ~pool
    tumor_rows = np.zeros(cn.shape[0], dtype=bool)
    tumor_rows[n_norm:n_norm + n_tumor] = True

    if held_out.any():
        called_tumor = int((cls[held_out] == "tumor").sum())
        assert called_tumor == 0, (
            f"pool={pool.sum()}/{truth_normal.sum()}: {called_tumor} of "
            f"{int(held_out.sum())} unlabelled true normals called tumor — the mixture "
            "is not separating tumor from normal"
        )
    assert (cls[tumor_rows] == "tumor").mean() >= 0.9, (
        f"pool={pool.sum()}/{truth_normal.sum()}: tumor recall "
        f"{(cls[tumor_rows] == 'tumor').mean()}"
    )
    # The anti-aligned population is held out of the FIT at every pool size, including
    # frac=1.0 where it is entirely inside normal_mask. Restricting the low side of the
    # fence to ~normal_mask left the broken mixture unfixed in exactly the case the
    # label override would hide.
    assert df.attrs["n_outlier_fenced"] >= n_art


def test_heldout_pool_mixture_separates_tumor_from_normal():
    """Pin the mechanism, not just the labels: check the split the GMM actually made.

    ``classify_cells`` output can look perfect while the mixture underneath is
    inverted, because the normal_mask override repairs it. Calling gmm_classify with a
    subset pool removes that repair for the held-out cells, so their labels report what
    the mixture really did.
    """
    cn, truth_normal, _, n_norm, n_tumor, n_art = _anti_aligned_cohort()
    pool = _heldout_pool(truth_normal, 0.5)
    segments = _segments_for(cn.shape[1])
    w = segments["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, pool, seg_weights=w)
    scores = template_projection(dev, consensus_template(dev, w), w)

    df = gmm_classify(
        scores, normal_mask=pool,
        coherence_of=lambda idx: _coherent_fraction(dev[idx], segments),
    )
    cls = df["class"].to_numpy()
    # Plain normals on the diploid side, the clone on the tumor side, the anti-aligned
    # population on neither — all without a gate or an override involved.
    assert (cls[:n_norm] == "tumor").sum() == 0
    assert (cls[n_norm:n_norm + n_tumor] == "tumor").mean() >= 0.9
    assert (cls[n_norm + n_tumor:] == "tumor").sum() == 0
    assert df.attrs["n_outlier_fenced"] >= n_art


@pytest.mark.parametrize("n_tumor", [50, 10, 5, 2])
def test_anti_alignment_gate_declines_below_the_tumor_cell_floor(n_tumor):
    """The gate must not estimate "the malignant population's level" from a few cells.

    Both of its thresholds are medians over the cells called tumor. Below the floor
    that median IS one or two cells: measured across 6 seeds differing in nothing else,
    the score threshold spanned 8.2x at 5 tumor calls, and the gate downgraded 5 cells
    on two seeds and 0 on the other four at the same purity. Above the floor it runs
    normally; below it, it declines and says so via ``anti_alignment_tumor_n``.
    """
    seen = set()
    for seed in range(4):
        cn, truth_normal, _, n_norm, nt, n_art = _anti_aligned_cohort(
            n_norm=400, n_tumor=n_tumor, n_art=5, art_amp=0.35, seed=100 + seed,
        )
        pool = np.zeros(cn.shape[0], dtype=bool)
        pool[:n_norm] = True          # artifacts unlabelled, so gate-eligible
        df = classify_cells(
            cn_matrix=cn, segments=_segments_for(cn.shape[1]), normal_mask=pool,
            barcodes=[f"cell{i}" for i in range(cn.shape[0])],
            discover_subclones_enabled=False,
        )
        n_called = int((df["class"].to_numpy() == "tumor").sum())
        ref_n = df.attrs["anti_alignment_tumor_n"]
        if n_called < MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT:
            assert ref_n == 0, (
                f"gate ran on {n_called} tumor calls, below the floor of "
                f"{MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT}"
            )
            assert df.attrs["n_anti_aligned"] == 0
        else:
            assert ref_n == n_called
        seen.add(df.attrs["n_anti_aligned"])
    # Whatever it does, it does the same thing on every seed — the flapping this floor
    # exists to stop was 5/0/5/0 across seeds at the same purity.
    assert len(seen) == 1, f"gate output varies with the RNG seed alone: {seen}"


@pytest.mark.parametrize("n_ref", [200, 100, 50, 30, 10])
def test_outlier_fence_holds_when_reference_pool_is_a_subset(n_ref):
    """The reference pool is a SUBSET of the normals, and the guard must survive it.

    ``normal_mask`` is never the full normal population: pick_baseline's signature,
    variance and gmm_fallback tiers each return a subset, and a supervised run gets
    whatever barcodes the user labelled. So ``scores[~normal_mask]`` is dominated by
    UNLABELLED NORMALS, and any guard phrased against its median (or any low quantile)
    is measuring the normal mode while believing it is measuring the tumor mode.

    Regression: an earlier guard compared the fence to ``median(scores[~nm])``. With
    every normal supplied as the reference pool — the only case the sibling tests
    cover — that median IS the tumor mode and the guard behaves. Take the pool down
    to a subset and it becomes ~0, the guard passes trivially, the 12x-reference-IQR
    fence lands below the tumor mode, and EVERY malignant cell is locked to "normal":
    recall 1.00 at 200/200 and 0.00 at 100, 50 or 30 (fence 0.159, tumor mode 0.398,
    non-reference median 0.008). The fence's second condition is now the candidate's
    own coherent fraction, which does not depend on how much of the normal population
    was labelled — or on any other property of the score distribution.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(
        n_normal=200, n_tumor=25, n_segments=20, n_diploid_padding=40,
    )
    pool = normal_mask.copy()
    pool[np.where(normal_mask)[0][n_ref:]] = False

    df = classify_cells(
        cn_matrix=cn, segments=_segments_for(cn.shape[1]), normal_mask=pool,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    cls = df["class"].to_numpy()
    recall = (cls[~normal_mask] == "tumor").mean()
    assert recall >= 0.9, f"reference pool {n_ref}/200: tumor recall {recall}"
    # The unlabelled normals must not be swept up either.
    unlabelled_normal = normal_mask & ~pool
    if unlabelled_normal.any():
        assert (cls[unlabelled_normal] == "tumor").mean() <= 0.1
    assert df.attrs["n_outlier_fenced"] == 0


def test_gmm_classify_bounds_both_tails_of_a_signed_score():
    """Winsorization is symmetric now that the score can go negative.

    An unbounded negative tail lets a single anti-aligned cell claim the GMM's low
    component, after which the ordinary normal and tumor modes share the high one.
    """
    rng = np.random.default_rng(0)
    scores = np.concatenate([
        rng.normal(0.0, 0.02, size=100),   # normals
        rng.normal(0.5, 0.05, size=50),    # tumor
        np.array([-40.0]),                 # one extreme anti-aligned cell
    ])
    normal_mask = np.array([True] * 100 + [False] * 51)

    df = gmm_classify(scores, normal_mask=normal_mask)
    cls = df["class"].to_numpy()

    assert (cls[100:150] == "tumor").sum() >= 45, (
        "the negative outlier displaced the tumor component"
    )
    # tumor_score is reported unmodified — winsorization is a fitting-time input
    # transform, never a change to what is written out.
    np.testing.assert_allclose(df["tumor_score"].to_numpy(), scores)


def test_classify_cells_tolerates_empty_segmentation():
    """detect_segments() legitimately returns zero rows; that must not crash.

    No chromosome reaching min_seg_genes is a supported outcome. The per-cell
    weighted median used to raise `argmax of an empty sequence` on the resulting
    (n_cells, 0) matrix — a regression against the previous classifier, which
    completed with all-zero scores.
    """
    empty = pd.DataFrame({
        "chr": pd.Series([], dtype=str), "n_genes": pd.Series([], dtype=int),
    })
    df = classify_cells(
        cn_matrix=np.zeros((10, 0)), segments=empty,
        normal_mask=np.ones(10, dtype=bool),
        barcodes=[f"cell{i}" for i in range(10)],
        discover_subclones_enabled=False,
    )
    assert len(df) == 10
    assert (df["class"].to_numpy() == "normal").all()
    assert (df["tumor_score"].to_numpy() == 0).all()
    assert (df["cn_burden"].to_numpy() == 0).all()


def _diploid_panels(cn, normal_mask):
    """(artifacts kept when ranking on burden, when ranking on the signed score)."""
    from kopya.heatmap import _confident_diploid_mask

    segments = _segments_for(cn.shape[1])
    df = classify_cells(
        cn_matrix=cn, segments=segments, normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    cls = df["class"].to_numpy()
    low_c = df["low_complexity"].to_numpy()
    on_burden, _ = _confident_diploid_mask(cls, low_c, df["cn_burden"].to_numpy(), 0.5)
    on_signed, _ = _confident_diploid_mask(cls, low_c, df["tumor_score"].to_numpy(), 0.5)
    # Artifacts are rows 100+.
    return int(np.asarray(on_burden)[100:].sum()), int(np.asarray(on_signed)[100:].sum()), df


def test_diploid_baseline_ranks_on_magnitude_not_signed_score():
    """heatmap's recentering baseline must not select anti-aligned cells.

    _confident_diploid_mask keeps normals at or below a score quantile. Under a
    signed projection the MINIMUM is the most anti-aligned cell — one carrying a
    real event in the opposite direction — not the most diploid one. That baseline
    feeds _recenter and propagates into the heatmap, the denoised matrix,
    chr_cnv_matrix.csv, {sample}_clones.seg and the matched-bulk concordance.

    Uses MILD artifacts (0.2 against a 0.5 clone) precisely because the
    anti-alignment gate is deliberately conservative and does not fire on them.
    That is what keeps this a test of the ranking rather than of the gate: the gate
    removes only the extreme tail, so ranking on `cn_burden` remains load-bearing
    for everything below it. The complementary regime is the next test.
    """
    on_burden, on_signed, df = _diploid_panels(*_cohort_with_artifacts(10, 0.2))

    assert df.attrs["n_anti_aligned"] == 0, "fixture is meant to sit below the gate"
    assert on_burden == 0
    assert on_signed > 0, (
        "ranking the diploid panel on the signed score no longer selects "
        "anti-aligned cells — if a change made that safe, this test and the "
        "cn_burden ranking in heatmap._confident_diploid_mask should be revisited "
        "together"
    )


def test_anti_alignment_gate_keeps_extreme_artifacts_out_of_the_diploid_panel():
    """Above the gate's threshold, neither ranking can select the artifacts.

    The gate downgrades a large, coherent, anti-aligned cell to "uncertain", and
    _confident_diploid_mask only ever keeps cells called "normal" — so the extreme
    tail is removed from the recentering baseline before the ranking question even
    arises. This is the defence-in-depth half of the previous test: burden ranking
    handles the mild cases, the gate handles the extreme ones.
    """
    on_burden, on_signed, df = _diploid_panels(*_cohort_with_artifacts(10, 6.0))

    assert df.attrs["n_anti_aligned"] == 10
    assert on_burden == 0
    assert on_signed == 0


def test_segment_burden_is_the_reported_cn_burden():
    """cn_burden is exactly segment_burden over the shared deviation.

    Also pins the third name for this same quantity: ``compute_tumor_scores`` is
    the public entry point CHANGELOG.md offers as the migration path for code that
    relied on the pre-2.0 unsigned ``tumor_score``, and ``classify_cells`` computes
    the value inline rather than calling it. Two implementations of one definition
    can drift; this asserts they do not.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(n_segments=20, n_diploid_padding=40)
    segments = _segments_for(cn.shape[1])
    w = segments["n_genes"].to_numpy()

    df = classify_cells(
        cn_matrix=cn, segments=segments, normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)
    np.testing.assert_allclose(
        df["cn_burden"].to_numpy(), segment_burden(np.abs(dev), w), rtol=1e-12,
    )
    np.testing.assert_allclose(
        df["cn_burden"].to_numpy(),
        compute_tumor_scores(cn, normal_mask, seg_weights=w),
        rtol=1e-12,
    )
