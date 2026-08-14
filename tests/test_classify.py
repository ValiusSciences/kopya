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


def test_consensus_template_opposing_subclones_is_a_known_limitation():
    """Two equal, exactly-opposing clones: one is recovered, one is erased.

    consensus_template estimates ONE direction. With two opposing high-burden
    subclones the seed set holds both, and per-cell noise — not population size —
    breaks the tie: the template locks onto whichever clone's direction wins, that
    clone scores positive and is called tumor, and the other scores symmetrically
    NEGATIVE and is called normal. Measured here: template L1 7.63 (it does not
    collapse), clone A median score -0.43, clone B +0.44, calls 0/50 and 50/50.

    This is the dangerous shape of the limitation, and it is the one armish's review
    reproduced: a whole malignant population is hidden behind a confident-looking
    result rather than both clones failing visibly.

    NOTE on this test's history. It previously asserted the opposite — that NEITHER
    clone is recovered — and passed, because the outlier fence's activation guard was
    comparing against median(scores[~normal_mask]) and firing spuriously, locking both
    clones to "normal". The symmetric outcome was an artifact of that bug, and fixing
    the guard (see test_outlier_fence_holds_when_reference_pool_is_a_subset) exposed
    the real behaviour. Resolving it needs multiple templates with best-aligned
    scoring, which changes the scoring contract rather than the estimator and is
    deliberately out of scope here.

    This pins the CURRENT behaviour so the limitation stays visible and a future
    multi-template change has something to flip. It is a bug with a documented shape,
    not a property worth preserving.
    """
    rng = np.random.default_rng(1)
    n_norm = n_a = n_b = 50
    n_seg, n_pad = 20, 40
    total = n_seg + n_pad
    half = n_seg // 2
    cn = rng.normal(0.0, 0.05, size=(n_norm + n_a + n_b, total))
    cn[n_norm:n_norm + n_a, :half] += 0.6
    cn[n_norm:n_norm + n_a, half:n_seg] -= 0.6
    cn[n_norm + n_a:, :half] -= 0.6
    cn[n_norm + n_a:, half:n_seg] += 0.6
    normal_mask = np.array([True] * n_norm + [False] * (n_a + n_b))

    df = classify_cells(
        cn_matrix=cn, segments=_segments_for(total), normal_mask=normal_mask,
        barcodes=[f"cell{i}" for i in range(cn.shape[0])],
        discover_subclones_enabled=False,
    )
    cls = df["class"].to_numpy()

    called_a = int((cls[n_norm:n_norm + n_a] == "tumor").sum())
    called_b = int((cls[n_norm + n_a:] == "tumor").sum())
    recovered, erased = sorted((called_a, called_b))
    # Exactly one clone survives, the other is silently called normal. When the
    # multi-template change lands, BOTH should be >= 45 and this assertion is what
    # it has to flip.
    assert erased >= 45, f"expected one clone recovered, got {called_a}/50 and {called_b}/50"
    assert recovered <= 5, (
        f"clone silently erased: {called_a}/50 and {called_b}/50 called tumor — "
        "if both are now recovered, the multi-template fix landed and this "
        "known-limitation test should be replaced"
    )
    # The erased clone is not merely unconfident: it scores on the wrong SIDE of
    # diploid, which is why nothing downstream flags it.
    assert df["tumor_score"].to_numpy()[cls == "normal"].min() < -0.2


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
    """The fence's activation guard must be stated in score units.

    It used to be compared against clip_ceiling. Once that ceiling became a global
    quantile the two were incommensurable — the fence is in reference-pool IQR
    units — and the comparison silently stopped passing, disabling the exclusion
    entirely. The guard now asks whether the fence sits above the fitted tumor mode,
    which is the thing it must not cut into.
    """
    cn, normal_mask, _ = _bimodal_cn_matrix(n_segments=20, n_diploid_padding=40)
    w = _segments_for(cn.shape[1])["n_genes"].to_numpy()
    dev = reference_relative_deviation(cn, normal_mask, seg_weights=w)
    scores = template_projection(dev, consensus_template(dev, w), w)

    df = gmm_classify(scores, normal_mask=normal_mask)
    # A cleanly-separating score must not have its whole malignant population
    # fenced to "normal" — the failure the old reference-relative fence produced
    # whenever the guard did let it run.
    assert (df["class"].to_numpy()[50:] == "tumor").sum() >= 45
    assert df.attrs["n_outlier_fenced"] == 0


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
    non-reference median 0.008). The guard now compares against a provisional GMM's
    fitted high component, which does not depend on how much of the normal population
    was labelled.
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


def test_diploid_baseline_ranks_on_magnitude_not_signed_score():
    """heatmap's recentering baseline must not select anti-aligned cells.

    _confident_diploid_mask keeps normals at or below a score quantile. Under a
    signed projection the MINIMUM is the most anti-aligned cell — one carrying a
    large real event in the opposite direction — not the most diploid one. That
    baseline feeds _recenter and propagates into the heatmap, the denoised matrix,
    chr_cnv_matrix.csv, {sample}_clones.seg and the matched-bulk concordance.
    """
    from kopya.heatmap import _confident_diploid_mask

    cn, normal_mask = _cohort_with_artifacts(10, 6.0)
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

    # Artifacts are rows 100+. Ranking on the signed score pulls them in (they are
    # the most negative cells in the cohort); ranking on burden excludes them.
    assert np.asarray(on_burden)[100:].sum() == 0
    assert np.asarray(on_signed)[100:].sum() > 0


def test_segment_burden_is_the_reported_cn_burden():
    """cn_burden is exactly segment_burden over the shared deviation."""
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
