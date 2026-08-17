"""Step §3.5 — classify cells as tumor/normal/uncertain and discover subclones.

Three steps:
    1. scoring, in three parts: reference_relative_deviation() recovers the
       per-cell/per-segment deviation (per-segment normal-pool median, then the
       cell's own gene-count-weighted genome-wide median); consensus_template()
       estimates this sample's CN profile label-free from its most-aneuploid
       cells; template_projection() scores each cell by its SIGNED alignment with
       that template. Cells with no CN score near 0, cells carrying the clone's
       events score positive, cells deviating against it score negative.

       compute_tumor_scores() computes the older non-negative magnitude burden
       over the same deviation. It is no longer on the classify_cells() path —
       classify_cells inlines deviation -> template -> projection — but remains
       exported and supported for callers that want the magnitude directly; it is
       the same quantity now reported as the ``cn_burden`` column.

    2. gmm_classify(): 2-component Gaussian Mixture on the tumor-score vector.
       Higher-mean component = tumor; lower = normal. Posterior probabilities
       below --call-confidence (default 0.5) collapse to the "uncertain" label
       so downstream notebooks can filter them out cleanly.

    3. discover_subclones(): Leiden clustering on the per-cell × per-segment
       reference-relative deviation, restricted to tumor cells. Resolution is
       data-driven via a sweep, capped at max_subclones (default 5) to keep the
       clone-tree interpretable.

classify_cells() composes the three into a single per-cell DataFrame.
"""

from anndata import AnnData
from numpy import (
    abs as np_abs,
    arange,
    argmax,
    argsort,
    asarray,
    clip,
    cumsum,
    dtype,
    empty,
    full,
    median,
    minimum as np_minimum,
    ones,
    percentile,
    sign as np_sign,
    subtract,
    take_along_axis,
    unique,
    where,
    zeros,
    zeros_like,
)
from pandas import DataFrame, Series
from scipy.ndimage import uniform_filter1d
from scanpy.pp import neighbors as sc_neighbors
from scanpy.pp import pca as sc_pca
from scanpy.tl import leiden as sc_leiden
from sklearn.mixture import GaussianMixture


# Posterior threshold below which a cell is labeled "uncertain". 0.5 keeps
# clear calls while reducing false negatives from borderline tumor cells,
# especially important for hematological tumors with modest CNV signal.
DEFAULT_CALL_CONFIDENCE = 0.5

# Outlier fence multiplier for the IQR-based exclusion in gmm_classify.
# Non-baseline cells scoring above Q75 + OUTLIER_FENCE_MULT * IQR of the baseline
# score distribution become CANDIDATES for exclusion from the GMM fit; whether a
# candidate is actually held out turns on its coherent fraction, because scoring
# far above the reference pool is equally true of a rare real clone (see the fence
# for the measurements). This keeps transcriptomically extreme cell types
# (Erythrocytes, Platelets), whose high scores are driven by transcriptome mismatch
# rather than CNV, from dragging a mixture component onto themselves.
#
# The height of the fence is therefore not a recall/precision dial the way it reads:
# in a cleanly separating sample every malignant cell clears it and is held only by
# its coherence. Widening it from 7.0 to 12.0 (1.0.1) was a no-op for that reason and
# because no version of the second condition that shipped was ever reachable.
DEFAULT_OUTLIER_FENCE_MULT = 12.0

# Maximum number of subclones to emit. Capping keeps the per-clone .seg file
# interpretable and matches CopyKAT's typical post-hoc dendrogram cut at k<=5.
DEFAULT_MAX_SUBCLONES = 5

# Leiden resolution sweep used to find a partition that respects max_subclones.
# Walks from coarse → fine; first resolution yielding <= max_subclones wins.
SUBCLONE_RESOLUTION_SWEEP = (0.3, 0.5, 0.8, 1.2, 1.8)

# Minimum tumor cells required to even attempt subclone discovery. Below this,
# subclones are noise; we collapse to a single clone.
MIN_TUMOR_CELLS_FOR_SUBCLONES = 30

# Minimum tumor calls the anti-alignment gate's reference level may be estimated from.
# The gate places both of its thresholds at ANTI_ALIGNED_FRACTION of the median score
# and median burden of the cells called tumor, so when few cells are called the median
# IS one or two cells and the whole gate moves with the noise in them. Measured on a
# 400-normal fixture with a planted clone, varying only the RNG seed across 6 runs:
# the score threshold spans 1.1x at ~35 tumor calls, 2.4x at 10, and 8.2x at 5 — and
# the gate's output flips with it, downgrading 5 cells on two seeds and 0 on the other
# four at the same purity. There is no meaningful "typical tumor cell" to be half of
# below this, so the gate declines to run rather than run off noise; the count it
# rested on reaches qc.json as ``anti_alignment_tumor_n`` (0 = did not run) so a
# skipped gate is visible instead of silent. Set to match
# MIN_TUMOR_CELLS_FOR_SUBCLONES, the file's existing answer to the same question.
MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT = 30

# Cells whose detected-gene count is below this fraction of the cohort-median
# detected-gene count are flagged low_complexity (ambient / empty-droplet-like).
# Their CN signal is dominated by sampling noise, so a "tumor" call on them is
# untrustworthy — the classifier downgrades it to "uncertain" rather than
# emitting a false positive. The cell is still scored and reported (flagged),
# never silently dropped.
#
# The threshold is cohort-relative (fraction of the median), so it adapts to
# sequencing depth but is only as good as the median: in a cohort dominated by
# ambient cells the median itself is low and fewer cells clear the bar. It is a
# subtractive safety net (only downgrades tumor→uncertain), so at worst it does
# nothing — never worse than leaving these cells as tumor. Supply curated
# barcodes / upstream QC when ambient contamination is severe.
DEFAULT_LOW_COMPLEXITY_FRAC = 0.5

# Minimum fraction of a cell's total CN deviation that must be coherent
# (chromosome-arm-scale) for a "tumor" call to stand. Real CNVs shift whole
# chromosome arms together, so a genuine tumor cell's deviation is mostly
# coherent; a high score built from scattered, single-segment spikes is more
# likely transcriptional/lineage noise (a "loud" specialized normal cell) than
# copy number, and is downgraded to "uncertain". Conservative by design —
# broadly-aneuploid tumor cells clear it easily.
DEFAULT_COHERENCE_GATE = 0.45

# Burden quantile above which a cell seeds the consensus CN template (see
# consensus_template). 0.75 = the top quartile by aneuploidy burden. High enough
# that the seed set is dominated by genuinely altered cells on a low-purity
# sample, low enough that the average is over thousands of cells on a real
# cohort, so clonal CN survives it and per-cell noise does not.
DEFAULT_TEMPLATE_SEED_QUANTILE = 0.75

# Moving-average width (in segments) for the local-contiguity coherence measure.
# Small so it captures short contiguous runs without spanning whole chromosomes.
COHERENCE_WINDOW = 5

# Fraction of the malignant population's own level at which the anti-alignment gate
# (see classify_cells) starts calling a "normal" cell suspicious. The gate asks
# whether a cell is in the same LEAGUE as the tumor population while pointing the
# opposite way — not whether it out-does it. Anchoring at the tumor median instead
# makes the test a knife edge on the case it exists for: two symmetric opposing
# clones sit at ~1.0x each other's level, so a median threshold splits the opposing
# clone roughly in half by noise (measured on the opposing-subclone fixture: 9 of 50
# cells over the score threshold, 22 of 50 over the burden one). The populations this
# has to separate are ~4x apart in both quantities on real data — an anti-aligned
# artifact carries ~0.2x the malignant burden, an opposing clone ~1.0x — so the
# midpoint clears both with a wide margin rather than bisecting either.
ANTI_ALIGNED_FRACTION = 0.5


def weighted_median_rows(mat, weights):
    """Per-row weighted median of ``mat`` with per-column weights.

    The weighted median is the value at which the cumulative column weight — in
    ascending value order — first reaches half the total. Weighting by segment
    gene count keeps a cell's genome-wide center tied to the genomic majority
    rather than the segment count: otherwise many short segments packed into one
    altered region could pull the per-cell baseline and make the unchanged
    majority read as a gain or loss.

    Lives here (rather than in the heatmap module that first needed it) because
    the per-cell genome-wide center is part of recovering the CN signal, not part
    of drawing it — ``compute_tumor_scores`` and ``heatmap._recenter`` must use
    the same definition or the classifier and the figures disagree about where
    diploid sits.

    Computed in row blocks rather than over the whole matrix at once. The
    calculation needs three (n_rows × n_cols) intermediates — an int64 ``order``,
    the reordered values, and a float64 cumulative weight — so a whole-matrix
    version costs ~8x the input in temporaries: at the 800k cells x 500 segments
    this package documents as commodity-memory-safe that is roughly 8 GB, plus a
    cumsum temporary, on a path both the classifier and the heatmap now take. The
    block loop caps the intermediates at ~64 MB each while keeping every inner
    operation vectorized; results are identical (blocks are independent — the
    weighted median is per-row).

    Args:
        mat: (n_rows × n_cols) ndarray.
        weights: (n_cols,) non-negative column weights.

    Returns:
        (n_rows,) ndarray of per-row weighted medians. All-zero when there are no
        columns — detect_segments() legitimately returns zero rows when no
        chromosome reaches min_seg_genes, and an empty argmax would otherwise
        raise on an input the pipeline is documented to tolerate.
    """
    n_rows, n_cols = mat.shape
    if n_cols == 0:
        return zeros(n_rows, dtype="float64")
    w = asarray(weights, dtype="float64")
    total = float(w.sum())
    if total <= 0.0:
        # No weight anywhere: there is no majority to sit at, so there is no center
        # to report. Returning zeros leaves the deviation uncentered, which is the
        # honest answer; falling through would set half=0, make ``cumw >= half`` true
        # at position 0, and return each row's MINIMUM — subtracting which would make
        # every deviation non-negative and erase all loss signal. segment_burden
        # already guards its own total the same way; this is now a public function
        # shared with heatmap._recenter, so the asymmetry is worth closing even
        # though detect_segments (n_genes >= min_seg_genes) cannot currently produce it.
        return zeros(n_rows, dtype="float64")
    half = 0.5 * total
    out = zeros(n_rows, dtype="float64")
    # ~8M float64 per intermediate == 64 MB per block, whatever the segment count.
    block = max(1, int(8_000_000 // n_cols))
    for start in range(0, n_rows, block):
        chunk = mat[start:start + block]
        order = argsort(chunk, axis=1)
        sorted_vals = take_along_axis(chunk, order, axis=1)
        cumw = cumsum(w[order], axis=1)
        # First position whose cumulative weight reaches the halfway mark.
        idx = argmax(cumw >= half, axis=1)
        out[start:start + block] = sorted_vals[arange(chunk.shape[0]), idx]
    return out


def _weighted_row_sum(mat, weights):
    """``(mat * weights).sum(axis=1)`` without a full-size float64 temporary.

    Written out because the obvious spelling silently defeats the float32 invariant
    the rest of this path maintains. ``mat`` is float32 (see
    reference_relative_deviation, which goes to explicit trouble to keep it that way)
    while a gene-count weight vector is float64, and ``float32 * float64`` promotes:
    the elementwise product materializes a full ``(n_cells x n_segments)`` FLOAT64
    array before the reduction ever runs. At the 800k x 500 shape this package's
    docstrings use as the commodity-memory budget that is ~3.2 GB per call, on the
    same path where ``weighted_median_rows`` blocks its own intermediates to 64 MB.

    Blocking the reduction bounds the temporary at one block instead, and the
    arithmetic is UNCHANGED — the reduction is per-row and rows are independent, so
    each row sees the same values summed in the same order as the unblocked
    expression. Casting the weights down to float32 would also fix the memory but
    would round every product, so results would move; this way they do not.

    Args:
        mat: (n_rows x n_cols) ndarray, any float dtype.
        weights: (n_cols,) per-column weights.

    Returns:
        1-D float64 array len == mat.shape[0].
    """
    w = asarray(weights, dtype="float64")
    n_rows, n_cols = mat.shape
    out = empty(n_rows, dtype="float64")
    # ~8M float64 per intermediate == 64 MB per block, whatever the segment count.
    block = max(1, int(8_000_000 // max(n_cols, 1)))
    for start in range(0, n_rows, block):
        out[start:start + block] = (mat[start:start + block] * w).sum(axis=1)
    return out


def _normal_pool_baseline(cn_matrix, normal_mask):
    """Per-segment CN baseline: the median over the normal pool.

    Single source of truth for the "median over normals, else cohort median"
    fallback so the definition can't drift between the callers that need it
    (tumor score, coherence gate, n_segments_altered). If the normal pool is
    empty (defensive — pick_baseline always returns at least the GMM fallback),
    falls back to the whole-cohort median so scores stay meaningful.
    """
    if normal_mask.sum() > 0:
        return median(cn_matrix[normal_mask, :], axis=0)
    return median(cn_matrix, axis=0)


def _coherent_fraction(dev, segments, window=COHERENCE_WINDOW):
    """Per-cell fraction of CN deviation that is *locally contiguous*.

    A real CNV shows up as a run of same-sign deviation across neighbouring
    segments (a gained arm lifts every segment on it); transcriptional/lineage
    noise shows up as isolated spikes or sign-alternating scatter. We separate
    the two by asking, per segment, whether the local neighbourhood corroborates
    the deviation: the coherent part of a segment is ``min(|dev|, |local mean|)``
    where their signs agree, else 0. Summed and divided by ``L1(raw)`` this gives
    a value in [0, 1]:
      • contiguous run  -> local mean ≈ dev            -> counts ~fully  (→ ~1)
      • isolated spike  -> local mean ≈ dev / window   -> heavily diluted (→ ~1/window)
      • alternating     -> local mean ≈ 0              -> ~0

    Using a neighbourhood corroboration (not just the smoothed L1, which a mean
    filter conserves for isolated spikes) is what lets it downgrade a scattered
    *same-sign* cell — a "loud" specialized normal called tumor on magnitude
    alone — and not only strictly-alternating profiles.

    The window is local by design: on a chromosome with more than ``window``
    segments a gain and a loss on opposite arms stay two separate runs rather than
    cancelling. On a coarsely-segmented chromosome (≤ ``window`` segments — small
    chromosomes, or a large --min-seg-genes) the neighbourhood spans most of the
    chromosome, so a same-chromosome opposite-arm pair there partially cancels;
    this is inherent to having few segments and is tolerable because a cell's
    coherent fraction is dominated by its many well-segmented chromosomes.

    Args:
        dev: (n_cells × n_segments) reference-relative deviation from
            reference_relative_deviation(). Pass only the rows you need scored
            (e.g. the tumor-called cells) — the result is per-row and independent
            of which other rows are present. Taking the deviation rather than
            (cn_matrix, baseline) keeps the gate measuring the exact same
            quantity the tumor score does, per-cell centering included: on a
            profile that still carries its per-cell global offset every segment
            shares one sign, which reads as maximally coherent and makes the gate
            unable to fire at all.
        segments: DataFrame from detect_segments(); the ``chr`` column groups
            segments into chromosomes (smoothing never crosses a boundary).
        window: moving-average width in segments, clamped per chromosome.

    Returns:
        1-D float array len == dev.shape[0], each in [0, 1].
    """
    total = np_abs(dev).sum(axis=1)
    chrom = asarray(segments["chr"])
    coherent = zeros_like(dev)
    for c in unique(chrom):
        idx = where(chrom == c)[0]
        block = dev[:, idx]
        w = min(window, block.shape[1])
        # Local neighbourhood mean (includes self). A segment's deviation counts
        # as coherent only to the extent its neighbourhood corroborates it — same
        # sign AND comparable magnitude. So the coherent part per segment is
        # min(|dev|, |local|) where the signs agree, else 0. This measures
        # *contiguity*, not just sign-cancellation:
        #   • contiguous run  -> |local| ≈ |dev|      -> counts ~fully
        #   • isolated spike  -> |local| ≈ |dev|/w    -> heavily diluted
        #   • alternating     -> |local| ≈ 0          -> ~0
        # Local mean over the ACTUAL in-bounds neighbours: zero-padded window sum
        # divided by the count of in-bounds positions in that window. "nearest"
        # padding would replicate a chromosome-edge value and hand an isolated edge
        # spike a spuriously high mean (letting it clear the gate); plain "constant"
        # zero-padding would instead dilute a legitimate edge-adjacent CNV run and
        # wrongly gate it. Normalising by the in-bounds count does neither — an
        # isolated edge spike averages to ~d/(radius+1) (gated) while a contiguous
        # run keeps its amplitude (retained).
        wsize = max(w, 1)
        ssum = uniform_filter1d(block, size=wsize, axis=1, mode="constant", cval=0.0)
        cnts = uniform_filter1d(ones(block.shape[1]), size=wsize, mode="constant", cval=0.0)
        local = ssum / cnts[None, :]
        agree = np_sign(block) == np_sign(local)
        coherent[:, idx] = where(agree, np_minimum(np_abs(block), np_abs(local)), 0.0)
    return coherent.sum(axis=1) / (total + 1e-9)


def _low_complexity_mask(complexity, frac, n_cells):
    """Boolean mask flagging ambient/empty-droplet-like cells by low complexity.

    A cell is flagged when its detected-gene count falls below ``frac`` × the
    cohort-median detected-gene count. Returns an all-False mask when complexity
    is unavailable or the median is non-positive (nothing to compare against).
    """
    if complexity is None:
        return full(n_cells, False)
    complexity = asarray(complexity, dtype=float)
    med = median(complexity)
    if med <= 0:
        return full(len(complexity), False)
    return complexity < frac * med


def reference_relative_deviation(cn_matrix, normal_mask, baseline=None, seg_weights=None):
    """The per-cell, per-segment CN deviation the classifier should reason about.

    Two subtractions, in this order — the same two ``heatmap._recenter`` applies
    (and therefore the same frame the heatmap, ``--denoise-outputs`` and the
    matched-bulk concordance are all computed in):

      1. per-segment median over the normal pool → reference cells sit at ~0;
      2. each cell's own genome-wide (gene-count-weighted) median → the per-cell
         global offset is removed.

    Step 2 is what this function exists for. ``center_against_baseline``
    subtracts the per-gene median over the normal pool, but any gene detected in
    <= 50% of normal cells has a median of exactly 0 and so passes through
    UNCENTERED — on real data that is the large majority of the genes that
    survive the detection-rate filter. Their summed contribution is a function of
    how many genes the cell happened to detect, i.e. of its library depth, and it
    is near-constant across segments. So every cell carries a global pedestal
    proportional to its sequencing depth, on top of its real regional CN.

    A magnitude score taken before removing that pedestal measures mostly depth:
    the offset enters every segment, so it scales with the segment count while
    real focal events do not. (Measured on this repo's benchmark cohort it is
    50-90% of the raw L1, and within known-diploid reference cells the resulting
    score correlates with detected-gene count at up to r = +0.88 — a correlation
    that cannot be copy number, because those cells have none.)

    A weighted median is used, not a mean: it is robust to the cell's genuinely
    altered segments, so a cell with a real gain on 30% of its genome keeps that
    gain in the residual instead of having it averaged into its own center.

    Known limitation: this assumes the truly-altered segments are a MINORITY
    of the cell's genome. If "altered" instead covers half or more (e.g. a
    near-whole-genome clonal event split across a gain block and a loss
    block), there is no genuine diploid majority for the median to find —
    whichever block covers more of the genome (or, at an exact tie, whichever
    side the median's tie-break happens to favor) gets absorbed as if it were
    the depth pedestal, silently erasing that block's real signal rather than
    an artifact. See test_reference_relative_deviation_exact_tie_is_a_known_
    limitation in tests/test_classify.py for a worked example.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray from per_cell_segment_cn().
        normal_mask: bool array len == n_cells, True for cells in normal pool.
        baseline: optional precomputed per-segment baseline (from
            _normal_pool_baseline); computed from normal_mask when omitted.
        seg_weights: (n_segments,) per-segment gene counts. When omitted, step 2
            is skipped and the raw per-segment deviation is returned (the
            pre-1.1 behaviour), so callers without a segment table still work.

    Returns:
        (n_cells × n_segments) ndarray of reference-relative deviations.
    """
    if baseline is None:
        baseline = _normal_pool_baseline(cn_matrix, normal_mask)
    # Keep the input's precision. cn_matrix is float32 (segment.per_cell_segment_cn)
    # but the baseline is a float64 median, so a plain subtraction promotes the whole
    # matrix and doubles a value that classify_cells then retains for its entire run.
    # At the 800k x 500 this package documents as commodity-memory-safe that is 3.2 GB
    # held instead of 1.6 GB, on the same path where weighted_median_rows goes to the
    # trouble of blocking its own intermediates to 64 MB. float32 is ample here: these
    # are log-ratios of order 1e-2 to 1e0, and every downstream reduction
    # (weighted_median_rows' cumsum, segment_burden, template_projection) accumulates
    # in float64 regardless.
    out_dtype = cn_matrix.dtype if cn_matrix.dtype.kind == "f" else dtype("float64")
    dev = subtract(cn_matrix, baseline, dtype=out_dtype)
    if seg_weights is not None:
        dev -= weighted_median_rows(dev, seg_weights)[:, None].astype(out_dtype)
    return dev


def compute_tumor_scores(cn_matrix, normal_mask, baseline=None, seg_weights=None):
    """Per-cell L1 distance from the normal-pool median CN vector.

    Computed on the reference-relative deviation (see
    ``reference_relative_deviation``), i.e. after both the per-segment normal
    baseline and the cell's own genome-wide median have been removed.

    **Called with ``seg_weights``, this is exactly the ``cn_burden`` column of
    ``prediction.csv``** — ``segment_burden`` over the same deviation, same weights,
    same frame. The duplication is deliberate and is pinned by
    ``test_segment_burden_is_the_reported_cn_burden`` so the two cannot drift:
    ``classify_cells`` computes the value inline (it already holds ``abs_dev`` and
    would otherwise recompute the whole deviation matrix a second time), while this
    entry point exists for callers who have a raw CN matrix and want the magnitude
    without running the classifier. It is the migration path CHANGELOG.md points at
    for code that relied on the pre-2.0 unsigned ``tumor_score``.

    **Without ``seg_weights`` it is a different quantity, not a slightly different
    one.** Two things change together: ``reference_relative_deviation`` skips the
    per-cell centering step (so the result still carries the depth pedestal this
    release exists to remove), and the sum is unweighted (so it partly measures
    segmentation granularity). Measured on a 40x30 fixture: 1.2717 against a
    ``cn_burden`` of 0.0414, a ~30x scale difference in a different frame. There is
    no way to default the weights here — a per-segment gene count cannot be
    recovered from ``cn_matrix`` alone — so a caller migrating off the pre-2.0
    ``tumor_score`` has to pass ``segments["n_genes"]`` to get the column's value.

    Despite the name it is NOT what the classifier scores on. ``classify_cells``
    ranks cells by the signed consensus-template projection (see
    ``template_projection``); the name is retained because renaming it would break
    the same callers this function exists to serve.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray from per_cell_segment_cn().
        normal_mask: bool array len == n_cells, True for cells in normal pool.
        baseline: optional precomputed per-segment baseline (from
            _normal_pool_baseline); computed from normal_mask when omitted. Lets
            classify_cells compute the baseline once and share it across the
            score, coherence gate, and n_segments_altered.
        seg_weights: (n_segments,) per-segment gene counts, i.e.
            ``segments["n_genes"]``. Forwarded to reference_relative_deviation for
            the per-cell centering step AND used to weight the sum. Omitting it
            disables both — see the note above; pass it to get ``cn_burden``.

    Returns:
        1-D float array of tumor scores, len == n_cells.
    """
    deviations = np_abs(
        reference_relative_deviation(
            cn_matrix, normal_mask, baseline=baseline, seg_weights=seg_weights,
        )
    )
    return segment_burden(deviations, seg_weights)


def segment_burden(deviations, seg_weights=None):
    """Aggregate a per-segment magnitude into one per-cell aneuploidy burden.

    L1 across segments per cell, so cells with broader alterations score higher
    than cells with one focal event — the "aneuploidy load" intuition behind
    CopyKAT/SCEVAN tumor calling.

    The sum is weighted by segment gene count and normalized to the genome, so
    the burden is a **genome-fraction-weighted** mean deviation rather than a
    per-segment count. Segments are not comparable units: PELT emits them from
    25 genes up to ~350 on real data (>10x), and the number it emits per
    chromosome is set by the noise it finds there, not by how much genome the
    chromosome holds. An unweighted sum therefore lets a chromosome that happened
    to be cut into twenty short noisy segments contribute twenty noise terms while
    a single long clean segment contributes one — the score partly measures how
    finely each region got segmented. Weighting by ``n_genes`` makes a cell's
    burden depend on how much of its genome is altered, which is what the
    quantity is supposed to mean, and makes it comparable across samples whose
    segmentation granularity differs.

    Uses the same weights as the per-cell centering above, so the segment a cell
    is centered on and the segment that contributes to its burden carry the same
    importance.

    Args:
        deviations: (n_cells × n_segments) non-negative per-segment magnitude.
        seg_weights: (n_segments,) per-segment gene counts. When omitted, falls
            back to the unweighted per-segment sum.

    Returns:
        1-D float array len == n_cells.
    """
    if seg_weights is None:
        return deviations.sum(axis=1)
    w = asarray(seg_weights, dtype="float64")
    total = float(w.sum())
    if total <= 0.0:
        # No segments (or all zero-gene): nothing to average over. Zero burden is
        # the honest answer and keeps the empty-segmentation path warning-free.
        return zeros(deviations.shape[0], dtype="float64")
    return _weighted_row_sum(deviations, w) / total


def consensus_template(
    dev, seg_weights=None, seed_quantile=DEFAULT_TEMPLATE_SEED_QUANTILE, exclude=None,
    abs_dev=None,
):
    """The sample's own consensus CN profile, estimated without any labels.

    A per-segment template of "what this sample's copy-number alterations look
    like", built in three passes over the same deviation matrix:

      1. score every cell by its label-free aneuploidy burden (``segment_burden``);
      2. take the most-aneuploid ``1 - seed_quantile`` fraction of cells as seeds
         and rescale each one to unit weighted-L1 norm;
      3. average those unit directions.

    Real CNVs are shared by a clonal population, so they survive that average;
    per-cell transcriptional noise is independent between cells and averages away.
    The result is a noisy but unbiased estimate of the malignant population's CN
    profile.

    Step 2 — the unit rescaling — is what keeps a seed's *direction* independent
    of its *magnitude*, and it is load-bearing rather than cosmetic. Seeds are
    selected by burden, which selects for amplitude, so under a plain mean over
    raw deviations the cells most able to set the template's sign are the
    transcriptome-extreme ones this classifier exists to reject. Measured on a
    50-reference/50-tumor synthetic with a clonal event at ±0.5: three anti-aligned
    cells at amplitude 6.0 (2.9% of the cohort) invert the raw-mean template
    outright — gain block +0.510 -> -0.239, loss block -0.500 -> +0.254 — after
    which all 50 true tumor cells are called ``normal`` and the three artifacts are
    called ``tumor``. Rescaled to unit norm those three cells cast three votes out
    of twenty-six and the template holds.

    The protection is a majority argument over the SEED SET, not over the cohort,
    so it has an exact bound: extreme cells sort to the top of the burden ranking
    and therefore enter the seed set first, so they outvote the clone once they
    exceed half of it — ``0.5 * (1 - seed_quantile)`` = 12.5% of the cohort at the
    default quantile. Measured on the synthetic above, the template holds at 11.5%
    contamination and inverts at 12.3%. Beyond that no label-free single-template
    estimator can tell which direction is the malignant one; the outlier fence in
    ``gmm_classify`` is the layer meant to remove such cells before they get here.

    A mean over unit directions rather than a per-segment median, because the
    median additionally requires the seed set to be majority-malignant. On a
    low-purity sample the top burden quartile can be minority-tumor, and an
    aligned minority still sums coherently under a mean while independent noise
    directions cancel — so the mean degrades gracefully where the median would
    return ~0 and call the sample flat.

    Note the template's overall scale is irrelevant downstream:
    ``template_projection`` divides by ``sum |w t|``, so scaling the template by
    any positive constant leaves every score unchanged. Only its shape matters.

    Known limitation: this estimates ONE consensus direction, so two high-burden
    subclones carrying opposing profiles cannot both be represented. The template
    does NOT collapse to zero at equal population size, as one might expect from a
    mean over cancelling directions — per-cell noise breaks the tie and the estimate
    locks onto one clone's direction (measured on a 50/50 synthetic: template L1
    7.63, not ~0). That clone then scores positive and is called tumor while the
    other scores symmetrically negative and is called normal: one malignant
    population is silently erased behind a confident-looking result, rather than
    both failing visibly. See
    test_consensus_template_opposing_subclones_is_a_known_limitation. Resolving it
    needs multiple templates with best-aligned scoring, which is a scoring-contract
    change rather than an estimator fix and is deliberately not attempted here.

    Deliberately does NOT use ``normal_mask`` or ``segments.tumor_mean``, both of
    which are available here. ``tumor_mean`` is the pooled mean over the
    complement of the reference pool, so on a supervised run it is a function of
    the user's own tumor/normal split — using it as the scoring template would
    make the discriminant partly a restatement of that input rather than a
    measurement of copy number. Seeding on burden instead keeps the template a
    property of the data. (Empirically, on this repo's benchmark cohort the
    label-free template scores marginally BETTER than the pooled one — mean AUC
    0.925 vs 0.920 — so nothing is given up for the independence.)

    Args:
        dev: (n_cells × n_segments) reference-relative deviation.
        seg_weights: (n_segments,) per-segment gene counts.
        seed_quantile: burden quantile above which a cell seeds the template.
        exclude: optional bool mask len == n_cells of cells barred from seeding.
            Ambient / empty-droplet-like cells belong here: their CN signal is
            noise-dominated, that noise is large in magnitude, and burden
            selection therefore concentrates them in exactly the set that sets the
            template's direction. Ignored if it would leave no eligible cells.
        abs_dev: optional precomputed ``abs(dev)``. Purely an allocation saver —
            classify_cells needs the same array for cn_burden and the altered-segment
            count, and materializing it three times costs three full copies of a
            matrix that is already the largest thing in the run.

    Returns:
        (n_segments,) float array: the consensus per-segment deviation.
    """
    if dev.shape[1] == 0:
        return zeros(0, dtype="float64")
    eligible = ones(dev.shape[0], dtype=bool)
    if exclude is not None:
        candidate = ~asarray(exclude, dtype=bool)
        if candidate.any():
            eligible = candidate
    if abs_dev is None:
        abs_dev = np_abs(dev)
    burden = segment_burden(abs_dev, seg_weights)
    thr = float(percentile(burden[eligible], 100.0 * seed_quantile))
    seed = eligible & (burden >= thr)
    # Degenerate guard: an exactly-constant burden makes the threshold equal the
    # maximum and could select nothing. Fall back to the whole cohort, which gives
    # a ~zero template and hence ~zero scores — the honest answer for a sample
    # with no detectable CN structure.
    if not seed.any():
        seed = eligible
    # Keep the seed subset in the matrix's own dtype. It used to be copied up to
    # float64 here, which bought nothing — every reduction below accumulates in
    # float64 anyway — and cost three full float64 copies of the subset (the upcast,
    # the np_abs, and the division result). At the default seed quantile that subset
    # is ~25% of the cohort, so on the 800k x 500 budget each copy was ~800 MB.
    seeds = dev[seed]
    # Rescale each seed to unit weighted-L1 norm so it votes on direction only.
    # Same aggregation as the burden that selected it, so "one unit of deviation"
    # means the same thing in both places. abs_dev, when the caller already holds it,
    # spares recomputing np_abs over the subset.
    abs_seeds = abs_dev[seed] if abs_dev is not None else np_abs(seeds)
    norms = segment_burden(abs_seeds, seg_weights)
    keep = norms > 0
    if not keep.any():
        # Every seed is exactly diploid — no direction to estimate.
        return zeros(dev.shape[1], dtype="float64")
    # Mean of the unit-normed seeds, accumulated in float64 in bounded blocks rather
    # than by materializing the whole normalized subset at once. Unlike
    # _weighted_row_sum this one is NOT bit-identical to the unblocked spelling: the
    # reduction runs across cells, so blocking changes the summation order that
    # ndarray.mean's pairwise accumulation would have used. Measured at 50000 seeds
    # over 3 blocks the template moves by 2e-16 absolute / 4e-13 relative, i.e. float64
    # rounding on a quantity whose downstream use is a direction; the tiny_simulated
    # example reproduces bit-for-bit because its seed subset fits one block.
    kept = where(keep)[0]
    acc = zeros(dev.shape[1], dtype="float64")
    block = max(1, int(8_000_000 // max(dev.shape[1], 1)))
    for start in range(0, kept.size, block):
        idx = kept[start:start + block]
        acc += (seeds[idx] / norms[idx][:, None]).sum(axis=0)
    return acc / float(kept.size)


def template_projection(dev, template, seg_weights=None):
    """Per-cell signed alignment with the consensus CN template.

    ``sum_s w_s t_s dev_is / sum_s |w_s t_s|`` — a gene-count-weighted projection.
    A cell with no CN scores ~0 and a cell deviating opposite to the consensus
    scores negative.

    Units: the denominator is the weighted L1 norm of the template, not its
    squared norm, so this is NOT a dimensionless projection coefficient. For
    ``dev == template`` it returns the weight-mean of ``|t|`` — a template whose
    events sit at ±0.5 gives 0.5, one at ±2.0 gives 2.0. The score therefore reads
    in **deviation-amplitude units**: "how far, in log-ratio, does this cell move
    along the consensus direction". That is the useful scale here — it stays
    comparable to the ``cn_burden`` magnitude and to the 0.2 threshold
    ``n_segments_altered`` uses — but it does mean a fixed numeric cutoff is not
    portable between samples whose consensus amplitudes differ. Divide by
    ``sum_s w_s t_s^2`` instead if a true unit-normalized coefficient is wanted.

    Why this rather than the magnitude burden it replaces: the burden is
    ``sum |dev|``, which counts every departure from the reference as evidence of
    malignancy regardless of direction. But a specialized normal cell's
    transcriptional mismatch is *directionally arbitrary* — it has no reason to
    align with this tumor's particular gains and losses — while a malignant cell's
    deviation does align, because it is the same clonal event. Projecting onto the
    consensus keeps the aligned component and cancels the arbitrary one, so the
    classes separate on direction as well as magnitude. It is also the component
    of the signal matched-bulk truth actually constrains.

    Any residual per-cell global offset projects onto ``sum_s w_s t_s``, which is
    near zero whenever the consensus holds both gains and losses — so this is
    additionally robust to whatever the per-cell centering upstream did not catch.

    Args:
        dev: (n_cells × n_segments) reference-relative deviation.
        template: (n_segments,) consensus deviation from consensus_template().
        seg_weights: (n_segments,) per-segment gene counts.

    Returns:
        1-D float array len == dev.shape[0]. Signed.
    """
    w = asarray(template, dtype="float64")
    if seg_weights is not None:
        w = w * asarray(seg_weights, dtype="float64")
    denom = float(np_abs(w).sum())
    if denom <= 0.0:
        # No template at all (a sample with no detectable CN structure): every
        # cell scores 0 and gmm_classify's constant-score guard calls them normal.
        return zeros(dev.shape[0], dtype="float64")
    return _weighted_row_sum(dev, w) / denom


def gmm_classify(
    scores,
    normal_mask=None,
    confidence_threshold=DEFAULT_CALL_CONFIDENCE,
    outlier_fence_mult=DEFAULT_OUTLIER_FENCE_MULT,
    coherence_of=None,
    coherence_gate=DEFAULT_COHERENCE_GATE,
):
    """Fit a 2-component GMM on the tumor-score vector and call each cell.

    Args:
        scores: 1-D ndarray of per-cell tumor scores. On the classify_cells path
            this is the signed template_projection(); any monotone per-cell
            discriminant works, and it may be negative.
        normal_mask: Optional bool array; when supplied, cells flagged as
            confident-normal upstream remain "normal" regardless of GMM
            output. Without this, signature/supervised confidence is lost
            because GMM may put a confident-normal cell into the tumor
            cluster when the score happens to be middling.
        confidence_threshold: Cells with max posterior below this become
            "uncertain" rather than a hard tumor/normal call.
        outlier_fence_mult: Multiplier on the reference pool's IQR to define the
            outlier exclusion fence. Non-baseline cells scoring above
            Q75 + outlier_fence_mult * IQR are candidates for exclusion from the
            GMM FIT (not from being called — the fence does not label anything).
            This prevents transcriptomically extreme cell types (Erythrocytes,
            Platelets) — whose high scores reflect transcriptome mismatch rather
            than CNV — from contaminating the mixture and displacing the true
            tumor component. It is effectively untuned: see the fence itself for
            why no shipped version of it has ever fired on this repo's benchmark
            cohort.
        coherence_of: Optional callable taking a row-index array and returning
            the per-cell coherent fraction (see _coherent_fraction) for those
            rows. REQUIRED for the outlier fence to do anything: scoring high is
            not on its own distinguishable from being a rare real clone, so
            without a way to ask whether a candidate's deviation is spatially
            contiguous the fence stays inert rather than guessing. classify_cells
            supplies ``lambda idx: _coherent_fraction(dev[idx], segments)``,
            which is evaluated only on the fence's candidate rows.
        coherence_gate: Coherent-fraction floor below which a fence candidate is
            treated as a transcriptome artifact rather than a clone. Only used
            together with ``coherence_of``, and 0 makes the fence inert rather
            than unconditional — the coherence is what licenses the exclusion, so
            without a meaningful floor there is nothing to license it.

    Returns:
        DataFrame with columns:
            class: str in {tumor, normal, uncertain}.
            confidence: float in [0, 1] (max GMM posterior).
            tumor_score: float — the input score, unmodified. Winsorization
                applies to the GMM's fitting input only, never to what is
                reported, so the returned value is always the caller's own score.
    """
    # Winsorize scores before GMM fitting to neutralise transcriptional outliers.
    # Rare cell types (e.g. Erythrocytes, Dendritic cells) can have extremely
    # high scores that are driven by their unusual transcriptional state rather
    # than by CNV. These create a spurious high-score component in the GMM,
    # displacing the TRUE tumor cluster (moderately elevated scores) into the
    # "normal" component.
    #
    # Winsorization bounds: the 1st and 99th percentiles of the WHOLE score
    # distribution. Computed once, unconditionally — both branches below used to
    # arrive at the same global-P99 ceiling by different routes.
    #
    # The ceiling used to be reference-relative (min(refQ90, refQ75 + 1.5*refIQR),
    # floored at clip_floor_mult * global median), i.e. "just above where the
    # reference cells sit". That has the wrong sign of dependence: the
    # winsorization exists to stop a handful of transcriptome-extreme cells from
    # dragging the GMM's components, so it should bound the extreme TAIL — but
    # pinning it to the reference spread instead bounds the tumor population
    # itself, and does so more tightly the better the score gets. A score that
    # separates the classes cleanly has, by construction, a tight reference
    # distribution, so refQ90 lands far below the tumor mode and the entire
    # malignant population collapses onto one value before the GMM ever sees it.
    # Measured on this repo's benchmark cohort: 46-91% of true tumor cells clipped
    # to the ceiling under a signed projection score (0.8-66% under the old
    # magnitude score) — i.e. the defect scaled up exactly as the upstream signal
    # improved.
    #
    # A global tail quantile has no such coupling: it always bounds ~1% of cells
    # whatever the score's dynamic range.
    #
    # BOTH tails are bounded, because the score is signed. template_projection
    # scores an anti-aligned cell at roughly minus its deviation magnitude, so
    # against a template of amplitude ~0.1 a cell with |dev| ~ 1.5 lands at -1.5
    # versus a tumor mode of +0.1 — a 15x tail, and erythrocytes/platelets
    # anti-align about half the time. Leaving it open lets one such cell claim the
    # GMM's low component outright, after which the ordinary normal and tumor modes
    # share the high component and non-reference normals are called tumor. A
    # quantile floor (rather than a hard floor at 0) bounds that tail without
    # collapsing it onto a single point mass.
    #
    # LIMIT of that argument, and why the fence below is symmetric. A 1% quantile
    # bounds ONE-IN-A-HUNDRED cells, so it answers the "one such cell" case and
    # nothing larger. When the anti-aligned population is itself more than ~1% of the
    # sample, the P1 floor lands INSIDE it, most of it survives unclipped, and the
    # low component locks onto it after all — the exact outcome this paragraph claims
    # to prevent. Measured on a 200-normal / 50-tumor / 20-anti-aligned synthetic
    # (7.4% anti-aligned): component means -1.144 and +0.102, with all 200 plain
    # normals AND all 50 tumor cells inside the high component. The mixture had
    # stopped separating tumor from normal entirely and was separating artifacts from
    # everything else. Winsorization cannot fix this — it is a fixed-fraction tool
    # being asked about a population of unknown size — so the negative tail is
    # excluded from the FIT by the fence instead, where the threshold scales with the
    # reference pool's spread rather than with a cell count.
    #
    # Note how invisible that failure is from the output: every plain normal was in
    # ``normal_mask`` and got overridden to "normal" below, so the emitted labels were
    # exactly right (220 normal / 50 tumor) while the mechanism underneath was
    # inverted. It only becomes visible when the mask is a strict subset of the
    # normals, which is the configuration every fixture here used to omit. See
    # test_heldout_pool_does_not_call_unlabelled_normals_tumor.
    clip_ceiling = float(percentile(scores, 99))
    clip_floor = float(percentile(scores, 1))
    nm = asarray(normal_mask, dtype=bool) if normal_mask is not None else None

    # Outlier detection: non-baseline cells scoring far above the bulk of the
    # distribution are likely transcriptome-mismatch cells (Erythrocytes,
    # Platelets), not CNV-driven. Hold them out of the GMM fit so they cannot drag
    # a component onto themselves. They are still scored and called like every other
    # cell afterwards — see the note at the labelling step for why the fence no
    # longer forces them to "normal".
    #
    # The fence is a Tukey fence on the REFERENCE POOL's spread — "far above where
    # cells known to be diploid sit" — which is the only spread here that estimates
    # the normal mode's width without depending on how much tumor the sample holds.
    # A global-IQR fence was tried instead and is worse, not better: on a low-purity
    # sample the global IQR is dominated by normals and collapses, so the fence cuts
    # straight through the tumor mode (measured: recall 1.00 -> 0.50 at 2.9% tumor
    # fraction). The reference pool is the right population to measure the normal
    # mode's width.
    #
    # But "far above the normal mode" is also true of every genuine tumor cell, so
    # the fence needs a SECOND condition saying which of the high-scoring cells are
    # artifacts. Three attempts read that condition off the score distribution and
    # all three failed:
    #
    #   * versus ``clip_ceiling``: once the ceiling became a global quantile the two
    #     were no longer in the same units — the fence is in reference-pool IQR,
    #     which shrinks as the score's noise is removed — so the comparison silently
    #     stopped passing. Measured on a clean clonal synthetic, fence 0.146 vs
    #     ceiling 0.493: the exclusion never ran at all, which is the real reason
    #     commit 72d4a8c's 7.0 -> 12.0 widening looked "saturated".
    #   * versus ``median(scores[~nm])``: ``normal_mask`` is a reference SUBSET,
    #     never the full normal population (pick_baseline's signature / variance /
    #     gmm_fallback tiers each return a subset, and a supervised run gets whatever
    #     barcodes the user labelled), so ``scores[~nm]`` is dominated by UNLABELLED
    #     NORMALS and that median is the normal mode. The guard passed trivially and
    #     every malignant cell was locked to "normal": recall 1.00 with all 200
    #     normals supplied as the pool and 0.00 at 100, 50 or 30 of them — fence
    #     0.159 against a tumor mode of 0.398, non-reference median 0.008. See
    #     test_outlier_fence_holds_when_reference_pool_is_a_subset.
    #   * versus a provisional GMM's fitted high component: that fit ran on the
    #     WINSORIZED scores while the fence and the exclusion ran on the raw ones.
    #     Below ~1% tumor fraction the malignant population sits above the global P99
    #     clip ceiling, so the clipped vector holds no tumor mode at all, the fitted
    #     mode collapsed onto the normal mode, the guard passed — and the fence then
    #     selected exactly the tumor cells off the raw scores. Measured at 8 tumor /
    #     2000 normal: ceiling 0.0222, tumor median 0.1763, fitted mode 0.006, fence
    #     0.1368, recall 0.00. Fitting on the raw scores instead only relocates the
    #     failure, because with no real tumor present the artifacts ARE the high
    #     component, so the guard could never fire in the one case the fence exists
    #     for.
    #
    # The last two are one lesson. A small population sitting far above the normal
    # mode is the SAME 1-D distribution whether it is a rare real clone or a cluster
    # of transcriptome-extreme cells, so NO statistic of ``scores`` alone can
    # separate them — which means no activation guard phrased in score units can be
    # right, and the two failures above were not tuning mistakes.
    #
    # What does separate them is spatial. A clone's deviation is a contiguous run
    # across neighbouring segments; an artifact's is scattered spikes. That is
    # exactly ``_coherent_fraction``, already applied to the same deviation matrix by
    # the tumor-side gate in classify_cells, so the fence is not introducing a second
    # notion of "real CN" alongside the one the rest of the classifier uses.
    #
    # So the second condition is per-cell coherence, supplied as ``coherence_of`` and
    # evaluated ONLY on the cells the fence would actually take. The candidate set is
    # a handful of rows, so this costs one smoothing pass over those rows instead of
    # the whole matrix, and it removes the extra full GMM fit the previous guard
    # needed. Without ``coherence_of`` the two cases are indistinguishable here, so
    # the fence stays inert rather than guessing; a direct gmm_classify caller that
    # wants it must pass the callable.
    #
    # The fence is SYMMETRIC, and the second condition applies to the high side only.
    # That asymmetry is the whole point, so it is worth stating why:
    #
    #   * HIGH side — a rare real clone lives here, so excluding a cell needs positive
    #     evidence that it is not one. Hence the coherence condition, and hence the
    #     fence going inert when coherence is unavailable.
    #   * LOW side — under a signed score a clone projects POSITIVELY by construction,
    #     so nothing the GMM needs to resolve lives down here. Measured across this
    #     repo's 10 annotated benchmark patients: 0.4% of the most anti-aligned 2% of
    #     cells are ground-truth tumor, and their cn_burden is at or below the cohort
    #     median on 10 of 10. So no second condition is needed, and none is applied.
    #
    # The one thing that does land in the negative tail is a subclone whose profile
    # opposes the direction consensus_template locked onto. Holding it out of the FIT
    # is right even so: it is not recoverable under a single template either way (the
    # documented limitation), it is what was breaking the fit for everyone else, and
    # the anti-alignment gate in classify_cells still surfaces it as "uncertain" —
    # exclusion from the fit costs it nothing, because the fence does not label.
    #
    # Because the low side needs no coherence, it works for a bare gmm_classify caller
    # too, where the high side cannot.
    #
    # NOTE for re-validation: because no shipped guard ever passed correctly, the HIGH
    # side of this fence has effectively never run on the benchmark cohort at any
    # multiplier. ``outlier_fence_mult`` is therefore an untuned parameter, not a
    # settled one, and ``n_outlier_fenced`` is reported in qc.json so a run that does
    # fence cells says so in its own output. Note also that the fence runs inside
    # gmm_classify, i.e. AFTER consensus_template has already been estimated, so it
    # structurally cannot protect the template from the cells it excludes. Seed-set
    # robustness has to live in consensus_template itself, which is where the
    # unit-norm rescaling and the low-complexity exclusion now are.
    scores_for_fit = clip(scores, clip_floor, clip_ceiling)
    outlier_mask = zeros(len(scores), dtype=bool)
    if nm is not None and nm.sum() >= 10 and scores_for_fit.std() >= 1e-10:
        baseline_s = scores[nm]
        bq75 = float(percentile(baseline_s, 75))
        bq25 = float(percentile(baseline_s, 25))
        bq_iqr = bq75 - bq25

        # Low side: no second condition, for the reasons above, and no ~nm
        # restriction either. Holding a cell out of the FIT is not overriding its
        # label — the normal_mask override below still runs, and the fence no longer
        # labels anything — so the invariant "a supplied normal_mask is never
        # overridden" is untouched. And a reference cell sitting 12 IQRs BELOW its own
        # pool's Q25 is precisely the cell that must not anchor the low component: it
        # is anomalous by the pool's own yardstick. Restricting this to ~nm left the
        # broken-mixture case unfixed whenever the anti-aligned population happened to
        # be labelled — i.e. exactly when the label override would hide it.
        outlier_mask |= scores < bq25 - outlier_fence_mult * bq_iqr

        # High side: only with a coherent-fraction floor to license it. None and 0
        # both mean "no floor", hence no high-side fence.
        if coherence_of is not None and coherence_gate and coherence_gate > 0:
            cand_idx = where((~nm) & (scores > bq75 + outlier_fence_mult * bq_iqr))[0]
            if cand_idx.size:
                scattered = asarray(coherence_of(cand_idx), dtype=float) < coherence_gate
                outlier_mask[cand_idx[scattered]] = True

    # Exclude outlier cells from GMM fitting; they do not represent the
    # tumor/normal distribution we want to learn.
    fit_mask = ~outlier_mask
    fit_scores = scores_for_fit[fit_mask]

    # GMM fit on the winsorized 1-D score vector. random_state pinned for
    # reproducibility.
    gmm = GaussianMixture(n_components=2, random_state=0)
    # Guard: if all scores are (nearly) identical, GMM will degenerate; skip fitting
    # and label everything 'normal' with high confidence.
    if fit_scores.std() < 1e-10:
        import pandas as pd
        degenerate = pd.DataFrame({
            "class": ["normal"] * len(scores),
            "confidence": [1.0] * len(scores),
            "tumor_score": scores,
        })
        degenerate.attrs["n_outlier_fenced"] = int(outlier_mask.sum())
        return degenerate
    gmm.fit(fit_scores.reshape(-1, 1))

    # Predict for ALL cells using the clipped (but not outlier-excluded) scores.
    # Fenced cells are scored by the mixture like everyone else — the fence decides
    # what the mixture is FITTED on, not what any cell is called.
    posteriors = gmm.predict_proba(scores_for_fit.reshape(-1, 1))
    assignments = posteriors.argmax(axis=1)

    # Higher-mean component is the tumor cluster — tumor cells deviate more
    # from the normal baseline by construction.
    high_component = int(gmm.means_.ravel().argmax())
    is_tumor_component = assignments == high_component
    confidence = posteriors.max(axis=1)

    # Default labels from GMM. Cells below the posterior threshold land in
    # 'uncertain' regardless of which cluster won.
    labels = where(is_tumor_component, "tumor", "normal").astype(object)
    labels[confidence < confidence_threshold] = "uncertain"

    # Fenced cells are NOT forced to "normal". They used to be, and that made the
    # fence assert the one thing this classifier is careful never to assert about a
    # cell carrying a large unexplained deviation. Now that the fence's second
    # condition is the coherent fraction, it rules on exactly the evidence the
    # coherence gate in classify_cells rules on, and the two reached opposite
    # verdicts on the same cell — "normal" here versus "uncertain" there — with
    # precedence decided by nothing but which ran first. In a cleanly separating
    # sample that is not an edge case: measured on a 50-normal / 49-tumor synthetic,
    # all 49 malignant cells clear the fence and are held only by their coherence, so
    # the fence reaches every tumor-range cell in the sample.
    #
    # The module already settled this question in the mirror case. The anti-alignment
    # gate exists precisely because "confidently normal" is the wrong thing to assert
    # about a cell whose deviation is large but points the wrong way; a large
    # SCATTERED deviation is no more explained than a large opposing one. So the fence
    # keeps the job the whole comment block above is about — deciding what the mixture
    # is fitted on, which is where it protects the tumor component from being
    # displaced — and leaves labelling to the GMM and the gates. A fenced artifact
    # that still lands in the tumor component is then downgraded to "uncertain" by
    # the coherence gate on the same coherent fraction that fenced it, which is the
    # honest label for it. ``n_outlier_fenced`` reports the fit exclusion.

    # Honor the upstream normal_mask if supplied: cells flagged confident-normal
    # are NEVER reclassified to tumor. They can downgrade to uncertain if their
    # GMM posterior on the normal cluster is low.
    if normal_mask is not None:
        # For confident-normals: keep label 'normal' if confidence high, else
        # 'uncertain'. Never 'tumor' — that would override the upstream signal.
        upstream_normal_idx = asarray(normal_mask, dtype=bool)
        labels = labels.copy()
        for i in where(upstream_normal_idx)[0]:
            if confidence[i] >= confidence_threshold:
                labels[i] = "normal"
            else:
                labels[i] = "uncertain"

    out = DataFrame({
        "class": labels.astype(str),
        "confidence": confidence,
        "tumor_score": scores,
    })
    # Carried in .attrs rather than as a column: it is a per-run diagnostic, not a
    # per-cell value, and prediction.csv's schema is a public contract. classify_cells
    # propagates it and the CLI records it in qc.json.
    out.attrs["n_outlier_fenced"] = int(outlier_mask.sum())
    return out


def discover_subclones(
    cn_matrix,
    tumor_mask,
    max_subclones=DEFAULT_MAX_SUBCLONES,
    resolution_sweep=SUBCLONE_RESOLUTION_SWEEP,
    min_tumor_cells=MIN_TUMOR_CELLS_FOR_SUBCLONES,
):
    """Discover subclones via Leiden on the per-cell × per-segment CN matrix.

    Restricted to tumor cells — normal cells have ~0 CN signal and would
    collapse the clustering. Below min_tumor_cells we skip discovery and
    label every tumor cell "subclone_1" so the output schema stays uniform.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray.
        tumor_mask: bool array len == n_cells, True for tumor cells.
        max_subclones: Cap on the returned subclone count.
        resolution_sweep: Increasing-resolution tuple to walk until the
            partition fits under max_subclones.
        min_tumor_cells: Below this many tumor cells, return a single clone.

    Returns:
        1-D string array len == n_cells. Non-tumor cells get "".
        Tumor cells get "subclone_1", "subclone_2", etc.
    """
    # Pre-allocate the output with empty strings for non-tumor cells.
    n_cells = cn_matrix.shape[0]
    subclones = full(n_cells, "", dtype=object)

    n_tumor = int(tumor_mask.sum())
    if n_tumor < min_tumor_cells:
        # Too few tumor cells for stable clustering — collapse to one clone.
        subclones[tumor_mask] = "subclone_1"
        return subclones.astype(str)

    # Build a tiny AnnData around the tumor-cell CN matrix so we can reuse
    # scanpy's PCA + neighbors + Leiden pipeline. Each "gene" is one segment.
    tumor_cn = cn_matrix[tumor_mask, :]
    sub_adata = AnnData(X=tumor_cn.astype("float32"))

    # Bounded n_comps: segments often outnumber tumor cells in tiny cohorts,
    # so PCA dimension must not exceed min(n_obs, n_vars) - 1.
    n_comps = min(30, sub_adata.n_vars - 1, sub_adata.n_obs - 1)
    sc_pca(sub_adata, n_comps=n_comps)
    sc_neighbors(sub_adata, n_neighbors=min(15, sub_adata.n_obs - 1))

    # Walk the resolution sweep, taking the first partition that fits under
    # max_subclones. If none do, take the largest-resolution result and
    # collapse over-cap clusters into the largest k clusters.
    chosen_labels = None
    for res in resolution_sweep:
        sc_leiden(
            sub_adata,
            resolution=res,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            key_added=f"leiden_r{res:.2f}",
        )
        labels = sub_adata.obs[f"leiden_r{res:.2f}"].astype(str).to_numpy()
        n_clusters = len(set(labels))
        if n_clusters <= max_subclones:
            chosen_labels = labels
            break

    if chosen_labels is None:
        # Fall back: take the last partition we computed, keep the top-K largest
        # clusters, merge the rest into the largest one. Ensures we never exceed
        # max_subclones in the emitted output.
        labels = sub_adata.obs[f"leiden_r{resolution_sweep[-1]:.2f}"].astype(str).to_numpy()
        sizes_fallback = Series(labels).value_counts()
        top_labels = set(sizes_fallback.head(max_subclones).index)
        largest = sizes_fallback.idxmax()
        chosen_labels = asarray([lbl if lbl in top_labels else largest for lbl in labels])

    # Re-label to dense "subclone_1..subclone_k" with the largest cluster first
    # for stable ordering across runs. value_counts() on a Series returns a
    # plain Series whose .index is the unique labels in count-descending order.
    sizes = Series(chosen_labels).value_counts()
    relabel = {lbl: f"subclone_{i + 1}" for i, lbl in enumerate(sizes.index)}
    relabeled = asarray([relabel[lbl] for lbl in chosen_labels])

    subclones[tumor_mask] = relabeled
    return subclones.astype(str)


def classify_cells(
    cn_matrix,
    segments,
    normal_mask,
    barcodes,
    confidence_threshold=DEFAULT_CALL_CONFIDENCE,
    max_subclones=DEFAULT_MAX_SUBCLONES,
    discover_subclones_enabled=True,
    complexity=None,
    low_complexity_frac=DEFAULT_LOW_COMPLEXITY_FRAC,
    coherence_gate=DEFAULT_COHERENCE_GATE,
):
    """Orchestrator: tumor scores → GMM call → gates → subclones → DataFrame.

    After the GMM call, three *subtractive* gates downgrade untrustworthy calls to
    "uncertain". None of them ever promotes a cell to tumor, so none can inflate
    the tumor set or hurt bulk concordance:

      * low-complexity gate — ambient / empty-droplet-like cells (few detected
        genes) have noise-dominated CN signal; a tumor call on them is flagged
        rather than trusted. Requires ``complexity``; skipped if not supplied.
      * coherence gate — a "tumor" call whose deviation is mostly scattered
        single-segment spikes rather than chromosome-arm-scale (a "loud"
        specialized normal cell) is downgraded. Always applied.
      * anti-alignment gate — a "normal" call that carries a large, coherent
        deviation pointing *against* the consensus template is downgraded. This is
        the only gate acting on normal calls, and it exists because the signed
        score cannot by itself distinguish a confident diploid (≈ 0) from a cell
        whose real CN opposes the direction the template locked onto (strongly
        negative). See the gate itself for the four conditions and the cohort
        measurements behind them. Its thresholds are relative to the malignant
        population's own level, so it declines to run at all when fewer than
        MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT cells are called tumor.

    This encodes the "flag, don't force" philosophy: cells that are genuinely
    ambiguous (including same-lineage normals whose expression mimics CNV, and
    cells whose copy number is real but points the other way) land in "uncertain"
    rather than being asserted as either class.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray.
        segments: DataFrame from detect_segments(); ``chr`` groups segments into
            chromosomes (coherence gate) and it feeds the n_segments_altered count.
        normal_mask: bool array len == n_cells from pick_baseline.
        barcodes: array-like of cell barcodes, len == n_cells.
        confidence_threshold: Posterior cutoff for the "uncertain" label.
        max_subclones: Cap on emitted subclones.
        discover_subclones_enabled: If False, skip discovery entirely; every
            tumor cell gets subclone == "".
        complexity: Optional 1-D array len == n_cells of per-cell detected-gene
            counts. When supplied, cells below ``low_complexity_frac`` × the
            cohort median are flagged low_complexity and any tumor call on them
            is downgraded to uncertain. When None, no low-complexity gating and
            the low_complexity column is all-False.
        low_complexity_frac: Fraction-of-median detected-gene threshold.
        coherence_gate: Minimum coherent fraction (see _coherent_fraction) for a
            tumor call to stand; below it the call becomes uncertain. Set to 0 to
            disable that downgrade. Two other places consult the coherent fraction
            and 0 cannot make either of them fire on a cell it otherwise spares:
            the outlier fence goes inert, and the anti-alignment gate's third
            condition holds at DEFAULT_COHERENCE_GATE. See where the three are
            reconciled in the body.

    Returns:
        A DataFrame indexed by barcode with columns:
            class: str in {tumor, normal, uncertain}.
            confidence: float in [0, 1].
            tumor_score: float — the SIGNED consensus-template projection that
                drove the call. Positive = deviating with this sample's consensus
                CN profile, ~0 = no CN, negative = deviating against it. This is
                the discriminant, so it is what a ROC/AUC over the call should be
                computed on.
            cn_burden: float >= 0 — gene-count-weighted mean |deviation| across
                the genome. The magnitude companion to tumor_score, and the column
                to rank on when the question is "how close to diploid is this
                cell" rather than "how tumor-like". Kept separate rather than
                folding one into the other because the two orderings genuinely
                disagree: under the signed score the MINIMUM is the most
                anti-aligned cell, not the most diploid one, so a consumer picking
                a diploid baseline by lowest tumor_score selects cells carrying a
                large real event in the opposite direction.
            subclone: str ("subclone_1"... or "" for non-tumor).
            n_segments_altered: int — count of segments where the cell's CN
                deviates from the normal-pool median by more than 0.2 (~roughly
                a half-fold-change in log space; conservative heuristic).
            low_complexity: bool — flagged ambient/empty-droplet-like cell.
    """
    # ── tumor scores + GMM ────────────────────────────────────────────────
    # Compute the per-segment normal-pool baseline ONCE and share it across the
    # score, the coherence gate, and the n_segments_altered count below.
    baseline = _normal_pool_baseline(cn_matrix, normal_mask)
    # Segment gene counts weight the per-cell genome-wide center (see
    # reference_relative_deviation). Computed once and shared by the score, the
    # coherence gate and n_segments_altered so all three reason about the SAME
    # deviation — two different notions of "deviation from diploid" inside one
    # classifier is how a gate ends up gating something the score never saw.
    seg_weights = segments["n_genes"].to_numpy() if "n_genes" in segments else None
    dev = reference_relative_deviation(
        cn_matrix, normal_mask, baseline=baseline, seg_weights=seg_weights,
    )
    # Ambient / empty-droplet-like cells, computed BEFORE the template so they can
    # be barred from seeding it. Their CN signal is noise-dominated and large in
    # magnitude, so burden-based seed selection concentrates them in exactly the set
    # that sets the template's direction; the gate further down only downgrades
    # their own calls, which is too late to protect everyone else's score.
    low_complexity = _low_complexity_mask(complexity, low_complexity_frac, cn_matrix.shape[0])

    # Signed alignment with the sample's own consensus CN profile, rather than
    # the magnitude of any departure from the reference (see
    # template_projection for why direction is the discriminating part).
    # abs(dev) is needed three times below (the template's seed burden, cn_burden,
    # and the altered-segment count). Materialize it once — it is the same size as
    # the largest array in the run.
    abs_dev = np_abs(dev)
    template = consensus_template(
        dev, seg_weights, exclude=low_complexity, abs_dev=abs_dev,
    )
    scores = template_projection(dev, template, seg_weights)
    # Non-negative companion to the signed score: the gene-weighted L1 burden, in
    # the same per-cell-centered frame. This is what "how much CN does this cell
    # carry" means when direction is not the question — see the cn_burden note in
    # the Returns block above for why the two cannot be the same column.
    cn_burden = segment_burden(abs_dev, seg_weights)

    # Three places consult the coherent fraction, and ``coherence_gate=0`` has to
    # mean something different in each, because the role the coherence plays is
    # different. Wiring all three to the raw knob is what made ``--coherence-gate 0``
    # LOOSEN the anti-alignment gate instead of leaving it alone (measured on a
    # 60/50/15 synthetic: n_anti_aligned 0 at the default, 15 at 0 — it downgraded
    # exactly the scattered anti-aligned artifacts the default deliberately spares).
    #
    #   * tumor-side gate below — coherence IS the decision. 0 turns it off; that is
    #     the documented meaning of the knob and the only place it applies directly.
    #   * outlier fence in gmm_classify — coherence ENABLES an action (locking a
    #     high-scoring cell to "normal"). Threshold 0 there would fence every
    #     high-scoring non-reference cell, so 0 means the fence goes inert: it may
    #     only act on evidence the caller has said is meaningful. Strictly the safer
    #     direction, since an inert fence cannot mislabel a clone.
    #   * anti-alignment gate below — coherence RESTRICTS an action. Dropping the
    #     condition makes that gate fire more, so it holds at DEFAULT_COHERENCE_GATE
    #     regardless: "do not downgrade my focal tumor calls for being scattered" is
    #     not a claim that scattered anti-aligned artifacts are arm-scale.
    support_coherence = (
        float(coherence_gate) if coherence_gate and coherence_gate > 0
        else DEFAULT_COHERENCE_GATE
    )
    call_df = gmm_classify(
        scores,
        normal_mask=normal_mask,
        confidence_threshold=confidence_threshold,
        # Evaluated lazily on the fence's candidate rows only — see the fence.
        coherence_of=lambda idx: _coherent_fraction(dev[idx], segments),
        coherence_gate=coherence_gate,
    )
    labels = call_df["class"].to_numpy().astype(object)
    confidence = call_df["confidence"].to_numpy().astype(float).copy()
    gated = full(cn_matrix.shape[0], False)

    # ── gates: downgrade untrustworthy tumor calls to "uncertain" ─────────
    # low-complexity (ambient) gate — only when complexity is supplied. The mask
    # itself was computed above, before the template.
    lc_gate = (labels == "tumor") & low_complexity
    labels[lc_gate] = "uncertain"
    gated |= lc_gate

    # coherence gate — scattered-signal tumor calls become uncertain. Only the
    # still-tumor cells can be affected, so score just those rows (the coherent
    # fraction is per-row and independent of the others) and skip entirely when
    # none remain.
    if coherence_gate and coherence_gate > 0:
        tumor_idx = where(labels == "tumor")[0]
        if tumor_idx.size:
            coh_frac = _coherent_fraction(dev[tumor_idx], segments)
            coh_gate = tumor_idx[coh_frac < coherence_gate]
            labels[coh_gate] = "uncertain"
            gated[coh_gate] = True

    # anti-alignment gate — a "normal" call carrying real CN in the direction
    # OPPOSITE the consensus becomes "uncertain" rather than a confident normal.
    #
    # The score is signed, so the bottom of its range is not "most diploid" but
    # "most anti-aligned". Almost everything down there is a transcriptional
    # artifact: measured across this repo's 10 annotated benchmark patients, 0.4%
    # of the most anti-aligned 2% of cells are ground-truth tumor, and their
    # cn_burden sits at or below the cohort median on 10/10 — they carry LESS copy
    # number than an average cell, not more. The one thing that also lands there is
    # a genuine subclone whose profile opposes the direction consensus_template
    # locked onto (see its Known limitation), and such a cell is indistinguishable
    # from a confident diploid in every column prediction.csv emits. This gate
    # makes that case visible rather than silent.
    #
    # All four conditions must hold, so an ordinary anti-aligned artifact does not
    # qualify:
    #   * it projects AGAINST the consensus by at least ANTI_ALIGNED_FRACTION of how
    #     far the malignant population projects along it. Stated relative to the
    #     tumor mode rather than as an absolute cutoff because the score is in
    #     deviation-amplitude units, so no fixed number ports between samples;
    #   * it carries at least that same fraction of the malignant population's median
    #     burden — a large real deviation, not a flat cell at the noise floor;
    #   * its deviation is as chromosome-arm-scale as a tumor call is required to be;
    #   * it is not ambient/low-complexity, and was not asserted normal upstream
    #     (a supplied normal_mask is never overridden, here as everywhere else).
    #
    # Like the other two gates this only ever moves a cell TO "uncertain". We do not
    # know an anti-aligned cell is malignant — only that "confidently normal" is the
    # wrong thing to assert about it. ``n_anti_aligned`` reaches qc.json so the rate
    # is visible per run; a large value is the signal that this sample needs
    # multi-template scoring, which is the fix the single-template limitation defers.
    #
    # Both thresholds are medians over the cells called tumor, so the gate needs
    # enough of them for "the malignant population's level" to mean anything — see
    # MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT for the measured spread below that floor.
    # ``anti_alignment_tumor_n`` records what the estimate rested on, 0 when the gate
    # did not run, so a skipped gate is visible in the run's own output.
    n_anti_aligned = 0
    tumor_now = where(labels == "tumor")[0]
    anti_alignment_tumor_n = (
        int(tumor_now.size) if tumor_now.size >= MIN_TUMOR_CELLS_FOR_ANTI_ALIGNMENT else 0
    )
    if anti_alignment_tumor_n:
        tumor_ref = ANTI_ALIGNED_FRACTION * float(median(scores[tumor_now]))
        burden_ref = ANTI_ALIGNED_FRACTION * float(median(cn_burden[tumor_now]))
        # tumor_ref <= 0 means the "tumor" component does not project positively at
        # all — there is no consensus direction to be anti-aligned with, so there is
        # nothing this gate can meaningfully say.
        if tumor_ref > 0:
            candidate = (
                (labels == "normal")
                & (~low_complexity)
                & (scores <= -tumor_ref)
                & (cn_burden >= burden_ref)
            )
            if normal_mask is not None:
                candidate &= ~asarray(normal_mask, dtype=bool)
            cand_idx = where(candidate)[0]
            if cand_idx.size:
                # ``support_coherence``, not ``coherence_gate`` — this condition is
                # always evaluated. See where it is defined for why disabling the
                # tumor-side gate must not make this gate fire more often.
                coh_frac = _coherent_fraction(dev[cand_idx], segments)
                cand_idx = cand_idx[coh_frac >= support_coherence]
            if cand_idx.size:
                labels[cand_idx] = "uncertain"
                gated[cand_idx] = True
                n_anti_aligned = int(cand_idx.size)

    # A gate overrides a *confident* GMM call, so the reported confidence must
    # reflect the downgrade — otherwise a gate-flagged cell keeps its high posterior
    # and would pass a downstream `confidence >= threshold` filter despite being
    # uncertain. Report the posterior of the interpretation no longer being asserted
    # (1 - the winning posterior), which is < 0.5 for any former confident call,
    # preserving the invariant "uncertain ⇒ low confidence". This holds for both
    # directions of downgrade: tumor→uncertain and normal→uncertain.
    confidence[gated] = 1.0 - confidence[gated]

    # ── subclone discovery ───────────────────────────────────────────────
    tumor_mask = labels == "tumor"
    if discover_subclones_enabled:
        # Cluster the reference-relative deviation, not the raw matrix. The raw
        # per-segment CN still carries the per-cell depth pedestal — 50-90% of its
        # L1 on this repo's benchmark cohort, correlating with detected-gene count
        # at up to r=+0.88 within cells that have no CN at all — so PC1 of the raw
        # matrix is substantially library depth and the emitted subclone_N labels
        # would be depth strata rather than clones.
        subclones = discover_subclones(
            dev,
            tumor_mask=tumor_mask,
            max_subclones=max_subclones,
        )
    else:
        # Disabled: every tumor cell carries an empty subclone label.
        subclones = full(cn_matrix.shape[0], "", dtype=object).astype(str)

    # ── n_segments_altered per cell ───────────────────────────────────────
    # Count segments where the cell's reference-relative deviation exceeds 0.2 in
    # log space. Useful for downstream filtering and for spotting "barely-tumor"
    # calls with one or two focal events. Reuses the deviation computed above, so
    # the count is of genuinely regional departures rather than of the per-cell
    # global offset (which, being present in every segment, previously made this
    # a proxy for library depth on deep cells).
    n_segments_altered = (abs_dev > 0.2).sum(axis=1).astype(int)

    # ── assemble per-cell DataFrame ───────────────────────────────────────
    out = DataFrame(
        {
            "class": labels.astype(str),
            "confidence": confidence,
            "tumor_score": call_df["tumor_score"].to_numpy(),
            "cn_burden": cn_burden,
            "subclone": subclones,
            "n_segments_altered": n_segments_altered,
            "low_complexity": low_complexity,
        },
        index=asarray(barcodes),
    )
    out.index.name = "barcode"
    out.attrs["n_outlier_fenced"] = int(call_df.attrs.get("n_outlier_fenced", 0))
    out.attrs["n_anti_aligned"] = n_anti_aligned
    out.attrs["anti_alignment_tumor_n"] = anti_alignment_tumor_n
    return out
