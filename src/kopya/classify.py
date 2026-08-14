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
    full,
    median,
    minimum as np_minimum,
    ones,
    percentile,
    sign as np_sign,
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
# Non-baseline cells scoring above Q75 + OUTLIER_FENCE_MULT * IQR of the
# baseline score distribution are excluded from GMM fitting and locked to
# "normal". This prevents transcriptomically extreme cell types (Erythrocytes,
# Platelets) whose high scores are driven by transcriptome-mismatch rather
# than CNV from being incorrectly called as tumor. A tighter fence also
# forces genuine high-CN-burden tumor cells (multiple/large events, hence a
# very high tumor_score) into "normal" the same way -- unlike the coherence/
# low-complexity gates (which only ever downgrade "tumor" to "uncertain"),
# this is the one mechanism in the classifier that can force a real tumor
# cell all the way to "normal" and is therefore a plausible direct recall
# cost. Widened from 7.0 to 12.0 to exclude only the most extreme outliers,
# leaving more strong-signal tumor cells inside the GMM fit where their
# score can earn a tumor call on its own merits.
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
    half = 0.5 * float(w.sum())
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
    dev = cn_matrix - baseline
    if seg_weights is not None:
        dev = dev - weighted_median_rows(dev, seg_weights)[:, None]
    return dev


def compute_tumor_scores(cn_matrix, normal_mask, baseline=None, seg_weights=None):
    """Per-cell L1 distance from the normal-pool median CN vector.

    Computed on the reference-relative deviation (see
    ``reference_relative_deviation``), i.e. after both the per-segment normal
    baseline and the cell's own genome-wide median have been removed.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray from per_cell_segment_cn().
        normal_mask: bool array len == n_cells, True for cells in normal pool.
        baseline: optional precomputed per-segment baseline (from
            _normal_pool_baseline); computed from normal_mask when omitted. Lets
            classify_cells compute the baseline once and share it across the
            score, coherence gate, and n_segments_altered.
        seg_weights: (n_segments,) per-segment gene counts; forwarded to
            reference_relative_deviation for the per-cell centering step.

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
    return (deviations * w).sum(axis=1) / total


def consensus_template(
    dev, seg_weights=None, seed_quantile=DEFAULT_TEMPLATE_SEED_QUANTILE, exclude=None,
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

    Known limitation: this estimates ONE consensus direction. Two high-burden
    subclones carrying opposing profiles cancel here — at equal population size
    the template collapses toward zero, and an unequal minority clone projects
    negative and is called normal. See
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
    burden = segment_burden(np_abs(dev), seg_weights)
    thr = float(percentile(burden[eligible], 100.0 * seed_quantile))
    seed = eligible & (burden >= thr)
    # Degenerate guard: an exactly-constant burden makes the threshold equal the
    # maximum and could select nothing. Fall back to the whole cohort, which gives
    # a ~zero template and hence ~zero scores — the honest answer for a sample
    # with no detectable CN structure.
    if not seed.any():
        seed = eligible
    seeds = asarray(dev[seed], dtype="float64")
    # Rescale each seed to unit weighted-L1 norm so it votes on direction only.
    # Same aggregation as the burden that selected it, so "one unit of deviation"
    # means the same thing in both places.
    norms = segment_burden(np_abs(seeds), seg_weights)
    keep = norms > 0
    if not keep.any():
        # Every seed is exactly diploid — no direction to estimate.
        return zeros(dev.shape[1], dtype="float64")
    return (seeds[keep] / norms[keep][:, None]).mean(axis=0)


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
    return (dev * w).sum(axis=1) / denom


def gmm_classify(
    scores,
    normal_mask=None,
    confidence_threshold=DEFAULT_CALL_CONFIDENCE,
    outlier_fence_mult=DEFAULT_OUTLIER_FENCE_MULT,
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
            Q75 + outlier_fence_mult * IQR are excluded from GMM fitting
            and forced to "normal". This prevents transcriptomically extreme
            cell types (Erythrocytes, Platelets) — whose high scores reflect
            transcriptome mismatch rather than CNV — from contaminating the
            GMM and displacing the true tumor signal. See the activation guard
            below: the fence only applies when it sits above the bulk of the
            non-reference population, and it is effectively untuned because the
            guard it shipped with never passed.

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
    clip_ceiling = float(percentile(scores, 99))
    clip_floor = float(percentile(scores, 1))
    nm = asarray(normal_mask, dtype=bool) if normal_mask is not None else None

    # Outlier detection: non-baseline cells scoring far above the bulk of the
    # distribution are likely transcriptome-mismatch cells (Erythrocytes,
    # Platelets), not CNV-driven. Exclude them from GMM fitting and lock them to
    # "normal" in the output.
    #
    # The fence is a Tukey fence on the REFERENCE POOL's spread — "far above where
    # cells known to be diploid sit" — which is the only spread here that estimates
    # the normal mode's width without depending on how much tumor the sample holds.
    #
    # It needs an activation guard, because "far above the normal mode" is also
    # true of every genuine tumor cell; without one, a cleanly-separating score
    # locks the whole malignant population to "normal" (100/100 on a synthetic with
    # a clean clonal gain+loss). The guard that shipped compared the fence to
    # ``clip_ceiling``, and once that ceiling moved to a global quantile the two
    # were no longer in the same units: the fence is measured in reference-pool IQR,
    # which shrinks as the score's noise is removed, so the comparison silently
    # stopped passing. Measured on that same synthetic, fence 0.146 vs ceiling 0.493
    # at the shipped multiplier — the exclusion never ran at all. That made this
    # protection unreachable on any supervised run whose score separates, and made
    # commit 72d4a8c's 7.0 -> 12.0 widening a no-op, which is the real reason it
    # "saturated" rather than any property of the cohort.
    #
    # The guard is now stated in score units against the quantity it is actually
    # trying to protect: fence the tail only when the fence sits ABOVE the bulk of
    # the non-reference population, so it cannot cut into the tumor mode. Both sides
    # are per-cell scores, so the comparison means something at any dynamic range.
    #
    # A global-IQR fence was tried instead and is worse, not better: on a low-purity
    # sample the global IQR is dominated by normals and collapses, so the fence cuts
    # straight through the tumor mode (measured: recall 1.00 -> 0.50 at 2.9% tumor
    # fraction). The reference pool is the right population to measure; the guard is
    # the part that was broken.
    #
    # NOTE for re-validation: because the old guard never passed, this fence has
    # effectively never run on the benchmark cohort, at any multiplier. It stays
    # inert on both synthetics above under the new guard. ``outlier_fence_mult`` is
    # therefore an untuned parameter, not a settled one — and note the fence runs
    # inside gmm_classify, i.e. AFTER consensus_template has already been estimated,
    # so it structurally cannot protect the template from the cells it excludes.
    # Seed-set robustness has to live in consensus_template itself, which is where
    # the unit-norm rescaling and the low-complexity exclusion now are.
    if nm is not None and nm.sum() >= 10:
        baseline_s = scores[nm]
        bq75 = float(percentile(baseline_s, 75))
        bq25 = float(percentile(baseline_s, 25))
        outlier_fence = bq75 + outlier_fence_mult * (bq75 - bq25)
        # The bulk of the non-reference population. Fencing below this would be
        # cutting into whatever tumor mode the sample has.
        non_ref = scores[~nm]
        tumor_bulk = float(median(non_ref)) if non_ref.size else float("inf")
        if outlier_fence > tumor_bulk:
            outlier_mask = (~nm) & (scores > outlier_fence)
        else:
            outlier_mask = zeros(len(scores), dtype=bool)
    else:
        outlier_mask = zeros(len(scores), dtype=bool)

    scores_for_fit = clip(scores, clip_floor, clip_ceiling)

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
        return pd.DataFrame({
            "class": ["normal"] * len(scores),
            "confidence": [1.0] * len(scores),
            "tumor_score": scores,
        })
    gmm.fit(fit_scores.reshape(-1, 1))

    # Predict for ALL cells using the clipped (but not outlier-excluded) scores.
    # Outlier cells will get predictions too, but they are overridden below.
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

    # Force outlier cells to "normal" — they are transcriptome-extreme, not tumor.
    labels[outlier_mask] = "normal"

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

    After the GMM call, two *subtractive* gates downgrade untrustworthy "tumor"
    calls to "uncertain" (they never promote a cell to tumor, so they cannot
    inflate the tumor set or hurt bulk concordance):

      * low-complexity gate — ambient / empty-droplet-like cells (few detected
        genes) have noise-dominated CN signal; a tumor call on them is flagged
        rather than trusted. Requires ``complexity``; skipped if not supplied.
      * coherence gate — a "tumor" call whose deviation is mostly scattered
        single-segment spikes rather than chromosome-arm-scale (a "loud"
        specialized normal cell) is downgraded. Always applied.

    This encodes the "flag, don't force" philosophy: cells that are genuinely
    ambiguous (including same-lineage normals whose expression mimics CNV) land
    in "uncertain" rather than being asserted as tumor.

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
            disable.

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
    template = consensus_template(dev, seg_weights, exclude=low_complexity)
    scores = template_projection(dev, template, seg_weights)
    # Non-negative companion to the signed score: the gene-weighted L1 burden, in
    # the same per-cell-centered frame. This is what "how much CN does this cell
    # carry" means when direction is not the question — see the cn_burden note in
    # the Returns block above for why the two cannot be the same column.
    cn_burden = segment_burden(np_abs(dev), seg_weights)
    call_df = gmm_classify(
        scores,
        normal_mask=normal_mask,
        confidence_threshold=confidence_threshold,
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

    # A gate overrides a *confident* GMM tumor call, so the reported confidence
    # must reflect the downgrade — otherwise a gate-flagged cell keeps its high
    # tumor posterior and would pass a downstream `confidence >= threshold` filter
    # despite being uncertain. Report the posterior of the surviving (non-tumor)
    # interpretation (1 - tumor posterior), which is < 0.5 for a former tumor call,
    # preserving the invariant "uncertain ⇒ low confidence".
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
    n_segments_altered = (np_abs(dev) > 0.2).sum(axis=1).astype(int)

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
    return out
