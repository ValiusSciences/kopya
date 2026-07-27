"""Step §3.5 — classify cells as tumor/normal/uncertain and discover subclones.

Three steps:
    1. compute_tumor_scores(): per-cell L1 distance from the normal-pool median
       across the per-segment CN matrix. Cells with no CN deviation score near 0;
       cells with large amp/del events score high.

    2. gmm_classify(): 2-component Gaussian Mixture on the tumor-score vector.
       Higher-mean component = tumor; lower = normal. Posterior probabilities
       below --call-confidence (default 0.5) collapse to the "uncertain" label
       so downstream notebooks can filter them out cleanly.

    3. discover_subclones(): Leiden clustering on the per-cell × per-segment CN
       matrix, restricted to tumor cells. Resolution is data-driven via a
       sweep, capped at max_subclones (default 5) to keep the clone-tree
       interpretable.

classify_cells() composes the three into a single per-cell DataFrame.
"""

from anndata import AnnData
from numpy import (
    abs as np_abs,
    asarray,
    clip,
    full,
    median,
    minimum as np_minimum,
    ones,
    percentile,
    sign as np_sign,
    unique,
    where,
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
# than CNV from being incorrectly called as tumor.
DEFAULT_OUTLIER_FENCE_MULT = 7.0

# Floor multiplier for the global-median-based clip ceiling guard.
# clip_ceiling is clamped to at least CLIP_FLOOR_MULT × global_median so
# the ceiling never falls below the true tumor signal in high-tumor-fraction
# samples (where most cells are malignant and pull the global median up).
# 1.24 is slightly more permissive than the original 1.2, which prevents
# over-clipping in datasets where the tumor signal is moderate.
DEFAULT_CLIP_FLOOR_MULT = 1.24

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

# Moving-average width (in segments) for the local-contiguity coherence measure.
# Small so it captures short contiguous runs without spanning whole chromosomes.
COHERENCE_WINDOW = 5


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


def _coherent_fraction(cn_matrix, segments, baseline, window=COHERENCE_WINDOW):
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
        cn_matrix: (n_cells × n_segments) ndarray. Pass only the rows you need
            scored (e.g. the tumor-called cells) — the result is per-row and
            independent of which other rows are present.
        segments: DataFrame from detect_segments(); the ``chr`` column groups
            segments into chromosomes (smoothing never crosses a boundary).
        baseline: (n_segments,) per-segment normal-pool baseline from
            _normal_pool_baseline(); subtracted from every row before smoothing.
        window: moving-average width in segments, clamped per chromosome.

    Returns:
        1-D float array len == cn_matrix.shape[0], each in [0, 1].
    """
    dev = cn_matrix - baseline
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


def compute_tumor_scores(cn_matrix, normal_mask, baseline=None):
    """Per-cell L1 distance from the normal-pool median CN vector.

    The normal-pool median is the segment-wise median CN across cells in the
    normal pool — by construction, a per-segment baseline of ~0 (because the
    upstream centering already subtracted the per-gene normal median, and
    segment means inherit that). Each cell's score is the sum of absolute
    deviations from this baseline across all segments.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray from per_cell_segment_cn().
        normal_mask: bool array len == n_cells, True for cells in normal pool.
        baseline: optional precomputed per-segment baseline (from
            _normal_pool_baseline); computed from normal_mask when omitted. Lets
            classify_cells compute the baseline once and share it across the
            score, coherence gate, and n_segments_altered.

    Returns:
        1-D float array of tumor scores, len == n_cells.
    """
    if baseline is None:
        baseline = _normal_pool_baseline(cn_matrix, normal_mask)

    # L1 across segments per cell. Sum (not mean) so cells with broader
    # alterations score higher than cells with one focal event — matches the
    # "aneuploidy load" intuition behind CopyKAT/SCEVAN tumor calling.
    deviations = np_abs(cn_matrix - baseline)
    scores = deviations.sum(axis=1)
    return scores


def gmm_classify(
    scores,
    normal_mask=None,
    confidence_threshold=DEFAULT_CALL_CONFIDENCE,
    outlier_fence_mult=DEFAULT_OUTLIER_FENCE_MULT,
    clip_floor_mult=DEFAULT_CLIP_FLOOR_MULT,
):
    """Fit a 2-component GMM on the tumor-score vector and call each cell.

    Args:
        scores: 1-D ndarray of per-cell tumor scores from compute_tumor_scores().
        normal_mask: Optional bool array; when supplied, cells flagged as
            confident-normal upstream remain "normal" regardless of GMM
            output. Without this, signature/supervised confidence is lost
            because GMM may put a confident-normal cell into the tumor
            cluster when the score happens to be middling.
        confidence_threshold: Cells with max posterior below this become
            "uncertain" rather than a hard tumor/normal call.
        outlier_fence_mult: Multiplier on the baseline IQR to define the
            outlier exclusion fence. Non-baseline cells scoring above
            Q75 + outlier_fence_mult * IQR are excluded from GMM fitting
            and forced to "normal". This prevents transcriptomically extreme
            cell types (Erythrocytes, Platelets) — whose high scores reflect
            transcriptome mismatch rather than CNV — from contaminating the
            GMM and displacing the true tumor signal.
        clip_floor_mult: Floor multiplier applied to the global score median
            when computing the clip ceiling. Ensures the ceiling does not
            fall below clip_floor_mult × median (guards against over-clipping
            in high-tumor-fraction samples where tumor cells pull the median
            up). Default 1.24.

    Returns:
        DataFrame with columns:
            class: str in {tumor, normal, uncertain}.
            confidence: float in [0, 1] (max GMM posterior).
            tumor_score: float (the per-cell L1 score that drove the call).
    """
    # Winsorize scores before GMM fitting to neutralise transcriptional outliers.
    # Rare cell types (e.g. Erythrocytes, Dendritic cells) can have extremely
    # high scores that are driven by their unusual transcriptional state rather
    # than by CNV. These create a spurious high-score component in the GMM,
    # displacing the TRUE tumor cluster (moderately elevated scores) into the
    # "normal" component.
    #
    # Strategy: if a normal_mask is available, use those known-normal cells to
    # define the "baseline score distribution". Use the stricter of:
    #   - Baseline Q90 (the 90th percentile of normal-pool scores)
    #   - Baseline Q75 + 1.5*IQR (Tukey fence for outlier removal)
    # This bounds the GMM fitting range without losing the tumor signal.
    # Also ensure the clip ceiling doesn't fall below 1.2x the global median
    # (to avoid over-clipping in high-tumor-fraction samples).
    # Without a normal_mask, fall back to the global 99th percentile.
    if normal_mask is not None and asarray(normal_mask, dtype=bool).sum() >= 10:
        nm = asarray(normal_mask, dtype=bool)
        baseline_s = scores[nm]
        bq75 = float(percentile(baseline_s, 75))
        bq25 = float(percentile(baseline_s, 25))
        bq90 = float(percentile(baseline_s, 90))
        biqr = bq75 - bq25
        tukey_fence = bq75 + 1.5 * biqr
        clip_ceiling = min(bq90, tukey_fence)
        global_q50 = float(percentile(scores, 50))
        clip_ceiling = max(clip_ceiling, global_q50 * clip_floor_mult)

        # Outlier detection: non-baseline cells scoring well above the baseline
        # distribution (> outlier_fence_mult * IQR above Q75) are likely
        # transcriptome-mismatch cells, not CNV-driven. Exclude them from
        # GMM fitting and lock them to "normal" in the output.
        #
        # Guard: only apply outlier exclusion when the fence is above the
        # clip ceiling. When fence <= clip_ceiling, the outlier fence is
        # tighter than the clip (typical of very homogeneous baselines with
        # small IQR). In that case the fence would incorrectly exclude
        # legitimate high-scoring tumor cells. The clip itself already handles
        # the compression needed for GMM stability.
        outlier_fence = bq75 + outlier_fence_mult * biqr
        if outlier_fence > clip_ceiling:
            outlier_mask = (~nm) & (scores > outlier_fence)
        else:
            outlier_mask = asarray([False] * len(scores), dtype=bool)
    else:
        clip_ceiling = float(percentile(scores, 99))
        nm = asarray(normal_mask, dtype=bool) if normal_mask is not None else None
        outlier_mask = asarray([False] * len(scores), dtype=bool)

    scores_for_fit = clip(scores, 0, clip_ceiling)

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
            tumor_score: float.
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
    scores = compute_tumor_scores(cn_matrix, normal_mask, baseline=baseline)
    call_df = gmm_classify(
        scores,
        normal_mask=normal_mask,
        confidence_threshold=confidence_threshold,
    )
    labels = call_df["class"].to_numpy().astype(object)
    confidence = call_df["confidence"].to_numpy().astype(float).copy()
    gated = full(cn_matrix.shape[0], False)

    # ── gates: downgrade untrustworthy tumor calls to "uncertain" ─────────
    # low-complexity (ambient) gate — only when complexity is supplied.
    low_complexity = _low_complexity_mask(complexity, low_complexity_frac, cn_matrix.shape[0])
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
            coh_frac = _coherent_fraction(cn_matrix[tumor_idx], segments, baseline)
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
        subclones = discover_subclones(
            cn_matrix,
            tumor_mask=tumor_mask,
            max_subclones=max_subclones,
        )
    else:
        # Disabled: every tumor cell carries an empty subclone label.
        subclones = full(cn_matrix.shape[0], "", dtype=object).astype(str)

    # ── n_segments_altered per cell ───────────────────────────────────────
    # Count segments where the cell's deviation from the normal-pool baseline
    # exceeds 0.2 in log space. Useful for downstream filtering and for
    # spotting "barely-tumor" calls with one or two focal events. Reuses the
    # baseline computed above.
    deviations = np_abs(cn_matrix - baseline)
    n_segments_altered = (deviations > 0.2).sum(axis=1).astype(int)

    # ── assemble per-cell DataFrame ───────────────────────────────────────
    out = DataFrame(
        {
            "class": labels.astype(str),
            "confidence": confidence,
            "tumor_score": call_df["tumor_score"].to_numpy(),
            "subclone": subclones,
            "n_segments_altered": n_segments_altered,
            "low_complexity": low_complexity,
        },
        index=asarray(barcodes),
    )
    out.index.name = "barcode"
    return out
