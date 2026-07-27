"""Step §3.4 — segment the smoothed signal and compute per-cell CN per segment.

Two-stage, SCEVAN-inspired (multichannel first, then per-cell within):

    1. detect_segments():
       Pool the smoothed signal across putative-tumor cells (mean over
       ~normal_mask) per chromosome, then run PELT changepoint detection
       (ruptures.Pelt with the L2 cost — detects piecewise-constant mean
       shifts, which is exactly what a CNV looks like in centered expression
       space). Penalty scales with log(n_genes_in_chr) so longer chromosomes
       are not penalized into fewer segments.

    2. per_cell_segment_cn():
       For every (cell, segment) pair, compute the mean smoothed signal
       within the segment. Vectorized as a single (n_cells × n_segments)
       allocation; the inner loop is over segments, which is small
       (typically 100-500 globally).

Returned segment table is the segment-level CN call across the whole cohort,
with one row per global segment. Segments are uniform across all cells —
SCEVAN's multichannel trick — so per-cell CN is just an aggregation, not a
re-segmentation per cell.
"""

from numpy import asarray, empty, log as np_log, median as np_median, ones, std as np_std, where
from pandas import DataFrame
from ruptures import Pelt
from ruptures.exceptions import BadSegmentationParameters


# Default PELT penalty multiplier.
# Penalty coefficient for PELT. Operates on the per-chromosome signal normalized to
# zero-mean/unit-variance, so 1.0 means a jump of ~0.3σ on a 1000-gene chromosome
# is detectable. Lower values increase sensitivity; raise to 1.5–2.0 to suppress
# noise segments on low-purity samples.
DEFAULT_PENALTY_COEF = 1.0

# Minimum segment length in genes. Matches CopyKAT's win.size=25 default;
# CNVs spanning fewer than 25 ordered genes are well below the noise floor
# of expression-based CN calling.
DEFAULT_MIN_SEG_GENES = 25


def per_segment_normal_baseline(cn_matrix, normal_mask):
    """Per-segment diploid reference: the median CN over the normal pool.

    This is the segment-level analogue of the per-gene centering done in
    ``smooth.center_against_baseline`` — and the single definition of "where a
    diploid cell sits" that the ABSOLUTE outputs (``tumor_mean``,
    ``chr_cnv_matrix``, ``.seg``) subtract so that 0 = diploid (log space) /
    1.0 = diploid (ratio). It is identical to the baseline the classifier
    already uses internally (``classify._normal_pool_baseline``).

    Why it is needed: ``center_against_baseline`` subtracts the per-gene normal
    median, but genes detected in <=50% of normals have a median of 0 and pass
    through uncentered, stamping a near-constant positive log pedestal on every
    cell. Taking the per-segment median over the normal pool recovers where that
    pedestal sits so the absolute outputs can subtract it and honor the "0 =
    diploid" contract. The relative signal (tumor_score, classification) already
    subtracts this same baseline internally, so it is unaffected.

    Args:
        cn_matrix: (n_cells x n_segments) ndarray from per_cell_segment_cn().
        normal_mask: bool array len == n_cells, True for the normal pool.

    Returns:
        (n_segments,) ndarray: the per-segment median over the normal pool, or
        the whole-cohort median when the normal pool is empty (defensive — so
        the outputs stay centered somewhere sensible rather than on the pedestal).
    """
    if asarray(normal_mask).sum() > 0:
        return np_median(cn_matrix[normal_mask, :], axis=0)
    return np_median(cn_matrix, axis=0)


def detect_segments(
    smoothed,
    normal_mask,
    chr_labels,
    penalty_coef=DEFAULT_PENALTY_COEF,
    min_seg_genes=DEFAULT_MIN_SEG_GENES,
):
    """Detect global segments from the pooled tumor signal, per chromosome.

    The pooled signal is computed as the mean smoothed value across cells
    NOT in the normal pool (i.e. putative tumor cells). PELT then finds
    changepoints in that 1-D signal per chromosome; segments span from one
    changepoint to the next.

    Before running PELT, each chromosome's signal is normalized to zero-mean /
    unit-variance (median-centered, divided by std). This makes the penalty
    coefficient scale-invariant across samples of varying tumor purity: on
    real single-cell data the pooled signal amplitude is 0.1–0.4 (3–10× below
    bulk exome), so without normalization a fixed penalty is never exceeded and
    every chromosome collapses to one segment. Segment boundary positions come
    from the normalized signal; all stored values (tumor_mean, per-cell CN)
    continue to use the original scale.

    Args:
        smoothed: (n_cells × n_genes) ndarray from smooth_along_chromosomes.
        normal_mask: bool array len == n_cells; True for the normal pool.
            Pooled signal averages over the complement (tumor cells).
        chr_labels: array-like of length n_genes giving each gene's chromosome.
        penalty_coef: BIC-style coefficient on log(n_genes_in_chr) for PELT.
        min_seg_genes: Minimum segment length in genes.

    Returns:
        A DataFrame with one row per segment, columns:
            chr        — chromosome label
            start_idx  — inclusive gene-axis start (global gene index)
            end_idx    — exclusive gene-axis end (global gene index)
            n_genes    — segment length in genes
            tumor_mean — the CN call for the segment: the pooled-tumor mean
                         signal minus the per-segment diploid reference (the
                         median over the normal pool). Centered on 0 so
                         positive = gain, negative = loss. The reference removes
                         the positive log pedestal left by center_against_baseline
                         (see per_segment_normal_baseline), without which every
                         segment reads as a gain even where the tumor has a real
                         loss.
    """
    labels = asarray(chr_labels)
    n_cells = smoothed.shape[0]

    # Defensive default: if everyone is in the normal pool, the tumor-pool
    # mean is undefined. Fall back to the cohort mean — the segmentation
    # becomes a fit to the full-cohort signal, which is the right thing to
    # do when no tumor cells exist (the resulting segments will all be
    # near-zero by construction).
    tumor_mask = ~normal_mask
    if tumor_mask.sum() == 0:
        pool_rows = ones(n_cells, dtype=bool)
    else:
        pool_rows = tumor_mask

    # Pooled signal across the tumor pool, per gene. One scalar per gene —
    # this is the 1-D signal PELT segments per chromosome.
    pooled = smoothed[pool_rows, :].mean(axis=0)

    # Changepoint detection + raw (uncentered) tumor_mean per segment. Shared
    # with the blocked M3 path, which computes `pooled` by streaming rather than
    # from a materialized `smoothed` matrix.
    segments_df = segments_from_pooled(
        pooled, chr_labels, penalty_coef=penalty_coef, min_seg_genes=min_seg_genes,
    )

    # Center tumor_mean against the per-segment diploid reference so 0 = diploid.
    # `pooled` (and hence the raw tumor_mean) carries the positive log pedestal
    # that center_against_baseline leaves on genes detected in <=50% of normals,
    # so every segment reads as a gain even where the tumor has a real loss.
    # Subtract the per-segment median over the normal pool — the same diploid
    # reference per_segment_normal_baseline / the classifier / the heatmap use.
    # detect_segments runs before per_cell_segment_cn, so we compute the reference
    # straight from `smoothed` over the normal cells (whole-cohort fallback when
    # the pool is empty); this equals
    # per_segment_normal_baseline(per_cell_segment_cn(smoothed, segments_df), normal_mask).
    if len(segments_df):
        ref_rows = normal_mask if bool(asarray(normal_mask).any()) else ones(n_cells, dtype=bool)
        starts = segments_df["start_idx"].to_numpy()
        ends = segments_df["end_idx"].to_numpy()
        baseline = empty(len(segments_df), dtype="float64")
        for i in range(len(segments_df)):
            baseline[i] = np_median(smoothed[ref_rows, starts[i]:ends[i]].mean(axis=1))
        segments_df["tumor_mean"] = segments_df["tumor_mean"].to_numpy() - baseline

    return segments_df


def segments_from_pooled(
    pooled,
    chr_labels,
    penalty_coef=DEFAULT_PENALTY_COEF,
    min_seg_genes=DEFAULT_MIN_SEG_GENES,
):
    """Detect segments from a per-gene pooled tumor signal (PELT per chromosome).

    Factored out of detect_segments so the blocked M3 path can supply `pooled`
    computed by streaming over cell blocks, without materializing the full
    smoothed matrix. Returns segments with the RAW (uncentered) tumor_mean; the
    caller subtracts the per-segment diploid reference.

    Args:
        pooled: (n_genes,) per-gene mean of the smoothed signal over the tumor
            pool (or the cohort when no tumor cells exist).
        chr_labels: array-like of length n_genes giving each gene's chromosome.
        penalty_coef: BIC-style coefficient on log(n_genes_in_chr) for PELT.
        min_seg_genes: Minimum segment length in genes.

    Returns:
        DataFrame with columns chr, start_idx, end_idx, n_genes, tumor_mean
        (tumor_mean uncentered = pooled mean over the segment's genes).
    """
    labels = asarray(chr_labels)
    pooled = asarray(pooled)

    # Walk each chromosome's contiguous gene block; PELT detects changepoints
    # within that block, producing one set of segments per chromosome.
    seen_chroms = []
    for lbl in labels:
        if seen_chroms and seen_chroms[-1] == lbl:
            continue
        seen_chroms.append(lbl)

    rows = []
    for chrom in seen_chroms:
        col_mask = labels == chrom
        # Translate the boolean mask to a contiguous (start, end) span by
        # finding the first and last indices; the matrix layout guarantees
        # contiguity but using where() is defensive against future changes.
        chr_idx = where(col_mask)[0]
        if chr_idx.size == 0:
            continue
        start_global = int(chr_idx[0])
        end_global = int(chr_idx[-1]) + 1
        chr_signal = pooled[start_global:end_global].astype("float64")
        n_chr_genes = chr_signal.size

        # PELT requires the signal to be at least 2*min_size long; for tiny
        # chromosomes (e.g. our synthetic chr21/22 with few mapped genes),
        # there is no room for an internal changepoint anyway, so emit the
        # whole chromosome as one segment without invoking PELT.
        effective_min = max(min_seg_genes, 2)
        if n_chr_genes < 2 * effective_min:
            bkps = [n_chr_genes]
        else:
            # Normalize the per-chromosome signal to zero-mean/unit-variance before
            # changepoint detection. PELT's L2 cost is scale-sensitive: on real
            # single-cell data the pooled signal amplitude is 0.1-0.4 (3-10x below
            # bulk exome), so the fixed penalty penalty_coef*log(n) is never exceeded
            # and every chromosome gets exactly one segment. Normalizing makes
            # penalty_coef scale-invariant across samples of varying tumor purity.
            # Segment boundary positions come from the normalized signal; all stored
            # values (tumor_mean, per-cell CN) continue to use the original scale.
            signal_std = float(np_std(chr_signal))
            if signal_std < 1e-4:
                # Near-flat chromosome — no detectable variation, emit as one segment.
                bkps = [n_chr_genes]
            else:
                chr_signal_pelt = (
                    (chr_signal - float(np_median(chr_signal))) / signal_std
                ).astype("float64")
                algo = Pelt(model="l2", min_size=effective_min, jump=1)
                algo.fit(chr_signal_pelt.reshape(-1, 1))
                penalty = float(penalty_coef) * float(np_log(max(n_chr_genes, 2)))
                try:
                    # ruptures returns breakpoints as the index ONE-PAST each segment end.
                    # The final breakpoint always equals n_chr_genes.
                    bkps = algo.predict(pen=penalty)
                except BadSegmentationParameters:
                    # PELT refused (degenerate cost surface, near-uniform signal).
                    # Fall back to one segment for the whole chromosome.
                    bkps = [n_chr_genes]

        # Walk breakpoints to enumerate segments. start = previous endpoint
        # (or 0 for the first segment), end = current breakpoint.
        prev = 0
        for bk in bkps:
            seg_start_local = prev
            seg_end_local = bk
            seg_len = seg_end_local - seg_start_local
            if seg_len < min_seg_genes:
                # Merge tiny tail segments into the previous one rather than
                # emit them. Rare given min_size=min_seg_genes above, but the
                # final segment can be short if the chromosome length isn't a
                # multiple of min_size.
                if rows and rows[-1]["chr"] == chrom:
                    rows[-1]["end_idx"] = start_global + seg_end_local
                    rows[-1]["n_genes"] = rows[-1]["end_idx"] - rows[-1]["start_idx"]
                    rows[-1]["tumor_mean"] = float(
                        pooled[rows[-1]["start_idx"]:rows[-1]["end_idx"]].mean()
                    )
                # chromosome has too few genes (< min_seg_genes) to produce a usable segment;
                # it is silently excluded from segmentation. This is intentional: fewer than
                # min_seg_genes genes cannot support changepoint detection reliably.
                prev = seg_end_local
                continue
            seg_start_global = start_global + seg_start_local
            seg_end_global = start_global + seg_end_local
            seg_tumor_mean = float(pooled[seg_start_global:seg_end_global].mean())
            rows.append({
                "chr": chrom,
                "start_idx": int(seg_start_global),
                "end_idx": int(seg_end_global),
                "n_genes": int(seg_len),
                "tumor_mean": seg_tumor_mean,
            })
            prev = seg_end_local

    segments_df = DataFrame(rows, columns=["chr", "start_idx", "end_idx", "n_genes", "tumor_mean"])
    return segments_df


def per_cell_segment_cn(smoothed, segments):
    """Compute the per-cell CN matrix from the global segment table.

    For every (cell, segment) pair, computes the mean of the smoothed signal
    over the segment's gene span. Result is a dense (n_cells × n_segments)
    ndarray that downstream classification (M4) consumes.

    Args:
        smoothed: (n_cells × n_genes) ndarray from smooth_along_chromosomes.
        segments: DataFrame from detect_segments. Must have start_idx, end_idx.

    Returns:
        ndarray of shape (n_cells, len(segments)). Column i corresponds to
        segments.iloc[i]. Values are centered on 0 (per-cell deviation from
        the normal-pool baseline averaged across the segment).
    """
    # Pre-allocate the output — float32 matches the smoothed dtype and keeps
    # memory bounded. For 10k cells × 500 segments that's 20 MB.
    n_cells = smoothed.shape[0]
    n_segments = len(segments)
    out = empty((n_cells, n_segments), dtype=smoothed.dtype)

    # Vectorize per-segment: column i = row-mean over the segment's gene span.
    # Iterating over segments rather than cells is the right axis because
    # n_segments << n_cells in practice.
    starts = segments["start_idx"].to_numpy()
    ends = segments["end_idx"].to_numpy()
    for i in range(n_segments):
        out[:, i] = smoothed[:, starts[i]:ends[i]].mean(axis=1)

    return out
