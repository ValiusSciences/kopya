"""M4 output writers.

Three artifacts:
    write_prediction_csv(): per-cell call CSV. Drop-in for copykat_prediction.csv —
        same shape and column semantics so pipeline_outputs can consume both.
    write_chr_cnv_matrix_csv(): cells × chromosomes summary CSV, values centered
        on 1.0 (gain > 1, loss < 1) matching CopyKAT's downstream convention.
    write_clones_seg(): IGV-loadable per-clone consensus .seg file. Genomic
        coordinates from var.chr/start/end; one row per (clone, segment).

Plus one optional transform of the second artifact:
    center_chr_cnv_matrix(): divide each cell's row by its own median chromosome,
        removing the per-cell offset that compute_chr_cnv_matrix() leaves in. Used
        by the CLI's --centered-chr-matrix to write chr_cnv_matrix_centered.csv
        alongside — never instead of — the raw file.
"""

from pathlib import Path

from numpy import arange, asarray, empty, exp, full, median, where
from pandas import DataFrame, unique


# Each chromosome's "CN value" in chr_cnv_matrix is the mean per-segment CN
# across all segments in that chromosome, then mapped from log-space deviation
# to multiplicative scale centered on 1.0 via this conversion. exp() is
# accurate at any magnitude; log-space deviations are typically small so the
# result lands near 1.0 for diploid cells.
def _logspace_to_one_centered(values):
    """Convert log-space CN deviations (centered on 0) to gain/loss scale on 1.

    Args:
        values: ndarray of deviations from the diploid baseline in log space.

    Returns:
        ndarray of the same shape, with values ~1 for diploid, >1 for gain,
        <1 for loss. Uses exp() so the mapping is exact and reversible.
    """
    # exp(0) == 1, so diploid (zero deviation) maps to 1.0 by construction.
    # exp() preserves direction and magnitude of departures from baseline.
    one_centered = exp(values)
    return one_centered


def write_prediction_csv(prediction_df, out_path):
    """Write the per-cell prediction DataFrame to CSV.

    The schema mirrors CopyKAT's copykat_prediction.csv with our
    additional columns. Column order is fixed for downstream stability.

    Args:
        prediction_df: DataFrame from classify_cells() indexed by barcode.
        out_path: Destination CSV path (parent dir must exist).

    Returns:
        The output path as a Path object (convenience for chaining).
    """
    # Force a stable column order so downstream consumers parse positionally
    # if they want; column names remain authoritative for keyed access.
    # low_complexity is appended last so the historical column positions are
    # unchanged for any positional reader of the CopyKAT-compatible prefix.
    cols = ["class", "confidence", "tumor_score", "subclone", "n_segments_altered"]
    if "low_complexity" in prediction_df.columns:
        cols = cols + ["low_complexity"]
    # cn_burden is appended after low_complexity for the same reason low_complexity
    # itself is appended last: it keeps every historical column position intact for
    # a positional reader of the CopyKAT-compatible prefix.
    if "cn_burden" in prediction_df.columns:
        cols = cols + ["cn_burden"]
    ordered = prediction_df[cols]

    target = Path(out_path)
    ordered.to_csv(target, index_label="barcode")
    return target


def compute_chr_cnv_matrix(cn_matrix, segments, chrom_order, baseline=None):
    """Aggregate the per-cell × per-segment CN matrix to per-chromosome means.

    For each chromosome, averages every segment's CN value weighted by segment
    length (n_genes). The result is converted from log-space deviation to a
    1.0-centered multiplicative scale via exp().

    Args:
        cn_matrix: (n_cells × n_segments) ndarray.
        segments: DataFrame from detect_segments(); must have columns
            chr and n_genes.
        chrom_order: Iterable of chromosome labels in the order columns
            should appear in the output (typically CANONICAL_CHROM_ORDER).
        baseline: optional diploid reference subtracted from CN before aggregation
            so diploid cells land at 0 in log space and 1.0 after exp() — honoring
            the "1.0 = diploid" contract. Either a scalar (a single global pedestal)
            or a (n_segments,) per-segment vector; both broadcast over the CN matrix.
            Without it (None) the raw CN carries the positive per-gene pedestal left
            by centering and diploid cells sit above 1.0 (~1.27 on real data).

            The CLI passes a SCALAR here on purpose. chr_cnv_matrix is the output
            validated against matched bulk (per-chromosome Pearson). Pearson is
            invariant to a constant shift, so subtracting the global pedestal
            removes the gross diploid offset without touching the tumor profile's
            per-chromosome shape. A per-segment vector would also subtract the
            normal pool's per-chromosome residual — which on a transcriptionally
            heterogeneous normal pool is partly tumor-correlated (the known
            reference-frame limitation), and removing it measurably regresses the
            matched-bulk concordance. So the finer per-segment reference is used
            for the interpretability outputs (tumor_mean, .seg) that are not
            benchmarked against bulk, and only the scalar pedestal here.

            Accepted trade-off: a scalar removes the gross offset but cannot
            perfectly center a chromosome whose normal baseline deviates from the
            genome-wide median — such a neutral chromosome lands near
            exp(b_chr - median(baseline)), i.e. within a few percent of 1.0 rather
            than exactly 1.0 (e.g. diploid normals in the 0.95-1.07 range, median ~1.01).
            Removing that last per-chromosome residual would require the per-segment
            vector, which regresses the validated concordance, so it is left in.

    Calibrated only up to a per-cell scale factor. The upstream per-gene
    centering leaves a positive log pedestal on every cell — genes detected in
    <=50% of the normal pool have a normal-median of 0 and pass through
    uncentered, so a cell's pedestal grows with how many genes it detected — and
    the scalar `baseline` above removes only the pool-wide part of it.

    What survives is a factor on a cell's WHOLE row. It cancels in any comparison
    made *within* one cell (a chromosome ratio, a per-cell ranking of
    chromosomes), and in any average taken *over* cells (a pseudobulk — which is
    what the matched-bulk Pearson grades, so that metric cannot see it). It
    does NOT cancel when cells are compared to each other at a fixed chromosome:
    on real data that offset correlates strongly with sequencing depth and can
    exceed the per-chromosome biology, so a heatmap or ranking built on the raw
    values can show depth rather than copy number. Anything that ranks, sorts or
    colours individual cells should call center_chr_cnv_matrix() first.

    Left in deliberately: the alternative is to make 1.0 mean "this cell's median
    chromosome" instead of "diploid", which is a relative frame that cannot state
    ploidy — a whole-genome doubling would be invisible by construction. This
    function keeps the absolute frame; center_chr_cnv_matrix() offers the
    relative one as a separate, additive output.

    Returns:
        (matrix, chroms_present):
            matrix: (n_cells × n_chroms_present) ndarray, centered on 1.0
                (up to the per-cell factor described above).
            chroms_present: list[str] of chromosomes that had at least one
                segment, in chrom_order order.
    """
    # Subtract the diploid reference so a diploid cell's per-segment deviation is
    # ~0; the exp() below then maps it to ~1.0. A scalar or a (n_segments,) vector
    # both broadcast over (n_cells, n_segments). Cast to the CN dtype so the
    # subtraction does not silently upcast the working matrix.
    if baseline is not None:
        cn_matrix = cn_matrix - asarray(baseline, dtype=cn_matrix.dtype)

    # Pre-compute per-chromosome segment lookups for fast per-chr aggregation.
    segments_by_chr = {chrom: [] for chrom in chrom_order}
    for i, row in enumerate(segments.itertuples(index=False)):
        if row.chr in segments_by_chr:
            segments_by_chr[row.chr].append((i, int(row.n_genes)))

    chroms_present = [c for c in chrom_order if segments_by_chr[c]]
    n_cells = cn_matrix.shape[0]
    out = empty((n_cells, len(chroms_present)), dtype=cn_matrix.dtype)

    # Length-weighted mean per chromosome — a longer segment with the same
    # mean carries more weight than a tiny boundary segment, matching how a
    # human reads a CNV heatmap.
    for col_idx, chrom in enumerate(chroms_present):
        seg_indices, seg_weights = zip(*segments_by_chr[chrom])
        chr_block = cn_matrix[:, list(seg_indices)]  # (n_cells, n_chr_segs)
        weights = asarray(seg_weights, dtype=cn_matrix.dtype)
        # weighted mean: sum(values * weights) / sum(weights), along axis=1.
        weighted_sum = (chr_block * weights).sum(axis=1)
        total_weight = float(weights.sum())
        out[:, col_idx] = weighted_sum / total_weight if total_weight > 0 else 0.0

    one_centered = _logspace_to_one_centered(out)
    return one_centered, chroms_present


def center_chr_cnv_matrix(matrix):
    """Divide each cell's row by its own median chromosome.

    The one operation that removes the per-cell offset described in
    compute_chr_cnv_matrix(). Nothing else is applied — no log, no scaling, no
    per-chromosome correction:

        centered[cell, chrom] = matrix[cell, chrom] / median(matrix[cell, :])

    The median is unweighted over the chromosome columns present. (heatmap.py's
    _recenter() uses a gene-count-weighted median over *segments*; here the
    segments have already been collapsed to one value per chromosome, and the
    unweighted form is the one validated on the patient cohort.)

    Output stays on the same 1.0-centered multiplicative scale as the input, so
    it is a drop-in for any consumer of the raw matrix, and every within-row
    ratio is preserved exactly — dividing a row by a constant cannot create or
    remove a chromosome-level gain or loss. For a symmetric plotting scale, take
    log2() of the result; that is a separate transform and is left to the caller.

    Note the frame changes: 1.0 now means "this cell's median chromosome", not
    "diploid". For a cell whose median chromosome is genuinely altered — e.g. a
    whole-genome doubling, or a tumor cell with most chromosomes gained — that
    misstates ploidy. Use this for visualization and relative per-cell CNV
    interpretation; keep the raw matrix as the source of truth for absolute
    copy-number level.

    Args:
        matrix: (n_cells × n_chroms) ndarray from compute_chr_cnv_matrix(),
            on the 1.0-centered multiplicative scale (strictly positive).

    Returns:
        ndarray of the same shape and dtype, each row scaled so its median is 1.0.
    """
    values = asarray(matrix)
    # keepdims so the (n_cells, 1) medians broadcast back over the chromosome axis.
    row_medians = median(values, axis=1, keepdims=True)
    # The input is exp()-derived and strictly positive, so a zero median can only
    # come from a degenerate/empty input; leave those rows untouched rather than
    # emitting inf/nan into a file consumers read as a ratio.
    safe = where(row_medians == 0, 1.0, row_medians)
    return (values / safe).astype(values.dtype, copy=False)


def write_chr_cnv_matrix_csv(matrix, barcodes, chroms, out_path):
    """Write the cells × chromosomes CN matrix to CSV.

    Args:
        matrix: (n_cells × n_chroms) ndarray, already 1.0-centered.
        barcodes: array-like of cell barcodes, len == n_cells.
        chroms: list[str] of chromosome labels, len == n_chroms.
        out_path: Destination CSV path.

    Returns:
        The output path as a Path object.
    """
    # Build the DataFrame with the correct index/columns so downstream loaders
    # do not have to guess at orientation.
    df = DataFrame(matrix, index=asarray(barcodes), columns=chroms)
    df.index.name = "barcode"

    target = Path(out_path)
    df.to_csv(target)
    return target


def segments_with_coordinates(segments, var_coords):
    """Enrich the segment table with genomic coordinates for reuse across results.

    detect_segments() addresses each segment by gene-axis position — ``start_idx``
    (inclusive) and ``end_idx`` (exclusive) index into the ordered gene axis, not
    the genome. This helper maps those indices to genomic base pairs using
    ``var_coords`` (adata.var post-M1, one row per gene with chr/start/end), so the
    persisted ``segments.parquet`` is a self-contained *segment → chromosome-location*
    mapping. Row ``i`` of the returned table corresponds to column ``i`` of the
    per-cell CN matrix (``cn_per_segment.npz``); the explicit ``segment_id`` column
    makes that positional key stable even if a consumer filters or reorders rows.

    Args:
        segments: DataFrame from detect_segments() with at least the columns
            chr, start_idx, end_idx, n_genes, tumor_mean.
        var_coords: DataFrame indexed by gene symbol with chr/start/end columns
            (i.e. adata.var post-M1), addressed positionally to match the gene
            axis the segment indices refer to.

    Returns:
        A new DataFrame with the original columns (positions unchanged) plus three
        appended columns: ``segment_id`` (0-based positional key), ``start_bp`` and
        ``end_bp`` — the genomic span of the segment, i.e. the minimum gene start
        and maximum gene end across every gene it contains. The input is not
        mutated. Columns are appended (not reordered) so positional consumers of
        the original schema are unaffected.
    """
    var_start = var_coords["start"].to_numpy()
    var_end = var_coords["end"].to_numpy()
    # Coerce to an integer dtype before using as fancy indices: an empty segment
    # table (detect_segments() found no segments) has object-dtype index columns,
    # which NumPy rejects as indices. asarray(..., dtype=int) handles both the
    # empty-object and the populated-int cases.
    start_idx = asarray(segments["start_idx"].to_numpy(), dtype=int)
    end_idx = asarray(segments["end_idx"].to_numpy(), dtype=int)

    # Span the WHOLE gene range [start_idx, end_idx) for the genomic bounds. Genes
    # are ordered by start, but ends are not monotonic (genes overlap / nest), so
    # the last gene's end is not necessarily the segment's rightmost base — take
    # the min start and max end over all the segment's genes. (A list comp over
    # the segment axis is fine: n_segments is small, and it handles the empty case
    # cleanly, yielding an empty array.)
    start_bp = asarray([int(var_start[s:e].min()) for s, e in zip(start_idx, end_idx)], dtype=int)
    end_bp = asarray([int(var_end[s:e].max()) for s, e in zip(start_idx, end_idx)], dtype=int)

    # Append the new columns (do not reorder) so the change is purely additive and
    # positional readers of the original schema keep working.
    out = segments.copy()
    out["segment_id"] = arange(len(out), dtype=int)
    out["start_bp"] = start_bp
    out["end_bp"] = end_bp
    return out


def _compute_clone_consensus_segments(cn_matrix, segments, prediction_df, var_coords, baseline=None):
    """Build a per-clone consensus segment table for IGV .seg export.

    For each subclone, computes the median CN per segment across cells in
    that subclone and emits it as the .seg value column. Values are in the
    pipeline's native log space — median-centered log1p(CP10k), i.e. a
    natural-log expression ratio relative to the normal-pool reference, NOT a
    literal DNA copy-number log2 ratio. IGV renders seg.mean on a diverging
    color scale, so the sign/relative magnitude read correctly; only the
    absolute base differs from the log2 convention.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray.
        segments: DataFrame from detect_segments() with start_idx/end_idx.
        prediction_df: classify_cells() output (indexed by barcode).
        var_coords: DataFrame indexed by gene_symbol with chr/start/end columns
            (i.e. adata.var post-M1). Used to map gene-index → genomic coords.
        baseline: optional (n_segments,) per-segment diploid reference from
            segment.per_segment_normal_baseline(). When provided it is subtracted
            per segment so seg.mean is centered on 0 (diploid) in log space,
            rather than sitting on the positive per-gene centering pedestal.

    Returns:
        DataFrame in IGV .seg format with columns:
            ID, chrom, loc.start, loc.end, num.mark, seg.mean.
    """
    # Build the per-clone cell groupings. Tumor cells without a subclone label
    # ("uncertain" or supervised-normal-only cohorts) are bundled as 'all_tumor'
    # so we never silently drop a tumor cell from the .seg.
    rows = []
    tumor_pred = prediction_df[prediction_df["class"] == "tumor"]
    if tumor_pred.empty:
        # No tumor cells — return an empty frame with the right columns so
        # downstream code can write an empty .seg without failing.
        return DataFrame(columns=["ID", "chrom", "loc.start", "loc.end", "num.mark", "seg.mean"])

    # Subtract the per-segment diploid reference so seg.mean is centered on 0.
    # A (n_segments,) vector broadcasts over (n_cells, n_segments); done once here
    # so the median over each clone's cells is already in reference-relative space.
    if baseline is not None:
        cn_matrix = cn_matrix - asarray(baseline, dtype=cn_matrix.dtype)

    # Normalize NaN → "" so the checks below are stable whether the DataFrame
    # was built in-memory (subclone always "") or reloaded from prediction.csv
    # (where pandas' default NA handling turns "" back into NaN).
    subclone_col = tumor_pred["subclone"].fillna("")
    if (subclone_col == "").all():
        groups = {"all_tumor": tumor_pred.index.to_numpy()}
    else:
        groups = {
            clone: tumor_pred.index[subclone_col == clone].to_numpy()
            for clone in unique(subclone_col)
            if clone != ""
        }

    # Pre-resolve gene-index → genomic-coord lookups; var_coords is indexed by
    # gene symbol and we need to address its rows by integer position.
    var_chr = var_coords["chr"].to_numpy()
    var_start = var_coords["start"].to_numpy()
    var_end = var_coords["end"].to_numpy()

    # Cell-name to row-index lookup (the cn_matrix rows are in barcode order).
    barcode_to_idx = {bc: i for i, bc in enumerate(prediction_df.index)}

    # Walk each clone × segment combination by positional index so we can
    # reach into cn_matrix columns directly. itertuples loses the segment
    # row index when index=False; positional iteration is simpler and faster.
    seg_starts = segments["start_idx"].to_numpy()
    seg_ends = segments["end_idx"].to_numpy()
    n_segs = len(segments)
    for clone, clone_barcodes in groups.items():
        clone_rows = asarray([barcode_to_idx[bc] for bc in clone_barcodes])
        for s_i in range(n_segs):
            s_start = int(seg_starts[s_i])
            s_end = int(seg_ends[s_i])
            chrom = str(var_chr[s_start])
            loc_start = int(var_start[s_start])
            loc_end = int(var_end[s_end - 1])
            # IGV renders seg.mean centred on 0; by convention it is a log2
            # ratio, but we emit the pipeline's native natural-log (log1p)
            # values — the sign and relative magnitude read correctly, only the
            # log base differs. cn_matrix has had the per-segment diploid
            # reference subtracted above (when a baseline was supplied), so its
            # values are a log ratio relative to the normal pool — emitted
            # directly, no further transformation. Without a baseline they still
            # sit on the centering pedestal (legacy behavior, preserved for
            # callers that omit it).
            consensus_cn = float(median(cn_matrix[clone_rows, s_i]))
            num_mark = s_end - s_start
            rows.append({
                "ID": clone,
                "chrom": chrom,
                "loc.start": loc_start,
                "loc.end": loc_end,
                "num.mark": int(num_mark),
                "seg.mean": consensus_cn,
            })

    seg_df = DataFrame(rows, columns=["ID", "chrom", "loc.start", "loc.end", "num.mark", "seg.mean"])
    return seg_df


def write_clones_seg(cn_matrix, segments, prediction_df, var_coords, sample, out_path, baseline=None):
    """Write an IGV-loadable per-clone consensus .seg file.

    Args:
        cn_matrix: (n_cells × n_segments) ndarray.
        segments: detect_segments() output.
        prediction_df: classify_cells() output indexed by barcode.
        var_coords: adata.var post-M1 (chr, start, end columns).
        sample: Sample identifier; prepended to clone IDs in the .seg file
            so multi-sample IGV sessions stay disambiguated.
        out_path: Destination .seg path.
        baseline: optional (n_segments,) per-segment diploid reference from
            segment.per_segment_normal_baseline(); subtracted so seg.mean is
            centered on 0 (diploid) in the pipeline's native log space (natural
            log / log1p), not the base-2 of the strict IGV convention.

    Returns:
        The output path as a Path object.
    """
    consensus = _compute_clone_consensus_segments(
        cn_matrix=cn_matrix,
        segments=segments,
        prediction_df=prediction_df,
        var_coords=var_coords,
        baseline=baseline,
    )

    # Prefix the clone ID with the sample so IGV displays "{sample}::{clone}"
    # and segments from different samples never collide.
    if not consensus.empty:
        consensus = consensus.copy()
        consensus["ID"] = sample + "::" + consensus["ID"].astype(str)

    target = Path(out_path)
    # IGV .seg is tab-delimited; the first row is the header.
    consensus.to_csv(target, sep="\t", index=False)
    return target
