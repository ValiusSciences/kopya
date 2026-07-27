"""Memory-bounded (row-blocked) M3: center → smooth → segment → per-cell CN.

The direct path (``center_against_baseline`` → ``smooth_along_chromosomes``)
densifies the whole ``(n_cells × n_genes)`` matrix, so peak memory scales with
cell count and large cohorts need tens of GB (see issue #24). This module
produces byte-for-byte-equivalent segments + per-cell CN while never holding
more than one cell-block of the dense matrix at a time:

  1. per-gene centering baseline (median over the normal pool) — computed from
     the SPARSE matrix, no densify;
  2. pass 1 — stream cell-blocks, accumulate the per-gene tumor-pool mean
     (``pooled``), then run PELT on it (``segments_from_pooled``);
  3. pass 2 — stream cell-blocks again, write each block's per-segment CN into
     the ``(n_cells × n_segments)`` output (small);
  4. center ``tumor_mean`` against the per-segment normal baseline (identical to
     ``detect_segments``).

Peak dense footprint is ``O(block_size × n_genes)``, independent of ``n_cells``.
"""
from numpy import array, asarray, concatenate, empty, float32, float64, ones, zeros
from scipy.sparse import issparse

from kopya.segment import (
    DEFAULT_MIN_SEG_GENES,
    DEFAULT_PENALTY_COEF,
    per_cell_segment_cn,
    per_segment_normal_baseline,
    segments_from_pooled,
)
from kopya.smooth import DEFAULT_SMOOTH_WINDOW, smooth_along_chromosomes

# Target dense-block footprint in bytes (one (block × genes) float32 array).
# Block size is chosen so a block lands near this; ~1.5 GB keeps peak modest
# while amortizing per-block overhead.
_DEFAULT_BLOCK_BYTES = 1_500_000_000


def auto_block_size(n_cells, n_genes, target_bytes=_DEFAULT_BLOCK_BYTES):
    """Cells per block so one dense (block × genes) float32 array ≈ target_bytes."""
    per_cell = max(1, n_genes * 4)
    block = max(1, int(target_bytes // per_cell))
    return min(block, n_cells)


def per_gene_normal_median(X, normal_mask):
    """Per-gene median over the normal pool, exact, from the sparse matrix.

    Reproduces ``median(X_dense[normal_mask], axis=0)`` (what
    ``center_against_baseline`` computes) without densifying. For each gene it
    combines the explicit nonzeros in normal rows with the implicit zeros and
    takes the median; ``log1p(CP10k)`` values are ≥ 0 so zeros sort first.

    Empty-pool fallback matches ``center_against_baseline``: when ``normal_mask``
    selects no cells, the median is taken over the WHOLE cohort (all cells), not
    zeroed. (An all-zero result only occurs in the degenerate case of a matrix
    with no rows at all.)

    Args:
        X: (n_cells × n_genes) scipy sparse matrix (or dense — falls back to
            ``np.median`` on the normal rows, or all cells if the pool is empty).
        normal_mask: bool array, True for the normal pool.

    Returns:
        (n_genes,) float32 per-gene median over the normal pool (or over all
        cells when the pool is empty).
    """
    nm = asarray(normal_mask, dtype=bool)
    if not issparse(X):
        from numpy import median as np_median
        sub = asarray(X)[nm] if nm.any() else asarray(X)
        return np_median(sub, axis=0).astype(float32)

    Xn = (X[nm] if nm.any() else X).tocsc()
    # A legal but non-canonical sparse matrix can store duplicate (i, j) entries;
    # tocsc() does not necessarily merge them, which would make vals.size count
    # stored entries rather than cells and corrupt the zero count. Canonicalize so
    # the per-column value list has one entry per nonzero cell (matches the dense
    # np.median, which sums duplicates).
    Xn.sum_duplicates()
    n = Xn.shape[0]
    n_genes = Xn.shape[1]
    out = zeros(n_genes, dtype=float32)
    if n == 0:
        return out
    indptr = Xn.indptr
    data = Xn.data
    lo = (n - 1) // 2   # lower-middle index of the full n-length sorted column
    hi = n // 2         # upper-middle (== lo when n is odd); numpy averages both
    from numpy import sort as np_sort
    for g in range(n_genes):
        vals = data[indptr[g]:indptr[g + 1]]
        nz = vals.size
        if nz == 0:
            continue  # all zeros → median 0 (already set)
        n_zero = n - nz
        if n_zero >= hi + 1 and not (vals < 0).any():
            continue  # both middle indices fall in the zero run → median 0
        if (vals < 0).any():
            # General fallback: materialize zeros and sort (rare — signal ≥ 0).
            merged = concatenate([vals, zeros(n_zero, dtype=vals.dtype)])
            merged.sort()
            out[g] = (merged[lo] + merged[hi]) / 2.0
        else:
            sv = np_sort(vals)
            lo_v = 0.0 if lo < n_zero else float(sv[lo - n_zero])
            hi_v = 0.0 if hi < n_zero else float(sv[hi - n_zero])
            out[g] = (lo_v + hi_v) / 2.0
    return out


def _densify_center(X, start, end, baseline_gene):
    """Dense, centered cell-block ``X[start:end] - baseline_gene`` as float32.

    Always returns a fresh array: the in-place subtraction must never alias the
    source. ``toarray()`` copies for sparse input; for a dense ndarray, ``X[s:e]``
    is a view, so ``array(..., copy=True)`` is required — otherwise the block is
    a view into the caller's matrix and the subtraction would mutate it (and, in
    the two-pass flow, double-subtract on pass 2).
    """
    block = X[start:end]
    dense = block.toarray().astype(float32, copy=False) if issparse(block) else array(block, dtype=float32)
    dense -= baseline_gene  # broadcast per-gene subtraction, in place on our copy
    return dense


def blocked_segment_and_cn(
    adata,
    normal_mask,
    chr_labels,
    block_size=None,
    penalty_coef=DEFAULT_PENALTY_COEF,
    min_seg_genes=DEFAULT_MIN_SEG_GENES,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
):
    """Row-blocked M3. Returns ``(segments, cn_matrix, seg_baseline)``.

    Equivalent to::

        centered = center_against_baseline(adata, normal_mask)
        smoothed = smooth_along_chromosomes(centered, chr_labels, smooth_window)
        segments = detect_segments(smoothed, normal_mask, chr_labels, ...)
        cn_matrix = per_cell_segment_cn(smoothed, segments)
        seg_baseline = per_segment_normal_baseline(cn_matrix, normal_mask)

    but never materializes the full dense matrix. ``segments.tumor_mean`` is
    centered against ``seg_baseline`` exactly as ``detect_segments`` does.
    """
    X = adata.X
    n_cells, n_genes = X.shape
    nm = asarray(normal_mask, dtype=bool)
    chr_labels = asarray(chr_labels)
    if not block_size or block_size <= 0:
        block_size = auto_block_size(n_cells, n_genes)

    baseline_gene = per_gene_normal_median(X, nm)

    tumor_mask = ~nm
    pool_rows = tumor_mask if tumor_mask.sum() > 0 else ones(n_cells, dtype=bool)

    # ── pass 1: per-gene tumor-pool mean of the smoothed signal ───────────────
    pooled_sum = zeros(n_genes, dtype=float64)
    n_pool = 0
    for start in range(0, n_cells, block_size):
        end = min(start + block_size, n_cells)
        smoothed_block = smooth_along_chromosomes(
            _densify_center(X, start, end, baseline_gene), chr_labels, window=smooth_window,
        )
        pr = pool_rows[start:end]
        if pr.any():
            # Reduce each block in float64 so the pooled signal (and thus the PELT
            # boundaries) is independent of --block-size, not sensitive to float32
            # summation order across differently-sized blocks.
            pooled_sum += smoothed_block[pr].sum(axis=0, dtype=float64)
            n_pool += int(pr.sum())
        del smoothed_block
    pooled = pooled_sum / n_pool if n_pool else pooled_sum

    segments = segments_from_pooled(
        pooled, chr_labels, penalty_coef=penalty_coef, min_seg_genes=min_seg_genes,
    )
    n_seg = len(segments)

    # ── pass 2: per-cell × per-segment CN, block by block ─────────────────────
    cn_matrix = empty((n_cells, n_seg), dtype=float32)
    if n_seg:
        for start in range(0, n_cells, block_size):
            end = min(start + block_size, n_cells)
            smoothed_block = smooth_along_chromosomes(
                _densify_center(X, start, end, baseline_gene), chr_labels, window=smooth_window,
            )
            cn_matrix[start:end, :] = per_cell_segment_cn(smoothed_block, segments)
            del smoothed_block

    # ── center tumor_mean against the per-segment diploid reference ───────────
    seg_baseline = per_segment_normal_baseline(cn_matrix, nm)
    if n_seg:
        segments = segments.copy()
        segments["tumor_mean"] = segments["tumor_mean"].to_numpy() - seg_baseline

    return segments, cn_matrix, seg_baseline
