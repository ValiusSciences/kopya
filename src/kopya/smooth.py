"""Step §3.3 — center against baseline and smooth along chromosomes.

Two public functions:
    center_against_baseline(): subtract per-gene median over the normal pool
                               from every cell's expression vector.
    smooth_along_chromosomes(): per-chromosome moving-average smoothing of the
                                centered matrix; produces the continuous CNV
                                signal track that segmentation consumes.

Memory: the centered + smoothed matrix is dense float32 of shape
(n_cells × n_genes). At 10k cells × 20k genes that's ~800 MB. For very large
cohorts the CLI's blocked path (``blocked.py``) streams along the cell axis
to bound peak memory; this in-memory path keeps it simple.
"""

from numpy import asarray, empty_like, float32, median
from scipy.ndimage import uniform_filter1d


# Default smoothing window in genes. Matches inferCNV's default; CopyKAT uses
# 25 but on a much coarser bin grid. 100 genes gives ~3-5 Mb effective
# resolution on the gene-ordered chromosome track.
DEFAULT_SMOOTH_WINDOW = 100


def center_against_baseline(adata, normal_mask):
    """Center every cell's expression against the per-gene median over normals.

    For each gene, computes the median of its expression across cells in the
    normal pool and subtracts that median from every cell's value for that
    gene. The result is a dense float32 matrix where ~0 = diploid baseline,
    >0 = above-baseline (gain), <0 = below-baseline (loss).

    Densifies the input — after subtraction the matrix is no longer sparse
    (zero entries become -median wherever any normal cell expressed the gene),
    so storing sparse would actually cost more memory.

    Args:
        adata: AnnData post-M1 with log1p(CP10k) values in .X (sparse CSR).
        normal_mask: bool array len == n_obs, True for cells in the normal pool.

    Returns:
        A 2-D numpy float32 array, shape (n_cells, n_genes), centered against
        the per-gene normal-pool median. Column order matches adata.var_names.
    """
    # Densify once; CSR row slicing into a dense array is the cheapest path
    # given that we have to densify for smoothing anyway.
    X_dense = adata.X.toarray().astype(float32, copy=False)

    # Per-gene median of the normal pool. axis=0 reduces across cells so the
    # result has shape (n_genes,). Median is robust to outlier normal cells.
    normal_block = X_dense[normal_mask, :]
    if normal_block.shape[0] == 0:
        # Defensive: no normal cells supplied. Use the global per-gene median
        # so centering still happens and the user sees a degraded-but-coherent
        # baseline rather than a NaN explosion.
        baseline = median(X_dense, axis=0)
    else:
        baseline = median(normal_block, axis=0)

    # Subtract the per-gene baseline from every row. Broadcasting takes care
    # of the (n_cells, n_genes) - (n_genes,) shape; no allocation surprises.
    centered = X_dense - baseline
    return centered


def smooth_along_chromosomes(matrix, chr_labels, window=DEFAULT_SMOOTH_WINDOW):
    """Apply a moving-average filter per chromosome along the gene axis.

    Within each chromosome, runs `scipy.ndimage.uniform_filter1d` along
    axis=1 of the corresponding gene slice. `mode='nearest'` is used at
    chromosome edges so signal does not leak across chromosome boundaries.

    The window is clamped per chromosome to min(window, n_genes_in_chr) so
    tiny chromosomes (e.g. chr21) still produce a meaningful smoothed track
    rather than an entirely-uniform output.

    Args:
        matrix: 2-D ndarray (cells × genes), already centered. Gene order must
            be consistent with chr_labels.
        chr_labels: 1-D array-like (length n_genes) of chromosome strings,
            ordered to match the matrix columns. Genes are assumed pre-sorted
            within each chromosome (project_onto_genome guarantees this).
        window: Moving-average window size in genes (default 100).

    Returns:
        A dense ndarray of the same shape and dtype as `matrix` with the
        per-chromosome smoothed signal.
    """
    # Allocate output up front; we will fill chromosome-by-chromosome so the
    # column order of the output exactly matches the input.
    smoothed = empty_like(matrix)

    # Convert to numpy for cheap equality slicing; pandas Categoricals are fine
    # but their per-element comparison is slower than ndarray ==.
    labels = asarray(chr_labels)

    # Iterate over distinct chromosome labels in their order of appearance.
    # The matrix layout guarantees each chromosome is a contiguous block, so
    # iterating in label-discovery order is exactly genomic order.
    seen = []
    for lbl in labels:
        if seen and seen[-1] == lbl:
            continue
        seen.append(lbl)

    for chrom in seen:
        # Boolean mask along the gene axis for this chromosome.
        col_mask = labels == chrom
        block = matrix[:, col_mask]
        # Adaptive clamp: cap window to one-third of the chromosome gene count
        # so small chromosomes (e.g. chr21 with ~30 genes) are not entirely
        # averaged away, while large chromosomes (chr1, chr2) get the full
        # window. The // 3 + 1 gives a meaningful window even for tiny blocks.
        # Minimum 10 prevents degenerate single-gene windows; maximum is the
        # full requested window (for large chromosomes).
        block_window = max(10, min(window, block.shape[1] // 3 + 1))
        if block_window < 1:
            # Empty chromosome would be a bug, but defensively no-op rather
            # than crash on uniform_filter1d's validation.
            smoothed[:, col_mask] = block
            continue
        smoothed[:, col_mask] = uniform_filter1d(
            block,
            size=block_window,
            axis=1,
            mode="nearest",
        )

    return smoothed
