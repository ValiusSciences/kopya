"""Step §3.1 — normalize, filter, and project onto the genome.

Three public functions, applied in this order:
    1. filter_cells_and_genes() — drop low-quality cells and uninformative genes.
    2. normalize_log1p()        — CP10k library-size scaling + log1p.
    3. project_onto_genome()    — join genes against the gene-order table and
                                  sort by genomic position within chromosomes.

All three operate on (and return) AnnData. The matrix stays sparse CSR
throughout — log1p preserves sparsity since log1p(0)=0.
"""

import warnings
from copy import deepcopy

import numpy as np
from anndata import AnnData
from numpy import asarray, diff, float32, float64, log1p, repeat
from pandas import Categorical
from scipy.sparse import csr_matrix, issparse

from kopya.annotations import (
    CANONICAL_CHROM_ORDER,
    gene_filter_mask,
    load_cycle_genes,
    load_gene_order,
)


# Defaults match the spec (kopya.md §3.1). Exposed at module level so
# tests can reference them and the CLI can advertise them in --help.
DEFAULT_MIN_GENES = 200       # min genes per cell
DEFAULT_MIN_CELLS = 3         # min cells per gene (absolute floor)
DEFAULT_LOW_DR = 0.05         # min gene detection rate (fraction of cells)
DEFAULT_UP_DR = 1.0           # max gene detection rate (1.0 = no upper cutoff)
DEFAULT_TARGET_SUM = 1e4      # CP10k scaling target

# --- Raw-count validation tolerances (see _raw_count_reason) -----------------
# Absolute tolerance for treating a stored value as a whole number. h5ad files
# routinely store integer counts in a float dtype, so an exact == check is too
# strict; anything within this of its nearest integer counts as integral.
_INT_TOL = 1e-6
# Library-size normalization (CP10k, CPM, ...) forces every cell to (almost) the
# same total. Raw UMI counts never do — per-cell depth varies substantially. A
# coefficient of variation below this threshold is a necessary (not sufficient)
# fingerprint of a fixed-target rescaling that was then rounded back to integers.
_LIBSIZE_CV_THRESHOLD = 0.01
# Only trust the near-uniform-library-size signal when there are enough cells
# that a low CV cannot arise by chance in a small synthetic matrix.
_LIBSIZE_MIN_CELLS = 10
# A low CV alone is not proof of normalization — a homogeneous high-depth count
# simulation can also be near-uniform. We additionally require the typical total
# to sit near a canonical normalization target (CP10k or CPM), which raw depth
# has no reason to do. Both conditions together isolate rounded-normalized data.
_LIBSIZE_TARGETS = (1e4, 1e6)
_LIBSIZE_TARGET_RTOL = 0.02
# Values are scanned in bounded chunks so validation never allocates whole-buffer
# temporaries. A real cohort can hold hundreds of millions of nonzeros; a full
# rint/subtract/abs pass over a float buffer that size would add gigabytes of
# peak memory (and can OOM before the pipeline starts). One chunk of scratch is
# a few MB regardless of matrix size, and the scan short-circuits on first hit.
_SCAN_CHUNK = 1_000_000


def filter_cells_and_genes(
    adata,
    min_genes=DEFAULT_MIN_GENES,
    min_cells=DEFAULT_MIN_CELLS,
    low_dr=DEFAULT_LOW_DR,
    up_dr=DEFAULT_UP_DR,
):
    """Drop low-quality cells and uninformative genes.

    Two passes (semantics match scanpy filter_cells(min_genes) /
    filter_genes(min_cells), computed directly from the sparse structure):
        Cell pass: keep cells with >= min_genes detected (value > 0) genes.
        Gene pass: over the *surviving* cells, keep genes with >= min_cells
                   detected cells AND a detection rate in [low_dr, up_dr].

    The detection-rate window is what protects against ubiquitous housekeeping
    genes (UP_DR) and ultra-rare transcripts (LOW_DR) — both add noise to the
    smoothed CNV signal without contributing real CN information.

    Args:
        adata: AnnData with sparse CSR raw counts in .X.
        min_genes: Min genes/cell threshold (cells below are dropped).
        min_cells: Absolute min cells/gene floor (defensive against zero columns).
        low_dr: Minimum gene detection rate (fraction of post-filter cells).
        up_dr: Maximum gene detection rate.

    Returns:
        A new, independent filtered AnnData (a single boolean subset of the
        input; the caller's adata is never mutated). The QC counts scanpy's
        filters populate — obs['n_genes'] and var['n_cells'] — are attached.
    """
    # Memory-bounded filtering: compute the cell/gene keep masks straight from the
    # sparse structure, then materialize the kept sub-matrix in a SINGLE subset.
    # The previous code did `out = adata.copy()` (a full copy of the raw matrix,
    # tens of GB on a ~1M-cell cohort) before shrinking it in place; here nothing
    # larger than the final filtered matrix + a boolean (X > 0) mask is allocated.
    # Semantics are identical to scanpy filter_cells(min_genes) /
    # filter_genes(min_cells) + the [low_dr, up_dr] detection window.
    #
    # Count DETECTED (value > 0) cells/genes, not stored entries: a legal sparse
    # input can carry explicit zeros, duplicate coordinates, or negatives, and
    # counting storage would let zero-only genes pass low_dr or push a detection
    # rate above 1. `X > 0` is a boolean sparse (~nnz), matching the prior
    # `(out.X > 0).sum(...)` exactly.
    positive = adata.X.tocsr() > 0

    # Cell pass: keep cells with >= min_genes detected genes.
    genes_per_cell = asarray(positive.sum(axis=1)).ravel()
    cell_keep = genes_per_cell >= min_genes

    # Gene stats over the SURVIVING cells (the right denominator).
    n_cells = int(cell_keep.sum())
    cells_per_gene = asarray(positive[cell_keep].sum(axis=0)).ravel()
    detection_rate = cells_per_gene / max(n_cells, 1)

    # min_cells floor (defensive against all-zero genes) AND the inclusive
    # [low_dr, up_dr] window (mirrors CopyKAT's LOW.DR / UP.DR semantics).
    gene_keep = (
        (cells_per_gene >= min_cells)
        & (detection_rate >= low_dr)
        & (detection_rate <= up_dr)
    )

    # Single subset → the filtered matrix directly (copy() materializes the view).
    out = adata[cell_keep, gene_keep].copy()
    # Re-attach the QC counts scanpy's filter_cells/filter_genes used to populate,
    # so callers of this public helper still see them (n_genes over all genes at
    # cell-filter time; n_cells over the surviving cells).
    out.obs["n_genes"] = genes_per_cell[cell_keep]
    out.var["n_cells"] = cells_per_gene[gene_keep]
    return out


def _stored_values(X):
    """Return the stored matrix entries as a flat 1-D array in their native dtype.

    For sparse matrices this is the data buffer (explicit nonzeros only); for
    dense arrays it is the raveled matrix. Callers reason about the *stored*
    values because a CSR/CSC matrix never materializes its implicit zeros.

    The native dtype is preserved deliberately: forcing a float64 view would copy
    the entire data buffer, and on a real cohort (hundreds of millions of
    nonzeros stored as int32) that copy is gigabytes. The pipeline keeps these
    matrices sparse precisely to bound memory, so validation must not undo that.

    Args:
        X: A scipy sparse matrix or a dense (numpy) array/matrix.

    Returns:
        A 1-D numpy array of the stored entries (empty if X has none).
    """
    values = X.data if issparse(X) else np.asarray(X)
    return np.ravel(values)


def _is_already_normalized(adata):
    """Heuristic check for pre-normalized data (e.g. SMART-seq2 FPKM/log2 TPM).

    Returns True if the matrix contains negative values, which cannot occur in
    raw count data. Pre-normalized matrices (log2-TPM, log2-RPKM, z-scored) are
    common in legacy datasets and should not be run through library-size
    normalization + log1p again.

    This is deliberately narrow: it guards normalize_log1p against log1p(x)
    producing NaN when x < -1. The broader raw-vs-normalized classification
    used to warn the user at load time lives in _raw_count_reason /
    validate_raw_counts.

    Args:
        adata: AnnData to inspect.

    Returns:
        bool: True if the data appears to be pre-normalized.
    """
    values = _stored_values(adata.X)
    return bool((values < 0).any())


def _scan_value_reason(values):
    """Scan stored values in bounded chunks for negative/non-finite/fractional.

    Processes the buffer _SCAN_CHUNK elements at a time so peak scratch memory
    stays a few MB regardless of matrix size, and returns on the first violation
    (already-normalized float data is caught in its first chunk). The rounding
    and finiteness tests are skipped for integer dtypes, which are exactly
    integral and always finite.

    Args:
        values: 1-D numpy array of the stored matrix entries (native dtype).

    Returns:
        None if every chunk is consistent with raw counts, else a reason string.
    """
    is_integer_dtype = np.issubdtype(values.dtype, np.integer)
    for start in range(0, values.size, _SCAN_CHUNK):
        chunk = values[start:start + _SCAN_CHUNK]
        if (chunk < 0).any():
            return "matrix contains negative values"
        if not is_integer_dtype:
            if not np.isfinite(chunk).all():
                return "matrix contains non-finite values (NaN or Inf)"
            # Compare each value against its nearest integer. h5ad files often
            # store integer counts as float, so tolerate tiny rounding error.
            if np.abs(chunk - np.rint(chunk)).max() > _INT_TOL:
                return "matrix contains non-integer (fractional) values"
    return None


def _raw_count_reason(X):
    """Classify a matrix as raw UMI counts or explain why it is not.

    Raw single-cell UMI counts are non-negative integers whose per-cell library
    sizes vary widely. This inspects the stored values and returns a short human
    reason string when the matrix violates one of those properties, or None when
    it is consistent with raw counts.

    The checks, in order:
        1. Negative values      -> log2-TPM / z-scored / other centered data.
        2. Non-finite values    -> NaN / Inf cannot occur in raw counts.
        3. Fractional values    -> log1p, CP10k, TPM, or any float transform.
        4. Near-uniform library -> integer values whose per-cell totals barely
                                    vary *and* cluster at a canonical target
                                    (CP10k / CPM), the signature of a fixed-target
                                    rescaling rounded back to int.

    Checks 2-3 are skipped for integer dtypes (exactly integral, always finite)
    and, for float dtypes, run over the buffer in bounded chunks so validation
    never allocates whole-buffer temporaries — see _scan_value_reason. Check 4
    fires only with enough cells (>= _LIBSIZE_MIN_CELLS) and requires both a very
    low coefficient of variation and a canonical-target median, so genuine
    (high-variance, arbitrary-depth) raw counts are never flagged. Real raw
    counts return None.

    Args:
        X: A scipy sparse matrix or dense array of the candidate counts.

    Returns:
        None if X looks like raw counts, else a short reason string.
    """
    values = _stored_values(X)
    if values.size == 0:
        # An all-zero (or empty) matrix carries no evidence either way; treat it
        # as raw so a degenerate input does not trip a spurious warning.
        return None

    value_reason = _scan_value_reason(values)
    if value_reason is not None:
        return value_reason

    # Non-negative integer-valued data remains. Library-size normalization to a
    # canonical target forces near-identical per-cell totals; rounding that back
    # to integers lands here. A low CV alone is not proof (a uniform high-depth
    # simulation is also near-uniform), so we additionally require the median
    # total to sit near CP10k / CPM — something raw depth has no reason to do.
    #
    # Uniformity is measured over cells with a positive library only: a single
    # all-zero cell (a normalized matrix can still carry one) would otherwise
    # inflate the CV and hide the normalization. The minimum-cell requirement
    # therefore applies to the count of positive-library cells.
    totals = asarray(X.sum(axis=1)).ravel().astype(float64)
    positive = totals[totals > 0]
    if positive.size >= _LIBSIZE_MIN_CELLS:
        mean_total = positive.mean()
        if mean_total > 0:
            cv = positive.std() / mean_total
            median_total = float(np.median(positive))
            near_target = any(
                abs(median_total - target) <= _LIBSIZE_TARGET_RTOL * target
                for target in _LIBSIZE_TARGETS
            )
            if cv < _LIBSIZE_CV_THRESHOLD and near_target:
                return (
                    "per-cell library sizes are near-uniform "
                    f"(coefficient of variation {cv:.2e}) and cluster at "
                    f"~{median_total:.0f}, which indicates library-size-"
                    "normalized data rather than raw counts"
                )

    return None


def _looks_like_raw_counts(X):
    """Return True when X is consistent with raw integer UMI counts.

    Thin boolean wrapper over _raw_count_reason for call sites that only need
    the yes/no answer and not the explanation.

    Args:
        X: A scipy sparse matrix or dense array.

    Returns:
        bool: True if the matrix looks like raw counts.
    """
    return _raw_count_reason(X) is None


def validate_raw_counts(adata, action="warn"):
    """Check that adata carries raw integer UMI counts and flag it if not.

    kopya applies its own CP10k + log1p normalization and therefore
    expects raw counts as input. Passing already-normalized or log-transformed
    values silently produces wrong CNV calls, so this guards the load boundary.

    When layers["counts"] is present it is treated as the authoritative
    raw-count source (that is what the loader uses), so .X is not inspected and
    the call is a no-op. Otherwise .X is classified via _raw_count_reason and,
    if it does not look like raw counts, this warns (default), raises, or stays
    silent depending on ``action``.

    Args:
        adata: AnnData to validate.
        action: One of "warn" (emit a UserWarning; the default), "raise"
            (raise ValueError — opt-in strict mode), or "ignore" (skip the
            check entirely).

    Returns:
        bool: True if the data looks like raw counts (or a counts layer is
            present, or the check was ignored); False if a problem was detected
            and only warned about.

    Raises:
        ValueError: if ``action`` is unrecognized, or if a non-raw matrix is
            detected while ``action="raise"``.
    """
    if action not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"action must be 'warn', 'raise', or 'ignore', got {action!r}."
        )
    if action == "ignore":
        return True

    # A raw-count layer is the trusted source; the loader reads counts from it
    # regardless of what .X holds, so there is nothing to validate here.
    if "counts" in getattr(adata, "layers", {}):
        return True

    reason = _raw_count_reason(adata.X)
    if reason is None:
        return True

    message = (
        f"Input .X does not look like raw integer UMI counts ({reason}). "
        "kopya expects raw counts and applies its own CP10k + log1p "
        "normalization; passing pre-normalized or log-transformed data "
        "(e.g. log1p(CP10k), CPM, TPM) will produce incorrect CNV results. "
        "Provide raw counts in .X, store them in layers['counts'], or disable "
        "this check (--raw-count-check ignore on the CLI, or action='ignore' "
        "when calling validate_raw_counts directly)."
    )
    if action == "raise":
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    return False


def _anndata_with_new_x(adata, new_x):
    """New AnnData with .X replaced by ``new_x``, preserving everything else.

    Deliberately NOT ``adata.copy()``: that would duplicate the old, about-to-be-
    discarded .X (a full-matrix copy — the exact cost the memory-lean normalize
    path exists to avoid). We instead reconstruct with the new .X and copy the
    other fields so the result is still independent and preserves obs/var/uns and
    any layers/obsm/varm/obsp/varp/raw. In this pipeline those mappings are empty
    so the comprehensions are free; external callers that carry them keep them,
    exactly as the previous adata.copy() did.
    """
    out = AnnData(
        X=new_x,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=deepcopy(dict(adata.uns)),
        obsm={k: v.copy() for k, v in adata.obsm.items()},
        varm={k: v.copy() for k, v in adata.varm.items()},
        layers={k: v.copy() for k, v in adata.layers.items()},
        obsp={k: v.copy() for k, v in adata.obsp.items()},
        varp={k: v.copy() for k, v in adata.varp.items()},
    )
    if adata.raw is not None:
        out.raw = adata.raw.to_adata()
    return out


def normalize_log1p(adata, target_sum=DEFAULT_TARGET_SUM):
    """Library-size normalize then log1p, returning a new independent AnnData.

    Sparse-preserving: scales nonzero entries by per-cell factors and applies
    log1p. log1p(0) = 0 so the CSR structure is unchanged; we avoid scanpy's
    normalize_total because it would densify on some paths.

    If the input matrix contains negative values (indicating pre-normalized
    data such as SMART-seq2 log2-TPM or FPKM), the normalization step is
    skipped and the data is returned as-is after recording the detection in
    uns. This preserves the centering information already present in such
    data and prevents NaN from log1p of values < -1.

    Args:
        adata: AnnData with sparse CSR raw counts in .X, or pre-normalized
            continuous values.
        target_sum: Per-cell total after library-size scaling (default 1e4).
            Ignored when pre-normalized data is detected.

    Returns:
        A new AnnData with .X = log1p(CP10k counts) as **float32** sparse CSR
        (raw-count input), or the original data (pre-normalized input). obs/var/
        uns and any layers/obsm/varm/obsp/varp/raw are preserved, and the result
        is fully independent of the input.
    """
    # Detect pre-normalized data early (read-only on the caller's matrix);
    # skipping normalization on such data prevents log1p(x) NaN when x < -1,
    # which occurs with log2-TPM values typical of SMART-seq2 pipelines.
    if _is_already_normalized(adata):
        out = adata.copy()
        out.uns["kopya_normalized"] = False
        out.uns["kopya_pre_normalized"] = True
        out.uns["kopya_target_sum"] = None
        return out

    # --- PERFORMANCE-CRITICAL: readability traded for bounded memory ---
    # The obvious implementation, `out = adata.copy(); out.X = log1p(...)`, is a
    # memory trap on large cohorts: it copies the whole matrix only to discard
    # .X, and the intermediate `data * scale` / `log1p` steps each allocate a
    # full-nnz float64 buffer. At >1e9 nonzeros that is tens of GB. Instead we
    # build exactly one fresh data buffer and mutate it in place, so at most one
    # extra full-nnz temporary (the repeated scale vector) is ever live.
    csr = adata.X.tocsr()  # no copy when already CSR

    # Per-cell totals; guard zero-count rows so the divide is a no-op for them.
    totals_safe = asarray(csr.sum(axis=1)).ravel().astype(float64)
    totals_safe[totals_safe == 0] = 1.0
    scale = (target_sum / totals_safe).astype(float32)  # per-cell, n_cells long
    row_lengths = diff(csr.indptr)

    # Fresh float32 buffer (astype copies, so the caller's raw counts are never
    # touched); scale per nonzero then log1p, both IN PLACE on that one buffer.
    # float32 is ample — downstream (center_against_baseline) casts to float32
    # anyway — and halves every per-nonzero temporary vs the old float64 path.
    data = csr.data.astype(float32)
    data *= repeat(scale, row_lengths)
    log1p(data, out=data)

    # Copy the structural arrays (indices/indptr) so the result is fully
    # independent: a caller that keeps `adata` and later structurally mutates the
    # returned matrix (sort_indices / eliminate_zeros / sum_duplicates) must not
    # reach back and corrupt the input's structure. (This copy is still far
    # cheaper than the old full-matrix + float64-triple-buffer path.)
    new_X = csr_matrix((data, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)
    out = _anndata_with_new_x(adata, new_X)
    out.uns["kopya_normalized"] = True
    out.uns["kopya_target_sum"] = float(target_sum)
    return out


def project_onto_genome(adata, gene_order=None):
    """Join genes against the gene-order table and sort by genomic position.

    After this step, adata.var carries (chr, start, end) columns and the genes
    are in canonical genomic order (chr1, chr2, ..., chr22; within each
    chromosome by start coordinate). Genes that fail the gene_filter_mask
    (unmappable, chrX, chrY, MT-, HLA-, cycle) are dropped.

    Args:
        adata: AnnData post-normalization. var index = gene symbols.
        gene_order: Optional override DataFrame from load_gene_order(); default
            loads the bundled GENCODE v49 hg38 table.

    Returns:
        A new AnnData with genes reduced to the mappable, kept set and sorted
        by genomic position. var has new columns: chr (ordered Categorical),
        start (int), end (int).
    """
    # Deduplicate var_names before any positional look-up. Some h5ad files
    # (e.g. multi-library merges) carry duplicate gene symbols; var_names.get_loc
    # returns a slice for duplicates, which breaks the int-array positional index
    # built below. Making names unique first collapses them to "GENE-1", "GENE-2"
    # etc.; the duplicates are then removed by gene_filter_mask anyway because
    # the suffixed names won't match the gene_order table.
    if not adata.var_names.is_unique:
        adata = adata.copy()
        adata.var_names_make_unique()

    # Load the bundled annotation if the caller did not pre-pass one.
    if gene_order is None:
        gene_order = load_gene_order()

    # Load the cycle list once; gene_filter_mask would otherwise reload it
    # per call (cheap, but explicit reuse keeps the loader path obvious).
    cycle_genes = load_cycle_genes()

    # Compute the keep mask against the current var order. This is faster than
    # reindexing first because we skip the gene_order DataFrame lookup for
    # genes that fail the prefix / cycle / drop_chroms filters anyway.
    keep_mask = gene_filter_mask(
        var_names=adata.var_names,
        gene_order=gene_order,
        cycle_genes=cycle_genes,
    )

    # Subset to kept genes; defensive copy because boolean indexing on AnnData
    # returns a view by default and we are about to mutate .var.
    out = adata[:, keep_mask].copy()

    # Attach genomic coordinates from the gene_order table. .reindex() against
    # the subset's var_names is the explicit way to align positions — every
    # gene here is guaranteed present in gene_order by the mask above.
    coords = gene_order.reindex(out.var_names)
    out.var["chr"] = coords["chr"].values
    out.var["start"] = coords["start"].astype(int).values
    out.var["end"] = coords["end"].astype(int).values

    # Re-coerce chr to the canonical ordered Categorical so the sort below is
    # in genomic order (chr1 < chr2 < ... < chr22), not lexicographic.
    out.var["chr"] = Categorical(
        out.var["chr"],
        categories=CANONICAL_CHROM_ORDER,
        ordered=True,
    )

    # Sort genes by (chr, start). AnnData's __getitem__ accepts a list of var
    # names; build that order from the sorted var frame.
    var_sorted = out.var.sort_values(["chr", "start"])
    sorted_order = list(var_sorted.index)
    # Defensive in-place copy via boolean reindex against the sorted index.
    # Using positional indexing (.iloc-style int array) is faster than passing
    # a list of strings for large gene counts.
    pos = asarray(
        [out.var_names.get_loc(name) for name in sorted_order],
        dtype="int64",
    )
    out = out[:, pos].copy()

    return out


def filter_normalize_project(adata, gene_order=None, **filter_kwargs):
    """Run the M1 trio (filter → normalize → project) as a single call.

    Convenience wrapper used by the CLI; tests cover the three steps individually.

    Args:
        adata: AnnData with sparse CSR raw counts in .X.
        gene_order: Optional pre-loaded gene-order DataFrame; default = bundled.
        **filter_kwargs: Forwarded to filter_cells_and_genes (min_genes, etc.).

    Returns:
        The fully processed AnnData ready for §3.2 (baseline picking).
    """
    # The three steps are sequential, never independent — running them as a
    # single call also produces a well-defined timing budget for the QC JSON.
    filtered = filter_cells_and_genes(adata, **filter_kwargs)
    normalized = normalize_log1p(filtered)
    projected = project_onto_genome(normalized, gene_order=gene_order)
    return projected
