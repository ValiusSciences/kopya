"""In-memory pipeline orchestration — the CLI-free core behind the Python API.

``run_pipeline(adata)`` runs the M1->M4 pipeline in memory and returns structured
results **without writing any files**. It is the single orchestration shared by:

- ``kopya.tl.cnv`` — the scanpy-style API, which writes the results back
  into the AnnData (``obs``/``obsm``/``uns``);
- the external gold-standard test harness (``tests/external/ground_truth``),
  which wraps this and writes the CSV/.seg outputs.

so the same code path the API exposes is the one the gold-standard suite
validates. The ``kopya run`` CLI keeps its own memory-lean, block-
streaming path for very large cohorts (issue #24); ``run_pipeline`` is the
straightforward in-memory equivalent (numerically equivalent on the datasets we
validate).
"""

from dataclasses import dataclass

from numpy import asarray as _asarray


def detected_gene_counts(X):
    """Per-cell count of detected (numerically nonzero) genes.

    Feeds the low-complexity gate. Counts values that are actually nonzero — not
    stored entries — so a matrix that carries explicitly-stored zeros (some 10x /
    Flex ``.h5`` and mtx inputs do) does not inflate the count and let ambient
    cells escape the gate. For sparse input, drop explicit zeros then use the fast
    nnz count; for a dense (possibly signed pre-normalized) matrix, count ``!= 0``.

    Args:
        X: post-M1 expression matrix (scipy sparse or numpy ndarray).

    Returns:
        1-D float array, length n_cells.
    """
    if hasattr(X, "getnnz"):
        X = X.copy()
        X.eliminate_zeros()  # so getnnz reflects numerical, not stored, nonzeros
        return _asarray(X.getnnz(axis=1), dtype=float).ravel()
    return _asarray((_asarray(X) != 0).sum(axis=1), dtype=float).ravel()


@dataclass
class PipelineResult:
    """Structured output of run_pipeline (in-memory, no files written).

    Attributes:
        prediction_df: per-cell classify_cells() DataFrame, indexed by the
            post-M1 (filtered) barcodes; columns include class, tumor_score,
            subclone, low_complexity.
        chr_matrix: (n_filtered_cells x n_chroms) 1.0-centered per-chromosome
            CNV matrix (>1 gain, <1 loss).
        chroms_present: list[str] of chromosome labels for chr_matrix columns.
        segments: detect_segments() table (+ start_bp/end_bp/segment_id).
        cn_matrix: (n_filtered_cells x n_segments) raw per-segment CN.
        seg_baseline: (n_segments,) per-segment normal baseline.
        adata_m1: the filtered/normalized AnnData (its obs_names give the
            barcode order of all the per-cell arrays above — used to align
            results back to the caller's original cells).
        baseline: the pick_baseline() result dict (mask/method/n_normal/diag).
        qc: summary counts dict.
    """

    prediction_df: object
    chr_matrix: object
    chroms_present: list
    segments: object
    cn_matrix: object
    seg_baseline: object
    adata_m1: object
    baseline: dict
    qc: dict


def run_pipeline(
    adata,
    *,
    norm_cell_path=None,
    norm_cell_names=None,
    complexity_gate=False,
    filter_kwargs=None,
):
    """Run M1->M4 in memory and return a PipelineResult (no files written).

    Args:
        adata: raw-counts AnnData (cells x genes). Not mutated — M1 subsets to a
            fresh copy internally.
        norm_cell_path: optional path to a known-normal barcode file (supervised).
        norm_cell_names: optional iterable of known-normal barcodes (supervised,
            in-memory). Pass at most one of path/names; if neither, the
            unsupervised baseline cascade runs.
        complexity_gate: when True, feed per-cell detected-gene counts to the
            low-complexity gate (mirrors the CLI run).
        filter_kwargs: optional dict forwarded to filter_normalize_project
            (e.g. {"min_genes": 100}).

    Returns:
        PipelineResult.
    """
    from kopya.annotations import CANONICAL_CHROM_ORDER, load_gene_order
    from kopya.baseline import pick_baseline
    from kopya.classify import classify_cells
    from kopya.normalize import filter_normalize_project
    from kopya.outputs import compute_chr_cnv_matrix, segments_with_coordinates
    from kopya.segment import (
        detect_segments,
        per_cell_segment_cn,
        per_segment_normal_baseline,
    )
    from kopya.smooth import center_against_baseline, smooth_along_chromosomes

    import anndata as ad
    import numpy as np
    from scipy.sparse import csr_matrix, issparse

    kwargs = filter_kwargs or {}

    # ── M1: filter, normalize, project ───────────────────────────────────────
    # Dense .X is a valid AnnData input, but downstream M1 calls .X.tocsr(); coerce
    # to CSR at this boundary (as the file loaders do) so dense arrays work too.
    # Build a thin wrapper from the CSR (no dense copy) rather than adata.copy().
    n_cells_raw = adata.n_obs
    if not issparse(adata.X):
        adata = ad.AnnData(X=csr_matrix(adata.X), obs=adata.obs, var=adata.var)
    # filter_normalize_project subsets to its own copy, so `adata` is untouched.
    gene_order = load_gene_order()
    adata_m1 = filter_normalize_project(adata, gene_order=gene_order, **kwargs)

    # ── M2: baseline selection ────────────────────────────────────────────────
    baseline = pick_baseline(
        adata_m1, norm_cell_path=norm_cell_path, norm_cell_names=norm_cell_names)

    # ── M3: center -> smooth -> segment -> per-cell CN ───────────────────────
    chr_labels = adata_m1.var["chr"].to_numpy()
    centered = center_against_baseline(adata_m1, baseline["mask"])
    smoothed = smooth_along_chromosomes(centered, chr_labels=chr_labels)
    segments = detect_segments(smoothed, normal_mask=baseline["mask"], chr_labels=chr_labels)
    segments = segments_with_coordinates(segments, adata_m1.var)
    cn_matrix = per_cell_segment_cn(smoothed, segments)
    seg_baseline = per_segment_normal_baseline(cn_matrix, baseline["mask"])
    chr_pedestal = float(np.median(seg_baseline))

    # ── M4: classify + per-chromosome matrix ─────────────────────────────────
    complexity = detected_gene_counts(adata_m1.X) if complexity_gate else None
    prediction_df = classify_cells(
        cn_matrix=cn_matrix,
        segments=segments,
        normal_mask=baseline["mask"],
        barcodes=adata_m1.obs_names.to_numpy(),
        complexity=complexity,
    )
    chr_matrix, chroms_present = compute_chr_cnv_matrix(
        cn_matrix, segments, CANONICAL_CHROM_ORDER, baseline=chr_pedestal
    )

    class_counts = prediction_df["class"].value_counts().to_dict()
    qc = {
        "n_cells_raw": n_cells_raw,
        "n_cells_filtered": adata_m1.n_obs,
        "n_genes_filtered": adata_m1.n_vars,
        "baseline_method": baseline["method"],
        "n_tumor": class_counts.get("tumor", 0),
        "n_normal": class_counts.get("normal", 0),
        "n_uncertain": class_counts.get("uncertain", 0),
        "n_segments": len(segments),
    }

    return PipelineResult(
        prediction_df=prediction_df,
        chr_matrix=chr_matrix,
        chroms_present=chroms_present,
        segments=segments,
        cn_matrix=cn_matrix,
        seg_baseline=seg_baseline,
        adata_m1=adata_m1,
        baseline=baseline,
        qc=qc,
    )
