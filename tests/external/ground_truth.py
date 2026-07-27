"""Ground-truth evaluation helpers for external benchmark tests.

This module provides:
    compute_chr_concordance()    — compare sc pseudobulk vs known per-chr CNV.
    compute_tumor_normal_recall() — precision/recall for tumor/normal calls.
    run_full_pipeline()          — end-to-end M1→M4 pipeline via Python API.

All functions are designed to be called from pytest test functions and raise
plain AssertionErrors / return numeric results that tests can assert on.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Evaluation functions
# =============================================================================


def compute_chr_concordance(
    prediction_df: pd.DataFrame,
    chr_matrix: np.ndarray,
    chroms_present: list,
    known_cnv_dict: dict,
    n_normal: int,
) -> dict:
    """Compare sc tumor pseudobulk vs known per-chromosome CNV direction.

    Builds a pseudobulk CNV profile for tumor cells by taking the column-wise
    mean of chr_matrix restricted to tumor-called cells. Compares the sign of
    each altered chromosome's mean deviation against the known CNV direction.

    Args:
        prediction_df: DataFrame from classify_cells(), indexed by barcode,
            with a 'class' column containing 'tumor'/'normal'/'uncertain'.
        chr_matrix: (n_cells × n_chroms) 1.0-centered ndarray from
            compute_chr_cnv_matrix(). Values >1 indicate gain, <1 indicate loss.
        chroms_present: list[str] of chromosome labels corresponding to
            columns of chr_matrix (in order).
        known_cnv_dict: dict mapping chromosome label (e.g. "chr7") to
            +1 (gain), -1 (loss), or 0 (neutral/ignore).
        n_normal: int, number of cells in the normal reference pool. Used
            to derive a per-chromosome normal baseline when the prediction_df
            tumor/normal labels are not available for subsetting.

    Returns:
        dict with keys:
            direction_recall: float in [0, 1] — fraction of altered chromosomes
                where the sc tumor pseudobulk agrees with the known direction.
            n_altered_in_truth: int — number of chromosomes in known_cnv_dict
                with non-zero value (the denominator for direction_recall).
            per_chr_agreement: dict[str, bool] — per-chromosome agreement flag
                for the altered chromosomes only.
    """
    # Identify tumor-called cells. Fall back to positional slice if the
    # prediction_df does not cover the same barcodes as chr_matrix rows.
    tumor_mask = (prediction_df["class"] == "tumor").to_numpy()
    normal_mask = (prediction_df["class"] == "normal").to_numpy()

    # If no cells are called tumor, fall back to the positional split.
    if tumor_mask.sum() == 0:
        n_total = chr_matrix.shape[0]
        normal_mask_pos = np.zeros(n_total, dtype=bool)
        normal_mask_pos[:n_normal] = True
        tumor_mask = ~normal_mask_pos
        normal_mask = normal_mask_pos

    # Compute per-chromosome means for tumor and normal cells.
    tumor_chr_means = chr_matrix[tumor_mask, :].mean(axis=0)  # shape: (n_chroms,)
    if normal_mask.sum() > 0:
        normal_chr_means = chr_matrix[normal_mask, :].mean(axis=0)
    else:
        # No explicit normals — use the first n_normal cells positionally.
        normal_chr_means = chr_matrix[:n_normal, :].mean(axis=0)

    # Build direction agreement per altered chromosome.
    altered_chroms = {c: d for c, d in known_cnv_dict.items() if d != 0}
    n_altered = len(altered_chroms)

    per_chr_agreement = {}
    for chrom, known_direction in altered_chroms.items():
        if chrom not in chroms_present:
            # Chromosome not represented in this dataset — treat as no-call.
            per_chr_agreement[chrom] = False
            continue
        col = chroms_present.index(chrom)
        # Deviation of tumor from normal in 1.0-centered space.
        # Gain → tumor_mean > normal_mean → positive deviation.
        # Loss → tumor_mean < normal_mean → negative deviation.
        deviation = tumor_chr_means[col] - normal_chr_means[col]
        called_direction = int(np.sign(deviation))
        per_chr_agreement[chrom] = (called_direction == known_direction)

    direction_recall = (
        sum(per_chr_agreement.values()) / n_altered if n_altered > 0 else 0.0
    )

    return {
        "direction_recall": direction_recall,
        "n_altered_in_truth": n_altered,
        "per_chr_agreement": per_chr_agreement,
    }


def compute_tumor_normal_recall(
    prediction_df: pd.DataFrame,
    normal_barcodes: Optional[list] = None,
    tumor_barcodes: Optional[list] = None,
) -> dict:
    """Precision/recall for tumor/normal classification against ground-truth labels.

    At least one of normal_barcodes or tumor_barcodes must be provided. Cells
    whose barcode is not in either ground-truth set are excluded from both
    precision and recall calculations.

    Args:
        prediction_df: DataFrame from classify_cells(), indexed by barcode,
            with a 'class' column.
        normal_barcodes: Optional iterable of barcodes known to be normal.
        tumor_barcodes: Optional iterable of barcodes known to be tumor.

    Returns:
        dict with keys:
            tumor_precision: TP_tumor / (TP_tumor + FP_tumor), or None if
                no tumor ground truth is provided.
            tumor_recall: TP_tumor / (TP_tumor + FN_tumor), or None.
            normal_precision: TP_normal / (TP_normal + FP_normal), or None.
            normal_recall: TP_normal / (TP_normal + FN_normal), or None.
            n_tumor_truth: int, number of ground-truth tumor barcodes found
                in prediction_df.
            n_normal_truth: int, number of ground-truth normal barcodes found
                in prediction_df.
    """
    pred_index = set(prediction_df.index)
    result = {
        "tumor_precision": None,
        "tumor_recall": None,
        "normal_precision": None,
        "normal_recall": None,
        "n_tumor_truth": 0,
        "n_normal_truth": 0,
    }

    if tumor_barcodes is not None:
        tb = [b for b in tumor_barcodes if b in pred_index]
        result["n_tumor_truth"] = len(tb)
        if tb:
            subset = prediction_df.loc[tb, "class"]
            tp = (subset == "tumor").sum()
            fn = (subset != "tumor").sum()
            result["tumor_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if normal_barcodes is not None:
        nb = [b for b in normal_barcodes if b in pred_index]
        result["n_normal_truth"] = len(nb)
        if nb:
            subset = prediction_df.loc[nb, "class"]
            tp = (subset == "normal").sum()
            fn = (subset != "normal").sum()
            result["normal_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Precision for tumor: among all cells called tumor, how many are in
    # tumor_barcodes?
    if tumor_barcodes is not None:
        tb_set = set(tumor_barcodes) & pred_index
        called_tumor = prediction_df[prediction_df["class"] == "tumor"].index
        if len(called_tumor) > 0:
            tp = sum(1 for b in called_tumor if b in tb_set)
            result["tumor_precision"] = tp / len(called_tumor)
        else:
            result["tumor_precision"] = 0.0

    if normal_barcodes is not None:
        nb_set = set(normal_barcodes) & pred_index
        called_normal = prediction_df[prediction_df["class"] == "normal"].index
        if len(called_normal) > 0:
            tp = sum(1 for b in called_normal if b in nb_set)
            result["normal_precision"] = tp / len(called_normal)
        else:
            result["normal_precision"] = 0.0

    return result


def run_full_pipeline(
    h5ad_path=None,
    sample_name: str = "sample",
    out_dir=None,
    extra_kwargs: Optional[dict] = None,
    adata=None,
    complexity_gate: bool = False,
    norm_cell_path=None,
) -> tuple:
    """Run the full M1→M4 pipeline via the Python API (not CLI).

    This is the canonical way for external tests to exercise the pipeline end-
    to-end. It mirrors the pipeline_outputs fixture in tests/test_end_to_end.py
    but operates on an arbitrary input rather than the committed fixture.

    The input is supplied either as an ``h5ad_path`` (read with read_h5ad) or as
    a pre-loaded ``adata`` — the latter lets a caller feed an AnnData produced by
    any of the package loaders (e.g. the CellRanger-H5 loader in kopya.io)
    without first round-tripping through an .h5ad file.

    Args:
        h5ad_path: Path-like pointing at the input .h5ad file. Ignored when
            ``adata`` is supplied; exactly one of the two must be given.
        sample_name: Sample identifier used for output file naming.
        out_dir: Directory where output CSVs and .seg are written. Must exist.
        extra_kwargs: Optional dict of overrides passed to filter_normalize_project
            as keyword arguments (e.g. {'min_genes': 100}).
        adata: Optional pre-loaded raw-counts AnnData. When supplied it is used
            directly (a copy) and ``h5ad_path`` is not read.
        complexity_gate: When True, mirror the CLI ``run`` command by computing
            per-cell detected-gene counts and passing them to classify_cells so
            the low-complexity gate is active. The other external tests leave it
            off (the shared default); datasets whose published baseline was
            measured with the CLI (which always feeds complexity) turn it on so
            the test reproduces those numbers.
        norm_cell_path: Optional path to a file listing known-normal barcodes
            (one per line). When given, pick_baseline runs in SUPERVISED mode
            against that reference instead of the unsupervised signature/variance
            cascade — required for datasets where the unsupervised baseline
            inverts (e.g. mesenchymal tumors such as sarcomas, whose
            fibroblast/osteoblast programs mislabel tumor as normal).

    Returns:
        (prediction_df, chr_matrix, chroms_present, segments, qc) tuple:
            prediction_df: classify_cells() output DataFrame (indexed by barcode).
            chr_matrix: (n_cells × n_chroms_present) 1.0-centered ndarray.
            chroms_present: list[str] of chromosome labels for chr_matrix columns.
            segments: detect_segments() output DataFrame.
            qc: dict with summary QC stats:
                n_cells_raw, n_cells_filtered, n_genes_filtered,
                n_tumor, n_normal, n_uncertain, n_segments.
    """
    from anndata import read_h5ad

    from kopya.outputs import (
        write_chr_cnv_matrix_csv,
        write_clones_seg,
        write_prediction_csv,
    )
    from kopya.pipeline import run_pipeline

    if (adata is None) == (h5ad_path is None):
        raise ValueError("Pass exactly one of h5ad_path or adata.")

    if out_dir is None:
        raise ValueError("out_dir is required; pass a directory for the pipeline outputs.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Delegate the M1->M4 computation to the shared in-memory core so this
    # harness exercises exactly the code path the public API (kopya.tl.cnv)
    # exposes. This function only adds the file-writing the CLI-style tests want.
    adata_in = adata if adata is not None else read_h5ad(h5ad_path)
    res = run_pipeline(
        adata_in,
        norm_cell_path=norm_cell_path,
        complexity_gate=complexity_gate,
        filter_kwargs=extra_kwargs,
    )
    barcodes = res.adata_m1.obs_names.to_numpy()

    write_prediction_csv(res.prediction_df, out_dir / f"{sample_name}_prediction.csv")
    write_chr_cnv_matrix_csv(
        res.chr_matrix, barcodes=barcodes, chroms=res.chroms_present,
        out_path=out_dir / f"{sample_name}_chr_cnv_matrix.csv",
    )
    write_clones_seg(
        cn_matrix=res.cn_matrix, segments=res.segments,
        prediction_df=res.prediction_df, var_coords=res.adata_m1.var,
        sample=sample_name, out_path=out_dir / f"{sample_name}_clones.seg",
        baseline=res.seg_baseline,
    )

    return res.prediction_df, res.chr_matrix, res.chroms_present, res.segments, res.qc
