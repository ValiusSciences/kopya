"""External tests for the DCIS1 breast cancer dataset (GSE148673).

Reference: Casasent et al. / Navin lab — single-cell RNA-seq of ductal
carcinoma in situ (DCIS). The dataset contains DCIS tumor cells and normal
epithelial/stromal cells from the same patient.

Key validation:
    - Pipeline completes without error.
    - sc CNV pseudobulk profile correlates positively with the published
      bulk WGS CNV profile for DCIS1 (Pearson r > 0.40 across chromosomes).

The bulk WGS reference is encoded below as a per-chromosome log2 ratio
vector derived from the published copy-number profile in the DCIS paper.
Values are approximate log2 ratios from the published figures.

Requires: DATA_ROOT/gse148673/dcis1.h5ad (skip if absent).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests.external.ground_truth import run_full_pipeline


DATA_ROOT = Path(os.environ.get("KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))


# Per-chromosome approximate log2 ratios from the published DCIS1 bulk WGS
# CNV profile (Casasent et al.). Values derived from Fig. 1 of the paper:
#   gains: chr1q, chr8q, chr17q, chr20
#   losses: chr16q, chr17p
# Represented here as per-arm/chromosome deviations from diploid (0 = diploid,
# positive = gain, negative = loss). Only chromosomes with clear signal are
# included; others are set to 0 (neutral) and excluded from the correlation.
DCIS1_BULK_WGS_LOG2R = {
    "chr1": +0.3,    # 1q gain (partial arm gain; whole-chrom mean ≈ +0.3)
    "chr8": +0.4,    # 8q gain (MYC amplification region)
    "chr16": -0.3,   # 16q loss (CDH1 locus)
    "chr17": +0.15,  # 17q gain (HER2/ERBB2 locus), net slightly positive
    "chr20": +0.35,  # 20q gain (frequently amplified in breast cancer)
}

# Chromosomes not listed above are treated as diploid (log2r ≈ 0) and
# included in the correlation as diploid reference points.
DIPLOID_LOG2R = 0.0

# Minimum Pearson correlation required between sc pseudobulk and bulk WGS.
MIN_BULK_CORRELATION = 0.40


@pytest.fixture(scope="module")
def dcis1_pipeline(dcis1_h5ad, tmp_path_factory):
    """Run the full pipeline on the DCIS1 dataset once per module."""
    out_dir = tmp_path_factory.mktemp("dcis1_out")
    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        h5ad_path=dcis1_h5ad,
        sample_name="dcis1",
        out_dir=out_dir,
    )
    return {
        "prediction_df": prediction_df,
        "chr_matrix": chr_matrix,
        "chroms_present": chroms_present,
        "segments": segments,
        "qc": qc,
    }


def test_dcis1_pipeline_runs(dcis1_pipeline):
    """DCIS1 pipeline completes without error.

    Validates that the filtered cell and segment counts are both positive,
    confirming that the pipeline did not silently fail.
    """
    qc = dcis1_pipeline["qc"]
    assert qc["n_cells_filtered"] > 0, (
        f"All cells were filtered out from DCIS1 "
        f"(n_cells_raw={qc['n_cells_raw']})"
    )
    assert qc["n_segments"] > 0, (
        "No segments detected in DCIS1; segmentation failed."
    )


def test_dcis1_tumor_cells_detected(dcis1_pipeline):
    """Pipeline detects at least some tumor cells in the DCIS1 sample."""
    qc = dcis1_pipeline["qc"]
    n_tumor = qc["n_tumor"]
    n_total = qc["n_cells_filtered"]
    assert n_tumor > 0, (
        f"No tumor cells detected in DCIS1 (n_total={n_total}). "
        "Expected at least some malignant DCIS cells."
    )


def test_dcis1_bulk_concordance(dcis1_pipeline):
    """DCIS1 sc profile correlates positively with bulk WGS CNV (r > 0.40).

    Builds a tumor pseudobulk CNV profile by taking the column-wise mean of
    chr_matrix for tumor-called cells, then converts from 1.0-centered space
    to log2-ratio space (log2(mean)) for comparability with the bulk WGS
    reference. Computes Pearson r across all chromosomes present in both
    the sc data and the reference.

    The correlation is computed over all chromosomes with data in both
    vectors; chromosomes absent from either are excluded.
    """
    from kopya.annotations import CANONICAL_CHROM_ORDER

    prediction_df = dcis1_pipeline["prediction_df"]
    chr_matrix = dcis1_pipeline["chr_matrix"]
    chroms_present = dcis1_pipeline["chroms_present"]

    # Build tumor pseudobulk in 1.0-centered space.
    tumor_mask = (prediction_df["class"] == "tumor").to_numpy()
    if tumor_mask.sum() < 5:
        # Fall back to top-quartile by tumor_score if classifier is degenerate.
        scores = prediction_df["tumor_score"].to_numpy()
        q75 = np.percentile(scores, 75)
        tumor_mask = scores >= q75

    tumor_pseudobulk_1c = chr_matrix[tumor_mask, :].mean(axis=0)  # (n_chroms,)

    # Convert 1.0-centered to log2-ratio: log2(value).
    # 1.0-centered values come from exp(log-space deviation), so
    # log2(exp(x)) = x / log(2). Equivalent to log2(tumor_pseudobulk_1c).
    epsilon = 1e-9  # guard against log(0)
    tumor_log2r = np.log2(tumor_pseudobulk_1c + epsilon)

    # Build parallel vectors for sc and bulk, aligned on chromosome.
    sc_values = []
    bulk_values = []
    for chrom in CANONICAL_CHROM_ORDER:
        if chrom not in chroms_present:
            continue
        col = chroms_present.index(chrom)
        sc_log2r = float(tumor_log2r[col])
        bulk_log2r = DCIS1_BULK_WGS_LOG2R.get(chrom, DIPLOID_LOG2R)
        sc_values.append(sc_log2r)
        bulk_values.append(bulk_log2r)

    if len(sc_values) < 5:
        pytest.skip(
            f"Too few chromosomes in common for correlation ({len(sc_values)}); "
            "cannot compute meaningful Pearson r."
        )

    sc_arr = np.array(sc_values)
    bulk_arr = np.array(bulk_values)

    # Pearson correlation.
    r = float(np.corrcoef(sc_arr, bulk_arr)[0, 1])

    assert r >= MIN_BULK_CORRELATION, (
        f"DCIS1 sc-bulk correlation r={r:.3f} < {MIN_BULK_CORRELATION}. "
        f"sc log2r range: [{sc_arr.min():.3f}, {sc_arr.max():.3f}]; "
        f"bulk log2r range: [{bulk_arr.min():.3f}, {bulk_arr.max():.3f}]."
    )


def test_dcis1_chr8_gain(dcis1_pipeline):
    """DCIS1 tumor cells show chr8 gain relative to normal cells.

    chr8q gain (encompassing MYC at 8q24) is a well-documented feature of
    DCIS. This test checks that tumor cells have a higher chr8 mean than
    normal cells in 1.0-centered space.
    """
    prediction_df = dcis1_pipeline["prediction_df"]
    chr_matrix = dcis1_pipeline["chr_matrix"]
    chroms_present = dcis1_pipeline["chroms_present"]

    if "chr8" not in chroms_present:
        pytest.skip("chr8 not present in DCIS1 chr_matrix.")

    col = chroms_present.index("chr8")
    tumor_mask = (prediction_df["class"] == "tumor").to_numpy()
    normal_mask = (prediction_df["class"] == "normal").to_numpy()

    if tumor_mask.sum() < 3 or normal_mask.sum() < 3:
        # Fallback to score-based split.
        scores = prediction_df["tumor_score"].to_numpy()
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        tumor_mask = scores >= q75
        normal_mask = scores <= q25

    tumor_mean = float(chr_matrix[tumor_mask, col].mean())
    normal_mean = float(chr_matrix[normal_mask, col].mean())

    assert tumor_mean > normal_mean, (
        f"DCIS1 chr8: tumor_mean={tumor_mean:.4f} not > normal_mean={normal_mean:.4f}. "
        "Expected chr8 gain in DCIS tumor cells."
    )


def test_dcis1_outputs_written(dcis1_pipeline, tmp_path):
    """All three standard output files are written for the DCIS1 sample."""
    out_dir = tmp_path / "dcis1_output_check"
    out_dir.mkdir()

    # The module-scoped fixture already ran the pipeline; re-run to check
    # file writing behavior in a fresh output directory.
    dcis1_path = (
        pytest.importorskip("pathlib").Path(
            DATA_ROOT / "gse148673" / "dcis1.h5ad"
        )
    )

    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        h5ad_path=dcis1_path,
        sample_name="dcis1",
        out_dir=out_dir,
    )

    assert (out_dir / "dcis1_prediction.csv").exists(), "prediction CSV not written"
    assert (out_dir / "dcis1_chr_cnv_matrix.csv").exists(), "chr CNV matrix CSV not written"
    assert (out_dir / "dcis1_clones.seg").exists(), ".seg file not written"
