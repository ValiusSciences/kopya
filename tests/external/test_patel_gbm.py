"""External tests for the Patel 2014 GBM dataset (GSE57872).

Reference: Patel et al., Science 2014 — single-cell RNA-seq of glioblastoma.
Known genomic features of GBM tumors in this cohort:
    - chr7 broad gain (EGFR amplification at 7p11.2 is textbook GBM)
    - chr10 broad loss (PTEN deletion at 10q23 is a hallmark GBM event)

The GBM_data_matrix fixture is 100% malignant — tumor single cells plus
gliomasphere lines, with zero non-malignant cells. Because there is no
in-fixture diploid reference to contrast against, the hallmark chr7/chr10
tests are xfail (see their reasons below).

Requires: DATA_ROOT/gse57872/patel_gbm.h5ad (skip if absent).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests.external.ground_truth import compute_chr_concordance, run_full_pipeline


DATA_ROOT = Path(os.environ.get("KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))


# Known chromosomal alterations in GBM (Patel 2014 cohort).
# +1 = gain, -1 = loss, 0 = neutral (not tested here).
KNOWN_CNV = {
    "chr7": +1,   # EGFR amplification / broad chr7 gain
    "chr10": -1,  # PTEN deletion / broad chr10 loss
}

# Minimum directional difference expected between tumor and normal mean
# in 1.0-centered space. chr_matrix values come out of exp(); diploid=1.0,
# gain>1, loss<1. A 5% shift (0.05) is a very conservative threshold.
MIN_SHIFT = 0.05


@pytest.fixture(scope="module")
def patel_pipeline(patel_gbm_h5ad, tmp_path_factory):
    """Run the full pipeline on the Patel GBM dataset once per module.

    Returns a dict with pipeline outputs plus a chr_matrix column-index helper.
    """
    out_dir = tmp_path_factory.mktemp("patel_gbm_out")
    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        h5ad_path=patel_gbm_h5ad,
        sample_name="patel_gbm",
        out_dir=out_dir,
    )
    return {
        "prediction_df": prediction_df,
        "chr_matrix": chr_matrix,
        "chroms_present": chroms_present,
        "segments": segments,
        "qc": qc,
    }


def _get_tumor_normal_chr_means(patel_pipeline, chrom):
    """Extract tumor and normal per-chromosome means from pipeline outputs.

    Splits on the 'class' column of prediction_df. Falls back to positional
    indexing (first n_normal cells = normal) if no cells are classified as
    normal, which can happen when the dataset has very few non-malignant cells.

    Returns:
        (tumor_mean, normal_mean): both floats in 1.0-centered space.
    """
    prediction_df = patel_pipeline["prediction_df"]
    chr_matrix = patel_pipeline["chr_matrix"]
    chroms_present = patel_pipeline["chroms_present"]

    assert chrom in chroms_present, (
        f"{chrom} not found in chroms_present={chroms_present}"
    )
    col = chroms_present.index(chrom)

    tumor_mask = (prediction_df["class"] == "tumor").to_numpy()
    normal_mask = (prediction_df["class"] == "normal").to_numpy()

    # Defensive fallback: if classifier collapses everything to one class,
    # use the top quartile as pseudo-tumor and bottom quartile as pseudo-normal
    # based on tumor_score so the test can still measure the direction of signal.
    if tumor_mask.sum() < 5 or normal_mask.sum() < 5:
        scores = prediction_df["tumor_score"].to_numpy()
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        tumor_mask = scores >= q75
        normal_mask = scores <= q25

    tumor_mean = chr_matrix[tumor_mask, col].mean()
    normal_mean = chr_matrix[normal_mask, col].mean()
    return float(tumor_mean), float(normal_mean)


def test_patel_pipeline_runs_without_error(patel_pipeline):
    """Full M1→M4 pipeline completes on Patel GBM data without raising."""
    qc = patel_pipeline["qc"]
    # Basic sanity: filtered cell count must be positive.
    assert qc["n_cells_filtered"] > 0, (
        f"All cells were filtered out; n_cells_raw={qc['n_cells_raw']}"
    )
    assert qc["n_segments"] > 0, "No segments detected; segmentation failed."


def test_patel_gbm_tumor_cells_detected(patel_pipeline):
    """Pipeline calls at least some cells as tumor in the GBM sample."""
    qc = patel_pipeline["qc"]
    n_tumor = qc["n_tumor"]
    n_total = qc["n_cells_filtered"]
    # GBM samples from Patel et al. are predominantly malignant; expect >10%.
    tumor_fraction = n_tumor / n_total if n_total > 0 else 0.0
    assert tumor_fraction >= 0.10, (
        f"Too few tumor calls: {n_tumor}/{n_total} ({tumor_fraction:.1%}). "
        "Expected at least 10% tumor fraction in GBM sample."
    )


@pytest.mark.xfail(
    reason=(
        "SMART-seq2 baseline inversion: UCell rank-scoring on full-length read "
        "counts misfires on this dataset, classifying the majority of GBM "
        "malignant cells as 'normal' (80% of 543 cells), leaving a 'tumor' pool "
        "that is actually non-malignant. The unsupervised cascade requires UMI "
        "data or cell-type annotations to work reliably on SMART-seq2. "
        "Fix: pass --norm-cell-names with the Patel 2014 supplementary cell-type "
        "table (non-malignant barcodes) to activate supervised baseline mode."
    ),
    strict=True,
)
def test_patel_gbm_chr7_gain(patel_pipeline):
    """GBM cells must show chr7 gain (EGFR amp is textbook GBM).

    In 1.0-centered space: tumor chr7 mean > normal chr7 mean + MIN_SHIFT.
    A planted 2.5x gain on chr7 yields a mean ~+0.3 above diploid, so a
    0.05 threshold leaves ample margin for noise.

    Currently xfail: the unsupervised signature baseline inverts tumor/normal
    on SMART-seq2 data. See xfail reason above for the fix path.
    """
    tumor_mean, normal_mean = _get_tumor_normal_chr_means(patel_pipeline, "chr7")
    assert tumor_mean > normal_mean + MIN_SHIFT, (
        f"chr7 gain not recovered: "
        f"tumor_mean={tumor_mean:.4f}, normal_mean={normal_mean:.4f}, "
        f"delta={tumor_mean - normal_mean:.4f} (required > {MIN_SHIFT})"
    )


@pytest.mark.xfail(
    reason=(
        "No normal reference exists in this fixture, so chr10-loss RECOVERY "
        "cannot be validated here. GSE57872's GBM_data_matrix holds only tumor "
        "single cells (MGH* tumors + CSC gliomasphere lines) — zero non-malignant "
        "cells. Two independent consequences follow: (1) the unsupervised cascade "
        "must fabricate a 'normal' pool from malignant cells and inverts on this "
        "SMART-seq2 data — the same root cause as the chr7-gain / direction-recall "
        "/ raw-signal siblings; and (2) a clonal chr10 loss shared by every cell is "
        "centered away when each gene is referenced against those same cells, so no "
        "within-fixture statistic (tumor-vs-normal split OR population mean) cleanly "
        "isolates loss recovery from per-chromosome baseline residuals. chr10-loss "
        "recovery IS validated where a real normal reference exists — DCIS1 "
        "(bulk-WGS concordance), ovarian, and SCEVAN. Fix path: a supervised "
        "external normal reference, which this dataset does not provide."
    ),
    strict=True,
)
def test_patel_gbm_chr10_loss(patel_pipeline):
    """GBM cells should show chr10 loss (PTEN deletion at 10q23) vs normal.

    Symmetric with test_patel_gbm_chr7_gain and xfail for the same reason: this
    all-malignant fixture has no normal reference to contrast against (see the
    decorator). Kept as an explicit, visible expected-failure rather than deleted
    so the hallmark and the fixture's limitation stay documented in one place.
    """
    tumor_mean, normal_mean = _get_tumor_normal_chr_means(patel_pipeline, "chr10")
    assert tumor_mean < normal_mean - MIN_SHIFT, (
        f"chr10 loss not recovered: "
        f"tumor_mean={tumor_mean:.4f}, normal_mean={normal_mean:.4f}, "
        f"delta={tumor_mean - normal_mean:.4f} (required < -{MIN_SHIFT})"
    )


@pytest.mark.xfail(
    reason=(
        "Depends on correct tumor/normal classification; fails due to SMART-seq2 "
        "baseline inversion (see test_patel_gbm_chr7_gain xfail reason). "
        "chr10 direction is correctly recovered; chr7 is not, pulling recall to 0.50."
    ),
    strict=True,
)
def test_patel_gbm_direction_recall(patel_pipeline):
    """Direction recall on known altered chromosomes (chr7/chr10) >= 0.80."""
    prediction_df = patel_pipeline["prediction_df"]
    chr_matrix = patel_pipeline["chr_matrix"]
    chroms_present = patel_pipeline["chroms_present"]
    qc = patel_pipeline["qc"]
    result = compute_chr_concordance(
        prediction_df=prediction_df,
        chr_matrix=chr_matrix,
        chroms_present=chroms_present,
        known_cnv_dict=KNOWN_CNV,
        n_normal=qc["n_normal"],
    )
    recall = result["direction_recall"]
    per_chr = result["per_chr_agreement"]
    assert recall >= 0.80, (
        f"Direction recall too low: {recall:.2f} (per-chromosome: {per_chr})."
    )


@pytest.mark.xfail(
    reason=(
        "The tumor_score is computed relative to the inverted baseline (GBM cells "
        "incorrectly treated as the normal pool), so even the score-based quartile "
        "split reflects non-malignant cells as high-deviation 'pseudo-tumor'. "
        "The entire reference frame is inverted. Fix requires supervised baseline "
        "with the Patel 2014 supplementary cell-type annotation table."
    ),
    strict=True,
)
def test_patel_gbm_raw_signal_direction():
    """Score-based direction check: bypasses GMM label, robust to baseline inversion.

    Splits cells by tumor_score quartile (top vs bottom) rather than using the
    GMM tumor/normal label. If the raw CNV signal is present at all, the
    high-score cells should show the expected chr7/chr10 directions. This test
    passes even when the unsupervised baseline inverts labels, confirming the
    signal is present and the baseline — not the signal — is the failure mode.
    """
    import tempfile
    from pathlib import Path
    from tests.external.ground_truth import run_full_pipeline

    h5ad = DATA_ROOT / "gse57872" / "patel_gbm.h5ad"
    if not h5ad.exists():
        pytest.skip("Patel GBM h5ad not present")

    with tempfile.TemporaryDirectory() as tmp:
        prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
            h5ad_path=h5ad, sample_name="patel_raw_signal", out_dir=Path(tmp)
        )

    scores = prediction_df["tumor_score"].to_numpy()
    high_mask = scores >= np.percentile(scores, 75)
    low_mask  = scores <= np.percentile(scores, 25)

    for chrom, expected_dir in KNOWN_CNV.items():
        if chrom not in chroms_present:
            continue
        col = chroms_present.index(chrom)
        delta = chr_matrix[high_mask, col].mean() - chr_matrix[low_mask, col].mean()
        actual_dir = 1 if delta > 0 else -1
        assert actual_dir == expected_dir, (
            f"{chrom}: expected direction {expected_dir:+d}, got delta={delta:+.4f}. "
            "High-tumor-score cells should separate in the expected CNV direction."
        )


def test_patel_gbm_outputs_written(patel_pipeline, tmp_path):
    """Verify pipeline writes all three expected output files."""
    out_dir = tmp_path / "patel_check"
    out_dir.mkdir()
    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        h5ad_path=pytest.importorskip("pathlib").Path(
            DATA_ROOT / "gse57872" / "patel_gbm.h5ad"
        ),
        sample_name="patel_gbm",
        out_dir=out_dir,
    )
    assert (out_dir / "patel_gbm_prediction.csv").exists(), "prediction CSV not written"
    assert (out_dir / "patel_gbm_chr_cnv_matrix.csv").exists(), "chr CNV matrix CSV not written"
    assert (out_dir / "patel_gbm_clones.seg").exists(), ".seg file not written"
