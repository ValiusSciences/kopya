"""End-to-end pipeline test on tests/fixtures/tiny_simulated.h5ad.

This is the CI gate: every commit must preserve the algorithm's ability
to recover the three planted events on the synthetic fixture:

  - chr7 broad gain
  - chr10 broad loss
  - focal MYC amp (chr8)

The test runs the full M1→M4 chain via the Python API (no shell-out to the
CLI) and asserts ground-truth recovery on the resulting prediction frame
and chr_cnv_matrix. If the algorithm drifts so the fixture stops behaving,
this test fails and the regression is caught before a release tag.

Regenerate the fixture (only when intentionally changing the synthetic):
    python -m tests.fixtures.build_tiny_simulated
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad

from kopya.annotations import CANONICAL_CHROM_ORDER, load_gene_order
from kopya.baseline import pick_baseline
from kopya.classify import classify_cells
from kopya.normalize import filter_normalize_project
from kopya.outputs import compute_chr_cnv_matrix
from kopya.segment import detect_segments, per_cell_segment_cn, per_segment_normal_baseline
from kopya.smooth import center_against_baseline, smooth_along_chromosomes


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tiny_simulated.h5ad"


@pytest.fixture(scope="module")
def fixture_adata():
    """Load the committed synthetic fixture once per test session."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"fixture not present at {FIXTURE_PATH}; "
            "rebuild via `python -m tests.fixtures.build_tiny_simulated`",
        )
    adata = read_h5ad(FIXTURE_PATH)
    return adata


@pytest.fixture(scope="module")
def pipeline_outputs(fixture_adata):
    """Run the full M1-M4 pipeline once and reuse the outputs across tests.

    Returns:
        dict with keys:
            adata_m1: filtered + normalized + projected AnnData.
            baseline: pick_baseline() result dict.
            segments: detect_segments() output.
            cn_matrix: per_cell_segment_cn() output.
            prediction: classify_cells() per-cell DataFrame.
            chr_matrix: (n_cells × n_chroms) 1.0-centered matrix.
            chroms_present: list of chromosomes in chr_matrix column order.
            truth: adata.uns ground-truth dict from the fixture builder.
    """
    # ── M1 ───────────────────────────────────────────────────────────────
    gene_order = load_gene_order()
    adata_m1 = filter_normalize_project(fixture_adata, gene_order=gene_order)

    # ── M2 ───────────────────────────────────────────────────────────────
    # No supervised override — exercise the signature/variance cascade end-to-end.
    baseline = pick_baseline(adata_m1)

    # ── M3 ───────────────────────────────────────────────────────────────
    chr_labels = adata_m1.var["chr"].to_numpy()
    centered = center_against_baseline(adata_m1, baseline["mask"])
    smoothed = smooth_along_chromosomes(centered, chr_labels=chr_labels)
    segments = detect_segments(smoothed, normal_mask=baseline["mask"], chr_labels=chr_labels)
    cn_matrix = per_cell_segment_cn(smoothed, segments)
    # Diploid reference threaded into the absolute outputs, exactly as the CLI does:
    # chr_cnv_matrix gets the SCALAR global pedestal (concordance-safe).
    seg_baseline = per_segment_normal_baseline(cn_matrix, baseline["mask"])
    chr_pedestal = float(np.median(seg_baseline))

    # ── M4 ───────────────────────────────────────────────────────────────
    prediction = classify_cells(
        cn_matrix=cn_matrix,
        segments=segments,
        normal_mask=baseline["mask"],
        barcodes=adata_m1.obs_names.to_numpy(),
    )
    chr_matrix, chroms_present = compute_chr_cnv_matrix(
        cn_matrix, segments, CANONICAL_CHROM_ORDER, baseline=chr_pedestal,
    )

    return {
        "adata_m1": adata_m1,
        "baseline": baseline,
        "segments": segments,
        "cn_matrix": cn_matrix,
        "prediction": prediction,
        "chr_matrix": chr_matrix,
        "chroms_present": chroms_present,
        "truth": fixture_adata.uns["kopya_fixture"],
    }


# =============================================================================
# Ground-truth-recovery assertions
# =============================================================================


def test_tumor_normal_recall(pipeline_outputs):
    """≥80% of planted-tumor cells called tumor; ≥80% of planted-normal cells called normal.

    Both directions are checked so a degenerate caller that labels everything
    one class cannot pass. Tolerance is loose enough (80%) that minor GMM
    boundary noise does not fail the gate; tighter recall is a larger-cohort
    validation concern, not a unit-test concern.
    """
    truth = pipeline_outputs["truth"]
    pred = pipeline_outputs["prediction"]

    n_normal = int(truth["n_normal"])

    # The fixture orders cells normal-first then tumor — split via positional
    # index rather than re-parsing obs.
    predicted_class = pred["class"].to_numpy()
    normal_calls = predicted_class[:n_normal]
    tumor_calls = predicted_class[n_normal:]

    normal_recall = (normal_calls == "normal").mean()
    tumor_recall = (tumor_calls == "tumor").mean()

    assert normal_recall >= 0.8, f"normal recall too low: {normal_recall:.2f}"
    assert tumor_recall >= 0.8, f"tumor recall too low: {tumor_recall:.2f}"


def test_diploid_normals_center_on_one(pipeline_outputs):
    """Contract: planted-normal cells sit at ~1.0 on every chromosome.

    With the per-segment diploid reference subtracted (as the CLI does), the
    chr_cnv_matrix honors "1.0 = diploid": normal cells' per-chromosome median
    lands on 1.0 rather than on the centering pedestal.
    """
    truth = pipeline_outputs["truth"]
    chr_matrix = pipeline_outputs["chr_matrix"]
    n_normal = int(truth["n_normal"])

    normal_chr_median = np.median(chr_matrix[:n_normal, :], axis=0)
    np.testing.assert_allclose(normal_chr_median, 1.0, atol=0.1)


def test_chr7_amp_recovered(pipeline_outputs):
    """Tumor cells show clearly elevated chr7 CN; normals stay near baseline."""
    truth = pipeline_outputs["truth"]
    chr_matrix = pipeline_outputs["chr_matrix"]
    chroms = pipeline_outputs["chroms_present"]

    assert "chr7" in chroms, "chr7 missing from chr_matrix"
    col = chroms.index("chr7")

    n_normal = int(truth["n_normal"])
    normal_chr7 = chr_matrix[:n_normal, col].mean()
    tumor_chr7 = chr_matrix[n_normal:, col].mean()

    # Tumor mean should exceed normal mean by at least 0.2 in 1.0-centered space.
    # A planted 2.5x factor yields a log-deviation of ~+0.4, so 0.2 leaves
    # plenty of headroom for the CP10k global-shift drag-down on the chr_matrix.
    assert tumor_chr7 > normal_chr7 + 0.2, (
        f"chr7 gain not recovered: normal={normal_chr7:.3f}, tumor={tumor_chr7:.3f}"
    )


def test_chr10_loss_recovered(pipeline_outputs):
    """Tumor cells show clearly depressed chr10 CN; normals stay near baseline."""
    truth = pipeline_outputs["truth"]
    chr_matrix = pipeline_outputs["chr_matrix"]
    chroms = pipeline_outputs["chroms_present"]

    assert "chr10" in chroms, "chr10 missing from chr_matrix"
    col = chroms.index("chr10")

    n_normal = int(truth["n_normal"])
    normal_chr10 = chr_matrix[:n_normal, col].mean()
    tumor_chr10 = chr_matrix[n_normal:, col].mean()

    # Tumor mean must be below normal mean by at least 0.2 (planted 0.4x
    # factor → ~-0.9 log deviation → ~0.4 in exp-mapped 1.0-centered space).
    assert tumor_chr10 < normal_chr10 - 0.2, (
        f"chr10 loss not recovered: normal={normal_chr10:.3f}, tumor={tumor_chr10:.3f}"
    )


def test_focal_myc_amp_elevates_chr8_signal(pipeline_outputs):
    """Tumor cells' chr8 signal is at least slightly above normals'.

    The MYC region is only ~30 genes wide vs. ~1500 genes total on chr8 —
    a single broad chromosome metric will dilute the focal amp signal. We
    therefore use a relaxed threshold (>+0.05) and treat this as a coarse
    sensitivity check, not a focal-resolution test. Focal recall at gene
    resolution is a larger-cohort validation concern.
    """
    truth = pipeline_outputs["truth"]
    chr_matrix = pipeline_outputs["chr_matrix"]
    chroms = pipeline_outputs["chroms_present"]

    assert "chr8" in chroms, "chr8 missing from chr_matrix"
    col = chroms.index("chr8")

    n_normal = int(truth["n_normal"])
    normal_chr8 = chr_matrix[:n_normal, col].mean()
    tumor_chr8 = chr_matrix[n_normal:, col].mean()

    # Even a 30-gene focal amp leaves a measurable trace on the chromosome mean.
    # Threshold is intentionally loose; the assertion is that tumor > normal,
    # not that the amp is fully resolved.
    assert tumor_chr8 > normal_chr8 + 0.05, (
        f"focal MYC amp not visible on chr8 mean: "
        f"normal={normal_chr8:.3f}, tumor={tumor_chr8:.3f}"
    )


def test_pipeline_emits_segments_and_subclones(pipeline_outputs):
    """Sanity: segmentation produced >=1 segment per chromosome present, and
    at least one subclone was discovered (the planted population is uniform
    so we expect k=1 by construction)."""
    segments = pipeline_outputs["segments"]
    prediction = pipeline_outputs["prediction"]

    # At least one segment exists per chromosome that appears in segments.
    counts_per_chr = segments["chr"].value_counts()
    assert (counts_per_chr >= 1).all()

    # The tumor cohort is uniform — there is one underlying clone. The
    # discovery algorithm may emit k=1 or k=2 depending on Leiden noise;
    # both are acceptable for this synthetic.
    tumor_pred = prediction[prediction["class"] == "tumor"]
    distinct_clones = {s for s in tumor_pred["subclone"] if s}
    assert 1 <= len(distinct_clones) <= 2, (
        f"expected 1-2 subclones, got {len(distinct_clones)}: {distinct_clones}"
    )
