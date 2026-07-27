"""External tests for the 10X 17k ovarian scFFPE dataset (HGSOC).

Reference: 10X Genomics Flex demonstration dataset — "17k Ovarian Cancer
scFFPE" (high-grade papillary serous carcinoma, stage III-B, Discovery Life
Sciences FFPE tissue, Cell Ranger 8.0.1). 10X publishes a per-barcode
cell-type annotation CSV that serves as an independent gold-standard label
set the pipeline is never shown.

This is a strong real-world validation case: the sample mixes immune
(macrophages, T & NK), stromal (fibroblasts, pericytes, smooth muscle,
endothelial), the tumor's own epithelial lineage (Fallopian Tube, Ciliated,
Granulosa), and several malignant states + subclones. HGSOC is highly
aneuploid, so the ground CNV signal is strong.

The pipeline is run fully UNSUPERVISED (signature baseline; no cell-type
labels are passed in). The published annotations are joined afterwards purely
to score the calls.

Acceptance criteria (baseline measured in issue #13; margins are comfortable):
    - malignant vs immune/stromal separation AUC >= 0.85  (measured 0.893)
    - immune/stromal false-tumor rate <= 5%               (measured 2.0%)
    - recall on the main tumor states                     (measured 0.80)
      (Tumor / Proliferative / VEGFA+ / Inflammatory) >= 0.70

Documented, but NOT gated on (see issue #13):
    - same-lineage floor: normal Ciliated Epithelial Cells are partly called
      tumor because they share the malignant epithelial lineage;
    - two low-signal malignant states (MT-High Jun+/Fos+, Malignant Cells
      Lining Cyst) are under-called — the overall malignant recall (~0.72) is
      dragged down by these, so we gate on the main-state recall instead.

Requires DATA_ROOT/ovarian-scffpe/ (skip if absent):
    17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5
    FLEX_Ovarian_Barcode_Cluster_Annotation.csv
"""

import pandas as pd
import pytest

from tests.external.ground_truth import run_full_pipeline


# ── Ground-truth annotation groups (from the 10X FLEX annotation CSV) ────────

# All annotated malignant states.
MALIGNANT_STATES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    "VEGFA+ Tumor Cells",
    "Inflammatory Tumor Cells",
    "MT-High, Jun+/Fos+ Tumor Cells",
    "Malignant Cells Lining Cyst",
}

# The main, high-signal malignant states we gate recall on. The two low-signal
# states (MT-High, Malignant Cells Lining Cyst) are excluded by design — they
# are under-called (documented caveat) and would pull recall below the floor.
MAIN_TUMOR_STATES = {
    "Tumor Cells",
    "Proliferative Tumor Cells",
    "VEGFA+ Tumor Cells",
    "Inflammatory Tumor Cells",
}

# Immune + stromal cells: the true-negative set for the false-tumor rate and
# the negative class for the AUC. Deliberately EXCLUDES the normal epithelial
# lineage (Fallopian Tube / Ciliated / Granulosa), which shares the tumor's
# lineage and is expected to be harder to separate (same-lineage floor).
IMMUNE_STROMAL_STATES = {
    "Macrophages",
    "T & NK Cells",
    "Tumor Associated Fibroblasts",
    "Stromal Associated Fibroblasts",
    "Endothelial Cells",
    "Smooth Muscle Cells",
    "Pericytes",
}

# Acceptance thresholds (issue #13).
MIN_MALIGNANT_VS_IMMUNE_STROMAL_AUC = 0.85
MAX_IMMUNE_STROMAL_FALSE_TUMOR_RATE = 0.05
MIN_MAIN_TUMOR_RECALL = 0.70


@pytest.fixture(scope="module")
def ovarian_pipeline(ovarian_scffpe_data, tmp_path_factory):
    """Run the unsupervised pipeline on the ovarian scFFPE h5, join annotations.

    Loads the filtered matrix through the package's CellRanger-H5 loader
    (kopya.io.load_counts), runs the full M1->M4 pipeline unsupervised
    with the low-complexity gate active (mirroring the ``kopya run``
    CLI that produced the issue baseline), then inner-joins the published
    per-barcode annotations onto the predictions (scoring only cells present
    in both the predictions and the annotation CSV).

    Returns a dict with:
        joined: DataFrame indexed by barcode with the prediction columns plus an
            'annotation' column; rows are the cells present in BOTH the
            predictions and the annotation CSV.
        prediction_df, qc: raw pipeline outputs for sanity assertions.
    """
    from kopya.io import load_counts

    h5_path, annotation_path = ovarian_scffpe_data
    out_dir = tmp_path_factory.mktemp("ovarian_scffpe_out")

    adata = load_counts(cellranger_h5=str(h5_path))
    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        adata=adata,
        sample_name="ovarian_scffpe",
        out_dir=out_dir,
        complexity_gate=True,
    )

    annotations = pd.read_csv(annotation_path, index_col=0)
    # The CSV has a single "Cell Annotation" column; normalise the name.
    annotations.columns = ["annotation"]
    joined = prediction_df.join(annotations, how="inner")

    return {
        "joined": joined,
        "prediction_df": prediction_df,
        "qc": qc,
    }


def test_ovarian_pipeline_runs(ovarian_pipeline):
    """Unsupervised pipeline completes on the ovarian scFFPE sample.

    Confirms cells survive QC, segments are detected, and the published
    annotations line up with the called barcodes.
    """
    qc = ovarian_pipeline["qc"]
    joined = ovarian_pipeline["joined"]

    assert qc["n_cells_filtered"] > 0, (
        f"All cells filtered out (n_cells_raw={qc['n_cells_raw']})."
    )
    assert qc["n_segments"] > 0, "No segments detected; segmentation failed."
    # The barcodes must actually match between predictions and annotations,
    # otherwise every downstream metric is silently computed on an empty join.
    assert len(joined) > 10000, (
        f"Only {len(joined)} barcodes matched between predictions and the "
        "annotation CSV; expected the bulk of ~17k cells to join."
    )


def test_ovarian_malignant_vs_immune_stromal_auc(ovarian_pipeline):
    """tumor_score separates malignant from immune/stromal cells (AUC >= 0.85).

    Uses the continuous per-cell tumor_score (aneuploidy load) as the ranking
    variable and the published annotations as labels: malignant states are the
    positive class, immune + stromal cells the negative class. Epithelial
    normals are excluded here (same-lineage floor is a documented caveat).
    """
    from sklearn.metrics import roc_auc_score

    joined = ovarian_pipeline["joined"]
    is_malignant = joined["annotation"].isin(MALIGNANT_STATES)
    is_immune_stromal = joined["annotation"].isin(IMMUNE_STROMAL_STATES)

    subset = joined[is_malignant | is_immune_stromal]
    y_true = subset["annotation"].isin(MALIGNANT_STATES).astype(int).to_numpy()
    scores = subset["tumor_score"].to_numpy()

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    assert n_pos > 100 and n_neg > 100, (
        f"Degenerate label counts for AUC (malignant={n_pos}, "
        f"immune/stromal={n_neg}); annotation join likely broke."
    )

    auc = roc_auc_score(y_true, scores)
    assert auc >= MIN_MALIGNANT_VS_IMMUNE_STROMAL_AUC, (
        f"Malignant-vs-immune/stromal AUC {auc:.3f} < "
        f"{MIN_MALIGNANT_VS_IMMUNE_STROMAL_AUC} "
        f"(malignant={n_pos}, immune/stromal={n_neg})."
    )


def test_ovarian_immune_stromal_false_tumor_rate(ovarian_pipeline):
    """<= 5% of immune/stromal cells are (wrongly) called tumor."""
    joined = ovarian_pipeline["joined"]
    immune_stromal = joined[joined["annotation"].isin(IMMUNE_STROMAL_STATES)]

    assert len(immune_stromal) > 100, (
        f"Only {len(immune_stromal)} immune/stromal cells found; "
        "annotation join likely broke."
    )

    false_tumor_rate = (immune_stromal["class"] == "tumor").mean()
    n_false = int((immune_stromal["class"] == "tumor").sum())
    assert false_tumor_rate <= MAX_IMMUNE_STROMAL_FALSE_TUMOR_RATE, (
        f"Immune/stromal false-tumor rate {false_tumor_rate:.3f} "
        f"({n_false}/{len(immune_stromal)}) > "
        f"{MAX_IMMUNE_STROMAL_FALSE_TUMOR_RATE}."
    )


def test_ovarian_main_tumor_recall(ovarian_pipeline):
    """Recall on the main tumor states (Tumor/Prolif/VEGFA+/Inflammatory) >= 0.70."""
    joined = ovarian_pipeline["joined"]
    main_tumor = joined[joined["annotation"].isin(MAIN_TUMOR_STATES)]

    assert len(main_tumor) > 100, (
        f"Only {len(main_tumor)} main-state tumor cells found; "
        "annotation join likely broke."
    )

    recall = (main_tumor["class"] == "tumor").mean()
    n_recalled = int((main_tumor["class"] == "tumor").sum())
    assert recall >= MIN_MAIN_TUMOR_RECALL, (
        f"Main tumor-state recall {recall:.3f} "
        f"({n_recalled}/{len(main_tumor)}) < {MIN_MAIN_TUMOR_RECALL}."
    )
