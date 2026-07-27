"""External test for the Maynard 2020 lung dataset (maynard2020_3k).

Reference: Maynard et al., Cell 2020 — SMART-seq2 scRNA-seq of lung
adenocarcinoma. This is the 3k-cell demonstration subset used by the infercnvpy
tutorial, so it doubles as a neutral, third-party head-to-head between
kopya and inferCNV: both tools are given the SAME immune reference.

The malignant population is the annotated 'Epithelial cell' cluster (a mix of
tumor and some normal lung epithelium across ~15 patients); the immune lineages
are the supervised normal reference. The dataset ships log-normalized (no raw
UMIs), so the converter stores de-logged pseudo-counts — see convert_maynard_lung.py.

Acceptance criteria (measured values in comments; margins are comfortable):
    - malignant (Epithelial) vs immune AUC (tumor_score)   >= 0.80  (measured 0.885)
    - immune false-tumor rate                              <= 0.03  (measured 0.000)
    - fraction of Epithelial cells called tumor            >= 0.40  (measured 0.62;
      the cluster mixes tumor + normal epithelium, so a full recall is not expected)
    - chr7 gain in tumor (lung adenocarcinoma hallmark)    Epithelial > immune + 0.05
      (measured +0.149 vs +0.001)

Also (guarded, skips if infercnvpy is absent):
    - kopya vs inferCNV tumor per-chromosome concordance  Pearson >= 0.50 (measured 0.67)

Requires DATA_ROOT/maynard-lung/maynard_lung.h5ad (skip if absent) — build it with
``python tests/external/convert_maynard_lung.py`` (see GOLD_STANDARD_TESTING.md).
"""

import numpy as np
import pandas as pd
import pytest

from tests.external.ground_truth import run_full_pipeline

TUMOR_LABEL = "Epithelial cell"
IMMUNE_REF = {
    "B cell", "Macrophage", "Mast cell", "Monocyte", "NK cell", "Plasma cell",
    "T cell CD4", "T cell CD8", "T cell regulatory", "T cell dividing", "mDC", "pDC",
}
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

MIN_TUMOR_CELLS = 200
MIN_TUMOR_VS_REF_AUC = 0.80
MAX_IMMUNE_FALSE_TUMOR = 0.03
MIN_EPITHELIAL_TUMOR_FRACTION = 0.40
MIN_CHR7_GAIN_GAP = 0.05
MIN_INFERCNV_PEARSON = 0.50


@pytest.fixture(scope="module")
def maynard_pipeline(maynard_lung_h5ad, tmp_path_factory):
    """Run the SUPERVISED pipeline on Maynard lung (immune cells as reference)."""
    from anndata import read_h5ad

    out_dir = tmp_path_factory.mktemp("maynard_out")
    adata = read_h5ad(maynard_lung_h5ad)
    cell_type = adata.obs["cell_type"].astype(str)

    normal_bc = adata.obs_names[cell_type.isin(IMMUNE_REF).to_numpy()]
    norm_cell_path = out_dir / "immune_ref.txt"
    norm_cell_path.write_text("\n".join(normal_bc) + "\n")

    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        adata=adata, sample_name="maynard", out_dir=out_dir,
        norm_cell_path=str(norm_cell_path), complexity_gate=True)

    joined = prediction_df.join(cell_type.rename("cell_type"), how="inner")
    return {
        "prediction_df": prediction_df, "chr_matrix": chr_matrix,
        "chroms_present": chroms_present, "segments": segments, "qc": qc,
        "joined": joined, "cell_type": cell_type,
        "barcodes": prediction_df.index.to_numpy(),
    }


def _chr_log2(chr_matrix, chroms_present, mask, chrom):
    col = chroms_present.index(chrom)
    return float(np.log2(np.clip(chr_matrix[mask, col].mean(), 1e-6, None)))


def _epithelial_pseudobulk_log2(pipe):
    epi = set(pipe["cell_type"].index[pipe["cell_type"] == TUMOR_LABEL])
    vm = pd.DataFrame(pipe["chr_matrix"], index=pipe["barcodes"], columns=pipe["chroms_present"])
    sub = vm.loc[vm.index.intersection(epi)]
    return np.log2(sub.mean(axis=0).clip(lower=1e-6)).reindex(AUTOSOMES)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_maynard_pipeline_runs(maynard_pipeline):
    """Supervised pipeline completes and detects a tumor population."""
    qc = maynard_pipeline["qc"]
    assert qc["n_cells_filtered"] > 0, "All cells filtered out."
    assert qc["n_segments"] > 0, "No segments detected."
    assert qc["n_tumor"] >= MIN_TUMOR_CELLS, (
        f"Only {qc['n_tumor']} tumor cells called; expected >= {MIN_TUMOR_CELLS}."
    )


def test_maynard_tumor_vs_reference_auc(maynard_pipeline):
    """tumor_score separates Epithelial (tumor) from immune cells (AUC >= 0.80)."""
    from sklearn.metrics import roc_auc_score

    j = maynard_pipeline["joined"]
    epi = j["cell_type"] == TUMOR_LABEL
    imm = j["cell_type"].isin(IMMUNE_REF)
    sub = j[epi | imm]
    y = sub["cell_type"].eq(TUMOR_LABEL).astype(int).to_numpy()
    assert y.sum() > 100 and (y == 0).sum() > 100, "Degenerate label counts for AUC."
    auc = roc_auc_score(y, sub["tumor_score"].to_numpy())
    assert auc >= MIN_TUMOR_VS_REF_AUC, (
        f"Epithelial-vs-immune AUC {auc:.3f} < {MIN_TUMOR_VS_REF_AUC}."
    )


def test_maynard_immune_false_tumor_rate(maynard_pipeline):
    """<= 3% of immune reference cells are (wrongly) called tumor."""
    j = maynard_pipeline["joined"]
    imm = j[j["cell_type"].isin(IMMUNE_REF)]
    assert len(imm) > 100, "Too few immune cells joined."
    rate = (imm["class"] == "tumor").mean()
    assert rate <= MAX_IMMUNE_FALSE_TUMOR, (
        f"Immune false-tumor rate {rate:.3f} "
        f"({int((imm['class'] == 'tumor').sum())}/{len(imm)}) > {MAX_IMMUNE_FALSE_TUMOR}."
    )


def test_maynard_epithelial_tumor_fraction(maynard_pipeline):
    """A substantial fraction of Epithelial cells are called tumor (>= 0.40).

    Not a full recall: Maynard's Epithelial cluster mixes malignant cells with
    normal lung epithelium across many patients, so some Epithelial cells are
    genuinely diploid and correctly called normal.
    """
    j = maynard_pipeline["joined"]
    epi = j[j["cell_type"] == TUMOR_LABEL]
    assert len(epi) > 100, "Too few Epithelial cells joined."
    frac = (epi["class"] == "tumor").mean()
    assert frac >= MIN_EPITHELIAL_TUMOR_FRACTION, (
        f"Epithelial tumor fraction {frac:.3f} "
        f"({int((epi['class'] == 'tumor').sum())}/{len(epi)}) < {MIN_EPITHELIAL_TUMOR_FRACTION}."
    )


def test_maynard_chr7_gain(maynard_pipeline):
    """Tumor cells show chr7 gain (a lung adenocarcinoma hallmark) vs the reference."""
    pipe = maynard_pipeline
    j = pipe["joined"]
    chroms = pipe["chroms_present"]
    cm = pipe["chr_matrix"]
    bc = pd.Index(pipe["barcodes"])
    epi_mask = bc.isin(j.index[j["cell_type"] == TUMOR_LABEL])
    imm_mask = bc.isin(j.index[j["cell_type"].isin(IMMUNE_REF)])
    assert "chr7" in chroms, "chr7 not present in the chromosome matrix."
    epi_chr7 = _chr_log2(cm, chroms, epi_mask, "chr7")
    imm_chr7 = _chr_log2(cm, chroms, imm_mask, "chr7")
    assert epi_chr7 - imm_chr7 >= MIN_CHR7_GAIN_GAP, (
        f"chr7 gain not recovered: Epithelial {epi_chr7:+.3f} vs immune "
        f"{imm_chr7:+.3f} (gap {epi_chr7 - imm_chr7:+.3f}, required >= {MIN_CHR7_GAIN_GAP})."
    )


def test_maynard_vs_infercnv_concordance(maynard_pipeline):
    """kopya and inferCNV agree on the tumor per-chromosome profile (r >= 0.50).

    The head-to-head this dataset exists for: run inferCNV (infercnvpy) with the
    SAME immune reference and correlate the two tools' Epithelial-cell per-
    chromosome CNV profiles. Skips if infercnvpy is unavailable or its run fails
    in this environment.
    """
    cnv = pytest.importorskip("infercnvpy")
    import scipy.sparse as sp

    kopya = _epithelial_pseudobulk_log2(maynard_pipeline)

    try:
        a = cnv.datasets.maynard2020_3k()
        ref = [c for c in IMMUNE_REF if c in set(a.obs["cell_type"].astype("category").cat.categories)]
        cnv.tl.infercnv(a, reference_key="cell_type", reference_cat=ref,
                        window_size=250, n_jobs=1)
    except Exception as exc:  # noqa: BLE001 — env/spawn issues -> skip, don't fail
        pytest.skip(f"infercnvpy run unavailable in this environment: {exc}")

    xcnv = a.obsm["X_cnv"]
    xcnv = xcnv.toarray() if sp.issparse(xcnv) else np.asarray(xcnv)
    chr_pos = a.uns["cnv"]["chr_pos"]
    order = sorted(chr_pos.items(), key=lambda kv: kv[1])
    bounds = {c: (s, (order[i + 1][1] if i + 1 < len(order) else xcnv.shape[1]))
              for i, (c, s) in enumerate(order)}
    emask = a.obs["cell_type"].astype(str).eq(TUMOR_LABEL).to_numpy()
    infercnv = pd.Series({c: xcnv[emask, s:e].mean() for c, (s, e) in bounds.items()}).reindex(AUTOSOMES)

    x = kopya.to_numpy(float)
    y = infercnv.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    assert ok.sum() >= 18, f"Only {ok.sum()} chromosomes in common."
    r = float(np.corrcoef(x[ok] - np.median(x[ok]), y[ok] - np.median(y[ok]))[0, 1])
    assert r >= MIN_INFERCNV_PEARSON, (
        f"kopya vs inferCNV tumor per-chromosome Pearson {r:.3f} < {MIN_INFERCNV_PEARSON}."
    )
