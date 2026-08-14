"""Unit tests for the scanpy-style public API (kopya.tl / .pl)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad

import kopya as kp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_simulated.h5ad"


@pytest.fixture
def sim_adata():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE} (build via tests.fixtures.build_tiny_simulated)")
    return read_h5ad(FIXTURE)


def test_tl_cnv_writes_expected_keys(sim_adata):
    """tl.cnv annotates the AnnData with obs/obsm/uns results, aligned to n_obs."""
    adata = sim_adata
    n = adata.n_obs
    ret = kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    assert ret is None, "in-place (copy=False) must return None"

    for col in ("cnv_class", "cnv_score", "cnv_cn_burden", "cnv_subclone",
                "cnv_low_complexity"):
        assert col in adata.obs, f"missing obs[{col!r}]"
    # The scanpy surface must expose BOTH halves of the score contract: the signed
    # projection and its non-negative magnitude. Ranking near-diploid cells on the
    # signed score picks the most anti-aligned ones instead, so an API that exports
    # only cnv_score hands users the documented footgun with no way out.
    scored = adata.obs["cnv_cn_burden"].notna()
    assert (adata.obs.loc[scored, "cnv_cn_burden"] >= 0).all()
    assert adata.obs.loc[scored, "cnv_score"].min() < adata.obs.loc[scored, "cnv_cn_burden"].min()
    assert "cnv_chr" in adata.obsm
    assert "cnv" in adata.uns

    uns = adata.uns["cnv"]
    chroms = uns["chroms"]
    assert adata.obsm["cnv_chr"].shape == (n, len(chroms))
    assert uns["baseline_method"] == "supervised"
    assert uns["params"]["supervised"] is True
    # autosome-only: no sex chromosomes in the output
    assert not any(c in ("chrX", "chrY") for c in chroms)


def test_tl_cnv_aligns_filtered_cells(sim_adata):
    """Cells dropped by M1 QC are NaN in obs; scored cells match n_cells_filtered."""
    adata = sim_adata
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    n_scored = int(adata.obs["cnv_class"].notna().sum())
    assert n_scored == adata.uns["cnv"]["n_cells_filtered"]
    n_nan = adata.n_obs - n_scored
    # obsm rows for dropped cells are all-NaN; scored rows are finite.
    row_finite = np.isfinite(adata.obsm["cnv_chr"]).all(axis=1)
    assert int(row_finite.sum()) == n_scored
    assert int((~row_finite).sum()) == n_nan


def test_tl_cnv_separates_planted_tumor(sim_adata):
    """Supervised call recovers the planted tumor population (recall > 0.5)."""
    adata = sim_adata
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    scored = adata.obs["cnv_class"].notna()
    truth = adata.obs["true_class"].astype(str)
    tumor_truth = scored & (truth == "tumor")
    recall = (adata.obs.loc[tumor_truth, "cnv_class"] == "tumor").mean()
    assert recall > 0.5, f"planted-tumor recall {recall:.2f} too low"
    # immune/normal cells should rarely be called tumor
    normal_truth = scored & (truth == "normal")
    false_tumor = (adata.obs.loc[normal_truth, "cnv_class"] == "tumor").mean()
    assert false_tumor < 0.2, f"false-tumor rate {false_tumor:.2f} too high"


def test_tl_cnv_copy_does_not_mutate_input(sim_adata):
    """copy=True returns an annotated copy and leaves the input untouched."""
    adata = sim_adata
    out = kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"], copy=True)
    assert out is not None and out is not adata
    assert "cnv_class" in out.obs
    assert "cnv_class" not in adata.obs, "input must not be mutated when copy=True"


def test_tl_cnv_matches_run_pipeline(sim_adata):
    """tl.cnv's per-cell calls equal the shared run_pipeline core's (same code path)."""
    adata = sim_adata
    normals = adata.obs_names[adata.obs["true_class"].astype(str) == "normal"].tolist()
    res = kp.run_pipeline(adata, norm_cell_names=normals, complexity_gate=True)
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    api_class = adata.obs["cnv_class"].reindex(res.adata_m1.obs_names).astype(str)
    core_class = res.prediction_df["class"].astype(str)
    assert (api_class.to_numpy() == core_class.to_numpy()).all()


def test_tl_cnv_unsupervised_runs(sim_adata):
    """Omitting the reference runs the unsupervised baseline cascade."""
    adata = sim_adata
    kp.tl.cnv(adata)
    assert "cnv_class" in adata.obs
    assert adata.uns["cnv"]["baseline_method"] != "supervised"


def test_tl_cnv_reference_validation(sim_adata):
    """reference_key/reference_cat must be passed as a pair with matching cells."""
    adata = sim_adata
    with pytest.raises(ValueError):
        kp.tl.cnv(adata, reference_key="true_class")  # missing reference_cat
    with pytest.raises(ValueError):
        kp.tl.cnv(adata, reference_cat=["normal"])  # reference_cat without key
    with pytest.raises(ValueError):
        kp.tl.cnv(adata, reference_key="true_class", reference_cat=["nonexistent"])
    with pytest.raises(KeyError):
        kp.tl.cnv(adata, reference_key="no_such_column", reference_cat=["x"])


def test_tl_cnv_accepts_dense_X(sim_adata):
    """A dense .X (valid AnnData input) is coerced to CSR, not an AttributeError."""
    adata = sim_adata
    adata.X = np.asarray(adata.X.todense())
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    assert "cnv_class" in adata.obs


def test_tl_cnv_result_is_h5ad_serializable(sim_adata):
    """The annotated AnnData round-trips through write_h5ad even with dropped cells."""
    import tempfile
    from pathlib import Path
    import scipy.sparse as sp

    adata = sim_adata
    # Zero out 20 cells so M1 QC drops them -> NaN in the per-cell result columns.
    X = adata.X.toarray()
    X[:20, :] = 0
    adata.X = sp.csr_matrix(X)
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    assert int(adata.obs["cnv_class"].isna().sum()) >= 20
    assert str(adata.obs["cnv_low_complexity"].dtype) == "boolean"  # nullable, not object

    path = Path(tempfile.mkdtemp()) / "out.h5ad"
    adata.write_h5ad(path)  # must not raise
    reloaded = read_h5ad(path)
    assert "cnv_class" in reloaded.obs and "cnv_chr" in reloaded.obsm


def test_run_pipeline_supervised_zero_reference_raises(sim_adata):
    """A supervised reference that matches no surviving cell fails loudly."""
    with pytest.raises(ValueError, match="matched 0 cells"):
        kp.run_pipeline(sim_adata, norm_cell_names=["NOT_A_REAL_BARCODE"])


def test_tl_cnv_flags_normalized_matrix(sim_adata):
    """A log-normalized .X is caught (raise/warn), not silently double-normalized."""
    import scanpy as sc

    adata = sim_adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # now NOT raw counts
    with pytest.raises(ValueError):
        kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"],
                  raw_count_check="raise")
    # ignore lets deliberately-normalized/pseudo-count input through
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"],
              raw_count_check="ignore")
    assert "cnv_class" in adata.obs


def test_tl_cnv_reference_cat_type_handling(sim_adata):
    """Scalar categories and non-string (integer-coded) labels are matched."""
    adata = sim_adata
    adata.obs["ref_int"] = (adata.obs["true_class"].astype(str) == "normal").astype(int)
    # list of ints against an integer-coded column
    kp.tl.cnv(adata, reference_key="ref_int", reference_cat=[1])
    assert adata.uns["cnv"]["params"]["supervised"] is True

    other = read_h5ad(FIXTURE)
    # scalar (non-list) category
    kp.tl.cnv(other, reference_key="true_class", reference_cat="normal")
    assert other.uns["cnv"]["params"]["supervised"] is True


def test_pl_chromosome_heatmap_missing_groupby_raises(sim_adata):
    """A misspelled/absent groupby raises instead of rendering ungrouped."""
    import matplotlib
    matplotlib.use("Agg")

    adata = sim_adata
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    with pytest.raises(KeyError):
        kp.pl.chromosome_heatmap(adata, groupby="does_not_exist")


def test_tl_cnv_validates_selected_matrix_not_a_counts_layer(sim_adata):
    """The guard inspects the matrix actually run on, even if a counts layer exists.

    validate_raw_counts trusts a present layers['counts'] over .X; if we run on .X
    (layer=None) a normalized .X must still be flagged, and selecting layer='counts'
    must use the raw layer.
    """
    import scanpy as sc

    adata = sim_adata
    adata.layers["counts"] = adata.X.copy()          # raw counts in the layer
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)                                # .X now normalized
    with pytest.raises(ValueError):                  # must not skip .X because a layer exists
        kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"],
                  raw_count_check="raise")
    # selecting the raw counts layer works
    other = read_h5ad(FIXTURE)
    other.layers["counts"] = other.X.copy()
    sc.pp.normalize_total(other)
    sc.pp.log1p(other)
    kp.tl.cnv(other, reference_key="true_class", reference_cat=["normal"], layer="counts")
    assert "cnv_class" in other.obs


def test_tl_cnv_reference_cat_iterables(sim_adata):
    """reference_cat accepts any non-string iterable (Index/ndarray/generator)."""
    for make in (lambda: pd.Index(["normal"]),
                 lambda: np.array(["normal"]),
                 lambda: (x for x in ["normal"])):
        adata = read_h5ad(FIXTURE)
        kp.tl.cnv(adata, reference_key="true_class", reference_cat=make())
        assert adata.uns["cnv"]["params"]["supervised"] is True


def test_tl_cnv_numeric_reference_label(sim_adata):
    """reference_cat=[1] matches a numeric (NaN-promoted float 1.0) obs column."""
    adata = sim_adata
    adata.obs["ref_num"] = np.where(
        adata.obs["true_class"].astype(str) == "normal", 1.0, np.nan)
    kp.tl.cnv(adata, reference_key="ref_num", reference_cat=[1])
    assert adata.uns["cnv"]["params"]["supervised"] is True


def test_pick_baseline_positional_signatures_unshifted():
    """A positional signatures arg still binds to signatures, not norm_cell_names."""
    from kopya.annotations import load_gene_order
    from kopya.baseline import load_signatures, pick_baseline
    from kopya.normalize import filter_normalize_project

    m1 = filter_normalize_project(read_h5ad(FIXTURE), gene_order=load_gene_order())
    sigs, _ = load_signatures(None)
    # pick_baseline(adata, norm_cell_path=None, signatures=sigs) positionally
    result = pick_baseline(m1, None, sigs)
    assert result["method"] != "supervised"  # sigs must NOT be read as norm_cell_names


def test_pl_chromosome_heatmap_one_artist_per_group(sim_adata):
    """The group-color strip uses one Rectangle per contiguous run, not per cell."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle

    adata = sim_adata
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    ax = kp.pl.chromosome_heatmap(adata, groupby="cnv_class")
    n_rect = sum(isinstance(p, Rectangle) for p in ax.patches)
    n_groups = adata.obs["cnv_class"].dropna().nunique()
    assert n_rect <= n_groups + 1  # ~one per group, not ~n_cells


def test_tl_cnv_mixed_type_reference_cat(sim_adata):
    """reference_cat mixing native- and string-matched labels selects both."""
    adata = sim_adata
    tc = adata.obs["true_class"].astype(str).to_numpy()
    adata.obs["mixed"] = np.where(tc == "normal", "normal", "1")  # 200 "normal" + 300 "1"
    # [1, "normal"] must match the string "1" cells AND the "normal" cells.
    kp.tl.cnv(adata, reference_key="mixed", reference_cat=[1, "normal"])
    uns = adata.uns["cnv"]
    # every cell is a reference here, so nothing should be called tumor
    assert uns["n_tumor"] == 0 and uns["params"]["supervised"] is True


def test_tl_cnv_requires_unique_obs_names(sim_adata):
    """Duplicate obs_names fail fast (results can't be mapped back by barcode)."""
    adata = sim_adata
    adata.obs_names = ["dup"] * adata.n_obs
    with pytest.raises(ValueError, match="unique"):
        kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])


def test_pl_chromosome_heatmap_all_missing_group_raises(sim_adata):
    """A groupby that is all-missing among scored cells raises, not IndexError."""
    import matplotlib
    matplotlib.use("Agg")

    adata = sim_adata
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    adata.obs["empty_group"] = np.nan  # valid column, no labels
    with pytest.raises(ValueError):
        kp.pl.chromosome_heatmap(adata, groupby="empty_group")


def test_pl_chromosome_heatmap(sim_adata):
    """pl.chromosome_heatmap renders from the annotated AnnData without error."""
    import matplotlib
    matplotlib.use("Agg")

    adata = sim_adata
    kp.tl.cnv(adata, reference_key="true_class", reference_cat=["normal"])
    ax = kp.pl.chromosome_heatmap(adata, groupby="true_class")
    assert ax is not None
    # x-axis has one tick per chromosome
    assert len(ax.get_xticks()) == len(adata.uns["cnv"]["chroms"])
