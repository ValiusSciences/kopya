"""Unit tests for normalize.py — filter, normalize, project.

Exercises each of the three M1 steps independently plus the
filter_normalize_project() wrapper. Uses the tiny synthetic fixtures from
conftest.py so tests stay fast and assertions are deterministic.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from kopya.normalize import (
    DEFAULT_TARGET_SUM,
    _SCAN_CHUNK,
    _looks_like_raw_counts,
    _raw_count_reason,
    _scan_value_reason,
    filter_cells_and_genes,
    filter_normalize_project,
    normalize_log1p,
    project_onto_genome,
    validate_raw_counts,
)


def test_filter_cells_and_genes_drops_low_quality(tiny_adata):
    """Cells below min_genes and genes outside the detection-rate window are dropped."""
    # Zero-out one cell entirely so it falls below min_genes=200 (well below
    # the synthetic dense matrix's per-cell gene count).
    X = tiny_adata.X.toarray()
    X[0, :] = 0
    tiny_adata.X = csr_matrix(X)

    # min_genes=5 is below the per-cell gene count for the remaining 49 cells
    # but above 0 — so only cell_0000 should be dropped.
    out = filter_cells_and_genes(
        tiny_adata,
        min_genes=5,
        min_cells=1,
        low_dr=0.0,
        up_dr=1.0,
    )

    # 49 cells survive; no gene drop because low_dr=0 / up_dr=1 is permissive.
    assert out.n_obs == 49
    assert "cell_0000" not in out.obs_names


def test_filter_drops_low_detection_genes(tiny_adata):
    """Genes below low_dr are dropped via the detection-rate window."""
    # Zero out one gene's column so its detection rate becomes 0.
    X = tiny_adata.X.toarray()
    X[:, 0] = 0
    tiny_adata.X = csr_matrix(X)

    # low_dr=0.5 will drop the zero-column gene (detection rate 0).
    out = filter_cells_and_genes(
        tiny_adata,
        min_genes=1,
        min_cells=1,
        low_dr=0.5,
        up_dr=1.0,
    )

    # The first gene was zeroed out — it should be dropped.
    assert tiny_adata.var_names[0] not in out.var_names
    # The rest survive.
    assert out.n_vars == tiny_adata.n_vars - 1


def test_filter_does_not_mutate_input(tiny_adata):
    """filter_cells_and_genes operates on a defensive copy."""
    n_cells_before = tiny_adata.n_obs
    n_genes_before = tiny_adata.n_vars

    _ = filter_cells_and_genes(tiny_adata, min_genes=5, low_dr=0.0)

    # Original adata is unchanged after the call.
    assert tiny_adata.n_obs == n_cells_before
    assert tiny_adata.n_vars == n_genes_before


def test_normalize_log1p_preserves_sparsity(tiny_adata):
    """Output is still sparse CSR and zeros stay zero (log1p(0) = 0)."""
    out = normalize_log1p(tiny_adata)

    assert issparse(out.X)
    assert out.X.format == "csr"
    # nnz must not increase — log1p preserves the zero pattern exactly.
    assert out.X.nnz == tiny_adata.X.nnz


def test_normalize_log1p_dtype_is_float(tiny_adata):
    """Output dtype is float64 even though input was int counts."""
    assert tiny_adata.X.dtype == np.int32  # sanity check on the fixture

    out = normalize_log1p(tiny_adata)

    # log1p produces floats; we must not silently truncate back to int.
    assert np.issubdtype(out.X.dtype, np.floating)


def test_normalize_log1p_row_sums_match_target(tiny_adata):
    """After undo-ing log1p, per-cell totals equal target_sum (CP10k)."""
    out = normalize_log1p(tiny_adata, target_sum=DEFAULT_TARGET_SUM)

    # Reverse log1p to recover CP10k counts; rows sums should equal target_sum
    # within float tolerance.
    cp10k = np.expm1(out.X.toarray())
    totals = cp10k.sum(axis=1)

    np.testing.assert_allclose(totals, DEFAULT_TARGET_SUM, rtol=1e-5)


def test_normalize_log1p_handles_zero_row():
    """A cell with no counts (sum=0) must not produce NaNs in the output."""
    # Build a tiny AnnData with one zero-row and one normal row.
    counts = csr_matrix(np.array([[0, 0, 0], [5, 3, 2]], dtype=np.int32))
    obs = pd.DataFrame(index=["zero_cell", "normal_cell"])
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    adata = AnnData(X=counts, obs=obs, var=var)

    out = normalize_log1p(adata)

    # No NaNs anywhere — the safe-divide guards the zero-sum row.
    assert not np.any(np.isnan(out.X.toarray()))


def test_project_onto_genome_sorts_genes(tiny_adata, tiny_gene_order):
    """Genes are sorted by canonical genomic order after projection."""
    out = project_onto_genome(tiny_adata, gene_order=tiny_gene_order)

    # chrY (TY_GENE), chrX (G_CHRX_EARLY), MT-DROP, and MKI67 (cycle) are
    # dropped — the remaining 6 autosomal genes are sorted in (chr, start) order.
    expected_order = [
        "A_CHR1_EARLY",   # chr1, 100_000
        "B_CHR1_MID",     # chr1, 500_000
        "C_CHR1_LATE",    # chr1, 900_000
        "D_CHR2_EARLY",   # chr2, 200_000
        "E_CHR2_LATE",    # chr2, 800_000
        "F_CHR7_MID",     # chr7, 400_000
    ]
    assert list(out.var_names) == expected_order


def test_project_adds_genomic_columns(tiny_adata, tiny_gene_order):
    """var gains (chr, start, end) columns wired from the gene-order table."""
    out = project_onto_genome(tiny_adata, gene_order=tiny_gene_order)

    assert "chr" in out.var.columns
    assert "start" in out.var.columns
    assert "end" in out.var.columns
    # The first surviving gene is A_CHR1_EARLY at start=100_000.
    assert out.var.iloc[0]["start"] == 100_000


def _cp10k_log1p(counts):
    """Return log1p(CP10k) of a dense integer count matrix (as float ndarray)."""
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return np.log1p(counts / totals * DEFAULT_TARGET_SUM)


def test_looks_like_raw_counts_accepts_raw(tiny_adata):
    """Genuine non-negative integer counts are recognized as raw (no reason)."""
    assert _looks_like_raw_counts(tiny_adata.X)
    assert _raw_count_reason(tiny_adata.X) is None


def test_looks_like_raw_counts_tolerates_float_stored_integers(tiny_adata):
    """Integer counts stored in a float dtype (common in h5ad) still read as raw."""
    x_float = csr_matrix(tiny_adata.X.toarray().astype(np.float32))
    assert _looks_like_raw_counts(x_float)


def test_raw_count_reason_flags_fractional_values(tiny_adata):
    """log1p(CP10k) float data is flagged as non-integer, i.e. not raw counts."""
    log_x = csr_matrix(_cp10k_log1p(tiny_adata.X.toarray()))
    reason = _raw_count_reason(log_x)
    assert reason is not None
    assert "non-integer" in reason


def test_raw_count_reason_flags_negative_values():
    """Centered/z-scored data (negatives) cannot be raw counts."""
    x = csr_matrix(np.array([[1.0, -2.0, 3.0], [0.0, 4.0, -1.0]]))
    reason = _raw_count_reason(x)
    assert reason is not None
    assert "negative" in reason


def test_raw_count_reason_flags_non_finite_values():
    """NaN / Inf cannot occur in raw counts and must be flagged."""
    x = csr_matrix(np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]]))
    reason = _raw_count_reason(x)
    assert reason is not None
    assert "non-finite" in reason


def test_scan_value_reason_detects_violation_past_first_chunk():
    """Chunked scanning must not miss a fractional value in a later chunk.

    The buffer is scanned _SCAN_CHUNK elements at a time to bound memory; this
    guards against an off-by-chunk bug that would only inspect the first chunk.
    """
    values = np.ones(_SCAN_CHUNK + 5, dtype=np.float64)
    values[_SCAN_CHUNK + 2] = 1.5  # fractional, lives in the second chunk
    reason = _scan_value_reason(values)
    assert reason is not None
    assert "non-integer" in reason


def test_raw_count_reason_flags_uniform_library_size():
    """Integer data with near-uniform per-cell totals looks library-normalized.

    Simulates CP10k values rounded back to integers: values are whole numbers,
    every cell sums to ~1e4, and 1e4 is a canonical normalization target — the
    combined fingerprint of library-size normalization.
    """
    rng = np.random.default_rng(0)
    raw = rng.poisson(3.0, size=(30, 12)).astype(float)
    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    cp_rounded = np.rint(raw / totals * 1e4).astype(np.int64)

    reason = _raw_count_reason(csr_matrix(cp_rounded))
    assert reason is not None
    assert "library" in reason


def test_uniform_library_detected_despite_zero_cell():
    """A single all-zero cell must not mask an otherwise CP10k-normalized matrix.

    Uniformity is measured over positive-library cells, so an empty cell (which a
    normalized matrix can still carry) does not inflate the CV and hide it.
    """
    rng = np.random.default_rng(1)
    raw = rng.poisson(3.0, size=(29, 12)).astype(float)
    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    cp_rounded = np.rint(raw / totals * 1e4).astype(np.int64)
    # Prepend one all-zero cell -> 30 rows, 29 with a positive library.
    with_zero = np.vstack([np.zeros((1, 12), dtype=np.int64), cp_rounded])

    reason = _raw_count_reason(csr_matrix(with_zero))
    assert reason is not None
    assert "library" in reason


def test_uniform_library_not_near_target_is_raw():
    """Near-uniform integer totals that don't cluster at CP10k/CPM stay 'raw'.

    A low coefficient of variation alone is not proof of normalization — genuine
    (e.g. uniform-depth) integer counts whose totals sit far from any canonical
    target must not be rejected. Here every cell sums to exactly 1000.
    """
    counts = csr_matrix(np.full((40, 5), 200, dtype=np.int64))  # totals = 1000
    assert _looks_like_raw_counts(counts)


def test_uniform_library_check_ignores_small_matrices():
    """The uniform-library signal must not fire on tiny matrices (few cells).

    A handful of cells can share a total by chance; the heuristic only trusts
    the signal once there are enough cells, so small raw inputs stay 'raw'.
    """
    # 5 cells (< the min-cells floor) all summing to 10 — still treated as raw.
    counts = csr_matrix(np.full((5, 5), 2, dtype=np.int64))
    assert _looks_like_raw_counts(counts)


def test_raw_count_reason_empty_matrix_is_raw():
    """An all-zero matrix carries no evidence and must not trip a warning."""
    assert _raw_count_reason(csr_matrix((4, 4))) is None


def _adata(x, layers=None):
    """Build a minimal AnnData wrapping matrix x (n_cells × n_genes)."""
    n_obs, n_var = x.shape
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_var)])
    return AnnData(X=x, obs=obs, var=var, layers=layers)


def test_validate_raw_counts_no_warning_on_raw(tiny_adata):
    """Raw counts pass validation silently and report True."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise here
        assert validate_raw_counts(tiny_adata) is True


def test_validate_raw_counts_warns_on_normalized(tiny_adata):
    """log1p/CP10k float data triggers a UserWarning and reports False."""
    log_x = csr_matrix(_cp10k_log1p(tiny_adata.X.toarray()))
    adata = _adata(log_x)
    with pytest.warns(UserWarning, match="does not look like raw integer UMI counts"):
        result = validate_raw_counts(adata)
    assert result is False


def test_validate_raw_counts_uses_counts_layer(tiny_adata):
    """When layers['counts'] exists it is trusted; .X is not inspected."""
    raw = tiny_adata.X
    log_x = csr_matrix(_cp10k_log1p(raw.toarray()))
    # .X is normalized, but a raw-count layer is present -> no warning.
    adata = _adata(log_x, layers={"counts": raw})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate_raw_counts(adata) is True


def test_validate_raw_counts_raise_action(tiny_adata):
    """action='raise' turns a non-raw matrix into a hard error (opt-in)."""
    log_x = csr_matrix(_cp10k_log1p(tiny_adata.X.toarray()))
    adata = _adata(log_x)
    with pytest.raises(ValueError, match="does not look like raw integer UMI counts"):
        validate_raw_counts(adata, action="raise")


def test_validate_raw_counts_ignore_action(tiny_adata):
    """action='ignore' skips the check entirely, even on normalized data."""
    log_x = csr_matrix(_cp10k_log1p(tiny_adata.X.toarray()))
    adata = _adata(log_x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validate_raw_counts(adata, action="ignore") is True


def test_validate_raw_counts_rejects_bad_action(tiny_adata):
    """An unrecognized action is a programming error and raises immediately."""
    with pytest.raises(ValueError, match="action must be"):
        validate_raw_counts(tiny_adata, action="explode")


def test_filter_normalize_project_end_to_end(tiny_adata, tiny_gene_order):
    """The wrapper composes the three steps and produces a kept-gene AnnData."""
    out = filter_normalize_project(
        tiny_adata,
        gene_order=tiny_gene_order,
        min_genes=1,
        min_cells=1,
        low_dr=0.0,
    )

    # Same kept set as project_onto_genome's standalone test (autosomes only).
    expected_n_genes = 6
    assert out.n_vars == expected_n_genes
    # Counts went through log1p → values are floats in [0, ~log1p(target_sum)].
    assert np.issubdtype(out.X.dtype, np.floating)
    assert out.X.max() < np.log1p(DEFAULT_TARGET_SUM) + 1.0


# ---------------------------------------------------------------------------
# Memory-lean M1 rewrite: equivalence + immutability guards.
# The filter/normalize rewrites must match the prior semantics exactly and
# never mutate the caller's matrix (the whole point is a memory win, not a
# behavior change). These are deliberately generous.
# ---------------------------------------------------------------------------

def _random_raw_adata(seed=0, n_cells=80, n_genes=50):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.4, size=(n_cells, n_genes)).astype(np.int32)
    X[3, :] = 0                      # a dead cell (drops on min_genes)
    X[:, 7] = 0                      # an undetected gene (drops on low_dr)
    X[:, 11] = rng.integers(1, 5, n_cells)  # a ubiquitous gene (drops on up_dr)
    obs = pd.DataFrame(index=[f"c{i:03d}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i:03d}" for i in range(n_genes)])
    return AnnData(X=csr_matrix(X), obs=obs, var=var)


@pytest.mark.parametrize("min_genes,low_dr,up_dr", [(3, 0.05, 0.95), (1, 0.1, 0.9), (5, 0.0, 1.0)])
def test_filter_matches_scanpy_reference(min_genes, low_dr, up_dr):
    """The mask-based filter keeps exactly the cells/genes scanpy's would."""
    sc = pytest.importorskip("scanpy")
    adata = _random_raw_adata()

    got = filter_cells_and_genes(adata, min_genes=min_genes, min_cells=1,
                                 low_dr=low_dr, up_dr=up_dr)

    ref = adata.copy()
    sc.pp.filter_cells(ref, min_genes=min_genes)
    sc.pp.filter_genes(ref, min_cells=1)
    n = ref.n_obs
    dr = np.asarray((ref.X > 0).sum(axis=0)).ravel() / max(n, 1)
    ref = ref[:, (dr >= low_dr) & (dr <= up_dr)].copy()

    assert list(got.obs_names) == list(ref.obs_names)
    assert list(got.var_names) == list(ref.var_names)
    np.testing.assert_array_equal(got.X.toarray(), ref.X.toarray())
    # QC columns scanpy's filters populate must be reproduced.
    np.testing.assert_array_equal(got.obs["n_genes"].to_numpy(), ref.obs["n_genes"].to_numpy())
    np.testing.assert_array_equal(got.var["n_cells"].to_numpy(), ref.var["n_cells"].to_numpy())


def test_filter_counts_detected_not_stored():
    """Explicitly-stored zeros count as NOT detected (positive-cell counting)."""
    # 3 cells x 2 genes. gene0 is stored as explicit zeros in cells 0,1 (detected
    # in 0 cells); gene1 is positive in all 3. Storage-based counting would give
    # gene0 a detection rate of 2/3 and keep it — positive counting drops it.
    data = np.array([0.0, 5.0, 0.0, 3.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 1])
    indptr = np.array([0, 2, 4, 5])
    X = csr_matrix((data, indices, indptr), shape=(3, 2))
    adata = AnnData(X=X, obs=pd.DataFrame(index=["c0", "c1", "c2"]),
                    var=pd.DataFrame(index=["g0", "g1"]))

    out = filter_cells_and_genes(adata, min_genes=1, min_cells=1, low_dr=0.5, up_dr=1.0)

    assert list(out.var_names) == ["g1"]           # g0 dropped (detected in 0 cells)
    assert int(out.var["n_cells"].iloc[0]) == 3    # g1 detected in all 3


def test_filter_does_not_mutate_input_values():
    """filter_cells_and_genes leaves the caller's matrix values + shape unchanged."""
    adata = _random_raw_adata()
    before = adata.X.toarray().copy()
    _ = filter_cells_and_genes(adata, min_genes=3, low_dr=0.05, up_dr=0.95)
    np.testing.assert_array_equal(adata.X.toarray(), before)
    assert adata.n_obs == 80 and adata.n_vars == 50  # shape untouched


def test_normalize_does_not_mutate_input_and_preserves_metadata():
    """normalize_log1p leaves the input untouched and carries obs/var + uns flags."""
    adata = _random_raw_adata()
    adata.obs["celltype"] = "x"
    adata.var["symbol"] = adata.var_names
    before = adata.X.toarray().copy()

    out = normalize_log1p(adata)

    np.testing.assert_array_equal(adata.X.toarray(), before)   # input untouched
    assert list(out.obs_names) == list(adata.obs_names)
    assert list(out.var_names) == list(adata.var_names)
    assert "celltype" in out.obs and "symbol" in out.var       # metadata preserved
    assert out.uns["kopya_normalized"] is True
    assert issparse(out.X) and out.X.format == "csr"


def test_filter_normalize_project_wrapper_still_works():
    """The convenience wrapper (used by the external harness) still runs end to end."""
    from kopya.annotations import load_gene_order
    adata = _random_raw_adata(n_genes=50)
    # Give a handful of genes real gene-order symbols so projection keeps them.
    go = load_gene_order()
    real = list(go.index[:20])
    adata.var_names = real + [f"g{i:03d}" for i in range(len(real), adata.n_vars)]
    out = filter_normalize_project(adata, min_genes=1, low_dr=0.0, up_dr=1.0)
    assert out.n_obs > 0
    assert {"chr", "start", "end"} <= set(out.var.columns)


def test_normalize_preserves_mappings_and_is_independent():
    """normalize_log1p keeps layers/obsm/uns and the result can't corrupt the input."""
    adata = _random_raw_adata()
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 3), dtype=np.float32)
    adata.varm["loadings"] = np.zeros((adata.n_vars, 2), dtype=np.float32)
    adata.uns["note"] = {"k": 1}
    raw_before = adata.X.toarray().copy()

    out = normalize_log1p(adata)

    # Mappings preserved (regression: the lean rewrite must not drop them).
    assert "counts" in out.layers
    assert "X_pca" in out.obsm
    assert "loadings" in out.varm
    assert out.uns.get("note") == {"k": 1}

    # Independence: structurally mutating the returned matrix must not reach back
    # into the input (regression: shared indices/indptr).
    out.X.data[:] = 0.0
    out.X.eliminate_zeros()  # rewrites out.X.indices/indptr in place
    np.testing.assert_array_equal(adata.X.toarray(), raw_before)


def test_dedup_after_filter_keeps_surviving_duplicate():
    """A duplicate gene symbol whose first copy fails QC keeps its name.

    Regression for the CLI M1 ordering: dedup must happen AFTER filtering, else
    the surviving occurrence is suffixed (GENE -> GENE-1) and project_onto_genome
    drops it for having no gene-order match.
    """
    from kopya.annotations import gene_filter_mask, load_cycle_genes, load_gene_order
    go = load_gene_order()
    keepable = go.index[gene_filter_mask(var_names=go.index, gene_order=go,
                                         cycle_genes=load_cycle_genes())]
    real = keepable[0]  # a symbol guaranteed to survive projection

    n_cells = 10
    X = np.zeros((n_cells, 5), dtype=np.int32)
    X[0, 0] = 5                       # 'real' occurrence #1: detected in 1/10 cells -> fails low_dr
    X[:, 1] = np.arange(1, n_cells + 1)  # 'real' occurrence #2: detected in all -> passes
    X[:, 2:] = 1                      # fillers so cells clear min_genes
    var = pd.DataFrame(index=[real, real, "fillerA", "fillerB", "fillerC"])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)

    # CLI M1 order: filter -> dedup(owned) -> normalize -> project.
    filt = filter_cells_and_genes(adata, min_genes=1, min_cells=1, low_dr=0.5, up_dr=1.0)
    filt.var_names_make_unique()
    proj = project_onto_genome(normalize_log1p(filt), gene_order=go)
    assert real in list(proj.var_names)  # surviving 'real' retained, not suffixed away
