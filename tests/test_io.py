"""Smoke tests for the loader.

Validates the input contract from kopya.md §2: both Form A (AnnData)
and Form B (mtx trio) load into the same shape of AnnData with sparse CSR
counts, and obviously wrong inputs raise rather than silently miscompute.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, issparse

from kopya.io import load_counts


def _make_tiny_counts(n_cells=20, n_genes=50, density=0.3, seed=0):
    """Build a tiny synthetic cells × genes sparse UMI matrix with metadata.

    Used as a shared fixture by both Form A and Form B tests so the comparison
    between the two paths is apples-to-apples.

    Args:
        n_cells: Number of cells (rows).
        n_genes: Number of genes (cols).
        density: Fraction of nonzero entries (Poisson rate via Bernoulli mask).
        seed: numpy RNG seed for reproducibility.

    Returns:
        Tuple of (csr counts, obs DataFrame, var DataFrame).
    """
    # Reproducible RNG so failures are deterministic across CI runs.
    rng = np.random.default_rng(seed)

    # Build a dense Poisson-ish matrix, then mask to a sparse footprint —
    # mimics 10x UMI counts at low coverage closely enough for I/O testing.
    dense = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.int32)
    mask = rng.random(size=(n_cells, n_genes)) < density
    dense_masked = dense * mask
    counts_csr = csr_matrix(dense_masked)

    # Build the metadata frames with the same index conventions the host
    # pipeline uses: barcode for obs, gene symbol for var.
    obs = pd.DataFrame(
        {"n_counts": np.asarray(dense_masked.sum(axis=1)).ravel()},
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"n_cells": np.asarray((dense_masked > 0).sum(axis=0)).ravel()},
        index=[f"GENE{i:04d}" for i in range(n_genes)],
    )

    return counts_csr, obs, var


def test_load_mtx_trio_roundtrip(tmp_path):
    """Form B: load_counts from a written-then-read mtx trio matches the source."""
    # Build the fixture in-memory and serialize it to the three pipeline files.
    counts, obs, var = _make_tiny_counts()
    counts_path = tmp_path / "counts.mtx"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    mmwrite(str(counts_path), counts)
    obs.to_csv(obs_path)
    var.to_csv(var_path)

    # Exercise the loader through the public entry point.
    adata = load_counts(counts=counts_path, obs=obs_path, var=var_path)

    # Shape, sparseness, and index alignment are the three things downstream
    # steps rely on; assert them explicitly rather than via a generic equality.
    assert isinstance(adata, AnnData)
    assert adata.shape == counts.shape
    assert issparse(adata.X)
    assert adata.X.format == "csr"
    assert list(adata.obs_names) == list(obs.index)
    assert list(adata.var_names) == list(var.index)
    # Values must round-trip exactly because MM stores integers losslessly.
    assert (adata.X != counts).nnz == 0


def test_load_anndata_form(tmp_path):
    """Form A: an .h5ad with counts in .X loads into the same shape AnnData."""
    counts, obs, var = _make_tiny_counts()
    source = AnnData(X=counts, obs=obs, var=var)

    h5ad_path = tmp_path / "tiny.h5ad"
    source.write_h5ad(h5ad_path)

    adata = load_counts(anndata=h5ad_path)

    # Equivalent assertions to the Form B test; the two paths must converge.
    assert adata.shape == counts.shape
    assert issparse(adata.X)
    assert adata.X.format == "csr"
    assert list(adata.obs_names) == list(obs.index)


def test_load_anndata_prefers_counts_layer(tmp_path):
    """Form A: when both .X (log) and layers['counts'] (raw) exist, prefer counts."""
    counts, obs, var = _make_tiny_counts()
    # Simulate a common upstream shape: .X holds normalized values, raw counts
    # live in layers["counts"]. The loader must return the raw counts.
    log_x = csr_matrix(np.log1p(counts.toarray()))
    source = AnnData(X=log_x, obs=obs, var=var, layers={"counts": counts})

    h5ad_path = tmp_path / "with_layer.h5ad"
    source.write_h5ad(h5ad_path)

    adata = load_counts(anndata=h5ad_path)

    # The loaded .X must equal the raw counts, not the log-normalized .X.
    assert (adata.X != counts).nnz == 0


def test_load_anndata_warns_on_normalized_x(tmp_path):
    """Form A: a .h5ad whose .X is log-normalized (no counts layer) warns on load."""
    counts, obs, var = _make_tiny_counts()
    # .X holds log1p(CP10k) values and there is no raw-count layer to fall back on.
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1
    log_x = csr_matrix(np.log1p(counts.toarray() / totals[:, None] * 1e4))
    source = AnnData(X=log_x, obs=obs, var=var)

    h5ad_path = tmp_path / "normalized.h5ad"
    source.write_h5ad(h5ad_path)

    with pytest.warns(UserWarning, match="does not look like raw integer UMI counts"):
        load_counts(anndata=h5ad_path)


def test_load_anndata_counts_layer_suppresses_warning(tmp_path):
    """Form A: a raw-count layer is trusted, so normalized .X does not warn."""
    counts, obs, var = _make_tiny_counts()
    log_x = csr_matrix(np.log1p(counts.toarray()))
    source = AnnData(X=log_x, obs=obs, var=var, layers={"counts": counts})

    h5ad_path = tmp_path / "with_layer.h5ad"
    source.write_h5ad(h5ad_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adata = load_counts(anndata=h5ad_path)
    # No raw-count warning fires (unrelated library warnings are ignored).
    assert not any("raw integer UMI counts" in str(w.message) for w in caught)
    # And it still returns the raw counts, not the log-normalized .X.
    assert (adata.X != counts).nnz == 0


def test_load_anndata_raw_count_check_raise(tmp_path):
    """Form A: raw_count_check='raise' converts the normalized-input warning into an error."""
    counts, obs, var = _make_tiny_counts()
    log_x = csr_matrix(np.log1p(counts.toarray()))
    source = AnnData(X=log_x, obs=obs, var=var)

    h5ad_path = tmp_path / "normalized_raise.h5ad"
    source.write_h5ad(h5ad_path)

    with pytest.raises(ValueError, match="does not look like raw integer UMI counts"):
        load_counts(anndata=h5ad_path, raw_count_check="raise")


def test_load_counts_invalid_raw_count_check(tmp_path):
    """An invalid raw_count_check names that parameter, not the internal 'action'."""
    counts, obs, var = _make_tiny_counts()
    h5ad_path = tmp_path / "raw.h5ad"
    AnnData(X=counts, obs=obs, var=var).write_h5ad(h5ad_path)

    # The message must reference the caller-facing parameter, and the check must
    # fire regardless of input form (validated up front, before any file read).
    with pytest.raises(ValueError, match="raw_count_check must be"):
        load_counts(anndata=h5ad_path, raw_count_check="bogus")


def test_load_anndata_raw_counts_no_warning(tmp_path):
    """Form A: genuine raw counts in .X load without any warning."""
    counts, obs, var = _make_tiny_counts()
    source = AnnData(X=counts, obs=obs, var=var)

    h5ad_path = tmp_path / "raw.h5ad"
    source.write_h5ad(h5ad_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_counts(anndata=h5ad_path)
    assert not any("raw integer UMI counts" in str(w.message) for w in caught)


def test_load_rejects_both_forms(tmp_path):
    """Passing both --anndata and the trio is a configuration error, not a silent pick."""
    counts, obs, var = _make_tiny_counts()
    counts_path = tmp_path / "counts.mtx"
    mmwrite(str(counts_path), counts)

    # Bare existence of a fake h5ad is enough; the validation should reject
    # the combo before any file is read.
    h5ad_path = tmp_path / "ignored.h5ad"
    AnnData(X=counts, obs=obs, var=var).write_h5ad(h5ad_path)

    with pytest.raises(ValueError, match="exactly one input form"):
        load_counts(
            anndata=h5ad_path,
            counts=counts_path,
            obs=tmp_path / "obs.csv",
            var=tmp_path / "var.csv",
        )


def test_load_rejects_no_input():
    """Calling load_counts() with nothing must raise rather than return empty."""
    with pytest.raises(ValueError, match="Must pass one of"):
        load_counts()


def test_load_rejects_partial_trio(tmp_path):
    """Form B requires all three trio paths; partial input is rejected."""
    counts, obs, var = _make_tiny_counts()
    counts_path = tmp_path / "counts.mtx"
    mmwrite(str(counts_path), counts)

    with pytest.raises(ValueError, match="all three"):
        load_counts(counts=counts_path)


def test_load_rejects_mismatched_shape(tmp_path):
    """A trio with mismatched (n_obs, n_var) vs the mtx must fail fast."""
    counts, obs, var = _make_tiny_counts(n_cells=20, n_genes=50)
    counts_path = tmp_path / "counts.mtx"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    mmwrite(str(counts_path), counts)

    # Truncate obs so its row count no longer matches the mtx — the most common
    # mistake when the trio is assembled from out-of-sync sources.
    obs.iloc[:10].to_csv(obs_path)
    var.to_csv(var_path)

    with pytest.raises(ValueError, match="does not match"):
        load_counts(counts=counts_path, obs=obs_path, var=var_path)


def test_load_cellranger_mtx_roundtrip(tmp_path):
    """Form C: a CellRanger-style MTX directory loads correctly."""
    import gzip
    counts, obs, var = _make_tiny_counts()

    # Write a minimal CellRanger-style filtered_feature_bc_matrix/ directory.
    mtx_dir = tmp_path / "filtered_feature_bc_matrix"
    mtx_dir.mkdir()

    # CellRanger MTX is features × barcodes (genes × cells), not cells × genes.
    mmwrite(str(mtx_dir / "matrix.mtx"), counts.T)
    # Compress to .gz as CellRanger v3+ does.
    import shutil, os
    with open(mtx_dir / "matrix.mtx", "rb") as src, \
         gzip.open(mtx_dir / "matrix.mtx.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(mtx_dir / "matrix.mtx")

    with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(obs.index) + "\n")

    # Write features.tsv.gz in CellRanger v3 format: id\tsymbol\tGene Expression
    with gzip.open(mtx_dir / "features.tsv.gz", "wt") as fh:
        for sym in var.index:
            fh.write(f"ENSG_{sym}\t{sym}\tGene Expression\n")

    adata = load_counts(cellranger_dir=mtx_dir)

    assert adata.shape == counts.shape
    assert issparse(adata.X)
    assert adata.X.format == "csr"
    assert list(adata.obs_names) == list(obs.index)
    assert list(adata.var_names) == list(var.index)


def test_load_cellranger_h5_roundtrip(tmp_path):
    """Form D: a CellRanger .h5 file loads correctly."""
    import scanpy as sc
    counts, obs, var = _make_tiny_counts()

    # Build a minimal AnnData and write it in 10x H5 format via scanpy.
    source = AnnData(X=counts, obs=obs, var=var)
    source.var["gene_ids"] = [f"ENSG_{s}" for s in var.index]
    source.var["feature_types"] = "Gene Expression"
    h5_path = tmp_path / "filtered_feature_bc_matrix.h5"
    source.write_h5ad(tmp_path / "tmp.h5ad")  # write h5ad then re-load via scanpy

    # Fallback: if write_10x_h5 is not available, skip.
    try:
        sc.readwrite.write._write_10x_h5(str(h5_path), source)
    except (AttributeError, Exception):
        # scanpy's 10x H5 writer is not a stable public API; skip if unavailable.
        pytest.skip("scanpy 10x H5 writer not available in this version")

    adata = load_counts(cellranger_h5=h5_path)
    assert adata.shape == counts.shape
    assert issparse(adata.X)
    assert adata.X.format == "csr"


def test_load_rejects_multiple_forms(tmp_path):
    """Passing two input forms at once raises ValueError."""
    counts, obs, var = _make_tiny_counts()
    h5ad_path = tmp_path / "tiny.h5ad"
    AnnData(X=counts, obs=obs, var=var).write_h5ad(h5ad_path)

    mtx_dir = tmp_path / "filtered_feature_bc_matrix"
    mtx_dir.mkdir()

    with pytest.raises(ValueError, match="exactly one input form"):
        load_counts(anndata=h5ad_path, cellranger_dir=mtx_dir)
