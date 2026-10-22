"""Shared pytest fixtures for kopya tests.

Provides:
- tiny_gene_order: a 10-gene synthetic gene-order DataFrame used by both
  annotations and normalize tests so neither imports the bundled 77k-row
  GENCODE table for every test (cheap, but unnecessary).
- tiny_adata: a 50-cell × 10-gene synthetic AnnData with sparse CSR counts
  whose var_names align with tiny_gene_order's index.

Also provides write_atera_store(), a plain helper (not a fixture) shared by the
loader and CLI tests.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, hstack


# Control features a released Atera matrix carries alongside its genes.
ATERA_CONTROL_TYPES = ("Negative Control Probe", "Negative Control Codeword", "Unassigned Codeword")


def write_atera_store(adata, out_dir, as_zip=True):
    """Write ``adata`` in the layout of a released 10x Atera count matrix.

    Mirrors Atera Onboard Analysis v1.0.4's cell_feature_matrix.zarr: an AnnData
    zarr store whose obs is indexed by integer position with the barcode in
    obs["barcode"], whose var holds feature_name / feature_type / feature_id,
    whose X stores int64 column indices, and whose genes are interleaved with
    control probes and codewords that a loader must drop. Cells also get an
    azimuth_broad label column, as real bundles do.

    Args:
        adata: cells x genes AnnData with raw counts and gene-symbol var_names.
        out_dir: directory to write into (created if missing).
        as_zip: write cell_feature_matrix.zarr.zip (as 10x ships it) when True,
            else leave the unzipped cell_feature_matrix.zarr directory.

    Returns:
        Path to the .zarr.zip file, or to the .zarr directory when as_zip=False.
    """
    import anndata

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cells = adata.n_obs
    rng = np.random.default_rng(7)

    # One feature of each control type, with a few counts so dropping them is visible.
    n_ctrl = len(ATERA_CONTROL_TYPES)
    ctrl = csr_matrix(rng.poisson(0.5, size=(n_cells, n_ctrl)))
    X = hstack([csr_matrix(adata.X), ctrl], format="csr")
    X.indices = X.indices.astype(np.int64)  # the released format stores int64 indices

    names = [str(g) for g in adata.var_names] + [f"CTRL_{i}" for i in range(n_ctrl)]
    var = pd.DataFrame({
        "feature_name": names,
        "feature_type": ["Gene Expression"] * adata.n_vars + list(ATERA_CONTROL_TYPES),
        "feature_id": [f"ENSG_{g}" for g in names],
    }, index=[str(i) for i in range(len(names))])
    obs = pd.DataFrame({
        "barcode": [str(b) for b in adata.obs_names],
        "cell_id": np.arange(n_cells),
        "azimuth_broad": pd.Categorical(np.where(np.arange(n_cells) % 2, "Immune cell", "Epithelial cell")),
    }, index=[str(i) for i in range(n_cells)])

    store_dir = out_dir / "cell_feature_matrix.zarr"
    if hasattr(anndata, "settings") and hasattr(anndata.settings, "zarr_write_format"):
        anndata.settings.zarr_write_format = 2  # 10x writes zarr v2
    AnnData(X=X, obs=obs, var=var).write_zarr(store_dir)
    if not as_zip:
        return store_dir
    # 10x zips the store's contents (keys at the archive root, no top folder).
    shutil.make_archive(str(out_dir / "cell_feature_matrix.zarr"), "zip", root_dir=store_dir)
    shutil.rmtree(store_dir)
    return out_dir / "cell_feature_matrix.zarr.zip"


@pytest.fixture
def tiny_gene_order():
    """Return a 10-gene synthetic gene-order DataFrame.

    Covers chr1, chr2, chr7, chrX, and chrY so tests can exercise the
    canonical-order sort and the sex-chromosome (chrX/chrY) filter without
    pulling in the bundled GENCODE table.

    Returns:
        DataFrame indexed by gene_symbol with columns [chr, start, end].
    """
    # Deliberately includes one chrX gene (G_CHRX_EARLY), one chrY gene
    # (TY_GENE), and one MT-* gene (MT-DROP) so the filter coverage is
    # end-to-end — all three are dropped for CNV calling.
    rows = [
        ("A_CHR1_EARLY",  "chr1",  100_000,  101_000),
        ("B_CHR1_MID",    "chr1",  500_000,  501_000),
        ("C_CHR1_LATE",   "chr1", 900_000,   901_000),
        ("D_CHR2_EARLY",  "chr2",  200_000,  201_000),
        ("E_CHR2_LATE",   "chr2", 800_000,   801_000),
        ("F_CHR7_MID",    "chr7",  400_000,  401_000),
        ("G_CHRX_EARLY",  "chrX",  100_000,  101_000),
        ("TY_GENE",       "chrY",  100_000,  101_000),
        ("MT-DROP",       "chr1",  600_000,  601_000),
        ("MKI67",         "chr10", 100_000,  101_000),
    ]
    df = pd.DataFrame(rows, columns=["gene_symbol", "chr", "start", "end"])
    df_indexed = df.set_index("gene_symbol")
    return df_indexed


@pytest.fixture
def tiny_adata(tiny_gene_order):
    """Return a 50-cell × 10-gene synthetic AnnData with sparse CSR counts.

    Counts are Poisson-distributed with a high enough mean (~3) that the
    per-gene detection rate clears LOW_DR=0.05 reliably for the kept genes.
    var_names match tiny_gene_order's index so projection tests don't have
    to fabricate alignments.

    Args:
        tiny_gene_order: pytest fixture for the shared gene-order frame.

    Returns:
        AnnData with sparse CSR raw counts, indexed obs/var.
    """
    # Reproducible RNG; the fixture must not silently change between runs.
    rng = np.random.default_rng(42)

    n_cells = 50
    gene_names = list(tiny_gene_order.index)
    n_genes = len(gene_names)

    # Poisson(3) so per-gene detection rate is ~0.95 — well clear of LOW_DR.
    dense = rng.poisson(lam=3.0, size=(n_cells, n_genes)).astype(np.int32)
    counts = csr_matrix(dense)

    obs = pd.DataFrame(
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        index=pd.Index(gene_names, name="gene_symbol"),
    )

    adata = AnnData(X=counts, obs=obs, var=var)
    return adata
