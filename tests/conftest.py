"""Shared pytest fixtures for kopya tests.

Provides:
- tiny_gene_order: a 10-gene synthetic gene-order DataFrame used by both
  annotations and normalize tests so neither imports the bundled 77k-row
  GENCODE table for every test (cheap, but unnecessary).
- tiny_adata: a 50-cell × 10-gene synthetic AnnData with sparse CSR counts
  whose var_names align with tiny_gene_order's index.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix


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
