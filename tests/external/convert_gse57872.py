"""Convert the Patel 2014 GBM expression matrix (GSE57872) to AnnData h5ad.

Standalone script — run directly with the kopya conda environment:

    conda run -n kopya python tests/external/convert_gse57872.py

Expected input:
    $KOPYA_TEST_DATA/gse57872/GBM_data_matrix.txt.gz
    (default: <repo-root>/external-benchmarks/gse57872/GBM_data_matrix.txt.gz)

Output:
    $KOPYA_TEST_DATA/gse57872/patel_gbm.h5ad
    (default: <repo-root>/external-benchmarks/gse57872/patel_gbm.h5ad)

The raw file is a tab-delimited matrix:
    - Rows: genes (HGNC symbols in the first column, labelled "Gene")
    - Columns: cells (header row contains cell barcodes)
    - Values: FPKM expression (float)

Preprocessing applied here:
    1. Log1p transformation: log1p(FPKM).  This approximates log1p(CP10k) close
       enough for the inferCNV-style pipeline; the pipeline's own normalize step
       will re-apply library-size normalization on the raw counts, but Patel FPKM
       is close enough at this stage.
    2. The gene index is stored as var_names; cell barcodes as obs_names.
    3. The matrix is stored as a dense float32 array (FPKM data are not sparse).

Cell-type annotations from the paper (malignant / non-malignant) are encoded in
the cell barcode suffix:
    MGH26, MGH28, MGH29, MGH30, MGH31 — glioblastoma patient IDs
    Non-malignant cell types contain suffixes like "_Microglia", "_Oligodendrocyte",
    "_Endothelial", "_Astrocyte" in the original GEO data.
We store the raw barcode as obs_names without parsing cell type here; the
pipeline's baseline picker uses its own marker-based detection.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))
INPUT_GZ = DATA_ROOT / "gse57872" / "GBM_data_matrix.txt.gz"
OUTPUT_H5AD = DATA_ROOT / "gse57872" / "patel_gbm.h5ad"


def main():
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}.  Activate the kopya environment.")

    if not INPUT_GZ.exists():
        sys.exit(
            f"Input file not found: {INPUT_GZ}\n"
            "Download with:\n"
            f"  mkdir -p {DATA_ROOT}/gse57872\n"
            f"  curl -L -o {DATA_ROOT}/gse57872/"
            "GBM_data_matrix.txt.gz \\\n"
            '    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57872/suppl/'
            'GSE57872_GBM_data_matrix.txt.gz"'
        )

    print(f"Reading {INPUT_GZ} ...")
    df = pd.read_csv(INPUT_GZ, sep="\t", index_col=0, compression="gzip")
    print(f"  Raw shape: {df.shape[0]} genes × {df.shape[1]} cells")

    # The first column in some GEO releases is labelled "Gene" and contains
    # HGNC symbols; when read with index_col=0 the index becomes the gene names.
    # Drop any all-NaN columns that can appear at the end of the file.
    df = df.dropna(axis=1, how="all")
    # Remove any duplicate gene rows (keep first occurrence).
    df = df[~df.index.duplicated(keep="first")]

    # Transpose to (cells × genes) and apply log1p.
    X = df.values.T.astype("float32")
    X = np.log1p(X)

    obs = pd.DataFrame(index=df.columns.astype(str))
    var = pd.DataFrame(index=df.index.astype(str))
    var.index.name = "gene_name"

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=obs,
        var=var,
    )
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    OUTPUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"  Written to {OUTPUT_H5AD}")


if __name__ == "__main__":
    main()
