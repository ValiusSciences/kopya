"""Build the Maynard 2020 lung (maynard2020_3k) AnnData for the external test.

Standalone, on-demand acquisition script for the Maynard gold-standard test
(tests/external/test_maynard_lung.py). The dataset is the 3k-cell demonstration
subset used by the infercnvpy tutorial (Maynard et al., Cell 2020, SMART-seq2
lung adenocarcinoma), so acquisition goes through infercnvpy's dataset loader.

Run (from the repo root, in an env with infercnvpy + anndata):

    conda run -n kopya python tests/external/convert_maynard_lung.py

Prerequisite: `infercnvpy` (only to *fetch* the bundled dataset; the test itself
does not need it). `pip install infercnvpy`.

Output:
    DATA_ROOT/maynard-lung/maynard_lung.h5ad
        cells x genes; obs['cell_type'] holds the paper's cell-type labels,
        including 'Epithelial cell' (the malignant/observation set) and the
        immune lineages used as the supervised normal reference.

Note on values: maynard2020_3k ships **log-normalized** expression (SMART-seq2,
no raw UMIs). kopya applies its own CP10k+log1p, so we store expm1(X) —
the de-logged normalized expression — as pseudo-counts. The pipeline re-
normalizes them; the relative per-gene/per-cell signal (all a CNV caller needs)
is preserved. This mirrors how the infercnvpy tutorial feeds the same values to
inferCNV, so the two tools operate on the same underlying expression.
"""

import os
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))
OUT_DIR = DATA_ROOT / "maynard-lung"
OUTPUT_H5AD = OUT_DIR / "maynard_lung.h5ad"
TUMOR_LABEL = "Epithelial cell"


def main():
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Activate the kopya environment.")
    try:
        import infercnvpy as cnv
    except ImportError:
        sys.exit(
            "infercnvpy is required to fetch maynard2020_3k (the test itself does "
            "not need it). Install with: pip install infercnvpy"
        )

    if OUTPUT_H5AD.exists():
        print(f"h5ad already present: {OUTPUT_H5AD} (delete it to rebuild)")
        return

    print("fetching maynard2020_3k via infercnvpy.datasets ...")
    src = cnv.datasets.maynard2020_3k()
    X = src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X, dtype=np.float32)
    pseudo = np.expm1(np.asarray(X, dtype=np.float32)).clip(min=0)   # de-log -> pseudo-counts

    adata = ad.AnnData(
        X=sp.csr_matrix(pseudo),
        obs=pd.DataFrame({"cell_type": src.obs["cell_type"].astype(str).values},
                         index=src.obs_names.astype(str)),
        var=pd.DataFrame(index=src.var_names.astype(str)),
    )
    adata.var_names_make_unique()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_H5AD.with_name(OUTPUT_H5AD.name + ".part")
    tmp.unlink(missing_ok=True)
    try:
        adata.write_h5ad(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(OUTPUT_H5AD)

    n_tumor = int((adata.obs["cell_type"] == TUMOR_LABEL).sum())
    print(f"wrote {OUTPUT_H5AD}: {adata.shape} "
          f"({n_tumor} {TUMOR_LABEL}, {adata.n_obs - n_tumor} other/reference cells)")


if __name__ == "__main__":
    main()
