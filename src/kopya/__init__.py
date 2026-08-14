"""kopya — expression-only single-cell CNV caller.

Two ways to use it:

- **Scanpy-style, in memory** — run on an AnnData and get the results written
  back onto it (like scanpy / infercnvpy)::

      import scanpy as sc
      import kopya as kp

      adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")   # raw counts
      kp.tl.cnv(adata, reference_key="cell_type",
                reference_cat=["T cell", "B cell", "Macrophage"])
      # -> adata.obs["cnv_class"|"cnv_score"|...], adata.obsm["cnv_chr"], adata.uns["cnv"]
      kp.pl.chromosome_heatmap(adata, groupby="cell_type")

- **CLI** — ``kopya run --anndata ... --out-dir ...`` writes CSV/parquet/
  npz outputs and has a memory-lean path for very large cohorts.

Low-level entry points (loader, pipeline core) are re-exported here too.
"""

from kopya import pl, tl
from kopya.io import load_counts
from kopya.pipeline import run_pipeline

# Left at the last released version deliberately: this branch's changes warrant a
# MAJOR bump (tumor_score changed meaning without changing name or type — see
# CHANGELOG.md's Unreleased section), but the number is the maintainer's call and is
# set when the release is cut, not by the branch that prompts it.
__version__ = "1.0.1"

__all__ = ["tl", "pl", "load_counts", "run_pipeline", "__version__"]
