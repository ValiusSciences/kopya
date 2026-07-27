"""Convert the UCSF osteosarcoma T1 scRNA-seq (IPISRC044_T1) to AnnData h5ad.

Standalone, on-demand acquisition script for the osteosarcoma gold-standard test
(tests/external/test_ipisrc044_osteosarcoma.py). It:

    1. downloads the published Seurat .rds on demand (~1 GB) if not present,
    2. exports raw counts + cell-type annotations via R/Seurat, and
    3. builds the cells x genes h5ad the test consumes, with obs['annot'].

Run (from the repo root, in an env with anndata/scipy/pandas):

    conda run -n kopya python tests/external/convert_ipisrc044_osteosarcoma.py

Prerequisites for the .rds export step:
    - curl (to fetch the .rds)
    - R with the Seurat and Matrix packages on PATH as `Rscript`
      (override with the RSCRIPT env var, e.g. a conda env's Rscript).

Output:
    DATA_ROOT/ipisrc044-osteosarcoma/ipisrc044_t1.h5ad
        cells x genes raw counts; obs['annot'] holds the paper's cell-type
        labels, including 'Putative_Tumor_Cells' (the tumor set) and the
        non-tumor immune/stromal lineages used as the supervised normal
        reference.

Data source: https://osteosarc.com/  (UCSF osteosarcoma, Tumor 1)
    .rds: https://b2.osteosarc.com/ucsf/T1/seurat_objects/
          IPISRC044_T1_scrna_live_processed_annot_101824.rds

DATA_ROOT resolves from the KOPYA_TEST_DATA environment variable, defaulting to
an `external-benchmarks/` directory at the repo root.

If you already have the derived exports (raw_counts.mtx / features.txt /
barcodes.txt / metadata.csv) from a prior R run, drop them in
DATA_ROOT/ipisrc044-osteosarcoma/exports/ and this script skips straight to the
h5ad build (no R needed).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))
OSTEO_DIR = DATA_ROOT / "ipisrc044-osteosarcoma"
RDS_NAME = "IPISRC044_T1_scrna_live_processed_annot_101824.rds"
RDS_URL = (
    "https://b2.osteosarc.com/ucsf/T1/seurat_objects/"
    "IPISRC044_T1_scrna_live_processed_annot_101824.rds"
)
OUTPUT_H5AD = OSTEO_DIR / "ipisrc044_t1.h5ad"
EXPORT_DIR = OSTEO_DIR / "exports"
TUMOR_LABEL = "Putative_Tumor_Cells"

# R script that turns the Seurat .rds into flat intermediates. Kept inline so
# this converter is a single self-contained file (like the other convert_*.py).
EXPORT_R = r"""
suppressPackageStartupMessages({library(Seurat); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
rds_path <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
seu <- readRDS(rds_path)
counts <- GetAssayData(seu, assay = "RNA", layer = "counts")
writeMM(counts, file = file.path(out_dir, "raw_counts.mtx"))       # genes x cells
writeLines(colnames(counts), file.path(out_dir, "barcodes.txt"))
writeLines(rownames(counts), file.path(out_dir, "features.txt"))
md <- seu@meta.data; md$barcode <- rownames(md)
write.csv(
  data.frame(barcode = md$barcode, annot = md$annot, stringsAsFactors = FALSE),
  file = file.path(out_dir, "metadata.csv"), row.names = FALSE
)
cat(sprintf("exported %d genes x %d cells\n", nrow(counts), ncol(counts)))
"""


def _download_rds(dest: Path) -> None:
    # A present `dest` is complete by construction: we only ever publish it via an
    # atomic rename after a clean download (below), so an existing file is never a
    # partial/errored leftover.
    if dest.exists():
        print(f"  .rds already present: {dest}")
        return
    if shutil.which("curl") is None:
        sys.exit("curl not found; install it or download the .rds manually:\n"
                 f"  curl -fL -o {dest} {RDS_URL}")
    print(f"  downloading {RDS_URL}\n    -> {dest} (~1 GB, one-time)")
    # Download to a sidecar temp path with --fail (so an HTTP 4xx/5xx body is not
    # saved as if it were the RDS), then atomically publish only on success.
    # A partial file from an interrupted or failed transfer is removed, never left
    # behind to be mistaken for a complete download on the next run.
    tmp = dest.with_name(dest.name + ".part")
    tmp.unlink(missing_ok=True)
    try:
        subprocess.run(["curl", "--fail", "-L", "-o", str(tmp), RDS_URL], check=True)
    except BaseException:  # CalledProcessError, KeyboardInterrupt, etc.
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)  # atomic on the same filesystem


def _run_r_export(rds_path: Path, out_dir: Path) -> None:
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        sys.exit(
            "Rscript not found. The .rds -> exports step needs R with Seurat + "
            "Matrix.\nInstall R/Seurat and re-run, set the RSCRIPT env var to a "
            "suitable Rscript, or drop pre-made exports (raw_counts.mtx / "
            "features.txt / barcodes.txt / metadata.csv) into\n"
            f"  {out_dir}\nand re-run to skip the R step."
        )
    script_path = out_dir.parent / "export_osteo_rds.R"
    script_path.write_text(EXPORT_R)
    # Export into a sidecar dir and publish atomically, so an interrupted R run
    # (e.g. killed after raw_counts.mtx but before metadata.csv) never leaves a
    # partial export set that a later run's completeness check would trust.
    tmp_dir = out_dir.with_name(out_dir.name + ".part")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  running R export via {rscript}")
    try:
        subprocess.run([rscript, str(script_path), str(rds_path), str(tmp_dir)], check=True)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.replace(out_dir)  # atomic publish of the complete export set


def _build_h5ad(export_dir: Path, out_path: Path) -> None:
    import anndata as ad
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp  # noqa: F401  (mmread returns COO; .tocsr below)

    mtx = export_dir / "raw_counts.mtx"
    feats = [l.strip() for l in (export_dir / "features.txt").read_text().splitlines() if l.strip()]
    bcs = [l.strip() for l in (export_dir / "barcodes.txt").read_text().splitlines() if l.strip()]
    print(f"  building h5ad from {mtx}")
    M = scipy.io.mmread(str(mtx)).tocsr()          # genes x cells
    if M.shape != (len(feats), len(bcs)):
        sys.exit(f"matrix {M.shape} does not match features({len(feats)}) x barcodes({len(bcs)})")
    X = M.T.tocsr()                                 # cells x genes
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=feats))
    meta = pd.read_csv(export_dir / "metadata.csv").set_index("barcode")
    adata.obs["annot"] = meta["annot"].reindex(adata.obs_names).values
    n_na = int(adata.obs["annot"].isna().sum())
    if n_na:
        print(f"  WARNING: {n_na} cells have no annotation label")
    adata.var_names_make_unique()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sidecar path and rename only once the file is fully written, so
    # an interrupted write never leaves a truncated h5ad that the top-level
    # existence check would treat as a complete, cached build.
    tmp_path = out_path.with_name(out_path.name + ".part")
    tmp_path.unlink(missing_ok=True)
    try:
        adata.write_h5ad(tmp_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(out_path)
    n_tumor = int((adata.obs["annot"] == TUMOR_LABEL).sum())
    n_norm = int((adata.obs["annot"].notna() & (adata.obs["annot"] != TUMOR_LABEL)).sum())
    print(f"  wrote {out_path}: {adata.shape} "
          f"({n_tumor} {TUMOR_LABEL}, {n_norm} non-tumor reference cells)")


def main():
    try:
        import anndata  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Activate the kopya environment.")

    OSTEO_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_H5AD.exists():
        print(f"h5ad already present: {OUTPUT_H5AD} (delete it to rebuild)")
        return

    # Fast path: pre-made exports already dropped in place -> skip download + R.
    have_exports = all(
        (EXPORT_DIR / f).exists()
        for f in ("raw_counts.mtx", "features.txt", "barcodes.txt", "metadata.csv")
    )
    if not have_exports:
        _download_rds(OSTEO_DIR / RDS_NAME)
        _run_r_export(OSTEO_DIR / RDS_NAME, EXPORT_DIR)
    else:
        print(f"  using existing exports in {EXPORT_DIR}")

    _build_h5ad(EXPORT_DIR, OUTPUT_H5AD)
    print("Done.")


if __name__ == "__main__":
    main()
