"""Convert SCEVAN synthetic count matrices (Zenodo 6628423) to AnnData h5ad.

Standalone script — run directly with the kopya conda environment:

    conda run -n kopya python tests/external/convert_scevan.py

Expected input layout:
    $KOPYA_TEST_DATA/scevan-synthetic/raw/
    (default: <repo-root>/external-benchmarks/scevan-synthetic/raw/)
        synthetic_tumor_01/
            count_matrix.txt   (genes × cells, tab-separated, with header)
            metadata.txt       (cell barcode + ground-truth labels)
        synthetic_tumor_02/
            ...
        ... (up to ~30 matrices in the Zenodo archive)

Alternatively, the script also accepts a flat directory of .txt/.tsv/.csv count
matrices with no subdirectory structure — it will attempt to pair each matrix
with a same-name metadata file, or fall back to positional label inference.

Output layout:
    $KOPYA_TEST_DATA/scevan-synthetic/h5ad/
    (default: <repo-root>/external-benchmarks/scevan-synthetic/h5ad/)
        synthetic_tumor_01.h5ad
        synthetic_tumor_02.h5ad
        ...

Ground-truth label encoding:
    SCEVAN synthetic matrices encode cell identity in the metadata file.  The
    standard column names used by the SCEVAN R package are:
        "is_tumour"  (logical/boolean 1/0)  ← primary target
        "class"      ("Tumor" / "Normal")
    Both are preserved in adata.obs when present so the test harness can
    pick them up via _infer_tumor_barcodes().

    When no metadata file is found, a positional 50/50 split is used
    (bottom half = normal, top half = tumor) consistent with the SCEVAN
    synthetic generation protocol.

Count matrix format:
    Rows = genes (HGNC symbols), columns = cells.
    Values are raw UMI counts (integer) or normalised counts (float).
    Library-size normalisation and log1p are applied here; the pipeline's
    own filter_normalize_project() step will re-normalise, but a clean
    starting representation is preferable.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))
RAW_DIR = DATA_ROOT / "scevan-synthetic" / "raw"
OUTPUT_DIR = DATA_ROOT / "scevan-synthetic" / "h5ad"


def _read_count_matrix(count_path: Path):
    """Read a genes × cells count matrix, return (df: cells × genes)."""
    import pandas as pd

    sep = "\t"
    if count_path.suffix.lower() == ".csv":
        sep = ","

    df = pd.read_csv(count_path, sep=sep, index_col=0)
    df = df.dropna(axis=1, how="all")
    df = df[~df.index.duplicated(keep="first")]
    # Transpose to cells × genes.
    return df.T


def _read_metadata(meta_path: Path):
    """Read a metadata TSV/CSV file, return as DataFrame indexed by barcode."""
    import pandas as pd

    if not meta_path.exists():
        return None
    sep = "\t"
    if meta_path.suffix.lower() == ".csv":
        sep = ","
    meta = pd.read_csv(meta_path, sep=sep, index_col=0)
    return meta


def _positional_labels(n_cells: int):
    """Assign positional 50/50 tumor/normal labels.

    SCEVAN synthetic matrices are generated with the normal cells in the first
    half and tumor cells in the second half, by convention.

    Returns a Series with barcode index (positional strings) and values
    1 (tumor) or 0 (normal) in column 'is_tumour'.
    """
    import pandas as pd
    import numpy as np

    split = n_cells // 2
    labels = np.zeros(n_cells, dtype=int)
    labels[split:] = 1
    return pd.Series(labels, name="is_tumour")


def _convert_one(count_path: Path, meta_path: Path, output_h5ad: Path):
    """Convert a single count matrix to h5ad.

    Args:
        count_path: Path to the count matrix file (genes × cells).
        meta_path: Path to the paired metadata file, or None.
        output_h5ad: Destination path.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    cells_df = _read_count_matrix(count_path)
    n_cells, n_genes = cells_df.shape

    # Log1p normalisation (library-size normalise to 1e4 first).
    X_raw = cells_df.values.astype("float32")
    lib_size = X_raw.sum(axis=1, keepdims=True)
    lib_size = np.where(lib_size == 0, 1.0, lib_size)  # avoid divide-by-zero
    X_norm = np.log1p(X_raw / lib_size * 1e4)

    obs = pd.DataFrame(index=cells_df.index.astype(str))
    var = pd.DataFrame(index=cells_df.columns.astype(str))

    # Attach metadata to obs if available.
    meta = _read_metadata(meta_path) if meta_path is not None else None
    if meta is not None:
        # Align on barcode index.
        common = obs.index.intersection(meta.index)
        if len(common) > 0:
            for col in meta.columns:
                obs[col] = meta[col].reindex(obs.index)
        else:
            # Metadata index may be positional integers — try aligning by
            # position if lengths match.
            if len(meta) == n_cells:
                for col in meta.columns:
                    obs[col] = meta[col].values
    else:
        # Positional fallback.
        obs["is_tumour"] = _positional_labels(n_cells).values

    adata = ad.AnnData(
        X=sp.csr_matrix(X_norm),
        obs=obs,
        var=var,
    )
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    return n_cells, n_genes


def main():
    try:
        import anndata  # noqa: F401
        import scipy.sparse  # noqa: F401
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}.  Activate the kopya environment.")

    # Discover input matrices.  Accept either subdirectory-per-sample layout or
    # flat directory of .txt/.tsv/.csv files.
    input_files = []  # list of (count_path, meta_path_or_None, stem)

    if RAW_DIR.is_dir():
        # Check subdirectory layout first.
        subdirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
        for sd in subdirs:
            for fname in ("count_matrix.txt", "count_matrix.tsv", "counts.txt",
                          "counts.tsv", "matrix.txt", "expression.txt"):
                cp = sd / fname
                if cp.exists():
                    meta = None
                    for mfname in ("metadata.txt", "metadata.tsv", "metadata.csv",
                                   "labels.txt", "labels.tsv"):
                        mp = sd / mfname
                        if mp.exists():
                            meta = mp
                            break
                    input_files.append((cp, meta, sd.name))
                    break

        # Flat layout fallback.
        if not input_files:
            for cp in sorted(RAW_DIR.glob("*.txt")) + sorted(RAW_DIR.glob("*.tsv")):
                stem = cp.stem
                meta = None
                for suffix in ("_metadata.txt", "_metadata.tsv", "_labels.txt"):
                    mp = cp.with_name(stem + suffix)
                    if mp.exists():
                        meta = mp
                        break
                input_files.append((cp, meta, stem))

    if not input_files:
        sys.exit(
            f"No count matrices found under {RAW_DIR}\n"
            "Download the SCEVAN synthetic benchmark from Zenodo (record 6628423):\n"
            f"  mkdir -p {RAW_DIR}\n"
            f"  cd {RAW_DIR}\n"
            "  curl -L -O https://zenodo.org/record/6628423/files/"
            "synthetic_benchmark_matrices.tar.gz\n"
            "  tar xzf synthetic_benchmark_matrices.tar.gz"
        )

    print(f"Found {len(input_files)} count matrices under {RAW_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converted = 0
    for count_path, meta_path, stem in input_files:
        out = OUTPUT_DIR / f"{stem}.h5ad"
        meta_note = f"+ {meta_path.name}" if meta_path else "(positional labels)"
        print(f"  {stem}: {count_path.name} {meta_note} -> {out.name} ...", end=" ")
        try:
            n_cells, n_genes = _convert_one(count_path, meta_path, out)
            print(f"{n_cells} cells, {n_genes} genes")
            converted += 1
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\nDone. Converted {converted}/{len(input_files)} matrices to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
