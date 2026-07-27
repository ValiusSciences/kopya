"""I/O for kopya.

Four accepted entry forms:
    A. AnnData h5ad  — preferred; handles upstream single-cell pipeline
                       outputs and any tool that writes h5ad (e.g. STARsolo,
                       alevin-fry).
    B. Matrix trio   — counts.mtx + obs.csv + var.csv, a matrix-trio format
                       produced by common single-cell pipelines.
    C. CellRanger directory — path to filtered_feature_bc_matrix/ (or
                       raw_feature_bc_matrix/), the standard output of
                       10x Genomics CellRanger, STARsolo, and most droplet
                       sequencing pipelines.
    D. CellRanger H5 — path to filtered_feature_bc_matrix.h5, the single-file
                       HDF5 alternative to the MTX directory from CellRanger.

All four produce an AnnData with .X = sparse CSR raw UMI counts, .obs indexed
by cell barcode, .var indexed by gene symbol.

Note: the Loupe Browser .cloupe file is not supported — it is a proprietary
binary format with no public parser.  Use the .h5 or MTX output from the same
CellRanger run instead.
"""

from pathlib import Path

from anndata import AnnData, read_h5ad
from pandas import read_csv
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
import scanpy as sc

from kopya.normalize import validate_raw_counts


def _load_anndata(anndata_path, raw_count_check="warn"):
    """Load an .h5ad file and coerce its X to sparse CSR raw counts.

    Prefers the "counts" layer when present (upstream pipelines often store
    raw counts there and may write log-normalized values into .X). Falls back
    to .X when no counts layer exists.

    When falling back to .X (no counts layer), the matrix is validated to look
    like raw integer UMI counts. h5ad files that carry normalized or
    log-transformed values in .X would otherwise be double-normalized silently,
    so validate_raw_counts flags them per ``raw_count_check``.

    Args:
        anndata_path: Path to a .h5ad file written by an upstream pipeline.
        raw_count_check: "warn" (default), "raise", or "ignore" — how to react
            when .X does not look like raw counts and no counts layer exists.

    Returns:
        An AnnData with .X as a sparse CSR matrix of raw UMI counts.
    """
    # Read the AnnData directly; this is cheap because we touch the small
    # metadata frames eagerly but matrices are only fully realized on access.
    adata = read_h5ad(anndata_path)

    # Validate before rebuilding the AnnData below (which drops layers): the
    # layer-aware check trusts layers["counts"] and only inspects .X otherwise.
    validate_raw_counts(adata, action=raw_count_check)

    # Prefer the raw-counts layer; upstream pipelines commonly store raw counts there.
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
    else:
        counts = adata.X

    # Coerce to sparse CSR — every downstream step assumes that exact layout.
    if not issparse(counts):
        counts_sparse = csr_matrix(counts)
    else:
        counts_sparse = counts.tocsr()

    # Rebuild a clean AnnData carrying only what we need; drops unrelated
    # layers/obsm/uns that would otherwise bloat memory through the run.
    out = AnnData(X=counts_sparse, obs=adata.obs.copy(), var=adata.var.copy())
    return out


def _load_mtx_trio(counts_path, obs_path, var_path):
    """Load an (counts.mtx, obs.csv, var.csv) trio.

    Expects counts as a cells × genes Matrix Market file alongside
    obs.csv (cell metadata, first column = barcode) and var.csv (gene metadata,
    first column = symbol). We preserve that orientation in AnnData.

    Args:
        counts_path: Path to counts.mtx (cells × genes sparse).
        obs_path: Path to obs.csv (row index = barcode).
        var_path: Path to var.csv (row index = gene symbol).

    Returns:
        An AnnData with .X = sparse CSR raw counts, indexed by barcode/symbol.
    """
    # mmread returns a COO sparse matrix; convert to CSR for row-slicing speed.
    counts_coo = mmread(str(counts_path))
    counts_csr = counts_coo.tocsr()

    # Both metadata frames use the first column as the index (barcode / symbol).
    obs = read_csv(obs_path, index_col=0)
    var = read_csv(var_path, index_col=0)

    # Sanity-check the orientation early — a swapped trio is the most common
    # pipeline-glue mistake and produces silently wrong results downstream.
    expected_shape = (len(obs), len(var))
    if counts_csr.shape != expected_shape:
        msg = (
            f"counts shape {counts_csr.shape} does not match "
            f"(n_obs={len(obs)}, n_var={len(var)}). "
            "The mtx trio is expected to be cells × genes."
        )
        raise ValueError(msg)

    # Assemble the AnnData with explicit obs/var indices (barcode/symbol).
    adata = AnnData(X=counts_csr, obs=obs, var=var)
    return adata


def _load_10x_mtx(mtx_dir):
    """Load a CellRanger (or STARsolo / alevin-fry) MTX directory.

    Accepts the standard filtered_feature_bc_matrix/ layout produced by
    CellRanger v2+ (genes.tsv) and v3+ (features.tsv), STARsolo, and most
    other droplet-sequencing pipelines that follow the 10x sparse format.

    Args:
        mtx_dir: Path to the directory containing matrix.mtx.gz (or .mtx),
            barcodes.tsv.gz (or .tsv), and features.tsv.gz / genes.tsv.gz.

    Returns:
        AnnData with .X = sparse CSR raw UMI counts, var_names = gene symbols.
    """
    adata = sc.read_10x_mtx(
        str(mtx_dir),
        var_names="gene_symbols",
        make_unique=True,
        gex_only=True,
    )
    # Ensure CSR layout — scanpy may return CSC on some paths.
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    return adata


def _load_10x_h5(h5_path):
    """Load a CellRanger filtered_feature_bc_matrix.h5 file.

    Args:
        h5_path: Path to the .h5 file written by CellRanger (v2 or v3+).

    Returns:
        AnnData with .X = sparse CSR raw UMI counts, var_names = gene symbols.
    """
    adata = sc.read_10x_h5(str(h5_path), gex_only=True)
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    return adata


def load_counts(anndata=None, counts=None, obs=None, var=None,
                cellranger_dir=None, cellranger_h5=None,
                raw_count_check="warn"):
    """Load raw single-cell UMI counts into an AnnData.

    Exactly one of the two input forms must be supplied:
        - anndata: path to a .h5ad file (Form A).
        - counts + obs + var: paths to the mtx-trio (Form B).

    Args:
        anndata: Path to .h5ad (Form A); mutually exclusive with the trio.
        counts: Path to counts.mtx (Form B).
        obs: Path to obs.csv (Form B).
        var: Path to var.csv (Form B).
        raw_count_check: How to react when a Form-A .h5ad supplies .X that does
            not look like raw integer UMI counts and has no layers["counts"].
            "warn" (default) emits a UserWarning, "raise" errors out (opt-in
            strict mode), "ignore" skips the check. The mtx-trio and CellRanger
            forms always provide raw counts by construction, so the check does
            not apply to them.

    Returns:
        An AnnData with .X as sparse CSR raw counts.

    Raises:
        ValueError: if zero or more than one input form is supplied, if the
            mtx trio's shape disagrees with the metadata frames, or if
            raw_count_check="raise" and non-raw data is detected.
    """
    # Validate here so an invalid value is reported against the parameter the
    # caller actually passed (raw_count_check), rather than surfacing later as
    # an error about validate_raw_counts's internal `action` parameter.
    if raw_count_check not in ("warn", "raise", "ignore"):
        raise ValueError(
            "raw_count_check must be 'warn', 'raise', or 'ignore', "
            f"got {raw_count_check!r}."
        )

    form_a = anndata is not None
    form_b = counts is not None or obs is not None or var is not None
    form_c = cellranger_dir is not None
    form_d = cellranger_h5 is not None

    n_forms = sum([form_a, form_b, form_c, form_d])
    if n_forms > 1:
        raise ValueError(
            "Pass exactly one input form: --anndata, (--counts/--obs/--var), "
            "--cellranger-dir, or --cellranger-h5."
        )
    if n_forms == 0:
        raise ValueError(
            "Must pass one of: --anndata, (--counts, --obs, --var), "
            "--cellranger-dir, or --cellranger-h5."
        )

    if form_b and (counts is None or obs is None or var is None):
        raise ValueError(
            "Form B requires all three of --counts, --obs, --var."
        )

    if form_a:
        return _load_anndata(Path(anndata), raw_count_check=raw_count_check)
    if form_b:
        return _load_mtx_trio(Path(counts), Path(obs), Path(var))
    if form_c:
        return _load_10x_mtx(Path(cellranger_dir))
    return _load_10x_h5(Path(cellranger_h5))
