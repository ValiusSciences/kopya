"""I/O for kopya.

Five accepted entry forms:
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
                       Xenium and the Atera WTA preview datasets write the same
                       layout as cell_feature_matrix.h5.
    E. Atera bundle  — path to a 10x Atera Onboard Analysis (v1.0+) output
                       directory or its cell_feature_matrix.zarr.zip. Released
                       Atera bundles ship the count matrix only as an AnnData
                       zarr store, with no .h5.

All five produce an AnnData with .X = sparse CSR raw UMI counts, .obs indexed
by cell barcode, .var indexed by gene symbol.

Note: the Loupe Browser .cloupe file is not supported — it is a proprietary
binary format with no public parser.  Use the .h5 or MTX output from the same
CellRanger run instead.
"""

from pathlib import Path

from anndata import AnnData, read_h5ad
from numpy import empty, int32, iinfo
from pandas import read_csv
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
import scanpy as sc

from kopya.normalize import validate_raw_counts

# File name of the count matrix inside an Atera Onboard Analysis output bundle.
ATERA_MATRIX_ZIP = "cell_feature_matrix.zarr.zip"

# Where the matrix can sit inside a bundle directory, in order of preference:
# the zip as 10x ships it, then a store someone unzipped, each at the bundle
# root or under outs/. Shared with find-inputs so the two never disagree.
ATERA_MATRIX_CANDIDATES = (
    ATERA_MATRIX_ZIP, f"outs/{ATERA_MATRIX_ZIP}",
    "cell_feature_matrix.zarr", "outs/cell_feature_matrix.zarr",
)

# Elements an Atera matrix store must carry for _load_atera to read it. Checked
# up front so a future format change fails with the missing names rather than
# a KeyError from deep inside the read.
ATERA_REQUIRED_ELEMENTS = (
    "X/data", "X/indices", "X/indptr",
    "obs/barcode", "var/feature_name", "var/feature_type",
)


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


def _resolve_atera_path(path):
    """Return the Atera matrix store inside ``path``.

    Accepts the cell_feature_matrix.zarr.zip itself, an unzipped .zarr
    directory, or an output bundle directory that holds either one (directly
    or under outs/). The zip is preferred when both are present.

    Args:
        path: Path given on the command line or to load_counts.

    Returns:
        Path to a .zarr.zip file or a .zarr directory.

    Raises:
        FileNotFoundError: if ``path`` is a directory with no matrix store in it.
    """
    if path.is_file() or path.suffix == ".zarr":
        return path
    for relative in ATERA_MATRIX_CANDIDATES:
        candidate = path / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {ATERA_MATRIX_ZIP} (or unzipped cell_feature_matrix.zarr) found in "
        f"{path} or {path / 'outs'}. Pass the Atera output directory or the matrix "
        "itself."
    )


def _load_atera(atera_path):
    """Load the count matrix from a 10x Atera Onboard Analysis output bundle.

    The released Atera format stores the cells x features matrix as an AnnData
    zarr store (cell_feature_matrix.zarr.zip): CSR counts under X, cell
    barcodes in obs["barcode"], gene symbols in var["feature_name"], and control
    probes / codewords alongside the genes, told apart by var["feature_type"].

    Mirrors the CellRanger loaders: only "Gene Expression" features are kept,
    var_names are gene symbols, obs_names are barcodes. The rest of obs (10x's
    Azimuth cell-type labels, cell area, centroids) is kept so callers can use
    it to pick a reference or check calls.

    The store keeps column indices as int64. They are read chunk by chunk into
    int32 (every Atera panel has far fewer than 2**31 features), which avoids
    holding an int64 copy of the largest array: ~2.4 GB at 600M nonzeros.

    Args:
        atera_path: Atera output directory, cell_feature_matrix.zarr.zip, or an
            unzipped cell_feature_matrix.zarr directory.

    Returns:
        AnnData with .X = sparse CSR raw counts (int32 indices), indexed by
        barcode / gene symbol.

    Raises:
        FileNotFoundError: if no matrix store is found at ``atera_path``.
        ValueError: if the store lacks an element this loader needs.
    """
    import zarr
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11 keeps it under experimental
        from anndata.experimental import read_elem

    store_path = _resolve_atera_path(Path(atera_path))
    if store_path.suffix == ".zip":
        store = zarr.storage.ZipStore(str(store_path), mode="r")
    else:
        store = str(store_path)
    root = zarr.open_group(store, mode="r")

    missing = []
    for element in ATERA_REQUIRED_ELEMENTS:
        try:
            root[element]
        except KeyError:
            missing.append(element)
    if missing:
        raise ValueError(
            f"{store_path} is not an Atera count matrix this version of kopya can "
            f"read: missing {', '.join(missing)}."
        )

    obs = read_elem(root["obs"])
    var = read_elem(root["var"])

    data_arr, indices_arr = root["X/data"], root["X/indices"]
    indptr = root["X/indptr"][:]
    n_obs, n_var = len(obs), len(var)
    # Read the largest array chunk by chunk straight into int32 so no full-size
    # int64 copy is ever held. data stays in its stored integer dtype.
    if n_var > iinfo(int32).max:
        raise ValueError(f"{store_path} has {n_var} features; too many for int32 indices.")
    indices = empty(indices_arr.shape[0], dtype=int32)
    step = indices_arr.chunks[0] * 16
    for start in range(0, indices_arr.shape[0], step):
        indices[start:start + step] = indices_arr[start:start + step]
    X = csr_matrix((data_arr[:], indices, indptr), shape=(n_obs, n_var))

    gex = (var["feature_type"] == "Gene Expression").to_numpy()
    obs_out = obs.drop(columns=["barcode"])
    obs_out.index = obs["barcode"].astype(str).to_numpy()
    var_out = var.loc[gex, ["feature_id"]] if "feature_id" in var else var.loc[gex, []]
    var_out.index = var.loc[gex, "feature_name"].astype(str).to_numpy()

    adata = AnnData(X=X[:, gex].tocsr(), obs=obs_out, var=var_out)
    adata.var_names_make_unique()
    return adata


def load_counts(anndata=None, counts=None, obs=None, var=None,
                cellranger_dir=None, cellranger_h5=None,
                raw_count_check="warn", *, atera=None):
    """Load raw single-cell UMI counts into an AnnData.

    Exactly one input form must be supplied:
        - anndata: path to a .h5ad file (Form A).
        - counts + obs + var: paths to the mtx-trio (Form B).
        - cellranger_dir: CellRanger MTX directory (Form C).
        - cellranger_h5: CellRanger / Xenium .h5 file (Form D).
        - atera: Atera output directory or cell_feature_matrix.zarr.zip (Form E).

    Args:
        anndata: Path to .h5ad (Form A); mutually exclusive with the trio.
        counts: Path to counts.mtx (Form B).
        obs: Path to obs.csv (Form B).
        var: Path to var.csv (Form B).
        cellranger_dir: Path to a filtered_feature_bc_matrix/ directory (Form C).
        cellranger_h5: Path to a filtered_feature_bc_matrix.h5 or Xenium-style
            cell_feature_matrix.h5 file (Form D).
        atera: Path to an Atera Onboard Analysis output directory, its
            cell_feature_matrix.zarr.zip, or an unzipped .zarr directory
            (Form E). Keyword-only, so existing positional calls that pass
            raw_count_check keep working.
        raw_count_check: How to react when a Form-A .h5ad supplies .X that does
            not look like raw integer UMI counts and has no layers["counts"].
            "warn" (default) emits a UserWarning, "raise" errors out (opt-in
            strict mode), "ignore" skips the check. The mtx-trio, CellRanger
            and Atera forms always provide raw counts by construction, so the
            check does not apply to them.

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
    form_e = atera is not None

    n_forms = sum([form_a, form_b, form_c, form_d, form_e])
    if n_forms > 1:
        raise ValueError(
            "Pass exactly one input form: --anndata, (--counts/--obs/--var), "
            "--cellranger-dir, --cellranger-h5, or --atera."
        )
    if n_forms == 0:
        raise ValueError(
            "Must pass one of: --anndata, (--counts, --obs, --var), "
            "--cellranger-dir, --cellranger-h5, or --atera."
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
    if form_e:
        return _load_atera(Path(atera))
    return _load_10x_h5(Path(cellranger_h5))
