"""Scanpy-style tools namespace: run CNV calling and store results in the AnnData.

Usage mirrors infercnvpy / scanpy:

    import scanpy as sc
    import kopya as kp

    adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")   # raw counts in .X
    # ... your usual cell-type annotation pass populates adata.obs["cell_type"] ...

    kp.tl.cnv(adata, reference_key="cell_type",
              reference_cat=["T cell", "B cell", "Macrophage", "Endothelial"])

    # results now live on the AnnData:
    adata.obs["cnv_class"]          # 'tumor' / 'normal' / 'uncertain' (NaN = QC-filtered)
    adata.obs["cnv_score"]          # per-cell aneuploidy score
    adata.obs["cnv_subclone"]       # subclone label within tumor cells
    adata.obs["cnv_low_complexity"] # bool — ambient / low-gene-count cell flag
    adata.obsm["cnv_chr"]           # cells x chromosomes, 1.0-centered (>1 gain, <1 loss)
    adata.uns["cnv"]                # chroms, baseline method, params, segment table

Give ``reference_key``/``reference_cat`` (the normal cell types) to run supervised
against a known reference — recommended, and required for mesenchymal tumors that
invert unsupervised. Omit them to run the unsupervised baseline cascade.
"""

import numpy as np


def cnv(
    adata,
    *,
    reference_key=None,
    reference_cat=None,
    layer=None,
    complexity_gate=True,
    raw_count_check="warn",
    key_added="cnv",
    copy=False,
):
    """Call CNVs and write the results back into ``adata`` (scanpy-style).

    Args:
        adata: AnnData with **raw counts** in ``.X`` (or in ``layer``). The tool
            applies its own CP10k+log1p, so do not pass pre-normalized values.
        reference_key: obs column holding cell-type labels. When given with
            ``reference_cat``, those categories are the supervised normal
            reference. Omit for the unsupervised baseline cascade.
        reference_cat: label or list of labels in ``adata.obs[reference_key]`` to
            use as the normal reference.
        layer: name of a layer holding raw counts; default ``None`` uses ``.X``.
        complexity_gate: feed per-cell detected-gene counts to the low-complexity
            gate (matches the CLI). Default True.
        raw_count_check: how to react if the selected matrix does not look like
            raw counts (it must be — the tool re-applies CP10k+log1p, so passing
            normalized values gives wrong calls). ``"warn"`` (default), ``"raise"``,
            or ``"ignore"`` (e.g. for deliberately de-logged pseudo-counts).
        key_added: prefix for the stored keys (default ``"cnv"``).
        copy: if True, operate on and return a copy; else annotate in place and
            return None.

    Returns:
        AnnData if ``copy=True``, else None (``adata`` is modified in place).

    Writes:
        obs[f"{key_added}_class"], obs[f"{key_added}_score"],
        obs[f"{key_added}_subclone"], obs[f"{key_added}_low_complexity"]
            per-cell results; NaN for cells dropped by M1 QC.
        obsm[f"{key_added}_chr"]  (n_obs x n_chroms) 1.0-centered CNV matrix;
            rows for dropped cells are NaN.
        uns[key_added]  metadata: chroms, baseline method, counts, params, and a
            compact segment table.
    """
    from kopya.pipeline import run_pipeline

    if copy:
        adata = adata.copy()

    # Results are mapped back onto cells by barcode; duplicate obs_names (common
    # after concatenation without index_unique) would make that ambiguous and
    # break the reindex/get_indexer below. Fail fast, before the expensive run.
    if not adata.obs_names.is_unique:
        raise ValueError(
            "adata.obs_names must be unique so CNV results can be mapped back "
            "onto cells; call adata.obs_names_make_unique() first."
        )

    # ── resolve the supervised reference (if any) ────────────────────────────
    if reference_cat is not None and reference_key is None:
        raise ValueError(
            "reference_cat requires reference_key (the obs column to match it in); "
            "otherwise the categories are silently ignored and the run would fall "
            "back to unsupervised. Pass both, or neither."
        )
    norm_cell_names = None
    cats = None
    if reference_key is not None:
        if reference_key not in adata.obs:
            raise KeyError(f"reference_key {reference_key!r} not in adata.obs.")
        if reference_cat is None:
            raise ValueError("reference_cat is required when reference_key is given.")
        # Accept a scalar (str/int/bool) or ANY non-string iterable of labels
        # (list/tuple/set, a pandas Index, a numpy array, a generator, ...).
        if isinstance(reference_cat, str):
            raw_cats = [reference_cat]
        else:
            try:
                raw_cats = list(reference_cat)
            except TypeError:            # a non-iterable scalar (int/bool/float)
                raw_cats = [reference_cat]
        # Match by native equality OR by string form — the UNION, not a fallback:
        # native handles reference_cat=[1] against int / NaN-promoted float 1.0
        # columns; the string form handles mixed types (categorical "1" vs int 1).
        # A fallback-only-when-zero-native would drop the string-only labels of a
        # mixed request like [1, "normal"].
        col = adata.obs[reference_key]
        match = col.isin(raw_cats) | col.astype(str).isin({str(c) for c in raw_cats})
        norm_cell_names = adata.obs_names[match.to_numpy()].tolist()
        cats = sorted(str(c) for c in raw_cats)
        if len(norm_cell_names) == 0:
            raise ValueError(
                f"No cells match reference_cat={cats} in "
                f"adata.obs[{reference_key!r}]; check the labels."
            )

    # ── build a LAYER-FREE view of the matrix we will run on ─────────────────
    # Wrap the selected matrix (X, or `layer`) in a fresh AnnData carrying only
    # obs/var names. This matters for the raw-count guard: validate_raw_counts
    # trusts a present layers['counts'] over .X, so if we handed it the original
    # object (which may have a counts layer) it would skip inspecting the .X we
    # actually run on — a normalized .X would then slip through. A layer-free
    # wrapper forces it to inspect the real matrix.
    import anndata as ad
    import pandas as pd
    selected = adata.layers[layer] if layer is not None else adata.X
    raw = ad.AnnData(
        X=selected,
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names),
    )

    # Guard the load boundary the way load_counts does: the tool re-normalizes, so
    # a log1p/CP10k matrix here would be double-normalized into wrong calls. Warn
    # (default), raise, or ignore per raw_count_check.
    from kopya.normalize import validate_raw_counts
    validate_raw_counts(raw, action=raw_count_check)

    # ── run the shared in-memory core ────────────────────────────────────────
    res = run_pipeline(
        raw, norm_cell_names=norm_cell_names, complexity_gate=complexity_gate)

    # ── align per-cell results back to the caller's (unfiltered) cells ───────
    pred = res.prediction_df  # indexed by the filtered barcodes
    obs_index = adata.obs_names
    adata.obs[f"{key_added}_class"] = pred["class"].reindex(obs_index).astype("category")
    adata.obs[f"{key_added}_score"] = pred["tumor_score"].reindex(obs_index).astype(float)
    if "subclone" in pred:
        adata.obs[f"{key_added}_subclone"] = pred["subclone"].reindex(obs_index).astype("category")
    if "low_complexity" in pred:
        # Reindexing a bool Series over dropped cells inserts NaN and would turn it
        # into an object column (bool + float) that write_h5ad refuses. Use pandas'
        # nullable boolean, which serializes cleanly and keeps True/False/<NA>.
        adata.obs[f"{key_added}_low_complexity"] = (
            pred["low_complexity"].reindex(obs_index).astype("boolean")
        )

    # per-chromosome matrix: full (n_obs x n_chroms), NaN for filtered cells
    chroms = list(res.chroms_present)
    full = np.full((adata.n_obs, len(chroms)), np.nan, dtype=float)
    pos = obs_index.get_indexer(res.adata_m1.obs_names)
    ok = pos >= 0
    full[pos[ok]] = np.asarray(res.chr_matrix)[ok]
    adata.obsm[f"{key_added}_chr"] = full

    seg = res.segments
    adata.uns[key_added] = {
        "chroms": chroms,
        "baseline_method": res.qc["baseline_method"],
        "n_tumor": res.qc["n_tumor"],
        "n_normal": res.qc["n_normal"],
        "n_uncertain": res.qc["n_uncertain"],
        "n_segments": res.qc["n_segments"],
        "n_cells_filtered": res.qc["n_cells_filtered"],
        "params": {
            "reference_key": reference_key,
            "reference_cat": cats,  # sorted list of str labels, or None
            "layer": layer,
            "complexity_gate": bool(complexity_gate),
            "supervised": norm_cell_names is not None,
        },
        # compact, h5ad-serializable segment table
        "segments": {
            "chr": seg["chr"].astype(str).to_list(),
            "start_bp": seg["start_bp"].astype(int).to_list(),
            "end_bp": seg["end_bp"].astype(int).to_list(),
            "n_genes": seg["n_genes"].astype(int).to_list(),
            "tumor_mean": seg["tumor_mean"].astype(float).to_list(),
        },
    }

    return adata if copy else None
