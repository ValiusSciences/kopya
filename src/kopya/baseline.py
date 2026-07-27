"""Step §3.2 — pick the diploid baseline.

Three modes, tried in order:
    1. supervised   — caller supplied a list of known-normal barcodes.
    2. signature    — UCell scoring against immune/stromal/endothelial signatures;
                      cells passing the threshold form the confident-normal pool.
    3. variance     — Leiden cluster on PCA, lowest-variance cluster = normal.
                      Triggered when the signature pool is too small (<50 cells).
    4. gmm_fallback — per-cell variance GMM if even variance clustering fails to
                      resolve a low-variance subpopulation.

pick_baseline() returns (mask, method, diagnostics). method ∈ {supervised,
signature, variance, gmm_fallback}; the chosen method is recorded in qc.json
so downstream knows how confident the baseline is.
"""

import json
import math
import os
from importlib.resources import files
from pathlib import Path

from anndata import AnnData
from numpy import argmin, asarray, isfinite, median
from pyucell import compute_ucell_scores
from scanpy.pp import neighbors as sc_neighbors
from scanpy.pp import pca as sc_pca
from scanpy.tl import leiden as sc_leiden
from sklearn.mixture import GaussianMixture


# Threshold on the top UCell score for a cell to enter the signature pool.
# 0.3 mirrors SCEVAN's confident-normal cutoff; UCell scores are in [0, 1]
# and 0.3 is "moderately above background" for a small signature.
DEFAULT_NORMAL_SCORE_THRESHOLD = 0.3

# Minimum signature pool size before we trust signature-based baseline.
# 50 cells is the practical floor for a stable per-gene median baseline.
DEFAULT_MIN_NORMAL_CELLS = 50

# Variance-fallback PCA + neighbors + leiden defaults; small numbers because
# this is a fallback and we want a fast, stable cluster solution rather than
# a publication-quality embedding.
DEFAULT_VARIANCE_N_COMPS = 30
DEFAULT_VARIANCE_N_NEIGHBORS = 15
DEFAULT_VARIANCE_LEIDEN_RES = 1.0


def load_signatures(path=None):
    """Load a non-malignant signature library from JSON (bundled by default).

    The schema matches the bundled ``normal_signatures.json``: an optional
    ``"_meta"`` object carrying ``"non_malignant_labels"`` (the allow-list of
    cell types treated as "normal"), plus one key per signature mapping a
    label to a list of HGNC gene symbols. When ``_meta.non_malignant_labels``
    is absent, every signature is treated as non-malignant.

    Args:
        path: Optional path to a signature JSON. Defaults to the bundled
            library shipped with the package.

    Returns:
        (signatures, non_malignant_labels):
            signatures: dict[str, list[str]] of name -> gene symbols.
            non_malignant_labels: list[str] — labels considered "normal" when
                a cell scores highest on them.

    Raises:
        ValueError: if the file defines no signatures, or lists a
            non-malignant label that is not a defined signature (a typo guard).
    """
    # Importlib.resources gives us a robust pointer to the bundled file that
    # works for editable + wheel installs alike; an explicit path bypasses it.
    if path is None:
        resource = files("kopya.data").joinpath("normal_signatures.json")
        text = Path(str(resource)).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)

    # _meta carries the non-malignant label set so the library can be extended
    # without code changes; everything else is a signature.
    meta = raw.pop("_meta", {})
    signatures = raw
    if not signatures:
        raise ValueError(
            f"signature library {path or '(bundled)'} defines no signatures "
            "(expected at least one '<label>': [gene, ...] entry)"
        )

    non_malignant_labels = meta.get("non_malignant_labels", list(signatures.keys()))
    unknown = [lbl for lbl in non_malignant_labels if lbl not in signatures]
    if unknown:
        raise ValueError(
            f"non_malignant_labels {unknown} in {path or '(bundled)'} are not "
            f"defined signatures (available: {sorted(signatures)})"
        )

    return signatures, non_malignant_labels


def _load_bundled_signatures():
    """Back-compat shim: load the bundled non-malignant signature library."""
    return load_signatures(None)


def _supervised_baseline(adata, norm_cell_path=None, norm_cell_names=None):
    """Mode 1: caller supplied a list of known-normal barcodes.

    The barcodes come either from a file (``norm_cell_path``, the CLI path) or
    directly as an iterable (``norm_cell_names``, the in-memory API path). Pass
    exactly one.

    Args:
        adata: AnnData (post-M1).
        norm_cell_path: Path to a text file with one barcode per line.
        norm_cell_names: Iterable of known-normal barcodes (alternative to the
            file), used by the scanpy-style API so no temp file is needed.

    Returns:
        (mask, diagnostics) where mask is a bool array len == n_obs and
        diagnostics carries the supplied count and the matched count.
    """
    if (norm_cell_path is None) == (norm_cell_names is None):
        raise ValueError("Pass exactly one of norm_cell_path or norm_cell_names.")

    if norm_cell_names is not None:
        supplied = {str(bc).strip() for bc in norm_cell_names if str(bc).strip()}
    else:
        # Read one barcode per line; ignore blanks and '#'-commented lines so
        # users can annotate the list.
        raw_lines = Path(norm_cell_path).read_text(encoding="utf-8").splitlines()
        supplied = {
            line.strip() for line in raw_lines
            if line.strip() and not line.strip().startswith("#")
        }

    # Intersect with the adata's obs index — missing barcodes (cells dropped
    # in M1 QC) are silently skipped but counted in diagnostics so the user
    # can see whether their list was usable.
    obs_set = set(adata.obs_names)
    matched = supplied & obs_set

    mask = asarray([bc in matched for bc in adata.obs_names], dtype=bool)

    diagnostics = {
        "n_supplied": len(supplied),
        "n_matched": int(mask.sum()),
        "n_unmatched": len(supplied - obs_set),
    }
    return mask, diagnostics


def _signature_baseline(
    adata,
    signatures=None,
    non_malignant_labels=None,
    threshold=DEFAULT_NORMAL_SCORE_THRESHOLD,
):
    """Mode 2: signature-based via UCell scoring (default unsupervised path).

    Cells are confident-normal iff:
        - their top UCell score across all signatures exceeds `threshold`, AND
        - the winning signature is in non_malignant_labels.

    Args:
        adata: AnnData with log1p(CP10k) values in .X (post-M1).
        signatures: dict[name, gene list]; defaults to the bundled library.
        non_malignant_labels: list of labels considered normal; defaults to
            the bundled library's _meta.non_malignant_labels.
        threshold: minimum top-score for a cell to enter the pool.

    Returns:
        (mask, diagnostics) where diagnostics records per-signature pool sizes
        and the winning-signature distribution.
    """
    # Default to the bundled library; allow overrides for testing and future
    # tumor-type-specific signature sets.
    if signatures is None or non_malignant_labels is None:
        bundled_sig, bundled_labels = _load_bundled_signatures()
        if signatures is None:
            signatures = bundled_sig
        if non_malignant_labels is None:
            non_malignant_labels = bundled_labels

    # compute_ucell_scores reads .X (read-only) and writes `{name}_UCell` columns
    # to .obs. Score on a lightweight AnnData that SHARES .X (no matrix copy) with
    # a fresh empty obs, so we neither duplicate the matrix nor pollute the
    # caller's obs.
    #
    # --- PERFORMANCE-CRITICAL: chunk_size tuned for two competing costs ---
    # pyucell's default chunk_size=500 makes thousands of tiny chunks on a large
    # cohort (~1,800 at 900K cells); the per-chunk joblib/loky overhead then
    # dominates (this was the M2 "superlinearity"). Bigger chunks fix that — but
    # each in-flight chunk materializes up to max_rank (1500) rank entries per
    # cell plus COO/CSR build buffers, and ~n_cores chunks run at once, so an
    # unbounded chunk_size would reintroduce a memory blow-up at scale.
    #
    # So: aim for a few chunks per core (kills the overhead), but CAP the chunk
    # size (bounds the per-wave rank buffers to a few GB: n_cores * cap * max_rank
    # * ~12 B). Both bounds keep the win; chunk_size only partitions cells, so the
    # scores are byte-identical regardless (verified; pinned by a test).
    n_cores = os.cpu_count() or 1
    _CHUNK_CAP = 8192
    chunk_size = min(_CHUNK_CAP, max(1000, math.ceil(adata.n_obs / (3 * n_cores))))
    # Score on a lightweight AnnData that SHARES .X (compute_ucell_scores only
    # reads it) with a fresh empty obs, so we neither copy the matrix nor pollute
    # the caller's obs with the `{sig}_UCell` columns it writes.
    scored = AnnData(X=adata.X, obs=adata.obs[[]].copy(), var=adata.var)
    compute_ucell_scores(scored, signatures=signatures, chunk_size=chunk_size)

    # The score columns follow `{signature_name}_UCell` convention.
    score_cols = [f"{name}_UCell" for name in signatures]
    score_mat = scored.obs[score_cols].to_numpy()

    # Winner per cell: index of the max score; ties broken by argmax's natural
    # behavior (first occurrence) which is fine — only matters for the label.
    winner_idx = score_mat.argmax(axis=1)
    winner_score = score_mat.max(axis=1)
    winner_labels = asarray([list(signatures.keys())[i] for i in winner_idx])

    # Confident-normal mask: score must clear the threshold AND the winning
    # signature must be in the non-malignant label set.
    non_malig = set(non_malignant_labels)
    above_thr = winner_score >= threshold
    is_normal = asarray([lbl in non_malig for lbl in winner_labels])
    mask = above_thr & is_normal

    # Per-label cell counts for diagnostics — useful for sanity-checking the
    # signature library on a new tissue (e.g. expecting B-cell signal in LN).
    per_label = {}
    for lbl in signatures:
        per_label[lbl] = int(((winner_labels == lbl) & above_thr).sum())

    diagnostics = {
        "n_passed_threshold": int(above_thr.sum()),
        "n_confident_normal": int(mask.sum()),
        "threshold": float(threshold),
        "per_label_counts": per_label,
        # Record the effective allow-list so a run's provenance is self-contained
        # — important now that it can be overridden per-run via the CLI.
        "non_malignant_labels": list(non_malignant_labels),
        "median_top_score": float(median(winner_score)),
    }
    return mask, diagnostics


def _variance_baseline(
    adata,
    n_comps=DEFAULT_VARIANCE_N_COMPS,
    n_neighbors=DEFAULT_VARIANCE_N_NEIGHBORS,
    resolution=DEFAULT_VARIANCE_LEIDEN_RES,
):
    """Mode 3: cluster on PCA + Leiden, return the lowest-variance cluster.

    The CopyKAT mechanism: normal cells are assumed to be the most homogeneous
    population in expression space, so the cluster with the smallest mean
    per-gene variance is treated as the diploid pool.

    Args:
        adata: AnnData with log1p(CP10k) values in .X (post-M1).
        n_comps: PCA components for the embedding.
        n_neighbors: kNN neighborhood for Leiden.
        resolution: Leiden resolution parameter.

    Returns:
        (mask, diagnostics) where diagnostics records the per-cluster variance
        ranking so we can see how decisive the winner was.
    """
    # Defensive copy — scanpy mutators write into obs/obsm/uns in place.
    working = adata.copy()

    # Standard scanpy embedding pipeline; we keep it minimal because this is a
    # fallback path used when signature scoring already failed.
    sc_pca(working, n_comps=min(n_comps, working.n_vars - 1, working.n_obs - 1))
    sc_neighbors(working, n_neighbors=min(n_neighbors, working.n_obs - 1))
    # Future scanpy default for leiden — pin explicitly so we are not
    # silently coupled to the deprecating leidenalg backend.
    sc_leiden(
        working,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    cluster_labels = working.obs["leiden"].astype(str).to_numpy()
    unique_clusters = sorted(set(cluster_labels))

    # Per-cluster mean of per-gene variance. Using mean-of-variance rather
    # than median because the latter is too forgiving of a single noisy gene
    # in tiny clusters.
    cluster_variances = {}
    X = working.X
    for cl in unique_clusters:
        cl_mask = cluster_labels == cl
        if cl_mask.sum() < 2:
            # Singletons have undefined variance; assign +inf so they lose.
            cluster_variances[cl] = float("inf")
            continue
        cl_data = X[cl_mask, :]
        # Sparse variance via E[X^2] - E[X]^2; avoids densifying the slice.
        mean = asarray(cl_data.mean(axis=0)).ravel()
        mean_sq = asarray(cl_data.multiply(cl_data).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
        # Mean across genes — bigger gives a single scalar per cluster.
        cluster_variances[cl] = float(var.mean())

    # Cluster with the smallest mean variance wins.
    winner_cluster = min(cluster_variances, key=cluster_variances.get)
    mask = cluster_labels == winner_cluster

    diagnostics = {
        "n_clusters": len(unique_clusters),
        "winner_cluster": winner_cluster,
        "winner_variance": cluster_variances[winner_cluster],
        "all_variances": cluster_variances,
        "n_normal": int(mask.sum()),
    }
    return mask, diagnostics


def _gmm_fallback(adata):
    """Mode 4: 2-component GMM on per-cell expression variance.

    Last-resort baseline picker: fit a 2-component GMM on per-cell variance
    across genes, take the lower-mean component as normal. CopyKAT does this
    too — included for parity when variance clustering fails to resolve.

    Args:
        adata: AnnData post-M1.

    Returns:
        (mask, diagnostics) — mask of cells assigned to the low-variance GMM
        component; diagnostics records the two component means.
    """
    # Per-cell variance across genes. Sparse-safe via E[X^2] - E[X]^2.
    X = adata.X
    cell_mean = asarray(X.mean(axis=1)).ravel()
    cell_mean_sq = asarray(X.multiply(X).mean(axis=1)).ravel()
    cell_var = cell_mean_sq - cell_mean ** 2

    # Replace any non-finite entries (zero-row pathologies) with the median so
    # the GMM fit does not blow up; this is rare but cheap to guard against.
    cell_var = cell_var.copy()
    bad = ~isfinite(cell_var)
    if bad.any():
        cell_var[bad] = median(cell_var[~bad]) if (~bad).any() else 0.0

    # 2-component GMM on the 1-D variance vector. random_state pinned so the
    # call is reproducible across runs.
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(cell_var.reshape(-1, 1))
    component = gmm.predict(cell_var.reshape(-1, 1))

    # Low-variance component is the one with the smaller mean. argmin returns
    # the component index; cells assigned there are the normal pool.
    low_component = int(argmin(gmm.means_.ravel()))
    mask = component == low_component

    diagnostics = {
        "component_means": [float(m) for m in gmm.means_.ravel()],
        "low_component": low_component,
        "n_normal": int(mask.sum()),
    }
    return mask, diagnostics


def pick_baseline(
    adata,
    norm_cell_path=None,
    signatures=None,
    non_malignant_labels=None,
    threshold=DEFAULT_NORMAL_SCORE_THRESHOLD,
    min_normal_cells=DEFAULT_MIN_NORMAL_CELLS,
    norm_cell_names=None,
):
    """Pick the diploid baseline mask using the §3.2 four-mode cascade.

    Order of attempts:
        1. supervised   if norm_cell_path or norm_cell_names is supplied.
        2. signature    UCell + bundled non-malignant library.
        3. variance     Leiden + lowest-variance cluster, if (2) pool too small.
        4. gmm_fallback if (3) also returns a too-small pool.

    Args:
        adata: AnnData post-M1 (log1p(CP10k), projected onto genome).
        norm_cell_path: Optional path to a known-normal barcode list (file).
        norm_cell_names: Optional iterable of known-normal barcodes (in-memory
            alternative to norm_cell_path, used by the scanpy-style API).
        signatures: Optional override of the bundled signature library.
        non_malignant_labels: Optional override of the bundled label set.
        threshold: UCell top-score threshold for the signature mode.
        min_normal_cells: Floor below which we cascade to the next mode.

    Returns:
        dict with keys:
            mask: bool array, len == adata.n_obs, True for normal cells.
            method: str in {supervised, signature, variance, gmm_fallback}.
            n_normal: int, count of cells in the mask.
            diagnostics: per-mode dict from the chosen path.
    """
    # ── Mode 1: supervised ────────────────────────────────────────────────
    # Supervised always wins when supplied — the user knows their data better
    # than any heuristic. We do NOT fall back even if the matched count is
    # small; that would silently override an explicit instruction.
    if norm_cell_path is not None or norm_cell_names is not None:
        mask, diag = _supervised_baseline(
            adata, norm_cell_path=norm_cell_path, norm_cell_names=norm_cell_names)
        # A supervised run with zero surviving reference cells would silently fall
        # through to an all-False mask (centering against the whole-cohort median),
        # while still reporting method="supervised" — i.e. calls that did NOT use
        # the requested reference. Fail loudly instead: this is almost always a
        # barcode/label mismatch or a reference that was entirely dropped by QC.
        if int(mask.sum()) == 0:
            raise ValueError(
                f"Supervised reference matched 0 cells after M1 QC "
                f"(supplied {diag['n_supplied']}, {diag['n_unmatched']} not in the "
                "data). Check that the reference barcodes/labels match the cells."
            )
        result = {
            "mask": mask,
            "method": "supervised",
            "n_normal": int(mask.sum()),
            "diagnostics": diag,
        }
        return result

    # ── Mode 2: signature ─────────────────────────────────────────────────
    mask, diag = _signature_baseline(
        adata,
        signatures=signatures,
        non_malignant_labels=non_malignant_labels,
        threshold=threshold,
    )
    n_sig = int(mask.sum())
    if n_sig >= min_normal_cells:
        result = {
            "mask": mask,
            "method": "signature",
            "n_normal": n_sig,
            "diagnostics": diag,
        }
        return result

    # ── Mode 3: variance fallback ─────────────────────────────────────────
    # Signature pool too small (typical of high-purity tumors or tissue types
    # where our signature library has no matching markers). Leiden + lowest-
    # variance cluster approach is topology-driven and agnostic to cell-type
    # identity.
    mask_var, diag_var = _variance_baseline(adata)
    if int(mask_var.sum()) >= min_normal_cells:
        result = {
            "mask": mask_var,
            "method": "variance",
            "n_normal": int(mask_var.sum()),
            "diagnostics": {"signature_attempted": diag, "variance": diag_var},
        }
        return result

    # ── Mode 4: GMM fallback ──────────────────────────────────────────────
    # Variance clustering also failed to surface a low-variance pool. Fall
    # back to per-cell variance GMM — accept whatever it returns even if
    # below min_normal_cells, because we have no further mode to try.
    mask_gmm, diag_gmm = _gmm_fallback(adata)
    result = {
        "mask": mask_gmm,
        "method": "gmm_fallback",
        "n_normal": int(mask_gmm.sum()),
        "diagnostics": {
            "signature_attempted": diag,
            "variance_attempted": diag_var,
            "gmm": diag_gmm,
        },
    }
    return result
