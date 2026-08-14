"""inferCNV-style genome heatmap from a kopya run directory.

Renders the per-cell × per-segment CN matrix as a genome-ordered heatmap, up to
three stacked panels — a **References** panel of confident-diploid cells on top,
a tumor **Observations** panel (grouped by subclone), and an **Uncertain** panel
of flagged cells when any exist — with columns laid out left-to-right along the
genome, widths proportional to gene content, and blue→white→red diverging colour
(loss→diploid→gain). Cells are ordered within each group by hierarchical
clustering.

Consumes three artifacts written by `kopya run` into the out-dir:
    cn_per_segment.npz   dense (cells × segments) signal + cell_barcodes
    segments.parquet     segment table (chr, start_idx, end_idx, n_genes)
    prediction.csv       per-cell class / subclone / tumor_score / cn_burden /
                         low_complexity

Choosing what to display
------------------------
The "normal" class on a real run is a mixed bag — genuinely-diploid cells, but
also ambient/empty-droplet-like cells and tumor cells the caller under-called as
normal (which still carry real CNV). Showing all of them as "references" makes
the panel read as noisy structure rather than a clean diploid baseline. So:

    * low_complexity cells (ambient) are dropped from the plot entirely.
    * the reference panel — and the baseline the whole figure is recentered
      against — is the *confident-diploid* subset: normal-called, non-junk cells
      whose ``cn_burden`` (CN magnitude, not the signed tumor_score) is in the
      lower ``ref_score_quantile`` of that pool. This
      is kopya's unsupervised analogue of inferCNV's curated reference set, and
      it sharpens the observation panel too (the frame is a true diploid median).

Recovering the signed gain/loss signal
---------------------------------------
The stored `cn` matrix is the per-segment mean of the median-centered smoothed
signal. It is NOT a clean signed log-ratio: median-centering against a
zero-inflated normal pool leaves a per-gene positive pedestal, and tumor cells
carry a per-cell magnitude offset (they are globally more active). So the raw
matrix reads as an all-gain wash. We recover the signal the way inferCNV
displays it — as a deviation relative to the reference cells:

    1. subtract each segment's median over the confident-diploid reference → references ≈ 0
    2. subtract each cell's own median across segments                      → drop per-cell offset
    3. denoise: soft-threshold within the reference noise band              → collapse the floor to white

The result is a log-space deviation centered on 0 (``scale="log"``). Passing
``scale="linear"`` exponentiates it to a copy-number ratio centered on 1.0
(diploid), matching the "Modified Expression" convention of inferCNV and the
1.0-centered `chr_cnv_matrix.csv` output.
"""

from pathlib import Path

import json

import numpy as np
import pandas as pd

from kopya.annotations import CANONICAL_CHROM_ORDER
# Single definition of the per-cell genome-wide center, shared with the
# classifier: the tumor score and this figure must agree on where diploid sits.
from kopya.classify import weighted_median_rows as _weighted_median_rows

# Categorical colours for the subclone annotation bar. Deliberately avoid the
# blue/red of the heatmap body so subclone identity never reads as gain/loss.
# Order + hues validated colourblind-safe (worst adjacent CVD ΔE 16.2).
SUBCLONE_COLORS = [
    "#008300",  # green
    "#eda100",  # amber
    "#4a3aa7",  # violet
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
NORMAL_COLOR = "#b8b7b0"  # neutral warm gray — reference (confident-diploid) cells
UNCERTAIN_COLOR = "#8f97a3"  # cool slate — flagged/uncertain cells, set apart from both
_UNASSIGNED_COLOR = "#999999"

# Cap the per-group hierarchical clustering cost: Ward linkage is O(n²) in both
# time and memory, so we subsample very large panels before ordering. These are
# display-only caps — the call outputs are unaffected.
DEFAULT_N_REF = 4000
DEFAULT_MAX_OBS = 8000


def _require_matplotlib():
    """Import matplotlib lazily, with an actionable message if it is absent.

    matplotlib is an optional dependency (the ``plot`` extra) so the core
    calling pipeline stays lean. Only the heatmap needs it.
    """
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        raise ModuleNotFoundError(
            "plotting requires matplotlib; install it with "
            "`pip install 'kopya[plot]'` or `pip install matplotlib`."
        ) from exc
    matplotlib.use("Agg")  # headless: never try to open a window
    return matplotlib


def _cluster_leaf_order(mat, cluster_max):
    """Row ordering from Ward hierarchical clustering (dendrogram leaf order).

    Returns identity order for degenerate (<3 rows) input, when the panel is
    larger than ``cluster_max`` (clustering would be too costly — cells keep
    their existing order), or on any linkage failure.

    Args:
        mat: (n_rows × n_features) ndarray to cluster over rows.
        cluster_max: skip clustering above this row count.

    Returns:
        1-D integer index array giving the row order.
    """
    n = mat.shape[0]
    if n < 3 or n > cluster_max:
        return np.arange(n)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        return leaves_list(linkage(mat, method="ward"))
    except Exception:  # pragma: no cover - defensive against degenerate input
        return np.arange(n)


def _recenter(cn, normal_mask, seg_weights):
    """Recover the signed, reference-relative CN deviation (see module docstring).

    Args:
        cn: (n_cells × n_segments) stored CN matrix.
        normal_mask: bool array, True for normal-called cells.
        seg_weights: (n_segments,) per-segment gene counts, aligned to cn's
            columns. Weights the per-cell genome-wide median so short segments
            do not dominate the row center.

    Returns:
        (n_cells × n_segments) float array, log-space, references ≈ 0.
    """
    if normal_mask.sum() == 0:
        # No reference pool — fall back to the whole-cohort median so the plot
        # is still centered somewhere sensible rather than all-gain.
        ref_median = np.median(cn, axis=0)
    else:
        ref_median = np.median(cn[normal_mask], axis=0)
    rc = cn - ref_median[None, :]
    weights = np.asarray(seg_weights, dtype=np.float32)
    rc = rc - _weighted_median_rows(rc, weights)[:, None]
    return rc


def _denoise(rc, normal_mask, sd_amplifier):
    """inferCNV-style denoise: shrink deviations within the reference band to 0.

    Real single-cell expression carries a per-segment noise floor — the incoherent
    gene-to-gene scatter that is present at ~the same magnitude in normal and tumor
    cells alike (see the reference-pool analysis in the design notes). Left in, it
    makes the reference (normal) panel read as coloured speckle rather than the near-
    white noise floor inferCNV shows. We remove it the way inferCNV's ``denoise``
    step does: estimate each segment's reference-cell spread and soft-threshold every
    cell's deviation toward zero by ``sd_amplifier`` times that spread. A true CNV
    event sits many spreads above the floor and survives largely intact; reference-band
    scatter collapses to zero.

    Robust spread = per-segment MAD of the reference cells (already ~0-centred by
    ``_recenter``), scaled to a standard-deviation equivalent. Soft- rather than hard-
    thresholding avoids a discontinuity at the band edge that would show as a ring of
    saturated pixels around real events.

    The band is estimated from the CLEANEST reference cells, not all of them: on
    real cohorts the normal-called pool is a mix of genuinely-diploid cells and
    noisier ones (off-lineage normals, low-complexity/ambient cells the caller
    could not exclude). Estimating the spread over all of them inflates the band
    and eats real CNV events (a tumor's chr7 gain vanishing along with the noise).
    Taking the lower-half of reference cells by total deviation gives a band that
    reflects the true diploid noise floor, so tumor events survive while the floor
    still collapses to white.

    Args:
        rc: (n_cells × n_segments) reference-relative signal from ``_recenter``.
        normal_mask: bool array, True for the reference (normal) cells whose spread
            defines the noise band.
        sd_amplifier: band half-width in reference-spread units. 0 disables denoise;
            larger values whiten more aggressively (at the cost of weak-event signal).

    Returns:
        (n_cells × n_segments) float array, soft-thresholded toward 0.
    """
    if sd_amplifier <= 0:
        return rc
    ref_idx = np.where(normal_mask)[0]
    if ref_idx.size == 0:
        ref = rc
    else:
        # Restrict to the cleanest half of reference cells (lowest total |deviation|)
        # so noisy normal-called cells do not widen the band. Below ~a handful of
        # reference cells the trim is pointless — use them all.
        total_dev = np.abs(rc[ref_idx]).sum(axis=1)
        keep = ref_idx if ref_idx.size < 8 else ref_idx[total_dev <= np.median(total_dev)]
        ref = rc[keep]
    # MAD→σ (×1.4826); ref is ~0-centred per segment so |ref| is the deviation.
    spread = np.median(np.abs(ref - np.median(ref, axis=0)[None, :]), axis=0) * 1.4826
    thr = sd_amplifier * spread[None, :]
    return np.sign(rc) * np.maximum(np.abs(rc) - thr, 0.0)


def _genome_layout(segments):
    """Order segments by genome position and build the column x-axis.

    Column widths are proportional to each segment's gene count so chromosome
    blocks occupy horizontal space in proportion to gene content — the same
    visual convention inferCNV uses.

    Args:
        segments: DataFrame with columns chr, n_genes (row order = matrix cols).

    Returns:
        dict with:
            col_order  — int array to reorder matrix columns into genome order
            x_edges    — (n_segments + 1,) cumulative gene-count edges
            bounds     — chromosome boundary x-positions (for separator lines)
            mids       — chromosome label x-midpoints
            names      — chromosome short names (no "chr" prefix)
    """
    seg = segments.reset_index(drop=True)
    rank = {c: i for i, c in enumerate(CANONICAL_CHROM_ORDER)}
    # Stable sort keeps within-chromosome segment order; unknown contigs sort last.
    seg = seg.assign(_rank=seg["chr"].map(lambda c: rank.get(c, len(rank))))
    seg = seg.sort_values("_rank", kind="stable")
    col_order = seg.index.to_numpy()
    seg = seg.reset_index(drop=True)

    widths = seg["n_genes"].to_numpy(dtype=float)
    x_edges = np.concatenate([[0.0], np.cumsum(widths)])

    bounds, mids, names = [0.0], [], []
    for chrom, block in seg.groupby("chr", sort=False):
        lo = x_edges[block.index[0]]
        hi = x_edges[block.index[-1] + 1]
        bounds.append(hi)
        mids.append((lo + hi) / 2.0)
        names.append(chrom.replace("chr", ""))
    return {"col_order": col_order, "x_edges": x_edges, "bounds": bounds, "mids": mids, "names": names}


def _confident_diploid_mask(cls, low_c, score, ref_score_quantile):
    """Recentering-baseline mask: confident-diploid reference cells.

    Non-low-complexity ``normal`` cells whose CN magnitude is at or below the
    given quantile of that pool. **Always** excludes low-complexity cells, so a
    display toggle (``--show-low-complexity``) can never shift the baseline. When
    no score is available, all non-junk normals qualify.

    Args:
        cls: per-cell class labels.
        low_c: per-cell low_complexity bool flags.
        score: per-cell CN magnitude to rank on, or None. Must be NON-NEGATIVE,
            i.e. ``cn_burden`` (or ``|tumor_score|``) — never the signed
            ``tumor_score`` itself, whose minimum is the most anti-aligned cell
            rather than the most diploid one. Callers pick the column; see
            render_heatmap and cli's --denoise-outputs path.
        ref_score_quantile: keep normals at/below this quantile of the non-junk pool.

    Returns:
        (is_ref, thr): the boolean mask and the score threshold (None if no score).
    """
    norm = np.asarray(cls) == "normal"
    elig = norm & (~np.asarray(low_c, dtype=bool))
    if score is not None and elig.sum() > 0:
        thr = float(np.quantile(np.asarray(score)[elig], ref_score_quantile))
        return elig & (np.asarray(score) <= thr), thr
    return elig, None


def reference_relative_signal(
    cn,
    cls,
    low_complexity,
    tumor_score,
    segments,
    sd_amplifier=1.0,
    ref_score_quantile=0.5,
):
    """Signed, reference-relative, denoised per-cell × per-segment signal.

    This is the exact transform the heatmap draws, factored out so it can also be
    written to disk (``kopya run --denoise-outputs``). Two display-only
    steps applied to the RAW stored matrix:
      1. recenter each cell against the confident-diploid reference (per-segment
         reference median removed, then the cell's own gene-count-weighted median),
      2. inferCNV-style denoise — soft-threshold each cell's deviation toward 0 by
         ``sd_amplifier`` reference-cell spreads, collapsing the noise floor while
         leaving real CNV events (many spreads above the floor) intact.

    The result is log-space and signed: ``0`` ≈ diploid, ``>0`` gain, ``<0`` loss.
    Columns stay in ``segments`` row order (``segment_id`` ``i`` ↔ column ``i``);
    nothing is subsampled or reordered, so it aligns 1:1 with ``cn_per_segment.npz``.

    Args:
        cn: (n_cells × n_segments) raw CN matrix (``cn_per_segment.npz`` 'cn').
        cls: per-cell class labels ('tumor'/'normal'/'uncertain'), len n_cells.
        low_complexity: per-cell bool flags (ambient cells never define the
            reference band); None treats every cell as non-low-complexity.
        tumor_score: per-cell NON-NEGATIVE CN magnitude — ``cn_burden``, or
            ``|tumor_score|`` for older prediction.csv files — used to select the
            lowest-signal normals as the confident-diploid reference. None uses
            every non-junk normal. Passing the raw signed ``tumor_score`` here is a
            bug: its minimum is the most anti-aligned cell, so the baseline fills
            with cells carrying a large real event in the opposite direction. (The
            parameter keeps its historical name for callers that pass by keyword.)
        segments: detect_segments() table; ``n_genes`` weights the per-cell median.
        sd_amplifier: denoise strength in reference-spread units (0 disables denoise;
            higher whitens more aggressively, at the cost of weak-event signal).
        ref_score_quantile: keep normals with that magnitude at/below this quantile
            of the non-junk normal pool as the reference.

    Returns:
        (rc, is_ref, thr):
            rc: (n_cells × n_segments) denoised signed signal.
            is_ref: bool mask of the confident-diploid reference cells.
            thr: the magnitude cutoff used to pick the reference (None if no score).

    Raises:
        ValueError: if no confident-diploid reference cell survives — without a
            reference pool the recentering frame is undefined.
    """
    # A non-finite strength would poison every cell: nan propagates to an all-NaN
    # matrix, inf soft-thresholds everything to 0. Reject it for any caller.
    if not np.isfinite(sd_amplifier):
        raise ValueError(f"sd_amplifier must be finite, got {sd_amplifier!r}.")
    low_c = np.zeros(len(cls), dtype=bool) if low_complexity is None else low_complexity
    is_ref, thr = _confident_diploid_mask(cls, low_c, tumor_score, ref_score_quantile)
    if is_ref.sum() == 0:
        raise ValueError(
            "no confident-diploid reference cells — cannot compute reference-relative signal."
        )
    rc = _recenter(cn, is_ref, np.asarray(segments["n_genes"].to_numpy()))
    rc = _denoise(rc, is_ref, sd_amplifier)
    return rc, is_ref, thr


def _subsample(idx, cap, rng):
    """Return a sorted subsample of ``idx`` capped at ``cap`` (identity if smaller)."""
    if cap is not None and len(idx) > cap:
        return np.sort(rng.choice(idx, cap, replace=False))
    return idx


def _order_panel(idx, rc, cluster_max, cap, rng, subclone=None):
    """Subsample, group (by subclone if given), and hierarchically order a panel.

    Args:
        idx: integer row indices belonging to this panel.
        rc: the full (n_cells × n_segments) recentered matrix (indexed by ``idx``).
        cluster_max: skip within-group hierarchical ordering above this row count.
        cap: display subsample cap (proportional across subclones when grouping).
        rng: numpy Generator for reproducible subsampling.
        subclone: optional per-cell subclone label array; when given, rows are
            grouped into contiguous per-subclone blocks.

    Returns:
        (ordered_idx, group_labels, block_edges, block_labels) — block_edges are
        the within-panel row separators (the final panel edge is dropped).
    """
    idx = np.asarray(idx)
    if cap is not None and len(idx) > cap:
        if subclone is not None:
            frac = cap / len(idx)
            kept = [
                _subsample(idx[subclone[idx] == sc], max(1, int(round((subclone[idx] == sc).sum() * frac))), rng)
                for sc in np.unique(subclone[idx])
            ]
            idx = np.sort(np.concatenate(kept)) if kept else idx
        else:
            idx = _subsample(idx, cap, rng)

    names = None
    if subclone is not None:
        names = sorted(s for s in np.unique(subclone[idx]) if s not in ("nan", ""))
        names = names or None

    blocks, parts, edges, labels, running = [], [], [], [], 0
    for sc in (names if names else [None]):
        lbl = sc if sc is not None else ""  # group value + label kept identical
        g = idx if sc is None else idx[subclone[idx] == sc]
        g = g[_cluster_leaf_order(rc[g], cluster_max)]
        blocks.append(g)
        parts.append(np.full(len(g), lbl))
        running += len(g)
        edges.append(running)
        labels.append(lbl)
    return np.concatenate(blocks), np.concatenate(parts), edges[:-1], labels


def render_heatmap(
    run_dir,
    out_path,
    sample=None,
    subtitle=None,
    scale="log",
    n_ref=DEFAULT_N_REF,
    max_obs=DEFAULT_MAX_OBS,
    vlim=0.15,
    sd_amplifier=1.0,
    ref_score_quantile=0.5,
    drop_low_complexity=True,
    cluster_max=8000,
    dpi=200,
    seed=0,
    title="kopya",
):
    """Render an inferCNV-style CNV heatmap from a run directory.

    Args:
        run_dir: directory containing cn_per_segment.npz, segments.parquet,
            prediction.csv (a `kopya run` out-dir).
        out_path: destination image path (.png/.pdf/.svg — inferred by suffix).
        sample: sample label for the header; defaults to qc.json's "sample".
        subtitle: header subtitle; defaults to a scale-aware generic line.
        scale: "log" (deviation centered on 0) or "linear" (ratio centered on 1.0).
        n_ref: subsample normal (reference) cells to at most this many.
        max_obs: subsample tumor (observation) cells to at most this many,
            proportionally across subclones.
        vlim: symmetric colour clip in log-space; the linear scale maps
            exp(±vlim) with white pinned to 1.0.
        sd_amplifier: inferCNV-style denoise strength. Each cell's per-segment
            deviation is soft-thresholded toward 0 by this many reference-cell
            spreads, collapsing the normal noise floor to white while leaving real
            CNV events (many spreads above the floor) intact. 0 disables denoise;
            1.0 (default) balances a clean floor against weak-event retention on
            cohorts whose normal-called pool still carries noisy cells. Raise to
            whiten more aggressively. The band is estimated from the cleanest
            reference cells (see ``_denoise``).
        ref_score_quantile: the reference panel (and the recentering baseline) is
            the *confident-diploid* subset — normal-called, non-low-complexity
            cells whose ``cn_burden`` (CN magnitude, not the signed tumor_score)
            is at or below this quantile of that pool.
            Excludes tumor cells the classifier under-called as normal (they carry
            real CNV) so the reference reads as a clean diploid baseline, the way
            inferCNV uses a curated reference. Default 0.5 (flattest half). 1.0
            keeps every non-junk normal.
        drop_low_complexity: when True (default), cells flagged low_complexity
            (ambient / empty-droplet-like) are dropped from every panel — they are
            not trustworthy CNV signal. Requires the low_complexity column; a no-op
            on runs that predate it.
        cluster_max: skip within-group clustering above this row count.
        dpi: raster resolution.
        seed: RNG seed for subsampling (reproducible figures).
        title: main title (top of figure).

    Returns:
        The output path as a Path.
    """
    if scale not in ("log", "linear"):
        raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}")
    for _name, _cap in (("n_ref", n_ref), ("max_obs", max_obs)):
        if _cap is not None and _cap < 1:
            raise ValueError(f"{_name} must be >= 1 (or None to disable subsampling), got {_cap}")

    matplotlib = _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    run = Path(run_dir)
    # allow_pickle=False: the run writer stores cn as float and cell_barcodes as
    # fixed-width unicode, so no pickle is needed — and disabling it prevents a
    # crafted object array in an untrusted .npz from executing code on load.
    z = np.load(run / "cn_per_segment.npz", allow_pickle=False)
    cn = z["cn"].astype(np.float32)
    barcodes = np.asarray(z["cell_barcodes"], dtype=str)
    segments = pd.read_parquet(run / "segments.parquet")
    # Read the barcode index as string *at parse time*: numeric-looking barcodes
    # (e.g. "001") would otherwise be inferred as integers — stripping the
    # leading zeros irrecoverably — and fail to match the unicode cell_barcodes,
    # yielding an all-NaN reindex that looks like "no tumor/normal cells".
    # (A post-hoc .astype(str) cannot help: the zeros are already gone.)
    pred_path = run / "prediction.csv"
    barcode_col = pd.read_csv(pred_path, nrows=0).columns[0]
    pred = pd.read_csv(pred_path, index_col=0, dtype={barcode_col: str}).reindex(barcodes)

    if sample is None:
        qc_path = run / "qc.json"
        sample = json.loads(qc_path.read_text()).get("sample", run.name) if qc_path.exists() else run.name
    if subtitle is None:
        subtitle = f"single-cell CNV calls · {scale} scale"

    cls = pred["class"].to_numpy()
    subclone = pred["subclone"].to_numpy().astype(str)
    low_c = (pred["low_complexity"].to_numpy().astype(bool)
             if "low_complexity" in pred.columns else np.zeros(len(pred), dtype=bool))
    # Rank candidate baseline cells on cn_burden — a non-negative magnitude — not on
    # tumor_score, which is a SIGNED template projection whose minimum is the most
    # anti-aligned cell rather than the most diploid one. Selecting the lowest signed
    # scores hands _recenter a baseline built from cells carrying a large real event
    # in the opposite direction to the consensus, and that frame then propagates into
    # the heatmap, cn_per_segment_denoised.npz, chr_cnv_matrix.csv, {sample}_clones.seg
    # and the matched-bulk concordance. Fall back to |tumor_score| for prediction.csv
    # files written before cn_burden existed, which is the same ordering wherever the
    # old score was non-negative anyway.
    if "cn_burden" in pred.columns:
        score = pred["cn_burden"].to_numpy()
    elif "tumor_score" in pred.columns:
        score = np.abs(pred["tumor_score"].to_numpy())
    else:
        score = None

    # Low-complexity (ambient) cells are dropped from the plot by default;
    # --show-low-complexity keeps them visible. Either way they NEVER define the
    # recentering baseline (see is_ref below), so a display toggle can't shift the
    # frame.
    plotted = ~low_c if drop_low_complexity else np.ones(len(pred), dtype=bool)
    n_dropped = int((~plotted).sum())

    norm = cls == "normal"
    is_tum = (cls == "tumor") & plotted
    is_unc = (cls == "uncertain") & plotted

    # Confident-diploid reference: non-junk normals with the lowest CNV signal, so
    # tumor cells under-called as normal (real CNV) don't contaminate the frame.
    #   is_ref     — the recentering/denoise baseline. ALWAYS excludes low-complexity
    #                cells, independent of drop_low_complexity.
    #   ref_panel  — the reference rows actually drawn (respects the display toggle).
    # Under the default they are identical; --show-low-complexity only widens the
    # drawn panel, never the baseline.
    # Signed, reference-relative signal recentered against the confident-diploid
    # reference then denoised — the SAME transform `run --denoise-outputs` writes,
    # so a rendered heatmap and a denoised output file agree cell-for-cell.
    rc, is_ref, thr = reference_relative_signal(
        cn, cls, low_c, score, segments,
        sd_amplifier=sd_amplifier, ref_score_quantile=ref_score_quantile,
    )
    ref_panel = norm & plotted & (score <= thr) if thr is not None else norm & plotted

    if is_tum.sum() == 0:
        raise ValueError("no tumor cells to plot as observations.")
    n_ref_total, n_tum_total, n_unc_total = int(ref_panel.sum()), int(is_tum.sum()), int(is_unc.sum())
    # Normals above the score quantile are held out of the reference and appear in
    # no panel; disclose the count in the legend rather than dropping them silently.
    n_excl_norm = int((norm & plotted & ~ref_panel).sum())

    # Reorder columns into genome position for the drawn image (display-only).
    layout = _genome_layout(segments)
    rc = rc[:, layout["col_order"]]
    x_edges, bounds, mids, names = layout["x_edges"], layout["bounds"], layout["mids"], layout["names"]

    rng = np.random.default_rng(seed)

    # Colour norm + display transform (log deviation, or exp() → 1.0-centred ratio).
    if scale == "linear":
        norm = TwoSlopeNorm(vmin=float(np.exp(-vlim)), vcenter=1.0, vmax=float(np.exp(vlim)))
        def tf(m):
            return np.exp(m)
        cbar_label = "CN ratio vs reference (linear)"
        cbar_ticks = [float(np.exp(-vlim)), 1.0, float(np.exp(vlim))]
        cbar_ticklabels = [f"{t:.2f}" for t in cbar_ticks]
    else:
        norm = Normalize(vmin=-vlim, vmax=vlim)
        def tf(m):
            return m
        cbar_label = "CN deviation vs reference (log-space)"
        cbar_ticks = [-vlim, 0.0, vlim]
        cbar_ticklabels = [f"{t:+.2f}" for t in cbar_ticks]

    # Order each panel: observations grouped by subclone, references/uncertain as
    # single clustered blocks. Row ordering uses the log-space rc so log and
    # linear figures stay pixel-aligned.
    ref_oidx, _, ref_edges, _ = _order_panel(np.where(ref_panel)[0], rc, cluster_max, n_ref, rng)
    tum_oidx, tum_groups, tum_edges, tum_labels = _order_panel(
        np.where(is_tum)[0], rc, cluster_max, max_obs, rng, subclone=subclone)
    color_of = {sc: SUBCLONE_COLORS[i % len(SUBCLONE_COLORS)] for i, sc in enumerate(tum_labels)}

    panels = [
        {"disp": tf(rc[ref_oidx]), "annot": [NORMAL_COLOR] * len(ref_oidx),
         "edges": ref_edges, "label": "References\n(confident diploid)"},
        {"disp": tf(rc[tum_oidx]), "annot": [color_of.get(g, _UNASSIGNED_COLOR) for g in tum_groups],
         "edges": tum_edges, "label": "Observations\n(tumor cells)"},
    ]
    if n_unc_total > 0:
        unc_oidx, _, unc_edges, _ = _order_panel(np.where(is_unc)[0], rc, cluster_max, n_ref, rng)
        panels.append({"disp": tf(rc[unc_oidx]), "annot": [UNCERTAIN_COLOR] * len(unc_oidx),
                       "edges": unc_edges, "label": "Uncertain\n(flagged)"})

    cmap = plt.get_cmap("RdBu_r")

    # ── figure ──────────────────────────────────────────────────────────────
    max_rows = max(p["disp"].shape[0] for p in panels)
    height_ratios = [max(p["disp"].shape[0], 1) for p in panels] + [max(max_rows, 1) * 0.04]
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(
        len(panels) + 1, 2, figure=fig,
        height_ratios=height_ratios,
        width_ratios=[0.11, 6],
        hspace=0.05, wspace=0.008,
        left=0.055, right=0.93, top=0.9, bottom=0.11,
    )

    def draw_heat(ax, mat):
        qm = ax.pcolormesh(x_edges, np.arange(mat.shape[0] + 1), mat, cmap=cmap,
                           norm=norm, rasterized=True, shading="flat")
        ax.set_ylim(mat.shape[0], 0)
        ax.set_xlim(0, x_edges[-1])
        for b in bounds:
            ax.axvline(b, color="black", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        return qm

    def draw_annot(ax, colors):
        rgb = np.array([to_rgb(c) for c in colors]).reshape(len(colors), 1, 3)
        ax.imshow(rgb, aspect="auto", extent=[0, 1, len(colors), 0], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def right_label(ax, text):
        r = ax.twinx()
        r.set_yticks([]); r.set_ylabel(text, fontsize=11)

    qm = None
    for i, p in enumerate(panels):
        draw_annot(fig.add_subplot(gs[i, 0]), p["annot"])
        ax = fig.add_subplot(gs[i, 1])
        q = draw_heat(ax, p["disp"])
        qm = qm if qm is not None else q
        for e in p["edges"]:  # thin separators between subclone blocks
            ax.axhline(e, color="black", lw=0.6)
        right_label(ax, p["label"])

    # chromosome label strip
    ax_lab = fig.add_subplot(gs[len(panels), 1])
    ax_lab.set_xlim(0, x_edges[-1]); ax_lab.set_ylim(0, 1)
    for b in bounds:
        ax_lab.axvline(b, color="black", lw=0.6)
    for mid, name in zip(mids, names):
        ax_lab.text(mid, 0.5, name, ha="center", va="center", fontsize=8)
    ax_lab.set_xticks([]); ax_lab.set_yticks([])
    for sp in ax_lab.spines.values():
        sp.set_visible(False)
    ax_lab.set_xlabel("Genomic region", fontsize=12, labelpad=6)

    # title + subtitle
    fig.suptitle(title, fontsize=22, fontweight="bold", x=0.49, y=0.965)
    fig.text(0.49, 0.925, f"{sample} · {subtitle}", ha="center", fontsize=12, color="#444444")

    # colorbar (top-left, echoing inferCNV's legend placement)
    cax = fig.add_axes([0.055, 0.945, 0.16, 0.014])
    cb = fig.colorbar(qm, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=8, labelpad=3)
    cb.set_ticks(cbar_ticks)
    cb.set_ticklabels(cbar_ticklabels)
    cb.ax.tick_params(labelsize=7)
    cax.text(-0.04, 0.5, "loss", transform=cax.transAxes, ha="right", va="center", fontsize=8)
    cax.text(1.04, 0.5, "gain", transform=cax.transAxes, ha="left", va="center", fontsize=8)

    # categorical legend (bottom). Counts are the full per-group totals among the
    # plotted cells (not the possibly-subsampled display).
    handles = [Patch(facecolor=NORMAL_COLOR, label=f"Confident-diploid reference (n={n_ref_total:,})")]
    for label in tum_labels:
        n_full = int((is_tum & (subclone == label)).sum()) if label else n_tum_total
        handles.append(Patch(facecolor=color_of[label], label=f"{label or 'tumor'} (n={n_full:,})"))
    if n_unc_total > 0:
        handles.append(Patch(facecolor=UNCERTAIN_COLOR, label=f"Uncertain (n={n_unc_total:,})"))
    if n_excl_norm:
        handles.append(Patch(facecolor="#ffffff", edgecolor="#bbbbbb",
                             label=f"normal, reference-excluded (n={n_excl_norm:,})"))
    if n_dropped:
        handles.append(Patch(facecolor="#ffffff", edgecolor="#bbbbbb",
                             label=f"low-complexity dropped (n={n_dropped:,})"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=True, fontsize=9, bbox_to_anchor=(0.49, 0.02))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
