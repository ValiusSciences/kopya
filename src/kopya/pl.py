"""Scanpy-style plotting namespace for CNV results stored on an AnnData.

    import kopya as kp
    kp.tl.cnv(adata, reference_key="cell_type", reference_cat=[...])
    kp.pl.chromosome_heatmap(adata, groupby="cell_type")

This draws the per-chromosome matrix that ``tl.cnv`` writes to
``adata.obsm["cnv_chr"]`` (chromosome resolution — one column per autosome).
For the full per-segment, genome-position heatmap, use the CLI:
``kopya plot-heatmap --run-dir <out>``.
"""

import numpy as np


def chromosome_heatmap(adata, *, groupby=None, key="cnv", vmax=0.15, ax=None,
                       show_group_colors=True):
    """Heatmap of the per-chromosome CNV matrix, rows optionally grouped.

    Args:
        adata: AnnData annotated by ``tl.cnv`` (needs obsm[f"{key}_chr"] and
            uns[key]).
        groupby: obs column to sort/group rows by (e.g. "cell_type" or
            f"{key}_class"). Cells with a NaN group are dropped. Default: the
            per-cell class, f"{key}_class".
        key: the key prefix used by ``tl.cnv`` (default "cnv").
        vmax: symmetric colour clip in log2 space (blue=loss, red=gain).
        ax: optional matplotlib Axes to draw into.
        show_group_colors: draw a left colour bar for the groupby categories.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mkey = f"{key}_chr"
    if mkey not in adata.obsm or key not in adata.uns:
        raise KeyError(
            f"adata is missing {mkey!r}/uns[{key!r}]; run kopya.tl.cnv first."
        )
    chroms = list(adata.uns[key]["chroms"])
    M = np.asarray(adata.obsm[mkey], dtype=float)          # (n_obs x n_chroms), 1.0-centered
    if groupby is None:
        groupby = f"{key}_class"
    if groupby not in adata.obs:
        raise KeyError(
            f"groupby column {groupby!r} not in adata.obs; pass a valid obs column "
            "(or run tl.cnv, which adds the default cnv_class)."
        )
    groups = adata.obs[groupby]

    # keep only scored cells (non-NaN across the matrix) and a non-NaN group
    scored = np.isfinite(M).all(axis=1) & groups.notna().to_numpy()
    if not scored.any():
        raise ValueError(
            f"No cells to plot: every scored cell has a missing {groupby!r} value "
            "(or no cells survived scoring). Pass a groupby with labelled cells."
        )
    M = M[scored]
    g = groups[scored].astype(str).to_numpy()
    order = np.argsort(g, kind="stable")
    M = M[order]
    row_group = g[order]

    # 1.0-centered -> log2 deviation for a symmetric diverging map.
    L = np.log2(np.clip(M, 1e-6, None))

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(L, aspect="auto", cmap="RdBu_r",
              norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
              interpolation="nearest")
    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels([c.replace("chr", "") for c in chroms], fontsize=8)
    ax.set_xlabel("chromosome")
    ax.set_yticks([])
    ax.set_ylabel(f"cells (grouped by {groupby})")

    if show_group_colors:
        cats = list(dict.fromkeys(row_group))
        cmap = plt.get_cmap("tab20")
        cat_color = {c: cmap(i % 20) for i, c in enumerate(cats)}
        # imshow spans x in [-0.5, n_chroms-0.5] with row i centered at y=i
        # (cell i-0.5..i+0.5). Put the strip strictly left of the image and align
        # to row bounds so it neither overlaps chr1 nor shifts half a row. Rows are
        # group-sorted, so draw ONE rectangle per contiguous run (not per cell) —
        # a per-cell loop dominates rendering on tens-of-thousands-cell objects.
        width = max(0.4, len(chroms) * 0.03)
        left = -0.5 - width
        changes = np.nonzero(row_group[1:] != row_group[:-1])[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(row_group)]])
        for s, e in zip(starts, ends):
            ax.add_patch(plt.Rectangle((left, s - 0.5), width, e - s,
                                       color=cat_color[row_group[s]], lw=0, clip_on=False))
        ax.set_xlim(left, len(chroms) - 0.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=cat_color[c], label=c) for c in cats],
                  bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, frameon=False)
    return ax
