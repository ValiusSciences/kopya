"""Command-line interface for kopya.

Subcommands:
    kopya run          --anndata|--counts/--obs/--var|--cellranger-* --out-dir --sample
    kopya find-inputs  --patient-dir ...
    kopya plot-heatmap --run-dir ... --out ...

`run` is the end-to-end pipeline: load → filter+normalize → baseline →
smooth+segment → classify+write. Each step appends to a single qc.json so the
per-sample diagnostics live in one file alongside the call outputs.
"""

import gc
import json
import time
from math import isfinite
from pathlib import Path

from click import BadParameter
from click import Choice
from click import Path as ClickPath
from click import echo, group, option, version_option
from numpy import abs as np_abs
from numpy import asarray as np_asarray
from numpy import median as np_median
from numpy import savez_compressed

from kopya import __version__
from kopya.annotations import CANONICAL_CHROM_ORDER, load_gene_order
from kopya.baseline import (
    DEFAULT_MIN_NORMAL_CELLS,
    DEFAULT_NORMAL_SCORE_THRESHOLD,
    load_signatures,
    pick_baseline,
)
from kopya.classify import (
    DEFAULT_CALL_CONFIDENCE,
    DEFAULT_COHERENCE_GATE,
    DEFAULT_LOW_COMPLEXITY_FRAC,
    DEFAULT_MAX_SUBCLONES,
    classify_cells,
)
from kopya.heatmap import DEFAULT_MAX_OBS, DEFAULT_N_REF
from kopya.io import load_counts
from kopya.outputs import (
    compute_chr_cnv_matrix,
    segments_with_coordinates,
    write_chr_cnv_matrix_csv,
    write_clones_seg,
    write_prediction_csv,
)
from kopya.normalize import (
    DEFAULT_LOW_DR,
    DEFAULT_MIN_GENES,
    DEFAULT_UP_DR,
    filter_cells_and_genes,
    normalize_log1p,
    project_onto_genome,
)
from kopya.blocked import auto_block_size, blocked_segment_and_cn
from kopya.segment import (
    DEFAULT_MIN_SEG_GENES,
    DEFAULT_PENALTY_COEF,
)
from kopya.smooth import DEFAULT_SMOOTH_WINDOW
# detected_gene_counts is defined in pipeline.py (shared with the in-memory API);
# re-exported here because it is part of the CLI module's historical surface.
from kopya.pipeline import detected_gene_counts


@group()
@version_option(version=__version__, prog_name="kopya", message="%(prog)s %(version)s")
def cli():
    """kopya — expression-only single-cell CNV caller."""


@cli.command("run")
@option(
    "--anndata",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Path to processed_ad.h5ad (Form A). Mutually exclusive with the trio.",
)
@option(
    "--cellranger-dir",
    type=ClickPath(exists=True, file_okay=False),
    default=None,
    help="Path to a CellRanger filtered_feature_bc_matrix/ directory (Form C).",
)
@option(
    "--cellranger-h5",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Path to a CellRanger filtered_feature_bc_matrix.h5 file (Form D).",
)
@option(
    "--counts",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Path to counts.mtx (Form B). Pair with --obs and --var.",
)
@option(
    "--obs",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Path to obs.csv (Form B). Row index = cell barcode.",
)
@option(
    "--var",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Path to var.csv (Form B). Row index = gene symbol.",
)
@option(
    "--raw-count-check",
    type=Choice(["warn", "raise", "ignore"]),
    default="warn",
    show_default=True,
    help="How to react when a Form-A .h5ad supplies non-raw-count .X (e.g. "
         "log1p/CP10k) and has no layers['counts']: warn, raise, or ignore.",
)
@option(
    "--out-dir",
    type=ClickPath(file_okay=False),
    required=True,
    help="Output directory; created if missing. All artifacts land here.",
)
@option(
    "--sample",
    required=True,
    help="Sample identifier; used as a prefix on the IGV .seg output.",
)
@option(
    "--threads",
    type=int,
    default=8,
    show_default=True,
    help="Parallelism for vectorized steps.",
)
@option(
    "--subclones/--no-subclones",
    default=True,
    show_default=True,
    help="Discover subclones within tumor cells.",
)
@option(
    "--norm-cell-names",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Path to file with one normal-cell barcode per line (supervised override).",
)
@option(
    "--signatures",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help=(
        "Override the bundled non-malignant signature library (JSON, same schema "
        "as the built-in normal_signatures.json). Tunes the signature baseline "
        "mode for tissues the defaults do not fit."
    ),
)
@option(
    "--non-malignant-labels",
    default=None,
    help=(
        "Comma-separated subset of signature names to treat as 'normal' "
        "(e.g. 'T_cell,B_cell,Endothelial'). Overrides the library's allow-list "
        "— add back a label the defaults exclude (e.g. Fibroblast on "
        "fibroblast-rich normal stroma, Plasma_cell on reactive infiltrate) or "
        "drop one your tumor scores on, for a given sample."
    ),
)
@option(
    "--gene-order",
    type=ClickPath(exists=True, dir_okay=False),
    default=None,
    help="Override path to the GENCODE gene-order TSV; default = bundled v49 hg38.",
)
@option(
    "--min-genes",
    type=int,
    default=DEFAULT_MIN_GENES,
    show_default=True,
    help="Drop cells with fewer than this many genes detected.",
)
@option(
    "--low-dr",
    type=float,
    default=DEFAULT_LOW_DR,
    show_default=True,
    help="Drop genes with detection rate below this fraction of cells.",
)
@option(
    "--up-dr",
    type=float,
    default=DEFAULT_UP_DR,
    show_default=True,
    help="Drop genes with detection rate above this fraction of cells.",
)
@option(
    "--normal-score-threshold",
    type=float,
    default=DEFAULT_NORMAL_SCORE_THRESHOLD,
    show_default=True,
    help="UCell top-score threshold for the signature-based baseline.",
)
@option(
    "--min-normal-cells",
    type=int,
    default=DEFAULT_MIN_NORMAL_CELLS,
    show_default=True,
    help="Cascade to variance/GMM modes if the signature pool is below this size.",
)
@option(
    "--smooth-window",
    type=int,
    default=DEFAULT_SMOOTH_WINDOW,
    show_default=True,
    help="Moving-average window (in genes) used to smooth the centered signal along chromosomes.",
)
@option(
    "--penalty-coef",
    type=float,
    default=DEFAULT_PENALTY_COEF,
    show_default=True,
    help="BIC-style coefficient on log(n_genes) for PELT segmentation. Smaller = more segments.",
)
@option(
    "--min-seg-genes",
    type=int,
    default=DEFAULT_MIN_SEG_GENES,
    show_default=True,
    help="Minimum segment length in genes; shorter segments are merged into neighbors.",
)
@option(
    "--call-confidence",
    type=float,
    default=DEFAULT_CALL_CONFIDENCE,
    show_default=True,
    help="GMM posterior threshold below which a cell is labeled 'uncertain'.",
)
@option(
    "--max-subclones",
    type=int,
    default=DEFAULT_MAX_SUBCLONES,
    show_default=True,
    help="Cap on the number of subclones reported within tumor cells.",
)
@option(
    "--coherence-gate",
    type=float,
    default=DEFAULT_COHERENCE_GATE,
    show_default=True,
    help="Downgrade a 'tumor' call to 'uncertain' when less than this fraction of "
         "its CN deviation is contiguous (chromosome-arm-scale) rather than scattered. "
         "0 disables that downgrade, and cannot make any other gate fire on a cell it "
         "otherwise spares. Lower it for focal-amplification-dominated tumors whose "
         "signal is concentrated in few segments.",
)
@option(
    "--low-complexity-frac",
    type=float,
    default=DEFAULT_LOW_COMPLEXITY_FRAC,
    show_default=True,
    help="Flag a cell low_complexity (and downgrade any 'tumor' call on it to "
         "'uncertain') when its detected-gene count is below this fraction of the "
         "cohort median. 0 disables the gate.",
)
@option(
    "--denoise-outputs/--no-denoise-outputs",
    default=False,
    show_default=True,
    help="Additionally write cn_per_segment_denoised.npz: the recentered, "
         "inferCNV-style denoised per-segment signal (the transform plot-heatmap "
         "draws), for a clean gain/loss matrix. The raw cn_per_segment.npz is "
         "always written unchanged — this is an extra file, never a replacement.",
)
@option(
    "--sd-amplifier",
    type=float,
    default=1.0,
    show_default=True,
    help="Denoise strength for --denoise-outputs, in reference-cell spread units. "
         "0 disables denoise (recenter only); higher whitens the noise floor more "
         "aggressively, at the cost of weak-event signal. Ignored without "
         "--denoise-outputs.",
)
@option(
    "--block-size",
    type=int,
    default=0,
    show_default=True,
    help="Cells per block for the memory-bounded M3 step. 0 auto-sizes each block "
         "to ~1.5 GB of dense signal (recommended). Lower it to cap peak memory "
         "further on very large cohorts; raise it for a small speedup when RAM is "
         "ample. Does not change results.",
)
def run(
    anndata,
    cellranger_dir,
    cellranger_h5,
    counts,
    obs,
    var,
    raw_count_check,
    out_dir,
    sample,
    threads,
    subclones,
    norm_cell_names,
    signatures,
    non_malignant_labels,
    gene_order,
    min_genes,
    low_dr,
    up_dr,
    normal_score_threshold,
    min_normal_cells,
    smooth_window,
    penalty_coef,
    min_seg_genes,
    call_confidence,
    max_subclones,
    coherence_gate,
    low_complexity_frac,
    denoise_outputs,
    sd_amplifier,
    block_size,
):
    """Run the full kopya pipeline on one sample.

    Stages: load → filter+normalize+project → baseline → smooth+segment+CN
    → classify+write outputs. All artifacts land under --out-dir; the qc.json
    written at the end summarizes counts and per-step timings.
    """
    # Validate the denoise strength up front (before the expensive load) so a
    # nonsensical value fails fast instead of silently writing junk: nan/inf
    # would propagate into an all-NaN/all-zero denoised matrix, and a negative
    # value is treated as "disabled" yet recorded as the applied strength.
    if denoise_outputs and not (isfinite(sd_amplifier) and sd_amplifier >= 0):
        raise BadParameter(
            f"must be a finite number >= 0 (0 disables denoise), got {sd_amplifier}.",
            param_hint="--sd-amplifier",
        )

    # Resolve the output directory eagerly so any permissions issue surfaces
    # before the (potentially expensive) load.
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 0: load raw counts ────────────────────────────────────────────
    t0 = time.time()
    echo(f"[kopya] loading counts for sample={sample!r}...")
    adata = load_counts(
        anndata=anndata,
        counts=counts, obs=obs, var=var,
        cellranger_dir=cellranger_dir,
        cellranger_h5=cellranger_h5,
        raw_count_check=raw_count_check,
    )
    load_secs = time.time() - t0
    n_cells_loaded = int(adata.n_obs)
    n_genes_loaded = int(adata.n_vars)
    echo(
        f"[kopya] loaded {n_cells_loaded} cells × {n_genes_loaded} genes "
        f"in {load_secs:.1f}s"
    )

    # ── Step 1: filter + normalize + project (§3.1) ───────────────────────
    # Load the gene-order table once and pass it through; loading the bundled
    # 77k-row table inside the helper would also work but the explicit hand-off
    # keeps the timing attributable to step 1 rather than to the inner call.
    t1 = time.time()
    echo("[kopya] step 1: filter + normalize + project onto genome...")
    gene_order_df = load_gene_order(path=gene_order)
    # Run M1 step-by-step (not via filter_normalize_project) so we can free the
    # raw matrix and each intermediate as soon as it is consumed — the raw and
    # filtered matrices are the memory high-water marks on a ~1M-cell cohort, and
    # nothing downstream needs them once M1 has produced adata_m1.
    filtered = filter_cells_and_genes(
        adata, min_genes=min_genes, low_dr=low_dr, up_dr=up_dr,
    )
    del adata
    gc.collect()
    # Deduplicate gene symbols AFTER filtering, on the matrix we own, so
    # project_onto_genome does not have to defensively copy the whole thing to do
    # it (that copy is many GB on large cohorts). Doing it post-filter is also
    # correct: if a duplicated symbol's first occurrence is dropped by QC, the
    # surviving occurrence keeps its original name (and so still matches the
    # gene-order table) — renaming pre-filter could suffix the survivor and drop
    # it. This mirrors the original placement (project_onto_genome deduped the
    # post-filter survivors).
    filtered.var_names_make_unique()
    normalized = normalize_log1p(filtered)
    del filtered
    gc.collect()
    adata_m1 = project_onto_genome(normalized, gene_order=gene_order_df)
    del normalized
    gc.collect()
    m1_secs = time.time() - t1
    echo(
        f"[kopya]   after M1: {adata_m1.n_obs} cells × {adata_m1.n_vars} genes "
        f"({m1_secs:.1f}s)"
    )

    # ── Step 2: pick the diploid baseline (§3.2) ──────────────────────────
    # Four-mode cascade: supervised → signature → variance → gmm_fallback.
    # The chosen method is the most important diagnostic in the qc JSON
    # because it tells downstream how confident the baseline call is.
    #
    # Resolve optional signature-library / allow-list overrides. A custom
    # --signatures file supplies both the gene sets and its own allow-list;
    # --non-malignant-labels then overrides just the allow-list (against the
    # custom library if given, else the bundled one). Both default to None so
    # the signature baseline falls back to the bundled library.
    #
    # Supervised mode (--norm-cell-names) bypasses the signature baseline
    # entirely, so the overrides are never consulted. In that case we skip
    # loading/validating them altogether — failing a supervised run on a custom
    # signature file or label it will not even read would be surprising — and
    # just note that they are ignored. `overrides_applied` then drives the
    # qc.json provenance flags so we never claim an override that wasn't used.
    sigs = None
    nm_labels = None
    overrides_applied = norm_cell_names is None
    if not overrides_applied:
        if signatures is not None or non_malignant_labels is not None:
            echo(
                "[kopya]   note: --signatures/--non-malignant-labels are ignored "
                "in supervised mode (--norm-cell-names takes precedence)."
            )
    else:
        if signatures is not None:
            sigs, nm_labels = load_signatures(signatures)
        if non_malignant_labels is not None:
            nm_labels = [lbl.strip() for lbl in non_malignant_labels.split(",") if lbl.strip()]
            if not nm_labels:
                raise BadParameter(
                    "no labels parsed from the value", param_hint="--non-malignant-labels"
                )
            available = sigs if sigs is not None else load_signatures(None)[0]
            unknown = [lbl for lbl in nm_labels if lbl not in available]
            if unknown:
                raise BadParameter(
                    f"{unknown} are not defined signatures; available: {sorted(available)}",
                    param_hint="--non-malignant-labels",
                )

    t2 = time.time()
    echo("[kopya] step 2: picking diploid baseline...")
    baseline = pick_baseline(
        adata_m1,
        norm_cell_path=norm_cell_names,
        signatures=sigs,
        non_malignant_labels=nm_labels,
        threshold=normal_score_threshold,
        min_normal_cells=min_normal_cells,
    )
    m2_secs = time.time() - t2
    echo(
        f"[kopya]   method={baseline['method']!r} "
        f"n_normal={baseline['n_normal']}/{adata_m1.n_obs} ({m2_secs:.1f}s)"
    )

    # ── Step 3: center + smooth + segment + per-cell CN (§3.3-§3.4) ──────
    # Three sub-steps run serially; we report a single combined timing
    # because they share a dense float32 working matrix and splitting the
    # timing across them would mostly measure allocation costs.
    t3 = time.time()
    echo("[kopya] step 3: center + smooth + segment + per-cell CN...")
    chr_labels = adata_m1.var["chr"].to_numpy()
    # M3 runs row-blocked so peak memory is bounded by one cell-block of the
    # dense matrix, not the full (n_cells × n_genes) — this is what lets large
    # cohorts (100Ks–~1M cells) run without the dense matrix blowing out RAM
    # Numerically equivalent to the direct
    # center→smooth→segment→per-cell-CN path. --block-size <= 0 auto-sizes each
    # block to ~1.5 GB of dense float32. Resolve here (any non-positive value →
    # auto) and pass the resolved value on, so the echo and qc.json record the
    # block size actually used rather than a raw sentinel like -1.
    resolved_block = auto_block_size(adata_m1.n_obs, adata_m1.n_vars) if block_size <= 0 else block_size
    echo(f"[kopya]   blocked M3: {resolved_block} cells/block")
    segments, cn_matrix, seg_baseline = blocked_segment_and_cn(
        adata_m1,
        normal_mask=baseline["mask"],
        chr_labels=chr_labels,
        block_size=resolved_block,
        penalty_coef=penalty_coef,
        min_seg_genes=min_seg_genes,
        smooth_window=smooth_window,
    )
    # Per-segment diploid reference: the median CN over the normal pool. The stored
    # cn_matrix / cn_per_segment.npz stays RAW — the relative signal (tumor_score,
    # classification) subtracts this baseline internally and is unchanged.
    #
    # This reference decomposes into a global pedestal + a per-chromosome residual.
    # The residual, on a heterogeneous normal pool, is partly tumor-correlated (the
    # known reference-frame limitation), so subtracting it from the bulk-validated
    # chr_cnv_matrix regresses the matched-bulk Pearson. We therefore split it:
    #   - tumor_mean (centered inside detect_segments) and .seg get the FULL
    #     per-segment reference — maximally correct per-segment signs (e.g. chr10
    #     reads as a clean loss); neither feeds the bulk concordance metric.
    #   - chr_cnv_matrix gets only the SCALAR global pedestal (median of the
    #     per-segment reference). Pearson is invariant to a constant shift, so this
    #     removes the gross ~1.27 diploid offset while leaving the tumor profile's
    #     per-chromosome shape — and thus the validated concordance — untouched.
    # seg_baseline is returned by blocked_segment_and_cn (== the per-segment
    # normal-pool median it already used to center tumor_mean).
    chr_pedestal = float(np_median(seg_baseline))
    m3_secs = time.time() - t3
    echo(
        f"[kopya]   {len(segments)} segments across "
        f"{int(segments['chr'].nunique())} chromosomes "
        f"({m3_secs:.1f}s)"
    )

    # Persist the intermediate M3 artifacts so M4 (and any debugging) can
    # consume them without re-running the upstream pipeline. parquet is the
    # right format for a wide DataFrame; npz for the dense CN matrix.
    segments_path = out_path / "segments.parquet"
    # Persist genomic coordinates alongside the gene-axis indices so segments.parquet
    # is a self-contained segment-to-chromosome-location mapping. Row i
    # still corresponds to column i of cn_per_segment.npz.
    segments_out = segments_with_coordinates(segments, adata_m1.var)
    segments_out.to_parquet(segments_path)
    cn_path = out_path / "cn_per_segment.npz"
    # Cast barcodes to fixed-width unicode so np.load works without allow_pickle.
    # adata.obs_names.to_numpy() returns dtype=object which npz disallows by default.
    savez_compressed(
        cn_path,
        cn=cn_matrix,
        cell_barcodes=np_asarray(adata_m1.obs_names, dtype="U"),
    )
    echo(f"[kopya]   wrote {segments_path.name} + {cn_path.name}")

    # ── Step 4: classify cells + write outputs (§3.5, §4) ─────────────────
    # Classification is fast (GMM + Leiden on a small matrix); the output
    # writers dominate this stage's timing.
    t4 = time.time()
    echo("[kopya] step 4: classify + write outputs...")
    complexity = detected_gene_counts(adata_m1.X)
    prediction_df = classify_cells(
        cn_matrix=cn_matrix,
        segments=segments,
        normal_mask=baseline["mask"],
        barcodes=adata_m1.obs_names.to_numpy(),
        confidence_threshold=call_confidence,
        max_subclones=max_subclones,
        discover_subclones_enabled=bool(subclones),
        complexity=complexity,
        coherence_gate=coherence_gate,
        low_complexity_frac=low_complexity_frac,
    )

    # prediction.csv — drop-in for copykat_prediction.csv.
    pred_path = write_prediction_csv(prediction_df, out_path / "prediction.csv")

    # chr_cnv_matrix.csv — drop-in for copykat_chr_cnv_matrix.csv, 1.0-centered.
    # Scalar pedestal (not the per-segment vector) to preserve bulk concordance.
    chr_matrix, chroms_present = compute_chr_cnv_matrix(
        cn_matrix, segments, CANONICAL_CHROM_ORDER, baseline=chr_pedestal,
    )
    chr_path = write_chr_cnv_matrix_csv(
        chr_matrix,
        barcodes=adata_m1.obs_names.to_numpy(),
        chroms=chroms_present,
        out_path=out_path / "chr_cnv_matrix.csv",
    )

    # {sample}_clones.seg — IGV-loadable per-clone consensus.
    seg_path = write_clones_seg(
        cn_matrix=cn_matrix,
        segments=segments,
        prediction_df=prediction_df,
        var_coords=adata_m1.var,
        sample=sample,
        out_path=out_path / f"{sample}_clones.seg",
        baseline=seg_baseline,
    )

    # cn_per_segment_denoised.npz — optional, purely additive. The recentered +
    # inferCNV-style denoised signal (the exact transform plot-heatmap draws),
    # baked into a file for users who just want a clean "is the hallmark event
    # there?" matrix (0 ≈ diploid, >0 gain, <0 loss). The raw cn_per_segment.npz
    # is always written unchanged above; this never replaces it.
    #
    # Always clear a stale copy first: re-running the same --out-dir with the flag
    # OFF (or a run whose denoise is skipped for lack of a reference) must not
    # leave a previous run's denoised matrix behind — its barcodes/dimensions
    # would no longer match the freshly-written raw outputs, silently exposing
    # mismatched data as a current artifact.
    den_path = out_path / "cn_per_segment_denoised.npz"
    den_path.unlink(missing_ok=True)
    if denoise_outputs:
        from kopya.heatmap import reference_relative_signal
        low_c = (prediction_df["low_complexity"].to_numpy()
                 if "low_complexity" in prediction_df.columns else None)
        # Rank baseline candidates on the non-negative magnitude, never on the
        # signed tumor_score — whose minimum is the most ANTI-aligned cell, not the
        # most diploid one. Same rule (and same fallback for older prediction.csv
        # files) as heatmap.render_heatmap.
        ref_rank = (prediction_df["cn_burden"].to_numpy()
                    if "cn_burden" in prediction_df.columns
                    else np_abs(prediction_df["tumor_score"].to_numpy()))
        try:
            rc, is_ref, _ = reference_relative_signal(
                cn_matrix,
                prediction_df["class"].to_numpy(),
                low_c,
                ref_rank,
                segments,
                sd_amplifier=sd_amplifier,
            )
            savez_compressed(
                den_path,
                cn=rc.astype("float32"),
                cell_barcodes=np_asarray(adata_m1.obs_names, dtype="U"),
                sd_amplifier=np_asarray(float(sd_amplifier)),
            )
            echo(
                f"[kopya]   wrote {den_path.name} "
                f"(sd_amplifier={sd_amplifier}, {int(is_ref.sum())} reference cells)"
            )
        except ValueError as exc:
            # No confident-diploid reference — skip the optional file (already
            # cleared above), don't fail the run (raw outputs are all written).
            echo(f"[kopya]   WARNING: --denoise-outputs skipped ({exc})")

    m4_secs = time.time() - t4

    # Per-class cell counts for the QC summary; subclone breakdown when enabled.
    class_counts = prediction_df["class"].value_counts().to_dict()
    n_tumor = int(class_counts.get("tumor", 0))
    n_normal = int(class_counts.get("normal", 0))
    n_uncertain = int(class_counts.get("uncertain", 0))
    subclone_counts = (
        prediction_df.loc[prediction_df["class"] == "tumor", "subclone"]
        .value_counts()
        .to_dict()
    )
    n_subclones_observed = len([s for s in subclone_counts if s])

    echo(
        f"[kopya]   class: tumor={n_tumor}, normal={n_normal}, "
        f"uncertain={n_uncertain}; subclones={n_subclones_observed} ({m4_secs:.1f}s)"
    )
    echo(
        f"[kopya]   wrote {pred_path.name} + {chr_path.name} + {seg_path.name}"
    )

    # ── QC JSON ───────────────────────────────────────────────────────────
    # M4 milestone: final per-class counts, subclone breakdown, and the full
    # timing budget for the run.
    qc = {
        "sample": sample,
        "kopya_version": __version__,
        "milestone": "M4",
        "threads": int(threads),
        "subclones_enabled": bool(subclones),
        "supervised": norm_cell_names is not None,
        "gene_order_overridden": gene_order is not None,
        # True only when the override was actually consumed; supervised mode
        # ignores both, so they read False there even if the flags were passed.
        "signatures_overridden": signatures is not None and overrides_applied,
        "non_malignant_labels_overridden": non_malignant_labels is not None and overrides_applied,
        "params": {
            "min_genes": int(min_genes),
            "low_dr": float(low_dr),
            "up_dr": float(up_dr),
            "normal_score_threshold": float(normal_score_threshold),
            "min_normal_cells": int(min_normal_cells),
            "smooth_window": int(smooth_window),
            "penalty_coef": float(penalty_coef),
            "min_seg_genes": int(min_seg_genes),
            "call_confidence": float(call_confidence),
            "max_subclones": int(max_subclones),
            "coherence_gate": float(coherence_gate),
            "low_complexity_frac": float(low_complexity_frac),
            "denoise_outputs": bool(denoise_outputs),
            "sd_amplifier": float(sd_amplifier) if denoise_outputs else None,
            "block_size": int(resolved_block),
        },
        "n_cells_loaded": n_cells_loaded,
        "n_genes_loaded": n_genes_loaded,
        "n_cells_post_m1": int(adata_m1.n_obs),
        "n_genes_post_m1": int(adata_m1.n_vars),
        "baseline_method": baseline["method"],
        "n_normal_seed": baseline["n_normal"],
        "baseline_diagnostics": baseline["diagnostics"],
        "n_segments": int(len(segments)),
        "n_segments_per_chr": segments["chr"].value_counts().to_dict(),
        "median_segment_genes": int(segments["n_genes"].median()) if len(segments) else 0,
        "tumor_signal_max_abs": float(segments["tumor_mean"].abs().max()) if len(segments) else 0.0,
        "n_tumor": n_tumor,
        "n_normal_called": n_normal,
        "n_uncertain": n_uncertain,
        "n_low_complexity": int(prediction_df["low_complexity"].sum()),
        # Cells the outlier fence held out of the GMM fit as probable transcriptome
        # artifacts — see pipeline.py for why this is worth recording. Normally 0.
        "n_outlier_fenced": int(prediction_df.attrs.get("n_outlier_fenced", 0)),
        # Cells the anti-alignment gate moved from "normal" to "uncertain" for
        # carrying large, coherent copy number pointing AGAINST the consensus
        # template — see pipeline.py. Normally 0 or near it.
        "n_anti_aligned": int(prediction_df.attrs.get("n_anti_aligned", 0)),
        # Tumor calls the gate's thresholds rested on; 0 means it did not run — see
        # pipeline.py.
        "anti_alignment_tumor_n": int(
            prediction_df.attrs.get("anti_alignment_tumor_n", 0)
        ),
        "n_subclones_observed": n_subclones_observed,
        "subclone_counts": subclone_counts,
        "timings_secs": {
            "load": round(load_secs, 2),
            "m1_filter_normalize_project": round(m1_secs, 2),
            "m2_pick_baseline": round(m2_secs, 2),
            "m3_smooth_segment_cn": round(m3_secs, 2),
            "m4_classify_write_outputs": round(m4_secs, 2),
        },
    }
    qc_path = out_path / "qc.json"
    # default=str so numpy ints / floats inside diagnostics serialize cleanly.
    qc_path.write_text(json.dumps(qc, indent=2, default=str) + "\n")
    echo(f"[kopya] wrote {qc_path}")


@cli.command("find-inputs")
@option(
    "--patient-dir",
    required=True,
    type=ClickPath(exists=True, file_okay=False),
    help="Root directory for a single patient (searched recursively for inputs).",
)
@option(
    "--sample",
    default=None,
    help="Sample name to use in the suggested run command (defaults to directory name).",
)
@option(
    "--out-dir",
    default=None,
    help="Output directory to include in the suggested run command.",
)
def find_inputs(patient_dir, sample, out_dir):
    """Locate kopya input files inside a patient directory tree.

    Searches for any of the four accepted input forms — CellRanger MTX directory
    (Form C) or H5 (Form D), an AnnData .h5ad (Form A), or a counts.mtx + obs.csv
    + var.csv trio (Form B) — following common single-cell pipeline output
    layouts. Prints the best candidate found and the ready-to-run 'kopya run'
    command.
    """
    import anndata

    root = Path(patient_dir)
    sample_name = sample or root.name
    out_dir_str = out_dir or f"results/{sample_name}"

    # ── Candidate search order ────────────────────────────────────────────────
    # Prefer raw counts (layers["counts"] or integer X) over processed adatas.
    # Check CellRanger layouts first (most common external format), then a broad
    # glob for any other .h5ad, then the mtx-trio layout.

    # CellRanger: the run outputs land in an outs/ subdirectory; the matrix
    # directory or H5 file may be at the root or one level down.
    CELLRANGER_MTX_CANDIDATES = [
        root / "outs" / "filtered_feature_bc_matrix",
        root / "filtered_feature_bc_matrix",
        root / "outs" / "raw_feature_bc_matrix",
        root / "raw_feature_bc_matrix",
    ]
    CELLRANGER_H5_CANDIDATES = [
        root / "outs" / "filtered_feature_bc_matrix.h5",
        root / "filtered_feature_bc_matrix.h5",
        root / "outs" / "raw_feature_bc_matrix.h5",
        root / "raw_feature_bc_matrix.h5",
    ]
    # Loupe file detection — not usable, but we can give a helpful message.
    LOUPE_CANDIDATES = list(root.glob("**/*.cloupe"))

    # Any .h5ad in the tree. Raw-count adatas are preferred over processed ones
    # by X dtype below, so no fixed filename ordering is needed.
    H5AD_CANDIDATES = sorted(root.glob("**/*.h5ad"))

    MTX_SEARCH_ROOTS = [
        root / "normalization",
        root,
    ]

    # ── Probe CellRanger formats ──────────────────────────────────────────────
    def _probe_10x_mtx(dir_path):
        """Check that an MTX directory has the required files; return cell count."""
        required = ["matrix.mtx.gz", "barcodes.tsv.gz"]
        # CellRanger v2 uses genes.tsv, v3+ uses features.tsv
        feature_files = ["features.tsv.gz", "genes.tsv.gz",
                         "features.tsv", "genes.tsv"]
        has_matrix = any((dir_path / f).exists() for f in ["matrix.mtx.gz", "matrix.mtx"])
        has_barcodes = any((dir_path / f).exists() for f in ["barcodes.tsv.gz", "barcodes.tsv"])
        has_features = any((dir_path / f).exists() for f in feature_files)
        if not (has_matrix and has_barcodes and has_features):
            return None
        # Count barcodes (= cells) from the barcodes file
        import gzip
        for bc_name in ["barcodes.tsv.gz", "barcodes.tsv"]:
            bc_path = dir_path / bc_name
            if bc_path.exists():
                try:
                    if bc_name.endswith(".gz"):
                        with gzip.open(bc_path, "rt") as fh:
                            return sum(1 for _ in fh)
                    else:
                        return sum(1 for _ in open(bc_path))
                except Exception:
                    return -1
        return -1

    def _probe_h5ad(path):
        """Return (n_obs, n_vars, has_counts_layer, x_dtype) or None on error."""
        try:
            ad = anndata.read_h5ad(path, backed="r")
            result = (
                ad.n_obs,
                ad.n_vars,
                "counts" in ad.layers,
                str(ad.X.dtype),
            )
            ad.file.close()
            return result
        except Exception:
            return None

    def _find_mtx_trio(search_root):
        """Return (counts, obs, var) paths if a valid trio exists."""
        counts = search_root / "counts.mtx"
        obs = search_root / "obs.csv"
        var = search_root / "var.csv"
        if counts.exists() and obs.exists() and var.exists():
            return counts, obs, var
        return None

    # ── Evaluate CellRanger MTX directories ──────────────────────────────────
    chosen_cellranger_mtx = None
    for candidate in CELLRANGER_MTX_CANDIDATES:
        if not candidate.exists():
            continue
        n_cells = _probe_10x_mtx(candidate)
        if n_cells is None:
            continue
        label = "filtered" if "filtered" in candidate.name else "raw"
        echo(
            f"  [found] {candidate.relative_to(root)} — "
            f"CellRanger MTX ({label}), {n_cells:,} cells"
            + (" ✓ preferred" if chosen_cellranger_mtx is None else "")
        )
        if chosen_cellranger_mtx is None:
            chosen_cellranger_mtx = candidate

    # ── Evaluate CellRanger H5 files ─────────────────────────────────────────
    chosen_cellranger_h5 = None
    for candidate in CELLRANGER_H5_CANDIDATES:
        if not candidate.exists():
            continue
        label = "filtered" if "filtered" in candidate.name else "raw"
        size_mb = candidate.stat().st_size / 1e6
        echo(
            f"  [found] {candidate.relative_to(root)} — "
            f"CellRanger H5 ({label}), {size_mb:.0f} MB"
            + (" ✓ preferred" if chosen_cellranger_h5 is None else "")
        )
        if chosen_cellranger_h5 is None:
            chosen_cellranger_h5 = candidate

    # ── Warn about Loupe files ────────────────────────────────────────────────
    for loupe in LOUPE_CANDIDATES:
        echo(
            f"  [skip]  {loupe.relative_to(root)} — Loupe .cloupe files are "
            "not supported (proprietary format). Use the filtered_feature_bc_matrix.h5 "
            "or filtered_feature_bc_matrix/ directory from the same CellRanger run."
        )

    # ── Evaluate h5ad candidates ──────────────────────────────────────────────
    chosen_h5ad = None
    seen_h5ad = set()
    for candidate in H5AD_CANDIDATES:
        if not candidate.exists() or candidate in seen_h5ad:
            continue
        seen_h5ad.add(candidate)
        info = _probe_h5ad(candidate)
        if info is None:
            echo(f"  [skip] {candidate.relative_to(root)} — could not open")
            continue
        n_obs, n_vars, has_counts, dtype = info
        label = "raw_counts" if "raw" in candidate.name else "processed"
        has_int = "int" in dtype or has_counts
        echo(
            f"  [found] {candidate.relative_to(root)} — "
            f"{n_obs:,} cells × {n_vars:,} genes | "
            f"X dtype={dtype} | layers.counts={has_counts}"
            + (" ✓ preferred" if has_int and chosen_h5ad is None else "")
        )
        if chosen_h5ad is None and has_int:
            chosen_h5ad = candidate
        elif chosen_h5ad is None:
            # Accept processed adata as fallback even if float (loader handles it)
            chosen_h5ad = candidate

    # ── Evaluate Form B (mtx trio) ────────────────────────────────────────────
    chosen_mtx = None
    for mtx_root in MTX_SEARCH_ROOTS:
        trio = _find_mtx_trio(mtx_root)
        if trio:
            counts_p, obs_p, var_p = trio
            echo(
                f"  [found] MTX trio at {mtx_root.relative_to(root)} — "
                f"counts.mtx + obs.csv + var.csv"
            )
            if chosen_mtx is None:
                chosen_mtx = trio

    # ── Decision + suggested command ─────────────────────────────────────────
    # Priority: CellRanger MTX > CellRanger H5 > h5ad > MTX trio.
    # CellRanger formats come first because they are the most common external
    # format and require no preprocessing.
    echo("")
    if chosen_cellranger_mtx is not None:
        echo(f"Recommended input: Form C — CellRanger MTX directory")
        echo(f"  {chosen_cellranger_mtx}")
        echo("\nSuggested run command:")
        echo(
            f"  kopya run \\\n"
            f"      --cellranger-dir {chosen_cellranger_mtx} \\\n"
            f"      --out-dir  {out_dir_str} \\\n"
            f"      --sample   {sample_name} \\\n"
            f"      --threads  8"
        )
    elif chosen_cellranger_h5 is not None:
        echo(f"Recommended input: Form D — CellRanger H5")
        echo(f"  {chosen_cellranger_h5}")
        echo("\nSuggested run command:")
        echo(
            f"  kopya run \\\n"
            f"      --cellranger-h5 {chosen_cellranger_h5} \\\n"
            f"      --out-dir  {out_dir_str} \\\n"
            f"      --sample   {sample_name} \\\n"
            f"      --threads  8"
        )
    elif chosen_h5ad is not None:
        echo(f"Recommended input: Form A — {chosen_h5ad.relative_to(root)}")
        echo("\nSuggested run command:")
        echo(
            f"  kopya run \\\n"
            f"      --anndata  {chosen_h5ad} \\\n"
            f"      --out-dir  {out_dir_str} \\\n"
            f"      --sample   {sample_name} \\\n"
            f"      --threads  8"
        )
    elif chosen_mtx is not None:
        counts_p, obs_p, var_p = chosen_mtx
        echo("Recommended input: Form B — MTX trio")
        echo("\nSuggested run command:")
        echo(
            f"  kopya run \\\n"
            f"      --counts  {counts_p} \\\n"
            f"      --obs     {obs_p} \\\n"
            f"      --var     {var_p} \\\n"
            f"      --out-dir {out_dir_str} \\\n"
            f"      --sample  {sample_name} \\\n"
            f"      --threads 8"
        )
    else:
        echo(
            "No usable input found. Expected one of:\n"
            "  Form C: <dir>/outs/filtered_feature_bc_matrix/  (CellRanger MTX)\n"
            "  Form D: <dir>/outs/filtered_feature_bc_matrix.h5  (CellRanger H5)\n"
            "  Form A: <dir>/**/raw_counts_ad.h5ad  (or any .h5ad)\n"
            "  Form B: <dir>/**/normalization/{counts.mtx,obs.csv,var.csv}\n\n"
            "Note: .cloupe files are not supported. Use the .h5 or MTX output\n"
            "from the same CellRanger run.\n\n"
            "If the download is still in progress, re-run once it completes."
        )


@cli.command("plot-heatmap")
@option(
    "--run-dir",
    required=True,
    type=ClickPath(exists=True, file_okay=False),
    help="A completed run out-dir (must hold cn_per_segment.npz, segments.parquet, prediction.csv).",
)
@option(
    "--out",
    "out_path",
    required=True,
    type=ClickPath(dir_okay=False),
    help="Destination image path; format inferred from the suffix (.png / .pdf / .svg).",
)
@option(
    "--scale",
    type=Choice(["log", "linear"]),
    default="log",
    show_default=True,
    help="'log' = deviation centered on 0; 'linear' = copy-number ratio centered on 1.0 (inferCNV style).",
)
@option(
    "--sample",
    default=None,
    help="Header label; defaults to qc.json's sample (else the run-dir name).",
)
@option(
    "--subtitle",
    default=None,
    help="Header subtitle; defaults to a scale-aware generic line.",
)
@option(
    "--n-ref",
    type=int,
    default=DEFAULT_N_REF,
    show_default=True,
    help="Subsample normal (reference) cells to at most this many (display only).",
)
@option(
    "--max-obs",
    type=int,
    default=DEFAULT_MAX_OBS,
    show_default=True,
    help="Subsample tumor (observation) cells to at most this many total, allocated proportionally across subclones (display only).",
)
@option(
    "--vlim",
    type=float,
    default=0.15,
    show_default=True,
    help="Symmetric colour clip in log-space; linear scale maps exp(±vlim) with white at 1.0.",
)
@option(
    "--sd-amplifier",
    type=float,
    default=1.0,
    show_default=True,
    help="inferCNV-style denoise strength: soft-threshold each cell's per-segment "
         "deviation toward 0 by this many reference-cell spreads, collapsing the "
         "normal noise floor to white while keeping real CNV events. 0 disables it.",
)
@option(
    "--ref-score-quantile",
    type=float,
    default=0.5,
    show_default=True,
    help="The reference panel (and recentering baseline) is the confident-diploid "
         "subset: non-junk normals with cn_burden (the non-negative CN magnitude, "
         "NOT the signed tumor_score) at or below this quantile of that pool. "
         "Excludes under-called tumor cells from the reference. 1.0 keeps every "
         "non-junk normal.",
)
@option(
    "--show-low-complexity",
    is_flag=True,
    default=False,
    help="Keep low-complexity (ambient / empty-droplet-like) cells in the plot. "
         "By default they are dropped from every panel as untrustworthy CNV signal.",
)
@option(
    "--dpi",
    type=int,
    default=200,
    show_default=True,
    help="Raster resolution for the saved figure.",
)
@option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed for subsampling (reproducible figures).",
)
def plot_heatmap(run_dir, out_path, scale, sample, subtitle, n_ref, max_obs, vlim, sd_amplifier,
                 ref_score_quantile, show_low_complexity, dpi, seed):
    """Render an inferCNV-style genome CNV heatmap from a run out-dir.

    References (normal cells) on top, observations (tumor cells) below, genome
    left-to-right with chromosome widths proportional to gene content, and
    blue→white→red diverging colour (loss→diploid→gain). Tumor cells are grouped
    by their subclone call and annotated with a colour bar.
    """
    # Fail fast with a clear message before importing the heavy plotting stack.
    run = Path(run_dir)
    missing = [
        name for name in ("cn_per_segment.npz", "segments.parquet", "prediction.csv")
        if not (run / name).exists()
    ]
    if missing:
        raise BadParameter(
            f"{run_dir} is missing required artifact(s): {', '.join(missing)}. "
            "Point --run-dir at a completed `kopya run` out-dir."
        )

    from kopya.heatmap import render_heatmap

    out = render_heatmap(
        run_dir=run_dir,
        out_path=out_path,
        sample=sample,
        subtitle=subtitle,
        scale=scale,
        n_ref=n_ref,
        max_obs=max_obs,
        vlim=vlim,
        sd_amplifier=sd_amplifier,
        ref_score_quantile=ref_score_quantile,
        drop_low_complexity=not show_low_complexity,
        dpi=dpi,
        seed=seed,
    )
    echo(f"[kopya] wrote {out} ({scale} scale)")


if __name__ == "__main__":
    cli()
