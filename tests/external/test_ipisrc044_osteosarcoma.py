"""External test for the UCSF osteosarcoma T1 (IPISRC044_T1).

A real-world, DNA-truthed validation case with a twist the other gold standards
don't have: a **mesenchymal** tumor. Its fibroblast/osteoblast expression program
inverts the unsupervised baseline (the cascade mislabels tumor as normal — only a
handful of cells survive as 'tumor'), so this sample is run **supervised**: the
paper's annotated non-tumor immune/stromal lineages are the normal reference and
the annotated ``Putative_Tumor_Cells`` are the tumor set.

Ground truth is matched **bulk whole-exome** CNV, distilled per-chromosome and
per-arm from two independent DNA platforms and committed at
``tests/external/data/ipisrc044_t1_bulk_cnv_truth.csv``:
    - CNVkit (log2 vs matched normal)
    - a ploidy-normalized integer-CN platform (cross-check)

What this pins down:
    1. Supervised calling recovers a large tumor population (the unsupervised
       inversion is avoided).
    2. The per-chromosome sc pseudobulk correlates strongly with bulk exome.
    3. **chr10 is a genuine arm split** — DNA-discordant: p-arm retained/gained,
       q-arm lost. The tool recovers both the arm ordering (p > q) and q-arm as a
       relative loss (below the tumor's genome-wide median). This is the headline
       "arm-flip" finding: real biology, not an artifact. (We assert the relative
       ordering + relative loss, not an absolute q-arm deletion — vs the matched
       normal, chr10q reads ~neutral, concordant with CNVkit; the "deletion" is
       only relative to tumor ploidy.)
    4. **chr22 is neutral** (guards against a historical spurious over-call).

Requires DATA_ROOT/ipisrc044-osteosarcoma/ipisrc044_t1.h5ad (skip if absent) — build
it with ``python tests/external/convert_ipisrc044_osteosarcoma.py`` (see
GOLD_STANDARD_TESTING.md).
"""

import numpy as np
import pandas as pd
import pytest

from tests.external.ground_truth import run_full_pipeline

TUMOR_LABEL = "Putative_Tumor_Cells"

# hg38 centromere midpoints (Mb) for the arm-discordant chromosomes we test.
CENTROMERE_MB = {"chr1": 123.4, "chr10": 39.8, "chr17": 25.1}

# Acceptance thresholds (measured values in comments; margins are comfortable).
MIN_TUMOR_CELLS = 400            # supervised recovers ~861; unsupervised inverts to ~7
MIN_ANNOTATED_TUMOR_RECALL = 0.70  # fraction of Putative_Tumor_Cells called tumor (~0.9)
MIN_BULK_PEARSON = 0.80          # autosome per-chr Pearson vs CNVkit (~0.877)
MIN_CHR10_ARM_GAP = 0.02         # chr10 p-arm minus q-arm tumor signal (~0.095)
MIN_CHR10Q_REL_LOSS = 0.02       # how far chr10q must sit below the genome median (~0.08)
MAX_CHR22_ABS_DEVIATION = 0.05   # |chr22 centered log2| — neutral (~0.001)

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


@pytest.fixture(scope="module")
def osteo_pipeline(ipisrc044_data, tmp_path_factory):
    """Run the SUPERVISED pipeline on IPISRC044_T1 and load the committed bulk truth.

    Builds the supervised normal reference from obs['annot'] (every non-tumor,
    non-null lineage), runs the full M1->M4 pipeline against it, and returns the
    pipeline outputs plus the annotation labels and the parsed truth table.
    """
    from anndata import read_h5ad

    h5ad_path, truth_csv_path = ipisrc044_data
    out_dir = tmp_path_factory.mktemp("ipisrc044_t1_out")

    adata = read_h5ad(h5ad_path)
    annot = adata.obs["annot"]

    # Supervised normal reference = all annotated non-tumor lineages.
    normal_barcodes = adata.obs_names[annot.notna() & (annot != TUMOR_LABEL)]
    norm_cell_path = out_dir / "norm_cells.txt"
    norm_cell_path.write_text("\n".join(normal_barcodes) + "\n")

    prediction_df, chr_matrix, chroms_present, segments, qc = run_full_pipeline(
        adata=adata,
        sample_name="ipisrc044_t1",
        out_dir=out_dir,
        norm_cell_path=str(norm_cell_path),
        complexity_gate=True,
    )

    truth = pd.read_csv(truth_csv_path, comment="#")

    return {
        "prediction_df": prediction_df,
        "chr_matrix": chr_matrix,
        "chroms_present": chroms_present,
        "segments": segments,
        "qc": qc,
        "annot": annot,
        "truth": truth,
    }


def _tumor_pseudobulk_log2(osteo_pipeline):
    """Per-chromosome log2 of the tumor-cell mean of the 1.0-centered matrix."""
    pred = osteo_pipeline["prediction_df"]
    chr_matrix = osteo_pipeline["chr_matrix"]
    chroms = osteo_pipeline["chroms_present"]
    tumor_mask = (pred["class"] == "tumor").to_numpy()
    pseudobulk = chr_matrix[tumor_mask, :].mean(axis=0)
    return pd.Series(np.log2(np.clip(pseudobulk, 1e-9, None)), index=chroms)


def _chr_truth(osteo_pipeline, column):
    """Per-chromosome truth Series (autosomes) from the committed CSV."""
    t = osteo_pipeline["truth"]
    chrom = t[t["level"] == "chromosome"].set_index("region")[column]
    return chrom.reindex(AUTOSOMES)


def _arm_mean(segments, chrom, arm):
    """n_genes-weighted mean of per-segment tumor_mean for one arm of a chrom.

    Segments are assigned to p/q by their genomic midpoint vs the hg38 centromere.
    """
    cen = CENTROMERE_MB[chrom]
    seg = segments[segments["chr"] == chrom].copy()
    mid_mb = (seg["start_bp"] + seg["end_bp"]) / 2 / 1e6
    keep = (mid_mb < cen) if arm == "p" else (mid_mb >= cen)
    seg = seg[keep]
    if seg["n_genes"].sum() == 0:
        return np.nan
    return float(np.average(seg["tumor_mean"], weights=seg["n_genes"]))


def _genome_median_tumor_mean(segments):
    """Gene-count-weighted median of per-segment tumor_mean across the genome.

    The reference level a per-arm mean is 'relatively lost' below. tumor_mean is
    already centered against the supervised normal reference, so this median is
    the tumor's own genome-wide center — a chromosome/arm below it is a loss
    relative to the (aneuploid) genome, which is the frame the bulk truth is in.
    """
    vals = np.repeat(segments["tumor_mean"].to_numpy(),
                     segments["n_genes"].to_numpy().astype(int))
    return float(np.median(vals))


# ── Tests ────────────────────────────────────────────────────────────────────

def test_osteo_pipeline_runs(osteo_pipeline):
    """Supervised pipeline completes and recovers a real tumor population.

    Guards against the mesenchymal-inversion failure mode: run unsupervised, the
    cascade collapses to a handful of 'tumor' cells; supervised must recover the
    bulk of the annotated tumor.
    """
    qc = osteo_pipeline["qc"]
    assert qc["n_cells_filtered"] > 0, "All cells filtered out."
    assert qc["n_segments"] > 0, "No segments detected; segmentation failed."
    assert qc["n_tumor"] >= MIN_TUMOR_CELLS, (
        f"Only {qc['n_tumor']} tumor cells called (supervised); expected "
        f">= {MIN_TUMOR_CELLS}. A tiny tumor pool signals the mesenchymal "
        "baseline inversion — the supervised reference did not take."
    )


def test_osteo_supervised_tumor_recall(osteo_pipeline):
    """Most annotated Putative_Tumor_Cells are called tumor (recall >= 0.70)."""
    pred = osteo_pipeline["prediction_df"]
    annot = osteo_pipeline["annot"]
    joined = pred.join(annot.rename("annot"), how="inner")
    annotated_tumor = joined[joined["annot"] == TUMOR_LABEL]
    assert len(annotated_tumor) > 100, (
        f"Only {len(annotated_tumor)} annotated tumor cells joined; "
        "annotation/prediction barcode join likely broke."
    )
    recall = (annotated_tumor["class"] == "tumor").mean()
    assert recall >= MIN_ANNOTATED_TUMOR_RECALL, (
        f"Annotated-tumor recall {recall:.3f} "
        f"({int((annotated_tumor['class'] == 'tumor').sum())}/{len(annotated_tumor)}) "
        f"< {MIN_ANNOTATED_TUMOR_RECALL}."
    )


def test_osteo_bulk_concordance(osteo_pipeline):
    """sc tumor pseudobulk correlates with bulk exome across autosomes (r >= 0.80).

    Both profiles are median-centered across chromosomes before the Pearson r
    (shift-invariant; keeps the comparison in the same relative frame the bulk
    truth was distilled in).
    """
    sc = _tumor_pseudobulk_log2(osteo_pipeline).reindex(AUTOSOMES)
    bulk = _chr_truth(osteo_pipeline, "cnvkit_log2")

    x = bulk.to_numpy(dtype=float)
    y = sc.to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    assert ok.sum() >= 20, f"Only {ok.sum()} autosomes in common; join broke."

    xc = x[ok] - np.median(x[ok])
    yc = y[ok] - np.median(y[ok])
    r = float(np.corrcoef(xc, yc)[0, 1])
    assert r >= MIN_BULK_PEARSON, (
        f"Autosome bulk-exome Pearson r {r:.3f} < {MIN_BULK_PEARSON} "
        f"(n={ok.sum()} chromosomes)."
    )


def test_osteo_chr10_arm_split(osteo_pipeline):
    """chr10 is a real arm split — p-arm above q-arm, and q-arm relatively lost.

    Bulk truth is arm-discordant: CNVkit p +0.585 / q -0.041; the ploidy-normalized
    platform p Neutral / q 100% Deletion. Two things the expression caller can and
    must recover:

    1. **Arm ordering** — chr10p sits above chr10q (both DNA platforms agree on
       this regardless of their differing absolute levels; it is the frame-robust
       fact).
    2. **q-arm relative loss** — chr10q sits below the tumor's genome-wide median,
       i.e. it is among the most-lost arms (measured 4th-lowest of ~39).

    We deliberately do NOT assert an absolute q-arm loss: tumor_mean is centered
    against the matched normal reference, where chr10q reads ~neutral (+0.01,
    concordant with CNVkit's -0.04). The 'deletion' call is only relative to tumor
    ploidy. Asserting relative ordering + relative loss captures the real,
    DNA-consistent biology without overclaiming an absolute deletion the
    matched-normal frame does not show.
    """
    segments = osteo_pipeline["segments"]
    p = _arm_mean(segments, "chr10", "p")
    q = _arm_mean(segments, "chr10", "q")
    genome_median = _genome_median_tumor_mean(segments)
    assert np.isfinite(p) and np.isfinite(q), (
        f"Missing chr10 arm signal (p={p}, q={q}); segmentation/coords issue."
    )
    # (1) arm ordering: p sits above q.
    assert p - q >= MIN_CHR10_ARM_GAP, (
        f"chr10 arm ordering not recovered: p-arm mean {p:.4f} vs q-arm mean "
        f"{q:.4f} (gap {p - q:+.4f}, required >= {MIN_CHR10_ARM_GAP})."
    )
    # (2) q-arm is a relative loss: below the tumor's genome-wide median.
    assert q <= genome_median - MIN_CHR10Q_REL_LOSS, (
        f"chr10q not a relative loss: q-arm mean {q:.4f} vs genome median "
        f"{genome_median:.4f} (required q <= median-{MIN_CHR10Q_REL_LOSS})."
    )

    # Sanity: the committed truth itself encodes the same arm ordering.
    truth = osteo_pipeline["truth"].set_index("region")
    assert truth.loc["chr10p", "cnvkit_log2"] > truth.loc["chr10q", "cnvkit_log2"]


def test_osteo_chr22_neutral(osteo_pipeline):
    """chr22 is called ~neutral, matching bulk (guards a historical over-call).

    An earlier ad-hoc run showed a spurious chr22 ~+0.12 gain; bulk (both DNA
    platforms) says chr22 is neutral. The current tool must keep chr22 on the
    diagonal.
    """
    sc = _tumor_pseudobulk_log2(osteo_pipeline).reindex(AUTOSOMES)
    y = sc.to_numpy(dtype=float)
    ok = np.isfinite(y)
    chr22_centered = float(sc["chr22"] - np.median(y[ok]))
    assert abs(chr22_centered) <= MAX_CHR22_ABS_DEVIATION, (
        f"chr22 centered log2 {chr22_centered:+.4f} exceeds "
        f"+/-{MAX_CHR22_ABS_DEVIATION}; bulk says chr22 is neutral."
    )
