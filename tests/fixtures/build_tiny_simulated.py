"""Builder for tests/fixtures/tiny_simulated.h5ad — the synthetic CI fixture.

Builds a small (500 cells × ~2000 genes) AnnData with three planted CNV events
spanning the full range of difficulty for an expression-based caller:

  - chr7 broad gain    (easy: many genes, large fold change)
  - chr10 broad loss   (easy: many genes, large fold change)
  - focal MYC amp      (hard: ~30 genes around MYC on chr8 — tests focal recall)

200 cells are "normal" (no planted events) and 300 are "tumor" (all events).
Ground truth is stored in adata.uns so the end-to-end test can assert recovery
without hard-coding expected segment indices.

Run from the package root after the package is installed:
    python -m tests.fixtures.build_tiny_simulated

The output .h5ad is committed so CI does not depend on this script at test time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from kopya.annotations import load_gene_order


# Deterministic seed — the fixture must be byte-identical across rebuilds so
# CI assertions never drift under regeneration.
SEED = 2026

# Cell counts: small enough to keep the .h5ad tiny but large enough that
# Leiden subclone discovery + GMM tumor call are above the noise floor.
N_NORMAL = 200
N_TUMOR = 300

# Per-gene Poisson rate for the baseline noisy counts. ~lam=3 is a realistic
# 10x droplet average for moderately-expressed genes.
BACKGROUND_LAM = 3.0

# Multiplicative shifts for the three planted events. Tumor cells multiply
# their raw counts in these regions by the listed factor.
CHR7_AMP_FACTOR = 2.5    # chr7 broad gain → ~+0.4 to +0.5 in log space
CHR10_LOSS_FACTOR = 0.4  # chr10 broad loss → ~-0.4 to -0.5 in log space
MYC_AMP_FACTOR = 5.0     # focal — large factor compensates for the tiny region


def _build_gene_list(rng):
    """Assemble a gene list mixing chr7 / chr10 / chr8-MYC regions with filler.

    The function does NOT enforce genomic order in the var index — M1's
    project_onto_genome step does that. We just need the symbols to exist
    in the bundled GENCODE table.

    Args:
        rng: numpy Generator for reproducible filler-gene sampling.

    Returns:
        Tuple (gene_names: list[str], regions: dict). `regions` maps event
        name → list of gene symbols in that event's footprint; used for
        both planting and ground-truth recording.
    """
    go = load_gene_order()

    # Broad chr7 amp footprint: first 200 chr7 genes by genomic position.
    chr7_amp_genes = go[go["chr"] == "chr7"].index.tolist()[:200]

    # Broad chr10 loss footprint: first 200 chr10 genes by genomic position.
    chr10_loss_genes = go[go["chr"] == "chr10"].index.tolist()[:200]

    # Focal MYC amp: MYC + ~15 genes flanking it on chr8. MYC sits at
    # chr8:127_736_062–127_741_434 in GENCODE v49 hg38.
    chr8 = go[go["chr"] == "chr8"].copy().sort_values("start")
    if "MYC" not in chr8.index:
        # Defensive fallback: pick any 30 contiguous chr8 genes near the middle.
        myc_window = chr8.index.tolist()[len(chr8) // 2 - 15 : len(chr8) // 2 + 15]
    else:
        myc_pos = chr8.index.get_loc("MYC")
        lo = max(0, myc_pos - 15)
        hi = min(len(chr8), myc_pos + 15)
        myc_window = chr8.index.tolist()[lo:hi]

    # Filler genes from other autosomes so M1 has a healthy multi-chromosome
    # working set. Avoid chr7/chr10/chr8 (already planted) and chrY/chrX (
    # M1 drops chrY; we keep chrX out for simplicity).
    autosomes = [f"chr{n}" for n in range(1, 23)]
    used = set(chr7_amp_genes) | set(chr10_loss_genes) | set(myc_window)
    filler_pool = go[
        (go["chr"].isin(autosomes))
        & (~go["chr"].isin(["chr7", "chr10", "chr8"]))
        & (~go.index.isin(used))
    ].index.tolist()
    # Target ~1400 filler genes — combined with the three planted regions we
    # land near ~1800 total, comfortably above pyucell's max_rank=1500 default.
    filler = list(rng.choice(filler_pool, size=1400, replace=False))

    # Stable gene order for the synthetic var index: planted regions first
    # so the test can locate them by name, then filler. M1's project step
    # will re-sort everything by genomic position.
    gene_names = list(chr7_amp_genes) + list(chr10_loss_genes) + list(myc_window) + list(filler)

    regions = {
        "chr7_amp": list(chr7_amp_genes),
        "chr10_loss": list(chr10_loss_genes),
        "focal_myc_amp": list(myc_window),
    }
    return gene_names, regions


def _plant_events(counts, gene_names, regions, tumor_start):
    """Apply multiplicative CN shifts to the tumor rows for each planted event.

    Args:
        counts: 2-D int32 ndarray (cells × genes). Modified in place.
        gene_names: list[str] — column-to-symbol mapping for `counts`.
        regions: dict from _build_gene_list().
        tumor_start: row index where tumor cells begin (planted events apply
            from this row onward).

    Returns:
        None — `counts` is modified in place.
    """
    # Build a name → column-index lookup once so we can scatter event regions
    # without repeated linear scans.
    name_to_col = {g: i for i, g in enumerate(gene_names)}

    def _scatter(symbols, factor):
        """Multiply tumor rows in the listed gene columns by `factor`."""
        cols = [name_to_col[g] for g in symbols if g in name_to_col]
        if not cols:
            return
        block = counts[tumor_start:, cols]
        counts[tumor_start:, cols] = (block * factor).astype(np.int32)

    _scatter(regions["chr7_amp"], CHR7_AMP_FACTOR)
    _scatter(regions["chr10_loss"], CHR10_LOSS_FACTOR)
    _scatter(regions["focal_myc_amp"], MYC_AMP_FACTOR)


def build(out_path):
    """Build tiny_simulated.h5ad at out_path with all ground truth in .uns.

    Args:
        out_path: Destination .h5ad path (parent dir must exist).
    """
    rng = np.random.default_rng(SEED)
    gene_names, regions = _build_gene_list(rng)

    n_cells = N_NORMAL + N_TUMOR
    n_genes = len(gene_names)

    # Background Poisson counts for every cell; tumor rows then get
    # event-specific multipliers layered on top.
    counts = rng.poisson(lam=BACKGROUND_LAM, size=(n_cells, n_genes)).astype(np.int32)
    _plant_events(counts, gene_names, regions, tumor_start=N_NORMAL)

    # AnnData assembly. .obs carries the ground-truth class label so the
    # end-to-end test can directly score recall without recomputing the split.
    obs = pd.DataFrame({
        "true_class": ["normal"] * N_NORMAL + ["tumor"] * N_TUMOR,
    }, index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))

    adata = AnnData(
        X=csr_matrix(counts),
        obs=obs,
        var=var,
    )

    # Ground truth in .uns so the test can pull it without parsing obs.
    adata.uns["kopya_fixture"] = {
        "seed": SEED,
        "n_normal": N_NORMAL,
        "n_tumor": N_TUMOR,
        "events": {
            "chr7_amp": {
                "chr": "chr7",
                "factor": CHR7_AMP_FACTOR,
                "n_genes_planted": len(regions["chr7_amp"]),
            },
            "chr10_loss": {
                "chr": "chr10",
                "factor": CHR10_LOSS_FACTOR,
                "n_genes_planted": len(regions["chr10_loss"]),
            },
            "focal_myc_amp": {
                "chr": "chr8",
                "factor": MYC_AMP_FACTOR,
                "n_genes_planted": len(regions["focal_myc_amp"]),
                "gene_symbols": regions["focal_myc_amp"],
            },
        },
    }

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(target)
    print(f"wrote {target} — {n_cells} cells × {n_genes} genes")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "tiny_simulated.h5ad"
    build(out)
