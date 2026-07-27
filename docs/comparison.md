# kopya: how it relates to the other callers

Where kopya sits relative to the existing scRNA-seq CNV callers.

## One-line summary

> "What you'd get if you took the best ideas from CopyKAT and SCEVAN, dropped the R install pain, kept everything sparse, and shipped opinionated defaults you can validate against the originals."

Numbat stays as the allele-aware companion. The external gold-standard test suite ([`../tests/external/GOLD_STANDARD_TESTING.md`](../tests/external/GOLD_STANDARD_TESTING.md)) is how kopya is validated against independently-established CNV ground truth.

## Closest peer: infercnvpy

[infercnvpy](https://github.com/icbi-lab/infercnvpy) is the most direct comparison, and the one worth reading carefully: it is also pure-Python, matrix-only, scanpy-native, and derived from Broad inferCNV. The honest distinction is scope, not language or speed.

**infercnvpy is a CNV-signal toolkit; kopya is an end-to-end caller.** infercnvpy computes a smoothed CNV matrix and leaves baseline selection, segmentation, and the tumor/normal decision to you (or delegates the actual call to R CopyKAT via `tl.copykat`). kopya automates all three.

### Same core idea

Both center position-ordered expression against a normal reference and smooth along the chromosome. infercnvpy uses a pyramidal (triangular) running mean (window 100 genes, step 10) over clipped log-fold-change; kopya uses a uniform moving average. On the pure CNV-signal step the two are the same family, and infercnvpy's core is actually faster than kopya's (see the head-to-head below). The divergence is entirely in what happens after the smoothing.

### What kopya adds that infercnvpy leaves to you

| Layer | infercnvpy | kopya |
|---|---|---|
| Baseline / normal pool | You label normals (`reference_cat`), or it averages all cells (which silently inverts on high-purity or mesenchymal samples) | Automatic 4-mode cascade (supervised, then UCell signatures, then variance cluster, then GMM fallback) that finds the diploid pool |
| CNV representation | Fixed-resolution smoothed matrix (window/step); no boundaries, no discrete states | PELT changepoint segmentation into discrete, variable-length segments (one shared cohort table) |
| Tumor/normal call | None native: `cnv_score` is mean-absolute CNV per Leiden cluster and you decide by eye, or `tl.copykat` shells out to R CopyKAT | Native 2-component GMM on the L1 aneuploidy score, with an explicit `uncertain` band, a Tukey outlier fence, and coherence + low-complexity QC gates |
| Subclones | Generic scanpy Leiden on all cells | Leiden on tumor-only segment CN, resolution sweep, capped at `max_subclones` |
| Outputs | `X_cnv` matrix on the AnnData | CopyKAT-drop-in `prediction.csv` / `chr_cnv_matrix.csv`, IGV `.seg`, per-segment parquet, `qc.json` |
| Gene hygiene | You prepare `.var` positions | GENCODE projection plus drops chrY / MT / HLA / cell-cycle / immunoglobulin genes |

The three that matter most: kopya **segments** the signal (it finds where copy number changes, which is what enables IGV `.seg`, focal boundaries, and CopyKAT-style output) where infercnvpy blurs at a fixed resolution; kopya **finds the baseline automatically** where infercnvpy needs labeled normals or silently inverts on the hard cases; and kopya **makes the call** with false-positive controls where infercnvpy stops at a matrix and, tellingly, hands the actual calling back to R CopyKAT.

### Where infercnvpy is equal or ahead

- Faster core CNV inference and lower peak memory at a given cell count (compact windowed matrix).
- A multi-reference "bounded" difference (min/max across reference cell types before centering) that mitigates HLA / immunoglobulin bias; kopya instead drops those genes.
- Intratumoral-heterogeneity metrics (`ithcna` / `ithgex`, the IQR of intra-group CNV/expression correlation) that kopya does not compute.
- More mature: published, maintained, citable, and long embedded in the scanpy ecosystem.

### Head to head (same real 900k-cell NSCLC matrix, subsampled)

Both tools run their full standard workflow: kopya's `run`, and infercnvpy's `infercnv` plus the PCA + Leiden + `cnv_score` needed to turn the matrix into calls.

| cells | kopya wall | infercnvpy wall | infercnvpy core | infercnvpy downstream |
|---:|---:|---:|---:|---:|
| 5,000 | 14 s | 7 s | 1.7 s | 3.6 s |
| 25,000 | 18 s | 24 s | 2.6 s | 19 s |
| 50,000 | 26 s | 61 s | 6 s | 52 s |
| 100,000 | 46 s | 167 s | 15 s | 149 s |

infercnvpy's CNV inference is fast at every size; its end-to-end time is dominated by the generic PCA + Leiden clustering required to turn the matrix into tumor calls, which scales super-linearly. kopya folds calling into the pipeline and stays near-linear, so it pulls ahead from roughly 10k cells upward. (One operational note: infercnvpy's default multiprocessing failed on macOS + Python 3.13 and needed the `fork` start method to run at all.)

## What we share with the matrix-only callers

Same lane as CopyKAT, SCEVAN, inferCNV / infercnvpy, CONICSmat:

- Matrix-only input (counts.mtx or AnnData); no BAM, no SNP pileup
- Expression-only signal; copy-neutral LOH and balanced rearrangements are invisible to all of us
- Same 5-step skeleton: normalize, pick a baseline, smooth along chromosomes, segment, call

## What we borrow deliberately

| Concept                                                          | Borrowed from | Why                                                                                |
|------------------------------------------------------------------|---------------|------------------------------------------------------------------------------------|
| Signature-based "confident normal" pool (UCell scoring)          | SCEVAN §3.2   | More principled than CopyKAT's variance trick at high tumor purity                 |
| Variance-cluster fallback when signatures don't fire             | CopyKAT §3.2  | Robust when the tumor microenvironment lacks our bundled cell types                |
| GMM fallback when variance clustering fails                      | CopyKAT §3.2  | Last-resort baseline so we never crash on degenerate inputs                        |
| Multichannel segmentation (one segment table for the whole cohort, per-cell CN inside) | SCEVAN §3.4 | Where SCEVAN's ~3-5× speedup over per-cell HMMs comes from              |
| 220 kb hg38 bin grid (CopyKAT-compatible)                        | CopyKAT §3.4  | IGV `.seg` consumers expect this resolution                                        |
| Tirosh 2016 cell-cycle filter                                    | every tool    | Cell-cycle expression dominates if you don't strip it                              |

## What we do differently

| Axis                  | Other tools                                                                                       | kopya                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Language             | R (all four matrix-only callers)                                                                  | Python (numpy/scipy/scanpy/sklearn)                                                                  |
| Install              | bioconda broken (inferCNV), GitHub two-step (SCEVAN), Rgraphviz hell (CaSpER)                     | one pip install, one conda env, no JAGS / no Rgraphviz                                              |
| Matrix layout        | Densified genes × cells (CopyKAT: 30-60 GB at 10k cells)                                          | Sparse CSR throughout M1+M2; dense only in M3+M4, ~1.8 GB peak at 10k cells vs CopyKAT's 30-60 GB   |
| Segmentation         | CopyKAT: KS-driven MCMC. SCEVAN: greedy VegaMC. inferCNV: 6-state HMM + JAGS Bayesian.            | PELT (`ruptures`, L2 cost): non-parametric changepoint detection with log-scaled penalty            |
| Smoothing            | CopyKAT: order-1 Kalman (dlm). inferCNV: pyramidinal MA, window=101.                              | Plain moving average (`scipy.ndimage.uniform_filter1d`); Kalman complexity buys no measurable accuracy and is hard to vectorize |
| GENCODE reference    | inferCNV: v27 (2017). CopyKAT: vendored ~v22 vintage.                                             | v49 (2025), current symbols, ~21k more entries                                                      |
| Tumor/normal call    | CopyKAT: Ward hclust + 2-way cut. SCEVAN: cluster on segments. inferCNV: needs supplied normals.  | 2-component GMM on L1 distance from normal-pool median, with explicit `uncertain` band               |
| Supervised override  | CopyKAT supports it; SCEVAN supports it                                                           | Same: `--norm-cell-names` short-circuits the cascade                                               |
| Subclone discovery   | CopyKAT: cut Ward dendrogram. SCEVAN: re-segment per cluster.                                     | Leiden on per-cell × per-segment CN, with resolution sweep capped at `max_subclones`                |
| Output contract      | Each tool's own format                                                                            | CSVs match CopyKAT semantics (drop-in for CopyKAT outputs); also writes IGV `.seg` + QC JSON        |

## What we don't do (deliberate non-goals)

| Capability                                          | Owned by                | Why we skip                                                                          |
|----------------------------------------------------|-------------------------|--------------------------------------------------------------------------------------|
| Allele-aware (LOH, copy-neutral, biallelic amp/del) | Numbat                  | Needs BAM ingest + Eagle2 + 10 GB phasing panel; entirely different scope            |
| BAF-based calling                                   | CaSpER, Numbat          | Same, and CaSpER is unrealistic without BAM ingest anyway                           |
| Per-segment Bayesian posteriors                     | inferCNV (HMM + JAGS)   | Our PELT segments + GMM confidence are inspectable and don't need 8-24 h of JAGS sampling |
| Mouse                                               | most tools support it   | v1 hg38 only; mouse is a GENCODE-table swap when needed                              |
| Multi-sample integration                            | SCEVAN's `multiSampleComparisonClonalCN()` | Per-sample only in v1                                                          |

## Capability-at-a-glance

| Tool          | Input         | Allele-aware | Needs normal ref?            | Subclones                | Runtime (~10k cells) | RAM (~10k cells) | Install friction |
|--------------|---------------|--------------|------------------------------|--------------------------|----------------------|------------------|------------------|
| **kopya** | matrix    | No           | No (signature/variance cascade) | Yes (Leiden)             | ~12 s                | ~1.8 GB          | Low (one pip)    |
| infercnvpy   | matrix        | No           | No (mean-of-all default)     | Via generic Leiden       | ~16 s (calling downstream dominates at scale) | ~1.2 GB | Low (one pip) |
| CopyKAT      | matrix        | No           | No (auto baseline)           | Ward dendrogram          | ~1-2 h               | 30-60 GB         | Medium (GitHub install) |
| SCEVAN       | matrix        | No           | No (signature library)       | Yes, native + tree       | ~30-60 min           | 30-60 GB         | Medium (GitHub two-step) |
| CONICSmat    | matrix + BED  | No           | No (GMM)                     | From binarized matrix    | minutes-hour         | low              | Low (R + biomaRt) |
| inferCNV     | matrix + annotations + gene-order | No | **Yes (required)**           | Leiden, HMM              | 8-24 h               | 100-250 GB       | High (JAGS, dead upstream, bioconda broken) |
| Numbat       | BAM + matrix + 1000G panel | **Yes (haplotype)** | No | Yes, lineage tree       | 1-4 h pileup + 30-90 min | 20-60 GB         | Medium-High (Eagle2 + 10 GB panel) |
| CaSpER       | matrix + BAM + known normals + BAF | **Yes (BAF)** | **Yes (required)** | No (segment-level)       | 5-30 min + BAM merge     | varies           | High (BAFExtract + Rgraphviz pain) |

## Reading guide

If you want to know:
- **Why we built our own**: [README, Motivation](../README.md#motivation)
- **The exact algorithm choices**: [README, How it works](../README.md#how-it-works)
- **How it's validated**: [README, Validation](../README.md#validation), [`../tests/external/GOLD_STANDARD_TESTING.md`](../tests/external/GOLD_STANDARD_TESTING.md)
