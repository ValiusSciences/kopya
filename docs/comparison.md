# kopya — how it relates to the other callers

Where kopya sits relative to the existing scRNA-seq CNV callers.

## One-line summary

> "What you'd get if you took the best ideas from CopyKAT and SCEVAN, dropped the R install pain, kept everything sparse, and shipped opinionated defaults you can validate against the originals."

Numbat stays as the allele-aware companion. The external gold-standard test suite ([`../tests/external/GOLD_STANDARD_TESTING.md`](../tests/external/GOLD_STANDARD_TESTING.md)) is how kopya is validated against independently-established CNV ground truth.

## What we share with the matrix-only callers

Same lane as CopyKAT, SCEVAN, inferCNV, CONICSmat:

- Matrix-only input (counts.mtx or AnnData); no BAM, no SNP pileup
- Expression-only signal — copy-neutral LOH and balanced rearrangements are invisible to all of us
- Same 5-step skeleton: normalize → pick a baseline → smooth along chromosomes → segment → call

## What we borrow deliberately

| Concept                                                          | Borrowed from | Why                                                                                |
|------------------------------------------------------------------|---------------|------------------------------------------------------------------------------------|
| Signature-based "confident normal" pool (UCell scoring)          | SCEVAN §3.2   | More principled than CopyKAT's variance trick at high tumor purity                 |
| Variance-cluster fallback when signatures don't fire             | CopyKAT §3.2  | Robust when the tumor microenvironment lacks our bundled cell types                |
| GMM fallback when variance clustering fails                      | CopyKAT §3.2  | Last-resort baseline so we never crash on degenerate inputs                        |
| Multichannel segmentation (one segment table for the whole cohort, per-cell CN inside) | SCEVAN §3.4 | Where SCEVAN's ~3–5× speedup over per-cell HMMs comes from              |
| 220 kb hg38 bin grid (CopyKAT-compatible)                        | CopyKAT §3.4  | IGV `.seg` consumers expect this resolution                                        |
| Tirosh 2016 cell-cycle filter                                    | every tool    | Cell-cycle expression dominates if you don't strip it                              |

## What we do differently

| Axis                  | Other tools                                                                                       | kopya                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Language             | R (all four matrix-only callers)                                                                  | Python (numpy/scipy/scanpy/sklearn)                                                                  |
| Install              | bioconda broken (inferCNV), GitHub two-step (SCEVAN), Rgraphviz hell (CaSpER)                     | one pip install, one conda env, no JAGS / no Rgraphviz                                              |
| Matrix layout        | Densified genes × cells (CopyKAT: 30–60 GB at 10k cells)                                          | Sparse CSR throughout M1+M2; dense only in M3+M4 — `mem_mb=32000` vs `128000`                       |
| Segmentation         | CopyKAT: KS-driven MCMC. SCEVAN: greedy VegaMC. inferCNV: 6-state HMM + JAGS Bayesian.            | PELT (`ruptures`, L2 cost) — non-parametric changepoint detection with log-scaled penalty            |
| Smoothing            | CopyKAT: order-1 Kalman (dlm). inferCNV: pyramidinal MA, window=101.                              | Plain moving average (`scipy.ndimage.uniform_filter1d`) — Kalman complexity buys no measurable accuracy and is hard to vectorize |
| GENCODE reference    | inferCNV: v27 (2017). CopyKAT: vendored ~v22 vintage.                                             | v49 (2025) — current symbols, ~21k more entries                                                      |
| Tumor/normal call    | CopyKAT: Ward hclust + 2-way cut. SCEVAN: cluster on segments. inferCNV: needs supplied normals.  | 2-component GMM on L1 distance from normal-pool median, with explicit `uncertain` band               |
| Supervised override  | CopyKAT supports it; SCEVAN supports it                                                           | Same — `--norm-cell-names` short-circuits the cascade                                               |
| Subclone discovery   | CopyKAT: cut Ward dendrogram. SCEVAN: re-segment per cluster.                                     | Leiden on per-cell × per-segment CN, with resolution sweep capped at `max_subclones`                |
| Output contract      | Each tool's own format                                                                            | CSVs match CopyKAT semantics (drop-in for CopyKAT outputs); also writes IGV `.seg` + QC JSON        |

## What we don't do (deliberate non-goals)

| Capability                                          | Owned by                | Why we skip                                                                          |
|----------------------------------------------------|-------------------------|--------------------------------------------------------------------------------------|
| Allele-aware (LOH, copy-neutral, biallelic amp/del) | Numbat                  | Needs BAM ingest + Eagle2 + 10 GB phasing panel; entirely different scope            |
| BAF-based calling                                   | CaSpER, Numbat          | Same — and CaSpER is unrealistic without BAM ingest anyway                           |
| Per-segment Bayesian posteriors                     | inferCNV (HMM + JAGS)   | Our PELT segments + GMM confidence are inspectable and don't need 8–24 h of JAGS sampling |
| Mouse                                               | most tools support it   | v1 hg38 only; mouse is a GENCODE-table swap when needed                              |
| Multi-sample integration                            | SCEVAN's `multiSampleComparisonClonalCN()` | Per-sample only in v1                                                          |

## Capability-at-a-glance

| Tool          | Input         | Allele-aware | Needs normal ref?            | Subclones                | Runtime (~10k cells) | RAM (~10k cells) | Install friction |
|--------------|---------------|--------------|------------------------------|--------------------------|----------------------|------------------|------------------|
| **kopya** | matrix    | No           | No (signature/variance cascade) | Yes (Leiden)             | ~5–7 min             | ~32 GB           | Low (one pip)    |
| CopyKAT      | matrix        | No           | No (auto baseline)           | Ward dendrogram          | ~1–2 h               | 30–60 GB         | Medium (GitHub install) |
| SCEVAN       | matrix        | No           | No (signature library)       | Yes, native + tree       | ~30–60 min           | 30–60 GB         | Medium (GitHub two-step) |
| CONICSmat    | matrix + BED  | No           | No (GMM)                     | From binarized matrix    | minutes–hour         | low              | Low (R + biomaRt) |
| inferCNV     | matrix + annotations + gene-order | No | **Yes (required)**           | Leiden, HMM              | 8–24 h               | 100–250 GB       | High (JAGS, dead upstream, bioconda broken) |
| Numbat       | BAM + matrix + 1000G panel | **Yes (haplotype)** | No | Yes, lineage tree       | 1–4 h pileup + 30–90 min | 20–60 GB         | Medium-High (Eagle2 + 10 GB panel) |
| CaSpER       | matrix + BAM + known normals + BAF | **Yes (BAF)** | **Yes (required)** | No (segment-level)       | 5–30 min + BAM merge     | varies           | High (BAFExtract + Rgraphviz pain) |

## Reading guide

If you want to know:
- **Why we built our own** → [README → Motivation](../README.md#motivation)
- **The exact algorithm choices** → [README → How it works](../README.md#how-it-works)
- **How it's validated** → [README → Validation](../README.md#validation), [`../tests/external/GOLD_STANDARD_TESTING.md`](../tests/external/GOLD_STANDARD_TESTING.md)
