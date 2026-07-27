# kopya

Expression-only single-cell CNV caller — a pure-Python, pip-installable alternative to the R-based tools for calling copy-number variation from scRNA-seq.

---

## Motivation

Four R-based CNV callers dominate this space, each with a different sharp edge:

| Tool | Problem |
|------|---------|
| **CopyKAT** | Densifies a genes × cells matrix at 30–60 GB RAM on 10k cells; runs 1–2 h |
| **SCEVAN** | Brittle two-step GitHub install; `parallelDist` missing on osx-arm64 conda |
| **inferCNV** | Dead upstream (README redirects elsewhere), needs JAGS, broken on bioconda, demands user-supplied normal cells |
| **CONICSmat** | Arm-level only — no focal events |

We wanted **one Python package, one pip install, sparse-first, with opinionated defaults** that we can validate against the originals. Allele-aware calling (LOH, copy-neutral) stays the domain of Numbat — see [`docs/comparison.md`](docs/comparison.md) for the full positioning.

---

## What we borrowed from the field

The algorithm is not novel — it is a careful synthesis of the best ideas from the published tools:

| Idea | Taken from |
|------|-----------|
| CP10k + log1p normalization, then project onto GENCODE coordinates | Standard scRNA-seq practice; CopyKAT's preprocessing |
| UCell scoring against non-malignant signatures to find the diploid baseline | SCEVAN |
| Lowest-variance Leiden cluster as a diploid fallback | CopyKAT |
| Pool tumor signal across cells before segmentation, then fill per-cell CN inside the shared segment table | SCEVAN's multichannel trick |
| PELT changepoint detection (L2 cost) with a BIC-style penalty | ruptures library; used by SCEVAN |
| 2-component GMM on L1 deviation from baseline → tumor / normal | CopyKAT |
| Per-segment CN matrix → Leiden → subclones | SCEVAN |

Our contribution is engineering: a single sparse-first Python package that runs the same ideas 10× faster, installs in one step, and produces CopyKAT-compatible output files so an existing CopyKAT-based pipeline needs no changes.

---

## How it works

Five-step pipeline:

| Step | Module | What it does |
|------|--------|-------------|
| 1. Normalize | `normalize.py` | CP10k + log1p; project genes onto GENCODE v49 hg38 coordinates; drop chrY / MT- / HLA- / cell-cycle genes |
| 2. Pick baseline | `baseline.py` | 4-mode cascade: **supervised** (barcodes you supply) → **signature** (UCell on bundled immune/stromal/endothelial gene sets, à la SCEVAN) → **variance cluster** (Leiden + lowest-variance cluster, à la CopyKAT) → **GMM fallback** |
| 3. Smooth + segment | `smooth.py`, `segment.py` | Center each cell against the per-gene normal-pool median; per-chromosome moving average; PELT changepoint detection on the **pooled tumor signal**; per-cell CN computed inside the shared segment table |
| 4. Classify + subclones | `classify.py` | 2-component GMM on per-cell L1 distance from baseline → tumor / normal / uncertain; Leiden on tumor cells' per-segment CN matrix → subclones (max 5) |
| 5. Write outputs | `outputs.py` | Drop-in `prediction.csv`, `chr_cnv_matrix.csv` (CopyKAT-compatible), IGV `.seg`, `qc.json` |

End-to-end on 10k cells: **~5–7 minutes, ~32 GB peak RSS** — vs CopyKAT's 1–2 h and 30–60 GB.

---

## Install

```bash
# Create the conda env, run from the repo root (Python + numpy/scipy/scanpy/sklearn/ruptures/leidenalg/pyucell).
# The yaml's `-e .` editable-installs this package, and it resolves against the
# directory you invoke mamba from, so the working directory must be the repo root.
mamba env create -f envs/kopya.yaml
mamba activate kopya

# The env already editable-installs this package. Re-run to add the optional
# plotting stack (matplotlib, for `plot-heatmap` and the Python-API heatmap)
# and the test tools (pytest):
pip install -e ".[plot,dev]"

# Confirm the install (full unit-test suite, ~2 min).
pytest
```

The package ships a bundled GENCODE v49 hg38 gene-order table (~77k genes, 2.6 MB) and a Tirosh 2016 cell-cycle gene list — no external resources required to run.

---

## Quickstart

A runnable first pass on the committed 500-cell demo fixture — no data to download:

```bash
# Run the pipeline, then render the inferCNV-style heatmap.
kopya run --anndata tests/fixtures/tiny_simulated.h5ad --out-dir /tmp/kopya-demo --sample demo
kopya plot-heatmap --run-dir /tmp/kopya-demo --out /tmp/kopya-demo/heatmap.png --scale linear
```

This writes `prediction.csv`, `chr_cnv_matrix.csv`, `segments.parquet`, `qc.json`, an IGV `.seg`, and the heatmap into `/tmp/kopya-demo/`. The fixture has three planted events (chr7 gain, chr10 loss, focal MYC amp), so it's a good place to build intuition — see [`examples/tiny_simulated/`](examples/tiny_simulated) for a pre-computed copy of these outputs and [`notebooks/working_with_results.ipynb`](notebooks/working_with_results.ipynb) for a walkthrough of every file.

---

## Inputs

Four accepted entry forms — all produce identical downstream results, listed most-common first.

### CellRanger directory (Form C) — standard external format

The `filtered_feature_bc_matrix/` directory that CellRanger, STARsolo, alevin-fry, and most droplet-sequencing pipelines write:

```
outs/
└── filtered_feature_bc_matrix/
    ├── barcodes.tsv.gz
    ├── features.tsv.gz      # genes.tsv.gz for CellRanger v2
    └── matrix.mtx.gz
```

```bash
kopya run --cellranger-dir outs/filtered_feature_bc_matrix/ --out-dir results/ --sample my-sample
```

### CellRanger H5 (Form D)

The single-file HDF5 alternative from the same CellRanger run:

```bash
kopya run --cellranger-h5 outs/filtered_feature_bc_matrix.h5 --out-dir results/ --sample my-sample
```

> **Loupe files (`.cloupe`) are not supported** — they are a proprietary format with no public parser. Use the `.h5` or `filtered_feature_bc_matrix/` output from the same CellRanger run.

### AnnData h5ad (Form A)

`adata.X` (or `layers["counts"]`) holds raw integer UMI counts. `var_names` are HGNC gene symbols. `obs_names` are cell barcodes.

```bash
kopya run --anndata processed_ad.h5ad --out-dir results/ --sample my-sample
```

### Matrix trio — Matrix Market + metadata CSVs (Form B)

A Matrix Market counts file alongside cell- and gene-metadata CSVs, in this layout:

```
results/
├── normalization/
│   └── counts.mtx
└── metadata/
    ├── obs.csv
    └── var.csv
```

### Finding inputs automatically

`find-inputs` searches a directory tree for any of the four forms and prints the ready-to-run command. Detects the CellRanger `outs/` layout, the matrix-trio layout, and any `.h5ad`. Warns clearly if a `.cloupe` file is found.

```bash
kopya find-inputs --patient-dir /path/to/patient --sample sample-01
```

### Optional inputs

- `--norm-cell-names path.txt` — one barcode per line; bypasses the baseline cascade and uses these cells as the diploid pool. Useful when you have manual immune-cell annotations. **For some tumors this is not optional — see [When to use supervised mode](#when-to-use-supervised-mode).**
- `--signatures path.json` — override the bundled non-malignant signature library (same schema as the built-in `normal_signatures.json`: an optional `_meta.non_malignant_labels` allow-list plus one `"<label>": [genes...]` entry per signature). Retunes the **signature** baseline mode for tissues the defaults don't fit.
- `--non-malignant-labels A,B,C` — override just the allow-list of cell types treated as "normal", without authoring a whole JSON. Drop an inversion-prone label (e.g. `Fibroblast` on a mesenchymal tumor) or add one (e.g. `Plasma_cell` where plasma cells are reactive infiltrate). Validated against the active signature set, so a typo fails fast.
- `--gene-order path.tsv` — overrides the bundled GENCODE table (4 columns: `gene_symbol chr start end`).

> **The bundled signatures and allow-list are defaults, not universal truth.** They're curated for immune-infiltrated solid tumors. On mesenchymal (sarcoma/osteosarcoma) or hematologic samples the defaults can mislabel the malignant population as "normal" — `Plasma_cell` is intentionally **excluded** from the allow-list, for instance, because plasma cells are the malignancy in myeloma. If a run looks off (see [When to use supervised mode](#when-to-use-supervised-mode)), retune with `--non-malignant-labels` / `--signatures`, or pin the baseline outright with `--norm-cell-names`. The effective allow-list is recorded in `qc.json` (`baseline_diagnostics.non_malignant_labels`).

---

## Running

### Quick run, all defaults

```bash
kopya run \
    --counts  /path/to/counts.mtx \
    --obs     /path/to/obs.csv \
    --var     /path/to/var.csv \
    --out-dir results/ \
    --sample  sample-01
```

### All the knobs

```bash
kopya run \
    --anndata /path/to/processed_ad.h5ad \
    --out-dir results/ \
    --sample  sample-01 \
    --threads 8 \
    \
    # cell + gene filtering
    --min-genes 200       \
    --low-dr    0.05      \
    --up-dr     1.0       \
    \
    # baseline picking
    --normal-score-threshold 0.3   \
    --min-normal-cells       50    \
    --norm-cell-names        immune_barcodes.txt   `# optional supervised override` \
    --non-malignant-labels   T_cell,B_cell,Endothelial   `# optional allow-list override` \
    --signatures             custom_signatures.json       `# optional signature library` \
    \
    # smoothing + segmentation
    --smooth-window  100   \
    --penalty-coef   1.0   \
    --min-seg-genes  25    \
    \
    # classification
    --call-confidence      0.5  \
    --max-subclones        5    \
    --coherence-gate       0.45 `# downgrade scattered-signal tumor calls → uncertain; 0 disables, lower for focal-amp tumors` \
    --low-complexity-frac  0.5  `# flag ambient cells below this fraction of the median detected-gene count; 0 disables` \
    --subclones            `# or --no-subclones`
```

Every knob has a sensible default; the bare-minimum invocation above is all most runs need. See `kopya run --help` for the full surface.

---

## Python API (scanpy-style)

Prefer to work in memory alongside scanpy / [infercnvpy](https://infercnvpy.readthedocs.io/)? Run on an AnnData and the results are written back onto it — no files, no CLI:

```python
import scanpy as sc
import kopya as kp

adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")   # raw counts in .X
# ... your usual cell-type annotation pass populates adata.obs["cell_type"] ...

kp.tl.cnv(adata, reference_key="cell_type",
          reference_cat=["T cell", "B cell", "Macrophage", "Endothelial"])

adata.obs["cnv_class"]          # 'tumor' / 'normal' / 'uncertain'  (NaN = QC-filtered)
adata.obs["cnv_score"]          # per-cell aneuploidy score
adata.obs["cnv_subclone"]       # Leiden subclone among tumor cells ('' for non-tumor)
adata.obs["cnv_low_complexity"] # bool — ambient / low-gene-count cell flag
adata.obsm["cnv_chr"]           # cells × chromosomes, 1.0-centered (>1 gain, <1 loss)
adata.uns["cnv"]                # chroms, baseline method, params, segment table

kp.pl.chromosome_heatmap(adata, groupby="cell_type")   # needs the [plot] extra (matplotlib)
```

Pass `reference_key`/`reference_cat` (your normal cell types) to run supervised — recommended, and required for mesenchymal tumors (see below). Omit them for the unsupervised cascade. Raw counts are required (the tool does its own CP10k+log1p); if they live in a layer, pass `layer="counts"`. The heatmap needs the `[plot]` extra (`pip install -e ".[plot]"`).

**Coming from Seurat?** See the round-trip walkthrough — annotate once in Seurat, run kopya, bring the calls back — in the executed notebook [`docs/tutorials/seurat.ipynb`](docs/tutorials/seurat.ipynb) (previews inline on GitHub, with a rendered CNV heatmap).

For ~1M-cell cohorts, the CLI's `run` command has a memory-lean streaming path; `tl.cnv` is the in-memory convenience path for typical objects.

---

## When to use supervised mode

The unsupervised baseline cascade (signature → variance → GMM) works out of the box for most samples, but it has a failure mode worth knowing about: **on some tumors it inverts.** It picks the *tumor* itself as the diploid baseline, so every call comes out backwards — tumor cells look normal, normal cells look aneuploid, and the per-chromosome profile anti-correlates with matched bulk.

This happens when the tumor's expression profile matches one of the bundled "normal" signatures:

- **Mesenchymal / non-immune tumors** (osteosarcoma, other sarcomas). The malignant cells express fibroblast / osteoblast markers, so the signature step labels the tumor population "normal."
- **Full-length protocols (SMART-seq2).** UCell rank-scoring misfires on read-length counts — see the Patel 2014 XFAIL under [Validation](#validation).
- More generally, any sample where the genuinely-diploid cells are too small a minority for the cascade to lock onto.

**How to spot an inverted baseline:**

- `qc.json` → `baseline_method` is `signature`/`variance` with an implausibly large `n_normal_seed` (most of the sample called "normal") and a compressed `tumor_signal_max_abs`.
- Known-normal/immune cells (T cells, plasma cells) score *highest*, not lowest — the tell-tale sign the reference frame is flipped.
- If you have matched bulk, the single-cell profile anti-correlates with it.

**The fix — seed the baseline yourself:**

```bash
kopya run --anndata processed_ad.h5ad --out-dir results/ --sample my-sample \
    --norm-cell-names normal_barcodes.txt
```

`--norm-cell-names` (one barcode per line) bypasses the cascade and uses exactly those cells as the diploid pool (`baseline_method: supervised`, the top tier of the cascade). Source the barcodes from any upstream cell-typing — a T-cell / immune cluster from your Leiden annotation is the usual choice, since lymphocytes are reliably diploid.

> **Worked example — a mesenchymal tumor (e.g. osteosarcoma).** Unsupervised, the osteoblast/fibroblast signatures capture the mesenchymal tumor itself as the baseline, so the single-cell profile anti-correlates with matched bulk (negative concordance) and plasma cells come out as the highest-scoring "tumor" population. Re-seeding the baseline with an annotated T-cell barcode list flips the sign back to positive concordance — strong-event recall goes from 0 to complete — and clears the plasma-cell false positives. The lesson: for mesenchymal and other inversion-prone tumors, **supervised mode is not a fallback — it's the right call.**

**Lighter-weight alternative.** If you don't have curated barcodes but you *do* know the cascade is picking the wrong cell type as normal, you can keep the unsupervised cascade and just retune which signatures count as "normal" — e.g. `--non-malignant-labels T_cell,B_cell,Endothelial,NK_cell` to exclude the mesenchymal labels, or supply a whole custom `--signatures` library. See [Optional inputs](#optional-inputs).

---

## Outputs

All artifacts land under `--out-dir`:

| File | Shape / Format | Purpose |
|------|----------------|---------|
| `prediction.csv` | cells × `{class, confidence, tumor_score, subclone, n_segments_altered, low_complexity}` | Per-cell call. Drop-in replacement for `copykat_prediction.csv` — the CopyKAT-compatible columns keep their positions; `low_complexity` is appended last |
| `chr_cnv_matrix.csv` | cells × chromosomes | Per-chromosome CN summary, centered on 1.0 (`>1` gain, `<1` loss). Drop-in for `copykat_chr_cnv_matrix.csv` |
| `{sample}_clones.seg` | IGV `.seg` | Per-clone consensus segments, loadable directly into IGV. Clone IDs prefixed with the sample name so multi-sample sessions stay disambiguated |
| `segments.parquet` | per-segment summary (one row per segment) | The **segment → genome map**. Original columns `chr`, `start_idx`/`end_idx` (gene-axis indices, start inclusive/end exclusive), `n_genes`, `tumor_mean` (pooled-tumor log-deviation: `>0` gain, `<0` loss), plus three appended: `segment_id` (0-based key), `start_bp`/`end_bp` (genomic span of the segment — the min gene start and max gene end across its genes). `segment_id`/`start_bp`/`end_bp` make this self-contained — you can locate any segment without the source `adata.var`. Segment row `i` == column `i` of `cn_per_segment.npz` |
| `cn_per_segment.npz` | dense (cells × segments) | Per-cell × per-segment CN matrix (`cn`) plus `cell_barcodes`; column `i` corresponds to `segment_id == i` in `segments.parquet`. Stored **raw** (log-space, carries a per-gene pedestal + per-cell offset) — re-center before plotting (see [How the signal is recovered](#visualize-the-calls--infercnv-style-heatmap)). Optimized for numpy consumers |
| `cn_per_segment_denoised.npz` | dense (cells × segments) | **Optional** — written only with `--denoise-outputs`. The **recentered + inferCNV-denoised** signal (the exact transform `plot-heatmap` draws), for a clean, ready-to-read gain/loss matrix: `0` ≈ diploid, `>0` gain, `<0` loss. Strong hallmark events survive; the normal noise floor collapses to ~0. Additive — the raw `cn_per_segment.npz` is always written unchanged. Also stores `sd_amplifier` |
| `qc.json` | dict | Run-level diagnostics: cell/gene counts, baseline method, subclone counts, per-step timings |

> **New to the output files?** Start with the worked example and notebook:
> [`examples/tiny_simulated/`](examples/tiny_simulated) holds a full set of
> results from a tiny 500-cell run, and
> [`notebooks/working_with_results.ipynb`](notebooks/working_with_results.ipynb)
> walks through every file — what each column means, the numeric scale of each
> value (log-space vs 1.0-centered, when negatives are expected), and how to
> re-center `cn_per_segment.npz` into gain/loss.

#### Value scales at a glance

Different files center their values differently — this trips people up, so keep it straight:

| File | Scale | Diploid value | Negatives? |
|------|-------|---------------|-----------|
| `chr_cnv_matrix.csv` | 1.0-centered multiplicative ratio | `1.0` | no (always > 0) |
| `segments.parquet` `tumor_mean` | natural-log deviation | `0` | **yes** |
| `cn_per_segment.npz` | natural-log signal (raw; re-center first) | `~0` after re-centering | **yes** |
| `{sample}_clones.seg` `seg.mean` | natural-log ratio vs normal reference | `0` | **yes** |
| `cn_per_segment_denoised.npz` (opt-in) | natural-log deviation, denoised | `0` | **yes** |
| `prediction.csv` `tumor_score` | Σ \|deviation\| (within-run rank, not portable) | `~0` | no |

#### Denoised outputs — scope and rationale

`--denoise-outputs` writes **one** extra file, `cn_per_segment_denoised.npz`, and nothing else. Denoise is a *per-cell, per-segment* soft-threshold against the reference noise band, so it only helps at the full resolution where that noise actually lives. It is applied at plot/output time only — the raw `cn_per_segment.npz` (and every other file) is always written unchanged. Use the denoised matrix for a clean "is the hallmark event there?" read (strong events survive, the normal floor collapses to ~0); keep the raw matrix as the source of truth for magnitude, since denoise is lossy and can attenuate weak/focal events.

The other outputs deliberately have **no** `_denoised` twin — they don't need one:

- **`chr_cnv_matrix.csv`** is already aggregated to chromosome level, which averages per-segment noise toward 0 (it's why raw chr7 already reads ~1.8 in tumor vs ~1.0 in normal). Chromosome aggregation is itself a form of denoising, and this file uses a scalar pedestal to preserve bulk concordance.
- **`{sample}_clones.seg`** is a per-subclone *median* across cells (with the per-segment baseline already subtracted); a cross-cell median is a strong denoiser, so an explicit denoise would be ~a no-op.
- **`segments.parquet` `tumor_mean`** is a pooled summary vector, not a per-cell matrix — there is nothing per-cell to threshold.
- **`prediction.csv`** holds calls/scores, not a signal matrix. (`tumor_score` is computed on the *raw* matrix, by design.)

Want a denoised **chromosome-level** view? Aggregate the denoised per-segment matrix — denoise first, then length-weight-average per chromosome. Aggregating the raw matrix and denoising afterward is not equivalent; the order matters.

### Example `qc.json`

```json
{
  "sample": "sample-01",
  "n_cells_loaded": 77418,
  "n_genes_loaded": 18487,
  "n_cells_post_m1": 77418,
  "n_genes_post_m1": 11379,
  "baseline_method": "signature",
  "n_normal_seed": 40063,
  "n_segments": 182,
  "n_tumor": 18096,
  "n_normal_called": 56981,
  "n_uncertain": 2441,
  "n_low_complexity": 9663,
  "n_subclones_observed": 5,
  "subclone_counts": {"subclone_1": 10097, "subclone_2": 3045, "subclone_3": 2606, "subclone_4": 2049, "subclone_5": 299},
  "timings_secs": {
    "load": 0.6,
    "m1_filter_normalize_project": 5.0,
    "m2_pick_baseline": 6.2,
    "m3_smooth_segment_cn": 19.1,
    "m4_classify_write_outputs": 10.4
  }
}
```

`baseline_method` is the most important diagnostic — tells downstream how confident the baseline call is (`supervised` > `signature` > `variance` > `gmm_fallback`).

### Interpreting `prediction.csv`

The intended per-cell call is the **`class`** column (`tumor` / `normal` / `uncertain`) — it is the GMM-derived label, not a threshold you apply yourself to a raw score. The other columns are supporting signals for triage:

| Column | Meaning | How to read it |
|--------|---------|----------------|
| `class` | Tumor / normal / uncertain call from the 2-component GMM on `tumor_score` | The call. `uncertain` = GMM posterior below `--call-confidence` (default 0.5); filter these out for clean downstream analysis |
| `confidence` | Max GMM posterior in `[0, 1]` | How sure the call is. Cells near 0.5 are genuinely ambiguous; sort tumor calls by this to find borderline ones |
| `tumor_score` | Per-cell L1 distance: the **sum** of absolute per-segment CN deviations from the normal-pool baseline (log-space) | Higher = more aneuploidy burden. See caveat below |
| `n_segments_altered` | Count of segments deviating >0.2 log-units from baseline | Often more interpretable than `tumor_score` — "how much of the genome is rearranged" rather than a magnitude. A real tumor cell lights up many segments; a high score from one or two noisy focal events will have a low count here |
| `subclone` | Leiden cluster among tumor cells (`subclone_1…`, `""` for non-tumor) | Largest clone is `subclone_1`; capped at `--max-subclones` (default 5) |
| `low_complexity` | bool — ambient / empty-droplet-like cell (few detected genes) | Flagged, never silently dropped. A tumor call on such a cell is untrustworthy (noise-dominated CN signal), so the classifier downgrades it to `uncertain`. Filter these out for a clean tumor set |

**On `uncertain` (flag, don't force).** Two gates downgrade untrustworthy `tumor` calls to `uncertain` rather than asserting them: the **low-complexity** gate (above) and a **coherence** gate — a high score built from scattered, single-segment spikes rather than contiguous chromosome-arm-scale runs is more likely a transcriptionally-"loud" normal cell than copy number. Both gates are *subtractive* (they never promote a cell to tumor), so they cannot inflate the tumor set. Genuinely ambiguous cells — including same-lineage normals whose expression mimics CNV (e.g. normal astrocytes vs AC-like glioma, which expression-only CNV cannot separate) — land in `uncertain` by design. `n_uncertain` and `n_low_complexity` are recorded in `qc.json`.

**Caveat on `tumor_score`:** it is a *sum* across segments, so its absolute magnitude scales with the number of segments (`n_segments` in `qc.json`) and the strength of the CN signal. There is **no fixed unit or universal cutoff** — a value is only meaningful *relative to other cells in the same run*. Normal cells sit near 0 (or a low baseline) and tumor cells form a separate higher cluster, but the actual numbers (e.g. a tumor cluster at 5–40) are **not comparable across samples**. For any cross-sample comparison, rely on `class` / `confidence` or `n_segments_altered` instead of the raw score.

A good triage filter is to cross-tab `tumor_score` against `n_segments_altered`: confident tumor cells score high on *both*, while a high score with few altered segments usually signals a noisy or transcriptionally extreme cell rather than a true malignant one.

---

## Visualize the calls — inferCNV-style heatmap

`plot-heatmap` renders a genome-ordered CNV heatmap from a completed run's
out-dir, in the familiar inferCNV layout — up to three stacked panels:
**References** (confident-diploid cells) on top, **Observations** (tumor cells,
grouped by subclone) below, and an **Uncertain** panel of flagged cells when any
exist. Genome runs left-to-right with chromosome widths proportional to gene
content; blue→white→red diverging colour (loss→diploid→gain).

The `normal` class on a real run is a mixed bag (genuinely-diploid cells, ambient
junk, and tumor cells under-called as normal), so plotting all of it as
"references" reads as noisy structure. Two curation steps fix that, matching how
inferCNV uses a hand-picked reference:

- **Low-complexity (ambient) cells are dropped** from every panel — they are not
  trustworthy CNV signal. Pass `--show-low-complexity` to keep them.
- **The reference panel — and the baseline the whole figure is recentered
  against — is the *confident-diploid* subset**: non-junk normals whose
  `tumor_score` is in the lowest `--ref-score-quantile` (default 0.5) of that
  pool. This keeps under-called tumor cells out of the reference frame, which
  also sharpens the observation panel. `--ref-score-quantile 1.0` keeps every
  non-junk normal.

```bash
# needs the plotting extra: pip install -e ".[plot]"
kopya plot-heatmap --run-dir results/ --out results/sample-01_cnv.png
```

Two colour scales:

- `--scale log` (default) — per-segment deviation in log-space, centered on 0.
- `--scale linear` — copy-number **ratio** centered on 1.0 (diploid), matching
  inferCNV's "Modified Expression" 0.8–1.2 convention and the 1.0-centered
  `chr_cnv_matrix.csv` output.

On a glioblastoma sample the tumor panel shows the textbook GBM signature —
solid **chr7 gain** and **chr10 loss**, plus chr12/18/19/20 gains — over a clean
confident-diploid reference panel.
(A residual note: any structure left in the reference panel is real biology/QC —
under-called tumor or same-lineage normals that mimic CNV — not a plotting
artifact. Curating the reference with `--norm-cell-names` removes it entirely.)

> **How the signal is recovered.** The stored `cn_per_segment.npz` is the
> per-segment mean of the median-centered smoothed signal — it carries a per-gene
> positive pedestal and a per-cell magnitude offset, so plotted raw it reads as an
> all-gain wash. `plot-heatmap` recovers the signed gain/loss signal the way
> inferCNV displays it: subtract each segment's median over the confident-diploid
> reference (references → ~0), then subtract each cell's own genome-wide median
> (removes the per-cell offset), then **denoise** — soft-threshold each cell's
> per-segment deviation toward 0 by `--sd-amplifier` reference-cell spreads
> (default 1.0), collapsing the normal noise floor to white while leaving real CNV
> events intact. The band is estimated from the *cleanest* reference cells so
> residual noisy cells don't widen it and erase weak tumor signal. Set
> `--sd-amplifier 0` to disable denoise and see the raw reference-relative signal.

Common knobs: `--ref-score-quantile` (reference strictness, default 0.5),
`--show-low-complexity` (keep ambient cells, dropped by default),
`--sd-amplifier` (denoise strength, default 1.0; 0 disables),
`--vlim` (colour clip, default 0.15 log-units), `--n-ref` /
`--max-obs` (display subsampling caps for the reference / observation panels),
`--sample` / `--subtitle` (header text), `--dpi`. Cells are ordered within each
panel by hierarchical clustering. See
`kopya plot-heatmap --help` for the full surface.

---

## Validation

We test at two levels. Each level is self-contained and can be run independently.

### Level 1 — Synthetic unit tests (always run in CI)

Every algorithm module has a dedicated test file with synthetic fixtures that have known answers. The end-to-end test uses a 500-cell AnnData with three planted CNV events:

- **chr7 broad gain** (200 genes × 2.5×) — EGFR amplification, the textbook GBM hallmark
- **chr10 broad loss** (200 genes × 0.4×) — PTEN deletion, the other GBM hallmark
- **focal MYC amp** (~30 genes on chr8, × 5.0) — a focal event to probe sub-chromosomal resolution

The test asserts ground-truth recovery: tumor/normal recall ≥ 80% both directions, chr7/chr10 signals clearly separated, focal MYC visible.

```bash
pytest                          # full unit suite, ~2 min
pytest tests/test_segment.py -v # just segmentation
pytest tests/test_cli.py -v     # includes CellRanger format tests
```

These run on every push and pull request via CI. **If this passes, the implementation is correct on synthetic data.**

The CLI tests cover all four input forms end-to-end on the same 500-cell fixture: `--anndata`, `--cellranger-dir`, `--cellranger-h5`, and `--counts/--obs/--var`. The CellRanger fixtures are built programmatically — no 10x Genomics tools required.

Regenerate the fixture only when intentionally changing the planted events:

```bash
python -m tests.fixtures.build_tiny_simulated
```

### Level 2 — External gold-standard datasets

We test against publicly available datasets whose CNV ground truth is established independently — not derived from this tool. The tests skip automatically when the data files are absent.

**Current results:**

| Dataset | Source | Ground truth | Result |
|---|---|---|---|
| Patel 2014 GBM | GSE57872 (SMART-seq2) | Published chr7 gain / chr10 loss | 3 PASS, 4 XFAIL¹ |
| DCIS1 breast cancer | GSE148673 (CopyKAT paper) | Paired bulk WGS (Gao et al. 2021) | 5 PASS |
| SCEVAN synthetic matrices | Zenodo 6628423 | Planted tumor/normal labels | 3 PASS |
| 10X ovarian scFFPE (HGSOC) | 10x Genomics 17k dataset | FLEX cluster annotations | 4 PASS |
| Maynard 2020 lung | `maynard2020_3k` (infercnvpy) | Cell-type labels + kopya-vs-inferCNV | 6 PASS |
| UCSF osteosarcoma T1 | osteosarc.com (IPISRC044_T1) | Matched bulk WES (CNVkit) | 5 PASS |
| HCC1395 breast cell line | SEQC2 / GIAB karyotype | Known amplifications / losses | SKIP² |

¹ The Patel SMART-seq2 dataset triggers a known baseline inversion — the unsupervised UCell cascade misfires on full-length read counts, so the four hallmark-direction tests are `xfail(strict)`. Fix path: supervised mode with cell-type annotations. Full diagnosis in [`tests/external/GOLD_STANDARD_TESTING.md §6.1`](tests/external/GOLD_STANDARD_TESTING.md).

² HCC1395 scRNA-seq data not staged locally; tests are ready to run once data is placed at the expected path.

**To reproduce:**

```bash
# One-time: acquire and convert each dataset (exact commands in the doc below)
# See tests/external/GOLD_STANDARD_TESTING.md §2 for download URLs and conversion scripts.

# Then run:
pytest tests/external/ -v
```

Full documentation — acquisition commands, acceptance criteria, per-metric results, known limitations, improvement history:
[`tests/external/GOLD_STANDARD_TESTING.md`](tests/external/GOLD_STANDARD_TESTING.md)

---

## Documentation

- [`docs/tutorials/seurat.ipynb`](docs/tutorials/seurat.ipynb) — executed end-to-end tutorial: annotate in Seurat, run kopya, bring the CNV calls back (with a rendered heatmap; previews on GitHub)
- [`notebooks/working_with_results.ipynb`](notebooks/working_with_results.ipynb) — a hands-on walkthrough of every output file: what each column means, the numeric scale of each value, and how to re-center the CN matrix into gain/loss
- [`examples/tiny_simulated/`](examples/tiny_simulated) — a complete, committed result set (500-cell demo) to explore or diff against
- [`docs/comparison.md`](docs/comparison.md) — detailed positioning vs CopyKAT, SCEVAN, inferCNV, CONICSmat, Numbat, CaSpER
- [`tests/external/GOLD_STANDARD_TESTING.md`](tests/external/GOLD_STANDARD_TESTING.md) — external dataset acquisition, acceptance criteria, results, known limitations
- `kopya run --help` — full CLI surface with every knob
