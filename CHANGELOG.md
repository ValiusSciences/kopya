# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- **`--centered-chr-matrix` writes `chr_cnv_matrix_centered.csv`**, an optional,
  purely additive companion to `chr_cnv_matrix.csv` with each cell's row divided
  by its own median chromosome (`outputs.center_chr_cnv_matrix()`). Off by
  default; the raw file is always written unchanged, and nothing in the pipeline
  reads the centered one, so calls are unaffected. `1.0` in the centered file
  means *that cell's median chromosome*, not *diploid* — it is for visualization
  and relative per-cell interpretation, and the raw file remains the source of
  truth for absolute copy-number level.

### Documentation

- **Documented the per-cell offset in `chr_cnv_matrix.csv`.** The file is
  calibrated only up to a per-cell scale factor: the per-gene centering leaves a
  pedestal that scales with how many genes a cell detected, and the scalar
  baseline subtracted at write time removes only the pool-wide part. The residual
  cancels within a cell; across cells it makes a pseudobulk a factor-weighted
  average rather than a plain mean, which the scale-invariant per-chromosome
  Pearson cannot see — hence the matched-bulk concordance never flagged it — and
  it does not cancel at all when cells are compared to each other at one
  chromosome, where it correlates strongly with sequencing depth and can exceed
  the per-chromosome biology. The README, the `compute_chr_cnv_matrix` docstring
  and the results notebook now say so and point at the one-line fix; the notebook
  previously recommended the file for per-cell work without qualification.
- Corrected `heatmap.py`'s module docstring (and the matching `--scale linear`
  note in the README), which said the recentered heatmap shares the "1.0-centered"
  frame of `chr_cnv_matrix.csv`. The heatmap subtracts each cell's own median;
  the CSV does not.

### Changed

- **`Fibroblast` is no longer an allowed normal reference in unsupervised mode**
  (`normal_signatures.json` v4). Mesenchymal / ECM-like tumor cells express the
  fibroblast program, won the `Fibroblast` signature, and so entered the
  confident-normal pool, letting the tumor partly seed its own diploid baseline.
  The `Fibroblast` gene set is still defined, so those cells keep resolving to it
  and are kept out of the seed rather than falling through to an allow-listed
  stromal label. Supervised runs (`--norm-cell-names`, or `reference_key` /
  `reference_cat` in the API) are unaffected: they never consult the allow-list.
  Samples whose normal stroma is genuinely fibroblast-rich can add the label back
  with `--non-malignant-labels`. On the 10X ovarian scFFPE benchmark this leaves
  AUC unchanged (0.9745), improves tumor recall (0.9756 to 0.9813), and raises
  the immune/stromal false-tumor rate (0.0262 to 0.0432, cap 0.05).

## 2.0.0 — 2026-08-19

> **Why major.** `tumor_score` changed meaning without changing its name or type, so
> code that thresholded, sorted, or filtered on it keeps running and returns wrong
> answers with no error to notice. See *Breaking* and *Migration* below before
> upgrading.

### Breaking

- **`prediction.csv` `tumor_score` is now a signed consensus-template projection**,
  not the unsigned L1 burden `Σ |deviation|`. Positive = deviating *with* the
  sample's clonal CN profile, `≈ 0` = no copy number, **negative = deviating
  against it** — a large real deviation in the opposite direction, typically a
  transcriptionally extreme cell type (erythrocytes, platelets), *not* a
  confidently diploid cell. The minimum of the column is now the most
  anti-aligned cell rather than the most diploid one.
- The score no longer scales with segment count (it is genome-fraction weighted and
  normalized), so previously calibrated numeric cutoffs do not carry over. It does
  scale with the amplitude of the sample's own consensus profile.
- `kopya.tl.cnv()` writes the signed score to `adata.obs[f"{key_added}_score"]`,
  with the same change of meaning.

### Migration

| if you were doing | do this instead |
|---|---|
| `pred.nsmallest(n, "tumor_score")` to find diploid cells | `pred.nsmallest(n, "cn_burden")` |
| `pred["tumor_score"] > cutoff` as a tumor filter | `pred["class"] == "tumor"`, or re-derive the cutoff on this run |
| ranking cells by aneuploidy load | `cn_burden`, or `n_segments_altered` for a portable measure |
| reading `prediction.csv` from an older run | `abs(tumor_score)` is the closest stand-in for `cn_burden` |

`kopya.classify.compute_tumor_scores()` still returns the old unsigned burden and is
still exported; it is simply no longer what the classifier scores on. **Pass
`seg_weights=segments["n_genes"]`** if you want the `cn_burden` column's value — with
the weights omitted it skips the per-cell centering *and* leaves the sum unweighted,
which is a different quantity in a different frame (measured on a small fixture:
1.2717 vs a `cn_burden` of 0.0414). The weights cannot be defaulted, because a
per-segment gene count is not recoverable from the CN matrix alone.

### Added

- **`cn_burden` column in `prediction.csv`** — the non-negative gene-count-weighted
  mean `|deviation|` across the genome, in the same per-cell-centered frame as the
  signed score. Appended after `low_complexity`, so every historical column position
  is unchanged for positional readers of the CopyKAT-compatible prefix.
- **`adata.obs[f"{key_added}_cn_burden"]`** from `kopya.tl.cnv()`.
- **`n_outlier_fenced` in `qc.json`** — how many cells the outlier fence held out of
  the GMM fit as probable transcriptome artifacts. Normally `0`; a large value means
  the sample carries a sizeable population of high-scoring cells whose deviation is
  scattered rather than clonal.
- **`n_anti_aligned` in `qc.json`** — how many cells the anti-alignment gate moved
  from `normal` to `uncertain` (see below). Normally `0` or near it; a large value
  means this sample carries a population the single-template score cannot represent.
- **`anti_alignment_tumor_n` in `qc.json`** — how many tumor calls that gate's
  thresholds were estimated from. `0` means the gate **did not run** because too few
  cells were called tumor, so `n_anti_aligned == 0` only means "nothing flagged" when
  this is non-zero. Recorded because a silently skipped gate is otherwise
  indistinguishable from a gate that found nothing.

### Changed — classifier

Measured on a 16-patient benchmark cohort (10 with annotation truth, 9 with matched
bulk WES), against the previous release:

| metric | 1.0.1 | 2.0.0 |
|---|---|---|
| mean recall | 0.326 | 0.856 |
| mean accuracy | 0.695 | 0.930 |
| mean AUC | 0.656 | 0.927 |
| mean bulk-WES concordance r | 0.463 | 0.618 |

On a held-out protocol where half of each patient's known-normal cells are withheld
from the tool (143k genuine normals, so false positives are measurable for the first
time): mean precision 0.741 → 0.858, and **at the 1.0.1 false-positive rate the new
score reaches 2.37× its recall** (0.728 vs 0.308), on 9 of 10 patients.

- **Per-cell recentering.** The classifier now subtracts each cell's own
  gene-count-weighted median across segments, matching what `heatmap._recenter` has
  always done for the figures. That offset was 50-90% of the old score's magnitude
  and correlated with detected-gene count at up to r=+0.88 *within known-diploid
  cells* — i.e. much of the old score was library depth.
- **Gene-count-weighted burden**, normalized to the genome, so the score no longer
  partly measures segmentation granularity (PELT emits segments spanning >10× in
  size).
- **Consensus-template projection.** Cells are scored by signed alignment with the
  sample's own consensus CN profile, estimated label-free from its most-aneuploid
  cells. Each seed is rescaled to unit norm so it votes on direction rather than
  amplitude, and low-complexity cells cannot seed the template.
- **Winsorization** is on the global 1st/99th percentiles, both tails, instead of a
  ceiling pinned to the reference pool's spread — which tightened as the score
  improved and was collapsing 46-91% of true tumor cells onto one value.
- **The negative tail is excluded from the GMM fit, not just clipped.** A fixed 1%
  quantile bounds one-in-a-hundred cells, so it answers the "one extreme cell" case
  and nothing larger. With an anti-aligned population above ~1% of the sample the
  floor lands *inside* it and the mixture spends a component describing it: measured
  on a 7.4%-anti-aligned synthetic, component means −1.144 and +0.102 with **all**
  plain normals in the same component as **all** tumor cells — the tumor/normal split
  gone entirely, while the emitted labels stayed correct because every normal was in
  `normal_mask` and got overridden. The outlier fence is now symmetric and its low
  side excludes those cells from the fit regardless of whether they were labelled,
  since holding a cell out of the fit is not overriding its label. The low side needs
  no coherence condition (under a signed score no clone projects negatively), which is
  also why it works for callers who cannot supply one.
- **The anti-alignment gate declines to run below 30 tumor calls.** Both of its
  thresholds are medians over the cells called tumor, so at low purity the median is
  one or two cells: across 6 runs differing only in RNG seed, the score threshold
  spanned 8.2× at 5 tumor calls and the gate downgraded 5 cells on two seeds and 0 on
  the other four at identical purity. `anti_alignment_tumor_n` reports what the
  estimate rested on.
- **`discover_subclones`** clusters the reference-relative deviation instead of the
  raw depth-confounded matrix, so `subclone_N` labels are not depth strata.
- **Outlier fence** now asks whether a high-scoring cell's deviation is *spatially
  contiguous* rather than where it sits in the score distribution. Scoring far above
  the reference pool is equally true of a rare real clone and of a cluster of
  transcriptome-extreme cells, so no threshold on the score alone can tell them apart;
  the coherent fraction can. The fence also no longer forces its cells to `normal` —
  it decides what the mixture is *fitted* on, and leaves labelling to the GMM and the
  gates, so a large unexplained deviation is never asserted diploid.
- **Anti-alignment gate.** A cell called `normal` that carries a large, coherent
  deviation pointing *against* the consensus template is now downgraded to
  `uncertain` rather than asserted normal. Under a signed score such a cell sits at
  the bottom of the range, next to genuinely diploid cells, and no emitted column
  distinguished the two. Like the other two gates it only ever moves a cell *to*
  `uncertain` — it never promotes anything to `tumor`, so it cannot inflate the
  tumor set, and recall and precision are unchanged by construction. Worst-case
  specificity cost measured on the benchmark cohort with the reference-pool
  protection deliberately disabled: 0.45% of known normals (1,275 of 286,101), and
  zero on the supervised runs where those cells are protected.

### Fixed

- `classify_cells` no longer raises `ValueError: attempt to get argmax of an empty
  sequence` on an empty segmentation, which `detect_segments` legitimately returns.
- `weighted_median_rows` is computed in row blocks (~64 MB of intermediates whatever
  the segment count) instead of ~8 GB of whole-matrix temporaries at 800k × 500, and
  guards against zero total weight rather than returning each row's minimum.
- The reference-relative deviation keeps its input's `float32` precision instead of
  promoting to `float64`, and `abs(dev)` is materialized once rather than three times.

### Packaging and tooling

These three landed on `main` after the `v1.0.1` tag was cut, so despite their commit
dates they ship to users for the first time here, not in 1.0.1.

- The package version is read from `src/kopya/__init__.py` at build time, so
  `pyproject.toml` no longer carries a second copy to keep in sync.
- The classifier's outlier-exclusion fence default widens from 7.0 to 12.0 IQR. Under
  this release's score the change is a no-op — the fence's activation guard was not
  passing either way (see *Known limitations*).
- README: PyPI version, Python version and license badges.

### Known limitations

- `consensus_template` estimates **one** direction, so two roughly equal opposing
  subclones cannot both be *scored*: the template locks onto one, that clone is
  called tumor, and the other projects symmetrically negative. The opposing clone is
  no longer silently called `normal` — the anti-alignment gate downgrades it to
  `uncertain` and counts it in `qc.json` — but it is still not recovered as tumor.
  Pinned by `test_consensus_template_opposing_subclones_surface_as_uncertain`.
  Resolving it properly needs multiple templates with best-aligned scoring, which
  changes the scoring contract and is deliberately deferred to its own change.

  How much this matters in practice, measured rather than assumed: across the 12
  benchmark patients with more than one discovered subclone, all 56 pairwise
  correlations between subclone CN profiles average **r = +0.65**, 53 of 56 are
  positive, and only one falls below −0.3. Real subclones share a clonal backbone
  and differ by private events; the equal-and-opposite pair this limitation needs is
  a synthetic construct, not a shape the cohort exhibits. The gate exists so that a
  sample which *does* exhibit it says so in its own output.
- The GMM's decision threshold is uncalibrated and errs liberal — on the held-out
  protocol, specificity falls on 8 of 10 patients (mean 0.900 → 0.824), badly on two.
  The score is better at every operating point; the *cut point* is not yet a
  deliberate choice. An explicit operating-point control is the intended follow-up.
- `outlier_fence_mult` has never fired on the benchmark cohort at any multiplier —
  every version of the fence's second condition that shipped was either unreachable or
  wrong — so the multiplier itself remains untuned. Now that the condition is the
  coherent fraction, `n_outlier_fenced` is the number to watch on a real run.
- The gold-standard AUCs recorded in `tests/external/GOLD_STANDARD_TESTING.md` were
  measured under the previous score and have not been re-measured; the datasets are
  not present in the environment where the classifier changed. That document now
  carries a banner saying so at the top, so its tables cannot be read as current.

## 1.0.1

- Publish as `valius-kopya` on PyPI, and add the Trusted-Publishing release workflow.

## 1.0.0

- First public release.
