# Example results

A complete set of `kopya run` outputs, committed so you can explore the
file formats without running the pipeline yourself. Paired with the walkthrough
notebook: [`../notebooks/working_with_results.ipynb`](../notebooks/working_with_results.ipynb).

## `tiny_simulated/`

Generated from the 500-cell simulated fixture
[`tests/fixtures/tiny_simulated.h5ad`](../tests/fixtures/tiny_simulated.h5ad)
(200 normal + 300 tumor cells) with three planted events:

| Event | Truth |
|-------|-------|
| **chr7** | whole-chromosome gain (2.5×) |
| **chr10** | whole-chromosome loss (0.4×) |
| **MYC (chr8)** | focal amplification (5×) |

Because the ground truth is known, this is a good place to build intuition for
what the numbers mean. The tumor population reads chr7 ≈ 1.8 and chr10 ≈ 0.34 in
`chr_cnv_matrix.csv`; normals sit near 1.0.

> **Note on `qc.json`'s `baseline_method: gmm_fallback`.** This synthetic fixture
> has no immune/stromal expression signatures for the baseline cascade to lock
> onto, so it correctly falls through to the GMM fallback tier. The calls are
> still exact (300 tumor / 200 normal); on real immune-infiltrated tumors the
> cascade normally settles on the higher `signature` or `variance` tier.

### Files

| File | What it is |
|------|-----------|
| `prediction.csv` | per-cell `class` / `confidence` / `tumor_score` / `subclone` / `n_segments_altered` / `low_complexity` |
| `segments.parquet` | segment-to-genome map (`segment_id`, `chr`, gene-axis indices, `start_bp`/`end_bp`, `n_genes`, `tumor_mean`) |
| `cn_per_segment.npz` | dense per-cell × per-segment CN matrix (`cn`) + `cell_barcodes` |
| `cn_per_segment_denoised.npz` | same shape, but **recentered + denoised** (opt-in via `--denoise-outputs`): a clean gain/loss matrix (0 ≈ diploid, >0 gain, <0 loss) where the normal floor collapses to ~0 and hallmark events survive |
| `chr_cnv_matrix.csv` | per-cell × per-chromosome summary, 1.0-centered |
| `tiny_simulated_clones.seg` | IGV-loadable per-subclone consensus segments |
| `qc.json` | run metadata (parameters, counts, baseline method, timings) |
| `heatmap.png` | inferCNV-style genome heatmap rendered by `plot-heatmap` |

See the [Outputs](../README.md#outputs) section of the main README for the full
value-scale reference (which files are log-space, which are 1.0-centered, and
where negative values are expected).

## Regenerating

```bash
kopya run \
    --anndata tests/fixtures/tiny_simulated.h5ad \
    --out-dir examples/tiny_simulated \
    --sample tiny_simulated \
    --denoise-outputs        # also write cn_per_segment_denoised.npz

kopya plot-heatmap \
    --run-dir examples/tiny_simulated \
    --out examples/tiny_simulated/heatmap.png \
    --scale linear
```

`qc.json` records per-step wall-clock timings and the package version, so those
fields will differ slightly on each regeneration; the analysis outputs are
deterministic.
