"""Unit tests for outputs.py — prediction CSV, chr CN matrix, IGV .seg."""

import numpy as np
import pandas as pd
import pytest

from kopya.outputs import (
    _logspace_to_one_centered,
    center_chr_cnv_matrix,
    compute_chr_cnv_matrix,
    segments_with_coordinates,
    write_chr_cnv_matrix_csv,
    write_clones_seg,
    write_prediction_csv,
)
from kopya.segment import (
    detect_segments,
    per_cell_segment_cn,
    per_segment_normal_baseline,
)
from kopya.smooth import (
    DEFAULT_SMOOTH_WINDOW,
    center_against_baseline,
    smooth_along_chromosomes,
)

# Reuse the zero-inflated pedestal fixture (see test_segment).
from tests.test_segment import _pedestal_adata


def _tiny_pipeline_state(seed=0):
    """Build a coherent (segments, cn_matrix, prediction_df, var_coords) tuple.

    Mirrors what the CLI passes to outputs.write_* — three chromosomes, mixed
    gain/loss segments, and a prediction frame with planted tumor/normal calls
    plus two subclones.

    Returns:
        Tuple (cn_matrix, segments, prediction_df, var_coords, barcodes).
    """
    rng = np.random.default_rng(seed)

    # Three chromosomes, two segments each (gain on first, neutral on second).
    segments = pd.DataFrame({
        "chr": ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"],
        "start_idx": [0, 50, 100, 150, 200, 250],
        "end_idx":   [50, 100, 150, 200, 250, 300],
        "n_genes":   [50, 50, 50, 50, 50, 50],
        "tumor_mean": [0.4, 0.0, -0.4, 0.0, 0.0, 0.0],
    })

    n_cells = 20
    n_segments = len(segments)
    cn_matrix = rng.normal(loc=0.0, scale=0.05, size=(n_cells, n_segments)).astype(np.float32)
    # Tumor cells (rows 10-19) carry the planted CN shifts on segments 0 and 2.
    cn_matrix[10:, 0] += 0.4
    cn_matrix[10:, 2] -= 0.4

    barcodes = [f"cell_{i:04d}" for i in range(n_cells)]
    prediction_df = pd.DataFrame({
        "class": ["normal"] * 10 + ["tumor"] * 10,
        "confidence": [0.95] * 20,
        "tumor_score": [0.1] * 10 + [0.8] * 10,
        "subclone": [""] * 10 + ["subclone_1"] * 5 + ["subclone_2"] * 5,
        "n_segments_altered": [0] * 10 + [2] * 10,
    }, index=pd.Index(barcodes, name="barcode"))

    # var coords: 300 genes across the 3 synthetic chromosomes.
    var_rows = []
    chr_lookup = []
    for i, row in segments.iterrows():
        for j in range(row["start_idx"], row["end_idx"]):
            var_rows.append({
                "chr": row["chr"],
                "start": j * 1000,
                "end": j * 1000 + 800,
            })
            chr_lookup.append(row["chr"])
    var_coords = pd.DataFrame(var_rows, index=[f"GENE{i:04d}" for i in range(len(var_rows))])

    return cn_matrix, segments, prediction_df, var_coords, barcodes


def test_logspace_to_one_centered_mapping():
    """exp(0)=1, exp(0.69)~2, exp(-0.69)~0.5 — the documented gain/loss scale."""
    values = np.array([0.0, np.log(2.0), -np.log(2.0)])
    out = _logspace_to_one_centered(values)
    np.testing.assert_allclose(out, [1.0, 2.0, 0.5], rtol=1e-6)


def test_segments_with_coordinates_maps_gene_indices_to_bp():
    """segments.parquet enrichment adds a self-contained segment→location mapping."""
    _, segments, _, var_coords, _ = _tiny_pipeline_state()

    enriched = segments_with_coordinates(segments, var_coords)

    # Additive: originals preserved, plus segment_id/start_bp/end_bp.
    for col in ("chr", "start_idx", "end_idx", "n_genes", "tumor_mean"):
        assert col in enriched.columns
    assert {"segment_id", "start_bp", "end_bp"} <= set(enriched.columns)
    # segment_id is the 0-based positional key matching cn_per_segment columns.
    assert enriched["segment_id"].tolist() == list(range(len(segments)))

    # Coordinates come from the first gene's start and the last gene's end
    # (end_idx exclusive). The fixture lays gene j at [j*1000, j*1000+800).
    first = enriched.iloc[0]
    assert first["start_bp"] == 0                      # gene 0 start
    assert first["end_bp"] == 49 * 1000 + 800          # gene 49 end (end_idx=50)
    second = enriched.iloc[1]
    assert second["start_bp"] == 50 * 1000             # gene 50 start
    assert second["end_bp"] == 99 * 1000 + 800         # gene 99 end (end_idx=100)

    # start_bp < end_bp for every segment; the input frame is not mutated.
    assert (enriched["start_bp"] < enriched["end_bp"]).all()
    assert "segment_id" not in segments.columns


def test_segments_with_coordinates_spans_overlapping_genes():
    """end_bp spans the max gene end in the segment, not just the last gene's.

    Genes are ordered by start, but with overlapping/nested genes the gene with
    the greatest *start* is not necessarily the one with the greatest *end*, so
    using only the last gene would truncate the segment's genomic span.
    """
    segments = pd.DataFrame({
        "chr": ["chr1"],
        "start_idx": [0],
        "end_idx": [3],
        "n_genes": [3],
        "tumor_mean": [0.0],
    })
    # Gene 1 nests a larger end (900) than the last-by-start gene 2 (350).
    var_coords = pd.DataFrame(
        {"chr": ["chr1", "chr1", "chr1"],
         "start": [100, 200, 300],
         "end": [150, 900, 350]},
        index=["G0", "G1", "G2"],
    )

    enriched = segments_with_coordinates(segments, var_coords)
    assert int(enriched.iloc[0]["start_bp"]) == 100   # min start across the segment
    assert int(enriched.iloc[0]["end_bp"]) == 900     # max end, NOT gene 2's 350


def test_segments_with_coordinates_appends_without_reordering():
    """New columns are appended; the original schema keeps its positions."""
    _, segments, _, var_coords, _ = _tiny_pipeline_state()
    original_cols = list(segments.columns)

    enriched = segments_with_coordinates(segments, var_coords)
    # Original columns keep their exact leading positions; new ones are appended.
    assert list(enriched.columns)[: len(original_cols)] == original_cols
    assert list(enriched.columns)[len(original_cols):] == ["segment_id", "start_bp", "end_bp"]


def test_segments_with_coordinates_handles_empty():
    """A zero-segment run yields an empty enriched table, not an IndexError.

    detect_segments() builds ``DataFrame([], columns=[...])`` when it finds no
    segments, so the index columns are object-dtype. NumPy rejects object arrays
    as fancy indices, so the helper must coerce them — reproduce that exact frame
    (do NOT pre-cast) to guard the crash.
    """
    _, _, _, var_coords, _ = _tiny_pipeline_state()
    empty = pd.DataFrame([], columns=["chr", "start_idx", "end_idx", "n_genes", "tumor_mean"])
    assert empty["start_idx"].dtype == object  # matches detect_segments' empty output

    enriched = segments_with_coordinates(empty, var_coords)
    assert len(enriched) == 0
    assert {"segment_id", "start_bp", "end_bp"} <= set(enriched.columns)


def test_write_prediction_csv_schema(tmp_path):
    """prediction.csv has the documented columns in stable order, indexed by barcode."""
    _, _, prediction_df, _, _ = _tiny_pipeline_state()

    out_path = write_prediction_csv(prediction_df, tmp_path / "prediction.csv")

    parsed = pd.read_csv(out_path, index_col="barcode")
    assert list(parsed.columns) == ["class", "confidence", "tumor_score", "subclone", "n_segments_altered"]
    # Values round-trip exactly for non-float columns.
    assert (parsed["class"].astype(str) == prediction_df["class"].astype(str)).all()


def test_compute_chr_cnv_matrix_shape_and_values(tmp_path):
    """Per-chromosome aggregation has the right shape and lands near 1.0 for normals."""
    cn_matrix, segments, _, _, _ = _tiny_pipeline_state()
    chrom_order = ["chr1", "chr2", "chr3"]

    matrix, chroms_present = compute_chr_cnv_matrix(cn_matrix, segments, chrom_order)

    assert chroms_present == chrom_order
    assert matrix.shape == (cn_matrix.shape[0], 3)
    # Normal cells (rows 0-9) should sit near 1.0 (exp(0)=1) on every chromosome.
    np.testing.assert_allclose(matrix[:10, :], 1.0, atol=0.1)
    # Tumor cells (rows 10-19) on chr1 should be > 1 (planted gain) and on
    # chr2 should be < 1 (planted loss).
    assert matrix[10:, 0].mean() > 1.05
    assert matrix[10:, 1].mean() < 0.95
    # chr3 had no planted shift → tumor cells stay near 1.
    np.testing.assert_allclose(matrix[10:, 2], 1.0, atol=0.1)


def test_write_chr_cnv_matrix_csv_round_trip(tmp_path):
    """Round-trip the chr CN matrix CSV: barcodes match, columns match."""
    cn_matrix, segments, _, _, barcodes = _tiny_pipeline_state()
    matrix, chroms = compute_chr_cnv_matrix(cn_matrix, segments, ["chr1", "chr2", "chr3"])

    out_path = write_chr_cnv_matrix_csv(matrix, barcodes, chroms, tmp_path / "chr.csv")

    parsed = pd.read_csv(out_path, index_col="barcode")
    assert list(parsed.columns) == chroms
    np.testing.assert_allclose(parsed.to_numpy(), matrix, rtol=1e-5)


def test_write_clones_seg_igv_format(tmp_path):
    """The .seg file has IGV-canonical columns and one row per (clone, segment)."""
    cn_matrix, segments, prediction_df, var_coords, _ = _tiny_pipeline_state()

    out_path = write_clones_seg(
        cn_matrix=cn_matrix,
        segments=segments,
        prediction_df=prediction_df,
        var_coords=var_coords,
        sample="smoke",
        out_path=tmp_path / "smoke_clones.seg",
    )

    parsed = pd.read_csv(out_path, sep="\t")
    assert list(parsed.columns) == ["ID", "chrom", "loc.start", "loc.end", "num.mark", "seg.mean"]
    # Two subclones × 6 segments = 12 rows.
    assert len(parsed) == 12
    # Sample prefix is applied to every clone ID.
    assert (parsed["ID"].str.startswith("smoke::subclone_")).all()
    # loc.start and loc.end are integer genomic coordinates.
    assert parsed["loc.start"].dtype.kind == "i"
    assert parsed["loc.end"].dtype.kind == "i"
    # seg.mean for chr1 segment 0 (planted gain) is positive in both clones.
    chr1_seg0 = parsed[(parsed["chrom"] == "chr1") & (parsed["loc.start"] == 0)]
    assert (chr1_seg0["seg.mean"] > 0.2).all()
    # seg.mean for chr2 segment 0 (planted loss) is negative.
    chr2_seg0 = parsed[(parsed["chrom"] == "chr2") & (parsed["loc.start"] == 100_000)]
    assert (chr2_seg0["seg.mean"] < -0.2).all()


def _pedestal_pipeline_state(seed=0):
    """Full M3 state on the zero-inflated pedestal synthetic.

    Returns everything the absolute-output writers consume plus the per-segment
    diploid reference the CLI threads into them:
        (cn_matrix, segments, seg_baseline, prediction_df, var_coords,
         normal_mask, loss_label).
    """
    adata, normal_mask, loss_label = _pedestal_adata(seed=seed)
    chr_labels = adata.var["chr"].to_numpy()
    centered = center_against_baseline(adata, normal_mask)
    smoothed = smooth_along_chromosomes(centered, chr_labels=chr_labels, window=DEFAULT_SMOOTH_WINDOW)
    segments = detect_segments(smoothed, normal_mask=normal_mask, chr_labels=chr_labels)
    cn_matrix = per_cell_segment_cn(smoothed, segments)
    seg_baseline = per_segment_normal_baseline(cn_matrix, normal_mask)

    # Label planted-normal cells 'normal' and planted-tumor cells 'tumor' (one clone),
    # mirroring what classify_cells would produce on this clean synthetic.
    n = adata.n_obs
    prediction_df = pd.DataFrame(
        {
            "class": np.where(normal_mask, "normal", "tumor"),
            "confidence": [0.95] * n,
            "tumor_score": np.where(normal_mask, 0.1, 0.9),
            "subclone": np.where(normal_mask, "", "subclone_1"),
            "n_segments_altered": np.where(normal_mask, 0, 2),
        },
        index=pd.Index(adata.obs_names, name="barcode"),
    )
    var_coords = adata.var.copy()
    var_coords["start"] = np.arange(adata.n_vars) * 1000
    var_coords["end"] = np.arange(adata.n_vars) * 1000 + 800
    return cn_matrix, segments, seg_baseline, prediction_df, var_coords, normal_mask, loss_label


def test_chr_cnv_matrix_diploid_centers_on_one_with_baseline():
    """With the diploid pedestal subtracted, normal cells center on 1.0.

    The CLI passes a SCALAR pedestal (median of the per-segment reference) here to
    keep the bulk-validated per-chromosome shape intact; this test mirrors that.
    Without any baseline the per-gene centering pedestal pushes diploid normals
    well above 1.0 (the ~1.27 the issue reports); subtracting the pedestal
    collapses them back to ~1.0, and the planted loss stays clearly < normal.
    """
    cn_matrix, segments, seg_baseline, _, _, normal_mask, loss_label = _pedestal_pipeline_state()
    chrom_order = ["chr_synth_1", "chr_synth_2", "chr_synth_3"]
    chr_pedestal = float(np.median(seg_baseline))  # scalar, as the CLI uses

    fixed, chroms = compute_chr_cnv_matrix(cn_matrix, segments, chrom_order, baseline=chr_pedestal)
    raw, _ = compute_chr_cnv_matrix(cn_matrix, segments, chrom_order)

    # Contrast: uncorrected diploid normals sit well above 1.0 (the pedestal bug).
    assert np.median(raw[normal_mask], axis=0).max() > 1.2, (
        "fixture no longer exhibits the pedestal — the corrected assertion is vacuous"
    )
    # Corrected: normal cells' per-chromosome median lands on ~1.0.
    fixed_norm_median = np.median(fixed[normal_mask], axis=0)
    np.testing.assert_allclose(fixed_norm_median, 1.0, atol=0.1)

    # The planted loss is preserved: tumor << normal on the lost chromosome.
    col = chroms.index(loss_label)
    assert np.median(fixed[~normal_mask, col]) < np.median(fixed[normal_mask, col]) - 0.15


def test_chr_cnv_matrix_scalar_pedestal_preserves_tumor_shape():
    """The scalar-pedestal correction is a pure log-space shift of the tumor profile.

    This is the property that keeps matched-bulk Pearson concordance intact: a
    scalar pedestal only rescales chr_cnv by a constant factor, so the tumor
    pseudobulk log2 profile shifts by a constant and its *shape* (hence its
    correlation with bulk) is unchanged. A per-segment baseline would not have
    this property.
    """
    cn_matrix, segments, seg_baseline, _, _, normal_mask, _ = _pedestal_pipeline_state()
    chrom_order = ["chr_synth_1", "chr_synth_2", "chr_synth_3"]
    chr_pedestal = float(np.median(seg_baseline))

    raw, _ = compute_chr_cnv_matrix(cn_matrix, segments, chrom_order)
    fixed, _ = compute_chr_cnv_matrix(cn_matrix, segments, chrom_order, baseline=chr_pedestal)

    tumor = ~normal_mask
    raw_log2 = np.log2(raw[tumor].mean(axis=0))
    fixed_log2 = np.log2(fixed[tumor].mean(axis=0))
    # The two profiles differ by a single constant across chromosomes.
    diff = fixed_log2 - raw_log2
    np.testing.assert_allclose(diff, diff[0], atol=1e-5)


# ---------------------------------------------------------------------------
# center_chr_cnv_matrix() — the optional per-cell centering behind
# --centered-chr-matrix. Additive: none of this touches compute_chr_cnv_matrix.
# ---------------------------------------------------------------------------

def _offset_matrix():
    """A 1.0-centered chr matrix with a planted gain, loss, and per-cell offset.

    Column 2 is a gain and column 4 a loss in every cell; the per-cell factors
    span 0.8x-1.6x, the shape of the offset seen on real runs.
    """
    rng = np.random.default_rng(7)
    base = np.exp(rng.normal(0.0, 0.02, size=(8, 6)))
    base[:, 2] *= 1.4
    base[:, 4] *= 0.6
    offsets = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.4, 1.6])[:, None]
    return base, offsets, base * offsets


def test_center_chr_cnv_matrix_row_medians_are_one():
    """The defining property: every cell's median chromosome lands on 1.0."""
    _, _, raw = _offset_matrix()
    centered = center_chr_cnv_matrix(raw)
    np.testing.assert_allclose(np.median(centered, axis=1), 1.0, atol=1e-12)


def test_center_chr_cnv_matrix_preserves_shape_ids_and_within_row_ratios():
    """Dividing a row by a constant cannot add or remove chromosome-level signal.

    This is what makes the centered file safe: the ratio between any two
    chromosomes of the same cell — which is where a gain or loss actually lives —
    comes through untouched, and the matrix keeps its shape and dtype so the
    barcodes/columns written beside it still line up.
    """
    _, _, raw = _offset_matrix()
    centered = center_chr_cnv_matrix(raw)

    assert centered.shape == raw.shape
    assert centered.dtype == raw.dtype
    assert np.all(centered > 0), "1.0-centered values must stay strictly positive"

    # Every pairwise within-row ratio survives exactly.
    for j in range(raw.shape[1]):
        np.testing.assert_allclose(centered[:, j] / centered[:, 0], raw[:, j] / raw[:, 0], rtol=1e-10)

    # The planted gain and loss are still on the correct side of 1.0.
    assert np.all(centered[:, 2] > 1.2)
    assert np.all(centered[:, 4] < 0.8)


def test_center_chr_cnv_matrix_removes_a_planted_per_cell_offset():
    """A known per-cell factor is removed, and only that factor.

    The offset is what makes cells incomparable to each other at a fixed
    chromosome; after centering, the same chromosome reads the same across cells
    that differ only by their offset.
    """
    base, offsets, raw = _offset_matrix()
    centered = center_chr_cnv_matrix(raw)

    # Before: the row medians span the planted 0.8x-1.6x range.
    assert np.ptp(np.median(raw, axis=1)) > 0.7
    # After: they are identical, so the spread is gone.
    assert np.ptp(np.median(centered, axis=1)) < 1e-12

    # And centering recovers the offset-free matrix up to its own row medians —
    # i.e. it removed the planted factor, not some of the biology with it.
    expected = base / np.median(base, axis=1, keepdims=True)
    np.testing.assert_allclose(centered, expected, rtol=1e-10)

    # Ranking cells at a fixed chromosome is the use case this fixes: on the raw
    # matrix the deepest-offset cell tops the gain column regardless of biology.
    assert np.argmax(raw[:, 2]) == np.argmax(offsets[:, 0])
    assert np.median(centered[:, 2]) > 1.2


def test_center_chr_cnv_matrix_does_not_mutate_its_input():
    """The raw matrix the CLI just wrote must not change underneath it."""
    _, _, raw = _offset_matrix()
    before = raw.copy()
    center_chr_cnv_matrix(raw)
    np.testing.assert_array_equal(raw, before)


def test_center_chr_cnv_matrix_on_pedestal_fixture_keeps_the_planted_loss():
    """End of the real path: centering the pedestal synthetic's chr matrix.

    The scalar-pedestal correction leaves a per-cell spread behind (that is the
    documented trade-off); centering removes it, and the planted loss survives.
    """
    cn_matrix, segments, seg_baseline, _, _, normal_mask, loss_label = _pedestal_pipeline_state()
    chrom_order = ["chr_synth_1", "chr_synth_2", "chr_synth_3"]
    chr_pedestal = float(np.median(seg_baseline))

    raw, chroms = compute_chr_cnv_matrix(cn_matrix, segments, chrom_order, baseline=chr_pedestal)
    centered = center_chr_cnv_matrix(raw)

    # The raw file genuinely carries a per-cell spread — otherwise this is vacuous.
    assert np.median(raw, axis=1).std() > 0.01
    assert np.median(centered, axis=1).std() < 1e-12

    # The planted loss is still a loss relative to the normals.
    col = chroms.index(loss_label)
    assert np.median(centered[~normal_mask, col]) < np.median(centered[normal_mask, col])


def test_clones_seg_centered_on_zero_with_baseline(tmp_path):
    """With the baseline, .seg seg.mean is a log ratio centered on 0.

    Diploid segments read ~0 and the planted loss reads clearly negative — the
    "seg.mean centered on 0" the .seg header promises. Without the baseline the
    pedestal would lift every seg.mean positive.
    """
    cn_matrix, segments, seg_baseline, prediction_df, var_coords, _, loss_label = _pedestal_pipeline_state()

    out_path = write_clones_seg(
        cn_matrix=cn_matrix,
        segments=segments,
        prediction_df=prediction_df,
        var_coords=var_coords,
        sample="pedestal",
        out_path=tmp_path / "pedestal_clones.seg",
        baseline=seg_baseline,
    )
    parsed = pd.read_csv(out_path, sep="\t")

    loss_seg = parsed.loc[parsed["chrom"] == loss_label, "seg.mean"]
    diploid_seg = parsed.loc[parsed["chrom"] != loss_label, "seg.mean"]

    assert (loss_seg < -0.15).all(), f"planted loss not negative in .seg: {loss_seg.to_numpy()}"
    assert diploid_seg.abs().max() < 0.2, (
        f"diploid seg.mean not centered on 0: absmax = {diploid_seg.abs().max():.3f}"
    )


def test_write_clones_seg_empty_when_no_tumor(tmp_path):
    """No-tumor cohort produces an empty .seg with the right header."""
    cn_matrix, segments, prediction_df, var_coords, _ = _tiny_pipeline_state()
    # Flip every cell to 'normal' so no tumor remains.
    prediction_df = prediction_df.copy()
    prediction_df["class"] = "normal"
    prediction_df["subclone"] = ""

    out_path = write_clones_seg(
        cn_matrix=cn_matrix,
        segments=segments,
        prediction_df=prediction_df,
        var_coords=var_coords,
        sample="smoke",
        out_path=tmp_path / "smoke_clones.seg",
    )

    parsed = pd.read_csv(out_path, sep="\t")
    assert list(parsed.columns) == ["ID", "chrom", "loc.start", "loc.end", "num.mark", "seg.mean"]
    assert len(parsed) == 0
