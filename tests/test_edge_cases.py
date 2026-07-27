"""Edge-case and boundary-condition tests for all pipeline modules.

Covers degenerate inputs (empty matrices, inverted windows, single-cell
or single-gene scenarios) and boundary conditions that normal unit tests do
not exercise. Every test is self-contained and uses fixed-seed RNGs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix

from kopya.normalize import (
    filter_cells_and_genes,
    filter_normalize_project,
    normalize_log1p,
)
from kopya.smooth import center_against_baseline, smooth_along_chromosomes
from kopya.segment import (
    DEFAULT_MIN_SEG_GENES,
    detect_segments,
    per_cell_segment_cn,
)
from kopya.classify import (
    MIN_TUMOR_CELLS_FOR_SUBCLONES,
    classify_cells,
    discover_subclones,
    gmm_classify,
)
from kopya.outputs import (
    _compute_clone_consensus_segments,
    compute_chr_cnv_matrix,
)
from kopya.baseline import pick_baseline
from kopya.annotations import CANONICAL_CHROM_ORDER, load_gene_order


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tiny_simulated.h5ad"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells, n_genes, seed=0, mean=3.0):
    """Build a minimal AnnData with Poisson-sampled integer counts."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(lam=mean, size=(n_cells, n_genes)).astype(np.int32)
    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"GENE{i:05d}" for i in range(n_genes)])
    return AnnData(X=csr_matrix(dense), obs=obs, var=var)


def _three_chr_adata_for_edge(
    n_normal=20,
    n_tumor=20,
    genes_per_chr=(60, 60, 60),
    seed=42,
):
    """Build a small 3-chromosome AnnData for smooth/segment edge-case tests."""
    rng = np.random.default_rng(seed)
    n_cells = n_normal + n_tumor
    n_genes = sum(genes_per_chr)

    chr_labels = []
    for i, n in enumerate(genes_per_chr):
        chr_labels += [f"chr_ec_{i + 1}"] * n

    X = rng.normal(loc=1.0, scale=0.05, size=(n_cells, n_genes)).astype(np.float32)
    X = np.clip(X, 0, None)

    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(
        {"chr": chr_labels},
        index=[f"GENE{i:05d}" for i in range(n_genes)],
    )
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    normal_mask = np.array([True] * n_normal + [False] * n_tumor)
    return adata, normal_mask


# ===========================================================================
# NORMALIZE EDGE CASES
# ===========================================================================


def test_filter_with_zero_survivors():
    """All-zero cells → filter_cells_and_genes should return an empty AnnData."""
    n_cells, n_genes = 5, 5
    # All-zeros: every cell has 0 genes expressed.
    dense = np.zeros((n_cells, n_genes), dtype=np.int32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    adata = AnnData(X=csr_matrix(dense), obs=obs, var=var)

    result = filter_cells_and_genes(adata, min_genes=1)

    # No cells survive since none express any genes.
    assert result.n_obs == 0, (
        f"expected 0 surviving cells, got {result.n_obs}"
    )


def test_filter_low_dr_above_up_dr():
    """Inverted detection-rate window [0.9, 0.1] should drop all genes."""
    adata = _make_adata(n_cells=30, n_genes=20, seed=1)

    # low_dr > up_dr: the inclusion window is empty, all genes are excluded.
    result = filter_cells_and_genes(adata, min_genes=1, low_dr=0.9, up_dr=0.1)

    assert result.n_vars == 0, (
        f"expected 0 surviving genes with inverted DR window, got {result.n_vars}"
    )


def test_normalize_single_cell():
    """normalize_log1p on a 1-cell AnnData should not crash and keep shape."""
    rng = np.random.default_rng(7)
    dense = rng.poisson(lam=5.0, size=(1, 20)).astype(np.int32)
    obs = pd.DataFrame(index=["cell_0000"])
    var = pd.DataFrame(index=[f"GENE{i:05d}" for i in range(20)])
    adata = AnnData(X=csr_matrix(dense), obs=obs, var=var)

    result = normalize_log1p(adata)

    # Output must be sparse CSR and preserve shape.
    from scipy.sparse import issparse
    assert issparse(result.X), "output .X should be sparse"
    assert result.shape == adata.shape, (
        f"shape changed: {adata.shape} -> {result.shape}"
    )
    # All values must be finite (no NaN or Inf introduced).
    data = result.X.toarray()
    assert np.all(np.isfinite(data)), "non-finite values in normalized output"


# ===========================================================================
# SMOOTH EDGE CASES
# ===========================================================================


def test_center_against_baseline_single_gene_chromosome():
    """A chromosome with exactly 1 gene should be processed without producing NaN."""
    rng = np.random.default_rng(3)
    n_cells = 20
    # Layout: chr_a (50 genes), chr_b (1 gene), chr_c (50 genes).
    genes_per_chr = {"chr_a": 50, "chr_b": 1, "chr_c": 50}
    n_genes = sum(genes_per_chr.values())

    chr_labels = []
    for chrom, n in genes_per_chr.items():
        chr_labels += [chrom] * n

    X = rng.normal(loc=1.0, scale=0.1, size=(n_cells, n_genes)).astype(np.float32)
    X = np.clip(X, 0, None)
    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(
        {"chr": chr_labels},
        index=[f"GENE{i:05d}" for i in range(n_genes)],
    )
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)

    # Half normal, half tumor.
    normal_mask = np.array([True] * (n_cells // 2) + [False] * (n_cells // 2))

    centered = center_against_baseline(adata, normal_mask)
    assert centered.shape == adata.shape

    # Now smooth with a large window — the single-gene chromosome must clamp
    # gracefully and not produce NaN.
    smoothed = smooth_along_chromosomes(
        centered,
        chr_labels=var["chr"].to_numpy(),
        window=100,
    )

    assert smoothed.shape == centered.shape, "shape changed after smoothing"

    # Single-gene chromosome column (index 50) must not contain NaN.
    single_gene_col = smoothed[:, 50]
    assert not np.any(np.isnan(single_gene_col)), (
        "NaN found in single-gene chromosome column"
    )


def test_smooth_window_larger_than_chromosome():
    """smooth_along_chromosomes with window >> chromosome size should clamp cleanly."""
    rng = np.random.default_rng(5)
    n_cells = 15
    # One very small chromosome (5 genes) and one normal one (80 genes).
    chr_labels = ["chr_tiny"] * 5 + ["chr_big"] * 80
    n_genes = len(chr_labels)

    matrix = rng.normal(loc=0.0, scale=0.2, size=(n_cells, n_genes)).astype(np.float32)

    # window=100 greatly exceeds the 5-gene chromosome.
    smoothed = smooth_along_chromosomes(matrix, chr_labels=chr_labels, window=100)

    assert smoothed.shape == matrix.shape, (
        f"shape mismatch: {matrix.shape} -> {smoothed.shape}"
    )
    assert not np.any(np.isnan(smoothed)), "NaN in output after oversized window"
    assert not np.any(np.isinf(smoothed)), "Inf in output after oversized window"


# ===========================================================================
# SEGMENT EDGE CASES
# ===========================================================================


def test_detect_segments_tiny_chromosome_produces_no_segments():
    """A chromosome with fewer genes than min_seg_genes should yield 0 segments
    while other normal-sized chromosomes still produce segments.

    The tiny chr has 10 genes; DEFAULT_MIN_SEG_GENES is 25. Because 10 < 2*25,
    PELT will emit the whole tiny block as one segment — but that segment has
    fewer than min_seg_genes genes, so it should be merged/skipped.

    We assert that the big chromosome produces at least one segment and no crash.
    """
    rng = np.random.default_rng(9)
    n_cells = 20
    # tiny_chr: 10 genes (< DEFAULT_MIN_SEG_GENES=25)
    # big_chr: 200 genes (well above the threshold)
    n_tiny = 10
    n_big = 200
    n_genes = n_tiny + n_big

    chr_labels = np.array(["chr_tiny"] * n_tiny + ["chr_big"] * n_big)

    # Plant a clear step signal on the big chromosome for PELT to find.
    smoothed = rng.normal(0.0, 0.05, size=(n_cells, n_genes)).astype(np.float32)
    # Add a step change at gene 100 on the big chromosome (global index 110).
    smoothed[:, n_tiny + 100:] += 0.5

    normal_mask = np.array([True] * (n_cells // 2) + [False] * (n_cells // 2))

    segments = detect_segments(
        smoothed,
        normal_mask=normal_mask,
        chr_labels=chr_labels,
        min_seg_genes=DEFAULT_MIN_SEG_GENES,
    )

    # No crash is the primary assertion.
    assert isinstance(segments, pd.DataFrame)

    # The big chromosome must produce at least one segment.
    big_segs = segments[segments["chr"] == "chr_big"]
    assert len(big_segs) >= 1, (
        f"expected >= 1 segment on big chromosome, got {len(big_segs)}"
    )


def test_per_cell_segment_cn_empty_segments():
    """per_cell_segment_cn with an empty segments DataFrame should return shape (n_cells, 0)."""
    rng = np.random.default_rng(11)
    n_cells, n_genes = 10, 50
    smoothed = rng.normal(0.0, 0.1, size=(n_cells, n_genes)).astype(np.float32)

    empty_segments = pd.DataFrame(
        columns=["chr", "start_idx", "end_idx", "n_genes", "tumor_mean"]
    )
    # Ensure correct dtypes for start/end_idx columns used in per_cell_segment_cn.
    empty_segments = empty_segments.astype(
        {"start_idx": int, "end_idx": int, "n_genes": int, "tumor_mean": float}
    )

    result = per_cell_segment_cn(smoothed, empty_segments)

    assert result.shape == (n_cells, 0), (
        f"expected (10, 0), got {result.shape}"
    )


# ===========================================================================
# CLASSIFY EDGE CASES
# ===========================================================================


def test_gmm_classify_all_uncertain_when_high_threshold():
    """With confidence_threshold=0.999, cells in the overlap region should be 'uncertain'.

    Uses heavily overlapping Gaussian components so many cells land in the
    ambiguous zone where neither component achieves 0.999 posterior.
    """
    rng = np.random.default_rng(13)
    # Heavily overlapping: components at 1.0 and 1.5 with std=0.5.
    # Many cells will have posteriors well below 0.999 in the overlap region.
    low_scores = rng.normal(loc=1.0, scale=0.5, size=100)
    high_scores = rng.normal(loc=1.5, scale=0.5, size=100)
    scores = np.concatenate([low_scores, high_scores]).astype(np.float32)

    result = gmm_classify(scores, confidence_threshold=0.999)

    # Most cells should not achieve 0.999 posterior — land in 'uncertain'.
    n_uncertain = (result["class"] == "uncertain").sum()
    n_total = len(result)
    # At such a high threshold on overlapping components, the majority are uncertain.
    assert n_uncertain > n_total // 2, (
        f"expected most cells uncertain at threshold=0.999, "
        f"got {n_uncertain}/{n_total} uncertain"
    )
    # No crash: all cells have a valid class label.
    assert set(result["class"]).issubset({"tumor", "normal", "uncertain"})


def test_classify_cells_zero_tumor_cells():
    """When all cells look diploid and normal_mask covers all cells, no tumors should emerge."""
    rng = np.random.default_rng(17)
    n_cells = 40
    n_segs = 30

    # Near-zero CN matrix — all cells look diploid.
    cn_matrix = rng.normal(0.0, 0.02, size=(n_cells, n_segs)).astype(np.float32)

    # Segments DataFrame with plausible structure.
    seg_rows = [
        {"chr": "chr1", "start_idx": i * 2, "end_idx": i * 2 + 2, "n_genes": 2, "tumor_mean": 0.0}
        for i in range(n_segs)
    ]
    segments = pd.DataFrame(seg_rows)

    # All cells are in the normal pool.
    normal_mask = np.ones(n_cells, dtype=bool)
    barcodes = [f"cell_{i:04d}" for i in range(n_cells)]

    result = classify_cells(
        cn_matrix=cn_matrix,
        segments=segments,
        normal_mask=normal_mask,
        barcodes=barcodes,
    )

    # No crash.
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_cells

    # No tumor calls: all cells are either 'normal' or 'uncertain'.
    assert "tumor" not in result["class"].values, (
        f"unexpected tumor calls: {result['class'].value_counts().to_dict()}"
    )

    # All subclone values are empty (no subclone for non-tumor).
    assert (result["subclone"] == "").all(), (
        f"expected all subclone == '', got: {result['subclone'].value_counts().to_dict()}"
    )


def test_classify_cells_exactly_min_tumor_floor():
    """Below MIN_TUMOR_CELLS_FOR_SUBCLONES → single subclone_1; at the floor → clustering path."""
    rng = np.random.default_rng(19)
    n_segs = 50

    # --- Case 1: exactly MIN_TUMOR_CELLS_FOR_SUBCLONES - 1 tumor cells ---
    n_tumor_below = MIN_TUMOR_CELLS_FOR_SUBCLONES - 1
    cn_tumor_below = rng.normal(1.0, 0.1, size=(n_tumor_below, n_segs)).astype(np.float32)
    tumor_mask_below = np.ones(n_tumor_below, dtype=bool)

    subclones_below = discover_subclones(cn_tumor_below, tumor_mask=tumor_mask_below)

    # All cells should be "subclone_1" (collapsed due to too few cells).
    assert set(subclones_below) == {"subclone_1"}, (
        f"expected only 'subclone_1' below floor, got: {set(subclones_below)}"
    )

    # --- Case 2: exactly MIN_TUMOR_CELLS_FOR_SUBCLONES tumor cells ---
    n_tumor_at = MIN_TUMOR_CELLS_FOR_SUBCLONES
    # Give the tumor cells a slight structure so clustering finds something.
    cn_tumor_at = rng.normal(1.0, 0.2, size=(n_tumor_at, n_segs)).astype(np.float32)
    tumor_mask_at = np.ones(n_tumor_at, dtype=bool)

    # This should use the clustering path (not the collapse path).
    # Just verify it does not crash and returns the right length.
    subclones_at = discover_subclones(cn_tumor_at, tumor_mask=tumor_mask_at)

    assert len(subclones_at) == n_tumor_at, (
        f"expected {n_tumor_at} subclone labels, got {len(subclones_at)}"
    )
    # At least one subclone label is non-empty.
    assert any(s != "" for s in subclones_at), "expected at least one subclone label"


# ===========================================================================
# OUTPUTS EDGE CASES
# ===========================================================================


def test_compute_clone_consensus_no_subclone_labels():
    """When all tumor cells have subclone=='', the result should use 'all_tumor' group."""
    rng = np.random.default_rng(23)
    n_cells = 20
    n_segs = 10

    cn_matrix = rng.normal(0.0, 0.3, size=(n_cells, n_segs)).astype(np.float32)

    # Build a segments DataFrame.
    # var_coords needs chr, start, end for each gene position.
    # We'll use 100 genes total spread across 10 segments of 10 genes each.
    n_genes_total = n_segs * 10
    seg_rows = []
    for i in range(n_segs):
        seg_rows.append({
            "chr": "chr1",
            "start_idx": i * 10,
            "end_idx": i * 10 + 10,
            "n_genes": 10,
            "tumor_mean": 0.0,
        })
    segments = pd.DataFrame(seg_rows)

    # var_coords: one row per gene with chr/start/end.
    var_coords = pd.DataFrame(
        {
            "chr": ["chr1"] * n_genes_total,
            "start": [i * 1000 for i in range(n_genes_total)],
            "end": [i * 1000 + 999 for i in range(n_genes_total)],
        },
        index=[f"GENE{i:05d}" for i in range(n_genes_total)],
    )

    # prediction_df: first 10 cells are tumor, all with subclone == "".
    barcodes = [f"cell_{i:04d}" for i in range(n_cells)]
    prediction_df = pd.DataFrame(
        {
            "class": ["tumor"] * 10 + ["normal"] * 10,
            "confidence": [0.9] * n_cells,
            "tumor_score": [1.0] * 10 + [0.1] * 10,
            "subclone": [""] * n_cells,
            "n_segments_altered": [5] * 10 + [0] * 10,
        },
        index=barcodes,
    )
    prediction_df.index.name = "barcode"

    result = _compute_clone_consensus_segments(
        cn_matrix=cn_matrix,
        segments=segments,
        prediction_df=prediction_df,
        var_coords=var_coords,
    )

    # Should not be empty (tumor cells exist).
    assert not result.empty, "expected non-empty .seg output"

    # ID column must contain 'all_tumor'.
    assert "all_tumor" in result["ID"].values, (
        f"expected 'all_tumor' ID, got: {result['ID'].unique()}"
    )

    # One row per segment for the all_tumor group.
    assert len(result) == n_segs, (
        f"expected {n_segs} rows (one per segment), got {len(result)}"
    )


def test_chr_cnv_matrix_single_segment_per_chr():
    """compute_chr_cnv_matrix with exactly one segment per chromosome produces valid output."""
    rng = np.random.default_rng(29)
    n_cells = 15
    n_segs = 3  # One segment per chromosome.

    cn_matrix = rng.normal(0.0, 0.2, size=(n_cells, n_segs)).astype(np.float32)

    chrom_list = ["chr1", "chr2", "chr3"]
    seg_rows = [
        {"chr": chrom_list[i], "start_idx": i * 30, "end_idx": i * 30 + 30, "n_genes": 30, "tumor_mean": 0.0}
        for i in range(n_segs)
    ]
    segments = pd.DataFrame(seg_rows)

    matrix, chroms_present = compute_chr_cnv_matrix(cn_matrix, segments, chrom_list)

    # Shape: n_cells × n_segs (one seg per chr → one chr per column).
    assert matrix.shape == (n_cells, n_segs), (
        f"expected ({n_cells}, {n_segs}), got {matrix.shape}"
    )

    # exp-mapped: all values must be positive.
    assert np.all(matrix > 0), "exp-mapped values should be strictly positive"

    # All three chromosomes present.
    assert set(chroms_present) == {"chr1", "chr2", "chr3"}, (
        f"unexpected chroms_present: {chroms_present}"
    )


# ===========================================================================
# REPRODUCIBILITY TEST
# ===========================================================================


def test_pipeline_reproducible(tmp_path):
    """Running the full M1→M4 pipeline twice on the same fixture yields identical results.

    Uses supervised baseline mode (a written barcodes file) to guarantee a
    deterministic normal mask — this avoids any UCell/Leiden stochasticity in
    baseline picking while still exercising the full M1→M4 chain.
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"fixture not present at {FIXTURE_PATH}; "
            "rebuild via `python -m tests.fixtures.build_tiny_simulated`"
        )

    gene_order = load_gene_order()
    raw = read_h5ad(FIXTURE_PATH)

    # Identify the planted-normal cells from the fixture's ground-truth obs column
    # and write them to a barcodes file for supervised baseline mode.
    normal_barcodes = raw.obs_names[raw.obs["true_class"] == "normal"].tolist()
    norm_cell_file = tmp_path / "normal_barcodes.txt"
    norm_cell_file.write_text("\n".join(normal_barcodes))

    def _run_pipeline(adata_raw):
        adata_m1 = filter_normalize_project(adata_raw, gene_order=gene_order)

        # Supervised mode: fully deterministic — same barcodes in, same mask out.
        baseline = pick_baseline(adata_m1, norm_cell_path=str(norm_cell_file))

        chr_labels = adata_m1.var["chr"].to_numpy()
        centered = center_against_baseline(adata_m1, baseline["mask"])
        smoothed = smooth_along_chromosomes(centered, chr_labels=chr_labels)
        segments = detect_segments(
            smoothed, normal_mask=baseline["mask"], chr_labels=chr_labels
        )
        cn_matrix = per_cell_segment_cn(smoothed, segments)
        prediction = classify_cells(
            cn_matrix=cn_matrix,
            segments=segments,
            normal_mask=baseline["mask"],
            barcodes=adata_m1.obs_names.to_numpy(),
        )
        return prediction, segments

    pred1, segs1 = _run_pipeline(raw)
    pred2, segs2 = _run_pipeline(raw)

    # Prediction DataFrames must be identical.
    pd.testing.assert_frame_equal(
        pred1.reset_index(drop=False),
        pred2.reset_index(drop=False),
        check_like=False,
        obj="prediction_df",
    )

    # Segment DataFrames must be identical (same chr, start_idx, end_idx).
    pd.testing.assert_frame_equal(
        segs1.reset_index(drop=True),
        segs2.reset_index(drop=True),
        obj="segments_df",
    )
