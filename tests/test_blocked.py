"""Blocked M3 must equal the direct center→smooth→segment→CN path.

The whole point of blocking is a memory win with no change in results, so these
tests pin numerical equivalence (segment boundaries identical; cn_matrix and
tumor_mean within float32 noise) and the exactness of the sparse per-gene
normal-median baseline.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from kopya.blocked import (
    auto_block_size,
    blocked_segment_and_cn,
    per_gene_normal_median,
)
from kopya.segment import (
    detect_segments,
    per_cell_segment_cn,
    per_segment_normal_baseline,
)
from kopya.smooth import center_against_baseline, smooth_along_chromosomes


@pytest.fixture(scope="module")
def tiny_adata_m1():
    """(adata_m1, normal_mask, chr_labels) from the committed tiny fixture."""
    from pathlib import Path

    import scanpy as sc

    from kopya.annotations import load_gene_order
    from kopya.baseline import pick_baseline
    from kopya.normalize import filter_normalize_project

    fx = Path(__file__).resolve().parent / "fixtures" / "tiny_simulated.h5ad"
    if not fx.exists():
        pytest.skip(f"tiny fixture missing at {fx}")
    adata = filter_normalize_project(sc.read_h5ad(fx), gene_order=load_gene_order())
    mask = pick_baseline(adata)["mask"]
    return adata, mask, adata.var["chr"].to_numpy()


def test_per_gene_normal_median_matches_dense():
    """Sparse per-gene normal median == np.median over the dense normal rows."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 40
    dense = rng.random((n_cells, n_genes)).astype(np.float32)
    dense[dense < 0.6] = 0.0  # ~60% zeros so many genes have a zero median
    X = csr_matrix(dense)
    mask = np.zeros(n_cells, dtype=bool)
    mask[rng.choice(n_cells, 137, replace=False)] = True  # odd count → single middle

    got = per_gene_normal_median(X, mask)
    want = np.median(dense[mask], axis=0)
    np.testing.assert_allclose(got, want, atol=1e-6)

    # Even count exercises the two-middle average path.
    mask2 = np.zeros(n_cells, dtype=bool)
    mask2[rng.choice(n_cells, 200, replace=False)] = True
    np.testing.assert_allclose(
        per_gene_normal_median(X, mask2), np.median(dense[mask2], axis=0), atol=1e-6
    )


def test_auto_block_size_bounds_dense_footprint():
    # ~1.5 GB / (genes*4 bytes) cells, capped at n_cells.
    assert auto_block_size(1_000_000, 9300) == pytest.approx(1_500_000_000 // (9300 * 4), rel=0)
    assert auto_block_size(500, 9300) == 500  # never exceeds n_cells


def _direct_m3(adata, mask, chr_labels):
    centered = center_against_baseline(adata, mask)
    smoothed = smooth_along_chromosomes(centered, chr_labels=chr_labels)
    seg = detect_segments(smoothed, normal_mask=mask, chr_labels=chr_labels)
    cn = per_cell_segment_cn(smoothed, seg)
    segbase = per_segment_normal_baseline(cn, mask)
    return seg, cn, segbase


@pytest.mark.parametrize("block_size", [137, 500, 10_000])
def test_blocked_equals_direct_on_fixture(tiny_adata_m1, block_size):
    """Multi-block and single-block runs both reproduce the direct path."""
    adata, mask, chr_labels = tiny_adata_m1
    seg_d, cn_d, base_d = _direct_m3(adata, mask, chr_labels)
    seg_b, cn_b, base_b = blocked_segment_and_cn(adata, mask, chr_labels, block_size=block_size)

    # Segment boundaries are identical (PELT runs on the same pooled signal).
    assert list(seg_b.columns) == list(seg_d.columns)
    np.testing.assert_array_equal(seg_b["start_idx"].to_numpy(), seg_d["start_idx"].to_numpy())
    np.testing.assert_array_equal(seg_b["end_idx"].to_numpy(), seg_d["end_idx"].to_numpy())
    # Values agree to float32 accumulation noise.
    np.testing.assert_allclose(seg_b["tumor_mean"].to_numpy(), seg_d["tumor_mean"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(cn_b, cn_d, atol=1e-5)
    np.testing.assert_allclose(base_b, base_d, atol=1e-5)


def test_blocked_does_not_mutate_dense_input(tiny_adata_m1):
    """Dense-X input must not be mutated by in-place centering (aliasing regression).

    The dense branch of _densify_center must copy; otherwise pass 1 mutates the
    caller's matrix and pass 2 double-subtracts the baseline.
    """
    import anndata as ad

    adata, mask, chr_labels = tiny_adata_m1
    dense = ad.AnnData(
        X=np.asarray(adata.X.todense(), dtype=np.float32),
        obs=adata.obs.copy(), var=adata.var.copy(),
    )
    before = dense.X.copy()
    _, cn_dense, _ = blocked_segment_and_cn(dense, mask, chr_labels, block_size=137)

    np.testing.assert_array_equal(dense.X, before)  # input untouched
    _, cn_sparse, _ = blocked_segment_and_cn(adata, mask, chr_labels, block_size=137)
    np.testing.assert_allclose(cn_dense, cn_sparse, atol=1e-5)  # dense == sparse path


def test_blocked_pooled_signal_block_size_invariant(tiny_adata_m1):
    """Segment boundaries must not depend on --block-size (float64 block reduction)."""
    adata, mask, chr_labels = tiny_adata_m1
    seg_small, _, _ = blocked_segment_and_cn(adata, mask, chr_labels, block_size=53)
    seg_big, _, _ = blocked_segment_and_cn(adata, mask, chr_labels, block_size=100_000)
    np.testing.assert_array_equal(seg_small["start_idx"].to_numpy(), seg_big["start_idx"].to_numpy())
    np.testing.assert_array_equal(seg_small["end_idx"].to_numpy(), seg_big["end_idx"].to_numpy())


def test_per_gene_normal_median_sums_duplicate_coords():
    """Non-canonical CSR with duplicate (i,j) entries must be summed (match dense)."""
    # Row 0 stores gene 0 twice (2 + 3 = 5); a raw CSR built this way is legal
    # but non-canonical — .tocsc() need not merge the duplicates.
    data = np.array([2.0, 3.0, 4.0, 1.0], dtype=np.float32)
    indices = np.array([0, 0, 1, 2])
    indptr = np.array([0, 2, 4])
    X = csr_matrix((data, indices, indptr), shape=(2, 3))
    mask = np.array([True, True])

    got = per_gene_normal_median(X, mask)
    dense = np.array([[5.0, 0.0, 0.0], [0.0, 4.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(got, np.median(dense, axis=0), atol=1e-6)
