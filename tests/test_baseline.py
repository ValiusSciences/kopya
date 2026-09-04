"""Unit tests for baseline.py.

Constructs synthetic AnnDatas where the right baseline answer is engineered
(planted immune signatures, planted low-variance subpopulations) so each of
the four cascade modes can be exercised deterministically.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import json

from kopya.baseline import (
    DEFAULT_MIN_NORMAL_CELLS,
    _gmm_fallback,
    _signature_baseline,
    _supervised_baseline,
    _variance_baseline,
    load_signatures,
    pick_baseline,
)


def _synthetic_signature_adata(n_normal=60, n_tumor=40, seed=0, markers=None):
    """Build an AnnData where the first n_normal cells have planted signature signal.

    Cells 0..n_normal-1 get elevated counts on the planted marker set
    (canonical T-cell markers by default); the rest are diffuse low-signal
    noise. This is the minimum structure needed for the signature baseline to
    fire deterministically.

    Args:
        n_normal: Number of planted signature-positive cells (the "normal" pool).
        n_tumor: Number of background tumor-like cells.
        seed: RNG seed.
        markers: Gene symbols to plant; defaults to the canonical T-cell set.

    Returns:
        AnnData with log1p-style float values in .X. var_names include the
        planted markers plus a pool of generic gene IDs so the matrix is
        non-trivial.
    """
    rng = np.random.default_rng(seed)

    # Gene list: real T-cell markers first (they must match the bundled
    # signatures so UCell scoring picks them up), then filler symbols.
    # Filler is sized larger than pyucell's max_rank=1500 default so that
    # suppressed T-cell markers in tumor cells fall outside the rank cap
    # and are excluded from the score — replicating the real-data regime
    # where the signature is genuinely absent.
    planted_markers = markers if markers is not None else [
        "CD3D", "CD3E", "CD3G", "CD2", "TRAC",
        "TRBC1", "TRBC2", "CD8A", "CD8B", "CD4"]
    filler = [f"GENE{i:04d}" for i in range(2000)]
    gene_names = planted_markers + filler

    n_cells = n_normal + n_tumor
    n_genes = len(gene_names)
    n_markers = len(planted_markers)

    # Base diffuse expression on all genes (log1p-style).
    X = rng.uniform(0.0, 1.0, size=(n_cells, n_genes)).astype(np.float64)

    # Normal cells: boost T-cell markers to clearly elevated — their ranks
    # will dominate the per-cell ordering, so UCell scores near 1.
    X[:n_normal, :n_markers] += rng.uniform(
        4.0, 6.0, size=(n_normal, n_markers),
    )

    # Tumor cells: suppress T-cell markers to near zero. This puts those
    # genes near the bottom of the per-cell rank ordering, so UCell's
    # rank-based statistic for T_cell is forced low (not just average).
    # Without this, random per-cell ranking gives middling T-cell scores
    # in tumor cells too and the threshold can't separate them.
    X[n_normal:, :n_markers] = rng.uniform(
        0.0, 0.05, size=(n_tumor, n_markers),
    )

    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))

    # Sparse for input-shape consistency with M1 outputs.
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    return adata


def _synthetic_low_variance_adata(n_normal=60, n_tumor=40, seed=0):
    """Build an AnnData with a clearly identifiable low-variance subpopulation.

    The first n_normal cells get tightly clustered expression around mean 1.0;
    the rest get expression around mean 3.0 with high variance. Different
    means are required so the two groups occupy distinct regions of PCA space
    (variance differences alone don't separate clouds at the same mean).
    UCell will NOT pick the planted normals because no T-cell markers are
    loaded — so this fixture exercises the variance fallback only.

    Args:
        n_normal: Cells in the low-variance pool.
        n_tumor: Cells in the high-variance pool.
        seed: RNG seed.

    Returns:
        AnnData with float values in .X. var_names are generic (no T-cell
        markers) so the signature baseline returns an empty pool.
    """
    rng = np.random.default_rng(seed)
    # 2000 filler genes — keeps the synthetic in the same dimensional regime
    # as the signature fixture so cross-test behavior is comparable.
    gene_names = [f"GENE{i:04d}" for i in range(2000)]
    n_cells = n_normal + n_tumor
    n_genes = len(gene_names)

    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    # Low-variance group: tight Gaussian around 1.0 — homogeneous, small spread.
    X[:n_normal] = rng.normal(loc=1.0, scale=0.05, size=(n_normal, n_genes))
    # High-variance group: shifted mean AND high spread — distinct cloud in
    # PCA space so Leiden can find them, AND higher per-gene variance so the
    # lowest-variance-cluster rule picks the normals.
    X[n_normal:] = rng.normal(loc=3.0, scale=1.5, size=(n_tumor, n_genes))
    # Counts must be non-negative for downstream operations.
    X = np.clip(X, 0.0, None)

    obs = pd.DataFrame(index=[f"cell_{i:04d}" for i in range(n_cells)])
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))

    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    return adata


def test_supervised_mode(tmp_path):
    """Mode 1: a barcode list returns exactly those cells, with diagnostics."""
    adata = _synthetic_signature_adata(n_normal=20, n_tumor=20)

    # Pick 10 known barcodes plus 2 phantom barcodes that aren't in obs to
    # exercise the unmatched-count diagnostic.
    chosen = list(adata.obs_names[:10])
    phantoms = ["cell_NOMATCH_A", "cell_NOMATCH_B"]
    bc_file = tmp_path / "normals.txt"
    bc_file.write_text(
        "# header comment\n"
        + "\n".join(chosen + phantoms)
        + "\n",
    )

    mask, diag = _supervised_baseline(adata, bc_file)

    # Exactly the 10 chosen cells are flagged normal.
    assert mask.sum() == 10
    assert all(bc in chosen for bc in adata.obs_names[mask])
    assert diag["n_supplied"] == 12
    assert diag["n_matched"] == 10
    assert diag["n_unmatched"] == 2


def test_signature_mode_picks_planted_t_cells():
    """Mode 2: UCell + bundled T-cell signature returns the planted cells."""
    adata = _synthetic_signature_adata(n_normal=60, n_tumor=40)

    mask, diag = _signature_baseline(adata)

    # The first 60 cells were engineered to score high on T_cell.
    # We allow a few false positives/negatives — UCell rank statistics are
    # not exact thresholds — but the planted set should dominate the pool.
    n_called = int(mask.sum())
    assert n_called >= 50, f"signature pool too small: {n_called}"
    # At least 90% of called normals are within the planted set.
    planted = set(adata.obs_names[:60])
    called = set(adata.obs_names[mask])
    overlap = called & planted
    assert len(overlap) / max(len(called), 1) >= 0.9
    # T_cell is the dominant winning label.
    assert diag["per_label_counts"]["T_cell"] >= 50


def test_variance_mode_picks_low_variance_cluster():
    """Mode 3: variance fallback picks the engineered low-variance subpopulation."""
    adata = _synthetic_low_variance_adata(n_normal=60, n_tumor=40)

    mask, diag = _variance_baseline(adata)

    # The first 60 cells were planted with tight Gaussian counts.
    n_called = int(mask.sum())
    planted = set(adata.obs_names[:60])
    called = set(adata.obs_names[mask])
    # ≥90% precision of the picked cluster against the planted normal pool.
    overlap = called & planted
    assert len(overlap) / max(len(called), 1) >= 0.9
    # Multiple clusters were found (the partition wasn't degenerate).
    assert diag["n_clusters"] >= 2


def test_gmm_fallback_returns_a_pool():
    """Mode 4: the GMM fallback always returns a non-empty mask."""
    adata = _synthetic_low_variance_adata(n_normal=60, n_tumor=40)

    mask, diag = _gmm_fallback(adata)

    # Both GMM components should land on at least one cell.
    assert 0 < int(mask.sum()) < adata.n_obs
    # Two component means recorded.
    assert len(diag["component_means"]) == 2
    # The low component is identified.
    assert diag["low_component"] in (0, 1)


def test_pick_baseline_uses_supervised_when_supplied(tmp_path):
    """pick_baseline short-circuits to supervised mode when barcode file given."""
    adata = _synthetic_signature_adata(n_normal=60, n_tumor=40)
    chosen = list(adata.obs_names[:5])
    bc_file = tmp_path / "normals.txt"
    bc_file.write_text("\n".join(chosen) + "\n")

    result = pick_baseline(adata, norm_cell_path=bc_file)

    # Method is supervised; the small pool size does NOT trigger a cascade.
    assert result["method"] == "supervised"
    assert result["n_normal"] == 5


def test_pick_baseline_uses_signature_for_immune_rich_sample():
    """pick_baseline picks signature mode when the planted pool is large enough."""
    adata = _synthetic_signature_adata(n_normal=80, n_tumor=20)

    result = pick_baseline(adata)

    assert result["method"] == "signature"
    assert result["n_normal"] >= DEFAULT_MIN_NORMAL_CELLS


def test_pick_baseline_falls_back_to_variance_when_no_signal():
    """pick_baseline cascades to variance when no immune signature is present."""
    # Low-variance fixture has no T-cell markers in var, so UCell finds nothing.
    adata = _synthetic_low_variance_adata(n_normal=60, n_tumor=40)

    result = pick_baseline(adata)

    # Either variance or gmm_fallback is acceptable depending on Leiden's
    # behavior on the synthetic data; both are valid fallbacks for this fixture.
    assert result["method"] in ("variance", "gmm_fallback")
    assert result["n_normal"] > 0


# ---------------------------------------------------------------------------
# Configurable signature library + non-malignant allow-list
# ---------------------------------------------------------------------------

def test_load_signatures_bundled():
    """The bundled library loads with the expected shape and allow-list."""
    signatures, labels = load_signatures()

    # A few canonical signatures must be present, each a non-empty gene list.
    for name in ("T_cell", "B_cell", "Myeloid", "Plasma_cell"):
        assert name in signatures
        assert len(signatures[name]) > 0
    # Plasma_cell is defined as a signature but deliberately NOT in the
    # non-malignant allow-list (it can be the malignancy in myeloma).
    assert "Plasma_cell" in signatures
    assert "Plasma_cell" not in labels
    # Fibroblast is the same shape of exclusion: the signature stays defined so
    # mesenchymal/ECM-like tumor cells still resolve to it and are therefore
    # kept OUT of the seed, but it is not itself treated as normal.
    assert "Fibroblast" in signatures
    assert "Fibroblast" not in labels
    # The other stromal labels are unaffected by that exclusion.
    for name in ("T_cell", "B_cell", "NK_cell", "Myeloid", "Neutrophil",
                 "Endothelial", "Pericyte", "Osteoblast", "Smooth_muscle"):
        assert name in labels
    # Every allow-listed label resolves to a defined signature.
    assert set(labels).issubset(set(signatures))


def test_load_signatures_custom_file(tmp_path):
    """A custom JSON overrides both the gene sets and the allow-list."""
    path = tmp_path / "sig.json"
    path.write_text(json.dumps({
        "_meta": {"non_malignant_labels": ["Stroma"]},
        "Stroma": ["COL1A1", "DCN", "LUM"],
        "Tumorish": ["MKI67", "TOP2A"],
    }))

    signatures, labels = load_signatures(path)

    assert set(signatures) == {"Stroma", "Tumorish"}
    assert signatures["Stroma"] == ["COL1A1", "DCN", "LUM"]
    # Only the label named in _meta is treated as normal.
    assert labels == ["Stroma"]


def test_load_signatures_defaults_labels_to_all_when_meta_absent(tmp_path):
    """Without _meta, every signature is treated as non-malignant."""
    path = tmp_path / "sig.json"
    path.write_text(json.dumps({"A": ["G1"], "B": ["G2"]}))

    signatures, labels = load_signatures(path)

    assert set(signatures) == {"A", "B"}
    assert set(labels) == {"A", "B"}


def test_load_signatures_rejects_empty(tmp_path):
    """A file with no signature entries is an error, not a silent empty pool."""
    path = tmp_path / "sig.json"
    path.write_text(json.dumps({"_meta": {"non_malignant_labels": []}}))

    with pytest.raises(ValueError, match="no signatures"):
        load_signatures(path)


def test_load_signatures_rejects_unknown_label(tmp_path):
    """An allow-list label that isn't a defined signature is a typo guard."""
    path = tmp_path / "sig.json"
    path.write_text(json.dumps({
        "_meta": {"non_malignant_labels": ["Tcell"]},  # typo: should be T_cell
        "T_cell": ["CD3D", "CD3E"],
    }))

    with pytest.raises(ValueError, match="not.*defined signatures"):
        load_signatures(path)


def test_signature_baseline_respects_non_malignant_labels():
    """Overriding the allow-list changes which winners count as normal.

    The fixture plants T-cell signal, so cells win the T_cell signature. With
    T_cell allow-listed they form the pool; excluding it (allowing only B_cell)
    collapses the pool even though the same cells still win T_cell.
    """
    adata = _synthetic_signature_adata(n_normal=60, n_tumor=40)

    keep, diag_keep = _signature_baseline(adata, non_malignant_labels=["T_cell"])
    drop, diag_drop = _signature_baseline(adata, non_malignant_labels=["B_cell"])

    assert int(keep.sum()) >= 50
    assert int(drop.sum()) == 0
    # The effective allow-list is recorded in diagnostics for provenance.
    assert diag_keep["non_malignant_labels"] == ["T_cell"]
    assert diag_drop["non_malignant_labels"] == ["B_cell"]


def test_fibroblast_winners_are_not_seeded_by_default():
    """Cells that win the Fibroblast signature stay OUT of the default seed.

    Mesenchymal / ECM-like tumor cells express the fibroblast program, so under
    the old allow-list they entered the diploid seed and the tumor partly seeded
    its own baseline. The signature is deliberately still defined: these cells
    must keep resolving TO Fibroblast (rather than falling through to a stromal
    label that IS allow-listed, e.g. Pericyte or Smooth_muscle), which is what
    keeps them out. Re-adding the label restores the old behavior per-sample.
    """
    fibroblast_markers = ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM",
                          "PDGFRA", "FAP", "POSTN", "FN1", "THY1"]
    adata = _synthetic_signature_adata(
        n_normal=60, n_tumor=40, markers=fibroblast_markers)

    mask, diag = _signature_baseline(adata)

    # They do clear the threshold, and Fibroblast is what they resolve to.
    assert diag["per_label_counts"]["Fibroblast"] >= 50
    # ... but none of them are admitted to the diploid seed.
    assert int(mask.sum()) == 0
    assert "Fibroblast" not in diag["non_malignant_labels"]

    # The --non-malignant-labels escape hatch still recovers them for a sample
    # whose stroma is genuinely fibroblast-rich.
    readd, _ = _signature_baseline(adata, non_malignant_labels=["Fibroblast"])
    assert int(readd.sum()) >= 50


def test_signature_baseline_accepts_custom_signature_library():
    """A custom signatures dict + matching label drives the pool."""
    adata = _synthetic_signature_adata(n_normal=60, n_tumor=40)
    # Map the planted T-cell markers to a custom label name.
    custom = {"MyNormal": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"]}

    mask, diag = _signature_baseline(
        adata, signatures=custom, non_malignant_labels=["MyNormal"],
    )

    assert int(mask.sum()) >= 50
    assert diag["non_malignant_labels"] == ["MyNormal"]
    assert "MyNormal" in diag["per_label_counts"]


def test_pick_baseline_threads_label_override_through_signature_mode():
    """pick_baseline forwards non_malignant_labels into the signature mode."""
    adata = _synthetic_signature_adata(n_normal=60, n_tumor=40)

    result = pick_baseline(adata, non_malignant_labels=["T_cell"])

    assert result["method"] == "signature"
    assert result["n_normal"] >= 50
    assert result["diagnostics"]["non_malignant_labels"] == ["T_cell"]


def test_signature_baseline_does_not_pollute_caller_obs():
    """_signature_baseline scores on a shared-.X view; the caller's obs stays clean.

    Regression for the memory-lean rewrite: it must not leave `{sig}_UCell`
    columns on the caller's AnnData (previously it copied the whole AnnData).
    """
    adata = _synthetic_signature_adata()
    cols_before = list(adata.obs.columns)
    _signature_baseline(adata)
    assert list(adata.obs.columns) == cols_before
    assert not any(c.endswith("_UCell") for c in adata.obs.columns)


def test_ucell_chunk_size_is_score_invariant():
    """chunk_size is a pure performance knob — scores must be identical.

    The M2 speedup tunes compute_ucell_scores' chunk_size; this pins the
    assumption that doing so never changes the scores (guards a future pyucell
    bump from silently breaking it).
    """
    import numpy as np
    from pyucell import compute_ucell_scores

    from kopya.baseline import _load_bundled_signatures

    adata = _synthetic_signature_adata()
    sigs, _ = _load_bundled_signatures()
    cols = [f"{k}_UCell" for k in sigs]
    a = adata.copy(); compute_ucell_scores(a, signatures=sigs, chunk_size=7)
    b = adata.copy(); compute_ucell_scores(b, signatures=sigs, chunk_size=1000)
    np.testing.assert_array_equal(a.obs[cols].to_numpy(), b.obs[cols].to_numpy())
