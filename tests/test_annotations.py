"""Unit tests for annotations.py.

Covers gene-order loading (bundled GENCODE v49 + override), cycle-gene
parsing, and the gene_filter_mask combinatorics. The bundled GENCODE file
is exercised here once; everywhere else uses the tiny synthetic fixture
so tests stay fast.
"""

import pandas as pd
import pytest

from kopya.annotations import (
    CANONICAL_CHROM_ORDER,
    gene_filter_mask,
    load_cycle_genes,
    load_gene_order,
)


def test_load_gene_order_bundled():
    """The bundled GENCODE v49 hg38 table loads with the expected shape."""
    df = load_gene_order()

    # v49 has 77,041 gene rows by construction (see build_gencode_table.py).
    assert len(df) == 77041
    # Required columns; gene_symbol is the index.
    assert list(df.columns) == ["chr", "start", "end"]
    assert df.index.name == "gene_symbol"
    # chr is an ordered Categorical in canonical genomic order.
    assert isinstance(df["chr"].dtype, pd.CategoricalDtype)
    assert df["chr"].cat.ordered


def test_load_gene_order_override(tmp_path):
    """An explicit override path bypasses the bundled table."""
    override = tmp_path / "tiny.tsv"
    override.write_text(
        "gene_symbol\tchr\tstart\tend\n"
        "FOO\tchr1\t100\t200\n"
        "BAR\tchr2\t300\t400\n"
    )

    df = load_gene_order(path=override)

    # Round-trip the two rows; chr remains a Categorical post-load.
    assert len(df) == 2
    assert set(df.index) == {"FOO", "BAR"}


def test_load_gene_order_sorted_canonical():
    """Rows are sorted by canonical chromosome order, not lexicographic."""
    df = load_gene_order()

    # Build the observed chromosome sequence; collapse adjacent duplicates so
    # we are checking the ORDER of unique chromosomes, not their frequencies.
    seen = []
    for chrom in df["chr"]:
        if not seen or seen[-1] != chrom:
            seen.append(chrom)

    # Every chr in seen should appear in canonical order. We allow chrY to
    # appear at the end (it is filtered by gene_filter_mask, not by the loader).
    canonical = CANONICAL_CHROM_ORDER + ["chrY"]
    seen_idxs = [canonical.index(c) for c in seen if c in canonical]
    assert seen_idxs == sorted(seen_idxs)


def test_load_cycle_genes_bundled():
    """The bundled Tirosh list parses to a non-trivial set of HGNC symbols."""
    symbols = load_cycle_genes()

    # Spot-check well-known members from each phase.
    assert "MCM5" in symbols          # S phase
    assert "MKI67" in symbols         # G2/M
    # Aliases for renamed symbols ship alongside the originals.
    assert "MLF1IP" in symbols and "CENPU" in symbols
    # No comment lines or blanks leaked through.
    assert all(not s.startswith("#") and s.strip() == s for s in symbols)


def test_load_cycle_genes_override(tmp_path):
    """An override path replaces the bundled list, including comment handling."""
    override = tmp_path / "cycle.txt"
    override.write_text(
        "# header comment\n"
        "\n"
        "FOO\n"
        "BAR\n"
        "# inline comment line\n"
        "BAZ\n"
    )

    symbols = load_cycle_genes(path=override)

    assert symbols == {"FOO", "BAR", "BAZ"}


def test_gene_filter_mask_basic(tiny_gene_order):
    """The mask drops chrY, MT-*, cycle genes, and unmappable symbols."""
    var_names = [
        "A_CHR1_EARLY",   # kept
        "TY_GENE",        # dropped: chrY
        "MT-DROP",        # dropped: MT- prefix
        "MKI67",          # dropped: cycle gene
        "UNKNOWN_SYMBOL", # dropped: not in gene_order
        "G_CHRX_EARLY",   # dropped: chrX
    ]

    mask = gene_filter_mask(
        var_names=var_names,
        gene_order=tiny_gene_order,
    )

    # Only A_CHR1_EARLY survives; the sex chromosomes (chrX, chrY) are dropped.
    assert mask.tolist() == [True, False, False, False, False, False]


def test_gene_filter_mask_hla_prefix(tiny_gene_order):
    """HLA-prefixed symbols are dropped even when present in the gene-order table."""
    # Add an HLA-A row to the gene_order fixture so we exercise the prefix
    # filter rather than the unmappable-symbol path.
    extra = pd.DataFrame(
        [["HLA-A", "chr6", 100, 200]],
        columns=["gene_symbol", "chr", "start", "end"],
    ).set_index("gene_symbol")
    gene_order = pd.concat([tiny_gene_order, extra])
    # Re-coerce chr to a Categorical that covers every value present in the
    # combined frame; CANONICAL_CHROM_ORDER already includes chr6 / chr10.
    gene_order["chr"] = gene_order["chr"].astype(
        pd.CategoricalDtype(
            categories=CANONICAL_CHROM_ORDER + ["chrY", "chrM"],
            ordered=True,
        )
    )

    mask = gene_filter_mask(
        var_names=["HLA-A", "A_CHR1_EARLY"],
        gene_order=gene_order,
    )

    # HLA-A dropped (prefix); A_CHR1_EARLY kept.
    assert mask.tolist() == [False, True]


def test_gene_filter_mask_drops_nan_chromosome(tiny_gene_order):
    """A gene whose chromosome resolves to NaN is dropped, not silently kept.

    A custom gene-order table can carry an ordered Categorical that omits some
    chromosome (e.g. CANONICAL_CHROM_ORDER, which excludes chrX). Coercing to it
    turns the unmatched label into float NaN. NaN is neither None nor a member of
    drop_chroms, so the mask must treat it as unplaceable and drop the gene —
    otherwise chrX (or any off-Categorical) genes leak through with no position.
    """
    gene_order = tiny_gene_order.copy()
    gene_order["chr"] = gene_order["chr"].astype(
        pd.CategoricalDtype(categories=CANONICAL_CHROM_ORDER, ordered=True)
    )
    # Under a chrX-less Categorical, G_CHRX_EARLY's chromosome is NaN.
    assert pd.isna(gene_order.loc["G_CHRX_EARLY", "chr"])

    mask = gene_filter_mask(
        var_names=["A_CHR1_EARLY", "G_CHRX_EARLY"],
        gene_order=gene_order,
    )
    # A_CHR1_EARLY kept; the NaN-chromosome gene dropped.
    assert mask.tolist() == [True, False]


def test_gene_filter_mask_custom_cycle_list(tiny_gene_order):
    """Caller-supplied cycle_genes replaces the bundled default."""
    var_names = ["A_CHR1_EARLY", "B_CHR1_MID"]
    # Override Tirosh with a list that drops A_CHR1_EARLY explicitly.
    mask = gene_filter_mask(
        var_names=var_names,
        gene_order=tiny_gene_order,
        cycle_genes={"A_CHR1_EARLY"},
    )

    assert mask.tolist() == [False, True]


def test_gene_filter_mask_drops_immunoglobulin_genes(tiny_gene_order):
    """IG-locus genes are dropped only on their locus chromosome.

    Immunoglobulin expression tracks clonal antibody load, not copy number, and
    manufactures a focal pseudo-amplification at the IGH@chr14 / IGK@chr2 /
    IGL@chr22 loci in B / plasma cells. The filter is chromosome-guarded so that
    genes which merely share an IG prefix but sit elsewhere are kept:
      - IGHMBP2 (a helicase on chr11) — not an antibody gene,
      - IGLON5 (neural adhesion on chr19),
      - and V-segment orphons scattered off-locus.
    """
    # IG-locus genes at their real loci, plus two prefix look-alikes that sit
    # on other chromosomes and must survive (IGHMBP2 chr11, IGLON5 chr19).
    extra = pd.DataFrame(
        [
            ["IGHG1", "chr14", 100, 200],    # heavy-chain constant @ locus — drop
            ["IGKC", "chr2", 100, 200],      # kappa constant @ locus — drop
            ["IGLC1", "chr22", 100, 200],    # lambda constant @ locus — drop
            ["IGHMBP2", "chr11", 100, 200],  # helicase, not Ig — KEEP
            ["IGLON5", "chr19", 100, 200],   # neural adhesion — KEEP
        ],
        columns=["gene_symbol", "chr", "start", "end"],
    ).set_index("gene_symbol")
    gene_order = pd.concat([tiny_gene_order, extra])
    gene_order["chr"] = gene_order["chr"].astype(
        pd.CategoricalDtype(
            categories=CANONICAL_CHROM_ORDER + ["chrY", "chrM"],
            ordered=True,
        )
    )

    mask = gene_filter_mask(
        var_names=["IGHG1", "IGKC", "IGLC1", "IGHMBP2", "IGLON5", "A_CHR1_EARLY"],
        gene_order=gene_order,
    )

    # The three on-locus IG genes are dropped; the two off-locus look-alikes
    # and the ordinary gene are kept.
    assert mask.tolist() == [False, False, False, True, True, True]
