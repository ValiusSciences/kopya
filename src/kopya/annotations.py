"""Gene-annotation loaders and filter sets for kopya.

Provides:
- load_gene_order(): parse a (gene_symbol, chr, start, end) TSV into a DataFrame
- load_cycle_genes(): the bundled Tirosh cell-cycle gene list
- gene_filter_mask(): boolean mask for genes to drop (chrY, MT-, HLA-, cycle)
- CANONICAL_CHROM_ORDER: canonical chr1..chr22, chrX order used for sorting

The bundled tables live under kopya/data/. Users can override the
gene-order file via the --gene-order CLI flag. The default reference is
GENCODE v49 hg38 (2025-09); rebuild via kopya.data.build_gencode_table.
"""

from importlib.resources import files
from pathlib import Path

from pandas import read_csv


# Canonical human chromosome order used for sorting genes and chromosomes
# wherever genomic order matters. The sex chromosomes are filtered out before
# this point (see DEFAULT_DROP_CHROMS) so they do not appear here — CNV calling
# is autosome-only.
CANONICAL_CHROM_ORDER = [f"chr{n}" for n in range(1, 23)]

# Chromosomes never used for CNV calling. chrY is dropped to match CopyKAT;
# mitochondrial contigs because they are not nuclear copy number. chrX is
# dropped because expression is a poor proxy for X copy number: X-inactivation
# and dosage compensation flatten the expression response to a gain or loss, so
# the arm never tracks the true CNV. Empirically chrX was the single worst
# per-chromosome outlier against matched bulk exome (e.g. a real chrX gain reads
# as neutral in expression); excluding it lifts the bulk-concordance Pearson
# materially and never hurts it. inferCNV/CopyKAT users routinely drop the sex
# chromosomes for the same reason.
DEFAULT_DROP_CHROMS = frozenset({"chrX", "chrY", "chrM", "chrMT"})

# Gene-symbol prefix filters. MT-* are mitochondrial transcripts whose counts
# scale with mitochondrial content, not nuclear CN. HLA-* are filtered because
# their hyper-variable expression in immune cells creates spurious CN signal.
DEFAULT_DROP_PREFIXES = ("MT-", "HLA-")

# Immunoglobulin locus genes. Their expression in B / plasma cells is driven by
# clonal VDJ selection and antibody-secretion load, not by copy number, and is
# extreme enough (immunoglobulins are the dominant transcript in plasma cells)
# to manufacture a focal pseudo-amplification at each locus — exactly the
# spurious chr14 spike seen on plasma-rich samples. CopyKAT and inferCNV both
# exclude these for the same reason.
#
# Each prefix is paired with its locus chromosome (heavy @ 14q32, kappa @ 2p11,
# lambda @ 22q11) so a gene is dropped only when its symbol prefix AND its
# chromosome both match. The chromosome guard is essential: the prefixes alone
# also match unrelated genes — IGHMBP2 (a helicase on chr11), the neural IGLON*
# adhesion family on chr19, and ~50 scattered V-segment orphons on chr8/9/15/16
# /etc. — none of which belong to the antibody loci and all of which are kept.
DEFAULT_DROP_IG_LOCI = {
    "IGH": "chr14",
    "IGK": "chr2",
    "IGL": "chr22",
}


def _data_path(name):
    """Return an absolute filesystem path to a file under kopya/data/.

    Wraps importlib.resources so the package data is locatable whether the
    install is editable (live source tree) or a built wheel.

    Args:
        name: File name within the data/ directory (e.g. "cycle_genes.txt").

    Returns:
        A pathlib.Path pointing at the bundled file.
    """
    # files() returns a Traversable; joinpath + str + Path normalizes to a real
    # filesystem path that read_csv and open() can both consume directly.
    resource = files("kopya.data").joinpath(name)
    path = Path(str(resource))
    return path


def load_gene_order(path=None):
    """Load the (gene_symbol, chr, start, end) gene-order table.

    Defaults to the bundled GENCODE v49 hg38 table; override `path` to use
    a custom file (e.g. a different GENCODE release or mm10). The returned
    DataFrame is indexed by gene_symbol and sorted by canonical genomic order.

    Args:
        path: Optional override path; default = bundled gencode_v49_hg38.tsv.

    Returns:
        A DataFrame with index=gene_symbol and columns [chr, start, end].
        Sorted by (chr, start) using CANONICAL_CHROM_ORDER for chr ordering.
    """
    # Resolve the table path: explicit override > bundled default.
    if path is None:
        table_path = _data_path("gencode_v49_hg38.tsv")
    else:
        table_path = Path(path)

    # Parse the TSV; gene_symbol becomes the index for fast joins against var.
    df = read_csv(table_path, sep="\t", index_col="gene_symbol")

    # Coerce chromosome to the canonical ordered Categorical so a single sort
    # call gives genomic order rather than lexicographic ("chr10" < "chr2").
    # The dropped chromosomes (chrX, chrY, chrM, chrMT) are still listed as
    # categories so their rows sort validly here; gene_filter_mask() removes
    # them afterwards.
    df["chr"] = df["chr"].astype("category").cat.set_categories(
        CANONICAL_CHROM_ORDER + ["chrX", "chrY", "chrM", "chrMT"],
        ordered=True,
    )

    # Sort by (chr, start). DEFAULT_DROP_CHROMS rows still appear here; the
    # gene_filter_mask() step is responsible for removing them.
    df_sorted = df.sort_values(["chr", "start"])
    return df_sorted


def load_cycle_genes(path=None):
    """Load the bundled Tirosh cell-cycle gene symbol list.

    Args:
        path: Optional override path to a one-symbol-per-line text file.
            Lines starting with '#' and blank lines are ignored.

    Returns:
        A set of HGNC-style gene symbols (str).
    """
    # Resolve the file location, defaulting to the bundled list.
    if path is None:
        cycle_path = _data_path("cycle_genes.txt")
    else:
        cycle_path = Path(path)

    # Parse line-by-line so comments and blank lines are tolerated cleanly.
    symbols = set()
    with cycle_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            symbols.add(line)

    return symbols


def gene_filter_mask(
    var_names,
    gene_order,
    cycle_genes=None,
    drop_prefixes=DEFAULT_DROP_PREFIXES,
    drop_chroms=DEFAULT_DROP_CHROMS,
    drop_ig_loci=DEFAULT_DROP_IG_LOCI,
):
    """Compute a boolean mask of genes to KEEP for CNV inference.

    A gene is kept iff:
        - it appears in the gene_order table (so we know where it sits)
        - its chromosome is placeable (present, non-null) and not in drop_chroms
          (default: chrX, chrY, chrM, chrMT)
        - its symbol does not begin with any drop_prefix (default: MT-, HLA-)
        - it is not an immunoglobulin locus gene — i.e. NOT (symbol starts with an
          IG-locus prefix AND sits on that locus's chromosome: IGH@chr14,
          IGK@chr2, IGL@chr22). These track clonal antibody expression, not copy
          number. The chromosome guard keeps prefix look-alikes off the loci
          (e.g. IGHMBP2 on chr11, IGLON* on chr19, V-segment orphons elsewhere).
        - its symbol is not in the cycle_genes set (Tirosh list by default)

    Args:
        var_names: Iterable of gene symbols, in the order they appear in adata.var.
        gene_order: DataFrame from load_gene_order(), indexed by gene symbol.
        cycle_genes: Optional set of symbols to drop; default = bundled Tirosh list.
        drop_prefixes: Tuple of symbol prefixes to drop.
        drop_chroms: Set of chromosome labels to drop entirely.
        drop_ig_loci: Mapping of immunoglobulin-locus symbol prefix -> the
            chromosome that locus lives on. A gene is dropped only when both
            match, so genes that merely share a prefix but sit elsewhere are kept.

    Returns:
        A numpy bool array, len == len(var_names), True where the gene is kept.
    """
    # Late import keeps numpy out of the module-load critical path.
    import numpy as np

    # Default to the bundled cycle list; reading the file every call is cheap
    # (~100 lines) but a caller that pre-loaded it can pass it through to avoid.
    if cycle_genes is None:
        cycle_genes = load_cycle_genes()

    # Build a chr lookup keyed by gene symbol from gene_order. Using .to_dict
    # rather than .reindex() so missing genes resolve to None and the prefix /
    # cycle filters still get applied for diagnostics.
    chrom_by_symbol = gene_order["chr"].to_dict()

    keep = np.zeros(len(var_names), dtype=bool)
    for i, symbol in enumerate(var_names):
        # Drop genes we cannot place. Two ways a chromosome comes back unusable:
        #   - a mapping miss (symbol absent from gene_order) -> None
        #   - a value that resolved to NaN, e.g. a custom gene-order table whose
        #     ordered Categorical omits this chromosome; .to_dict() surfaces the
        #     unmatched label as float NaN.
        # NaN is neither None nor a member of drop_chroms, so without this guard
        # such a gene (a chrX gene under a chrX-less Categorical, say) would slip
        # through every filter and be retained with a missing chromosome label.
        chrom = chrom_by_symbol.get(symbol)
        if chrom is None or chrom != chrom:  # `chrom != chrom` is True only for NaN
            continue
        # Skip excluded chromosomes (chrX, chrY, chrM/chrMT).
        if chrom in drop_chroms:
            continue
        # Skip mitochondrial / HLA prefixes — high variance unrelated to CN.
        if symbol.startswith(drop_prefixes):
            continue
        # Skip immunoglobulin-locus genes: symbol starts with an IG-locus prefix
        # AND sits on that locus's chromosome. The chromosome guard preserves
        # prefix look-alikes off the loci (IGHMBP2 @ chr11, IGLON* @ chr19, etc.).
        ig_chrom = next(
            (c for prefix, c in drop_ig_loci.items() if symbol.startswith(prefix)),
            None,
        )
        if ig_chrom is not None and str(chrom) == ig_chrom:
            continue
        # Skip cell-cycle genes whose expression tracks cycling, not CN.
        if symbol in cycle_genes:
            continue
        keep[i] = True

    return keep
