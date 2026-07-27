"""Build the bundled GENCODE hg38 gene-order table from the upstream GTF.

Parses the GENCODE comprehensive annotation GTF, keeps one row per `gene`
feature, and writes a 4-column TSV: gene_symbol, chr, start, end. Sorted by
(chr, start) so the table can be consumed without re-sorting downstream.

The output table is small (~77k rows for v49, ~2.6 MB) and is committed
alongside the package data so users do not need to re-download the 89 MB GTF.

Current bundled table: GENCODE v49 (2025-09).

Usage (from the repo root, after activating the env):
    curl -sSL https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz -o /tmp/gencode.v49.gtf.gz
    python -m kopya.data.build_gencode_table \
        --gtf /tmp/gencode.v49.gtf.gz \
        --out src/kopya/data/gencode_v49_hg38.tsv

Reference: https://www.gencodegenes.org/human/
"""

import argparse
import gzip
import re
from pathlib import Path


# Standard 24 chromosome labels; everything else (scaffolds, alt contigs,
# chrM) is dropped because per-cell CNV calling does not use them.
CANONICAL_CHROMS = {f"chr{n}" for n in range(1, 23)} | {"chrX", "chrY"}

# GTF attribute parser: matches `gene_name "FOO"` style fields. Capturing
# group 1 is the value, sans surrounding double quotes.
_ATTR_RE = re.compile(r'(\w+) "([^"]*)"')


def _parse_attrs(attr_str):
    """Parse a GTF attribute column into a dict of {key: value}.

    GTF attributes are semicolon-delimited `key "value"` pairs. This is a
    deliberately small parser — full GFF3/GTF parsers are overkill for the
    handful of fields we use (gene_name, gene_type).

    Args:
        attr_str: The 9th column of a GTF line.

    Returns:
        A dict mapping attribute key to its (unquoted) string value.
    """
    # findall yields list of (key, value) tuples — convert to dict in one pass.
    pairs = _ATTR_RE.findall(attr_str)
    attrs = dict(pairs)
    return attrs


def _iter_gene_rows(gtf_path):
    """Yield (gene_symbol, chr, start, end) tuples for every canonical gene.

    Filters applied:
        - feature column must equal "gene" (skip transcripts/exons/etc.)
        - chr must be in the canonical autosome+X+Y set (skip scaffolds, chrM)
        - gene must have a non-empty gene_name attribute (skip unannotated)

    Args:
        gtf_path: Path to gencode.v27.annotation.gtf.gz.

    Yields:
        Tuples of (gene_symbol: str, chr: str, start: int, end: int).
    """
    # gzip.open in text mode handles the .gz transparently; the GENCODE GTF is
    # always UTF-8 (ASCII in practice) and small enough to stream once.
    with gzip.open(gtf_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            # GTF header lines begin with '##'; skip without parsing.
            if line.startswith("#"):
                continue

            # Split into the 9 GTF columns; defensive against malformed rows.
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            chrom, _source, feature, start_s, end_s, _score, _strand, _frame, attrs_s = fields

            # Only the "gene" feature gives us one row per gene with its full span.
            if feature != "gene":
                continue
            if chrom not in CANONICAL_CHROMS:
                continue

            attrs = _parse_attrs(attrs_s)
            gene_symbol = attrs.get("gene_name", "")
            if not gene_symbol:
                continue

            # GTF coordinates are 1-based, inclusive on both ends; we keep them
            # in that form because downstream CNV tools (CopyKAT, IGV) expect it.
            row = (gene_symbol, chrom, int(start_s), int(end_s))
            yield row


def _chrom_sort_key(chrom):
    """Return a sort key that orders chromosomes by canonical genomic order.

    Numeric autosomes come first (1..22), then X, then Y. Matches the order
    every CNV plot in this codebase already uses.

    Args:
        chrom: A "chrN" string in CANONICAL_CHROMS.

    Returns:
        An integer suitable as a sort key.
    """
    # Strip the 'chr' prefix; map X/Y to numeric values that sort after 22.
    bare = chrom[3:]
    if bare == "X":
        key = 23
    elif bare == "Y":
        key = 24
    else:
        key = int(bare)
    return key


def build(gtf_path, out_path):
    """Build the gene-order TSV from a GENCODE GTF.

    Reads every gene record from the GTF, dedupes (a handful of gene_name
    values appear twice across PAR regions on chrX/chrY — we keep the longest
    span as a deterministic tiebreaker), sorts by (chr, start), and writes a
    4-column TSV.

    Args:
        gtf_path: Path to gencode.v27.annotation.gtf.gz.
        out_path: Path to the output TSV (parent dir must exist).
    """
    # Collect rows into a per-symbol dict to handle the PAR-region duplicates;
    # keep the longest span on a tie since per-gene smoothing wants the widest
    # genomic footprint anyway.
    best_by_symbol = {}
    for symbol, chrom, start, end in _iter_gene_rows(gtf_path):
        span = end - start
        existing = best_by_symbol.get(symbol)
        if existing is None or span > (existing[3] - existing[2]):
            best_by_symbol[symbol] = (symbol, chrom, start, end)

    # Sort by (chr, start) using the canonical genomic order, not lexicographic.
    rows_sorted = sorted(
        best_by_symbol.values(),
        key=lambda r: (_chrom_sort_key(r[1]), r[2]),
    )

    # Write a header + tab-delimited body; downstream loads with pandas.read_csv(sep='\t').
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("gene_symbol\tchr\tstart\tend\n")
        for symbol, chrom, start, end in rows_sorted:
            fh.write(f"{symbol}\t{chrom}\t{start}\t{end}\n")

    n_rows = len(rows_sorted)
    print(f"wrote {n_rows} gene rows to {out}")


def main():
    """CLI entry point: parse args and dispatch to build()."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gtf", required=True, help="Path to gencode.v27.annotation.gtf.gz")
    parser.add_argument("--out", required=True, help="Output TSV path")
    args = parser.parse_args()

    build(args.gtf, args.out)


if __name__ == "__main__":
    main()
