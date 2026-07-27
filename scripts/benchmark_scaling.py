#!/usr/bin/env python
"""Reproducible scaling benchmark for the kopya pipeline.

Subsamples a large single-cell matrix down to a ladder of cell counts, runs the
full `kopya run` pipeline on each, N replicates each, and records peak
resident memory + wall-clock duration. Results stream to a CSV as they complete
and a mean/std summary table is printed (and saved as Markdown) at the end.

Design notes:
  * Reproducible: each (size, replicate) draws its cell subset from a seeded
    numpy Generator keyed on (--seed, size, rep), so re-running reproduces the
    exact same subsamples.
  * Clean measurement: every run is an isolated subprocess wrapped in
    /usr/bin/time, and it loads only its N-cell subsample (never the full
    matrix), so the reported peak RSS reflects the pipeline at N cells — not the
    one-time full-matrix load. Subsamples are pre-written in a separate prep
    subprocess so the full matrix is released before any measured run starts.
  * Cross-platform peak RSS: parses `/usr/bin/time -l` on macOS (bytes) and
    `/usr/bin/time -v` on Linux (kbytes).

Usage:
    python scripts/benchmark_scaling.py \
        --h5 /path/to/full_matrix.h5 \
        --out /path/to/bench_workdir \
        [--sizes 5000,10000,25000,50000,100000,200000,400000,800000] \
        [--reps 10] [--seed 0] [--timeout 3600] [--keep-subsamples]

The --h5 input may be a CellRanger .h5 (read via scanpy.read_10x_h5) or an
.h5ad (read via scanpy.read_h5ad); format is inferred from the extension.
"""
import argparse
import csv
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SIZES = [5000, 10000, 25000, 50000, 100000, 200000, 400000, 800000]


# ─────────────────────────────────────────────────────────────────────────────
# Subsample prep (runs as its own process so the full matrix is freed after)
# ─────────────────────────────────────────────────────────────────────────────
def _read_matrix(h5_path):
    import scanpy as sc
    p = str(h5_path)
    return sc.read_h5ad(p) if p.endswith(".h5ad") else sc.read_10x_h5(p)


def prep_subsamples(h5_path, subdir, sizes, reps, seed):
    """Write one raw-counts .h5ad per (size, rep) into subdir. Deterministic."""
    import numpy as np
    subdir = Path(subdir)
    subdir.mkdir(parents=True, exist_ok=True)
    print(f"[prep] loading {h5_path}", flush=True)
    adata = _read_matrix(h5_path)
    n_obs = adata.n_obs
    print(f"[prep] full: {n_obs:,} cells x {adata.n_vars:,} genes", flush=True)
    for size in sizes:
        for rep in range(reps):
            out = subdir / f"sub_{size}_{rep}.h5ad"
            if out.exists():
                continue
            m = min(size, n_obs)
            # Keyed on (seed, size, rep) -> reproducible, independent per cell.
            rng = np.random.default_rng([seed, size, rep])
            idx = np.sort(rng.choice(n_obs, m, replace=False))
            # Write RAW counts (no var-name dedup here): the pipeline does its own
            # filtering/dedup, so we benchmark exactly what `run` does end to end.
            adata[idx].copy().write_h5ad(out)
        print(f"[prep] size={size}: {reps} subsamples ready", flush=True)
    print("[prep] DONE", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Measured run
# ─────────────────────────────────────────────────────────────────────────────
def _time_cmd_and_parser():
    """(/usr/bin/time prefix, regex, kb->? ) for the current OS."""
    if platform.system() == "Darwin":
        return (["/usr/bin/time", "-l"],
                re.compile(r"(\d+)\s+maximum resident set size"),
                1 / 1073741824)          # bytes -> GiB
    return (["/usr/bin/time", "-v"],
            re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)"),
            1024 / 1073741824)           # kbytes -> GiB


def measure_run(sub_h5ad, out_dir, timeout):
    """Run the pipeline on one subsample; return (status, wall_s, peak_gb, qc)."""
    time_prefix, rss_re, rss_to_gib = _time_cmd_and_parser()
    cmd = time_prefix + [
        "kopya", "run",
        "--anndata", str(sub_h5ad),
        "--out-dir", str(out_dir),
        "--sample", "bench",
    ]
    t0 = time.time()
    status, stderr = "ok", ""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, timeout=timeout)
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            status = f"ERR{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stderr = (exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr) or ""
    wall = time.time() - t0

    peak_gb = None
    match = rss_re.search(stderr)
    if match:
        peak_gb = round(int(match.group(1)) * rss_to_gib, 3)

    qc = {}
    qc_path = Path(out_dir) / "qc.json"
    if qc_path.exists():
        try:
            qc = json.loads(qc_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    return status, round(wall, 1), peak_gb, qc


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration + summary
# ─────────────────────────────────────────────────────────────────────────────
def summarize(rows, sizes):
    """Return a Markdown table of mean±std peak & duration per size."""
    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—", "—"
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, sd

    lines = [
        "| cells (input) | n runs | cells post-M1 | peak RSS GiB (mean±sd) | "
        "duration s (mean±sd) | min–max s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        srows = [r for r in rows if r["size"] == size and r["status"] == "ok"]
        n = len(srows)
        pk_m, pk_sd = agg([r["peak_gb"] for r in srows])
        du_m, du_sd = agg([r["wall_s"] for r in srows])
        durs = [r["wall_s"] for r in srows]
        cm = next((r["cells_post_m1"] for r in srows if r["cells_post_m1"]), "—")
        pk = f"{pk_m:.2f}±{pk_sd:.2f}" if isinstance(pk_m, float) else pk_m
        du = f"{du_m:.1f}±{du_sd:.1f}" if isinstance(du_m, float) else du_m
        rng = f"{min(durs):.0f}–{max(durs):.0f}" if durs else "—"
        lines.append(f"| {size:,} | {n} | {cm} | {pk} | {du} | {rng} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="Full matrix (.h5 CellRanger or .h5ad).")
    ap.add_argument("--out", required=True, help="Work dir for subsamples + results.")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                    help="Comma-separated cell counts.")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600, help="Per-run timeout (s).")
    ap.add_argument("--keep-subsamples", action="store_true",
                    help="Do not delete the subsample .h5ad files at the end.")
    # Internal: prep worker mode (spawned as a subprocess for memory isolation).
    ap.add_argument("--_prep", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    out = Path(args.out)
    subdir = out / "subsamples"

    if args._prep:
        prep_subsamples(args.h5, subdir, sizes, args.reps, args.seed)
        return

    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    runs_dir.mkdir(exist_ok=True)
    csv_path = out / "results.csv"

    # Phase 1: write all subsamples in an isolated subprocess (frees the full
    # matrix before any measured run) — load the big matrix exactly once.
    print(f"==== prep {len(sizes)} sizes x {args.reps} reps ====", flush=True)
    subprocess.run([sys.executable, __file__, "--_prep", "--h5", args.h5, "--out",
                    str(out), "--sizes", args.sizes, "--reps", str(args.reps),
                    "--seed", str(args.seed)], check=True)

    # Phase 2: measured runs (small -> large so partial results land early).
    rows = []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "size", "rep", "seed", "status", "wall_s", "peak_gb",
            "cells_post_m1", "n_segments", "n_tumor"])
        writer.writeheader()
        for size in sizes:
            for rep in range(args.reps):
                sub = subdir / f"sub_{size}_{rep}.h5ad"
                rdir = runs_dir / f"run_{size}_{rep}"
                status, wall, peak, qc = measure_run(sub, rdir, args.timeout)
                row = {
                    "size": size, "rep": rep, "seed": args.seed, "status": status,
                    "wall_s": wall, "peak_gb": peak,
                    "cells_post_m1": qc.get("n_cells_post_m1"),
                    "n_segments": qc.get("n_segments"), "n_tumor": qc.get("n_tumor"),
                }
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                print(f"[bench] size={size:>7,} rep={rep} status={status} "
                      f"wall={wall}s peak={peak}GiB", flush=True)
                # Free disk: drop per-run outputs immediately.
                import shutil
                shutil.rmtree(rdir, ignore_errors=True)

    table = summarize(rows, sizes)
    (out / "summary.md").write_text(table + "\n")
    print("\n==== SUMMARY ====\n" + table, flush=True)
    print(f"\nper-run CSV: {csv_path}\nsummary:     {out / 'summary.md'}", flush=True)

    if not args.keep_subsamples:
        import shutil
        shutil.rmtree(subdir, ignore_errors=True)
        print("[cleanup] removed subsamples", flush=True)


if __name__ == "__main__":
    main()
