"""Shared fixtures for external gold-standard dataset tests.

All fixtures skip gracefully when the referenced file or directory is absent,
so the external suite is always safe to collect in CI even without the large
benchmark datasets mounted.

Dataset layout under DATA_ROOT:
    gse57872/patel_gbm.h5ad          — Patel 2014 GBM (GSE57872)
    gse148673/dcis1.h5ad             — DCIS1 dataset (GSE148673)
    scevan-synthetic/h5ad/*.h5ad     — SCEVAN synthetic matrices
    hcc1395/*.h5ad                   — HCC1395 breast cancer cell line
    ovarian-scffpe/                  — 10X 17k ovarian scFFPE (HGSOC)
        17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5
        FLEX_Ovarian_Barcode_Cluster_Annotation.csv
    maynard-lung/maynard_lung.h5ad   — Maynard 2020 lung (maynard2020_3k, the infercnvpy
                                       tutorial dataset); build via
                                       tests/external/convert_maynard_lung.py.
    ipisrc044-osteosarcoma/ipisrc044_t1.h5ad
                                     — UCSF osteosarcoma T1 (IPISRC044_T1); build via
                                       tests/external/convert_ipisrc044_osteosarcoma.py.
                                       Bulk truth is committed at
                                       data/ipisrc044_t1_bulk_cnv_truth.csv.
"""

import os
from pathlib import Path

import pytest


DATA_ROOT = Path(os.environ.get("KOPYA_TEST_DATA", Path(__file__).resolve().parents[2] / "external-benchmarks"))


@pytest.fixture(scope="session")
def patel_gbm_h5ad():
    """Path to the Patel 2014 GBM AnnData file.

    Skips the test session if the file is not present.

    Returns:
        Path to DATA_ROOT/gse57872/patel_gbm.h5ad.
    """
    path = DATA_ROOT / "gse57872" / "patel_gbm.h5ad"
    if not path.exists():
        pytest.skip(
            f"Patel GBM dataset not found at {path}; "
            "download GSE57872 and convert to h5ad to run these tests."
        )
    return path


@pytest.fixture(scope="session")
def dcis1_h5ad():
    """Path to the DCIS1 AnnData file.

    Skips the test session if the file is not present.

    Returns:
        Path to DATA_ROOT/gse148673/dcis1.h5ad.
    """
    path = DATA_ROOT / "gse148673" / "dcis1.h5ad"
    if not path.exists():
        pytest.skip(
            f"DCIS1 dataset not found at {path}; "
            "download GSE148673 and convert to h5ad to run these tests."
        )
    return path


@pytest.fixture(scope="session")
def scevan_h5ad_list():
    """List of up to 10 SCEVAN synthetic h5ad files.

    Scans DATA_ROOT/scevan-synthetic/h5ad/ and returns the first 10 files
    found (sorted for reproducibility). Skips if the directory is absent or
    empty.

    Returns:
        list[Path] of up to 10 h5ad files.
    """
    h5ad_dir = DATA_ROOT / "scevan-synthetic" / "h5ad"
    if not h5ad_dir.is_dir():
        pytest.skip(
            f"SCEVAN synthetic directory not found at {h5ad_dir}; "
            "generate or download SCEVAN synthetic matrices to run these tests."
        )
    files = sorted(h5ad_dir.glob("*.h5ad"))[:10]
    if not files:
        pytest.skip(
            f"No .h5ad files found under {h5ad_dir}; "
            "generate or download SCEVAN synthetic matrices to run these tests."
        )
    return files


@pytest.fixture(scope="session")
def ovarian_scffpe_data():
    """Paths to the 10X 17k ovarian scFFPE matrix + published annotations.

    Skips the test session if either file is missing. Both are expected under
    DATA_ROOT/ovarian-scffpe/ with the same filenames 10X publishes so
    acquisition is a plain download (see tests/external/GOLD_STANDARD_TESTING.md).

    Returns:
        (h5_path, annotation_path) tuple of Paths:
            h5_path: CellRanger filtered_feature_bc_matrix.h5.
            annotation_path: per-barcode cell-type annotation CSV.
    """
    ovarian_dir = DATA_ROOT / "ovarian-scffpe"
    h5_path = ovarian_dir / "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
    annotation_path = ovarian_dir / "FLEX_Ovarian_Barcode_Cluster_Annotation.csv"
    missing = [p for p in (h5_path, annotation_path) if not p.exists()]
    if missing:
        pytest.skip(
            "10X ovarian scFFPE dataset not found "
            f"(missing: {', '.join(str(p) for p in missing)}); "
            "download the filtered matrix h5 and the FLEX barcode annotation CSV "
            "into DATA_ROOT/ovarian-scffpe/ to run these tests "
            "(URLs in tests/external/GOLD_STANDARD_TESTING.md)."
        )
    return h5_path, annotation_path


@pytest.fixture(scope="session")
def hcc1395_h5ad():
    """Path to the HCC1395 AnnData file.

    Scans DATA_ROOT/hcc1395/ for the first .h5ad present. Skips if none
    found.

    Returns:
        Path to the first .h5ad under DATA_ROOT/hcc1395/.
    """
    hcc_dir = DATA_ROOT / "hcc1395"
    if not hcc_dir.is_dir():
        pytest.skip(
            f"HCC1395 directory not found at {hcc_dir}; "
            "download the HCC1395 dataset and convert to h5ad to run these tests."
        )
    candidates = sorted(hcc_dir.glob("*.h5ad"))
    if not candidates:
        pytest.skip(
            f"No .h5ad files found under {hcc_dir}; "
            "download the HCC1395 dataset and convert to h5ad to run these tests."
        )
    return candidates[0]


@pytest.fixture(scope="session")
def maynard_lung_h5ad():
    """Path to the Maynard 2020 lung (maynard2020_3k) AnnData.

    Built on demand by tests/external/convert_maynard_lung.py (which fetches the
    dataset through infercnvpy). Skips the session if absent.

    Returns:
        Path to DATA_ROOT/maynard-lung/maynard_lung.h5ad — cells x genes with
        obs['cell_type'] (incl. 'Epithelial cell' as the malignant/observation
        set and the immune lineages used as the supervised normal reference).
    """
    path = DATA_ROOT / "maynard-lung" / "maynard_lung.h5ad"
    if not path.exists():
        pytest.skip(
            f"Maynard lung dataset not found at {path}; run "
            "`python tests/external/convert_maynard_lung.py` to build it "
            "(see tests/external/GOLD_STANDARD_TESTING.md)."
        )
    return path


@pytest.fixture(scope="session")
def ipisrc044_data():
    """Paths to the UCSF osteosarcoma T1 (IPISRC044_T1) single-cell h5ad + bulk truth.

    The single-cell h5ad (built on demand by
    tests/external/convert_ipisrc044_osteosarcoma.py) is gated under DATA_ROOT and
    the session skips if it is absent. The distilled bulk-exome CNV truth is
    committed in-repo (tests/external/data/), so it is always available — only the
    single-cell matrix needs downloading.

    Returns:
        (h5ad_path, truth_csv_path) tuple of Paths:
            h5ad_path: DATA_ROOT/ipisrc044-osteosarcoma/ipisrc044_t1.h5ad — cells x
                genes raw counts; obs['annot'] holds cell-type labels including
                'Putative_Tumor_Cells' (tumor) and the non-tumor immune/stromal
                lineages used as the supervised normal reference.
            truth_csv_path: committed per-chromosome / per-arm bulk-CNV truth.
    """
    h5ad_path = DATA_ROOT / "ipisrc044-osteosarcoma" / "ipisrc044_t1.h5ad"
    truth_csv_path = Path(__file__).parent / "data" / "ipisrc044_t1_bulk_cnv_truth.csv"
    if not h5ad_path.exists():
        pytest.skip(
            f"Osteosarcoma (IPISRC044_T1) single-cell data not found at {h5ad_path}; "
            "run `python tests/external/convert_ipisrc044_osteosarcoma.py` to "
            "download + build it (see tests/external/GOLD_STANDARD_TESTING.md)."
        )
    return h5ad_path, truth_csv_path
