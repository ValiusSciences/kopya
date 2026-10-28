# Running kopya on 10x Atera data

An end-to-end guide: from a 10x Atera output bundle to tumor / normal calls you can check. Every command here was run with kopya 2.2.0 from PyPI in a fresh Python 3.12 environment, on 10x's public Atera datasets.

Atera is 10x Genomics' whole-transcriptome in-situ platform (about 18,000 genes, single-cell resolution, built on Xenium). kopya reads its count matrix like any other single-cell matrix; the in-situ setting changes a few defaults worth knowing about, covered below.

## Which input flag

Atera data comes in two formats, depending on the software that produced it:

| Data | Count matrix | kopya flag |
| --- | --- | --- |
| Atera WTA preview datasets (Xenium-style files) | `cell_feature_matrix.h5` | `--cellranger-h5` |
| Atera Onboard Analysis v1.0 and later (released format) | `cell_feature_matrix.zarr.zip` | `--atera` |

The preview bundles also contain a `cell_feature_matrix.zarr.zip`, but in Xenium's own layout, which `--atera` does not read: it stops with `missing X/data, X/indices, ...`. Use the `.h5` for preview data. `kopya find-inputs` (step 3) picks the right file for either format.

## 1. Install

```bash
pip install "valius-kopya>=2.2.0" "pynndescent<0.7"
```

Python 3.11 to 3.13. Version 2.2.0 is the first with `--atera`, and it pins pyarrow below 26 so a fresh install can write its outputs.

Keep the `pynndescent<0.7` pin. pynndescent 0.7.0 returns a sparse type that scanpy 1.11.5's neighbour graph cannot handle, so with it every `kopya run` stops at the subclone step with `AttributeError: 'csr_array' object has no attribute 'getnnz'`, after the run has already spent its full time. pip picks 0.7.0 by default, so a fresh install is affected. Already installed? Run `pip install "pynndescent<0.7"`.

## 2. Get the count matrix

kopya needs only the count matrix, not the 45 to 90 GB outs bundle. Either download the bundle and unzip one file:

```bash
unzip outs.zip cell_feature_matrix.h5          # preview data
unzip outs.zip cell_feature_matrix.zarr.zip    # released data
```

or pull just that file from 10x's server, which accepts byte-range requests, with [remotezip](https://pypi.org/project/remotezip/):

```bash
pip install remotezip
remotezip -l <outs.zip URL>                          # list the bundle
remotezip <outs.zip URL> cell_feature_matrix.h5      # extract one file
```

On the breast preview bundle this extracts the 398 MB matrix from a 55 GB zip in about 25 seconds.

### The 10X preview datasets

The two public preview datasets used in this guide. The count matrix is in `_outs.zip`; the smaller `_xe_outs.zip` (Xenium Explorer files) does not contain `cell_feature_matrix.h5` and is not needed. `_cell_groups.csv` holds 10X's cell annotations, which the [reference results](#reference-results) compare kopya's calls against.

Breast cancer ([dataset page](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer)):

```bash
mkdir -p breast && cd breast
remotezip https://s3-us-west-2.amazonaws.com/10x.files/samples/atera/dev/WTA_Preview_FFPE_Breast_Cancer/WTA_Preview_FFPE_Breast_Cancer_outs.zip cell_feature_matrix.h5
curl -O https://cf.10xgenomics.com/samples/atera/dev/WTA_Preview_FFPE_Breast_Cancer/WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv
cd ..
```

Cervical cancer ([dataset page](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer)):

```bash
mkdir -p cervical && cd cervical
remotezip https://s3-us-west-2.amazonaws.com/10x.files/samples/atera/dev/WTA_Preview_FFPE_Cervical_Cancer/WTA_Preview_FFPE_Cervical_Cancer_outs.zip cell_feature_matrix.h5
curl -O https://cf.10xgenomics.com/samples/atera/dev/WTA_Preview_FFPE_Cervical_Cancer/WTA_Preview_FFPE_Cervical_Cancer_cell_groups.csv
cd ..
```

To download a full bundle instead, `curl -O` the same `_outs.zip` URL. The dataset pages also list a gene-group table, an H&E image and its alignment files; kopya uses none of them.

## 3. Find the input

```bash
kopya find-inputs --patient-dir path/to/bundle --sample my-sample
```

It lists the candidate files and prints a ready-to-run command: `--atera` for a released bundle, `--cellranger-h5` for a preview one.

## 4. Remove probe-artifact genes

```bash
grep -v -P "^(OR4F17|TRIM49D1|PDE4DIP)\t" \
  "$(python -c 'import kopya, pathlib; print(pathlib.Path(kopya.__file__).parent / "data" / "gencode_v49_hg38.tsv")')" \
  > gene_order_atera.tsv
```

This writes a copy of kopya's bundled gene table without three genes whose Atera probes read as background rather than expression. The table is passed to `--gene-order` below.

**OR4F17** matters most. It is an olfactory-receptor gene that should be silent in these tissues, yet the Atera panel detects it in most cells: 95% of cells in the breast preview, where it is the second most abundant gene, and 65% in a released v1.0.4 kidney section. Its share of a cell's counts falls with cell size, so small cells (lymphocytes, neutrophils) carry more of it. It is the first gene on chromosome 19, and kopya turns that cell-type-dependent offset into a fake segment at the chromosome tip. Removing it improved every sample tested.

**TRIM49D1** (11q14, normally testis-only) and **PDE4DIP** (1q21) carry the same signature, though weaker. They are detected in about half of breast cells (TRIM49D1 52.7%, PDE4DIP 50.2%). Their counts barely rise with cell size: they sit at the 2nd to 3rd percentile among panel genes of similar detection, where a real transcript scales with the cell. Their levels move together with OR4F17 across 10X's cell groups, highest in myeloid cells, neutrophils and T cells and lowest in fibroblasts, instead of following any lineage. Removing them on top of OR4F17 raised breast tumor recall from 92.5% to 93.2%; on cervical the change was within noise. They cost nothing to drop.

All three are on 10x's probe disallow list for Flex v2, as is **ZNG1A** (9p24). On Atera, ZNG1A behaves like a real transcript: its counts scale with cell size and are highest in tumor and mitotic cells. Removing it changed neither dataset's results, so this guide keeps it. Dropping it as well does no harm if you prefer to follow the Flex list as is.

## 5. Run

Preview data:

```bash
kopya run --cellranger-h5 cell_feature_matrix.h5 \
  --gene-order gene_order_atera.tsv \
  --out-dir results/ --sample my-sample --threads 8
```

Released data, passing the bundle directory or the zip:

```bash
kopya run --atera path/to/bundle \
  --gene-order gene_order_atera.tsv \
  --out-dir results/ --sample my-sample --threads 8
```

**Low-depth sections: lower `--low-dr`.** In-situ data has fewer transcripts per cell than droplet data, and kopya's default gene filter (`--low-dr 0.05`, keep a gene only if at least 5% of cells detect it) drops too many genes at low depth. Add `--low-dr 0.02` when the median transcripts per cell, from 10x's metrics summary, is below about 1,500. On the cervical preview (929 transcripts per cell) it kept 9,550 genes instead of 5,906 and brought the share of immune and stromal cells called tumor from 6.9% to 4.2%. Do not lower it on deeper sections: on the breast preview (2,116 transcripts per cell) it would let in 138 olfactory-receptor genes that are background, not expression.

## 6. Resources

kopya 2.2.0 at default settings, 8 threads, single runs:

| Dataset | Cells | Median transcripts per cell | Wall time | Peak memory |
| --- | --- | --- | --- | --- |
| Breast cancer, preview | 170,057 | 2,116 | 3 min 33 s | 8.4 GiB |
| Cervical cancer, preview | 717,576 | 929 | 7 min 52 s | 15.9 GiB |

`--low-dr 0.02` keeps more genes and costs time: cervical took about 11 minutes and 16.6 GiB. Plan for 16 GB of RAM for a full Atera section. A 7 GB machine ran out of memory on the 170,057-cell breast section.

## 7. Check the output before trusting it

The outputs are the usual kopya files (see the main README): `prediction.csv` with per-cell calls, `chr_cnv_matrix.csv`, `segments.parquet`, an IGV `.seg` and `qc.json`. Things that behave differently on in-situ data:

- **Start unsupervised.** On the cervical preview, seeding the baseline with lymphocytes only (`--norm-cell-names`) raised the share of immune and stromal cells called tumor from 6.9% to 24.7%. A small, low-count reference differs from myeloid and stromal cells in more than copy number. If you supervise, include several normal lineages.
- **Expect many low-complexity flags.** 16% (cervical) and 21% (breast) of cells were flagged `low_complexity`. That reflects in-situ depth, not empty droplets.
- **Treat short segments at chromosome ends with suspicion.** In every Atera section tested, several of the strongest segments in `segments.parquet` were 25 to 37 genes long and sat at a chromosome end, the same ends across unrelated tissues. They come from how smoothing handles chromosome edges, not from biology.
- **Expect some tumor calls on immune cells inside tumor nests.** In-situ segmentation assigns some neighbouring transcripts to the wrong cell. On the cervical preview, immune and stromal cells surrounded by tumor were called tumor 17.6% of the time, against 5.7% for those with no tumor neighbours; exhausted T and dendritic cells were affected most.
- **Do not run it on tumor-free tissue.** kopya always splits cells into two groups and has no "no tumor present" outcome. On a normal kidney section it called about 30% of cells tumor, mostly tubular epithelium, which differs from the immune and stromal reference in lineage, not copy number.

## 8. Python API

`load_counts(atera=...)` returns an AnnData that keeps 10x's per-cell annotations, including the Azimuth cell types, which can serve as the reference for `kp.tl.cnv`:

```python
import kopya as kp
from kopya.io import load_counts

adata = load_counts(atera="path/to/bundle")    # released format; cellranger_h5=... for preview data
adata.obs["azimuth_broad"].value_counts()

kp.tl.cnv(adata, reference_key="azimuth_broad",
          reference_cat=["Immune cell", "Endothelial cell"])
adata.obs["cnv_class"].value_counts()
```

`tl.cnv` holds the whole smoothed matrix in memory, so use it on a subset or a region of a section. For a full section, use `kopya run`, which processes cells in blocks. Loading the 399,120-cell kidney bundle and running `tl.cnv` on 20,000 of its cells peaked at 9.5 GiB.

## Reference results

kopya's calls on the two 10x preview datasets, compared with 10x's own cell-group annotations. Tumor groups are the positive class; immune and stromal groups the negative class. Normal cells of the tumor's own lineage (breast myoepithelial and apocrine, cervical endocervical) are left out of these scores.

| Dataset and setup | AUC | Immune / stromal called tumor | Tumor cells called tumor |
| --- | --- | --- | --- |
| Breast, defaults | 0.998 | 0.5% | 91.2% |
| Breast, OR4F17 removed | 0.997 | 0.6% | 92.5% |
| Breast, OR4F17, TRIM49D1 and PDE4DIP removed | 0.997 | 0.6% | 93.2% |
| Cervical, defaults | 0.945 | 6.9% | 80.8% |
| Cervical, OR4F17 removed | 0.966 | 4.4% | 80.0% |
| Cervical, OR4F17, TRIM49D1 and PDE4DIP removed | 0.966 | 4.3% | 79.9% |
| Cervical, OR4F17 removed, `--low-dr 0.02` | 0.977 | 3.7% | 84.2% |

AUC ranks cells by `tumor_score`; the rates use the `class` column. Breast tumor cells are the "11q13 High Grade DCIS" groups; cervical tumor cells are the seven squamous cell carcinoma groups. On breast, kopya also finds the 11q13 (CCND1) gain those cells are named for, plus a distal 11q loss the label does not capture; on cervical, the 3q gain typical of cervical squamous cell carcinoma is the strongest event.

These labels come from 10x's clustering and marker genes, not from DNA, so the comparison measures agreement between two expression-based classifications. Datasets: [Atera WTA FFPE human breast cancer](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer) and [Atera WTA FFPE human cervical cancer](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer), both 10x preview data under CC BY 4.0.

## Known issues

- `--atera` on a preview bundle's Xenium-style zarr reports missing elements but does not point to the `.h5`.
- `find-inputs` labels a preview bundle's zarr "Atera count matrix" and marks it preferred alongside the `.h5` it actually recommends.
- Removing the probe-artifact genes is a manual step; an in-situ probe denylist in kopya would make it automatic.
- kopya 2.2.0 does not pin pynndescent below 0.7, so a fresh install needs the pin from step 1.
- `--low-dr` is a fixed detection share; a depth-aware default would remove the low-depth tuning step.
- Edge smoothing creates the chromosome-end segments described above; a fix that averages over the genes a window covers, instead of repeating the edge gene, raised breast tumor recall from 92.5% to 94.6% in a patched run.
