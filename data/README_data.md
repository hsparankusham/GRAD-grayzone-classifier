# Data Access Instructions

This study uses individual-level data from two restricted-access cohorts. Per the respective Data Use Agreements, raw participant data **cannot be redistributed**. Researchers must obtain data access independently.

A synthetic demonstration dataset is provided in `data/synthetic/` for testing the pipeline without requiring data access.

---

## ADNI (Alzheimer's Disease Neuroimaging Initiative)

**Apply at:** https://adni.loni.usc.edu/data-samples/access-data/

**Required:** ADNI Data Use Agreement (free for qualified researchers)

### Required Data Files

Download the following files from the LONI Image & Data Archive and place them in a directory of your choice (passed as `--adni-dir` to scripts):

| File | LONI Location | Description |
|------|---------------|-------------|
| `UPENN_PlasmaBiomarkers.csv` | Biomarkers > Pathology | UPENN plasma p-tau217, AB42, AB40, NfL, GFAP |
| `JANSSEN_PLASMA_P217_TAU_18Dec2025.csv` | Biomarkers > Pathology | Janssen plasma p-tau217 assay |
| `ADNIMERGE2025.csv` | Study Data > Merged | Demographics, diagnosis, amyloid PET (AV45), MRI volumes |
| `APOERES_26Dec2025.csv` | Genetics > Reserve | APOE genotype |

### Expected Directory Structure

```
your-adni-data/
├── Pathology/
│   ├── UPENN_PlasmaBiomarkers.csv
│   └── JANSSEN_PLASMA_P217_TAU_18Dec2025.csv
├── ADNIMERGE2025.csv
└── Reserve/
    └── Genetics/
        └── APOERES_26Dec2025.csv
```

---

## A4 Study (Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease)

**Apply at:** https://www.a4studydata.org/

**Required:** A4 Study Data Use Agreement

### Required Data Files

| File | Portal Location | Description |
|------|-----------------|-------------|
| `biomarker_pTau217.csv` | Clinical > External Data | Lilly plasma p-tau217 |
| `biomarker_Plasma_Roche_Results.csv` | Clinical > External Data | Roche plasma GFAP, NfL, AB42, AB40 |
| `SUBJINFO.csv` | Clinical > Derived Data | Demographics, APOE4, amyloid PET (centiloid), MMSE |
| `ptdemog.csv` | Clinical > Raw Data | Participant demographics |

### Expected Directory Structure

```
your-a4-data/
├── Clinical/
│   ├── External Data/
│   │   ├── biomarker_pTau217.csv
│   │   └── biomarker_Plasma_Roche_Results.csv
│   ├── Derived Data/
│   │   └── SUBJINFO.csv
│   └── Raw Data/
│       └── ptdemog.csv
```

---

## Data Preparation

After obtaining access to both datasets, run the pipeline:

```bash
python scripts/run_grad_master.py --adni-dir /path/to/adni --a4-dir /path/to/a4
```

The script will load, merge, and harmonize the data automatically.

---

## Synthetic Demo Data

A synthetic dataset (`data/synthetic/synthetic_cohort.csv`) is provided for pipeline testing. This dataset:

- Preserves the statistical structure (distributions, correlations, class ratios) of the real data
- Contains **no real participant data** and cannot be used to identify individuals
- Produces similar (but not identical) metrics to the manuscript results
- Allows verification that the code runs correctly without requiring ADNI/A4 access

To run the demo:

```bash
python scripts/run_demo.py
```
