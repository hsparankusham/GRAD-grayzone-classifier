# GRAD: Gatekeeper-Reflex for Alzheimer's Diagnostics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18932865.svg)](https://doi.org/10.5281/zenodo.18932865)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A two-stage machine learning algorithm that resolves the plasma p-tau217 diagnostic gray zone for amyloid PET prediction.**

> Parankusham H, Vanderlip C, Birkenbihl CJ, Krishna E, Ugboaja C, Budson A, Frank B.
> *Alzheimer's Research & Therapy* (2026). [Under Review]

---

## Overview

Plasma phospho-tau 217 (p-tau217) is a leading blood-based biomarker for Alzheimer's disease, but 30-50% of patients fall into a diagnostic "gray zone" where the result is indeterminate. GRAD resolves this uncertainty using a two-stage algorithm:

1. **Gatekeeper (Stage 1):** A univariate logistic regression on Z-harmonized p-tau217 classifies high-confidence positive (P > 0.75) and negative (P < 0.25) cases, resolving 55.6% of patients with 88.8% accuracy.

2. **Reflex (Stage 2):** A 6-feature random forest classifier resolves the remaining gray zone cases using p-tau217, GFAP, Abeta42/40 ratio, age, APOE4 status, and a GFAP-tau interaction term.

### Key Results

| Metric | ADNI (Internal, LOOCV) | A4 + LEARN (External) |
|--------|:----------------------:|:---------------------:|
| **AUC** | 0.853 (95% CI: 0.808-0.894) | 0.821 (95% CI: 0.798-0.841) |
| **Accuracy** | 79.7% | 72.9% |
| **Sensitivity** | 79.4% | 70.0% |
| **Specificity** | 80.0% | 79.8% |
| **Gatekeeper resolved** | 55.6% @ 88.8% accuracy | 21.4% |

---

## Repository Structure

```
GRAD-grayzone-classifier/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── CITATION.cff                    # Machine-readable citation
├── .zenodo.json                    # Zenodo archival metadata
├── requirements.txt                # Python dependencies (pinned)
├── environment.yml                 # Conda environment specification
│
├── config/
│   ├── config.yaml                 # All hyperparameters, thresholds, seeds
│   └── data_dictionary.csv         # Variable definitions, sources, units
│
├── src/                            # Core algorithm modules
│   ├── data_loader.py              # Load and merge ADNI / A4 datasets
│   ├── harmonizer.py               # Z-score normalization (cross-platform)
│   ├── gatekeeper.py               # Stage 1: Univariate logistic regression
│   ├── reflex.py                   # Stage 2: 6-feature random forest
│   ├── pipeline.py                 # Two-stage pipeline orchestration
│   ├── validation.py               # Leave-one-out cross-validation
│   └── visualization.py            # Publication figure generation
│
├── scripts/
│   ├── run_all.sh                  # Single command to reproduce everything
│   ├── run_demo.py                 # Demo on synthetic data (no data access needed)
│   ├── run_grad_master.py          # Self-contained end-to-end pipeline
│   ├── run_authoritative_loocv.py  # ADNI internal validation (LOOCV)
│   ├── run_a4_binary_validation.py # A4 external validation
│   ├── run_nfl_ablation.py         # NfL feature ablation analysis
│   ├── run_subgroup_analysis.py    # Performance by cognitive status, APOE4, sex
│   ├── run_calibration_analysis.py # Calibration curves, Hosmer-Lemeshow test
│   ├── generate_synthetic_data.py  # Generate demo dataset
│   ├── generate_figure2_combined.py    # Figure 2: ROC curves
│   ├── generate_figure3_final.py       # Figure 3: Model characterization (8 panels)
│   ├── generate_figure4.py             # Figure 4: A4 external validation
│   ├── generate_manuscript_figures.py  # Figures 1, 5, 6
│   ├── generate_figure_s3_subgroup.py  # Figure S3: Subgroup analysis
│   └── generate_stard_diagram.py       # Figure S2: STARD flow diagram
│
├── data/
│   ├── README_data.md              # ADNI and A4 data access instructions
│   └── synthetic/
│       └── synthetic_cohort.csv    # Demo dataset (no real patient data)
│
├── results/                        # Expected outputs for verification
│   ├── adni_loocv_predictions.csv  # ADNI LOOCV predictions (N=320)
│   ├── a4_binary_validation_predictions.csv  # A4 predictions (N=1,644)
│   ├── figures/                    # Manuscript figures (PNG + PDF)
│   ├── tables/                     # Supplementary tables (CSV + JSON)
│   └── figure_3/                   # Individual Figure 3 panels
│
└── tests/                          # Unit tests
```

---

## System Requirements

- **Operating system:** Tested on macOS 14.x (Apple Silicon) and Ubuntu 22.04
- **Python:** 3.10+
- **Hardware:** Standard desktop/laptop (no GPU required)
- **RAM:** 4 GB minimum
- **Install time:** ~2 minutes
- **Demo runtime:** ~2 minutes (LOOCV on 320 synthetic samples)
- **Full reproduction runtime:** ~15 minutes (320 LOOCV folds + A4 validation + figures)

---

## Installation

### Option 1: pip

```bash
git clone https://github.com/hsparankusham/GRAD-grayzone-classifier.git
cd GRAD-grayzone-classifier
pip install -r requirements.txt
```

### Option 2: Conda

```bash
git clone https://github.com/hsparankusham/GRAD-grayzone-classifier.git
cd GRAD-grayzone-classifier
conda env create -f environment.yml
conda activate grad-classifier
```

---

## Quick Start: Demo (No Data Access Required)

Run the full GRAD pipeline on synthetic data:

```bash
python scripts/run_demo.py
```

This demonstrates the complete two-stage algorithm (harmonization, Gatekeeper, Reflex, LOOCV) using a synthetic dataset that preserves the statistical properties of the real cohort without containing any real patient data. The demo AUC will differ from manuscript values since the data is synthetic.

---

## Reproducing Manuscript Results

### Prerequisites

Manuscript reproduction requires access to two restricted-access datasets:

1. **ADNI:** Apply at https://adni.loni.usc.edu/data-samples/access-data/
2. **A4 Study:** Apply at https://www.a4studydata.org/

See [`data/README_data.md`](data/README_data.md) for the exact files needed and directory structure.

### Run All Analyses

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
# Step 1: ADNI internal validation (LOOCV, N=320)
python scripts/run_authoritative_loocv.py

# Step 2: A4 external validation (N=1,644)
python scripts/run_a4_binary_validation.py

# Step 3: Supplementary analyses
python scripts/run_nfl_ablation.py
python scripts/run_subgroup_analysis.py
python scripts/run_calibration_analysis.py

# Step 4: Generate all manuscript figures
python scripts/generate_figure2_combined.py
python scripts/generate_figure3_final.py
python scripts/generate_figure4.py
python scripts/generate_manuscript_figures.py
python scripts/generate_figure_s3_subgroup.py
python scripts/generate_stard_diagram.py
```

### Standalone Walkthrough

For a self-contained, annotated version of the entire pipeline in a single file:

```bash
python scripts/run_grad_master.py
```

This 900-line script inlines all algorithm logic with detailed comments explaining each step and the biological rationale behind design decisions. Recommended for reviewers who want to follow the complete methodology without navigating multiple modules.

### Expected Outputs

After full reproduction, verify against these values (also in `config/config.yaml`):

| Metric | Expected Value |
|--------|:--------------:|
| ADNI LOOCV AUC | 0.8527 |
| A4 External AUC | 0.8205 |
| Gatekeeper resolution rate | 55.6% |
| Gatekeeper accuracy | 88.8% |
| Spearman r (centiloid) | 0.631 |
| Brier score (A4) | 0.181 |

---

## Configuration

All hyperparameters, decision thresholds, random seeds, and model settings are documented in [`config/config.yaml`](config/config.yaml). Key parameters:

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `random_seed` | 42 | Global random seed for reproducibility |
| `gatekeeper.threshold_negative` | 0.25 | Below this probability: classify as amyloid-negative |
| `gatekeeper.threshold_positive` | 0.75 | Above this probability: classify as amyloid-positive |
| `reflex.n_estimators` | 100 | Number of trees in random forest |
| `reflex.max_depth` | 5 | Maximum tree depth |
| `reflex.min_samples_leaf` | 5 | Minimum samples per leaf |
| ADNI amyloid threshold | AV45 SUVR > 1.11 | Amyloid PET positivity cutoff |
| A4 amyloid threshold | Centiloid >= 20 | Amyloid PET positivity cutoff |

---

## Figure-to-Script Mapping

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 1 | `generate_manuscript_figures.py` | GRAD architecture + probability distribution |
| Figure 2 | `generate_figure2_combined.py` | ROC curves (overall, Gatekeeper, Reflex) |
| Figure 3 | `generate_figure3_final.py` | 8-panel model characterization |
| Figure 4 | `generate_figure4.py` | A4 external validation (ROC, centiloid, calibration) |
| Figure 5 | `generate_manuscript_figures.py` | MRI volumetric enhancement |
| Figure 6 | `generate_manuscript_figures.py` | Health economic cost comparison |
| Figure S2 | `generate_stard_diagram.py` | STARD 2015 participant flow |
| Figure S3 | `generate_figure_s3_subgroup.py` | Subgroup analysis + threshold sensitivity |

---

## Data Availability

- **ADNI data:** Available from the LONI Image & Data Archive (https://ida.loni.usc.edu/) upon completion of a Data Use Agreement.
- **A4 Study data:** Available from the A4 Study Data portal (https://www.a4studydata.org/) upon approval of a data access request.
- **Analysis code:** This repository (archived at [Zenodo DOI pending]).
- **Synthetic demo data:** Included in `data/synthetic/` for pipeline testing.

---

## Citation

If you use this code, please cite:

```bibtex
@article{parankusham2026grad,
  title={GRAD: A Two-Stage Machine Learning Algorithm Resolves the Plasma
         p-Tau217 Diagnostic Gray Zone for Amyloid PET Prediction},
  author={Parankusham, Harthik and Vanderlip, Casey and Birkenbihl, Colin Jan
          and Krishna, Eashwar and Ugboaja, Chizobam and Budson, Andrew
          and Frank, Brandon},
  journal={Alzheimer's Research \& Therapy},
  year={2026},
  note={Under Review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu) and the A4 Study (a4study.org). ADNI investigators and A4 Study investigators contributed to the design and implementation of their respective studies but did not participate in analysis or writing of this report. A complete listing of ADNI and A4 investigators can be found at their respective websites.
