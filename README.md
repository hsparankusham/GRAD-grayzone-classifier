# GRAD: Gatekeeper-Reflex for Alzheimer's Diagnostics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18932865.svg)](https://doi.org/10.5281/zenodo.18932865)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A two-stage plasma biomarker algorithm that resolves the p-tau217 diagnostic gray zone and reduces amyloid PET use.**

> Parankusham H, Vanderlip C, Birkenbihl C, Krishna E, Ugboaja C, Budson A, Frank B.
> *Annals of Clinical and Translational Neurology* (2026). [Under Review]

---

## Overview

Plasma phospho-tau 217 (p-tau217) is a leading blood-based biomarker for Alzheimer's
disease, but 30–50% of patients fall into a diagnostic "gray zone" where the result is
indeterminate. GRAD resolves that zone algorithmically, reserving Aβ-PET for the cases
that remain uncertain:

1. **Gatekeeper (Stage 1).** Univariate logistic regression on reference-anchored
   p-tau217. Predicted probability below 0.25 is called Aβ-negative, above 0.75
   Aβ-positive; the interval between is the gray zone and is routed forward.

2. **Reflex (Stage 2).** L2-regularized logistic regression (C = 0.1, balanced class
   weights, features standardized within each training fold) scores gray zone cases on
   seven features: p-tau217, GFAP, Aβ42/40, a tau−Aβ divergence ratio, a GFAP × p-tau217
   interaction, age, and APOE ε4 carrier status. Output outside 0.40–0.60 yields a
   definitive call; inside it, the case is referred for confirmatory Aβ-PET.

The model is developed in **cognitively impaired** ADNI participants and validated,
without refitting, in a **preclinical** A4 + LEARN cohort — two clinically distinct
populations measured on different p-tau217 assay platforms.

## Repository structure

```
GRAD-grayzone-classifier/
├── config/config.yaml              # Hyperparameters, thresholds, seeds, expected values
├── data/README_data.md             # ADNI and A4 access instructions
├── scripts/
│   ├── run_grad_impaired_logistic.py   # The pipeline. Regenerates every manuscript number
│   ├── GRAD_reference_sensitivity.py   # Reference-anchoring sensitivity analysis
│   ├── GRAD_fig2_panels.py             # Figure 2 panels A–E
│   ├── GRAD_fig3_panels.py             # Figure 3 panels A–D, plus Figure S1
│   ├── GRAD_fig4_panels.py             # Figure 4 panels A–D
│   ├── GRAD_supplementary_tables.py    # Supplementary Tables S1–S9
│   ├── _grad_paths.py                  # Data and output locations
│   └── _grad_style.py                  # Frozen figure style (600 dpi RGB, Helvetica 10 pt)
├── results/
│   ├── adni_loocv_predictions_v2.csv   # Per-participant LOOCV predictions
│   ├── a4_predictions_v2.csv           # Per-participant external predictions
│   ├── tables/grad_v2_numbers.json     # Every number reported in the manuscript
│   ├── tables/reference_sensitivity.json
│   └── figures/panels/                 # 14 publication panels
└── archive/v1_random_forest/       # Superseded random-forest design (see below)
```

**On `archive/`.** An earlier version of this work used a random forest for Stage 2 and
an unrestricted ADNI cohort. That design is superseded and its scripts, outputs and
drafts are retained under `archive/v1_random_forest/` for provenance only. Nothing there
reproduces the current manuscript, and its numbers do not match it.

---

## Requirements

- Python 3.10+, standard laptop (no GPU), 4 GB RAM
- Full reproduction: ~10 minutes (145 LOOCV folds, external scoring, figures, tables)

```bash
git clone https://github.com/hsparankusham/GRAD-grayzone-classifier.git
cd GRAD-grayzone-classifier
pip install -r requirements.txt      # or: conda env create -f environment.yml
```

---

## Reproducing the manuscript

Both datasets are restricted-access and must be obtained directly:

1. **ADNI** — https://adni.loni.usc.edu/data-samples/access-data/
2. **A4 / LEARN** — https://www.a4studydata.org/

See [`data/README_data.md`](data/README_data.md) for the files required and the expected
directory layout, then set paths in `scripts/_grad_paths.py`.

```bash
# 1. The pipeline. Writes results/tables/grad_v2_numbers.json and both prediction files.
python scripts/run_grad_impaired_logistic.py

# 2. Reference-anchoring sensitivity analysis (Supplementary Table S3)
python scripts/GRAD_reference_sensitivity.py

# 3. Figure panels
python scripts/GRAD_fig2_panels.py
python scripts/GRAD_fig3_panels.py
python scripts/GRAD_fig4_panels.py

# 4. Supplementary Tables S1–S9 (.docx and .md)
python scripts/GRAD_supplementary_tables.py
```

Step 1 is the source of truth: every figure and table script reads from the JSON and the
two prediction files it writes, so the manuscript cannot drift from the code.

## License and citation

MIT (see [LICENSE](LICENSE)). If you use this code, please cite the manuscript above and
the archived release: [10.5281/zenodo.18932865](https://doi.org/10.5281/zenodo.18932865).
Citation metadata is in [CITATION.cff](CITATION.cff).
