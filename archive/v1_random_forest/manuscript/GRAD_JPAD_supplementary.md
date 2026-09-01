# GRAD: Supplementary Materials

**Manuscript title:** GRAD: A Two-Stage Algorithm for Resolving Diagnostic Uncertainty in the Plasma phospho-tau217 Gray Zone

**Authors:** Harthik Parankusham, Casey Vanderlip, Colin Birkenbihl, Eashwar Krishna, Chizobam Ugboaja, Andrew Budson, Brandon Frank, and for the Alzheimer's Disease Neuroimaging Initiative

---

## Supplementary Table S1. Complete Model Performance Metrics

GRAD model performance in the ADNI development cohort (N = 320) under leave-one-out cross-validation. 95% confidence intervals derived from 2,000 bootstrap resamples of the LOOCV predictions. Discrimination, calibration, and likelihood ratio metrics reported per STARD 2015 diagnostic accuracy guidelines.

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.857 | 0.813 – 0.897 |
| AUPRC | 0.827 | 0.757 – 0.897 |
| Accuracy | 80.6% | 76.2% – 84.7% |
| Sensitivity | 78.7% | 72.3% – 84.8% |
| Specificity | 82.4% | 76.4% – 88.0% |
| PPV | 80.8% | 74.5% – 86.7% |
| NPV | 80.5% | 74.6% – 86.2% |
| LR+ | 4.48 | 3.28 – 6.50 |
| LR− | 0.26 | 0.18 – 0.35 |
| Brier Score | 0.148 | — |

Abbreviations: AUC, area under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve; PPV, positive predictive value; NPV, negative predictive value; LR+, positive likelihood ratio; LR−, negative likelihood ratio.

---

## Supplementary Table S2. Cost-Impact Simulation by Diagnostic Method

Projected total cost, per-capita cost, and Aβ-PET utilization for four diagnostic strategies applied to a hypothetical cohort of 10,000 patients. Unit costs reflect 2024 U.S. Medicare reimbursement schedules: $3,000 per amyloid PET scan, $350 per single-analyte p-Tau217 test, $600 per multi-analyte plasma panel (p-Tau217, GFAP, Aβ42/40). Resolution rates derived from the LOOCV (ADNI) and external validation (A4/LEARN) results reported in Sections 3.2, 3.3, and 3.6 of the main text.

| Strategy | Total Cost | Per Capita Cost | PET Scans Required | Savings vs. Universal PET |
|---|---|---|---|---|
| Universal PET | $30,000,000 | $3,000 | 10,000 | Reference |
| p-Tau217 + PET (Gray Zone) | $16,820,000 | $1,682 | 4,440 | 44% |
| Staged Algorithm (GRAD) | $9,993,000 | $999 | 1,331 | 67% |
| Staged Algorithm + MRI | $8,661,000 | $866 | 887 | 71% |

---
