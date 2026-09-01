# GRAD — analysis spine and frozen values

**Frozen 2026-08-05, revised same day (impaired-restricted primary analysis).** Every number here was produced by the scripts named
against it and survives adversarial checking. Superseded, null and exploratory
analyses are deliberately excluded — this is the working skeleton of the paper,
not a record of everything attempted.

Regenerate the audit with `python3 scripts/GRAD_number_audit.py`. Do not edit
numbers in the manuscript by hand.

---

## 1. The problem

Plasma p-tau217 is the most accurate single blood marker for cerebral amyloid,
but a two-cutoff workflow leaves a substantial **indeterminate band** where it
cannot support a confident call. Current practice resolves that band with
amyloid PET.

**The band is largest exactly where it matters least to have it.** Under a common
cutoff rule (95% sensitivity / 95% specificity), the indeterminate fraction rises
as disease severity falls:

| population | this study | Giacomucci et al. (published) |
|---|---|---|
| Cognitively unimpaired | **53.1%** (ADNI CN) · **46.2%** (A4) | 22.6% (SCD) |
| MCI | 42.7% | 21.6% |
| Dementia | 35.7% | 6.8% |

The gradient replicates in two independent datasets. Our absolute values sit
higher because the gray-zone fraction under this rule is a deterministic function
of the marker's AUC — simulation gives 52% at AUC 0.85 and 13% at 0.96 — and
p-tau217 achieves **0.87** in our cohorts against **0.95** in a cohort that is 34%
AD dementia. *The size of the gray zone is a property of the population under
test, not of the threshold rule.*

`scripts/GRAD_fig1_grayzone_by_population.py` · `GRAD_fig1_grayzone_stacked.py`

---

## 2. Cohorts

| | ADNI (development) | A4 + LEARN (external validation) |
|---|---|---|
| Full cohort | 320 | 1,644 |
| **Analysis cohort** | **145 (MCI + dementia)** | **1,644 (all CU)** |
| Amyloid-positive | 58.6% | 69.6% |
| Ground truth | florbetapir SUVR > 1.11 | Centiloid ≥ 20 |
| p-tau217 assay | Lumipulse (Fujirebio) / Janssen | Lilly Research Labs MSD |
| GFAP, Aβ42/40, NfL | ADNI platform | Roche Elecsys (subset) |

**Why ADNI is restricted to cognitively impaired participants.** Blood-based
biomarkers are approved for clinical use in cognitively impaired patients, so a
model intended for clinical deployment should be developed in that population.
ADNI's 175 cognitively normal participants function as research controls rather
than as a population in which the test would be ordered. A4 + LEARN is retained
in full: its participants are cognitively unimpaired but screened for preclinical
AD, which is a distinct and legitimate deployment context (prevention-trial
enrichment).

**This restriction was adopted for clinical alignment, not for performance.**
It must be reported as such. Training on the impaired subset alone does not
improve external performance in A4 (0.794 vs 0.806 for Stage 2 trained on all
gray-zone cases); the gain in the ADNI figures comes from evaluating in a
population where p-tau217 separates better, not from a better model.

**Harmonisation.** Reference-anchored Z-scores, `Z = (log1p(x) − μ_ref)/σ_ref`,
reference = cognitively unimpaired amyloid-negative participants **within each
cohort** and **per assay**. Worth +0.077 AUC in ADNI (0.793 raw pooled → 0.870
anchored) because pooling Lumipulse and Janssen on raw concentrations scrambles
ranks across platforms.

Aβ42/40 enters as a **raw log ratio, not anchored** — it is a ratio of two
analytes from the same platform and sample, so scale factors cancel within it;
anchoring adds reference-subset noise without removing bias (standalone AUC
0.726 raw vs 0.652 anchored).

---

## 3. The model

**Stage 1 — Gatekeeper.** Univariate logistic regression on p-tau217. Resolve
P < 0.25 and P > 0.75; route 0.25 ≤ P ≤ 0.75 to Stage 2. Because logistic
regression is monotone in a single predictor, Stage 1's ranking *is* p-tau217's
ranking — no model class can improve it (verified: linear, spline, forest and
boosting are identical or worse).

**Stage 2 — Reflex.** L2-regularised logistic regression (C = 0.1, balanced class
weights), seven features:

```
pTau217_Z · tau_ab42_diff · GFAP_Z · AGE · APOE4_carrier
gfap_tau_interaction · AB4240_log
```

Regularised logistic chosen a priori on parsimony — seven predictors, small
training set — and publishable as an equation rather than a binary artefact.

**Validation.** ADNI by leave-one-out CV with harmonisation recomputed per fold.
A4 scored once by the ADNI-fitted model. Missing Stage 2 analytes imputed with
**training-set medians**, never the test cohort's own.

---

## 4. Results

### 4.1 Primary results

**ADNI (MCI + dementia, n = 145)** — Stage 1 resolves 97 (66.9%), gray zone 48 (33.1%)

| | n | AUC | 95% CI | P |
|---|---|---|---|---|
| Stage 1, on cases it resolves | 97 | **0.948** | — | — |
| Stage 2 — p-tau217 alone | 48 | 0.705 | — | — |
| **Stage 2 — GRAD** | 48 | **0.781** | 0.633–0.915 | 0.315 |
| **Full pipeline** | 145 | **0.929** | 0.875–0.972 | — |

**A4 + LEARN (n = 1,644)** — Stage 1 resolves 1,014 (61.7%), gray zone 630 (38.3%)

| | n | AUC | 95% CI | P |
|---|---|---|---|---|
| Stage 1, on cases it resolves | 1,014 | **0.904** | — | — |
| Stage 2 — p-tau217 alone | 630 | 0.719 | — | — |
| **Stage 2 — GRAD** | 630 | **0.793** | 0.750–0.832 | **0.001** |
| **Full pipeline** | 1,644 | **0.878** | 0.860–0.894 | — |

**The headline is the replicated Stage 2 gain: +0.076 in ADNI and +0.074 in A4.**
Nearly identical magnitude in independent cohorts, on different platforms, in
different populations. ADNI's gray zone (n = 48) is underpowered — the claim
rests on A4, where P = 0.001.

### 4.2 The gray zone is concentrated in early disease

Under a common 95%-sensitivity / 95%-specificity cutoff rule:

| population | this study | Giacomucci et al. (published) |
|---|---|---|
| Cognitively unimpaired | 53.1% (ADNI CN) · 46.2% (A4) | 22.6% (SCD) |
| MCI | 42.7% | 21.6% |
| Dementia | 35.7% | 6.8% |

Gray-zone size under this rule is a deterministic function of the marker's AUC
(simulation: 52% at AUC 0.85, 13% at 0.96). p-tau217 reaches **0.94 in ADNI
dementia, 0.91 in MCI, 0.81 in CN** — so the indeterminate band is a property of
the population, not the threshold rule. Prior estimates from cohorts enriched for
dementia understate it in the populations where plasma triage is deployed.

### 4.3 Prior gray-zone methods do not transport to these populations

**The comparison, recomputed under the impaired-trained design.** A4 gray zone
n = 630, of which **256 have plasma p-tau181** (55.5% amyloid-positive). The
p-tau181 cutoff is Youden-optimal derived **in A4 itself**, which favours the
comparator; GRAD was trained on ADNI and applied blind.

| method | AUC | 95% CI | resolved | correct | wrong | to PET |
|---|---|---|---|---|---|---|
| p-tau217 alone | 0.743 | 0.678–0.802 | 64.1% | 44.1% | 19.9% | 35.9% |
| **p-tau181 integration** | **0.597** | 0.532–0.664 | 100% | 59.8% | **40.2%** | 0% |
| **GRAD Stage 2** | **0.799** | 0.741–0.852 | 80.5% | 58.6% | 21.9% | **19.5%** |

DeLong: **GRAD vs p-tau181 P = 4×10⁻⁶**; GRAD vs p-tau217 P = 0.064.

**Two distinct failures of the p-tau181 approach.**

*It does not discriminate here.* AUC **0.597** in a preclinical gray zone, against
0.87 reported in a cohort that is 34% AD dementia. p-tau181 is another tau
species, correlated with the marker that has already failed. The markers that
help lie on an orthogonal axis — Spearman with p-tau217 inside the gray zone:
Aβ42/40 **−0.25**, GFAP **+0.16**, APOE ε4 **+0.27**.

*It never abstains.* A binary p-tau181 rule classifies 100% of the gray zone and
therefore reaches a comparable correct rate (59.8% vs 58.6%) only by issuing a
**wrong answer to 40.2% of patients with no scan ordered**. GRAD reaches the same
correct rate at 21.9% wrong and flags 19.5% for confirmatory PET. Giacomucci et
al. report the same pattern in their own data: overall accuracy *fell* 2.3% when
p-tau181 was added.

**Scope limitation to state explicitly.** ADNI's plasma p-tau181 (Gothenburg
Simoa, collected ~2010–2011) overlaps the analysis cohort in **4 of 320
participants (1.2%)**, so this comparison is possible only in A4. Report it as a
single-cohort head-to-head.

### 4.3b Why these are the populations that matter

The two cohorts map onto the two settings where amyloid confirmation currently
gates a decision:

- **A4 + LEARN — preclinical AD, prevention-trial enrichment.** Aβ-PET is the
  acknowledged pre-screening bottleneck. GRAD resolves 61.7% at Stage 1 and a
  further 80.5% of the remainder, leaving 19.5% of the gray zone needing a scan.
- **ADNI MCI + dementia — DMT eligibility.** The population in which blood-based
  biomarkers are approved for clinical use, and in which lecanemab and donanemab
  Appropriate Use Recommendations require biomarker-confirmed amyloid positivity.

The indeterminate band persists in both — **38.3% of A4 and 33.1% of ADNI's
impaired cohort** — and prior gray-zone methods were developed in cohorts where
it is far smaller (18.1% overall in Giacomucci et al., 6.8% in their dementia
subgroup) and where p-tau217 itself reaches 0.95. That is why they do not
transport, and why the problem is largest exactly where the treatment decisions
now sit.

### 4.4 What Stage 2 is doing

Odds ratios per SD (ADNI gray zone): p-tau217 **1.94** (1.05–3.60, P = .035),
APOE ε4 **1.81** (1.20–2.73, P = .005), tau-Aβ42/40 1.74 (0.68–4.44), GFAP 1.14,
GFAP × p-tau217 **1.06** (P = .90), age 0.83.

Discrimination rests on residual p-tau217 signal, **APOE ε4 genotype**, and the
**amyloid axis (Aβ42/40)**. The GFAP × p-tau217 interaction contributes **0.001
AUC** on ablation and should not be defended on pathophysiological grounds.
Every prediction is a printable sum of contributions.

### 4.5 Calibration, continuous validity, PET burden

**Calibration (A4):** Brier 0.148, slope **1.07**, intercept **+0.72**; mean
predicted 59.1% vs observed 69.6%. Discrimination and routing transfer unchanged;
only the intercept needs recalibration to local prevalence — monotone, so no AUC
or routing decision changes.

**Continuous validity:** Spearman **ρ = 0.728** (P < .001) between GRAD
probability and Centiloid across all 1,644 A4 participants.

**Outcome without PET**, same 0.40–0.60 indeterminate band on both scores:

| | resolved correctly | resolved wrongly | still needs PET |
|---|---|---|---|
| A4 p-tau217 | 43.2% | 23.1% | 33.7% |
| **A4 GRAD Stage 2** | **56.1%** | **18.3%** | **25.6%** |

All three improve simultaneously. Paired reclassification in A4: **109 corrected
vs 32 broken** (McNemar **P < .001**); ADNI 18 vs 14 (P = .60).

---

## 5. Integrity of the analysis

- **No test-set leakage.** A4 was never used for training, feature selection,
  model-class selection or hyperparameter choice. Model class was fixed a priori
  on parsimony; the gray-zone band (0.25/0.75) and the indeterminate band
  (0.40/0.60) were both pre-specified.
- **Harmonisation inside folds.** Reference parameters recomputed within each
  LOOCV fold; test-set values imputed with training medians.
- **Paired tests where paired.** DeLong for two ROC curves on the same
  participants; McNemar for paired reclassification. AUC intervals are percentile
  bootstrap over participants (2,000 resamples).
- **Comparisons matched.** Sensitivity and specificity are only compared at
  matched operating points; resolution rates only under an identical band applied
  to both scores.

---

## 6. Outstanding before submission

1. **Regenerate the cost simulation.** Current figures use the previous Stage 2's
   residual-referral rate. Rerun `GRAD_fig_cost.py` after the Stage 2 switch, with
   reflex billing (the multi-analyte add-on charged only to the ~44% reaching
   Stage 2, not to all 10,000) and a cost-per-correct-diagnosis column.
2. **Regenerate every Stage 2 number and panel** against the seven-feature
   logistic model.
3. **Methods text** to state: model class chosen a priori; imputation actually
   performed (§2.5 currently says none was); Aβ42/40 used raw and why; A4 panel
   availability differs by study arm (81.2% LEARN vs 44.5% A4).
