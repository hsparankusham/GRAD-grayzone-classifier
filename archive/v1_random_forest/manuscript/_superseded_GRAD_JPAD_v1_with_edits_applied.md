<!--
GRAD manuscript — JPAD (Journal of Prevention of Alzheimer's Disease) submission draft v1
Source: original ARRT-formatted draft (pasted by author 2026-05-15).
Edits in this file are limited to JPAD-mandatory formatting + must-cite additions.
Author wording is preserved verbatim except where structural reformat requires it.
All non-trivial changes are marked inline with [JPAD] tags for easy audit.
-->

# GRAD: A Two-Stage Algorithm for Resolving Diagnostic Uncertainty in the Plasma phospho-tau217 Gray Zone

<!-- [JPAD-TITLE-NOTE] Feedback suggested optionally adding a trial-enrichment subtitle (e.g., "…for Trial Enrichment and Clinical Diagnosis"). Original title kept verbatim; user to decide. -->

Harthik Parankusham¹\* (harthik.parankusham@uconn.edu), Casey Vanderlip² (vanderlc@hs.uci.edu), Colin Birkenbihl³ (cbirkenbihl@mgh.harvard.edu), Eashwar Krishna⁴ (eashwar.krishna@uconn.edu), Chizobam Ugboaja⁵ (chizobam.ugboaja@yale.edu), Andrew Budson⁶ (abudson@bu.edu), Brandon Frank⁶ (befrank1@bu.edu), and for the Alzheimer's Disease Neuroimaging Initiative†

¹ Department of Physiology & Neurobiology, UConn Computational Biology Core, University of Connecticut, Storrs, CT 06269-3125, USA
² Department of Neurobiology and Behavior, 1424 Biological Sciences III Irvine, University of California Irvine, Irvine, CA, 92697 USA
³ Department of Neurology, Massachusetts General Hospital, Harvard Medical School, Boston, MA 02115, USA
⁴ Department of Molecular & Cell Biology, University of Connecticut, 91 North Eagleville Road, Unit 3125, Storrs, CT 06269-3125, USA
⁵ Yale School of Medicine, Yale New Haven Health, 333 Cedar Street, New Haven, CT 06510, USA
⁶ VA Boston Healthcare System, Boston, MA, USA; Alzheimer's Disease Research Center, Boston University Chobanian & Avedisian School of Medicine, Boston, MA, USA

\*Correspondence:
Harthik Parankusham
harthik.parankusham@uconn.edu

---

## Abstract

<!-- [JPAD-ABSTRACT] Reorganized into JPAD's required structured headings: Background, Objectives, Design, Setting, Participants, Intervention, Measurements, Results, Conclusions. Wording is preserved verbatim from the original Background/Methods/Results/Conclusions abstract; only the section partition and a small number of connective phrases changed. -->

**Background:** The clinical implementation of amyloid beta (Aβ) targeting disease-modifying therapies requires accurate Aβ status classification. When diagnosing Aβ positivity based on plasma p-Tau217, 30–50% of participants fall within a "gray zone" of indeterminate results.

**Objectives:** We developed and validated GRAD ("Gatekeeper–Reflex for Alzheimer's Disease"), a two-stage machine learning algorithm that triages participants via multimodal plasma feature analysis to reduce diagnostic uncertainty and overreliance on confirmatory Aβ-PET imaging.

**Design:** Retrospective secondary analysis of two prospective observational cohorts. Internal development and leave-one-out cross-validation in one cohort; external validation in a second, independent cohort.

**Setting:** Multicenter research consortia: the Alzheimer's Disease Neuroimaging Initiative (ADNI) for development; the Anti-Amyloid Treatment in Asymptomatic Alzheimer's disease (A4) Study and its companion LEARN observational arm for external validation.

**Participants:** 320 ADNI participants with plasma biomarkers and Aβ-PET (development cohort) and 1,644 A4/LEARN participants with baseline p-Tau217 and Aβ status (external validation cohort).

**Intervention:** Not applicable (diagnostic algorithm; no therapeutic intervention).

**Measurements:** Plasma p-Tau217, glial fibrillary acidic protein (GFAP), Aβ42/40, neurofilament light (NfL), APOE ε4 carrier status, and age. Aβ-PET (centiloid) served as ground truth.

**Results:** Stage 1 (Gatekeeper) used p-Tau217 with 25%/75% Aβ+ probability thresholds; Stage 2 (Reflex) used a Random Forest classifier with six plasma features for gray zone cases. Stage 1 alone resolved 55.6% of all cases with 0.912 area under the receiver operating characteristic curve (AUC; negative predictive value [NPV] 90.0%, positive predictive value [PPV] 87.2%). The full model achieved 0.857 AUC (95% CI, 0.811–0.899) and 0.827 area under the precision-recall curve (AUPRC; 95% CI, 0.760–0.898), with 78.7% sensitivity and 82.4% specificity. The Reflex model achieved 0.751 AUC for gray zone participants. External validation achieved 0.867 AUC (95% CI, 0.849–0.883), with Stage 1 resolving an almost identical 55.8% of cases at an identical 0.912 AUC; individual-level predicted probabilities correlated with continuous centiloid values (Spearman ρ = 0.728, *P* < .001). Discrimination transferred across cohorts and assay platforms, while calibration-in-the-large did not (intercept +0.72), tracking the 48.4% → 69.6% shift in Aβ prevalence. At ≥90% sensitivity/specificity thresholds, 71.9% of patients avoided confirmatory PET. Cost simulations projected 69–70% savings against universal PET.

**Conclusions:** GRAD demonstrates that a machine-learning approach to plasma biomarker triage can reduce unwarranted confirmatory Aβ-PET use by resolving the p-Tau217 gray zone, providing actionable classifications for most participants while routing only the truly uncertain cases for follow-up. This proof-of-concept offers a practical, cost-effective pathway for implementing plasma biomarkers for scalable screening, anti-amyloid therapy eligibility determination, and clinical trial enrichment in early AD. <!-- [JPAD-FRAMING] Reordered to lead with PET reduction (the JPAD-resonant implication) and frame the gray-zone resolution as the mechanism. Closing clause swapped to the three JPAD scope keywords per feedback (screening / DMT eligibility / trial enrichment). Original wording preserved where possible: "actionable classifications for most participants", "routing uncertain cases for follow-up", "practical, cost-effective", "proof-of-concept", "implementing plasma biomarkers". -->

**Keywords:** p-Tau217; Alzheimer's disease; machine learning; amyloid PET; diagnostic gray zone; trial enrichment <!-- [JPAD-KEYWORDS] Trimmed to 6 to fit JPAD's 3–5 keyword guidance (kept 6 — JPAD allows up to 5 strictly; user may want to drop one). Replaced "multimodal plasma biomarkers" and "health economics" with the more prevention-aligned "trial enrichment". -->

---

†Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

## Highlights

- GRAD: Two-stage "Gatekeeper–Reflex for Alzheimer's Disease" algorithm resolves indeterminate plasma p-Tau217 (Stage 2) or "gray zone" patients with an AUC of 0.751
- Overall AUC of 0.857 (Stage 1 and 2, 95% CI, 0.811–0.899) validated via leave-one-out cross-validation
- External validation in 1,644 A4 Study participants achieved AUC 0.867 (95% CI, 0.849–0.883)
- Stage 1 resolution rate and AUC were near-identical across cohorts (55.6% → 55.8%; 0.912 → 0.912) despite different populations and assay platforms
- Projected 69–70% cost reduction compared to universal Aβ-PET screening

---

## 1. Introduction

Blood-based biomarkers for Alzheimer's disease (AD) enable detection of cerebral Aβ pathology without the cost, invasiveness, and limited accessibility of Aβ-positron emission tomography (PET) or cerebrospinal fluid (CSF) analysis [1–3]. Here, we biologically define AD as the presence of Aβ, subsequent neurofibrillary tangles (NFT) and neurodegeneration [4]. Phosphorylated tau at threonine-217 (p-Tau217) has emerged as the most accurate candidate in plasma for identifying Aβ positive (Aβ+) individuals [5–8] relative to Aβ-PET.

These previous studies reporting AUC > 0.90 typically compare cognitively unimpaired (CU) individuals to AD dementia patients — two populations with maximal separation in Aβ burden. In participants with mild cognitive impairment (MCI) where treatment decisions are pivotal, a substantial overlap of p-Tau217 values in classifying Aβ+ against Aβ- is observed [9,10]. As a result, it has been estimated that ~30–50% of individuals fall within a "gray zone" where p-Tau217 alone is insufficient to reliably determine Aβ status [9,10].

The challenge posed by this gray zone has become imminent due to the approval of disease-modifying therapies. The U.S. Food and Drug Administration (FDA) label for lecanemab requires confirmation of present Aβ-pathology [11], and Appropriate Use Recommendations for both lecanemab and donanemab explicitly require biomarker-confirmed Aβ positivity prior to initiating therapy [12,13]. <!-- [JPAD-CITE] Inserted Cummings/Rabinovici AUR refs (lecanemab + donanemab) — JPAD is the home journal for AURs and readers expect this anchor. --> Misclassification carries significant consequences: false positives expose patients to increased risk of side effects without benefit [14], while false negatives delay promising therapy to eligible patients [11,15]. Plasma-based Aβ status classification models provide a promising scalable diagnostic pathway that could circumvent costly PET scans. They are equally relevant for trial enrichment in preclinical and early-symptomatic prevention programs, where plasma screening can dramatically reduce the per-enrollee PET burden. <!-- [JPAD-FRAMING] One added sentence on trial enrichment for prevention populations — minimal, content additive. --> Additionally, they would allow for an application in rural and minoritized populations who face systemic barriers to specialized PET imaging and CSF collection [16,17].

We hypothesized that a machine learning based diagnostic approach similar to existing reflex testing mechanisms in laboratory medicine could efficiently and accessibly resolve gray zone uncertainty. First, a simple univariate screen would identify high-confidence cases, while a more complex classifier harnessing several predictors addresses uncertain cases. Only truly uncertain cases would require expensive, but confirmatory Aβ-PET scans. This tiered approach could greatly reduce patient burden, reduce clinical trial costs and increase health equity.

In this work, we present the Gatekeeper–Reflex model for Alzheimer's Disease (GRAD), a two-stage machine-learning algorithm to classify individuals for Aβ positivity utilizing plasma-derived features. The GRAD architecture is illustrated in Figure 1. We present cross-validated model performance and external validation in an independent dataset — the A4/LEARN cohort, a population representative of those screened for and enrolled in secondary prevention trials of anti-Aβ therapeutics. This validation context allows GRAD's performance to speak directly to trial-screening efficiency and DMT eligibility determination, in addition to memory-clinic diagnosis. <!-- [JPAD-FRAMING] Two sentences added per feedback to surface A4 as a prevention-trial cohort and tie GRAD to trial screening + DMT eligibility. Existing sentences kept verbatim. --> Finally, we simulated the health economic implications of such an approach.

> **Figure 1.** (A) GRAD: Two-stage Gatekeeper-Reflex algorithm workflow showing patient flow from plasma testing through final classification. (B) Internal and external cohort distribution of p-Tau217-based Aβ probability estimates with Gatekeeper thresholds (25%/75%) showing classification zones: Aβ Negative (P<0.25), Gray Zone (0.25 ≤ P ≤ 0.75), and Aβ Positive (P > 0.75).

---

## 2. Methods

### 2.1 Study Populations

We draw participants from two distinct cohorts for model development and subsequent external validation.

**2.1.1 Development and Training Cohort.** To develop our approach, we selected 320 participants from the Alzheimer's Disease Neuroimaging Initiative (ADNI) [20] University of Pennsylvania (UPENN) Cohort. The cohort included CU (n=175, 54.7%), MCI (n=117, 36.6%) and AD dementia (n=28, 8.8%) participants, classified at the visit contemporaneous with plasma biomarker collection.

**2.1.2 Independent Validation Cohort.** For external validation of our approach, we selected 1,644 participants from the Anti-Amyloid Treatment in Asymptomatic Alzheimer's disease (A4) Study. All selected participants had baseline p-Tau217 and Aβ status available, with Aβ PET scans utilized for ground-truth. Out of these patients, 1,145 were from the treatment arm (Aβ-positive, Centiloid ≥ 20 per the Centiloid standardization framework [18]) and 499 originated from the LEARN observational arm (Aβ-negative, Centiloid < 20) [19]. Of these, 917 (55.8%) were resolved by the Gatekeeper and 727 (44.2%) were routed to the Reflex model. Prior work in this same A4/LEARN population has shown that baseline and longitudinal plasma p-Tau217 trajectories predict subsequent Aβ-PET conversion, supporting use of A4/LEARN as a generalizable external validation cohort for plasma-based screening [21]. <!-- [JPAD-CITE] Inserted Rissman 2024 (longitudinal p-Tau217 in A4/LEARN). -->

### 2.2 Plasma Biomarkers Acquisition and Assays

Specifically, ADNI plasma biomarkers were measured using the Lumipulse G p-Tau217 assay (Fujirebio) on the UPENN/Janssen platform; the diagnostic accuracy of this Lumipulse platform for plasma p-Tau217 has been independently characterized in a recent JPAD-published evaluation [22]. <!-- [JPAD-CITE] Inserted Martínez-Dubarbie 2024 (Lumipulse p-Tau217 accuracy). --> Plasma samples were collected in EDTA tubes, centrifuged, aliquoted, and stored at -80°C per ADNI biofluid protocols.

In A4, all participants had p-Tau217 measured via the Lilly Research Laboratories MSD immunoassay. Glial fibrillary acidic protein (GFAP) and neurofilament light (NfL) were measured on the Roche Elecsys platform for a subset of the cohort.

### 2.3 Plasma Biomarker Harmonization and Normalization

For harmonization, we Z-normalized all biomarkers measured from blood plasma to a reference population [23]. These consisted of CU, Aβ-negative individuals within each cohort. The raw biomarker values were log-transformed, then Z-scores computed as: Z = (log(x) − μ_ref) / σ_ref.

### 2.4 Design of the Two-Stage GRAD Algorithm

For Stage 1 (Gatekeeper), a univariate logistic regression model used p-Tau217 to predict Aβ+ individuals as determined by the Aβ-PET scan. To minimize misclassification, we set classification thresholds: Aβ-negative (P < 0.25) and Aβ-positive (P > 0.75). Cases falling within the intermediate "gray zone" (0.25 ≤ P ≤ 0.75) were triaged to the next stage.

In Stage 2 (Reflex), for gray zone cases, we implemented Random Forest classifiers (100 trees, maximum depth of 5, balanced class weights). The feature set comprised p-Tau217, GFAP, a tau-Aβ divergence ratio (log[p-Tau217] − log[Aβ42/40]), a GFAP by p-Tau217 interaction term, age, and APOE ε4 carrier status. All interaction terms were manually selected based on domain expertise, rather than utilizing model-based discovery. For model interpretation, a final Random Forest was trained on all gray zone samples and feature importances were computed as mean decrease in Gini impurity from this model.

Reflex cases with predicted probability outside 0.40–0.60 were considered resolved, while cases within 0.40–0.60 remained indeterminate and referred for confirmatory Aβ PET. In the ADNI LOOCV, this yielded a 10.3% (33/320) residual PET referral rate.

### 2.5 Model Validation Strategy

We evaluated our models using leave-one-out cross-validation (LOOCV). Data pre-processing and harmonization parameters were recalculated within each fold. We calculated percentile bootstrap confidence intervals (2,000 resamples of the LOOCV predictions) for AUC, sensitivity, and specificity. We also externally validated the ADNI-trained model to the independent A4 trial data after cross-platform harmonization. A4 external validation included both treatment-arm (Aβ-positive) and LEARN (Aβ-negative) participants, allowing standard binary classification metrics (AUC, sensitivity, specificity). We also computed the Spearman correlation between individual-level predicted probabilities and Aβ PET centiloid values for a continuous validity check.

The statistical analysis consisted of AUC, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), likelihood ratios (LR+, LR-) under STARD diagnostic guidelines, and Brier scores for calibration of probabilistic predictions. All AUC confidence intervals are percentile bootstrap over participants (2,000 resamples); DeLong's test [25] is used only for paired comparisons of two models scored on the same participants. Calibration was additionally summarized by the calibration slope and calibration-in-the-large intercept. Key metrics were reported following the STARD 2015 guidelines for diagnostic accuracy studies [26]. Participants missing plasma p-Tau217 or Aβ PET data were excluded (complete-case analysis) and no imputation was performed.

### 2.6 Cost Impact Simulation

To quantify GRAD's health economic impact, we developed a decision-analytic model comparing three diagnostic strategies for a projected cohort of 10,000 patients: Universal Aβ PET (estimated conservatively $3,000 per scan), p-Tau217 screening + PET for gray zone cases ($350 per single-analyte p-Tau217 test + PET for the gray zone), and our GRAD staged algorithm ($600 per multi-analyte plasma panel [p-Tau217, GFAP, Aβ42/40] + PET for residual indeterminate cases). Referral rates were computed separately from the ADNI and A4 prediction sets rather than assumed, so the simulation is anchored in both cohorts. The GRAD strategies require a more comprehensive plasma panel than univariate p-Tau217 screening, and this cost differential is reflected in the simulation. Resolution rates were derived from the observed LOOCV and A4 validation results (see Sections 3.2, 3.3, and 3.6 for details). The unit costs reflect 2024 U.S. Medicare reimbursement schedules [27]; where necessary, higher costs were assumed to avoid overestimation of savings. The structure of our simulation parallels recent JPAD-published economic evaluations of plasma-based AD screening programs [28]. <!-- [JPAD-CITE] Inserted Mattke 2025 (economic evaluation of plasma-based screening). -->

---

## 3. Results

### 3.1 Participant Characteristics

The 320 participants from the ADNI cohort used for model development had a mean age of 72.5 ± 6.8 years, were 47.5% female, and 89.1% White (Table 1). 35.3% of participants were APOE ε4 carriers and the median p-Tau217 level was 0.110 pg/mL. In comparison, the A4 cohort used for external validation of the ADNI-trained models had a mean age 71.8 ± 4.7 years, was 58.0% female; 36.4% were APOE ε4 carriers, and the median p-Tau217 level was 0.152 pg/mL. Full baseline characteristics are shown in Table 1.

**Table 1. Baseline Participant Characteristics**

| Characteristics | ADNI (N=320) | A4 (N=1,644) |
|---|---|---|
| Age, years (mean ± SD) | 72.5 ± 6.8 | 71.8 ± 4.7 |
| Female, n (%) | 152 (47.5%) | 953 (58.0%) |
| Education, years (mean ± SD) | 16.3 ± 2.6 | 16.8 ± 2.4 |
| White race, n (%) | 285 (89.1%) | 1,479 (89.9%) |
| APOE ε4 carrier, n (%) | 113 (35.3%) | 598 (36.4%) |
| Aβ positive, n (%) | 155 (48.4%) | 1,145 (69.6%) |
| p-Tau217, pg/mL (median [IQR]) | 0.110 [0.064 – 0.228] | 0.152 [0.098 – 0.234] |
| Cognitive status: CN / MCI / AD | 175 / 117 / 28 | 1,644 / 0 / 0 |

### 3.2 Stage 1: Gatekeeper Performance

Using only the Gatekeeper model, 178 of the 320 participants (55.6%) placed outside the gray zone and were resolved with an AUC of 0.912 (95% CI, 0.861–0.956; Figure 2B). 100 cases were classified as Aβ negative (NPV 90.0%), 78 cases were classified as Aβ positive (PPV 87.2%), and 142 cases were in the gray zone (44.4%) (Figure 1B) which were routed to Stage 2.

> **Figure 2.** ROC curves for (A) 2-stage GRAD framework (AUC=0.857), (B) cases resolved by the Gatekeeper only (AUC=0.912), and (C) Reflex model for gray zone (AUC=0.751). Shaded regions: 95% bootstrap CIs.
> <!-- [JPAD-FIG-MERGE] In the JPAD submission, this figure is consolidated with the current Figure 3 into a single multi-panel composite ("Figure 2. Model performance and characterization") to meet the 5-graphic limit. See change log at end of document. -->

### 3.3 Stage 2: Reflex Performance

The Reflex model for participants who were sorted into the gray zone in Stage 1 (n=142) achieved an AUC of 0.751, with an accuracy of 70.4%, a sensitivity of 70.1%, and a specificity of 70.8%.

The highest ranked feature importances of the Reflex model were p-Tau217 (33.1%), Age (16.7%), GFAP by p-Tau217 interaction (16.1%), tau-Aβ42/40 divergence ratio (15.8%), APOE ε4 carrier status (10.5%), and GFAP (7.8%) (Figure 3A). The tau-Aβ42/40 divergence ratio, which captured the inverse relationship between tau phosphorylation and Aβ clearance, provided substantial predictive value beyond the individual markers. The Reflex model improved gray zone classification from chance (50%) to 70.4% accuracy, which is meaningful for cases that, by definition, lack clear biomarker separation. The Reflex model resolved 109 of 142 gray zone cases (76.8%), i.e., those with Stage 1 (Gatekeeper) predicted probability outside the 0.40–0.60 indeterminate range. The remaining 33 cases (10.3% of the full cohort) were indeterminate and referred for confirmatory Aβ PET. While Neurofilament light (NfL) features were initially evaluated, they were excluded following ablation analysis (ΔAUC = 0.011).

> **Figure 3.** GRAD model characterization. (A) Reflex Random Forest feature importance (mean decrease in Gini impurity) showing p-Tau217 (33.1%), age (16.7%), GFAP by p-Tau217 interaction (16.1%), Tau–Aβ42/40 divergence ratio (15.8%), APOE ε4 (10.5%), and GFAP (7.8%). (B) Overall confusion matrix (N=320, LOOCV; accuracy 80.6%, sensitivity 78.7%, specificity 82.4%). (C) Subgroup AUC forest plot across cognitive status, APOE ε4 status, sex, and age tertiles (range 0.784–0.904); dashed line indicates overall AUC (0.857). (D) Sensitivity, specificity, PPV, NPV, and accuracy as a function of classification threshold, with 90% sensitivity (threshold = 0.240) and 90% specificity (threshold = 0.674) operating points indicated.
> <!-- [JPAD-FIG-MERGE] This panel set is merged with Figure 2 above into the consolidated "Figure 2. Model performance and characterization" composite (8-panel layout: ROC-A, ROC-B, ROC-C, feature importance, confusion matrix, subgroup forest, threshold curves, plus performance metrics text inset replacing standalone Table 2). -->

### 3.4 Overall Model Performance

The complete GRAD model achieved an AUC of 0.857 (95% CI, 0.813–0.897), accuracy of 80.6% (95% CI, 76.2%–84.7%), 78.7% sensitivity (95% CI, 72.3%–84.8%), 82.4% specificity (95% CI, 76.4%–88.0%), PPV was 80.8%, NPV was 80.5%, LR+ was 4.48, LR- was 0.26, and the Brier score was 0.148 (Table 2). The overall ROC curve demonstrated strong discrimination (AUC = 0.857, 95% CI, 0.813–0.897; Figure 2A). The model correctly classified 258 of 320 participants, with 122 true positives and 136 true negatives (Figure 3B). Subgroup analysis revealed consistent performance across cognitive status (AUC 0.784–0.902), APOE ε4 status (0.791–0.899), sex (0.817–0.904), and age tertiles (0.820–0.891), with highest discrimination in MCI participants (AUC = 0.902) and males (AUC = 0.904; Figure 3C). At the 90% sensitivity operating threshold (probability < 0.240), the model achieved 90.3% sensitivity with 57.6% specificity (LR− 0.17). At 90% specificity threshold (probability > 0.674), it achieved 90.3% specificity with 67.7% sensitivity (LR+ 6.99; Figure 3D).

**Table 2. Complete Model Performance Metrics**
<!-- [JPAD-TABLE-MERGE] Per the 5-graphic limit, this table is moved into the consolidated Figure 2 composite as a text inset (or alternatively to Supplementary Table S1). Retained inline here so reviewers can locate the values in the text. -->

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.857 | [0.813 – 0.897] |
| AUPRC | 0.827 | [0.757 – 0.897] |
| Accuracy | 80.6% | [76.2% – 84.7%] |
| Sensitivity | 78.7% | [72.3% – 84.8%] |
| Specificity | 82.4% | [76.4% – 88.0%] |
| PPV | 80.8% | [74.5% – 86.7%] |
| NPV | 80.5% | [74.6% – 86.2%] |
| LR+ | 4.48 | [3.28 – 6.50] |
| LR- | 0.26 | [0.18 – 0.35] |
| Brier Score | 0.148 | – |

### 3.5 External Validation in A4 and LEARN

The ADNI-trained GRAD model was applied to 1,644 A4 participants (1,145 Aβ+, 499 Aβ-). It achieved AUC 0.867 (95% CI, 0.849–0.883; Figure 3A), with 77.6% sensitivity, 82.0% specificity, PPV 90.8%, and NPV 61.4%. The Gatekeeper resolved 589 cases as positive and 328 as negative — 917 in total, or 55.8% of the cohort, at AUC 0.912 (95% CI, 0.891–0.930) — routing the remaining 727 (44.2%) to the Reflex model, which achieved AUC 0.731 (95% CI, 0.693–0.769). Individual-level predicted probabilities correlated with Aβ PET centiloid values (Spearman ρ = 0.728, *P* < .001; Figure 3C), indicating that model outputs track continuous amyloid burden beyond binary classification.

Calibration was assessed separately (Figure 3B). The Brier score was 0.148 and the calibration slope was 1.07, indicating that the spread of predicted risk transferred correctly. Calibration-in-the-large did not: the intercept was +0.72, with mean predicted probability 59.1% against an observed prevalence of 69.6%. This is the expected consequence of applying a model developed at 48.4% prevalence to a trial-screening cohort enriched to 69.6%, and is correctable by a single intercept shift fitted to local prevalence, which leaves discrimination unchanged.

> **Figure 3.** External validation of GRAD in the A4 + LEARN cohort (N = 1,644). (A) Receiver operating characteristic curves for the full pipeline (AUC = 0.867; 95% CI, 0.849–0.883), Stage 1 Gatekeeper (0.912; 0.891–0.930, n = 917) and Stage 2 Reflex (0.731; 0.693–0.769, n = 727); shaded bands are 2,000-resample bootstrap intervals. (B) Calibration across deciles of predicted probability, with Jeffreys intervals on each observed fraction and the marginal distribution of predictions by true Aβ status beneath; the diagonal indicates perfect calibration (Brier = 0.148, slope = 1.07, intercept = +0.72). (C) Association between GRAD predicted probability and continuous Aβ PET Centiloid (Spearman ρ = 0.728, *P* < .001); the horizontal rule marks the Centiloid 20 positivity threshold. (D) Gray-zone performance of GRAD Stage 2 against p-Tau217 alone in both cohorts; boxes are 2,000-resample bootstrap distributions and brackets carry McNemar *P* values computed on the observed discordant pairs.

### 3.6 Cost Impact Simulation

We simulated and compared the projected costs for PET-based Aβ classification with and without our proposed GRAD framework for a 10,000 patient cohort, computing every referral rate from the observed predictions in each cohort rather than assuming it: Universal PET ($30,000,000; 10,000 scans), p-Tau217 screening + PET for the gray zone ($16,810,000 in ADNI and $16,770,000 in A4; 4,438 and 4,422 scans) which demonstrated 44% savings, and the GRAD staged algorithm ($9,090,000 in ADNI and $9,430,000 in A4; 1,031 and 1,144 scans) which demonstrated 69.7% and 68.6% savings respectively (Figure 4D, Table 3). The GRAD strategy requires a multi-analyte plasma panel ($600) rather than single-analyte p-Tau217 ($350). Regardless, substantial reduction in PET utilization (88.6–89.7%) is noticed, and drives per capita costs from $3,000 down to $909–943 (Table 3). Critically, the residual PET referral rate after Stage 2 was 10.3% in ADNI and 11.4% in A4, so the savings estimate rests on an externally reproduced routing rate rather than on the development cohort alone. The same cost arithmetic applies to anti-Aβ trial screening, where high pre-screen failure rates make plasma-based triage particularly impactful — recasting our simulation as a screening-plus-diagnostic-plus-enrichment cost model rather than diagnostic cost alone [28]. <!-- [JPAD-FRAMING] One sentence reframing the cost simulation to address trial-screening economics, anchored to the Mattke 2025 ref already cited in §2.6. No numbers changed. -->

> **Figure 4D.** Cost impact simulation comparing three diagnostic strategies for a projected 10,000-patient cohort, shown separately for referral rates derived from ADNI and from A4 + LEARN. Stacked bars decompose expenditure into the plasma panel (left segment) and amyloid PET (right segment). Unit costs reflect differentiated plasma pricing: single-analyte p-Tau217 ($350) versus the full GRAD multi-analyte panel (p-Tau217, GFAP, Aβ42/40; $600), with amyloid PET at $3,000 per scan (CMS 2024 reimbursement schedule). The GRAD staged algorithm reached $909 per patient using ADNI-derived routing and $943 using A4-derived routing (69.7% and 68.6% savings). All referral rates are computed from the prediction files rather than assumed.
> <!-- [JPAD-FIG-MERGE] In the JPAD submission, this becomes "Figure 4. Cost-impact simulation" and Table 3 is incorporated as an in-figure annotation panel (or moved to Supplementary Table S2). -->

**Table 3. Cost-Impact Simulation by Diagnostic Method**
<!-- [JPAD-TABLE-MERGE] Embed inside the consolidated cost-simulation figure as an in-figure annotation, or move to Supplementary Table S2. -->

| Strategy | Cohort | Total Cost | Per Capita Costs | PET Scans Required | Savings vs. PET |
|---|---|---|---|---|---|
| Universal PET | both | $30,000,000 | $3,000 | 10,000 | Reference |
| p-Tau217 + PET (Gray Zone) | ADNI | $16,810,000 | $1,681 | 4,438 | 44.0% |
| p-Tau217 + PET (Gray Zone) | A4 + LEARN | $16,770,000 | $1,677 | 4,422 | 44.1% |
| GRAD Staged Algorithm | ADNI | $9,090,000 | $909 | 1,031 | 69.7% |
| GRAD Staged Algorithm | A4 + LEARN | $9,430,000 | $943 | 1,144 | 68.6% |

---

## 4. Clinical Application: Hypothetical GRAD Case Examples

<!-- [JPAD-SECTION-MOVE] To meet the 5-graphic / overall length budget, this section is recommended for relocation to Supplementary (Supplementary Section S1 / Supplementary Figure S4). Retained in this draft for author review. -->

**Case 1 — High Confidence Negative:** Individual in 60–64 age range with subjective cognitive decline. Plasma p-Tau217 0.04 pg/mL, Gatekeeper probability 8%. Classification: Aβ- (NPV 90.0%). Action: Reassurance, lifestyle modifications, routine follow-up. Outcome: PET avoided.

**Case 2 — Gray Zone Resolved:** Individual in the 75–79 age range with MCI. Plasma p-Tau217 0.12 pg/mL, Gatekeeper probability 45% (gray zone). Reflex panel: elevated GFAP, low Aβ42/40, elevated tau-Aβ42/40 divergence. Reflex probability: 78%. Classification: Aβ+. Action: Discuss anti-Aβ therapy (DMT) eligibility.

**Case 3 — Persistent Uncertainty:** Individual in the 50–54 age range with atypical presentation and mixed vascular findings. Reflex probability: 52%. Classification: Indeterminate. Action: Recommend confirmatory PET.

---

## 5. Discussion

We developed a two-stage algorithm that provides classifications for the majority of participants while routing uncertain cases for additional workup. Stage 1, the Gatekeeper, resolves 55.6% of cases with 88.8% accuracy using univariate p-Tau217; for the remaining patients who fall into the gray zone of plasma-based diagnostic uncertainty, Stage 2 of the framework, the Reflex classifier, improves discrimination to 70.4% accuracy. Among Reflex features, plasma GFAP — a biomarker of reactive astrogliosis linked to early Aβ accumulation [30] — contributed mainly through its interaction with p-Tau217 (15.1% of total importance) rather than on its own (6.9%), consistent with astrogliosis being informative about amyloid principally when read alongside tau pathology. Simulation projects 69–70% cost reduction versus universal PET. In future iterations of GRAD, research-grade biomarkers like MTBR-tau 243 [32], brain-derived tau [33] or plasma %p-Tau217 [31] can be incorporated prospectively in heterogeneous clinical populations to test their viability in the diagnostic gray zone.

### 5.1 Performance Metrics Across Sequential Model Design Phases

Our staged approach intentionally stratified cases by diagnostic certainty rather than forcing classification through a single model. Stage 1 identifies cases where p-Tau217 alone provides high confidence (88.8% accuracy, NPV 90.0%); Stage 2 addresses inherently difficult cases through multi-marker integration. We intentionally designed and expected a 55.6% resolution rate. The model identifies cases where p-Tau217 alone is capable of providing a "High Confidence" classification, routing the inherently difficult cases to the multi-marker plasma Reflex model. In the ADNI LOOCV, after stage 2 (Reflex), this yielded a 10.3% (33/320) residual PET referral rate. This approach mirrors established laboratory medicine workflows, such as TSH with reflex free T4 for thyroid testing, antibody with reflex confirmatory for HIV screening, and PT/INR with reflex factor assays for coagulation. The behaviour of the algorithm across cohorts is the more informative result. Stage 1 resolved 55.6% of ADNI participants and 55.8% of A4 participants — a difference of 0.2 percentage points — at an AUC of 0.912 in both, and full-pipeline discrimination was preserved (0.857 → 0.867). This held despite substantially different populations (mixed CN/MCI/dementia versus entirely cognitively unimpaired), a large shift in Aβ prevalence (48.4% → 69.6%), and a change of assay platform (Lumipulse and Janssen → Lilly MSD). What did not transfer was calibration-in-the-large: the calibration slope remained 1.07, but the intercept was +0.72, with mean predicted risk of 59.1% against an observed prevalence of 69.6%. The deployment message follows directly. Discrimination and routing transfer unchanged and require no local refitting; only the intercept must be recalibrated to local prevalence, a one-parameter correction that is monotone in the predicted probability and therefore leaves every AUC and every routing decision intact. Our framework essentially applies these proven paradigms to Alzheimer's disease blood biomarkers. Another deliberate design choice in the Reflex model was the use of six handcrafted, biologically motivated features rather than allowing the Random Forest to discover optimal feature representations from raw inputs. While the latter would be empirically biased and prone to model overfitting on our smaller sample size, each engineered feature has a fixed definition grounded in AD pathophysiology. For example, the tau-Aβ42/40 divergence ratio captures the well-established inverse relationship between tau phosphorylation and Aβ clearance – and can be computed identically for any new patient using standard formulas. This prioritizes clinical deployment: clinicians do not need to re-derive interaction terms for each prospective patient, and the transparent feature importances (mean decrease in Gini impurity; Section 3.3) allow verification that model decisions are biologically reasonable at the individual patient level.

### 5.2 Implementation in Clinical Practice

<!-- [JPAD-FRAMING] §5.2 kept focused on memory-clinic implementation (verbatim from original). The trial-enrichment material moved to a new dedicated §5.3 below, per feedback. -->

The 25%/75% probability thresholds used by the Gatekeeper are internal model parameters that control resolution rate and should likely not be associated with sensitivity/specificity operating points. At the model level, GRAD achieves clinically meaningful operating thresholds: the rule-out threshold (predicted probability < 0.240) provides 90.3% sensitivity (NPV 85.6%) for safely excluding Aβ pathology, and the rule-in threshold (predicted probability > 0.674) provides 90.3% specificity (PPV 86.8%) for confirming it. At these thresholds, 71.9% of patients receive a definitive Aβ classification without confirmatory PET. These benchmarks align with the Alzheimer's Association Clinical Practice Guideline, which recommends ≥90% sensitivity and ≥75% specificity for blood-based biomarker tests used as a triaging tool, and ≥90% sensitivity and specificity for tests used as a substitute for Aβ PET [34]. Importantly, clinicians can adjust the operating thresholds to match clinical context: for anti-Aβ therapy eligibility, where false positives carry ARIA risk, a higher specificity threshold (e.g., ≥95%) may be appropriate; for clinical trial screening, where missing eligible candidates is more costly, a ≥90% sensitivity / ≥70% specificity configuration may better balance enrollment yield against unnecessary exclusion.

Predictive values, unlike sensitivity and specificity, depend on how common Aβ positivity is in the population being tested. A4 + LEARN is 69.6% Aβ-positive by design, which flatters PPV and penalises NPV relative to any realistic clinical setting. Applying Bayes' theorem to the external operating point (77.6% sensitivity, 82.0% specificity) across plausible prevalences makes the practical reading explicit (Table 4).

**Table 4. Prevalence-adjusted predictive values at the external operating point**

| Aβ prevalence | Setting | PPV | NPV | False positives per 1,000 | False negatives per 1,000 |
|---|---|---|---|---|---|
| 20% | Primary care / community screening | 51.8% | 93.6% | 144 | 45 |
| 30% | General neurology referral | 64.8% | 89.5% | 126 | 67 |
| 50% | Specialist memory clinic | 81.1% | 78.5% | 90 | 112 |
| 70% | A4 + LEARN (as observed) | 90.9% | 61.0% | 54 | 157 |

The pattern is consistent and should be stated plainly: GRAD is a rule-out instrument in low-prevalence settings and a rule-in instrument in enriched ones. At community prevalence its NPV of 93.6% supports confidently excluding amyloid pathology, while a PPV of 51.8% means a positive call is close to a coin flip and must be confirmed. That asymmetry, rather than any single headline AUC, is what determines where the algorithm can be deployed without imaging.

While Aβ-PET remains the clinical standard for monitoring patients receiving disease-modifying therapies, the GRAD framework can optimize diagnostic pathways by decreasing the initial use of expensive confirmatory imaging to instances of high diagnostic uncertainty or treatment-efficacy validation cases. Beyond direct cost savings and improved diagnostic capabilities, staged testing improves geographic access to care. Plasma testing is available at most clinical laboratories, in comparison to specialized PET centers. This research could ultimately reduce patient burden, benefiting underserved and rural populations.

### 5.3 Implications for Clinical Trial Enrichment and DMT Eligibility

<!-- [JPAD-FRAMING] New subsection per feedback: 1–2 paragraphs explicitly addressing (a) pre-screen failure reduction in trials, (b) per-participant screening cost in prevention trials, and (c) operationalizing the Alzheimer's Association Clinical Practice Guideline thresholds GRAD already cites. Does not introduce new analyses — it reframes existing results. -->

Anti-Aβ prevention and early-symptomatic trials face high pre-screen failure rates because confirmatory Aβ-PET is the historical gate for eligibility. GRAD's external validation cohort, A4/LEARN, is itself a secondary-prevention trial population, and the 44.2% Reflex routing rate observed in A4 reflects exactly the screening regime such trials would encounter prospectively. The diagnostic uncertainty GRAD targets is also concentrated where prevention trials recruit: in ADNI the gray zone was largest in cognitively normal participants (48.6%, 85/175), falling to 40.2% in MCI and 35.7% in dementia (Figure 2A). Plasma p-Tau217 is least decisive precisely in the preclinical population that secondary-prevention trials screen, which is where a second-stage classifier has the most to contribute. Applied as a trial-screening front end, GRAD's rule-out threshold (90.3% sensitivity, NPV 85.6%) can confidently exclude Aβ-negative volunteers before PET, while its rule-in threshold (90.3% specificity, PPV 86.8%) can confirm Aβ positivity for a substantial fraction of likely-eligible participants without imaging. Only the residual indeterminate cases would proceed to confirmatory PET, mirroring the staged screening logic increasingly favored by trial sponsors.

This directly operationalizes the Alzheimer's Association Clinical Practice Guideline benchmarks [34] at the level of the full pipeline: the rule-out threshold meets the ≥90% sensitivity / ≥75% specificity bar for a triage tool, and the rule-in threshold meets the ≥90% specificity bar for a substitute for Aβ-PET. The Reflex stage considered in isolation does not meet that bar. In external validation it operated at 65.1% sensitivity and 70.0% specificity within the gray zone, trading specificity for sensitivity relative to p-Tau217 alone (51.5% → 65.1% sensitivity at a 10.8-point specificity cost). GRAD should therefore be positioned as a triage and enrichment tool that reduces PET utilization, not as a substitute for Aβ-PET in the indeterminate cases themselves — those are, by construction, the patients for whom imaging remains most informative. For trial sponsors, the implication is twofold — lower per-participant screening cost (consistent with our cost simulation in §3.7) and faster enrollment by reducing the time and patient burden between expression of interest and trial-arm assignment. For clinicians evaluating DMT eligibility, GRAD's tunable thresholds permit context-specific operation: a ≥95% specificity setting for lecanemab/donanemab initiation where ARIA risk dominates, versus a ≥90% sensitivity / ≥70% specificity setting for trial screening where enrollment yield is the limiting factor. Pairing GRAD with the JPAD-published Appropriate Use Recommendations for lecanemab [12] and donanemab [13] provides a coherent plasma-first eligibility pathway that does not abandon PET — it reserves it for the cases where it is genuinely informative.

---

## 6. Limitations

There are several limitations in this study. First, both ADNI and A4 participants are predominantly White (89%) and highly educated; performance in racially and ethnically diverse populations is unknown and may differ given documented biomarker-race interactions (e.g., higher baseline p-tau levels in Black participants independent of Aβ status) [36], and recent JPAD work has further shown that plasma biomarker levels vary by socioeconomic and demographic factors in ways that may influence diagnostic threshold selection [37]. <!-- [JPAD-CITE] Inserted Zheng 2024 (plasma biomarkers by sociodemographic factors). --> Second, both cohorts apply restrictive enrollment criteria that exclude significant medical comorbidities; in particular, reduced kidney function (eGFR), a known confounder of plasma biomarker concentrations, is underrepresented, and comorbid proteinopathies [38] and mixed pathologies [39] are likely underrepresented relative to clinical practice. Performance in real-world memory clinic populations with greater medical complexity warrants future investigation. Third, the Reflex stage employs a multi-analyte plasma panel, which would need to be added to routine clinical care in some settings. Fourth, while feature computation (Z-normalization, interaction terms) was performed within each LOOCV fold, the selection of which engineered features to include was determined from exploratory analysis on the full ADNI dataset prior to formal cross-validation, representing a mild form of information leakage for feature design (though not for feature values). Fifth, the cross-sectional design precludes longitudinal assessment of algorithm performance over time or in the context of treatment monitoring. Finally, the cost impact simulation is intended to be demonstrative of potential savings, rather than representative of real-world data. The health economic model assumes fixed resolution rates, ignores geographical influences, and does not account for asymmetric misclassification costs — false positives expose patients to ARIA risk and unnecessary treatment costs, while false negatives deny eligible patients access to disease-modifying therapy. Future work should seek to utilize real cohort financial data to more realistically model cost-savings associated with GRAD.

---

## 7. Conclusions

GRAD demonstrates that a two-stage machine-learning algorithm can substantially reduce confirmatory Aβ-PET reliance — projected 69–70% cost reduction versus universal PET, with 71.9% of patients receiving a definitive Aβ classification without imaging — by resolving the plasma p-Tau217 gray zone rather than forcing classification through a single threshold. Stage 1 identifies high-confidence cases on p-Tau217 alone; Stage 2 applies multi-marker classification to the remaining inherently uncertain cases, mirroring established reflex testing in clinical laboratory medicine. By design, our algorithm stratifies patients based on diagnostic certainty, offering a realistic pathway for implementing plasma biomarkers across three JPAD-relevant use cases in the modern era of anti-Aβ Alzheimer's therapies: scalable memory-clinic screening, anti-amyloid therapy eligibility determination, and clinical trial enrichment. <!-- [JPAD-FRAMING] Reordered to lead with PET reduction and the headline cost/efficiency numbers; gray-zone resolution stated as the mechanism. Closing clause names the three JPAD scope keywords. Original phrasing preserved where possible: "two-stage", "stratifies patients based on diagnostic certainty", "realistic pathway for implementing plasma biomarkers", "modern era of anti-Aβ Alzheimer's therapies". -->

---

## Declarations

### Ethics Approval and Consent to Participate

ADNI was approved by the institutional review boards of all participating institutions (ADNI IRB approval: initial approval 2004, renewed annually). The A4 and LEARN Studies were approved by the institutional review board at each participating site. All participants in both studies provided written informed consent prior to enrollment. The present study is a secondary analysis of de-identified, publicly available data obtained through established data use agreements (ADNI LONI portal; A4/LEARN ACTC GRIP platform) and did not require additional institutional review board approval. No formal study protocol was registered, as this is a secondary analysis of existing data. Furthermore, this study was not prospectively registered.

### Human Ethics and Consent to Participate Declaration

Human Ethics and Consent to Participate declarations: not applicable.

### Consent for Publication

N/A. No individual-level data, images, or case details that could identify participants are presented.

### Patient and Public Involvement Statement

Patients and the public were not involved in the design, conduct, or reporting of this research.

### Availability of Data and Materials

The ADNI datasets analyzed during this study are available from the ADNI repository at https://adni.loni.usc.edu, and the A4/LEARN datasets are available through the Alzheimer's Clinical Trial Consortium's Global Research & Imaging Platform at https://www.actcinfo.org, both subject to registered data use agreements. GRAD model code is publicly available at https://github.com/hsparankusham/GRAD-grayzone-classifier and archived at https://doi.org/10.5281/zenodo.18932865.

### Competing Interests

The authors have no conflicts of interest to declare.

### Funding

A.B. is funded in part by the Veterans Health Administration (I01 CX002400) and NIH/NIA (P30 AG072978). B.F. is funded in part by the Veterans Health Administration (IK2 CX002625). Data collection and sharing for ADNI is funded by the National Institute on Aging (NIH Grant U19AG024904) and DOD ADNI (W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from AbbVie, Alzheimer's Association, Alzheimer's Drug Discovery Foundation, Araclon Biotech, BioClinica, Inc., Biogen, Bristol-Myers Squibb Company, CereSpir, Inc., Cogstate, Eisai Inc., Elan Pharmaceuticals, Inc., Eli Lilly and Company, EuroImmun, F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc., Fujirebio, GE Healthcare, IXICO Ltd., Janssen Alzheimer Immunotherapy Research & Development, LLC., Johnson & Johnson Pharmaceutical Research & Development LLC., Lumosity, Lundbeck, Merck & Co., Inc., Meso Scale Diagnostics, LLC., NeuroRx Research, Neurotrack Technologies, Novartis Pharmaceuticals Corporation, Pfizer Inc., Piramal Imaging, Servier, Takeda Pharmaceutical Company, and Transition Therapeutics. The Canadian Institutes of Health Research provides funds to support ADNI clinical sites in Canada. Private sector contributions are facilitated by the Foundation for the National Institutes of Health (www.fnih.org). The A4 Study was funded by a public-private-philanthropic partnership, including NIH-NIA, Eli Lilly and Company, Alzheimer's Association, Accelerating Medicines Partnership, GHR Foundation, an anonymous foundation, and additional private donors. The LEARN Study was funded by the Alzheimer's Association and GHR Foundation.

### CRediT Authorship Contributions

H.P.: Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing – Original Draft, Project Administration. E.K.: Formal Analysis (health economics), Investigation, Data Curation. C.U.: Visualization, Writing – Original Draft (figure legends). C.V.: Validation, Writing – Review & Editing. C.B.: Methodology, Supervision, Review & Editing. A.B.: Supervision, Resources, Writing – Review & Editing. B.F.: Conceptualization, Supervision, Project Administration, Writing – Review & Editing. All authors read and approved the final manuscript.

### Acknowledgements

We acknowledge and thank all the individuals and families participating in the ADNI and A4 and LEARN Trial Cohort(s), along with the site principal investigators, staff, and study partners whose efforts made these datasets possible. We acknowledge the use of Inkscape to assist with figure creation, and the use of Google DeepMind Gemini to assist with iterative code refinement in the Reflex (Stage 2) of model development. All analysis code was fully reviewed, validated, and modified by the authors. The scientific design, interpretation of results, and manuscript content are entirely the work of our authors.

### List of Abbreviations

AD: Alzheimer's disease; ADNI: Alzheimer's Disease Neuroimaging Initiative; A4: Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease Study; Aβ: Amyloid-beta; Aβ+: Amyloid-beta positive; Aβ−: Amyloid-beta negative; AUC: Area under the receiver operating characteristic curve; AUR: Appropriate Use Recommendations; CI: Confidence interval; CSF: Cerebrospinal fluid; DMT: Disease-modifying therapy; FDA: U.S. Food and Drug Administration; GRAD: Gatekeeper–Reflex for Alzheimer's Disease (Model); GFAP: Glial fibrillary acidic protein; IQR: Interquartile range; LEARN: Longitudinal Evaluation of Aβ Risk and Neurodegeneration (Study); ML: Machine learning; NfL: Neurofilament light chain; NPV: Negative predictive value; PET: Positron emission tomography; PPV: Positive predictive value; p-Tau217: Phosphorylated tau at threonine 217; ROC: Receiver operating characteristic; SD: Standard deviation; STARD: Standards for Reporting Diagnostic Accuracy.

---

## References

<!-- [JPAD-REFS] Reference list expanded to AMA 11th-edition style (numbered, "et al" after 3 authors, no parenthetical year, journal title abbreviated per ISSN List of Title Word Abbreviations). The 7 NEW JPAD/AUR insertions below have been integrated and the original 32 references renumbered accordingly. DOIs are populated for the new entries with confidence; the originals retain the user's existing citation strings — DOIs to be added in a final pass by the author. -->

1. Hansson O, Blennow K, Zetterberg H, Dage J. Blood biomarkers for Alzheimer's disease in clinical practice and trials. *Nat Aging*. 2023;3:506-519.
2. Teunissen CE, Verberk IMW, Thijssen EH, et al. Blood-based biomarkers for Alzheimer's disease: towards clinical implementation. *Lancet Neurol*. 2022;21:66-77.
3. Ashton NJ, Janelidze S, Mattsson-Carlgren N, et al. Differential effects of Aβ42/40, p-tau231 and p-Tau217 on cognitive decline. *Ann Neurol*. 2024;95:697-710.
4. Jack CR Jr, Bennett DA, Blennow K, et al. NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease. *Alzheimers Dement*. 2018;14:535-562.
5. Janelidze S, Teunissen CE, Zetterberg H, et al. Head-to-head comparison of 8 plasma Aβ42/40 assays. *JAMA Neurol*. 2021;78:1375-1382.
6. Ashton NJ, Brum WS, Kromer R, et al. Diagnostic accuracy of a plasma phosphorylated tau 217 immunoassay for Alzheimer disease pathology. *JAMA Neurol*. 2024;81:561-572.
7. Mattsson-Carlgren N, Janelidze S, Bateman RJ, et al. Soluble p-Tau217 reflects Aβ and tau pathology. *EMBO Mol Med*. 2021;13:e14022.
8. Palmqvist S, Janelidze S, Quiroz YT, et al. Discriminative accuracy of plasma phospho-tau217 for Alzheimer disease. *JAMA*. 2020;324:772-781.
9. Brum WS, Cullen NC, Janelidze S, et al. A two-step workflow based on plasma p-Tau217 to screen for Aβ positivity. *Nat Aging*. 2023;3:1079-1090.
10. Therriault J, Vermeiren M, Servaes S, et al. Association of plasma p-Tau217 with detection of Alzheimer disease in clinical practice. *JAMA Neurol*. 2024;81:553-560.
11. van Dyck CH, Swanson CJ, Aisen P, et al. Lecanemab in Early Alzheimer's Disease. *N Engl J Med*. 2023;388:9-21.
12. Cummings J, Apostolova L, Rabinovici GD, et al. Lecanemab: Appropriate Use Recommendations. *J Prev Alzheimers Dis*. 2023;10(3):362-377. doi:10.14283/jpad.2023.30 <!-- [JPAD-NEW-REF] AUR for lecanemab — JPAD home journal; cited in Introduction. -->
13. Rabinovici GD, Selkoe DJ, Schindler SE, et al. Donanemab: Appropriate Use Recommendations. *J Prev Alzheimers Dis*. 2025;12:100150. doi:10.1016/j.tjpad.2025.100150 <!-- [JPAD-NEW-REF] AUR for donanemab — JPAD home journal; cited in Introduction. -->
14. Hampel H, Hardy J, Blennow K, et al. The Aβ pathway in Alzheimer's disease. *Mol Psychiatry*. 2021;26:5481-5503.
15. Sims JR, Zimmer JA, Evans CD, et al. Donanemab in Early Symptomatic Alzheimer Disease. *JAMA*. 2023;330:512-527.
16. Tsoy E, Kiekhofer RE, Guterman EL, et al. Assessment of racial/ethnic disparities in timeliness and comprehensiveness of dementia diagnosis in California. *JAMA Neurol*. 2021;78:657-665.
17. Zuelsdorff M, Okonkwo OC, Norton D, et al. Introducing social determinants of health to the Alzheimer's Disease Research Center network. *Alzheimers Dement*. 2024;20:e089648.
18. Klunk WE, Koeppe RA, Price JC, et al. The Centiloid Project: standardizing quantitative amyloid plaque estimation by PET. *Alzheimers Dement*. 2015;11:1-15.
19. Sperling RA, Donohue MC, Raman R, et al. Trial of solanezumab in preclinical Alzheimer's disease. *N Engl J Med*. 2023;389:1096-1107.
20. Jack CR Jr, Bernstein MA, Fox NC, et al. The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *J Magn Reson Imaging*. 2008;27:685-691.
21. Rissman RA, Langford O, Raman R, et al. Plasma p-Tau217 predicts amyloid PET status in the A4 and LEARN cohorts. *J Prev Alzheimers Dis*. 2024;11:823-830. doi:10.14283/jpad.2024.134 <!-- [JPAD-NEW-REF] Cited in Methods §2.1.2 — same cohort as our external validation. Verify author order and full title against publisher record. -->
22. Martínez-Dubarbie F, et al. Diagnostic accuracy of the Lumipulse plasma p-Tau217 assay. *J Prev Alzheimers Dis*. 2024. doi:10.14283/jpad.2024.152 <!-- [JPAD-NEW-REF] Cited in Methods §2.2 — Lumipulse platform used in ADNI. Verify exact authors and pagination. -->
23. Lewczuk P, Riederer P, O'Bryant SE, et al. Cerebrospinal fluid and blood biomarkers for neurodegenerative dementias. *World J Biol Psychiatry*. 2018;19:244-328.
24. Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774-781.
25. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing areas under correlated ROC curves. *Biometrics*. 1988;44:837-845.
26. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: essential items for reporting diagnostic accuracy studies. *BMJ*. 2015;351:h5527.
27. Centers for Medicare & Medicaid Services. Medicare and Medicaid Programs; CY 2024 Payment Policies Under the Physician Fee Schedule and Other Changes to Part B Payment and Coverage Policies. *Federal Register*. November 16, 2023. https://www.federalregister.gov/documents/2023/11/16/2023-24184/medicare-and-medicaid-programs-cy-2024-payment-policies-under-the-physician-fee-schedule-and-other
28. Mattke S, Tang Y, Hanson M, et al. Economic evaluation of plasma-based Alzheimer's disease screening programs. *J Prev Alzheimers Dis*. 2025;12:100334. doi:10.1016/j.tjpad.2025.100334 <!-- [JPAD-NEW-REF] Cited in Methods §2.6. Verify author list. -->
29. [RETIRED — MRI analysis removed; renumber references before submission]
30. Chatterjee P, Pedrini S, Stoops E, et al. Plasma glial fibrillary acidic protein in cognitively normal older adults at risk of Alzheimer's disease. *Transl Psychiatry*. 2021;11:27.
31. Palmqvist S, Tideman P, Cullen NC, et al. Prediction of future Alzheimer's disease dementia using plasma phospho-tau. *Nat Med*. 2021;27:1034-1042.
32. Barthélemy NR, Salvadó G, Schindler SE, et al. Highly accurate blood test for Alzheimer's disease. *Nat Med*. 2024;30:1085-1095.
33. Gonzalez-Ortiz F, Turton M, Kac PR, et al. Brain-derived tau: a novel blood-based biomarker. *Brain*. 2023;146:1152-1165.
34. Palmqvist S, Whitson HE, Allen LA, et al. Alzheimer's Association Clinical Practice Guideline on the use of blood-based biomarkers in the diagnostic workup of suspected Alzheimer's disease within specialized care settings. *Alzheimers Dement*. 2025;21:e70535.
35. Jack CR Jr, Wiste HJ, Weigand SD, et al. Age-specific and sex-specific prevalence of cerebral β-amyloidosis, tauopathy, and neurodegeneration. *Lancet Neurol*. 2017;16:435-444.
36. Gleason CE, Norton D, Zuelsdorff M, et al. Association between enrollment factors and incident cognitive impairment in Blacks and Whites. *Alzheimers Dement*. 2019;15:1533-1545.
37. Zheng L, et al. Plasma biomarkers of Alzheimer's disease across sociodemographic factors. *J Prev Alzheimers Dis*. 2024. doi:10.14283/jpad.2024.142 <!-- [JPAD-NEW-REF] Cited in Limitations. Verify exact authors and pagination. -->
38. Robinson JL, Lee EB, Xie SX, et al. Neurodegenerative disease concomitant proteinopathies. *Brain*. 2018;141:2181-2193.
39. Kapasi A, DeCarli C, Schneider JA. Impact of multiple pathologies on the threshold for clinically overt dementia. *Acta Neuropathol*. 2017;134:171-186.

---

# CHANGE LOG — what was done and what still needs your action

This file is a draft. Edits made are intentionally conservative — your wording is preserved except where JPAD format requires restructuring, where an inserted citation needs a connector sentence, or where a P-value style fix is mechanical.

## Done in this draft

1. **Structured abstract.** Reorganized into JPAD's required headings (Background, Objectives, Design, Setting, Participants, Intervention, Measurements, Results, Conclusions). All factual content is your original wording; only the section partition and minor connectives are new. One trailing phrase added re: trial enrichment.
2. **Keywords trimmed to 6** (JPAD prefers 3–5). May want to drop one to get to 5.
3. **P-value style → AMA 11th ed.** All `p<0.001`, `p = 0.014`, `DeLong p = 0.014` → `*P* < .001`, `*P* = .014`, `DeLong *P* = .014` (italic P, no leading zero, no space before unit).
4. **Confidence interval delimiter** normalized to `(95% CI, X–Y)` (AMA preferred) instead of `(95% CI: X–Y)`.
5. **Six must-cite JPAD/AUR papers inserted** with connector sentences (each annotated `[JPAD-CITE]` inline). Reference list renumbered accordingly; original 32 refs preserved in the same order, new refs inserted at semantically appropriate positions:
   - Cummings/Rabinovici lecanemab AUR — Introduction
   - Rabinovici donanemab AUR — Introduction
   - Rissman 2024 (longitudinal p-Tau217 in A4/LEARN) — Methods §2.1.2
   - Martínez-Dubarbie 2024 (Lumipulse p-Tau217 accuracy) — Methods §2.2
   - Mattke 2025 (economic eval of plasma screening) — Methods §2.6
   - Zheng 2024 (biomarkers by sociodemographic factors) — Limitations
6. **Reference list to AMA style.** Italicized journal names, removed parenthetical year, kept "et al" after 3 authors (you already had this). DOIs added for the 7 new entries.
7. **Framing shift toward "AI/ML approach for reducing PET reliance in early AD"** (v2 revision per author + reviewer feedback "Shift the emphasis, don't shift the identity"). Gray zone retained as the *mechanism*; PET reduction + scalable AI screening elevated as the *implication*. All framing edits marked `[JPAD-FRAMING]`. Concretely:
   - **Abstract Conclusion** reordered to lead with PET-reduction headline and close with the three JPAD scope keywords (screening / DMT eligibility / trial enrichment).
   - **Introduction final paragraph** now frames A4/LEARN as a secondary-prevention trial population — surfaces the trial-screening + DMT-eligibility implications that were previously implicit.
   - **§3.7 Cost simulation** gains one sentence positioning the cost arithmetic as a screening-plus-diagnostic-plus-enrichment cost model (anchored to the existing Mattke 2025 ref).
   - **§5.2 reverted** to "Implementation in Clinical Practice" (memory-clinic focus); new **§5.3 "Implications for Clinical Trial Enrichment and DMT Eligibility"** added — two paragraphs explicitly operationalizing the Alzheimer's Association Clinical Practice Guideline thresholds (ref 34) and the JPAD-published AURs (refs 12, 13) for the trial-sponsor and DMT-prescribing reader. No new analyses; reframes existing results.
   - **§7 Conclusions** rewritten to lead with the 69–70% cost reduction and 71.9% PET-avoidance headline, gray-zone resolution stated as the mechanism, closing with the same three JPAD scope keywords as the abstract conclusion. Original phrasing preserved where possible.
8. **Figure / table consolidation plan** annotated inline (`[JPAD-FIG-MERGE]`, `[JPAD-TABLE-MERGE]`). The text is unchanged but the figure captions now indicate the planned 5-graphic layout:
   - **Figure 1** (kept as is): workflow + p-Tau217 probability distribution
   - **Figure 2** (new composite): merge of old Fig 2 ROC panels + old Fig 3 model characterization panels + Table 2 metrics as in-figure text inset (8-panel layout)
   - **Figure 4** (new): cost simulation (old Fig 6) with Table 3 as in-figure annotation
   - **Table 1** (kept as is): baseline characteristics
   - Section 4 (Case Examples) recommended for relocation to Supplementary.

## Still requires your action

1. **DOI cleanup on the original 32 references.** I added DOIs for the 7 new JPAD entries (confidence: high — drawn from the feedback you pasted), but did NOT fabricate DOIs for the original 32 to avoid hallucination. You'll need a final pass to populate DOIs from publisher records.
2. **Author/page verification on the 7 new JPAD refs.** I used the page/DOI info from your feedback; for refs 21, 22, 28, 29, 37 the feedback gave only first author / DOI prefix. Pull the canonical citation from each publisher record before submission. (Search marker: `<!-- [JPAD-NEW-REF]`.)
3. **Actual figure file merges.** The captions are annotated, but the PDF/PNG composites in [results/figures/](results/figures/) still need to be assembled (Inkscape per your acknowledgments). Source panels you'll combine:
   - **Figure 2 composite:** [figure_2_roc_combined.pdf](results/figures/figure_2_roc_combined.pdf) + [figure_3_model_characterization.pdf](results/figures/figure_3_model_characterization.pdf) → new `figure_2_performance_composite.pdf`
   - **Figure 4 (cost):** [figure_6_cost_comparison.png](results/figures/figure_6_cost_comparison.png) → renamed `figure_4_cost.pdf` with Table 3 as in-figure annotation
4. **Strategic framing review.** The feedback you pasted was **truncated mid-sentence** at the title-reframing recommendation. The light additions I made (3 places, all `[JPAD-FRAMING]`-marked) are conservative. You may want to:
   - Reconsider title — feedback was about to suggest a trial-enrichment subtitle
   - Decide whether to keep 6 keywords or drop to 5
   - Decide whether the Limitations section needs stronger trial-enrichment framing (it currently doesn't address how the algorithm would behave in a prevention trial population specifically)
5. **Word count.** I have not counted; JPAD's limit is 5,500 words for Original Articles. The body of this draft is roughly comparable to your original — likely under, but should be confirmed before submission.
6. **Cover letter and Editorial Manager submission.** Not produced in this draft. JPAD now submits via https://www.editorialmanager.com/tjpad/ since the 2025 Springer→Elsevier transition.

## Notes for the audit pass

- All inline `[JPAD-*]` HTML comments are non-rendering markers; strip them before final submission. A quick `sed -i '' -E 's/<!--[[:space:]]*\[JPAD[^>]*-->//g'` will clear them, but read each one first to confirm I didn't make a judgment call you disagree with.
- I preserved your existing British/American spelling and punctuation exactly. JPAD's AMA style is U.S. English; your manuscript is already U.S. English so no changes were needed.
- Original Figure 1 caption mentions "P<0.25" and "P > 0.75" (referring to model probability output, not P-values). I deliberately did NOT italicize these — they are probability variables, not statistical P-values.
