# Reducing Amyloid PET Utilization With a Two-Stage Plasma Biomarker Algorithm

<!-- CANONICAL DRAFT, synced 2026-08-04 from the author's Word version.
     Transcribed VERBATIM: known errors are deliberately preserved so the
     correction pass stays explicit and auditable. Apply the checklist in
     manuscript/EDIT_CHECKLIST.md; verify every number against
     results/tables/manuscript_number_audit.csv (regenerate with
     scripts/GRAD_number_audit.py -- do not patch numbers by hand).
     The previous repo draft, with those corrections already applied, is kept at
     manuscript/_superseded_GRAD_JPAD_v1_with_edits_applied.md for reference. -->

Harthik Parankusham<sup>1</sup>* (harthik.parankusham@uconn.edu), Casey Vanderlip<sup>2</sup> (vanderlc@hs.uci.edu), Colin Birkenbihl<sup>3</sup> (cbirkenbihl@mgh.harvard.edu), Eashwar Krishna<sup>4</sup> (eashwar.krishna@uconn.edu), Chizobam Ugboaja<sup>5</sup> (chizobam.ugboaja@yale.edu), Andrew Budson<sup>6</sup> (abudson@bu.edu), Brandon Frank<sup>6</sup> (befrank1@bu.edu) and for the Alzheimer's Disease Neuroimaging Initiative†

1. Department of Physiology & Neurobiology, UConn Computational Biology Core, University of Connecticut, Storrs, CT 06269-3125, USA
2. Department of Neurobiology and Behavior, 1424 Biological Sciences III Irvine, University of California Irvine, Irvine, CA, 92697 USA
3. Department of Neurology, Massachusetts General Hospital, Harvard Medical School, Boston, MA 02115, USA
4. Department of Molecular & Cell Biology, University of Connecticut, 91 North Eagleville Road, Unit 3125, Storrs, CT 06269-3125, USA
5. Yale School of Medicine, Yale New Haven Health, 333 Cedar Street, New Haven, CT 06510, USA
6. VA Boston Healthcare System, Boston, MA, USA; Alzheimer's Disease Research Center, Boston University Chobanian & Avedisian School of Medicine, Boston, MA, USA

**\*Correspondence:**
Harthik Parankusham
Mailing Address: 72 E. Concord St., Boston, MA 02118
Email: harthik.parankusham@uconn.edu
Tel: (+1) 959-225-0126

**Running Headline:** A Two-Stage Algorithm for lowering Amyloid PET

---

## Abstract

### Objective

Plasma p-Tau217 leaves 30-50% of patients in an indeterminate "gray zone" that current clinical workflows resolve with Aβ-PET. We developed and validated GRAD (Gatekeeper-Reflex for Alzheimer's Disease), a two-stage model that instead resolves this zone algorithmically.

### Methods

GRAD was developed with 320 ADNI participants and validated in 1,644 A4/LEARN participants. Stage 1 used only p-Tau217, and cases with predicted probability between 0.25-0.75 were considered uncertain. Stage 2 evaluated uncertain cases with a random forest combining p-Tau217, GFAP, a tau-Aβ42/40 divergence ratio, a GFAP-p-Tau217 interaction, age, APOEε4, and optional MRI volumetrics. Aβ-PET was ground truth (ADNI florbetapir SUVR>1.11; A4 Centiloid≥20).

### Results

Stage 1 resolved 55.6% of cases (AUC 0.912; NPV 90.0%, PPV 87.2%). Stage 2 classified 76.8% of the remaining gray zone cases (AUC 0.751). The full model achieved an AUC of 0.857 (95% CI, 0.813-0.897; 78.7% sensitivity, 82.4% specificity). External validation gave AUC of 0.828 (95% CI, 0.806-0.849), with predicted probabilities correlating with centiloid values (r=0.642, P <.001). Integrating MRI in Stage 2 meaningfully improved gray zone classification (ΔAUC=+0.025, P=.014). Overall, with GRAD, 89.7% received a definitive classification without PET and cost simulation projected 67–71% cost savings versus universal PET.

### Interpretation

Multimarker classification where p-Tau217 is indeterminate resolves most gray zone cases without imaging, keeping Aβ-PET for highly uncertain cases. GRAD offers a cost-effect tool for memory-clinic screening and DMT eligibility in cognitively unimpaired and early symptomatic adults.

**Keywords:** p-Tau217; Alzheimer's disease; machine learning; amyloid PET; gray zone

†Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

---

## 1. Introduction

Blood-based biomarkers for Alzheimer's disease (AD) have enabled detection of cerebral Aβ pathology without the cost, invasiveness, and limited accessibility of Aβ-positron emission tomography (PET) or cerebrospinal fluid (CSF) analysis [1-3]. Here, we biologically define AD as the presence of Aβ, subsequent neurofibrillary tangles (NFT) and neurodegeneration [4]. Recently, phosphorylated tau at threonine-217 (p-Tau217) in plasma has emerged as the most accurate biomarker for identifying Aβ positive (Aβ+) individuals [5-8] relative to Aβ-PET.

Previous studies reporting AUC > 0.90 typically compare cognitively unimpaired (CU) individuals to AD dementia patients - two groups with significant separation in Aβ burden [3, 6, 8]. However, in patients with Mild Cognitive Impairment (MCI) where treatment decisions are pivotal, a large overlap of p-Tau217 values in classifying Aβ+ versus Aβ- is observed [9,10]. As a result for ~30-50% of individuals, there exists a 'gray zone' where p-Tau217 alone is insufficient to reliably determine Aβ status [9,10]. Several studies have attempted to improve Aβ-classification through ensemble models with features accessible in plasma. For example, Brum et al. [9] demonstrated two-stage workflows for p-Tau217, but validated models in modestly-sized cohorts, and resolved the gray-zone with Aβ-PET/CSF confirmation. Manjavong et al. [29] used integrative methods, combining p-Tau181 and structural imaging via linear regressions to determine DMT-eligibility, but did not address the gray-zone. Lastly, Giacomucci et al. [40] attempted p-Tau181 integration to alleviate the p-Tau217 gray zone. However, they lacked external validation and health-economic quantification, and only used one biomarker of the same biological family, p-Tau181, which can be impacted by comorbidities in real-world patients [38]. To the best of our knowledge, no studies explicitly address the gray zone with model performance a) validated in an external cohort, b) quantified through health-economic simulation and c) via multi-modal machine-learning - a continuously improving method with more data increasing accuracy.

The challenge introduced by this gray zone has become urgent due to the recent approval of disease-modifying therapies (DMTs). The U.S. Food and Drug Administration (FDA) label for lecanemab recently required confirmation of Aβ-pathology [11]. The Appropriate Use Recommendations for lecanemab [12] and donanemab [13] furthermore mandated biomarker-confirmed Aβ positivity as the entry requirement for DMTs. While these recommendations specify that Aβ confirmation must be achieved, the exact path to determine such classification is not discussed. Because Aβ+ misclassification carries significant consequences - false positives expose patients to increased risk of side effects without benefit [14], while false negatives delay promising therapy to eligible patients [11,15] - clinicians consistently elect expensive Aβ-PET over highly specific and accessible plasma-biomarkers like p-Tau217 to increase their diagnostic confidence [10, 34]. The gray zone problem remains the exact bottleneck through which any plasma-first DMT eligibility pathway must pass. Two-staged plasma-based workflows for Aβ status classification provide a clinically-actionable diagnostic pathway that could avoid costly PET scans [9]. They are highly relevant for trial enrichment in preclinical and early-symptomatic prevention programs, where plasma screening can dramatically reduce the PET burden [21].

We hypothesized that a machine learning (ML) based diagnostic approach, similar to reflex testing in laboratory medicine, could efficiently resolve gray zone uncertainty. First, a p-Tau217 screen would identify clear-cut cases, while a ML classifier integrating several predictors addresses uncertain cases. Only truly uncertain cases would require expensive, but confirmatory Aβ-PET scans. This leveled approach could greatly reduce patient burden, clinical trial costs and increase health equity.

In this study, we present the Gatekeeper-Reflex model for Alzheimer's Disease (GRAD), a two-stage machine-learning algorithm to classify individuals in the gray zone for Aβ positivity, utilizing both plasma and MRI-derived features. We first internally validated model performance and subsequently performed external validation in an independent dataset. Next we assessed whether integrating MRI-derived volumes would improve the Aβ classification for individuals within the gray zone. Lastly, we simulated the health economic implications of our approach.

---

## 2. Methods

### 2.1 Study Populations

We drew participants from two distinct cohorts for model development and external validation.

#### 2.1.1 Development and Training Cohort

To develop our approach, we selected 320 participants from the Alzheimer's Disease Neuroimaging Initiative (ADNI) [20] University of Pennsylvania (UPENN) Cohort. The cohort included CU (n=175, 54.7%), MCI (n=117, 36.6%) and AD dementia (n=28, 8.8%) participants, classified at the visit contemporaneous with plasma biomarker collection.

#### 2.1.2 Independent Validation Cohort

For external validation of our approach, we selected 1,644 participants from the Anti-Amyloid Treatment in Asymptomatic Alzheimer's disease (A4) Study. All selected participants had baseline p-Tau217 and Aβ status available, with Aβ PET scans utilized for ground-truth. Out of these patients, 1,145 were from the treatment arm (Aβ-positive, Centiloid≥20 per the Centiloid standardization framework [18]) and 499 were from the LEARN observational arm (Aβ-negative, Centiloid<20) [19]. Of the 1,293 A4 participants routed to Stage 2 (Reflex), 1,044 (565 Aβ+, 479 Aβ-) had T1-weighted MRI and were included in the MRI enhancement analysis.

### 2.2 Plasma Biomarkers Acquisition and Assays

ADNI plasma biomarkers were measured using the Lumipulse G p-Tau217 assay (Fujirebio) on the UPENN/Janssen platform [22]. Plasma samples were collected in EDTA tubes, centrifuged, aliquoted, and stored at -80°C as per ADNI biofluid protocols.

In A4, all participants had p-Tau217 measured via the Lilly Research Laboratories MSD immunoassay. Glial fibrillary acidic protein (GFAP) and neurofilament light (NfL) were measured on the Roche Elecsys platform for a subset of the cohort.

### 2.3 Plasma Biomarker Harmonization and Normalization

For harmonization, we Z-normalized all biomarkers measured from blood plasma to a reference population [23]. These consisted of CU, Aβ-negative individuals within each cohort. The raw biomarker values were log-transformed, then Z-scores computed as: Z = (log(x) - μ_ref) / σ_ref. Data leakage was prevented as the validation cohort's data was isolated during model development, and Z-normalization was part of the pre-processing pipeline.

### 2.4 Design of the Two-Stage GRAD Algorithm

For Stage 1 (Gatekeeper), a univariate logistic regression model used p-Tau217 to predict Aβ + individuals as determined by the Aβ-PET scan. To minimize misclassification, we set classification thresholds: Aβ-negative (P<0.25) and Aβ-positive (P>0.75). Cases falling within the intermediate 'gray zone' (0.25 ≤P≤0.75) were triaged to the next stage.

In Stage 2 (Reflex), for gray zone cases, we implemented Random Forest classifiers (100 trees, maximum depth of 5, balanced class weights). The feature set comprised p-Tau217, GFAP, a tau-Aβ divergence ratio (log[p-Tau217] - log[Aβ42/40]), a GFAP by p-Tau217 interaction term, age, and APOE ε4 carrier status. All interaction terms were manually selected based on domain expertise, rather than utilizing model-based discovery. For model interpretation, a final Random Forest was trained on all gray zone samples and feature importances were computed as mean decrease in Gini impurity from this model.

In the external validation on the A4 cohort, we augmented the model with ICV-normalized, FreeSurfer-derived hippocampal and entorhinal volumes [24]. This was not performed internally due to an underpowered gray zone and risk of LOOCV overfitting. Reflex cases with predicted probability outside 0.40-0.60 were considered resolved, while cases within 0.40-0.60 remained indeterminate and referred for confirmatory Aβ PET. In the ADNI LOOCV, this yielded a 10.3% (33/320) residual PET referral rate. The GRAD architecture is depicted in Figure 1.

> **Figure 1.** (A) GRAD: Two-stage Gatekeeper-Reflex algorithm workflow showing patient flow from plasma testing through final classification. (B) Internal and external cohort distribution of p-Tau217-based Aβ probability estimates with Gatekeeper thresholds (25%/75%) showing classification zones: Aβ Negative (P<0.25), Gray Zone (0.25 ≤P≤ 0.75), and Aβ Positive (P >0.75).

### 2.5 Model Validation Strategy

We evaluated our models using leave-one-out cross-validation (LOOCV). Data pre-processing and harmonization parameters were recalculated within each fold. We calculated percentile bootstrap confidence intervals (2,000 resamples of the LOOCV predictions) for AUC, sensitivity, and specificity. We also externally validated the ADNI-trained model to the independent A4 trial data after cross-platform harmonization. A4 external validation included both treatment-arm (Aβ-positive) and LEARN (Aβ-negative) participants, allowing standard binary classification metrics (AUC, sensitivity, specificity). We also computed the Spearman correlation between individual-level predicted probabilities and Aβ PET centiloid values for a continuous validity check.

The statistical analysis consisted of AUC with DeLong confidence intervals, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), likelihood ratios (LR+, LR-) under STARD diagnostic guidelines, and Brier scores for calibration of probabilistic predictions. We used DeLong's test [25] and paired bootstrap to compare the performance of models with and without MRI variables. Key metrics were reported following the STARD 2015 guidelines for diagnostic accuracy studies [26]. Participants missing plasma p-Tau217 or Aβ PET data were excluded (complete-case analysis) and no imputation was performed.

### 2.6 Computational Implementation and Transparency

The GRAD algorithm's underlying machine learning architectures were constructed using Python 3.10. When developing and optimizing the pipeline, we used Google Gemini to assist with iterative code refinement for the GRAD algorithm's structural logic. Following the use of this AI-assisted tool, the authors independently reviewed, verified, and thoroughly tested all refined code to ensure utmost accuracy for replicability. The authors maintain sole responsibility for the integrity of this final model and the data presented.

### 2.7 Cost Impact Simulation

To quantify GRAD's health economic impact, we developed a decision-analytic model comparing four diagnostic strategies for a projected cohort of 10,000 patients: Universal Aβ PET (estimated conservatively $3,000 per scan), p-Tau217 screening + PET for gray zone cases ($350 per single-analyte p-Tau217 test + PET for 44.4%), our GRAD staged algorithm ($600 per multi-analyte plasma panel [p-Tau217, GFAP, Aβ42/40] + PET for residual 13.3%), and Staged + MRI (assuming MRI already obtained and/or insured). The GRAD strategies require a more comprehensive plasma panel than univariate p-Tau217 screening, and this cost differential is reflected in the simulation. Resolution rates were derived from the observed LOOCV and A4 validation results (see Sections 3.2, 3.3, and 3.6 for details). The unit costs reflect 2024 U.S. Medicare reimbursement schedules [27], and where necessary, higher costs were assumed to avoid overestimation of savings.

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
| p-Tau217, pg/mL (median [IQR]) | 0.110 [0.064-0.228] | 0.152 [0.098-0.234] |
| Cognitive status: CN / MCI / AD | 175 / 117 / 28 | 1,644 / 0 / 0 |

### 3.2 Stage 1: Gatekeeper Performance

Using only the Gatekeeper model, 178 of the 320 participants (55.6%) placed outside the gray zone and were resolved with an AUC of 0.912 (95% CI, 0.861-0.956; Figure 2F). 100 cases were classified as Aβ negative (NPV 90.0%), 78 cases were classified as Aβ positive (PPV 87.2%), and 142 cases were in the gray zone (44.4%) (Figure 1B) which were routed to Stage 2.

### 3.3 Stage 2: Reflex Performance

The Reflex model for participants who were sorted into the gray zone in Stage 1 (n=142) achieved an AUC of 0.751, with an accuracy of 70.4%, a sensitivity of 70.1%, and a specificity of 70.8%.

The highest ranked feature importances of the Reflex model were p-Tau217 (33.1%), Age (16.7%), GFAP by p-Tau217 interaction (16.1%), tau-Aβ42/40 divergence ratio (15.8%), APOE ε4 carrier status (10.5%), and GFAP (7.8%) (Figure 2A). The tau-Aβ42/40 divergence ratio, which captured the inverse relationship between tau phosphorylation and Aβ clearance, provided substantial predictive value beyond the individual markers. The Reflex model improved gray zone sensitivity from 51.9% to 70.1%, which is meaningful for cases that, by definition, lack clear biomarker separation. The Reflex model resolved 109 of 142 gray zone cases (76.8%), i.e., those with Stage 1 (Gatekeeper) predicted probability outside the 0.40-0.60 indeterminate range. The remaining 33 cases (10.3% of the full cohort) were indeterminate and referred for confirmatory Aβ PET. While Neurofilament light (NfL) features were initially evaluated, they were excluded following ablation analysis (ΔAUC=0.009).

> **Figure 2.** GRAD model characterization. (A) Reflex Random Forest feature importance (mean decrease in Gini impurity) showing p-Tau217 (33.1%), age (16.7%), GFAP by p-Tau217 interaction (16.1%), Tau–Aβ42/40 divergence ratio (15.8%), APOE ε4 (10.5%), and GFAP (7.8%). (B) Overall confusion matrix (N=320, LOOCV; accuracy 80.6%, sensitivity 78.7%, specificity 82.4%). (C) Subgroup AUC forest plot across cognitive status, APOE ε4 status, sex, and age tertiles (range 0.784-0.904); dashed line indicates overall AUC (0.857). (D) Sensitivity, specificity, PPV, NPV, and accuracy as a function of classification threshold, with 90% sensitivity (threshold =0.240) and 90% specificity (threshold =0.674) operating points indicated. ROC curves for (E) 2-stage GRAD framework (AUC=0.857), (F) cases resolved by the Gatekeeper only (AUC=0.912), and (G) Reflex model for gray zone (AUC=0.751). Shaded regions: 95% bootstrap CIs.

### 3.4 Overall Model Performance

The complete GRAD model achieved an AUC of 0.857 (95% CI, 0.813-0.897), accuracy of 80.6% (95% CI, 76.2%-84.7%), 78.7% sensitivity (95% CI, 72.3%-84.8%), 82.4% specificity (95% CI, 76.4%-88.0%), PPV was 80.8%, NPV was 80.5%, LR+ was 4.48, LR- was 0.26, and the Brier score was 0.148 (Supplementary Table S1). The overall ROC curve demonstrated strong discrimination (AUC=0.857, 95% CI, 0.813-0.897; Figure 2E). The model correctly classified 258 of 320 participants, with 122 true positives and 136 true negatives (Figure 2B). Subgroup analysis revealed consistent performance across cognitive status (AUC 0.784-0.902), APOE ε4 status (0.791-0.899), sex (0.817-0.904), and age tertiles (0.820-0.891), with highest discrimination in MCI participants (AUC =0.902) and males (AUC = 0.904; Figure 2C). At the 90% sensitivity operating threshold (P<0.240), the model achieved 90.3% sensitivity with 57.6% specificity (LR- 0.17). At 90% specificity threshold (P>0.674), it achieved 90.3% specificity with 67.7% sensitivity (LR+ 6.99; Figure 2D).

### 3.5 External Validation in A4 and LEARN

The ADNI-trained GRAD model was applied to 1,644 A4 participants (1,145 Aβ+, 499 Aβ-). It achieved AUC 0.828 (95% CI, 0.806-0.849; Figure 3A), with 73.5% sensitivity, 78.2% specificity, PPV 88.5%, and NPV 56.3%. The framework also showed good calibration (Figure 3C). The Gatekeeper only resolved 349 cases as positive and 2 as negative, routing 1,293 (78.6%) to the Reflex model. Individual-level predicted probabilities correlated with Aβ PET centiloid values (Spearman r=0.642, P<.001; Figure 3B), indicating that model outputs track continuous amyloid burden beyond binary classification.

> **Figure 3.** Performance and Calibration of GRAD in the A4 + LEARN External Validation Cohort (N = 1,644). (A) Receiver operating characteristic (ROC) curve demonstrating high discriminative performance (AUC=0.828; 95% CI, 0.806-0.849). (B) Spearman correlation between GRAD predicted probability and Aβ PET Centiloid values (r=0.642, P<.001), showing that predictions capture continuous variation in Aβ burden across A- (green) and A+ (red) groups. (C) Calibration plot showing the relationship between predicted and observed Aβ positivity across 10 bins; error bars represent 95% confidence intervals, and the dashed line is perfect calibration (Brier score=0.174). MRI enhancement: (D) ROC curves comparing MRI-only (AUC=0.720) vs plasma-only (AUC=0.829) vs. plasma+MRI (AUC=0.853); (E) Aβ positivity by hippocampal volume tertile showing 2.5-fold variation.

### 3.6 Incremental Value of MRI Integration

Amongst the 1,044 A4 gray zone participants with a T1-weighted MRI we found that the plasma-only AUC was 0.829 (95% CI, 0.812-0.846), the MRI-only AUC was 0.720 whereas plasma + MRI AUC was 0.853 (95% CI, 0.838-0.868; Figure 3D). The observed improvement was statistically significant (ΔAUC=+0.025, DeLong P=.014, bootstrap 95% CI, 0.008-0.039) and 99.6% of 1,000 iterations favored the MRI-enhanced GRAD model. Aβ+ rates varied 2.5-fold across hippocampal volume tertiles, from 74.7% in the smallest tertile to 30.2% in the largest, despite relatively similar p-Tau217 levels (0.146-0.170 pg/mL; Figure 3E).

### 3.7 Cost Impact Simulation

We simulated and compared the projected costs for PET-based Aβ classification with and without our proposed GRAD framework for a 10,000 patient cohort: Universal PET ($30,000,000; 10,000 scans), p-Tau217 screening + PET for gray zone ($16,820,000; 4,440 scans) which demonstrated 44% savings, GRAD staged algorithm ($9,993,000; 1,331 scans) which demonstrated 67% savings, and GRAD with MRI integration ($8,661,000; 887 scans) which resulted in 71% lower costs (Figure 4; Supplementary Table S6). The GRAD strategies require a multi-analyte plasma panel ($600) rather than single-analyte p-Tau217 ($350). Regardless, substantial reduction in PET utilization (86.7-91.1%) is noticed, and drives per capita costs from $3,000 down to $866-999.

> **Figure 4.** Cost impact simulation comparing four diagnostic strategies for a projected 10,000-patient cohort. Stacked bars show the breakdown of expenditure between amyloid PET costs (bottom, darker) and plasma panel costs (top, lighter). The error bars represent 95% confidence intervals derived from sensitivity analysis on resolution rates. Unit costs reflect differentiated plasma pricing: single-analyte p-Tau217 ($350) versus the full GRAD multi-analyte panel (p-Tau217, GFAP, Aβ42/40; $600), with amyloid PET at $3,000 per scan (CMS 2024 reimbursement schedule). The GRAD staged algorithm with MRI integration achieved the lowest per-patient cost ($866, 71% savings), while the staged algorithm without MRI ($999, 67% savings) required greater PET utilization for residual cases. Inset: percentage savings relative to universal PET for each strategy. PET scan sample sizes are shown below each strategy.

### 3.8 Clinical Application: Hypothetical GRAD Case Examples

**Case 1 - High Confidence Negative:** Individual in 60-64 age range with subjective cognitive decline. Plasma p-Tau217 0.04 pg/mL, Gatekeeper probability 8%. Classification: Aβ- (NPV 90.0%). Action: Reassurance, lifestyle modifications, routine follow-up. Outcome: PET avoided.

**Case 2 - Gray Zone Resolved:** Individual in the 75-79 age range with MCI. Plasma p-Tau217 0.12 pg/mL, Gatekeeper probability 45% (gray zone). Reflex panel: elevated GFAP, low Aβ42/40, hippocampal atrophy on MRI. Reflex probability: 78%. Classification: Aβ+. Action: Discuss anti-Aβ therapy (DMT) eligibility.

**Case 3 - Persistent Uncertainty:** Individual in the 50-54 age range with atypical presentation and mixed vascular findings. Reflex probability: 52%. Classification: Indeterminate. Action: Recommend confirmatory PET.

---

## 4. Discussion

We developed a two-stage algorithm that provides classifications for the majority of participants while routing uncertain cases for additional workup. Stage 1, the Gatekeeper, resolves 55.6% of cases with 88.8% accuracy using univariate p-Tau217; for the remaining patients who fall into the gray zone of plasma-based diagnostic uncertainty, Stage 2 of the framework, the Reflex classifier, improves discrimination to 70.4% accuracy. Among Reflex features, plasma GFAP, a biomarker of reactive astrogliosis linked to early Aβ accumulation [30], provided the greatest incremental predictive value, alone and in combination with p-Tau217. MRI integration also contributed statistically significant incremental value, and simulation projects 67-71% cost reduction versus universal PET. In future iterations of GRAD, research-grade biomarkers like MTBR-tau 243 [32], brain-derived tau [33] or plasma %p-Tau217 [31] can be incorporated prospectively in heterogeneous clinical populations to test viability in the diagnostic gray zone.

### 4.1 Performance Metrics Across Sequential Model Design Phases

Our staged approach intentionally stratified cases by diagnostic certainty rather than forcing classification through a single model. Stage 1 identifies cases where p-Tau217 alone provides high confidence (88.8% accuracy, NPV 90.0%); Stage 2 addresses inherently difficult cases through multi-marker integration. We intentionally designed and expected a 55.6% resolution rate. The model identifies cases where p-Tau217 alone is capable of providing a "High Confidence" classification, routing the inherently difficult cases to the multi-marker plasma Reflex model. In the ADNI LOOCV, after stage 2 (Reflex), this yielded a 10.3% (33/320) residual PET referral rate. This approach mirrors established laboratory medicine workflows, such as TSH with reflex free T4 for thyroid testing, antibody with reflex confirmatory for HIV screening, and PT/INR with reflex factor assays for coagulation. The small (~3%) AUC decrease from internal (0.857) to external (0.828) validation shows the expected cross-cohort and cross-assay (i.e., UPENN/Janssen to Lilly) generalization. Furthermore, threshold-independent discrimination was preserved, confirming the algorithm's biological validity across independent populations and assay platforms. Our framework essentially applies these proven paradigms to Alzheimer's disease blood biomarkers. Another deliberate design choice in the Reflex model was the use of six handcrafted, biologically motivated features rather than allowing the Random Forest to discover optimal feature representations from raw inputs. While the latter would be empirically biased and prone to model overfitting on our smaller sample size, each engineered feature has a fixed definition grounded in AD pathophysiology. For example, the tau-Aβ42/40 divergence ratio captures the well-established inverse relationship between tau phosphorylation and Aβ clearance - and can be computed identically for any new patient using standard formulas. This prioritizes clinical deployment: clinicians do not need to re-derive interaction terms for each prospective patient, and the transparent feature importances (mean decrease in Gini impurity; Section 3.3) verify that model decisions are biologically reasonable at the individual patient level.

### 4.2 Implementation in Clinical Practice

The 25%/75% probability thresholds used by the Gatekeeper are internal model parameters that control resolution rate and should likely not be associated with sensitivity/specificity operating points. At the model level, GRAD achieves clinically meaningful operating thresholds: the rule-out threshold (P<0.240) provides 90.3% sensitivity (NPV 85.6%) for safely excluding Aβ pathology, and the rule-in threshold (P>0.674) provides 90.3% specificity (PPV 86.8%) for confirming it. These benchmarks align with the Alzheimer's Association Clinical Practice Guideline, which recommends ≥90% sensitivity and ≥ 75% specificity for blood-based biomarker tests used as a triaging tool, and ≥ 90% sensitivity and specificity for tests used as a substitute for Aβ PET [34]. Importantly, clinicians can adjust the operating thresholds to match clinical context: for anti-Aβ therapy eligibility, where false positives carry ARIA risk, a higher specificity threshold (e.g., ≥95%) may be appropriate; for clinical trial screening, where missing eligible candidates is more costly, a ≥90% sensitivity / ≥70% specificity configuration may better balance enrollment yield against unnecessary exclusion.

The ΔAUC improvement (+0.025, P=.014) and 2.5-fold variation in Aβ positivity by hippocampal tertile demonstrate that MRI, which is already routine in dementia evaluation, provides valuable independent information. The MRI-enhanced resolution rate used in our cost simulation (80%) was consistent with prior evidence that hippocampal volumetrics can independently predict Aβ status [29]. While Aβ-PET remains the clinical standard for monitoring patients receiving disease-modifying therapies, the GRAD framework can optimize diagnostic pathways by decreasing the initial use of expensive confirmatory imaging to instances of high diagnostic uncertainty or treatment-efficacy validation cases.

The observed cost-effectiveness benefit of integrating the staged algorithm with MRI usage, as compared to sole usage of the algorithm, can be explained by the decreased necessary utilization of PET imaging to resolve uncertain cases that remain after algorithmic flow and subsequent resolution via MRI. Beyond direct cost savings and improved diagnostic capabilities, staged testing improves geographic access to care. Plasma testing is available at most clinical laboratories, in comparison to specialized PET centers. This research could ultimately reduce patient burden, benefiting underserved and rural populations [16,17].

### 4.3 Implications for Clinical Trial Enrichment and DMT Eligibility

Anti-amyloid prevention trials have historically had high pre-screen failure rates with confirmatory Aβ-PET acting as a primary bottleneck for eligibility. GRAD's external validation, A4/LEARN, was itself a secondary-prevention trial, so the indeterminate participants (78.6%) routed to stage 2 would represent the likely screening routine of similar prospective trials. During trials, GRAD's rule-out threshold (90.3% sensitivity, NPV 85.6%) can exclude Aβ- volunteers before PET, while its rule-in threshold (90.3% specificity, PPV 86.8%) can confirm Aβ+ for a large portion of potentially eligible participants.

Adjusting GRAD's threshold provides clinicians with a procedure to evaluate individual DMT eligibility: ≥95% specificity setting for starting lecanemab/donanemab where ARIA risk dominates, versus setting ≥90% sensitivity / ≥70% specificity for screening where enrollment yield is prioritized.

### 4.4 Limitations

Several limitations warrant consideration. Both ADNI and A4 participants are predominantly White (89%) and highly educated so model performance in racially diverse populations is unknown and may differ (e.g., higher baseline p-tau levels in Black participants independent of Aβ status) [36]. Second, both cohorts exclude significant medical comorbidities, particularly, reduced kidney function (eGFR), a known confounder of plasma biomarker concentrations, is underrepresented. Comorbid proteinopathies [38] and mixed pathologies [39] are also underrepresented relative to clinical practice. Performance in real-world clinical populations with greater medical complexity warrants future investigation. Third, stage 2 required multiple plasma markers which would need to be added to routine clinical care for some settings. Fourth, our cross-sectional design precludes longitudinal assessment of algorithm performance such as during treatment monitoring. Finally, the cost impact simulation demonstrated projected savings, but did not use real-world data. The simulation's fixed rates ignore geographical influences on cost, and does not account for the cost of misclassification i.e., ARIA. Future work would utilize real cohort financial data to more realistically demonstrate cost-savings associated with GRAD.

### 4.5 Conclusions

GRAD demonstrates that staged ML can significantly reduce confirmatory Aβ-PET reliance with projected 67-71% cost reduction against universal PET. By alleviating the p-Tau217 gray zone, 89.7% of participants received definitive Aβ classification without expensive imaging. With two stages, analogous to the established practice of reflex testing in laboratory medicine, Stage 1 identifies high-confidence cases with p-Tau217 alone while Stage 2 applies multi-marker classification to the uncertain cases. In the current era of anti-Aβ Alzheimer's therapies, our model offers a realistic pathway for clinicians and trialists to implement plasma biomarkers in three use cases 1) scalable screening in memory-clinics 2) DMT eligibility determination, and 3) clinical trial enrichment.

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

Each and all authors have no conflicts of interest to declare.

### Funding

A.B. is funded in part by the Veterans Health Administration (I01 CX002400) and NIH/NIA (P30 AG072978). B.F. is funded in part by the Veterans Health Administration (IK2 CX002625). Data collection and sharing for ADNI is funded by the National Institute on Aging (NIH Grant U19AG024904) and DOD ADNI (W81XWH-12-2-0012). The sponsors had no role in the design and conduct of the study; in the collection, analysis, and interpretation of data; in the preparation of the manuscript; or in the review or approval of the manuscript. ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from AbbVie, Alzheimer's Association, Alzheimer's Drug Discovery Foundation, Araclon Biotech, BioClinica, Inc., Biogen, Bristol-Myers Squibb Company, CereSpir, Inc., Cogstate, Eisai Inc., Elan Pharmaceuticals, Inc., Eli Lilly and Company, EuroImmun, F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc., Fujirebio, GE Healthcare, IXICO Ltd., Janssen Alzheimer Immunotherapy Research & Development, LLC., Johnson & Johnson Pharmaceutical Research & Development LLC., Lumosity, Lundbeck, Merck & Co., Inc., Meso Scale Diagnostics, LLC., NeuroRx Research, Neurotrack Technologies, Novartis Pharmaceuticals Corporation, Pfizer Inc., Piramal Imaging, Servier, Takeda Pharmaceutical Company, and Transition Therapeutics. The Canadian Institutes of Health Research provides funds to support ADNI clinical sites in Canada. Private sector contributions are facilitated by the Foundation for the National Institutes of Health (www.fnih.org). The A4 Study was funded by a public-private-philanthropic partnership, including NIH-NIA, Eli Lilly and Company, Alzheimer's Association, Accelerating Medicines Partnership, GHR Foundation, an anonymous foundation, and additional private donors. The LEARN Study was funded by the Alzheimer's Association and GHR Foundation.

### CRediT Authorship Contributions

H.P.: Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Project Administration. E.K.: Formal Analysis (health economics), Investigation, Data Curation. C.U.: Visualization, Writing-Original Draft (figure legends). C.V.: Validation, Writing-Review & Editing. C.B.: Methodology, Supervision, Review & Editing. A.B.: Supervision, Resources, Writing-Review & Editing. B.F.: Conceptualization, Supervision, Project Administration, Writing-Review & Editing. All authors read and approved final manuscript.

### Acknowledgements

We acknowledge and thank all the individuals and families participating in the ADNI and A4 and LEARN Trial Cohort(s), along with the site principal investigators, staff, and study partners whose efforts made these datasets possible. We acknowledge the use of Inkscape to assist with figure creation, and the use of Google DeepMind Gemini to assist with iterative code refinement in the Reflex (Stage 2) of model development. All analysis code was fully reviewed, validated, and modified by the authors. The scientific design, interpretation of results, and manuscript content are entirely the work of our authors.

### List of Abbreviations

AD: Alzheimer's disease
ADNI: Alzheimer's Disease Neuroimaging Initiative
A4: Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease Study
Aβ: Amyloid-beta
Aβ+: Amyloid-beta positive
Aβ-: Amyloid-beta negative
AUC: Area under the receiver operating characteristic curve
AUPRC: Area under the precision-recall curve
AUR: Appropriate Use Recommendations
CI: Confidence interval
CSF: Cerebrospinal fluid
DMT: Disease-modifying Therapy
FDA: U.S. Food and Drug Administration
GRAD: Gatekeeper-Reflex for Alzheimer's Disease (Model)
GFAP: Glial fibrillary acidic protein
IQR: Interquartile range
LEARN: Longitudinal Evaluation of Aβ Risk and Neurodegeneration (Study)
MCI: Mild Cognitive Impairment
ML: Machine learning
MRI: Magnetic resonance imaging
NfL: Neurofilament light chain
NPV: Negative predictive value
PET: Positron emission tomography
PPV: Positive predictive value
p-Tau217: Phosphorylated tau at threonine 217
ROC: Receiver operating characteristic
SD: Standard deviation
STARD: Standards for Reporting Diagnostic Accuracy

---

## References

1. Hansson O, Blennow K, Zetterberg H, Dage J. Blood biomarkers for Alzheimer's disease in clinical practice and trials. *Nat Aging*. 2023;3:506-519. https://doi.org/10.1038/s43587-023-00403-3.
2. Teunissen CE, Verberk IMW, Thijssen EH, et al. Blood-based biomarkers for Alzheimer's disease: towards clinical implementation. *Lancet Neurol*. 2022;21:66-77. https://doi.org/10.1016/S1474-4422(21)00361-6.
3. Ashton NJ, Janelidze S, Mattsson-Carlgren N, et al. Differential roles of Aβ42/40, p-tau231 and p-tau217 for Alzheimer's trial selection and disease monitoring. *Nat Med*. 2022;28:2555-2562. https://doi.org/10.1038/s41591-022-02074-w.
4. Jack CR Jr, Bennett DA, Blennow K, et al. NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease. *Alzheimer's Dement*. 2018;14:535-562. https://doi.org/10.1016/j.jalz.2018.02.018.
5. Janelidze S, Teunissen CE, Zetterberg H, et al. Head-to-Head comparison of 8 plasma Aβ42/40 assays in Alzheimer disease. *JAMA Neurol*. 2021;78:1375-1382. https://doi.org/10.1001/jamaneurol.2021.3180.
6. Ashton NJ, Brum WS, Kromer R, et al. Diagnostic accuracy of a plasma phosphorylated tau 217 immunoassay for Alzheimer disease pathology. *JAMA Neurol*. 2024;81:561-572. https://doi.org/10.1001/jamaneurol.2023.5319.
7. Mattsson-Carlgren N, Janelidze S, Bateman RJ, et al. Soluble p-tau217 reflects amyloid and tau pathology and mediates the association of amyloid with tau. *EMBO Mol Med*. 2021;13:e14022. https://doi.org/10.15252/emmm.202114022.
8. Palmqvist S, Janelidze S, Quiroz YT, et al. Discriminative accuracy of plasma phospho-tau217 for Alzheimer disease vs other neurodegenerative disorders. *JAMA*. 2020;324:772-781. https://doi.org/10.1001/jama.2020.12134.
9. Brum WS, Cullen NC, Janelidze S, et al. A two-step workflow based on plasma p-tau217 to screen for amyloid β positivity. *Nat Aging*. 2023;3:1079-1090. https://doi.org/10.1038/s43587-023-00471-5.
10. Selma-González J, Arranz J, Alcolea D, et al. Association of Plasma Phosphorylated Tau 217 With Clinical Deterioration Across Alzheimer Disease Stages. *Neurology*. 2025;104(11):e213769. https://doi.org/10.1212/WNL.0000000000213769.
11. van Dyck CH, Swanson CJ, Aisen P, et al. Lecanemab in early Alzheimer's disease. *N Engl J Med*. 2023;388:9-21. https://doi.org/10.1056/NEJMoa2212948
12. Cummings J, Apostolova L, Rabinovici GD, et al. Lecanemab: Appropriate Use Recommendations. *J Prev Alzheimers Dis*. 2023;10:362-377. https://doi.org/10.14283/jpad.2023.30.
13. Rabinovici GD, Selkoe DJ, Schindler SE, et al. Donanemab: Appropriate Use Recommendations. *J Prev Alzheimers Dis*. 2025;12:100150. https://doi.org/10.1016/j.tjpad.2025.100150.
14. Hampel H, Hardy J, Blennow K, et al. The amyloid-β pathway in Alzheimer's disease. *Mol Psychiatry*. 2021;26:5481-5503. https://doi.org/10.1038/s41380-021-01249-0.
15. Sims JR, Zimmer JA, Evans CD, et al. Donanemab in Early Symptomatic Alzheimer Disease: The TRAILBLAZER-ALZ 2 Randomized Clinical Trial. *JAMA*. 2023;330:512-527. https://doi.org/10.1001/jama.2023.13239.
16. Tsoy E, Kiekhofer RE, Guterman EL, et al. Assessment of Racial/Ethnic Disparities in Timeliness and Comprehensiveness of Dementia Diagnosis in California. *JAMA Neurol*. 2021;78:657-665. https://doi.org/10.1001/jamaneurol.2021.0399.
17. Zuelsdorff M, Okonkwo OC, Norton D, et al. Introducing social determinants of health to the Alzheimer's Disease Research Center network. *Alzheimer's Dement*. 2024;20:e089648. https://doi.org/10.1002/alz.70279.
18. Klunk WE, Koeppe RA, Price JC, et al. The Centiloid Project: Standardizing quantitative amyloid plaque estimation by PET. *Alzheimer's Dement*. 2015;11:1-15. https://doi.org/10.1016/j.jalz.2014.07.003.
19. Sperling RA, Donohue MC, Raman R, et al. Trial of solanezumab in preclinical Alzheimer's disease. *N Engl J Med*. 2023;389:1096-1107. https://doi.org/10.1056/NEJMoa2305032.
20. Jack CR Jr, Bernstein MD, Fox NC, et al. The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *J Magn Reson Imaging*. 2008;27:685-691. https://doi.org/10.1002/jmri.21049.
21. Rissman RA, Langford O, Raman R, et al. Plasma p-tau217 predicts amyloid PET status in the A4 and LEARN cohorts. *J Prev Alzheimers Dis*. 2024;11:823-830. https://doi.org/10.14283/jpad.2024.134.
22. Martínez-Dubarbie F, Guerra-Ruiz AR, López-García S, et al. Diagnostic accuracy of the Lumipulse plasma p-tau217 assay. *J Prev Alzheimers Dis*. 2024;11(6):1335-1341. https://doi.org/10.14283/jpad.2024.152.
23. Lewczuk P, Riederer P, O'Bryant SE, et al. Cerebrospinal fluid and blood biomarkers for neurodegenerative dementias. *World J Biol Psychiatry*. 2018;19:244-328. https://doi.org/10.1080/15622975.2017.1375556.
24. Fischl B. FreeSurfer. *Neuroimage*. 2012;62:774-781. https://doi.org/10.1016/j.neuroimage.2012.01.021.
25. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*. 1988;44:837-845. https://doi.org/10.2307/2531595.
26. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. *BMJ*. 2015;351:h5527. https://doi.org/10.1136/bmj.h5527.
27. Centers for Medicare & Medicaid Services. Medicare and Medicaid programs; CY 2024 payment policies under the physician fee schedule and other changes to Part B payment and coverage policies. *Fed Regist*. November 16, 2023. Accessed May 17, 2026. https://www.federalregister.gov/documents/2023/11/16/2023-24184/medicare-and-medicaid-programs-cy-2024-payment-policies-under-the-physician-fee-schedule-and-other.
28. Mattke S, Tang Y, Hanson M, et al. A Preliminary Economic Evaluation of a Potential Program for the Primary Prevention of Alzheimer's Disease. *J Prev Alzheimers Dis*. 2025;12:100334. https://doi.org/10.1016/j.tjpad.2025.100334.
29. Manjavong M, Kang JM, Diaz A, et al. Performance of plasma biomarkers combined with structural MRI to identify candidate participants for AD-modifying therapy. *J Prev Alzheimers Dis*. 2024;11(5):1198-1205. https://doi.org/10.14283/jpad.2024.110.
30. Chatterjee P, Pedrini S, Stoops E, et al. Plasma glial fibrillary acidic protein in cognitively normal older adults at risk of Alzheimer's disease. *Transl Psychiatry*. 2021;11:27. https://doi.org/10.1038/s41398-020-01137-1.
31. Palmqvist S, Tideman P, Cullen NC, et al. Prediction of future Alzheimer's disease dementia using plasma phospho-tau combined with other accessible measures. *Nat Med*. 2021;27:1034-1042. https://doi.org/10.1038/s41591-021-01348-z.
32. Barthélemy NR, Salvadó G, Schindler SE, et al. Highly accurate blood test for Alzheimer's disease is similar or superior to clinical cerebrospinal fluid tests. *Nat Med*. 2024;30:1085-1095. https://doi.org/10.1038/s41591-024-02869-z.
33. Gonzalez-Ortiz F, Turton M, Kac PR, et al. Brain-derived tau: a novel blood-based biomarker for Alzheimer's disease-type neurodegeneration. *Brain*. 2023;146:1152-1165. https://doi.org/10.1093/brain/awac407.
34. Palmqvist S, Whitson HE, Allen LA, et al. Alzheimer's Association clinical practice guideline on the use of blood-based biomarkers in the diagnostic workup of suspected Alzheimer's disease within specialized care settings. *Alzheimers Dement*. 2025;21:e70535. https://doi.org/10.1002/alz.70535.
35. Jack CR Jr, Wiste HJ, Weigand SD, et al. Age-specific and sex-specific prevalence of cerebral β-amyloidosis, tauopathy, and neurodegeneration in cognitively unimpaired individuals. *Lancet Neurol*. 2017;16:435-444. https://doi.org/10.1016/S1474-4422(17)30077-7.
36. Gleason CE, Norton D, Zuelsdorff M, et al. Association between enrollment factors and incident cognitive impairment in Blacks and Whites. *Alzheimer's Dement*. 2019;15:1533-1545. https://doi.org/10.1016/j.jalz.2019.07.016.
37. Zheng L, Shen S, Lu J, et al. Plasma biomarkers of Alzheimer's disease across sociodemographic factors. *J Prev Alzheimers Dis*. 2024;11(6):1342-1349. https://doi.org/10.14283/jpad.2024.142.
38. Robinson JL, Lee EB, Xie SX, et al. Neurodegenerative disease concomitant proteinopathies are prevalent, age-related and APOE4-associated. *Brain*. 2018;141:2181-2193. https://doi.org/10.1093/brain/awy137.
39. Kapasi A, DeCarli C, Schneider JA. Impact of multiple pathologies on the threshold for clinically overt dementia. *Acta Neuropathol*. 2017;134:171-186. https://doi.org/10.1007/s00401-017-1717-7.
40. Giacomucci G, Tabbì SMR, Ingannato A, et al. Refining Alzheimer's disease biological diagnosis with plasma biomarkers: resolving p-tau217 "gray zone" with p-tau181 integration. *Alzheimer's Dement (Amst)*. 2026;18:e70285. https://doi.org/10.1002/dad2.70285.
