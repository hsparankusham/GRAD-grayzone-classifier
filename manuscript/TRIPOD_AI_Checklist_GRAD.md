# TRIPOD+AI Checklist for GRAD Manuscript

**Study:** GRAD: A Two-Stage Algorithm for Resolving Diagnostic Uncertainty in the Plasma p-Tau217 Gray Zone
**Journal:** Alzheimer's Research & Therapy
**Study type:** Development + External Validation (D/E)
**Checklist version:** TRIPOD+AI 2024 (Collins et al., BMJ 2024;385:e078378)

---

## TITLE AND ABSTRACT

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **1** | Identify study as developing/evaluating a prediction model, target population, and outcome | Yes | Title: "Two-Stage Algorithm for Resolving Diagnostic Uncertainty in the Plasma p-Tau217 Gray Zone" |
| **2** | Structured abstract per TRIPOD+AI for Abstracts | Yes | Abstract: Background/Methods/Results/Limitations/Conclusions |

## INTRODUCTION

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **3a** | Healthcare context, rationale, references to existing models | Yes | Section 1 (Introduction), paragraphs 1-3; Research in Context: Literature Review |
| **3b** | Target population and intended purpose in care pathway, intended users | Yes | Section 1, paragraph 4: "workflow with clinical implementation potential, analogous to established laboratory medicine" |
| **3c** | Known health inequalities between sociodemographic groups | Yes | Section 1, paragraph 5: "rural and minoritized populations who face systemic barriers"; Limitations: "predominantly White (89%)" |
| **4** | Study objectives (development, validation, or both) | Yes | Section 1, paragraph 5: four numbered objectives |

## METHODS

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **5a** | Data source for development and evaluation | Yes | Section 2.1: ADNI (development), A4+LEARN (external validation) |
| **5b** | Key dates (accrual start/end, follow-up) | Partial | ADNI and A4 data downloads referenced by date; accrual periods not explicitly stated — **ADD: ADNI 2004-2024, A4 2014-2022** |
| **6a** | Study setting, number/location of centres | Partial | Multi-center studies referenced but specific centre counts not stated — **ADD: "ADNI: 57 sites across North America; A4: 67 sites across US, Canada, Australia, Japan"** |
| **6b** | Eligibility criteria | Yes | Section 2.1: biomarker + PET availability, cognitive status criteria |
| **6c** | Treatments received | Yes | Section 2.1: A4 treatment arm (solanezumab) vs LEARN observational arm |
| **7** | Data preparation and preprocessing | Yes | Section 2.2 (Harmonization), Section 2.3 (feature engineering, Z-scoring, log-transform) |
| **8a** | Outcome definition, time horizon, assessment method | Yes | Section 2.1: amyloid PET (AV45 SUVR >1.11 for ADNI, Centiloid >=20 for A4) |
| **8b** | Outcome assessor qualifications | N/A | Outcome is quantitative PET measurement, not subjective |
| **8c** | Blinding of outcome assessment | N/A | PET is objective quantification; model was not used clinically |
| **9a** | Choice of initial predictors, pre-selection | Yes | Section 2.3: domain-guided selection of 6 features; Limitations: acknowledges exploratory feature selection |
| **9b** | Predictor definitions, measurement | Yes | Section 2.3: all 6 features defined with formulas; Table 1 demographics |
| **9c** | Predictors measured same way across participants | Yes | Section 2.2: Z-harmonization addresses cross-platform differences (UPENN vs Janssen vs Lilly) |
| **10** | Sample size and justification | Yes | Section 2.1: N=320 ADNI with justification; N=1,644 A4 |
| **11** | Missing data handling | Partial | Complete-case analysis implied (required both biomarkers + PET); **ADD explicit statement: "Participants with missing p-tau217 or amyloid PET were excluded (complete-case analysis)"** |
| **12a** | How data used for development and evaluation | Yes | Section 2.4: LOOCV for internal, frozen model for external |
| **12b** | Predictor handling (transformation, scaling) | Yes | Section 2.2: log1p transform, Z-scoring; Section 2.3: StandardScaler for Reflex |
| **12c** | Model type, rationale, building steps, hyperparameter tuning, internal validation | Yes | Section 2.3: Gatekeeper (LR, l2, C=1.0), Reflex (RF, 100 trees, max_depth=5); Section 2.4: LOOCV |
| **12d** | For validation: how predictions calculated | Yes | Section 2.4/3.6: frozen ADNI-trained pipeline applied to A4 |
| **12e** | Performance measures and rationale | Yes | Section 2.4: AUC, accuracy, sensitivity, specificity, PPV, NPV, LR+, LR-, Brier score, bootstrap CIs |
| **12f** | Model updating from evaluation | Yes | No recalibration performed; stated in Results |
| **12g** | For evaluation: how model predictions calculated | Yes | Section 3.6 + Code Availability: full pipeline code provided |
| **13** | Class imbalance methods | Yes | Section 2.3: class_weight='balanced' in Random Forest |
| **14** | Fairness approaches | Yes | Section 3.5 (subgroup analysis by sex, APOE4, age, cognitive status); Limitations: demographic homogeneity acknowledged |
| **15** | Model output specification | Yes | Section 2.3: probabilities (0-1), then classified via thresholds (0.25/0.75) |
| **16** | Differences between development and evaluation data | Yes | Section 3.6: different assay platforms, clinical stages, amyloid thresholds |
| **17** | Ethics approval | Yes | Declarations: Ethics Approval section references ADNI/A4 IRB approvals and secondary analysis exemption |

## OPEN SCIENCE PRACTICES

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **18a** | Funding source and role | Yes | Declarations: Funding section |
| **18b** | Conflicts of interest | Yes | Declarations: Competing Interests |
| **18c** | Study protocol access | No | **ADD: "No formal study protocol was registered as this is a secondary analysis of existing data"** |
| **18d** | Study registration | No | **ADD: "This study was not prospectively registered as it is a retrospective secondary analysis"** |
| **18e** | Data sharing statement | Yes | Declarations: Availability of Data and Materials (ADNI LONI, A4 portal) |
| **18f** | Code availability | Yes | Declarations: Code Availability (GitHub + Zenodo DOI 10.5281/zenodo.18932865) |

## PATIENT AND PUBLIC INVOLVEMENT

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **19** | Patient/public involvement | No | **ADD: "Patients and the public were not involved in the design, conduct, or reporting of this study. Future prospective validation should incorporate patient and public perspectives"** |

## RESULTS

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **20a** | Flow of participants | Yes | Figure S2: STARD 2015 flow diagram |
| **20b** | Participant characteristics | Yes | Table 1: demographics; Section 3.1 |
| **20c** | For evaluation: compare with development data | Partial | Demographics compared implicitly; **ADD explicit comparison table or statement** |
| **21** | Number of participants/events per analysis | Yes | Section 3.1-3.6: sample sizes and event counts reported per analysis |
| **22** | Full prediction model (formula, code, API) | Yes | Code Availability: GitHub repo + Zenodo DOI; run_grad_master.py provides complete implementation |
| **23a** | Performance measures with CIs, subgroup results | Yes | Section 3.2-3.6: AUC with 95% CI; Section 3.5: subgroup analyses |
| **23b** | Sensitivity analyses | Yes | Section 3.5: subgroup analyses; Section 3.4: MRI enhancement; NfL ablation in supplementary |
| **24** | Model updating results | N/A | No model updating performed |

## DISCUSSION

| Item | Section | Reported | Location |
|------|---------|----------|----------|
| **25** | Overall interpretation with limitations, similar studies | Yes | Section 5 (Discussion) |
| **26** | Study limitations | Yes | Section 6 (Limitations): demographic homogeneity, cross-sectional design, feature selection |
| **27a** | Potential clinical use | Yes | Section 7 (Conclusions); Research in Context: Interpretation |
| **27b** | Requirements for clinical implementation | Yes | Section 5: discusses data availability, assay standardization needs |
| **27c** | Implications for future research | Yes | Research in Context: Future Directions; Section 7 |

---

## ITEMS REQUIRING ADDITION TO MANUSCRIPT

1. **5b** — Add ADNI accrual period (2004-2024) and A4 accrual period (2014-2022)
2. **6a** — Add centre counts (ADNI: 57 sites; A4: 67 sites)
3. **11** — Add explicit missing data statement
4. **18c** — Add protocol statement
5. **18d** — Add registration statement
6. **19** — Add PPI statement
7. **20c** — Add explicit development vs. evaluation data comparison

---

## REFERENCE

Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378
