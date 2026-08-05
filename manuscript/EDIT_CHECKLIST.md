# GRAD manuscript — correction checklist

Applies to `GRAD_JPAD_v1.md` (synced 2026-08-04 from the author's Word version).
Every number below comes from `results/tables/manuscript_number_audit.csv`.
**Regenerate with `python3 scripts/GRAD_number_audit.py` — never patch by hand.**

---

## A. Internal contradictions (fix first — these exist independent of any other edit)

- [ ] **§2.5 contradicts itself on confidence intervals.** It says "percentile bootstrap confidence intervals (2,000 resamples)" and then, one paragraph later, "AUC with DeLong confidence intervals." Bootstrap is what the code does. Replace the second phrase with:
  > All AUC confidence intervals are percentile bootstrap over participants (2,000 resamples); DeLong's test [25] is used only for paired comparisons of two models scored on the same participants.

- [ ] **§2.7 says residual 13.3%; §2.4 and §3.3 say 10.3%.** Correct: **10.3% (ADNI), 11.4% (A4)**.

---

## B. Number corrections

| Location | Current | Correct |
|---|---|---|
| Abstract, §3.4 (×2), Fig 2 legend | 0.857 (0.813–0.897) | 0.857 (**0.811–0.899**) |
| Abstract, §3.5, Fig 3 legend | 0.828 (0.806–0.849) | **0.867 (0.849–0.883)** |
| Abstract, §3.5, Fig 3 legend | r = 0.642 | **Spearman ρ = 0.728** |
| §3.5 | "349 positive and 2 negative" | **589 positive, 328 negative** |
| §3.5, §4.3 | 1,293 routed (78.6%) | **727 routed (44.2%)**; Stage 1 resolves 917 (55.8%) |
| Fig 3 legend | Brier = 0.174 | **0.148** |
| §3.2 | 0.912 (0.861–0.956) | 0.912 (**0.858–0.957**) |
| Abstract, §3.7, §4, §4.5 | 67–71% savings | **69–70%** |
| §2.7 | four strategies | **three** |
| Abstract, §4.2, §4.5 | 72.5% avoid PET | **71.9%** |

### §3.3 + Fig 2A feature importances — the ranking changed, not just the values

> p-Tau217 (**34.6%**), Age (**15.8%**), **tau-Aβ42/40 divergence ratio (15.4%)**, **GFAP by p-Tau217 interaction (15.1%)**, APOE ε4 (**12.3%**), GFAP (**6.9%**)

Items 3 and 4 swap places. They are 15.4 vs 15.1 with overlapping bootstrap
intervals — **do not rank them against each other in prose.**

---

## C. Remove MRI — 12 locations

- [ ] Abstract Methods — delete "and optional MRI volumetrics"
- [ ] Abstract Results — delete the whole ΔAUC sentence
- [ ] Intro final ¶ — "utilizing both plasma and MRI-derived features" → "utilizing plasma-derived features"
- [ ] Intro final ¶ — delete "Next we assessed whether integrating MRI-derived volumes would improve…"
- [ ] §2.1.2 — replace the 1,293/1,044 sentence with:
  > Of these, 917 (55.8%) were resolved by the Gatekeeper and 727 (44.2%) were routed to Stage 2 (Reflex).
- [ ] §2.4 — delete the two-sentence MRI paragraph
- [ ] §2.5 — "models with and without MRI variables" → "two models scored on the same participants"
- [ ] §2.7 — delete "and Staged + MRI"
- [ ] §3.6 — delete the entire subsection; renumber 3.7 → 3.6, 3.8 → 3.7
- [ ] §3.8 Case 2 — "elevated GFAP, low Aβ42/40, hippocampal atrophy on MRI" → "elevated GFAP, low Aβ42/40, elevated tau-Aβ42/40 divergence"
- [ ] §4 opening ¶ + §4.2 — delete both MRI paragraphs
- [ ] Abbreviations — delete the MRI line

**Reference [24] Fischl becomes orphaned.** [29] Manjavong survives (cited in the
Introduction). [35] Jack appears already uncited in this draft — worth checking.

**Why MRI is out:** hippocampal volume adds real value *within* a cohort
(+0.025 AUC, p = 0.001, n = 724) but does **not** survive transport from ADNI to
A4 (+0.008, 95% CI −0.000 to +0.018, p = 0.061), including as an atrophy × tau
interaction (+0.005, p = 0.205). ADNI's gray zone is genuinely atrophic
(hippocampal Z −0.396); A4's is preclinical (−0.049). See
`scripts/GRAD_mri_external.py`.

---

## D. Three rewrites

### D1. §4 opening — a claim that is now backwards

Currently: GFAP "provided the greatest incremental predictive value."
**GFAP is last at 6.9%.** Replace with:

> Among Reflex features, plasma GFAP — a biomarker of reactive astrogliosis
> linked to early Aβ accumulation [30] — contributed mainly through its
> interaction with p-Tau217 (15.1% of total importance) rather than on its own
> (6.9%), consistent with astrogliosis being informative about amyloid
> principally when read alongside tau pathology.

### D2. §4.1 — replace the "~3% AUC decrease" sentence

It is now false as written: external AUC is *higher* than internal. Replace with:

> Stage 1 resolved 55.6% of ADNI participants and 55.8% of A4 participants — a
> difference of 0.2 percentage points — at an AUC of 0.912 in both, and
> full-pipeline discrimination was preserved (0.857 → 0.867). This held despite
> different populations (mixed CN/MCI/dementia versus entirely cognitively
> unimpaired), a large shift in Aβ prevalence (48.4% → 69.6%), and a change of
> assay platform (Lumipulse/Janssen → Lilly MSD). What did not transfer was
> calibration-in-the-large: the calibration slope remained 1.07, but the
> intercept was +0.72, with mean predicted risk 59.1% against an observed
> prevalence of 69.6%. Discrimination and routing transfer unchanged; only the
> intercept requires recalibration to local prevalence, a one-parameter
> correction that leaves every AUC and routing decision intact.

### D3. §4.3 — own the Stage 2 shortfall before a reviewer states it

Add after the rule-in/rule-out sentence:

> The Reflex stage considered in isolation does not meet the ≥90%/≥90% bar for
> PET substitution. In external validation it operated at 65.1% sensitivity and
> 70.0% specificity within the gray zone, trading specificity for sensitivity
> relative to p-Tau217 alone (51.5% → 65.1% sensitivity at a 10.8-point
> specificity cost). GRAD should be positioned as a triage and enrichment tool,
> not a substitute for Aβ-PET in the indeterminate cases themselves.

Anchor the trial-enrichment argument on the strongest supporting number:
**the gray zone is largest in CN — 48.6% (85/175), versus 40.2% MCI and
35.7% dementia.** p-Tau217 is least decisive exactly where prevention trials
recruit.

---

## E. Add — prevalence-adjusted predictive values (§4.2)

From the external operating point (77.6% sensitivity, 82.0% specificity), by Bayes.
Source: `results/tables/supp_table_prevalence_ppv_npv.csv`.

| Aβ prevalence | Setting | PPV | NPV |
|---|---|---|---|
| 20% | Primary care / community screening | 51.8% | **93.6%** |
| 30% | General neurology referral | 64.8% | 89.5% |
| 50% | Specialist memory clinic | 81.1% | 78.5% |
| 70% | A4 + LEARN (as observed) | **90.9%** | 61.0% |

State the conclusion explicitly: **GRAD is a rule-out instrument at community
prevalence and a rule-in instrument in enriched cohorts.**

---

## F. Figure legends — all three are stale

- [ ] **Fig 2C** — "forest plot… dashed line indicates overall AUC". It is now a
  **grouped bar chart**: subgroups on x, AUC on y, bars anchored at 0.50, one
  muted green shade per stratum, no reference line, bootstrap CI whiskers.
- [ ] **Fig 3B** — "A- (green) and A+ (red)". The frozen palette is
  **blue (Aβ−) and coral (Aβ+)**. Panels also reorder: **A** = three ROCs
  (full / Stage 1 / Stage 2), **B** = calibration, **C** = centiloid,
  **D** = gray-zone comparison vs p-Tau217 alone.
- [ ] **Fig 4** — "error bars represent 95% CI from sensitivity analysis".
  **There are no error bars.** It is now three strategies × two cohorts, each bar
  stacked into plasma panel + amyloid PET.

---

## G. Smaller accuracy items

- [ ] **§2.1.1** says "University of Pennsylvania (UPENN) Cohort" but the cohort is
  **207 UPENN + 113 Janssen**, as §2.2 itself states.
- [ ] **§2.3** formula is `Z = (log(x) − μ_ref)/σ_ref`; the code uses **log1p**.
  Write `log(1+x)`.
- [ ] **Abstract Interpretation** — "cost-effect tool" → "cost-effective tool"
- [ ] **Abstract** — "89.7% received a definitive classification without PET" is
  ADNI-only. Say **"89.7% in ADNI and 88.6% in A4."**
- [ ] **Abstract** — "76.8% of the remaining gray zone cases" is 109/142, ADNI-only.
  Label it as such.
- [ ] **Figure 1A schematic** (author-built, not script-generated): typos
  "Classififcation", "reudces", "refferals", "Bioclincial"; and n = 1640 → **1,644**.

---

## Reference numbering

Orphaned after these edits: **[24]**. Already orphaned in this draft before any
edits: **[2], [6], [7], [35]**. Decide on all of them together before renumbering —
mechanical renumbering will corrupt in-text citations if done piecemeal.
