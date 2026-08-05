# GRAD — script manifest

Every executable in the project lives in this folder. `../src/` holds the
importable library modules the scripts call (`data_loader`, `harmonizer`,
`gatekeeper`, `reflex`, `pipeline`, `validation`, `visualization`); it is not a
second script folder and nothing in it is run directly.

Manuscript references below are to `../manuscript/GRAD_JPAD_v1.md`.

## Analyses

| Script | Produces | Manuscript |
|---|---|---|
| `run_authoritative_loocv.py` | ADNI LOOCV predictions, bootstrap CIs, operating points | §3.2–3.4, Table 2, Supp Table S1 |
| `run_a4_binary_validation.py` | A4 + LEARN external validation, centiloid correlation | §3.5 |
| `gray_zone_mri_enhancement.py` | MRI enhancement, DeLong test, hippocampal tertiles | §3.6 |
| `run_subgroup_analysis.py` | AUC by cognitive status / APOE4 / sex / age tertile | §3.4 |
| `run_calibration_analysis.py` | Calibration curve, Brier, Hosmer–Lemeshow | §3.4, Fig S1 |
| `run_nfl_ablation.py` | NfL feature ablation (ΔAUC = 0.011 → NfL excluded) | §3.3 |
| `run_reflex_training_comparison.py` | Justifies training Reflex on gray-zone-only vs all-data | §2.4 design choice |
| `run_grad_master.py` | Self-contained end-to-end reimplementation; writes `grad_master_summary.json` | reviewer walkthrough |

## Figures

| Script | Produces | Manuscript |
|---|---|---|
| `generate_all_figures.py` | **Figures 2, 3, 4, 5, S1, S2, S3** — the shipped renders | primary figure generator |
| `generate_manuscript_figures.py` | Figures 1 and 6 | Fig 1, Fig 6 |
| `generate_figure6_cost.py` | Figure 6, cost simulation (newer standalone) | Fig 6 |
| `generate_figure_panels.py` | Individual panels for Inkscape compositing | JPAD 5-graphic merge |
| `generate_figure2_combined.py` | Figure 2 (standalone) | superseded by `generate_all_figures.py` |
| `generate_figure3_final.py` | Figure 3 (standalone) | superseded by `generate_all_figures.py` |
| `generate_figure4.py` | Figure 4 (standalone) | superseded by `generate_all_figures.py` |
| `generate_figure_s3_subgroup.py` | Figure S3 | shipped S3 render came from here (Apr 19) |
| `generate_stard_diagram.py` | Figure S2 STARD flow (standalone) | superseded by `generate_all_figures.py` |

## Submission artifacts

| Script | Produces |
|---|---|
| `generate_supplementary_docx.py` | Supplementary Materials DOCX (JPAD) |
| `generate_tables_docx.py` | Tables 1–3 as DOCX for upload |

## Demo / orchestration

| Script | Purpose |
|---|---|
| `run_demo.py` | Full pipeline on synthetic data — no data access needed |
| `generate_synthetic_data.py` | Builds `../data/synthetic/synthetic_cohort.csv` |
| `run_all.sh` | Reproduce everything |

---

## Known issues — read before running

### Fixed 2026-08-02

Every script now resolves paths through `_grad_paths.py`. The 2026-03-09
flat → packaged reorganisation had moved files without updating their paths,
which left the scripts unable to import `src/` or write to `results/`. All 22
scripts run, and `bash scripts/run_all.sh` completes.

`_grad_paths.py` also routes outputs automatically — figures to
`results/figures/`, tables to `results/tables/`, prediction CSVs to `results/`
— for reads as well as writes, so scripts always agree on where a file lives.
Nothing counts directory levels any more, so moving the project cannot break it
again. Three scripts that wrote to the Desktop now write into `results/`.

### Still open

1. **NfL ablation ΔAUC changed.** The table had never been re-run after the AGE
   fix. Re-running gives base AUC **0.8566** (was 0.8527) and **ΔAUC = 0.0087**
   (manuscript §3.3 says 0.011). The conclusion — NfL excluded — is unchanged,
   but the quoted number needs updating.

2. **Figure 1 and Figure 6 need regenerating with the final numbers.** Fig 1
   dated from 2026-03-09 (pre-AGE-fix), and no Fig 6 render matched the final
   cost model in `../results/tables/supp_table_cost_simulation.csv` (67% / 71%;
   earlier renders showed 74/77% and 70/75%). Both regenerate cleanly now.

3. **`generate_figure2_combined.py`, `generate_figure3_final.py`,
   `generate_figure4.py` and `generate_stard_diagram.py` overlap with
   `generate_all_figures.py`** — they write the same filenames. `run_all.sh`
   calls only `generate_all_figures.py` to avoid one clobbering the other.
   Consider deleting the four standalone scripts once you have confirmed the
   composite generator covers everything you need.
