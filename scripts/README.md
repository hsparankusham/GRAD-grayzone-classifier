# GRAD — script manifest

Five executables reproduce the manuscript, plus two support modules. Run them in the
order below; step 1 is the source of truth and everything else reads its output.

| Script | Produces |
|---|---|
| `run_grad_impaired_logistic.py` | `results/tables/grad_v2_numbers.json`, both prediction CSVs |
| `GRAD_reference_sensitivity.py` | `results/tables/reference_sensitivity.json` (Table S3) |
| `GRAD_fig2_panels.py` | Figure 2 panels A–E |
| `GRAD_fig3_panels.py` | Figure 3 panels A–D and Supplementary Figure S1 |
| `GRAD_fig4_panels.py` | Figure 4 panels A–D |
| `GRAD_supplementary_tables.py` | Supplementary Tables S1–S9 (`.docx` and `.md`) |

Support modules, imported rather than run:

- `_grad_paths.py` — data and output locations. **Set your ADNI and A4 paths here.**
- `_grad_style.py` — frozen figure style: 600 dpi RGB PNG, Helvetica, uniform 10 pt.

Scripts from the superseded random-forest design are under
`../archive/v1_random_forest/scripts/`. They do not reproduce the current manuscript.
