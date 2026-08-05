#!/usr/bin/env python3
"""
Cost of three diagnostic strategies per 10,000 patients.

Every referral rate is computed from the prediction files, not hard-coded, and
both cohorts are shown so the simulation is anchored externally rather than on
ADNI alone. The MRI strategy is gone: hippocampal volume did not survive
external validation (see GRAD_mri_external.py).

Strategies
    Universal PET        everyone scanned
    p-tau217 + PET       single-analyte screen, PET for the gray zone
    GRAD staged          multi-analyte panel, PET only for residual uncertainty
                         (Reflex probability still within 0.40-0.60)

Unit costs are 2024 US Medicare reimbursement; where ambiguous the higher
figure is used so savings are not overstated.

Output: results/figures/panels/panel_cost.{png,pdf}
        results/tables/supp_table_cost_simulation.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _grad_paths import PANELS, RESULTS, TABLES
from _grad_style import apply_style, save_panel, INK, ROC_BLUE, GUIDE

apply_style(base=9)
PET_COLOUR = '#D9C77E'        # same amber that marks "referred for PET" elsewhere

N = 10_000
COST_PET, COST_PTAU, COST_PANEL = 3_000, 350, 600
REFLEX_BAND = (0.40, 0.60)    # Methods 2.4: still indeterminate after Stage 2

COHORTS = [('ADNI', 'adni_loocv_predictions.csv'),
           ('A4 + LEARN', 'a4_binary_validation_predictions.csv')]


def rates(fname):
    """Gray-zone and residual-PET rates, as fractions of the whole cohort."""
    d = pd.read_csv(RESULTS / fname)
    gz = d.stage == 'reflex'
    residual = gz & d.predicted_prob.between(*REFLEX_BAND)
    return gz.mean(), residual.mean()


def strategies(gz_rate, res_rate):
    """Plasma and PET spend per 10,000 patients, in millions."""
    return [
        ('Universal PET',   0,                  N * COST_PET,             1.00),
        ('p-tau217 + PET',  N * COST_PTAU,      gz_rate * N * COST_PET,   gz_rate),
        ('GRAD staged',     N * COST_PANEL,     res_rate * N * COST_PET,  res_rate),
    ]


rows = []
fig, ax = plt.subplots(figsize=(4.6, 2.5))
labels, ypos = [], []

# two bars per strategy, one per cohort; cohorts are adjacent so the reader
# compares the same strategy across cohorts, not strategies within a cohort
for k, (name, fname) in enumerate(COHORTS):
    gz_rate, res_rate = rates(fname)
    for j, (strat, plasma, pet, scan_rate) in enumerate(strategies(gz_rate, res_rate)):
        y = j * 2.6 + (0 if k == 0 else 0.85)
        ax.barh(y, plasma / 1e6, height=.78, color=ROC_BLUE, zorder=3)
        ax.barh(y, pet / 1e6, left=plasma / 1e6, height=.78,
                color=PET_COLOUR, zorder=3)
        total = (plasma + pet) / 1e6
        ax.text(total + .5, y, f'${total:.1f}M', va='center', fontsize=6.8,
                color=INK)
        ax.text(-.5, y, name, va='center', ha='right', fontsize=6.2, color=GUIDE)
        if k == 0:
            labels.append(strat); ypos.append(y + .43)
        rows.append(dict(cohort=name, strategy=strat,
                         plasma_cost=plasma, pet_cost=pet, total=plasma + pet,
                         per_patient=(plasma + pet) / N,
                         pet_scans=round(scan_rate * N),
                         savings_vs_universal=1 - (plasma + pet) / (N * COST_PET)))

ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Cost per 10,000 patients ($ millions)')
ax.set_xlim(0, 34)
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (ROC_BLUE, PET_COLOUR)]
ax.legend(handles, ['Plasma panel', 'Amyloid PET'], loc='lower right',
          frameon=False, fontsize=7, handlelength=1.1)
save_panel(fig, PANELS / 'panel_cost')

t = pd.DataFrame(rows)
t.to_csv(TABLES / 'supp_table_cost_simulation.csv', index=False)
print(f"{'cohort':12s}{'strategy':18s}{'total':>10s}{'per pt':>9s}"
      f"{'PET scans':>11s}{'savings':>9s}")
for r in t.itertuples():
    print(f'{r.cohort:12s}{r.strategy:18s}${r.total/1e6:>8.2f}M'
          f'${r.per_patient:>8,.0f}{r.pet_scans:>11,d}{r.savings_vs_universal:>9.1%}')
