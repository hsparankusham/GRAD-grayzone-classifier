#!/usr/bin/env python3
"""
Prevalence-adjusted PPV and NPV at the external operating point.

Sensitivity and specificity are properties of the test; PPV and NPV are not --
they move with how common amyloid positivity is in whoever walks through the
door. A4 + LEARN is 69.6% positive by design, which flatters PPV and punishes
NPV relative to any real clinic.

This re-expresses the SAME external operating point (sensitivity 77.6%,
specificity 82.0%) at prevalences a reader will actually face, by Bayes:

    PPV = sens*pi / (sens*pi + (1-spec)*(1-pi))
    NPV = spec*(1-pi) / (spec*(1-pi) + (1-sens)*pi)

Nothing is refitted; this is arithmetic on the observed operating point.

Output: results/tables/supp_table_prevalence_ppv_npv.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from _grad_paths import RESULTS, TABLES

# prevalence settings a reader recognises, plus the two cohorts' own
SETTINGS = [
    (0.20, 'Primary care / community screening'),
    (0.30, 'General neurology referral'),
    (0.48, 'ADNI development cohort'),
    (0.50, 'Specialist memory clinic'),
    (0.70, 'A4 + LEARN (as observed)'),
]


def operating_point(path):
    d = pd.read_csv(path)
    y, p = d.true_amyloid.values.astype(int), d.predicted_prob.values
    tn, fp, fn, tp = confusion_matrix(y, (p >= .5).astype(int)).ravel()
    return tp / (tp + fn), tn / (tn + fp), y.mean()


sens, spec, obs = operating_point(RESULTS / 'a4_binary_validation_predictions.csv')
print(f'External operating point: sensitivity {sens:.1%}, specificity {spec:.1%}')
print(f'Observed A4 + LEARN prevalence: {obs:.1%}\n')

rows = []
for pi, label in SETTINGS:
    ppv = sens * pi / (sens * pi + (1 - spec) * (1 - pi))
    npv = spec * (1 - pi) / (spec * (1 - pi) + (1 - sens) * pi)
    rows.append(dict(prevalence=pi, setting=label, ppv=ppv, npv=npv,
                     # per 1,000 tested, how many calls of each kind are wrong
                     false_pos_per_1000=round((1 - spec) * (1 - pi) * 1000),
                     false_neg_per_1000=round((1 - sens) * pi * 1000)))

t = pd.DataFrame(rows)
print(f"{'prevalence':>11s}  {'setting':36s}{'PPV':>8s}{'NPV':>8s}"
      f"{'FP/1000':>9s}{'FN/1000':>9s}")
for r in t.itertuples():
    print(f'{r.prevalence:>10.0%}  {r.setting:36s}{r.ppv:>8.1%}{r.npv:>8.1%}'
          f'{r.false_pos_per_1000:>9d}{r.false_neg_per_1000:>9d}')

t.to_csv(TABLES / 'supp_table_prevalence_ppv_npv.csv', index=False)
print(f'\nwritten: {TABLES / "supp_table_prevalence_ppv_npv.csv"}')
