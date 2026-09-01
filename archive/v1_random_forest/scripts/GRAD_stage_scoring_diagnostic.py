"""
Two-stage scoring diagnostic.

GRAD reports one probability per patient, but that number comes from two
different models: a logistic regression for Gatekeeper-resolved cases and a
Random Forest for gray-zone cases. Their outputs are never put on a common
scale, so Reflex probabilities can fall outside the [low, high] band the case
was routed into -- letting a deliberately-deferred case outrank a
high-confidence one. AUC is a global ranking metric and penalises this.

This quantifies the effect and reports the scale-reconciled alternative:
Reflex output mapped monotonically back into [low, high]. That transform
preserves within-gray-zone ordering exactly (gray-zone AUC is unchanged), so it
is a scale repair, not a refit.

Output: results/tables/supp_table_stage_scoring_diagnostic.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from _grad_paths import RESULTS  # noqa: F401

LOW, HIGH = 0.25, 0.75
rows = []
for cohort, f, has_gk in (('ADNI (LOOCV)', 'adni_loocv_predictions.csv', True),
                          ('A4 + LEARN', 'a4_binary_validation_predictions.csv', False)):
    p = pd.read_csv(RESULTS / f)
    y = p.true_amyloid.values
    fin = p.predicted_prob.values
    gz = (p.stage == 'reflex').values
    rec = fin.copy()
    v = fin[gz]
    rec[gz] = LOW + (v - v.min()) / (v.max() - v.min()) * (HIGH - LOW)
    row = dict(
        cohort=cohort, n=len(p), gray_zone_n=int(gz.sum()),
        reflex_outside_band=int(((v < LOW) | (v > HIGH)).sum()),
        gray_zone_auc=round(roc_auc_score(y[gz], fin[gz]), 4),
        gray_zone_auc_reconciled=round(roc_auc_score(y[gz], rec[gz]), 4),
        overall_auc_published=round(roc_auc_score(y, fin), 4),
        overall_auc_reconciled=round(roc_auc_score(y, rec), 4),
    )
    if has_gk:
        gk = p.gatekeeper_prob.values
        row['gatekeeper_alone_auc'] = round(roc_auc_score(y, gk), 4)
        row['reflex_gain_within_gray_zone'] = round(
            roc_auc_score(y[gz], fin[gz]) - roc_auc_score(y[gz], gk[gz]), 4)
    rows.append(row)

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(RESULTS / 'supp_table_stage_scoring_diagnostic.csv', index=False)
print(f"\nSaved {RESULTS / 'supp_table_stage_scoring_diagnostic.csv'}")
