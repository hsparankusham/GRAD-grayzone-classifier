"""
Corrected Gatekeeper threshold sensitivity sweep.

The previous supp_table_threshold_sensitivity.csv applied the 0.25/0.75 bands to
the FINAL two-stage probability, not to the Gatekeeper probability. It therefore
described post-hoc banding of the output, not the routing thresholds, and its
0.25/0.75 row disagreed with the main text (73.8% / n=84 vs 55.6% / n=142).

This runs a full LOOCV at each (low, high) pair -- refitting harmoniser,
Gatekeeper and Reflex inside every fold -- and reports what the thresholds
actually control. Also reports the scale-reconciled overall AUC.

Output: results/tables/supp_table_gatekeeper_threshold_sweep.csv
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from _grad_paths import RESULTS, ADNI_DIR                      # noqa: F401
from data_loader import ADNIDataLoader
from validation import LOOCVValidator

df = ADNIDataLoader(str(ADNI_DIR)).merge_data(use_baseline_only=True).reset_index(drop=True)
y = df['amyloid_positive'].values
rows = []
for low in (0.15, 0.20, 0.25, 0.30, 0.35):
    for high in (0.65, 0.70, 0.75, 0.80, 0.85):
        r = LOOCVValidator(gatekeeper_low=low, gatekeeper_high=high).validate_loocv(
            df, verbose=False)
        p = r.predictions
        gz = (p.stage == 'reflex').values
        fin = p.predicted_prob.values
        # scale-reconciled score: map Reflex output back into the [low, high] band
        rec = fin.copy()
        if gz.sum() > 1:
            v = fin[gz]
            rec[gz] = low + (v - v.min()) / (v.max() - v.min()) * (high - low)
        rows.append(dict(
            low=low, high=high,
            resolved_n=int((~gz).sum()),
            resolution_rate=round(float((~gz).mean()), 4),
            resolved_accuracy=round(float(
                ((p.gatekeeper_prob.values[~gz] >= .5) == y[~gz]).mean()), 4),
            gray_zone_n=int(gz.sum()),
            gray_zone_auc=round(float(roc_auc_score(y[gz], fin[gz])), 4) if gz.sum() > 5 else np.nan,
            overall_auc=round(float(roc_auc_score(y, fin)), 4),
            overall_auc_reconciled=round(float(roc_auc_score(y, rec)), 4),
        ))
        print(f"  low={low} high={high}  resolved={rows[-1]['resolution_rate']:.3f} "
              f"gz_n={rows[-1]['gray_zone_n']:3d} gz_auc={rows[-1]['gray_zone_auc']} "
              f"overall={rows[-1]['overall_auc']} reconciled={rows[-1]['overall_auc_reconciled']}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(RESULTS / 'supp_table_gatekeeper_threshold_sweep.csv', index=False)
print(f"\nSaved {RESULTS / 'supp_table_gatekeeper_threshold_sweep.csv'}")
