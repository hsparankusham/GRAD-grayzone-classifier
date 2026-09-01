#!/usr/bin/env python3
"""
Single source of truth for every number the manuscript cites.

Regenerates each figure from the current prediction files rather than trusting
anything transcribed. Run this, then reconcile the manuscript against its output
-- do not patch individual numbers by hand.

CI convention (state this once in Methods and use it everywhere):
    All AUC confidence intervals are PERCENTILE BOOTSTRAP over participants
    (2,000 resamples). DeLong is used only for PAIRED comparisons between two
    models scored on the same participants.

Output: results/tables/manuscript_number_audit.csv  (+ printed report)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)
from scipy.stats import spearmanr

from _grad_paths import RESULTS, TABLES

N_BOOT = 2000
RNG = np.random.default_rng(20260204)
rows = []


def rec(section, label, value):
    rows.append(dict(section=section, quantity=label, value=value))
    print(f'  {label:52s} {value}')


def boot_ci(y, p, fn=roc_auc_score):
    vals = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def operating(y, p, thr=.5):
    tn, fp, fn_, tp = confusion_matrix(y, (p >= thr).astype(int)).ravel()
    return dict(sens=tp / (tp + fn_), spec=tn / (tn + fp),
                ppv=tp / (tp + fp), npv=tn / (tn + fn_),
                acc=(tp + tn) / len(y))


def block(name, path):
    d = pd.read_csv(path)
    y, p = d.true_amyloid.values.astype(int), d.predicted_prob.values
    gk_mask = d.stage != 'reflex'
    gz_mask = d.stage == 'reflex'

    print(f'\n{"=" * 68}\n{name}\n{"=" * 68}')
    rec(name, 'cohort n', f'{len(d):,}')
    rec(name, 'amyloid-positive n (%)', f'{y.sum():,} ({y.mean():.1%})')

    lo, hi = boot_ci(y, p)
    rec(name, 'full pipeline AUC [95% bootstrap CI]',
        f'{roc_auc_score(y, p):.3f} [{lo:.3f}-{hi:.3f}]')
    lo2, hi2 = boot_ci(y, p, average_precision_score)
    rec(name, 'full pipeline AUPRC [95% CI]',
        f'{average_precision_score(y, p):.3f} [{lo2:.3f}-{hi2:.3f}]')

    o = operating(y, p)
    rec(name, 'sensitivity / specificity', f"{o['sens']:.1%} / {o['spec']:.1%}")
    rec(name, 'PPV / NPV', f"{o['ppv']:.1%} / {o['npv']:.1%}")
    rec(name, 'accuracy', f"{o['acc']:.1%}")
    rec(name, 'Brier score', f'{brier_score_loss(y, p):.3f}')

    # Stage 1
    n_gk = int(gk_mask.sum())
    ygk, pgk = y[gk_mask], d.gatekeeper_prob.values[gk_mask]
    lo3, hi3 = boot_ci(ygk, pgk)
    rec(name, 'Stage 1 resolved n (% of cohort)',
        f'{n_gk:,} ({n_gk / len(d):.1%})')
    rec(name, 'Stage 1 resolved as positive / negative',
        f'{int((pgk > .75).sum()):,} / {int((pgk < .25).sum()):,}')
    rec(name, 'Stage 1 AUC [95% CI]',
        f'{roc_auc_score(ygk, pgk):.3f} [{lo3:.3f}-{hi3:.3f}]')
    ogk = operating(ygk, pgk)
    rec(name, 'Stage 1 accuracy', f"{ogk['acc']:.1%}")
    rec(name, 'Stage 1 PPV / NPV', f"{ogk['ppv']:.1%} / {ogk['npv']:.1%}")

    # Stage 2
    n_gz = int(gz_mask.sum())
    ygz, pgz = y[gz_mask], p[gz_mask]
    lo4, hi4 = boot_ci(ygz, pgz)
    rec(name, 'Stage 2 (gray zone) n (% of cohort)',
        f'{n_gz:,} ({n_gz / len(d):.1%})')
    rec(name, 'Stage 2 AUC [95% CI]',
        f'{roc_auc_score(ygz, pgz):.3f} [{lo4:.3f}-{hi4:.3f}]')
    ogz = operating(ygz, pgz)
    rec(name, 'Stage 2 accuracy', f"{ogz['acc']:.1%}")
    rec(name, 'Stage 2 sensitivity / specificity',
        f"{ogz['sens']:.1%} / {ogz['spec']:.1%}")
    return d, y, p


adni, y_a, p_a = block('ADNI (development, LOOCV)',
                       RESULTS / 'adni_loocv_predictions.csv')
a4, y_b, p_b = block('A4 + LEARN (external validation)',
                     RESULTS / 'a4_binary_validation_predictions.csv')

# ---- cross-cohort stability, the §4.1 headline ---------------------------
print(f'\n{"=" * 68}\nCROSS-COHORT STABILITY (Discussion 4.1)\n{"=" * 68}')
r1 = (adni.stage != 'reflex').mean()
r2 = (a4.stage != 'reflex').mean()
rec('stability', 'Stage 1 resolution rate ADNI -> A4',
    f'{r1:.1%} -> {r2:.1%}  (difference {abs(r2 - r1) * 100:.1f} points)')
rec('stability', 'full pipeline AUC ADNI -> A4',
    f'{roc_auc_score(y_a, p_a):.3f} -> {roc_auc_score(y_b, p_b):.3f}')
rec('stability', 'Stage 1 AUC ADNI -> A4',
    f"{roc_auc_score(y_a[adni.stage != 'reflex'], adni.gatekeeper_prob[adni.stage != 'reflex']):.3f} -> "
    f"{roc_auc_score(y_b[a4.stage != 'reflex'], a4.gatekeeper_prob[a4.stage != 'reflex']):.3f}")
rec('stability', 'amyloid prevalence ADNI -> A4',
    f'{y_a.mean():.1%} -> {y_b.mean():.1%}')

# ---- calibration ---------------------------------------------------------
import statsmodels.api as sm
lg = np.log(np.clip(p_b, 1e-6, 1 - 1e-6) / np.clip(1 - p_b, 1e-6, 1 - 1e-6))
slope = sm.Logit(y_b, sm.add_constant(lg)).fit(disp=0).params[1]
inter = sm.GLM(y_b, np.ones((len(y_b), 1)),
               family=sm.families.Binomial(), offset=lg).fit().params[0]
rec('calibration', 'A4 Brier / slope / intercept',
    f'{brier_score_loss(y_b, p_b):.3f} / {slope:.2f} / {inter:+.2f}')
rec('calibration', 'A4 mean predicted vs observed',
    f'{p_b.mean():.1%} vs {y_b.mean():.1%}')

# ---- centiloid -----------------------------------------------------------
cl = a4.dropna(subset=['centiloid'])
rho, _ = spearmanr(cl.predicted_prob, cl.centiloid)
rec('centiloid', 'Spearman rho (predicted prob vs centiloid)',
    f'{rho:.3f}  (n = {len(cl):,})')

# ---- feature importance --------------------------------------------------
imp = pd.read_csv(TABLES / 'supp_table_reflex_importance.csv')
print(f'\n{"=" * 68}\nREFLEX FEATURE IMPORTANCE (%, bootstrap CI)\n{"=" * 68}')
for r in imp.sort_values('importance_pct', ascending=False).itertuples():
    rec('features', r.label,
        f'{r.importance_pct:.1f}% [{r.ci_lo_pct:.1f}-{r.ci_hi_pct:.1f}]')

pd.DataFrame(rows).to_csv(TABLES / 'manuscript_number_audit.csv', index=False)
print(f'\nwritten: {TABLES / "manuscript_number_audit.csv"}')
