#!/usr/bin/env python3
"""
What does a six-feature Stage 2 actually do, with no imputation anywhere?

The published Stage 2 was both TRAINED and TESTED on partly fabricated inputs.
A4 measured GFAP and Abeta42/40 on only ~57% of participants and ADNI on ~66%,
so `GFAP_Z`, `tau_ab42_diff` and `gfap_tau_interaction` were median-imputed for
34% of the training gray zone and 42% of the validation gray zone. A repeated
median is a dense spike in the column, which both flattens the structure the
forest is meant to learn and mis-scores the patients it is applied to -- and in
A4 the missingness is strongly informative (85.1% amyloid-positive among those
lacking the panel, versus 53.2% among those with it).

This script removes imputation entirely:

    train  ADNI gray-zone participants with all six features measured
    test   A4   gray-zone participants with all six features measured

and compares against p-tau217 alone on the identical test set, plus the
published imputation-trained model as a reference arm.

Output: results/tables/stage2_complete_panel.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from _grad_paths import RESULTS, TABLES
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features

LOCKED = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
          'APOE4_carrier', 'gfap_tau_interaction']
RF_KW = dict(n_estimators=100, max_depth=5, min_samples_leaf=5,
             random_state=42, n_jobs=-1, class_weight='balanced')
BANDS = [(.45, .55), (.42, .58), (.40, .60), (.35, .65), (.30, .70), (.25, .75)]
RNG = np.random.default_rng(11)


def boot_delta(y, a, b, n=2000):
    """Paired bootstrap on AUC(b) - AUC(a)."""
    d = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()


# ---- build both gray zones, flagged for panel completeness ---------------
adni, a4 = load_adni(), load_a4()
adni_h, a4_h = harmonize(adni, a4)
adni_h = engineer_features(adni_h).reset_index(drop=True)
a4_h = engineer_features(a4_h).reset_index(drop=True)

for df, f in ((adni_h, 'adni_loocv_predictions.csv'),
              (a4_h, 'a4_binary_validation_predictions.csv')):
    p = pd.read_csv(RESULTS / f)
    df['stage'] = p['stage'].values
    df['gk'] = p['gatekeeper_prob'].values
    df['published'] = p['predicted_prob'].values     # imputation-trained output
    df['y'] = p['true_amyloid'].values

# complete = every one of the six features actually measured
train = adni_h[(adni_h.stage == 'reflex') & adni_h[LOCKED].notna().all(axis=1)]
test = a4_h[(a4_h.stage == 'reflex') & a4_h[LOCKED].notna().all(axis=1)]

print('NO-IMPUTATION STAGE 2')
print(f'  train  ADNI gray zone, complete panel : {len(train):>4d} of '
      f'{int((adni_h.stage == "reflex").sum())}   '
      f'({train.y.mean():.1%} amyloid-positive)')
print(f'  test   A4   gray zone, complete panel : {len(test):>4d} of '
      f'{int((a4_h.stage == "reflex").sum())}   '
      f'({test.y.mean():.1%} amyloid-positive)')

X_tr, y_tr = train[LOCKED].astype(float).values, train.y.values.astype(int)
X_te, y_te = test[LOCKED].astype(float).values, test.y.values.astype(int)
assert not np.isnan(X_tr).any() and not np.isnan(X_te).any(), 'imputation leaked in'

sc = StandardScaler().fit(X_tr)
rf = RandomForestClassifier(**RF_KW).fit(sc.transform(X_tr), y_tr)
p_clean = rf.predict_proba(sc.transform(X_te))[:, 1]
p_ptau = test.gk.values
p_pub = test.published.values

print(f'\nOn the identical {len(test)} complete-panel A4 participants:')
rows = []
for lab, p in [('p-tau217 alone', p_ptau),
               ('Stage 2, published (imputation-trained)', p_pub),
               ('Stage 2, clean (no imputation anywhere)', p_clean)]:
    auc = roc_auc_score(y_te, p)
    print(f'   {lab:42s} AUC {auc:.4f}')
    rows.append(dict(arm=lab, n=len(test), auc=auc))

for lab, p in [('published', p_pub), ('clean', p_clean)]:
    m, (lo, hi), frac = boot_delta(y_te, p_ptau, p)
    print(f'   delta vs p-tau217, {lab:10s}: {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]'
          f'   favouring Stage 2 in {frac:.1%}')

m, (lo, hi), frac = boot_delta(y_te, p_pub, p_clean)
print(f'   clean vs published                        : {m:+.4f}  '
      f'[{lo:+.4f}, {hi:+.4f}]   clean better in {frac:.1%}')

# ---- does the advantage still concentrate where p-tau217 is ambiguous? ---
print('\nBand sweep with the clean model (nested bands -- estimates are correlated)')
print(f"{'band':>14s}{'n':>6s}{'Abeta+':>8s}{'p-tau217':>10s}{'clean S2':>10s}"
      f"{'delta':>9s}{'95% CI':>21s}")
t = test.copy(); t['clean'] = p_clean
for lo_b, hi_b in BANDS:
    s = t[t.gk.between(lo_b, hi_b)]
    if len(s) < 40 or s.y.nunique() < 2:
        print(f'  [{lo_b:.2f},{hi_b:.2f}]{len(s):>6d}   too small'); continue
    y = s.y.values.astype(int)
    ap, ag = roc_auc_score(y, s.gk.values), roc_auc_score(y, s.clean.values)
    _, (cl, ch), _ = boot_delta(y, s.gk.values, s.clean.values, n=1500)
    print(f'  [{lo_b:.2f},{hi_b:.2f}]{len(s):>6d}{y.mean():>8.1%}{ap:>10.3f}'
          f'{ag:>10.3f}{ag - ap:>+9.3f}   [{cl:+.3f}, {ch:+.3f}]')
    rows.append(dict(arm=f'band [{lo_b},{hi_b}] clean', n=len(s),
                     auc=ag, auc_ptau=ap, delta=ag - ap, ci_low=cl, ci_high=ch))

pd.DataFrame(rows).to_csv(TABLES / 'stage2_complete_panel.csv', index=False)
print(f'\nwritten: {TABLES / "stage2_complete_panel.csv"}')
