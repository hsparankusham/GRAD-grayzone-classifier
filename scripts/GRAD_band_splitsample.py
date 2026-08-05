#!/usr/bin/env python3
"""
Split-sample test: does a high-gain gray-zone band survive out-of-sample?

An exhaustive search inside A4 found a band ([0.50, 0.70]) where p-tau217 sits
at chance (AUC 0.521) and Stage 2 reaches 0.679 -- but permuting the labels and
repeating the same 111-band search showed noise alone produces a best-delta of
+0.094 (median) to +0.202 (95th pct), against an observed +0.158, p = 0.142.
Selecting the band on p-tau217's weakness in the same data that measures Stage
2's advantage manufactures the gap.

This removes that circularity:

    SELECT   the band in ADNI (development), on ADNI data only
    TEST     that single frozen band once, in A4

The A4 estimate is then uncontaminated, because nothing about A4 informed the
choice. One band, one test, no multiplicity.

Reported alongside: the pre-specified [0.25, 0.75] band, which is the honest
benchmark either way.

Output: results/tables/band_splitsample.csv
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
RNG = np.random.default_rng(23)
MIN_N_SELECT = 30      # ADNI's complete-panel gray zone is small
MIN_N_TEST = 60


def boot_delta(y, a, b, n=2000):
    d = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()


# ---- data ---------------------------------------------------------------
adni, a4 = load_adni(), load_a4()
adni_h, a4_h = harmonize(adni, a4)
adni_h = engineer_features(adni_h).reset_index(drop=True)
a4_h = engineer_features(a4_h).reset_index(drop=True)
for df, f in ((adni_h, 'adni_loocv_predictions.csv'),
              (a4_h, 'a4_binary_validation_predictions.csv')):
    p = pd.read_csv(RESULTS / f)
    df['stage'] = p['stage'].values
    df['gk'] = p['gatekeeper_prob'].values
    df['y'] = p['true_amyloid'].values

tr = adni_h[(adni_h.stage == 'reflex') & adni_h[LOCKED].notna().all(axis=1)].copy()
te = a4_h[(a4_h.stage == 'reflex') & a4_h[LOCKED].notna().all(axis=1)].copy()

# Stage 2 trained once on ADNI complete-panel cases, no imputation
sc = StandardScaler().fit(tr[LOCKED].values)
rf = RandomForestClassifier(**RF_KW).fit(sc.transform(tr[LOCKED].values),
                                         tr.y.values.astype(int))
te['s2'] = rf.predict_proba(sc.transform(te[LOCKED].values))[:, 1]

# ADNI's own Stage 2 scores must be out-of-fold, or band selection sees the
# training labels; use leave-one-out on the same 82 participants
loo = np.empty(len(tr))
Xtr, ytr = tr[LOCKED].values, tr.y.values.astype(int)
for i in range(len(tr)):
    k = np.ones(len(tr), bool); k[i] = False
    s = StandardScaler().fit(Xtr[k])
    m = RandomForestClassifier(**RF_KW).fit(s.transform(Xtr[k]), ytr[k])
    loo[i] = m.predict_proba(s.transform(Xtr[i:i + 1]))[0, 1]
tr['s2'] = loo

# ---- SELECT the band, in ADNI only --------------------------------------
GRID = [(lo, hi) for lo in np.arange(.25, .51, .025)
        for hi in np.arange(.50, .76, .025) if hi - lo >= .08]
cands = []
for lo, hi in GRID:
    m = tr.gk.between(lo, hi).values
    if m.sum() < MIN_N_SELECT or len(np.unique(ytr[m])) < 2:
        continue
    d = (roc_auc_score(ytr[m], tr.s2.values[m])
         - roc_auc_score(ytr[m], tr.gk.values[m]))
    cands.append((d, lo, hi, int(m.sum())))
cands.sort(reverse=True)
d_sel, LO, HI, n_sel = cands[0]

print('SPLIT-SAMPLE BAND TEST')
print(f'  selection set : ADNI complete-panel gray zone, n = {len(tr)}')
print(f'  test set      : A4   complete-panel gray zone, n = {len(te)}')
print(f'  candidates considered in ADNI: {len(cands)}')
print(f'\n  BAND SELECTED IN ADNI: [{LO:.3f}, {HI:.3f}]   '
      f'(ADNI n = {n_sel}, ADNI delta {d_sel:+.3f})')
print('  top 5 ADNI candidates:')
for d, lo, hi, n in cands[:5]:
    print(f'     [{lo:.3f}, {hi:.3f}]  n={n:>3d}  delta {d:+.3f}')

# ---- TEST it once, in A4 ------------------------------------------------
rows = []
print('\n  --- frozen band, tested once in A4 ---')
m = te.gk.between(LO, HI).values
y = te.y.values.astype(int)[m]
if m.sum() < MIN_N_TEST or len(np.unique(y)) < 2:
    print(f'  band holds only {m.sum()} A4 participants -- not evaluable')
else:
    ap = roc_auc_score(y, te.gk.values[m])
    ag = roc_auc_score(y, te.s2.values[m])
    mu, (lo_ci, hi_ci), frac = boot_delta(y, te.gk.values[m], te.s2.values[m])
    print(f'  A4 n = {m.sum()}   amyloid-positive {y.mean():.1%}')
    print(f'     p-tau217 AUC {ap:.3f}   Stage 2 AUC {ag:.3f}')
    print(f'     delta {ag - ap:+.3f}   95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]'
          f'   favouring Stage 2 in {frac:.1%}')
    rows.append(dict(band=f'[{LO:.3f},{HI:.3f}] selected in ADNI', n=int(m.sum()),
                     auc_ptau=ap, auc_stage2=ag, delta=ag - ap,
                     ci_low=lo_ci, ci_high=hi_ci))

# ---- benchmark: the pre-specified band ----------------------------------
print('\n  --- pre-specified [0.25, 0.75], for comparison ---')
y_all = te.y.values.astype(int)
ap = roc_auc_score(y_all, te.gk.values)
ag = roc_auc_score(y_all, te.s2.values)
mu, (lo_ci, hi_ci), frac = boot_delta(y_all, te.gk.values, te.s2.values)
print(f'  A4 n = {len(te)}   p-tau217 AUC {ap:.3f}   Stage 2 AUC {ag:.3f}')
print(f'     delta {ag - ap:+.3f}   95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]'
      f'   favouring Stage 2 in {frac:.1%}')
rows.append(dict(band='[0.25,0.75] pre-specified', n=len(te), auc_ptau=ap,
                 auc_stage2=ag, delta=ag - ap, ci_low=lo_ci, ci_high=hi_ci))

pd.DataFrame(rows).to_csv(TABLES / 'band_splitsample.csv', index=False)
print(f'\nwritten: {TABLES / "band_splitsample.csv"}')
