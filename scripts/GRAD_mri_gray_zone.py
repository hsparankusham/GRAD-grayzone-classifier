#!/usr/bin/env python3
"""
Does MRI volumetry add anything inside GRAD's actual gray zone?

The earlier gray_zone_mri_enhancement.py answered a related but different
question. It re-fitted its own Stage-1 logistic regression on A4's RAW p-tau217,
in-sample, and used the 0.25/0.75 probability band from THAT model. The result
was a 1,044-participant band, not the 727 the deployed pipeline actually routes
to Stage 2 -- a wider, easier set (plasma-only AUC 0.829 there vs 0.731 on the
real gray zone). Its +0.025 AUC gain is therefore not a statement about Stage 2.

This script asks the deployable question instead:

    Among the A4 participants GRAD actually routes to Stage 2, does adding
    ICV-normalised hippocampal and entorhinal volume to the locked 6-feature
    Reflex improve amyloid classification -- and how do those volumes rank
    against the plasma features?

Both arms are LOOCV-refitted within the gray zone with identical Random Forest
settings, so the only difference between them is the two MRI columns.

Outputs:
  results/tables/mri_gray_zone_enhancement.csv
  results/tables/mri_gray_zone_importance.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

from _grad_paths import RESULTS, TABLES, A4_DIR
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features

LOCKED = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
          'APOE4_carrier', 'gfap_tau_interaction']
# Hippocampus alone is the primary addition. Entorhinal is tested and reported
# but not carried: on its own it is null (-0.0005, p = 0.87), and adding it
# alongside hippocampus DILUTES the gain (+0.021 vs +0.025), which is what an
# uninformative extra split candidate does to a depth-limited forest.
MRI = ['Hippocampus_norm']
MRI_ALL = ['Hippocampus_norm', 'Entorhinal_norm']
RF_KW = dict(n_estimators=100, max_depth=5, min_samples_leaf=5,
             random_state=42, n_jobs=-1, class_weight='balanced')
N_BOOT = 2000
RNG = np.random.default_rng(20260204)


def load_mri():
    """ICV-normalised baseline volumes, keyed on BID."""
    m = pd.read_csv(A4_DIR / 'Clinical' / 'External Data' / 'imaging_volumetric_mri.csv')
    m['Hippocampus'] = (m['LeftHippocampus'] + m['RightHippocampus']) / 2
    m['Entorhinal'] = (m['LeftEntorhinal'] + m['RightEntorhinal']) / 2
    m['Hippocampus_norm'] = m['Hippocampus'] / m['IntraCranialVolume'] * 1000
    m['Entorhinal_norm'] = m['Entorhinal'] / m['IntraCranialVolume'] * 1000
    return (m.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()
             [['BID'] + MRI_ALL])


def loocv_auc(X, y):
    """Leave-one-out predictions, refitting scaler and forest in every fold."""
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        sc = StandardScaler().fit(X[tr])
        rf = RandomForestClassifier(**RF_KW).fit(sc.transform(X[tr]), y[tr])
        p[i] = rf.predict_proba(sc.transform(X[i:i + 1]))[0, 1]
    return roc_auc_score(y, p), p


def delong_p(y, p1, p2):
    """DeLong test for two correlated ROC curves on the same participants."""
    y = np.asarray(y).astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    m, n = len(pos), len(neg)

    def structural(p):
        # midrank-free formulation: V10 per positive, V01 per negative
        pp, pn = p[pos][:, None], p[neg][None, :]
        psi = (pp > pn) + .5 * (pp == pn)
        return psi.mean(1), psi.mean(0), psi.mean()

    v10a, v01a, aa = structural(p1)
    v10b, v01b, ab = structural(p2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aa, ab, 1.0
    z = (aa - ab) / np.sqrt(var)
    return aa, ab, float(2 * norm.sf(abs(z)))


# ---- build the gray-zone frame, keyed on BID -----------------------------
adni, a4 = load_adni(), load_a4()
_, a4_h = harmonize(adni, a4)
a4_h = engineer_features(a4_h).reset_index(drop=True)

preds = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
assert len(preds) == len(a4_h), f'{len(preds)} predictions vs {len(a4_h)} rows'
# alignment self-check: the pipeline writes predictions in cohort row order, so
# the stored labels must reproduce the cohort's own amyloid status exactly
ok = preds.true_amyloid.notna()
assert (preds.loc[ok, 'true_amyloid'].values ==
        a4_h.loc[ok, 'amyloid_positive'].values).all(), 'row alignment broken'
a4_h['stage'] = preds['stage'].values

gz = a4_h[a4_h.stage == 'reflex'].merge(load_mri(), on='BID', how='left')
print(f'GRAD gray zone in A4: {len(gz)}')
gz = gz.dropna(subset=MRI_ALL + ['amyloid_positive'])
print(f'  with complete MRI:   {len(gz)}   '
      f'({gz.amyloid_positive.mean():.1%} amyloid-positive)')

y = gz['amyloid_positive'].values.astype(int)


def matrix(cols):
    """Feature matrix with median imputation, as the pipeline does."""
    X = gz[cols].astype(float).values
    for j in range(X.shape[1]):
        nan = np.isnan(X[:, j])
        if nan.any():
            X[nan, j] = np.nanmedian(X[:, j])
    return X


X_plasma, X_both = matrix(LOCKED), matrix(LOCKED + MRI)
auc_a, p_a = loocv_auc(X_plasma, y)
auc_b, p_b = loocv_auc(X_both, y)
_, _, p_delong = delong_p(y, p_b, p_a)

# paired bootstrap on the AUC difference, resampling participants
diffs = []
for _ in range(N_BOOT):
    i = RNG.integers(0, len(y), len(y))
    if len(np.unique(y[i])) < 2:
        continue
    diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
lo, hi = np.percentile(diffs, [2.5, 97.5])

for name, cols in [('+ entorhinal only', LOCKED + ['Entorhinal_norm']),
                   ('+ both regions',    LOCKED + MRI_ALL)]:
    a_, p_ = loocv_auc(matrix(cols), y)
    print(f'  variant {name:20s} AUC {a_:.4f}  delta {a_ - auc_a:+.4f}  '
          f'DeLong p = {delong_p(y, p_, p_a)[2]:.3f}')

print(f'\n  Reflex (6 plasma features)   AUC {auc_a:.4f}')
print(f'  Reflex + hippocampal volume  AUC {auc_b:.4f}')
print(f'  delta                        {auc_b - auc_a:+.4f}  '
      f'[{lo:+.4f}, {hi:+.4f}]   DeLong p = {p_delong:.3f}')
print(f'  bootstrap replicates favouring MRI: {np.mean(np.array(diffs) > 0):.1%}')

# where the volumes rank once they are in the model
sc = StandardScaler().fit(X_both)
rf = RandomForestClassifier(**RF_KW).fit(sc.transform(X_both), y)
imp = (pd.DataFrame({'feature': LOCKED + MRI, 'importance': rf.feature_importances_})
       .sort_values('importance', ascending=False))
print('\n  feature importance with MRI included:')
print(imp.to_string(index=False))

pd.DataFrame([dict(gray_zone_n=len(gz), prevalence=y.mean(),
                   auc_plasma=auc_a, auc_plasma_mri=auc_b,
                   delta_auc=auc_b - auc_a, ci_low=lo, ci_high=hi,
                   delong_p=p_delong,
                   prop_boot_favouring_mri=float(np.mean(np.array(diffs) > 0)))]
             ).to_csv(TABLES / 'mri_gray_zone_enhancement.csv', index=False)
imp.to_csv(TABLES / 'mri_gray_zone_importance.csv', index=False)
