#!/usr/bin/env python3
"""
Does hippocampal volume improve Stage 2 -- trained on ADNI, validated in A4?

This is the deployable test: the Reflex is fitted on ADNI's gray zone and
applied to A4's, exactly as the published pipeline does, with and without
hippocampal volume. GRAD_mri_gray_zone.py answers the narrower incremental-value
question by refitting inside A4; this one answers whether the addition survives
transport.

TWO CONVENTIONS HAD TO BE RECONCILED FIRST
------------------------------------------
ADNIMERGE reports `Hippocampus` as the BILATERAL SUM in mm^3 against an ICV in
mm^3 (median 6,872 / 1,510,050). A4 reports Left and Right separately in mL
against an ICV in mL (median ~3.1 + ~3.3 / ~1,251). Units cancel inside the
ratio, but laterality does not: the earlier script averaged A4's two sides while
ADNI's column is a sum, which alone put the two cohorts a factor of two apart
(4.85 vs 2.09 per mille). Summing both sides in A4 brings them to 4.55 vs 5.12.

The residual ~12% is what reference-anchored Z-scoring exists to absorb, so the
volume is harmonised the same way the plasma markers are: standardised against
each cohort's own cognitively-unimpaired, amyloid-negative participants. That
keeps the MRI feature on the identical footing as pTau217_Z and GFAP_Z.

Outputs:
  results/tables/mri_external_validation.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from _grad_paths import RESULTS, TABLES, A4_DIR
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features
from GRAD_mri_gray_zone import LOCKED, RF_KW, delong_p

MRI_Z = 'Hippocampus_Z'
N_BOOT = 2000
RNG = np.random.default_rng(20260204)


def a4_hippocampus():
    """A4 baseline hippocampal volume as a BILATERAL SUM, matching ADNIMERGE."""
    m = pd.read_csv(A4_DIR / 'Clinical' / 'External Data' / 'imaging_volumetric_mri.csv')
    m['Hippocampus_norm'] = ((m['LeftHippocampus'] + m['RightHippocampus'])
                             / m['IntraCranialVolume'] * 1000)
    return (m.sort_values(['BID', 'VISCODE']).groupby('BID').first()
             .reset_index()[['BID', 'Hippocampus_norm']])


def anchor(df, col, ref_mask):
    """Z-score against this cohort's own CU, amyloid-negative reference."""
    ref = df.loc[ref_mask, col].dropna()
    return (df[col] - ref.mean()) / (ref.std() if ref.std() > 0 else 1.0)


# ---- ADNI: harmonised plasma + harmonised hippocampal volume -------------
adni, a4 = load_adni(), load_a4()
adni_h, a4_h = harmonize(adni, a4)
adni_h = engineer_features(adni_h).reset_index(drop=True)
a4_h = engineer_features(a4_h).reset_index(drop=True)

adni_h['Hippocampus_norm'] = adni_h['Hippocampus'] / adni_h['ICV'] * 1000
adni_h[MRI_Z] = anchor(adni_h, 'Hippocampus_norm',
                       adni_h['DX'].isin(['CN', 'NL', 'Normal']) &
                       (adni_h['amyloid_positive'] == 0))

a4_h = a4_h.merge(a4_hippocampus(), on='BID', how='left')
# A4 + LEARN is cognitively unimpaired throughout, so the reference is simply
# its amyloid-negative participants -- the same rule the plasma harmonisation
# uses under LOCAL_REFERENCE
a4_h[MRI_Z] = anchor(a4_h, 'Hippocampus_norm', a4_h['amyloid_positive'] == 0)

print('ICV-normalised hippocampus after matching laterality convention:')
for nm, d in (('ADNI', adni_h), ('A4  ', a4_h)):
    print(f'  {nm}: median {d.Hippocampus_norm.median():.3f}  '
          f'reference-anchored Z median {d[MRI_Z].median():+.3f}')

# ---- restrict both cohorts to their gray zones ---------------------------
ap = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
a4p = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
adni_h['stage'], a4_h['stage'] = ap['stage'].values, a4p['stage'].values

train = adni_h[(adni_h.stage == 'reflex') & adni_h[MRI_Z].notna()]
test = a4_h[(a4_h.stage == 'reflex') & a4_h[MRI_Z].notna()]
print(f'\ntrain: ADNI gray zone with MRI  {len(train)} of '
      f'{(adni_h.stage == "reflex").sum()}')
print(f'test : A4   gray zone with MRI  {len(test)} of '
      f'{(a4_h.stage == "reflex").sum()}')

y_tr = train['amyloid_positive'].values.astype(int)
y_te = test['amyloid_positive'].values.astype(int)


def fit_apply(cols):
    """Train on the ADNI gray zone, score the A4 gray zone. Median-impute."""
    Xtr, Xte = train[cols].astype(float).values, test[cols].astype(float).values
    for j in range(Xtr.shape[1]):
        med = np.nanmedian(Xtr[:, j])
        Xtr[np.isnan(Xtr[:, j]), j] = med
        Xte[np.isnan(Xte[:, j]), j] = med        # training median, not test's
    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(**RF_KW).fit(sc.transform(Xtr), y_tr)
    return rf.predict_proba(sc.transform(Xte))[:, 1], rf


p_plasma, _ = fit_apply(LOCKED)
p_mri, rf_mri = fit_apply(LOCKED + [MRI_Z])
auc_a, auc_b = roc_auc_score(y_te, p_plasma), roc_auc_score(y_te, p_mri)
_, _, pv = delong_p(y_te, p_mri, p_plasma)

diffs = []
for _ in range(N_BOOT):
    i = RNG.integers(0, len(y_te), len(y_te))
    if len(np.unique(y_te[i])) < 2:
        continue
    diffs.append(roc_auc_score(y_te[i], p_mri[i]) - roc_auc_score(y_te[i], p_plasma[i]))
lo, hi = np.percentile(diffs, [2.5, 97.5])

print(f'\nEXTERNAL VALIDATION (ADNI-trained -> A4 gray zone, n = {len(y_te)})')
print(f'  Reflex, 6 plasma features     AUC {auc_a:.4f}')
print(f'  Reflex + hippocampal volume   AUC {auc_b:.4f}')
print(f'  delta                         {auc_b - auc_a:+.4f}  '
      f'[{lo:+.4f}, {hi:+.4f}]   DeLong p = {pv:.4f}')
print(f'  bootstrap replicates favouring MRI: {np.mean(np.array(diffs) > 0):.1%}')

imp = (pd.DataFrame({'feature': LOCKED + [MRI_Z],
                     'importance': rf_mri.feature_importances_})
       .sort_values('importance', ascending=False))
print('\n  importance in the ADNI-trained model:')
print(imp.to_string(index=False))

pd.DataFrame([dict(train_n=len(y_tr), test_n=len(y_te),
                   auc_plasma=auc_a, auc_plasma_mri=auc_b,
                   delta_auc=auc_b - auc_a, ci_low=lo, ci_high=hi, delong_p=pv,
                   prop_boot_favouring_mri=float(np.mean(np.array(diffs) > 0)))]
             ).to_csv(TABLES / 'mri_external_validation.csv', index=False)
