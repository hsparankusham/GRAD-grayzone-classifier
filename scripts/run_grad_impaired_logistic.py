#!/usr/bin/env python3
"""
GRAD -- impaired-restricted cohort, regularised-logistic Stage 2.
================================================================

Regenerates every number in manuscript/GRAD_ACTN_v3.md. Nothing in the
manuscript should be edited by hand; edit this script and rerun it.

    python3 scripts/run_grad_impaired_logistic.py

Design (frozen in manuscript/ANALYSIS_SPINE.md, revised 2026-08-18):

    Stage 1  univariate logistic on pTau217_Z; resolve P < 0.25 and P > 0.75
    Stage 2  L2 logistic (C = 0.1, balanced weights), seven features
    ADNI     restricted to MCI + dementia (n = 145), LOOCV, harmonisation
             recomputed inside every fold, training-median imputation
    A4       scored once by the ADNI-fitted model, local reference harmonisation

Outputs
    results/tables/grad_v2_numbers.json      every reported number
    results/adni_loocv_predictions_v2.csv    per-participant LOOCV predictions
    results/a4_predictions_v2.csv            per-participant external predictions
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from scipy import stats
import statsmodels.api as sm

from _grad_paths import ADNI_DIR, A4_DIR, RESULTS  # noqa: E402

AV45_THRESHOLD = 1.11
CL_THRESHOLD = 20
GK_LOW, GK_HIGH = 0.25, 0.75
BAND_LOW, BAND_HIGH = 0.40, 0.60

CN_DX = ['CN', 'NL', 'Normal']
MCI_DX = ['MCI', 'EMCI', 'LMCI']
DEM_DX = ['Dementia', 'AD']
IMPAIRED_DX = MCI_DX + DEM_DX

FEATURES = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
            'APOE4_carrier', 'gfap_tau_interaction', 'AB4240_log']

HARMONISED = ['pTau217_raw', 'NfL_raw', 'GFAP_raw']


# ---------------------------------------------------------------- loading ---

def load_adni():
    adni = str(ADNI_DIR)
    upenn = pd.read_csv(os.path.join(adni, 'Pathology', 'UPENN_PlasmaBiomarkers.csv'))
    upenn = upenn.rename(columns={
        'pT217_F': 'pTau217_raw', 'AB42_F': 'AB42_raw', 'AB40_F': 'AB40_raw',
        'AB42_AB40_F': 'AB42_40_ratio', 'NfL_Q': 'NfL_raw', 'GFAP_Q': 'GFAP_raw'})
    for col in ['pTau217_raw', 'AB42_raw', 'AB40_raw', 'AB42_40_ratio', 'NfL_raw', 'GFAP_raw']:
        upenn[col] = pd.to_numeric(upenn[col], errors='coerce')
        upenn.loc[upenn[col] < 0, col] = np.nan
    upenn['assay'] = 'UPENN'
    upenn['PTID'] = upenn['PTID'].astype(str)

    janssen = pd.read_csv(os.path.join(adni, 'Pathology', 'JANSSEN_PLASMA_P217_TAU_18Dec2025.csv'))
    janssen = janssen.rename(columns={'DILUTION_CORRECTED_CONC': 'pTau217_raw'})
    janssen['pTau217_raw'] = pd.to_numeric(janssen['pTau217_raw'], errors='coerce')
    janssen['assay'] = 'Janssen'
    janssen['PTID'] = janssen['PTID'].astype(str)

    merge = pd.read_csv(os.path.join(adni, 'ADNIMERGE2025.csv'), low_memory=False)
    merge['AV45'] = pd.to_numeric(merge['AV45'], errors='coerce')
    merge['AV45_bl'] = pd.to_numeric(merge.get('AV45_bl', pd.Series(dtype=float)), errors='coerce')
    merge['amyloid_positive'] = (merge['AV45'] > AV45_THRESHOLD).astype(float)
    merge.loc[merge['AV45'].isna(), 'amyloid_positive'] = np.nan
    fallback = merge['AV45'].isna() & merge['AV45_bl'].notna()
    merge.loc[fallback, 'amyloid_positive'] = (merge.loc[fallback, 'AV45_bl'] > AV45_THRESHOLD).astype(float)
    merge['PTID'] = merge['PTID'].astype(str)

    def visit_match(df):
        return df[(df['VISCODE2'] == df['VISCODE']) |
                  (df['VISCODE2'].str.contains('bl', case=False, na=False) &
                   df['VISCODE'].str.contains('bl', case=False, na=False))].copy()

    upenn_m = visit_match(upenn.merge(merge, on=['PTID', 'RID'], how='inner', suffixes=('', '_merge')))
    janssen_m = visit_match(janssen.merge(merge, on=['PTID', 'RID'], how='inner'))
    janssen_only = janssen_m[~janssen_m['PTID'].isin(upenn_m['PTID'])]

    combined = pd.concat([upenn_m, janssen_only], ignore_index=True)
    combined = combined.sort_values(['PTID', 'VISCODE2']).groupby('PTID').first().reset_index()
    combined = combined[combined['amyloid_positive'].notna() & combined['pTau217_raw'].notna()].copy()

    combined['APOE4'] = pd.to_numeric(combined.get('APOE4', pd.Series(dtype=float)), errors='coerce')
    combined['APOE4_carrier'] = (combined['APOE4'] > 0).astype(float)
    combined.loc[combined['APOE4'].isna(), 'APOE4_carrier'] = np.nan
    combined['SEX_binary'] = (combined['PTGENDER'] == 'Male').astype(int)
    combined['amyloid_positive'] = combined['amyloid_positive'].astype(int)
    return combined.reset_index(drop=True)


def load_a4():
    a4 = str(A4_DIR)
    ptau = pd.read_csv(os.path.join(a4, 'Clinical', 'External Data', 'biomarker_pTau217.csv'))
    ptau['pTau217_raw'] = pd.to_numeric(ptau['ORRES'], errors='coerce')
    miss = ptau['pTau217_raw'].isna()
    ptau.loc[miss, 'pTau217_raw'] = pd.to_numeric(ptau.loc[miss, 'ORRESRAW'], errors='coerce')
    ptau['assay'] = 'Lilly'
    ptau_bl = ptau.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()

    roche = pd.read_csv(os.path.join(a4, 'Clinical', 'External Data', 'biomarker_Plasma_Roche_Results.csv'))
    roche['LABRESN'] = pd.to_numeric(roche['LABRESN'], errors='coerce')
    roche['LBTESTCD'] = roche['LBTESTCD'].str.strip()
    pivot = roche.pivot_table(index=['BID', 'VISCODE'], columns='LBTESTCD',
                              values='LABRESN', aggfunc='first').reset_index()
    pivot = pivot.rename(columns={'GFAP': 'GFAP_raw', 'NF-L': 'NfL_raw',
                                  'AMYLB42': 'AB42_raw', 'AMYLB40': 'AB40_raw'})
    if 'AB42_raw' in pivot.columns and 'AB40_raw' in pivot.columns:
        pivot['AB42_40_ratio'] = pivot['AB42_raw'] / (pivot['AB40_raw'] * 1000)
    if 'GFAP_raw' in pivot.columns:
        pivot['GFAP_raw'] = pivot['GFAP_raw'] * 1000
    roche_bl = pivot.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()

    subj = pd.read_csv(os.path.join(a4, 'Clinical', 'Derived Data', 'SUBJINFO.csv'), low_memory=False)
    cols = ['BID', 'TX', 'AGEYR', 'SEX', 'EDCCNTU', 'APOEGNPRSNFLG', 'SUVRCER', 'AMYLCENT', 'MMSETSV1']
    subj = subj[[c for c in cols if c in subj.columns]]
    subj = subj.rename(columns={'AGEYR': 'AGE', 'EDCCNTU': 'PTEDUCAT',
                                'APOEGNPRSNFLG': 'APOE4_carrier', 'MMSETSV1': 'MMSE'})
    subj['AMYLCENT'] = pd.to_numeric(subj['AMYLCENT'], errors='coerce')
    subj['amyloid_positive'] = (subj['AMYLCENT'] >= CL_THRESHOLD).astype(float)
    subj.loc[subj['AMYLCENT'].isna(), 'amyloid_positive'] = np.nan

    combined = subj.merge(ptau_bl[['BID', 'pTau217_raw', 'assay']], on='BID', how='inner')
    keep = ['BID'] + [c for c in ['GFAP_raw', 'NfL_raw', 'AB42_raw', 'AB40_raw', 'AB42_40_ratio']
                      if c in roche_bl.columns]
    combined = combined.merge(roche_bl[keep], on='BID', how='left')
    combined = combined[combined['pTau217_raw'].notna() & combined['amyloid_positive'].notna()].copy()
    combined['amyloid_positive'] = combined['amyloid_positive'].astype(int)
    combined['SEX_binary'] = (combined['SEX'] == 1).astype(int)
    return combined.reset_index(drop=True)


# --------------------------------------------------------- harmonisation ---

def _reference_stats(df, biomarker, cn_only=True):
    """mu/sigma per assay from cognitively unimpaired, amyloid-negative participants."""
    mask = df['amyloid_positive'] == 0
    if cn_only and 'DX' in df.columns:
        dx = df['DX'].isin(CN_DX)
        if 'DX_bl' in df.columns:
            dx = dx | df['DX_bl'].isin(CN_DX)
        mask = mask & dx
    out = {}
    pooled = np.log1p(df.loc[mask, biomarker].dropna())
    p_mean = pooled.mean() if len(pooled) else np.nan
    p_std = pooled.std() if len(pooled) > 1 and pooled.std() > 0 else 1.0
    for assay in df['assay'].dropna().unique():
        vals = np.log1p(df.loc[mask & (df['assay'] == assay), biomarker].dropna())
        if len(vals) >= 5:
            out[assay] = (vals.mean(), vals.std() if vals.std() > 0 else 1.0)
        else:
            out[assay] = (p_mean, p_std)
    return out


def harmonise(df, ref_source, cn_only=True):
    """Z = (log1p(x) - mu_ref)/sigma_ref, reference drawn from ref_source, per assay."""
    out = df.copy()
    for bm in HARMONISED:
        if bm not in df.columns or bm not in ref_source.columns:
            continue
        z = bm.replace('_raw', '_Z')
        stats = _reference_stats(ref_source, bm, cn_only=cn_only)
        out[z] = np.nan
        for assay in df['assay'].dropna().unique():
            if assay not in stats or not np.isfinite(stats[assay][0]):
                continue
            mu, sd = stats[assay]
            m = df['assay'] == assay
            out.loc[m, z] = (np.log1p(df.loc[m, bm]) - mu) / sd
    return out


def engineer(df):
    out = df.copy()
    if 'pTau217_raw' in df.columns and 'AB42_40_ratio' in df.columns:
        out['tau_ab42_diff'] = np.log1p(df['pTau217_raw']) - np.log1p(df['AB42_40_ratio'])
        out['AB4240_log'] = np.log1p(df['AB42_40_ratio'])
    if 'GFAP_Z' in df.columns and 'pTau217_Z' in df.columns:
        out['gfap_tau_interaction'] = df['GFAP_Z'] * df['pTau217_Z']
    return out


# -------------------------------------------------------------- modelling ---

def fit_stage2(train_df, train_y, features=FEATURES, C=0.1):
    """Returns (model, scaler, medians, features_used)."""
    used = [f for f in features if f in train_df.columns]
    X = train_df[used].astype(float).values.copy()
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=2000,
                               class_weight='balanced', random_state=42)
    model.fit(scaler.transform(X), train_y)
    return model, scaler, medians, used


def score_stage2(model, scaler, medians, used, test_df):
    X = test_df[used].astype(float).values.copy()
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]
    return model.predict_proba(scaler.transform(X))[:, 1]


def fit_stage1(train_df, train_y):
    X = train_df[['pTau217_Z']].values
    ok = ~np.isnan(X.flatten())
    m = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    m.fit(X[ok], np.asarray(train_y)[ok])
    return m


def run_loocv(full_adni, analysis_idx, features=FEATURES, C=0.1, verbose=True):
    """LOOCV over analysis_idx (positions in full_adni). Harmonisation per fold."""
    n = len(analysis_idx)
    rows = []
    for k, i in enumerate(analysis_idx):
        train_full = full_adni.drop(index=i)
        test_row = full_adni.loc[[i]]

        train_h = harmonise(train_full, train_full)
        test_h = harmonise(test_row, train_full)
        train_h = engineer(train_h)
        test_h = engineer(test_h)

        train_imp = train_h[train_h.index.isin(analysis_idx)]
        train_y = train_imp['amyloid_positive'].values

        gk = fit_stage1(train_imp, train_y)
        p_test = gk.predict_proba(test_h[['pTau217_Z']].values)[:, 1][0]
        p_train = gk.predict_proba(train_imp[['pTau217_Z']].values)[:, 1]

        if p_test < GK_LOW or p_test > GK_HIGH:
            rows.append(dict(idx=i, y=int(test_row['amyloid_positive'].iloc[0]),
                             gk_prob=p_test, final_prob=p_test, stage='gatekeeper'))
            continue

        gz = (p_train >= GK_LOW) & (p_train <= GK_HIGH)
        m2, sc, med, used = fit_stage2(train_imp[gz], train_y[gz], features=features, C=C)
        p2 = score_stage2(m2, sc, med, used, test_h)[0]
        rows.append(dict(idx=i, y=int(test_row['amyloid_positive'].iloc[0]),
                         gk_prob=p_test, final_prob=p2, stage='reflex'))
        if verbose and (k + 1) % 25 == 0:
            print(f"    ...{k + 1}/{n}", flush=True)
    return pd.DataFrame(rows)


def fit_full_and_score_a4(full_adni, analysis_idx, a4, features=FEATURES, C=0.1):
    adni_h = engineer(harmonise(full_adni, full_adni))
    a4_h = engineer(harmonise(a4, a4, cn_only=False))   # all A4 participants are CU

    train_imp = adni_h.loc[analysis_idx]
    y = train_imp['amyloid_positive'].values
    gk = fit_stage1(train_imp, y)

    p_train = gk.predict_proba(train_imp[['pTau217_Z']].values)[:, 1]
    gz = (p_train >= GK_LOW) & (p_train <= GK_HIGH)
    m2, sc, med, used = fit_stage2(train_imp[gz], y[gz], features=features, C=C)

    p_a4 = gk.predict_proba(a4_h[['pTau217_Z']].values)[:, 1]
    final = p_a4.copy()
    stage = np.where((p_a4 >= GK_LOW) & (p_a4 <= GK_HIGH), 'reflex', 'gatekeeper')
    gz_a4 = stage == 'reflex'
    final[gz_a4] = score_stage2(m2, sc, med, used, a4_h[gz_a4])

    out = pd.DataFrame({
        'y': a4_h['amyloid_positive'].values,
        'gk_prob': p_a4, 'final_prob': final, 'stage': stage,
        'centiloid': a4_h['AMYLCENT'].values,
    })
    return out, dict(gatekeeper=gk, stage2=m2, scaler=sc, medians=med,
                     features=used, adni_h=adni_h, a4_h=a4_h, train_gz=train_imp[gz])


RNG = np.random.RandomState(42)
NBOOT = 2000


# ------------------------------------------------------------------ stats ---

def boot_ci(y, p, fn, nboot=NBOOT, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(nboot):
        i = rng.choice(n, n, replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float)
    T2[J] = T
    return T2


def delong_p(y, p1, p2):
    """DeLong test for two correlated ROC curves."""
    y = np.asarray(y)
    order = np.argsort(-y)
    y = y[order]
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum())
    n = len(y) - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]

    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])

    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    s = s01 / m + s10 / n
    L = np.array([[1, -1]])
    num = float(L @ auc)
    den = float(np.sqrt(L @ s @ L.T))
    if den == 0:
        return auc[0], auc[1], 1.0
    z = num / den
    return float(auc[0]), float(auc[1]), float(2 * stats.norm.sf(abs(z)))


def mcnemar(correct_a, correct_b):
    """b = new score. Returns (a_wrong_b_right, a_right_b_wrong, p)."""
    b01 = int(((~correct_a) & correct_b).sum())
    b10 = int((correct_a & (~correct_b)).sum())
    if b01 + b10 == 0:
        return b01, b10, 1.0
    p = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue)
    return b01, b10, p


def binary_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(threshold=float(thr), n=len(y), TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=(tp + tn) / len(y), sensitivity=sens, specificity=spec,
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                npv=tn / (tn + fn) if tn + fn else np.nan,
                lr_pos=sens / (1 - spec) if spec < 1 else np.inf,
                lr_neg=(1 - sens) / spec if spec > 0 else np.nan)


def operating_point(y, p, target, mode):
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    if mode == 'sens':
        ok = np.where(tpr >= target)[0]
        idx = ok[np.argmin(fpr[ok])]
    else:
        ok = np.where(spec >= target)[0]
        idx = ok[np.argmax(tpr[ok])]
    return binary_metrics(y, p, thr[idx])


def calibration(y, p):
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)
    return dict(intercept=float(fit.params[0]), slope=float(fit.params[1]),
                brier=float(brier_score_loss(y, p)),
                mean_predicted=float(p.mean()), observed=float(y.mean()))


def band_outcome(y, p, lo=BAND_LOW, hi=BAND_HIGH):
    ind = (p >= lo) & (p <= hi)
    call = (p > hi).astype(int)
    correct = (~ind) & (call == y)
    wrong = (~ind) & (call != y)
    return dict(resolved_correct=float(correct.mean()), resolved_wrong=float(wrong.mean()),
                needs_pet=float(ind.mean()), n=int(len(y))), correct, ind


def gray_zone_fraction_9595(y, score):
    """Fraction between a 95%-sensitivity and a 95%-specificity cutoff."""
    pos = np.sort(score[y == 1])
    neg = np.sort(score[y == 0])
    lo = np.percentile(pos, 5)     # 95% sensitivity: rule out below this
    hi = np.percentile(neg, 95)    # 95% specificity: rule in above this
    if hi < lo:
        return 0.0
    return float(((score >= lo) & (score <= hi)).mean())


# ------------------------------------------------------------------- main ---

def core_numbers():
    out = {}
    adni = load_adni()
    a4 = load_a4()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]

    # ---------------------------------------------------------- Table 1 ---
    def describe(df, label, dx_counts=None):
        d = dict(label=label, n=int(len(df)),
                 age_mean=float(df['AGE'].mean()), age_sd=float(df['AGE'].std()),
                 female_n=int((df['SEX_binary'] == 0).sum()),
                 female_pct=float((df['SEX_binary'] == 0).mean() * 100),
                 educ_mean=float(pd.to_numeric(df['PTEDUCAT'], errors='coerce').mean()),
                 educ_sd=float(pd.to_numeric(df['PTEDUCAT'], errors='coerce').std()),
                 apoe4_n=int(df['APOE4_carrier'].sum()),
                 apoe4_pct=float(df['APOE4_carrier'].mean() * 100),
                 amyloid_n=int(df['amyloid_positive'].sum()),
                 amyloid_pct=float(df['amyloid_positive'].mean() * 100),
                 ptau_median=float(df['pTau217_raw'].median()),
                 ptau_q1=float(df['pTau217_raw'].quantile(.25)),
                 ptau_q3=float(df['pTau217_raw'].quantile(.75)))
        if 'PTRACCAT' in df.columns:
            white = df['PTRACCAT'].astype(str).str.strip().eq('White')
            d['white_n'] = int(white.sum())
            d['white_pct'] = float(white.mean() * 100)
        if dx_counts:
            d['dx'] = dx_counts
        return d

    imp_raw = adni.loc[idx]
    out['table1_adni_impaired'] = describe(
        imp_raw, 'ADNI MCI + dementia',
        dx_counts=dict(MCI=int(imp_raw['DX'].isin(MCI_DX).sum()),
                       dementia=int(imp_raw['DX'].isin(DEM_DX).sum())))
    out['table1_adni_full'] = describe(adni, 'ADNI full')
    out['table1_a4'] = describe(a4, 'A4 + LEARN')
    out['table1_a4']['learn_n'] = int((a4['TX'].astype(str).str.contains('LEARN|Placebo|placebo', na=False)).sum()) \
        if 'TX' in a4.columns else None
    out['table1_adni_impaired']['assay_upenn'] = int((imp_raw['assay'] == 'UPENN').sum())
    out['table1_adni_impaired']['assay_janssen'] = int((imp_raw['assay'] == 'Janssen').sum())
    out['a4_panel_availability'] = dict(
        gfap=float(a4['GFAP_raw'].notna().mean() * 100),
        ab4240=float(a4['AB42_40_ratio'].notna().mean() * 100))

    # ----------------------------------------------------- ADNI pipeline ---
    print('ADNI LOOCV...', flush=True)
    pred = run_loocv(adni, idx, verbose=False)
    pred.to_csv(RESULTS / 'adni_loocv_predictions_v2.csv', index=False)
    y = pred['y'].values
    p = pred['final_prob'].values
    gkp = pred['gk_prob'].values
    res = (pred['stage'] == 'gatekeeper').values
    gz = ~res

    m = binary_metrics(y, p)
    m['auc'] = float(roc_auc_score(y, p))
    m['auc_ci'] = boot_ci(y, p, roc_auc_score)
    m['brier'] = float(brier_score_loss(y, p))
    for k, fn in [('accuracy_ci', lambda a, b: ((b >= .5).astype(int) == a).mean()),
                  ('sensitivity_ci', lambda a, b: ((b >= .5)[a == 1]).mean()),
                  ('specificity_ci', lambda a, b: (~(b >= .5)[a == 0]).mean())]:
        m[k] = boot_ci(y, p, fn)
    out['adni_pipeline'] = m

    gk_call = (gkp[res] > GK_HIGH).astype(int)
    yr = y[res]
    out['adni_stage1'] = dict(
        n_resolved=int(res.sum()), pct_resolved=float(res.mean() * 100),
        n_negative=int((gkp[res] < GK_LOW).sum()), n_positive=int((gkp[res] > GK_HIGH).sum()),
        accuracy=float((gk_call == yr).mean()),
        npv=float((yr[gkp[res] < GK_LOW] == 0).mean()),
        ppv=float((yr[gkp[res] > GK_HIGH] == 1).mean()),
        auc=float(roc_auc_score(yr, gkp[res])),
        auc_ci=boot_ci(yr, gkp[res], roc_auc_score))

    a_p, a_g, a_pval = delong_p(y[gz], gkp[gz], p[gz])
    out['adni_stage2'] = dict(
        n=int(gz.sum()), pct=float(gz.mean() * 100),
        auc_ptau=a_p, auc_ptau_ci=boot_ci(y[gz], gkp[gz], roc_auc_score),
        auc_grad=a_g, auc_grad_ci=boot_ci(y[gz], p[gz], roc_auc_score),
        delta=a_g - a_p, delong_p=a_pval)
    out['adni_stage2'].update({'metrics_at_0.5': binary_metrics(y[gz], p[gz])})

    out['adni_operating'] = dict(rule_out=operating_point(y, p, 0.90, 'sens'),
                                 rule_in=operating_point(y, p, 0.90, 'spec'))

    # subgroups
    imp = adni.loc[idx].reset_index(drop=True)
    subs = []
    for name, mask in [
        ('MCI', imp['DX'].isin(MCI_DX).values),
        ('Dementia', imp['DX'].isin(DEM_DX).values),
        ('APOE e4 carrier', (imp['APOE4_carrier'] == 1).values),
        ('APOE e4 non-carrier', (imp['APOE4_carrier'] == 0).values),
        ('Female', (imp['SEX_binary'] == 0).values),
        ('Male', (imp['SEX_binary'] == 1).values),
        ('Age <=70', (imp['AGE'] <= imp['AGE'].quantile(1 / 3)).values),
        ('Age 70-77', ((imp['AGE'] > imp['AGE'].quantile(1 / 3)) & (imp['AGE'] <= imp['AGE'].quantile(2 / 3))).values),
        ('Age >77', (imp['AGE'] > imp['AGE'].quantile(2 / 3)).values),
    ]:
        mask = np.asarray(mask, bool)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        lo, hi = boot_ci(y[mask], p[mask], roc_auc_score)
        subs.append(dict(stratum=name, n=int(mask.sum()),
                         auc=float(roc_auc_score(y[mask], p[mask])), lo=lo, hi=hi))
    out['adni_subgroups'] = subs
    out['adni_age_tertiles'] = [float(imp['AGE'].quantile(1 / 3)), float(imp['AGE'].quantile(2 / 3))]

    # -------------------------------------------------------- A4 external ---
    print('A4 external...', flush=True)
    a4_out, fit = fit_full_and_score_a4(adni, idx, a4)
    a4_out.to_csv(RESULTS / 'a4_predictions_v2.csv', index=False)
    ya = a4_out['y'].values
    pa = a4_out['final_prob'].values
    gka = a4_out['gk_prob'].values
    resa = (a4_out['stage'] == 'gatekeeper').values
    gza = ~resa

    ma = binary_metrics(ya, pa)
    ma['auc'] = float(roc_auc_score(ya, pa))
    ma['auc_ci'] = boot_ci(ya, pa, roc_auc_score)
    ma['brier'] = float(brier_score_loss(ya, pa))
    out['a4_pipeline'] = ma
    out['a4_stage1'] = dict(
        n_resolved=int(resa.sum()), pct_resolved=float(resa.mean() * 100),
        n_negative=int((gka < GK_LOW).sum()), n_positive=int((gka > GK_HIGH).sum()),
        auc=float(roc_auc_score(ya[resa], gka[resa])),
        accuracy=float(((gka[resa] > GK_HIGH).astype(int) == ya[resa]).mean()),
        npv=float((ya[gka < GK_LOW] == 0).mean()), ppv=float((ya[gka > GK_HIGH] == 1).mean()))
    b_p, b_g, b_pval = delong_p(ya[gza], gka[gza], pa[gza])
    out['a4_stage2'] = dict(
        n=int(gza.sum()), pct=float(gza.mean() * 100),
        auc_ptau=b_p, auc_ptau_ci=boot_ci(ya[gza], gka[gza], roc_auc_score),
        auc_grad=b_g, auc_grad_ci=boot_ci(ya[gza], pa[gza], roc_auc_score),
        delta=b_g - b_p, delong_p=b_pval,
        metrics_at_0_5=binary_metrics(ya[gza], pa[gza]))
    out['a4_calibration'] = calibration(ya, pa)
    cl = a4_out['centiloid'].values
    ok = ~np.isnan(cl)
    sr = stats.spearmanr(pa[ok], cl[ok])
    out['a4_centiloid'] = dict(rho=float(sr.statistic), p=float(sr.pvalue), n=int(ok.sum()))
    out['a4_operating'] = dict(rule_out=operating_point(ya, pa, 0.90, 'sens'),
                               rule_in=operating_point(ya, pa, 0.90, 'spec'))

    # prevalence-adjusted PPV/NPV from the external operating point
    sens, spec = ma['sensitivity'], ma['specificity']
    out['prevalence_table'] = [
        dict(prevalence=pv,
             ppv=sens * pv / (sens * pv + (1 - spec) * (1 - pv)),
             npv=spec * (1 - pv) / ((1 - sens) * pv + spec * (1 - pv)))
        for pv in (0.20, 0.30, 0.50, 0.70)]

    # ------------------------------------------- band routing + reclass ---
    rows = []
    for label, yy, pp, gg in [('ADNI', y[gz], p[gz], gkp[gz]), ('A4', ya[gza], pa[gza], gka[gza])]:
        o_ptau, c_ptau, i_ptau = band_outcome(yy, gg)
        o_grad, c_grad, i_grad = band_outcome(yy, pp)
        b01, b10, pv = mcnemar(c_ptau, c_grad)
        rows.append(dict(cohort=label, score='p-tau217', **o_ptau))
        rows.append(dict(cohort=label, score='GRAD Stage 2', **o_grad))
        out[f'{label.lower()}_reclassification'] = dict(
            wrong_to_right=b01, right_to_wrong=b10, p=pv, n=int(len(yy)))
    out['band_outcomes'] = rows

    # ------------------------------------------------- odds ratios (ADNI) ---
    adni_h = engineer(harmonise(adni, adni))
    imp_h = adni_h.loc[idx]
    ytr = imp_h['amyloid_positive'].values
    gk_full = fit_stage1(imp_h, ytr)
    p_tr = gk_full.predict_proba(imp_h[['pTau217_Z']].values)[:, 1]
    gz_tr = (p_tr >= GK_LOW) & (p_tr <= GK_HIGH)
    gzdf = imp_h[gz_tr]
    X = gzdf[FEATURES].astype(float)
    X = X.fillna(X.median())
    Xs = (X - X.mean()) / X.std()
    try:
        lg = sm.Logit(ytr[gz_tr], sm.add_constant(Xs)).fit(disp=0)
        out['odds_ratios'] = [
            dict(feature=f, odds_ratio=float(np.exp(lg.params[f])),
                 lo=float(np.exp(lg.conf_int().loc[f, 0])),
                 hi=float(np.exp(lg.conf_int().loc[f, 1])),
                 p=float(lg.pvalues[f]))
            for f in FEATURES]
    except Exception as e:      # pragma: no cover
        out['odds_ratios'] = f'failed: {e}'

    # coefficients of the deployed Stage 2 model
    out['stage2_model'] = dict(
        C=0.1, features=fit['features'],
        coefficients={f: float(c) for f, c in zip(fit['features'], fit['stage2'].coef_[0])},
        intercept=float(fit['stage2'].intercept_[0]),
        n_train=int(len(fit['train_gz'])),
        stage1_intercept=float(fit['gatekeeper'].intercept_[0]),
        stage1_coef=float(fit['gatekeeper'].coef_[0, 0]))

    # ------------------------------------------------------- ablation ------
    print('ablation...', flush=True)
    abl = []
    base = out['adni_stage2']['auc_grad']
    for drop in FEATURES:
        red = [f for f in FEATURES if f != drop]
        pr = run_loocv(adni, idx, features=red, verbose=False)
        g2 = (pr['stage'] == 'reflex').values
        abl.append(dict(variant=f'drop {drop}', n_gz=int(g2.sum()),
                        gz_auc=float(roc_auc_score(pr['y'].values[g2], pr['final_prob'].values[g2])),
                        delta=float(roc_auc_score(pr['y'].values[g2], pr['final_prob'].values[g2]) - base)))
    pr = run_loocv(adni, idx, features=FEATURES + ['NfL_Z'], verbose=False)
    g2 = (pr['stage'] == 'reflex').values
    nfl_auc = float(roc_auc_score(pr['y'].values[g2], pr['final_prob'].values[g2]))
    abl.append(dict(variant='add NfL_Z', n_gz=int(g2.sum()), gz_auc=nfl_auc, delta=nfl_auc - base))
    out['ablation'] = dict(baseline_gz_auc=base, results=abl)

    # ------------------------------------- gray zone size by population ----
    frac = {}
    for label, sub in [('ADNI CN', adni_h[adni_h['DX'].isin(CN_DX)]),
                       ('ADNI MCI', adni_h[adni_h['DX'].isin(MCI_DX)]),
                       ('ADNI dementia', adni_h[adni_h['DX'].isin(DEM_DX)])]:
        frac[label] = dict(n=int(len(sub)),
                           pct=100 * gray_zone_fraction_9595(
                               sub['amyloid_positive'].values, sub['pTau217_Z'].values),
                           ptau_auc=float(roc_auc_score(sub['amyloid_positive'], sub['pTau217_Z'])))
    a4_h = fit['a4_h']
    frac['A4 + LEARN'] = dict(n=int(len(a4_h)),
                              pct=100 * gray_zone_fraction_9595(
                                  a4_h['amyloid_positive'].values, a4_h['pTau217_Z'].values),
                              ptau_auc=float(roc_auc_score(a4_h['amyloid_positive'], a4_h['pTau217_Z'])))
    out['gray_zone_by_population'] = frac

    # --------------------------------------------- p-tau181 comparison -----
    print('p-tau181...', flush=True)
    import os
    from _grad_paths import A4_DIR
    roche = pd.read_csv(os.path.join(str(A4_DIR), 'Clinical', 'External Data',
                                     'biomarker_Plasma_Roche_Results.csv'))
    roche['LABRESN'] = pd.to_numeric(roche['LABRESN'], errors='coerce')
    roche['LBTESTCD'] = roche['LBTESTCD'].str.strip()
    p181 = roche[roche['LBTESTCD'] == 'TPP181'].sort_values(['BID', 'VISCODE'])
    p181 = p181.groupby('BID').first().reset_index()[['BID', 'LABRESN']]
    p181 = p181.rename(columns={'LABRESN': 'ptau181'})
    a4_join = a4[['BID']].copy()
    a4_join['_row'] = np.arange(len(a4_join))
    a4_join = a4_join.merge(p181, on='BID', how='left')
    has181 = a4_join['ptau181'].notna().values & gza
    if has181.sum() > 20:
        y181 = ya[has181]
        v181 = a4_join['ptau181'].values[has181]
        gr = pa[has181]
        pt = gka[has181]
        # Youden-optimal p-tau181 cutoff derived in A4 itself (favours comparator)
        fpr, tpr, thr = roc_curve(y181, v181)
        cut = thr[np.argmax(tpr - fpr)]
        call181 = (v181 >= cut).astype(int)
        o_grad, c_grad, i_grad = band_outcome(y181, gr)
        o_ptau, c_ptau, i_ptau = band_outcome(y181, pt)
        auc_g, auc_1, p_g1 = delong_p(y181, gr, v181)
        _, _, p_t1 = delong_p(y181, pt, v181)
        _, _, p_gt = delong_p(y181, gr, pt)
        out['ptau181_comparison'] = dict(
            n=int(has181.sum()), prevalence=float(y181.mean() * 100),
            auc_ptau217=float(roc_auc_score(y181, pt)),
            auc_ptau217_ci=boot_ci(y181, pt, roc_auc_score),
            auc_ptau181=float(roc_auc_score(y181, v181)),
            auc_ptau181_ci=boot_ci(y181, v181, roc_auc_score),
            auc_grad=float(roc_auc_score(y181, gr)),
            auc_grad_ci=boot_ci(y181, gr, roc_auc_score),
            delong_grad_vs_181=p_g1, delong_grad_vs_217=p_gt, delong_217_vs_181=p_t1,
            ptau181_correct=float((call181 == y181).mean() * 100),
            ptau181_wrong=float((call181 != y181).mean() * 100),
            grad_resolved_correct=o_grad['resolved_correct'] * 100,
            grad_resolved_wrong=o_grad['resolved_wrong'] * 100,
            grad_to_pet=o_grad['needs_pet'] * 100,
            ptau217_resolved_correct=o_ptau['resolved_correct'] * 100,
            ptau217_resolved_wrong=o_ptau['resolved_wrong'] * 100,
            ptau217_to_pet=o_ptau['needs_pet'] * 100)
        # correlations inside the gray zone
        sub = a4_h.iloc[np.where(gza)[0]]
        out['gz_correlations'] = {
            'ab4240': float(stats.spearmanr(sub['pTau217_Z'], sub['AB42_40_ratio'],
                                            nan_policy='omit').statistic),
            'gfap': float(stats.spearmanr(sub['pTau217_Z'], sub['GFAP_Z'],
                                          nan_policy='omit').statistic),
            'apoe4': float(stats.spearmanr(sub['pTau217_Z'], sub['APOE4_carrier'],
                                           nan_policy='omit').statistic)}

    # ------------------------------------------------------------- cost ----
    PLASMA = 350.0        # single-analyte p-tau217
    ADDON = 250.0         # multi-analyte add-on, billed only at Stage 2
    PET = 3000.0
    N = 10000
    cost = []
    for label, gz_rate, pet_rate in [
            ('ADNI (MCI + dementia)', gz.mean(), out['band_outcomes'][1]['needs_pet'] * gz.mean()),
            ('A4 + LEARN', gza.mean(), out['band_outcomes'][3]['needs_pet'] * gza.mean())]:
        universal = N * PET
        ptau_first = N * PLASMA + N * gz_rate * PET
        grad = N * PLASMA + N * gz_rate * ADDON + N * pet_rate * PET
        correct_grad = None
        cost.append(dict(cohort=label, gray_zone_rate=float(gz_rate), residual_pet_rate=float(pet_rate),
                         universal=universal, ptau_first=ptau_first, grad=grad,
                         scans_universal=N, scans_ptau_first=int(round(N * gz_rate)),
                         scans_grad=int(round(N * pet_rate)),
                         saving_ptau_first=100 * (1 - ptau_first / universal),
                         saving_grad=100 * (1 - grad / universal),
                         per_patient_grad=grad / N))
    # cost per correct diagnosis (GRAD vs p-tau217-first), external cohort
    o_grad = out['band_outcomes'][3]
    o_ptau = out['band_outcomes'][2]
    gz_rate = gza.mean()
    resolved_correct_grad = (1 - gz_rate) * out['a4_stage1']['accuracy'] + gz_rate * o_grad['resolved_correct']
    resolved_correct_ptau = (1 - gz_rate) * out['a4_stage1']['accuracy'] + gz_rate * o_ptau['resolved_correct']
    pet_grad = gz_rate * o_grad['needs_pet']
    pet_ptau = gz_rate * o_ptau['needs_pet']
    cost.append(dict(cohort='A4 cost per correct diagnosis',
                     grad=(N * PLASMA + N * gz_rate * ADDON + N * pet_grad * PET) / (N * (resolved_correct_grad + pet_grad)),
                     ptau_first=(N * PLASMA + N * pet_ptau * PET) / (N * (resolved_correct_ptau + pet_ptau)),
                     universal=PET))
    out['cost'] = dict(unit_costs=dict(plasma=PLASMA, addon=ADDON, pet=PET, n=N), rows=cost)

    return out




def extra_numbers(prev):
    out = {}
    adni = load_adni()
    a4 = load_a4()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]

    # ------------------------------------------------------------------ Table 1 --
    a4_sub = pd.read_csv(os.path.join(str(A4_DIR), 'Clinical', 'Derived Data', 'SUBJINFO.csv'),
                         low_memory=False)[['BID', 'RACE']]
    a4 = a4.merge(a4_sub, on='BID', how='left')

    imp = adni.loc[idx]
    out['table1'] = {
        'ADNI_impaired': dict(
            n=len(imp), mci=int(imp['DX'].isin(MCI_DX).sum()), dementia=int(imp['DX'].isin(DEM_DX).sum()),
            age_mean=imp['AGE'].mean(), age_sd=imp['AGE'].std(),
            female_n=int((imp['PTGENDER'] == 'Female').sum()),
            female_pct=100 * (imp['PTGENDER'] == 'Female').mean(),
            educ_mean=pd.to_numeric(imp['PTEDUCAT'], errors='coerce').mean(),
            educ_sd=pd.to_numeric(imp['PTEDUCAT'], errors='coerce').std(),
            white_n=int((imp['PTRACCAT'] == 'White').sum()),
            white_pct=100 * (imp['PTRACCAT'] == 'White').mean(),
            apoe4_n=int(imp['APOE4_carrier'].sum()),
            apoe4_avail=int(imp['APOE4_carrier'].notna().sum()),
            apoe4_pct=100 * imp['APOE4_carrier'].mean(),
            amyloid_n=int(imp['amyloid_positive'].sum()),
            amyloid_pct=100 * imp['amyloid_positive'].mean(),
            ptau_median=imp['pTau217_raw'].median(),
            ptau_q1=imp['pTau217_raw'].quantile(.25), ptau_q3=imp['pTau217_raw'].quantile(.75),
            upenn=int((imp['assay'] == 'UPENN').sum()), janssen=int((imp['assay'] == 'Janssen').sum()),
            mmse_mean=pd.to_numeric(imp['MMSE'], errors='coerce').mean(),
            mmse_sd=pd.to_numeric(imp['MMSE'], errors='coerce').std()),
        'A4': dict(
            n=len(a4),
            age_mean=a4['AGE'].mean(), age_sd=a4['AGE'].std(),
            female_n=int((a4['SEX'] == 1).sum()), female_pct=100 * (a4['SEX'] == 1).mean(),
            educ_mean=pd.to_numeric(a4['PTEDUCAT'], errors='coerce').mean(),
            educ_sd=pd.to_numeric(a4['PTEDUCAT'], errors='coerce').std(),
            white_n=int((a4['RACE'] == 1).sum()), white_pct=100 * (a4['RACE'] == 1).mean(),
            apoe4_n=int(a4['APOE4_carrier'].sum()),
            apoe4_avail=int(a4['APOE4_carrier'].notna().sum()),
            apoe4_pct=100 * a4['APOE4_carrier'].mean(),
            amyloid_n=int(a4['amyloid_positive'].sum()),
            amyloid_pct=100 * a4['amyloid_positive'].mean(),
            ptau_median=a4['pTau217_raw'].median(),
            ptau_q1=a4['pTau217_raw'].quantile(.25), ptau_q3=a4['pTau217_raw'].quantile(.75),
            learn_n=int(a4['TX'].isna().sum()), a4arm_n=int(a4['TX'].notna().sum()),
            gfap_avail=100 * a4['GFAP_raw'].notna().mean(),
            ab4240_avail=100 * a4['AB42_40_ratio'].notna().mean(),
            mmse_mean=pd.to_numeric(a4['MMSE'], errors='coerce').mean(),
            mmse_sd=pd.to_numeric(a4['MMSE'], errors='coerce').std()),
    }
    # panel availability by arm
    out['a4_panel_by_arm'] = dict(
        learn_pct=100 * a4.loc[a4['TX'].isna(), 'AB42_40_ratio'].notna().mean(),
        a4_pct=100 * a4.loc[a4['TX'].notna(), 'AB42_40_ratio'].notna().mean())
    # LEARN vs arm amyloid split
    out['a4_arm_amyloid'] = pd.crosstab(a4['TX'].isna(), a4['amyloid_positive']).to_dict()

    # ------------------------------------------- 95/95 rule, cutoffs pooled per cohort --
    adni_h = engineer(harmonise(adni, adni))
    a4_h = engineer(harmonise(a4, a4, cn_only=False))


    def band_fraction(score, lo, hi):
        return 100 * float(((score >= lo) & (score <= hi)).mean())


    def cutoffs_9595(y, score):
        return np.percentile(score[y == 1], 5), np.percentile(score[y == 0], 95)


    lo_a, hi_a = cutoffs_9595(adni_h['amyloid_positive'].values, adni_h['pTau217_Z'].values)
    lo_4, hi_4 = cutoffs_9595(a4_h['amyloid_positive'].values, a4_h['pTau217_Z'].values)
    rows = []
    for label, sub in [('ADNI CN', adni_h[adni_h['DX'].isin(CN_DX)]),
                       ('ADNI MCI', adni_h[adni_h['DX'].isin(MCI_DX)]),
                       ('ADNI dementia', adni_h[adni_h['DX'].isin(DEM_DX)])]:
        rows.append(dict(population=label, n=len(sub),
                         pct_common=band_fraction(sub['pTau217_Z'].values, lo_a, hi_a),
                         ptau_auc=float(roc_auc_score(sub['amyloid_positive'], sub['pTau217_Z']))))
    rows.append(dict(population='A4 + LEARN', n=len(a4_h),
                     pct_common=band_fraction(a4_h['pTau217_Z'].values, lo_4, hi_4),
                     ptau_auc=float(roc_auc_score(a4_h['amyloid_positive'], a4_h['pTau217_Z']))))
    out['gray_zone_common_rule'] = rows

    # simulation: gray-zone width as a function of marker AUC (two normals)
    sim = []
    rng = np.random.RandomState(0)
    for auc_target in (0.80, 0.85, 0.90, 0.95, 0.96):
        # two unit-variance normals separated by delta give AUC = Phi(delta/sqrt(2))
        delta = np.sqrt(2) * stats.norm.ppf(auc_target)
        neg = rng.normal(0, 1, 200000)
        pos = rng.normal(delta, 1, 200000)
        allv = np.concatenate([neg, pos])
        lo = np.percentile(pos, 5)
        hi = np.percentile(neg, 95)
        sim.append(dict(auc=auc_target, gray_zone_pct=100 * float(((allv >= lo) & (allv <= hi)).mean())))
    out['gray_zone_simulation'] = sim

    # ---------------------------------------------------- A4-side ablation --------
    imp_h = adni_h.loc[idx]
    y_imp = imp_h['amyloid_positive'].values
    gk = fit_stage1(imp_h, y_imp)
    p_tr = gk.predict_proba(imp_h[['pTau217_Z']].values)[:, 1]
    gz_tr = (p_tr >= GK_LOW) & (p_tr <= GK_HIGH)
    p_a4 = gk.predict_proba(a4_h[['pTau217_Z']].values)[:, 1]
    gz_a4 = (p_a4 >= GK_LOW) & (p_a4 <= GK_HIGH)
    ya = a4_h['amyloid_positive'].values


    def a4_gz_auc(features):
        m2, sc, med, used = fit_stage2(imp_h[gz_tr], y_imp[gz_tr], features=features)
        p2 = score_stage2(m2, sc, med, used, a4_h[gz_a4])
        return float(roc_auc_score(ya[gz_a4], p2))


    base_a4 = a4_gz_auc(FEATURES)
    abl = [dict(variant='full 7-feature model', auc=base_a4, delta=0.0)]
    for f in FEATURES:
        red = [x for x in FEATURES if x != f]
        v = a4_gz_auc(red)
        abl.append(dict(variant=f'drop {f}', auc=v, delta=v - base_a4))
    v = a4_gz_auc(FEATURES + ['NfL_Z'])
    abl.append(dict(variant='add NfL_Z', auc=v, delta=v - base_a4))
    out['a4_ablation'] = abl

    # ------------------------------------------------ univariate odds ratios ------
    gzdf = imp_h[gz_tr]
    uni = []
    for f in FEATURES:
        x = gzdf[f].astype(float)
        x = x.fillna(x.median())
        xs = (x - x.mean()) / x.std()
        try:
            fit = sm.Logit(y_imp[gz_tr], sm.add_constant(xs)).fit(disp=0)
            ci = fit.conf_int()
            uni.append(dict(feature=f, odds_ratio=float(np.exp(fit.params.iloc[1])),
                            lo=float(np.exp(ci.iloc[1, 0])), hi=float(np.exp(ci.iloc[1, 1])),
                            p=float(fit.pvalues.iloc[1])))
        except Exception as e:
            uni.append(dict(feature=f, error=str(e)))
    out['univariate_or'] = uni

    # standalone AUC of Ab42/40, raw vs reference-anchored (spine claim)
    gz_ab = gzdf['AB42_40_ratio'].notna().values
    if gz_ab.sum() > 10:
        raw = gzdf.loc[gz_ab, 'AB42_40_ratio'].values
        yy = y_imp[gz_tr][gz_ab]
        out['ab4240_standalone'] = dict(n=int(gz_ab.sum()), auc_raw=float(roc_auc_score(yy, -raw)))

    # ------------------------------------------------------------------ cost ------
    PLASMA, ADDON, PET, N = 350.0, 250.0, 3000.0, 10000


    def strategy_costs(gz_rate, stage1_acc, band_pet_frac, band_correct_frac):
        """Returns dict of cost, scans, correct-diagnosis fraction for three strategies."""
        universal = dict(cost=N * PET, scans=N, correct=1.0)
        ptau_first = dict(cost=N * PLASMA + N * gz_rate * PET, scans=N * gz_rate,
                          correct=(1 - gz_rate) * stage1_acc + gz_rate * 1.0)
        grad_pet = gz_rate * band_pet_frac
        grad = dict(cost=N * PLASMA + N * gz_rate * ADDON + N * grad_pet * PET,
                    scans=N * grad_pet,
                    correct=(1 - gz_rate) * stage1_acc + gz_rate * (band_correct_frac + band_pet_frac))
        for d in (universal, ptau_first, grad):
            d['per_patient'] = d['cost'] / N
            d['saving_pct'] = 100 * (1 - d['cost'] / universal['cost'])
            d['cost_per_correct'] = d['cost'] / (N * d['correct'])
            d['scans'] = int(round(d['scans']))
        return dict(universal=universal, ptau_first=ptau_first, grad=grad)



    out['cost'] = dict(
        unit_costs=dict(plasma=PLASMA, addon=ADDON, pet=PET, n=N),
        adni=strategy_costs(prev['adni_stage2']['pct'] / 100, prev['adni_stage1']['accuracy'],
                            prev['band_outcomes'][1]['needs_pet'], prev['band_outcomes'][1]['resolved_correct']),
        a4=strategy_costs(prev['a4_stage2']['pct'] / 100, prev['a4_stage1']['accuracy'],
                          prev['band_outcomes'][3]['needs_pet'], prev['band_outcomes'][3]['resolved_correct']))

    # overall definitive-classification rate (no PET)
    out['definitive_without_pet'] = dict(
        adni=100 * (1 - prev['adni_stage2']['pct'] / 100 * prev['band_outcomes'][1]['needs_pet']),
        a4=100 * (1 - prev['a4_stage2']['pct'] / 100 * prev['band_outcomes'][3]['needs_pet']))

    return out


def main():
    core_out = core_numbers()
    core_out.update(extra_numbers(core_out))
    path = RESULTS / 'grad_v2_numbers.json'
    with open(path, 'w') as fh:
        json.dump(core_out, fh, indent=2, default=float)
    print(f'\nwrote {path}')
    print(f"ADNI  full pipeline AUC {core_out['adni_pipeline']['auc']:.3f}  "
          f"gray zone {core_out['adni_stage2']['auc_ptau']:.3f} -> {core_out['adni_stage2']['auc_grad']:.3f}")
    print(f"A4    full pipeline AUC {core_out['a4_pipeline']['auc']:.3f}  "
          f"gray zone {core_out['a4_stage2']['auc_ptau']:.3f} -> {core_out['a4_stage2']['auc_grad']:.3f} "
          f"(DeLong P = {core_out['a4_stage2']['delong_p']:.1e})")


if __name__ == '__main__':
    main()
