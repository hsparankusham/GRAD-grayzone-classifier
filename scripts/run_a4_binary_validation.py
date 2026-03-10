"""
A4 Binary External Validation (Corrected)
==========================================
Uses LEARN cohort (amyloid-negative screen failures) + treatment arm
(amyloid-positive) for true binary classification external validation.

Previous bug: All A4 participants were hardcoded as amyloid_positive = 1
Fix: Uses centiloid >= 20 for amyloid-positive classification

This enables proper AUC, sensitivity, specificity, and 90/90 analysis
on a fully independent external cohort.
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    brier_score_loss, confusion_matrix
)
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))),
    'syntropi-ai-data'
)
ADNI_DIR = os.path.join(DATA_DIR, 'syntropi-ai-ADNI')
A4_DIR = os.path.join(DATA_DIR, 'syntropi-ai-A4')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Thresholds
AV45_THRESHOLD = 1.11
CL_THRESHOLD = 20
GK_LOW = 0.25
GK_HIGH = 0.75


# ============================================================
# DATA LOADING
# ============================================================

def load_adni():
    """Load ADNI data: UPENN + Janssen biomarkers merged with ADNIMERGE."""
    # UPENN plasma biomarkers
    upenn = pd.read_csv(os.path.join(ADNI_DIR, 'Pathology', 'UPENN_PlasmaBiomarkers.csv'))
    upenn = upenn.rename(columns={
        'pT217_F': 'pTau217_raw', 'AB42_F': 'AB42_raw', 'AB40_F': 'AB40_raw',
        'AB42_AB40_F': 'AB42_40_ratio', 'NfL_Q': 'NfL_raw', 'GFAP_Q': 'GFAP_raw'
    })
    numeric_cols = ['pTau217_raw', 'AB42_raw', 'AB40_raw', 'AB42_40_ratio', 'NfL_raw', 'GFAP_raw']
    for col in numeric_cols:
        upenn[col] = pd.to_numeric(upenn[col], errors='coerce')
        upenn.loc[upenn[col] < 0, col] = np.nan
    upenn['assay'] = 'UPENN'
    upenn['PTID'] = upenn['PTID'].astype(str)

    # Janssen plasma p-tau217
    janssen_path = os.path.join(ADNI_DIR, 'Pathology', 'JANSSEN_PLASMA_P217_TAU_18Dec2025.csv')
    if os.path.exists(janssen_path):
        janssen = pd.read_csv(janssen_path)
        janssen = janssen.rename(columns={'DILUTION_CORRECTED_CONC': 'pTau217_raw'})
        janssen['pTau217_raw'] = pd.to_numeric(janssen['pTau217_raw'], errors='coerce')
        janssen['assay'] = 'Janssen'
        janssen['PTID'] = janssen['PTID'].astype(str)
    else:
        janssen = None

    # ADNIMERGE
    merge = pd.read_csv(os.path.join(ADNI_DIR, 'ADNIMERGE2025.csv'))
    merge['AV45'] = pd.to_numeric(merge['AV45'], errors='coerce')
    merge['AV45_bl'] = pd.to_numeric(merge.get('AV45_bl', pd.Series(dtype=float)), errors='coerce')
    merge['amyloid_positive'] = (merge['AV45'] > AV45_THRESHOLD).astype(float)
    merge.loc[merge['AV45'].isna(), 'amyloid_positive'] = np.nan
    merge.loc[merge['AV45'].isna() & merge['AV45_bl'].notna(), 'amyloid_positive'] = (
        merge.loc[merge['AV45'].isna() & merge['AV45_bl'].notna(), 'AV45_bl'] > AV45_THRESHOLD
    ).astype(float)
    merge['PTID'] = merge['PTID'].astype(str)

    # Merge UPENN with ADNIMERGE
    upenn_merged = upenn.merge(merge, on=['PTID', 'RID'], how='inner', suffixes=('', '_merge'))
    upenn_merged = upenn_merged[
        (upenn_merged['VISCODE2'] == upenn_merged['VISCODE']) |
        (upenn_merged['VISCODE2'].str.contains('bl', case=False, na=False) &
         upenn_merged['VISCODE'].str.contains('bl', case=False, na=False))
    ].copy()

    combined = upenn_merged.copy()

    # Merge Janssen with ADNIMERGE and add non-overlapping participants
    if janssen is not None:
        janssen_merged = janssen.merge(merge, on=['PTID', 'RID'], how='inner')
        janssen_merged = janssen_merged[
            (janssen_merged['VISCODE2'] == janssen_merged['VISCODE']) |
            (janssen_merged['VISCODE2'].str.contains('bl', case=False, na=False) &
             janssen_merged['VISCODE'].str.contains('bl', case=False, na=False))
        ].copy()
        # Add Janssen participants not already in UPENN
        janssen_only = janssen_merged[~janssen_merged['PTID'].isin(upenn_merged['PTID'])]
        if len(janssen_only) > 0:
            combined = pd.concat([combined, janssen_only], ignore_index=True)

    # Baseline only
    combined = combined.sort_values(['PTID', 'VISCODE2'])
    combined = combined.groupby('PTID').first().reset_index()

    # Filter: need both pTau217 and amyloid PET
    combined = combined[
        combined['amyloid_positive'].notna() & combined['pTau217_raw'].notna()
    ].copy()

    # APOE4
    combined['APOE4'] = pd.to_numeric(combined.get('APOE4', pd.Series(dtype=float)), errors='coerce')
    combined['APOE4_carrier'] = (combined['APOE4'] > 0).astype(float)
    combined.loc[combined['APOE4'].isna(), 'APOE4_carrier'] = np.nan

    n_upenn = (combined['assay'] == 'UPENN').sum()
    n_janssen = (combined['assay'] == 'Janssen').sum()
    print(f"ADNI: {len(combined)} participants (UPENN: {n_upenn}, Janssen: {n_janssen})")
    print(f"  Amyloid+: {int(combined['amyloid_positive'].sum())} ({combined['amyloid_positive'].mean()*100:.1f}%)")
    return combined


def load_a4():
    """Load A4 data with CORRECTED amyloid labels using centiloid."""
    # pTau217
    ptau = pd.read_csv(os.path.join(A4_DIR, 'Clinical', 'External Data', 'biomarker_pTau217.csv'))
    ptau['pTau217_raw'] = pd.to_numeric(ptau['ORRES'], errors='coerce')
    ptau.loc[ptau['pTau217_raw'].isna(), 'pTau217_raw'] = pd.to_numeric(
        ptau.loc[ptau['pTau217_raw'].isna(), 'ORRESRAW'], errors='coerce'
    )
    ptau['assay'] = 'Lilly'
    ptau_bl = ptau.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()

    # Roche biomarkers
    roche = pd.read_csv(os.path.join(A4_DIR, 'Clinical', 'External Data', 'biomarker_Plasma_Roche_Results.csv'))
    roche['LABRESN'] = pd.to_numeric(roche['LABRESN'], errors='coerce')
    roche['LBTESTCD'] = roche['LBTESTCD'].str.strip()
    pivot = roche.pivot_table(index=['BID', 'VISCODE'], columns='LBTESTCD', values='LABRESN', aggfunc='first').reset_index()
    rename_map = {'GFAP': 'GFAP_raw', 'NF-L': 'NfL_raw', 'AMYLB42': 'AB42_raw', 'AMYLB40': 'AB40_raw'}
    pivot = pivot.rename(columns=rename_map)
    if 'AB42_raw' in pivot.columns and 'AB40_raw' in pivot.columns:
        pivot['AB42_40_ratio'] = pivot['AB42_raw'] / (pivot['AB40_raw'] * 1000)
    if 'GFAP_raw' in pivot.columns:
        pivot['GFAP_raw'] = pivot['GFAP_raw'] * 1000
    roche_bl = pivot.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()

    # Subject info (with CORRECTED amyloid labels)
    subj = pd.read_csv(os.path.join(A4_DIR, 'Clinical', 'Derived Data', 'SUBJINFO.csv'))
    cols = ['BID', 'TX', 'AGEYR', 'SEX', 'EDCCNTU', 'APOEGNPRSNFLG', 'SUVRCER', 'AMYLCENT', 'MMSETSV1']
    subj = subj[[c for c in cols if c in subj.columns]]
    subj = subj.rename(columns={
        'AGEYR': 'AGE', 'EDCCNTU': 'PTEDUCAT', 'APOEGNPRSNFLG': 'APOE4_carrier', 'MMSETSV1': 'MMSE'
    })

    # CRITICAL FIX: Use centiloid for amyloid classification (not hardcoded)
    subj['AMYLCENT'] = pd.to_numeric(subj['AMYLCENT'], errors='coerce')
    subj['amyloid_positive'] = (subj['AMYLCENT'] >= CL_THRESHOLD).astype(float)
    subj.loc[subj['AMYLCENT'].isna(), 'amyloid_positive'] = np.nan

    # Merge
    combined = subj.merge(ptau_bl[['BID', 'pTau217_raw', 'assay']], on='BID', how='inner')
    roche_cols = ['BID'] + [c for c in ['GFAP_raw', 'NfL_raw', 'AB42_raw', 'AB40_raw', 'AB42_40_ratio'] if c in roche_bl.columns]
    combined = combined.merge(roche_bl[roche_cols], on='BID', how='left')
    combined = combined[combined['pTau217_raw'].notna() & combined['amyloid_positive'].notna()].copy()
    combined['amyloid_positive'] = combined['amyloid_positive'].astype(int)

    n_pos = int(combined['amyloid_positive'].sum())
    n_neg = len(combined) - n_pos
    print(f"A4: {len(combined)} participants with pTau217 + amyloid label")
    print(f"  Amyloid+ (CL>=20): {n_pos} ({n_pos/len(combined)*100:.1f}%)")
    print(f"  Amyloid- (CL<20):  {n_neg} ({n_neg/len(combined)*100:.1f}%)")
    print(f"  Centiloid range: {combined['AMYLCENT'].min():.1f} to {combined['AMYLCENT'].max():.1f}")
    return combined


# ============================================================
# HARMONIZATION
# ============================================================

def harmonize(train_df, test_df, biomarkers=None):
    """
    Reference-based Z-score harmonization.
    Fit on CN amyloid-negative from training, apply to both.
    """
    if biomarkers is None:
        biomarkers = ['pTau217_raw', 'NfL_raw', 'GFAP_raw']

    train_out = train_df.copy()
    test_out = test_df.copy()

    for bm in biomarkers:
        if bm not in train_df.columns:
            continue
        z_col = bm.replace('_raw', '_Z')

        # Reference: CN amyloid-negative from training
        ref_mask = pd.Series(True, index=train_df.index)
        if 'DX' in train_df.columns:
            ref_mask &= train_df['DX'].isin(['CN', 'NL', 'Normal']) | train_df.get('DX_bl', pd.Series(False, index=train_df.index)).isin(['CN', 'NL', 'Normal'])
        if 'amyloid_positive' in train_df.columns:
            ref_mask &= train_df['amyloid_positive'] == 0

        ref_values = np.log1p(train_df.loc[ref_mask, bm].dropna())
        if len(ref_values) < 5:
            ref_values = np.log1p(train_df[bm].dropna())

        ref_mean = ref_values.mean()
        ref_std = ref_values.std() if ref_values.std() > 0 else 1.0

        # Apply per assay for training
        for assay in train_df['assay'].unique():
            mask = train_df['assay'] == assay
            assay_ref = ref_mask & mask
            assay_ref_vals = np.log1p(train_df.loc[assay_ref, bm].dropna())
            if len(assay_ref_vals) >= 5:
                a_mean, a_std = assay_ref_vals.mean(), assay_ref_vals.std()
                if a_std <= 0: a_std = 1.0
            else:
                a_mean, a_std = ref_mean, ref_std

            train_out.loc[mask, z_col] = (np.log1p(train_df.loc[mask, bm]) - a_mean) / a_std

        # Apply to test (use overall reference since assay differs)
        for assay in test_df['assay'].unique():
            mask = test_df['assay'] == assay
            test_out.loc[mask, z_col] = (np.log1p(test_df.loc[mask, bm]) - ref_mean) / ref_std

    return train_out, test_out


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """Create biologically motivated features."""
    result = df.copy()

    if 'pTau217_raw' in df.columns and 'AB42_40_ratio' in df.columns:
        result['tau_ab42_diff'] = np.log1p(df['pTau217_raw']) - np.log1p(df['AB42_40_ratio'])

    if 'GFAP_Z' in df.columns and 'pTau217_Z' in df.columns:
        result['gfap_tau_interaction'] = df['GFAP_Z'] * df['pTau217_Z']

    if 'AGE' in df.columns:
        result['AGE_Z'] = (df['AGE'] - df['AGE'].mean()) / (df['AGE'].std() if df['AGE'].std() > 0 else 1.0)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("A4 BINARY EXTERNAL VALIDATION (CORRECTED)")
    print("Using LEARN cohort for amyloid-negative ground truth")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # STEP 1: Load data
    print("STEP 1: Loading data...")
    adni_df = load_adni()
    print()
    a4_df = load_a4()
    print()

    adni_y = adni_df['amyloid_positive'].astype(int)
    a4_y = a4_df['amyloid_positive'].astype(int)

    # STEP 2: Harmonize
    print("STEP 2: Harmonizing biomarkers...")
    adni_h, a4_h = harmonize(adni_df, a4_df)
    print(f"  ADNI pTau217_Z: mean={adni_h['pTau217_Z'].mean():.3f}, std={adni_h['pTau217_Z'].std():.3f}")
    print(f"  A4 pTau217_Z: mean={a4_h['pTau217_Z'].mean():.3f}, std={a4_h['pTau217_Z'].std():.3f}\n")

    # STEP 3: Fit Gatekeeper on ADNI
    print("STEP 3: Fitting Gatekeeper on ADNI...")
    gatekeeper = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    X_gk = adni_h[['pTau217_Z']].values
    valid_gk = ~np.isnan(X_gk.flatten()) & ~np.isnan(adni_y.values)
    gatekeeper.fit(X_gk[valid_gk], adni_y.values[valid_gk])
    print(f"  Intercept: {gatekeeper.intercept_[0]:.4f}")
    print(f"  Coefficient: {gatekeeper.coef_[0,0]:.4f}")

    # Classify ADNI training data
    adni_gk_probs = gatekeeper.predict_proba(X_gk[valid_gk])[:, 1]
    adni_gray_mask = (adni_gk_probs >= GK_LOW) & (adni_gk_probs <= GK_HIGH)
    n_gray = adni_gray_mask.sum()
    print(f"  ADNI gray zone: {n_gray} ({n_gray/valid_gk.sum()*100:.1f}%)\n")

    # STEP 4: Fit Reflex on ADNI gray zone
    print("STEP 4: Fitting Reflex on ADNI gray zone...")
    adni_valid = adni_h.iloc[np.where(valid_gk)[0]].copy()
    adni_valid_y = adni_y.iloc[np.where(valid_gk)[0]].copy()

    gray_df = adni_valid[adni_gray_mask].copy()
    gray_y = adni_valid_y[adni_gray_mask].copy()

    # Engineer features
    gray_eng = engineer_features(gray_df)

    # Select features (same as original model, minus NfL features)
    feature_priority = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE_Z', 'APOE4_carrier', 'gfap_tau_interaction']
    features_used = [f for f in feature_priority if f in gray_eng.columns and gray_eng[f].notna().sum() > len(gray_eng) * 0.5]

    print(f"  Features: {features_used}")
    X_reflex = gray_eng[features_used].values

    # Median imputation
    for i in range(X_reflex.shape[1]):
        col = X_reflex[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X_reflex[nan_mask, i] = np.nanmedian(col)

    # Scale
    scaler = StandardScaler()
    X_reflex_scaled = scaler.fit_transform(X_reflex)

    # Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_reflex_scaled, gray_y.values)
    print(f"  Trained on {len(gray_y)} gray zone samples")
    print(f"  Feature importance: {dict(zip(features_used, [f'{x:.3f}' for x in rf.feature_importances_]))}\n")

    # STEP 5: Apply pipeline to A4
    print("STEP 5: Applying pipeline to A4...")

    # Gatekeeper predictions on A4
    X_a4_gk = a4_h[['pTau217_Z']].values
    valid_a4 = ~np.isnan(X_a4_gk.flatten())
    a4_probs = np.full(len(a4_df), np.nan)
    a4_stages = np.full(len(a4_df), '', dtype=object)

    a4_gk_probs = gatekeeper.predict_proba(X_a4_gk[valid_a4])[:, 1]

    # Classify by gatekeeper
    neg_mask_gk = a4_gk_probs < GK_LOW
    pos_mask_gk = a4_gk_probs > GK_HIGH
    gray_mask_gk = ~neg_mask_gk & ~pos_mask_gk

    valid_indices = np.where(valid_a4)[0]
    a4_probs[valid_indices[neg_mask_gk]] = a4_gk_probs[neg_mask_gk]
    a4_stages[valid_indices[neg_mask_gk]] = 'gatekeeper_neg'

    a4_probs[valid_indices[pos_mask_gk]] = a4_gk_probs[pos_mask_gk]
    a4_stages[valid_indices[pos_mask_gk]] = 'gatekeeper_pos'

    print(f"  Gatekeeper: {neg_mask_gk.sum()} negative, {pos_mask_gk.sum()} positive, {gray_mask_gk.sum()} gray zone")

    # Reflex for gray zone
    if gray_mask_gk.sum() > 0:
        a4_gray = a4_h.iloc[valid_indices[gray_mask_gk]].copy()
        a4_gray_eng = engineer_features(a4_gray)

        X_a4_reflex = a4_gray_eng[features_used].values

        # Median imputation
        for i in range(X_a4_reflex.shape[1]):
            col = X_a4_reflex[:, i]
            nan_mask_col = np.isnan(col)
            if nan_mask_col.any():
                X_a4_reflex[nan_mask_col, i] = np.nanmedian(col) if not np.all(nan_mask_col) else 0

        X_a4_reflex_scaled = scaler.transform(X_a4_reflex)
        reflex_probs = rf.predict_proba(X_a4_reflex_scaled)[:, 1]

        a4_probs[valid_indices[gray_mask_gk]] = reflex_probs
        a4_stages[valid_indices[gray_mask_gk]] = 'reflex'
        print(f"  Reflex applied to {gray_mask_gk.sum()} gray zone samples\n")

    # STEP 6: Compute metrics
    print("=" * 60)
    print("A4 EXTERNAL VALIDATION RESULTS")
    print("=" * 60)

    y_true = a4_y.values
    valid_final = ~np.isnan(a4_probs) & ~np.isnan(y_true.astype(float))
    y_pred = (a4_probs[valid_final] >= 0.5).astype(int)
    y_t = y_true[valid_final]
    p = a4_probs[valid_final]

    # AUC
    auc = roc_auc_score(y_t, p)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    accuracy = accuracy_score(y_t, y_pred)
    brier = brier_score_loss(y_t, p)

    print(f"\n  N total:        {valid_final.sum()}")
    print(f"  N amyloid+:     {int(y_t.sum())} ({y_t.mean()*100:.1f}%)")
    print(f"  N amyloid-:     {int(len(y_t) - y_t.sum())} ({(1-y_t.mean())*100:.1f}%)")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"\n  AUC:            {auc:.4f}")
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  Sensitivity:    {sensitivity:.4f}")
    print(f"  Specificity:    {specificity:.4f}")
    print(f"  PPV:            {ppv:.4f}")
    print(f"  NPV:            {npv:.4f}")
    print(f"  Brier Score:    {brier:.4f}")

    # 95% CI for AUC via bootstrap
    print("\n  Bootstrap 95% CI for AUC (2000 iterations)...")
    n_boot = 2000
    boot_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y_t), size=len(y_t), replace=True)
        if len(np.unique(y_t[idx])) > 1:
            boot_aucs.append(roc_auc_score(y_t[idx], p[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  AUC 95% CI:    [{ci_low:.4f}, {ci_high:.4f}]")

    # STEP 7: 90/90 operating points
    print(f"\n{'='*60}")
    print("90/90 OPERATING POINT ANALYSIS (EXTERNAL)")
    print(f"{'='*60}")

    fpr, tpr, thresholds = roc_curve(y_t, p)

    # Rule-out: >=90% sensitivity
    sens_90_idx = np.where(tpr >= 0.90)[0]
    rule_out = {}
    if len(sens_90_idx) > 0:
        t = thresholds[sens_90_idx[0]]
        s = tpr[sens_90_idx[0]]
        sp = 1 - fpr[sens_90_idx[0]]
        rule_out = {'threshold': float(t), 'sensitivity': float(s), 'specificity': float(sp)}
        print(f"\n  Rule-Out (>=90% sensitivity):")
        print(f"    Threshold:    {t:.4f}")
        print(f"    Sensitivity:  {s:.4f}")
        print(f"    Specificity:  {sp:.4f}")

    # Rule-in: >=90% specificity
    spec_90_idx = np.where((1 - fpr) >= 0.90)[0]
    rule_in = {}
    if len(spec_90_idx) > 0:
        t = thresholds[spec_90_idx[-1]]
        s = tpr[spec_90_idx[-1]]
        sp = 1 - fpr[spec_90_idx[-1]]
        rule_in = {'threshold': float(t), 'sensitivity': float(s), 'specificity': float(sp)}
        print(f"\n  Rule-In (>=90% specificity):")
        print(f"    Threshold:    {t:.4f}")
        print(f"    Sensitivity:  {s:.4f}")
        print(f"    Specificity:  {sp:.4f}")

    # 95% sensitivity
    sens_95_idx = np.where(tpr >= 0.95)[0]
    if len(sens_95_idx) > 0:
        t = thresholds[sens_95_idx[0]]
        print(f"\n  Rule-Out (>=95% sensitivity):")
        print(f"    Threshold:    {t:.4f}")
        print(f"    Sensitivity:  {tpr[sens_95_idx[0]]:.4f}")
        print(f"    Specificity:  {1-fpr[sens_95_idx[0]]:.4f}")

    # STEP 8: Subgroup analysis
    print(f"\n{'='*60}")
    print("SUBGROUP ANALYSIS")
    print(f"{'='*60}")

    # By stage
    for stage in ['gatekeeper_neg', 'gatekeeper_pos', 'reflex']:
        mask = (a4_stages == stage) & valid_final
        if mask.sum() > 0:
            stage_y = y_true[mask]
            stage_p = a4_probs[mask]
            stage_pred = (stage_p >= 0.5).astype(int)
            n_pos = int(stage_y.sum())
            n_neg = int(mask.sum() - n_pos)
            acc = accuracy_score(stage_y, stage_pred)
            print(f"\n  {stage}: N={mask.sum()} (A+={n_pos}, A-={n_neg}), Accuracy={acc:.3f}")
            if len(np.unique(stage_y)) > 1:
                stage_auc = roc_auc_score(stage_y, stage_p)
                print(f"    AUC: {stage_auc:.3f}")

    # APOE4 subgroups
    if 'APOE4_carrier' in a4_df.columns:
        for label, val in [("non-carriers", 0), ("carriers", 1)]:
            mask = (a4_df['APOE4_carrier'].values == val) & valid_final
            if mask.sum() > 5 and len(np.unique(y_true[mask])) > 1:
                sub_auc = roc_auc_score(y_true[mask], a4_probs[mask])
                sub_acc = accuracy_score(y_true[mask], (a4_probs[mask] >= 0.5).astype(int))
                print(f"\n  APOE4 {label}: N={mask.sum()}, AUC={sub_auc:.3f}, Acc={sub_acc:.3f}")

    # Centiloid correlation
    print(f"\n{'='*60}")
    print("CENTILOID CORRELATION")
    print(f"{'='*60}")
    cl_valid = a4_df['AMYLCENT'].notna().values & valid_final
    if cl_valid.sum() > 0:
        r, p_val = stats.spearmanr(a4_df.loc[cl_valid, 'AMYLCENT'].values, a4_probs[cl_valid])
        print(f"  Spearman r = {r:.3f}, p = {p_val:.2e}, N = {cl_valid.sum()}")

    # STEP 9: Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    preds_df = pd.DataFrame({
        'true_amyloid': y_true,
        'predicted_prob': a4_probs,
        'predicted_class': (a4_probs >= 0.5).astype(int) if not np.all(np.isnan(a4_probs)) else np.nan,
        'stage': a4_stages,
        'centiloid': a4_df['AMYLCENT'].values,
    })
    preds_path = os.path.join(RESULTS_DIR, 'a4_binary_validation_predictions.csv')
    preds_df.to_csv(preds_path, index=False)
    print(f"  Predictions: {preds_path}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'description': 'A4 + LEARN binary external validation (centiloid >= 20 = amyloid+)',
        'training': {
            'cohort': 'ADNI', 'n': len(adni_df),
            'n_positive': int(adni_y.sum()),
            'n_negative': int(len(adni_y) - adni_y.sum()),
        },
        'external_validation': {
            'cohort': 'A4 + LEARN',
            'n': int(valid_final.sum()),
            'n_positive': int(y_t.sum()),
            'n_negative': int(len(y_t) - y_t.sum()),
            'prevalence': round(float(y_t.mean()), 4),
            'auc': round(float(auc), 4),
            'auc_95ci': [round(float(ci_low), 4), round(float(ci_high), 4)],
            'accuracy': round(float(accuracy), 4),
            'sensitivity': round(float(sensitivity), 4),
            'specificity': round(float(specificity), 4),
            'ppv': round(float(ppv), 4),
            'npv': round(float(npv), 4),
            'brier': round(float(brier), 4),
            'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
        },
        'operating_points': {
            'rule_out_90': rule_out,
            'rule_in_90': rule_in,
        },
        'pipeline': {
            'gatekeeper_thresholds': [GK_LOW, GK_HIGH],
            'gatekeeper_intercept': round(float(gatekeeper.intercept_[0]), 4),
            'gatekeeper_coefficient': round(float(gatekeeper.coef_[0, 0]), 4),
            'reflex_features': features_used,
            'reflex_importance': {f: round(float(imp), 4) for f, imp in zip(features_used, rf.feature_importances_)},
            'n_gatekeeper_neg': int(neg_mask_gk.sum()),
            'n_gatekeeper_pos': int(pos_mask_gk.sum()),
            'n_reflex': int(gray_mask_gk.sum()),
        }
    }

    summary_path = os.path.join(RESULTS_DIR, 'a4_binary_validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")

    return summary


if __name__ == '__main__':
    main()
