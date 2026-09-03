"""
Gray Zone MRI Enhancement Analysis
==================================

Enhancing Gray Zone Resolution with Structural MRI Biomarkers

This script extends the Gatekeeper-Reflex algorithm by incorporating
structural MRI biomarkers (hippocampal and entorhinal cortex volumes)
into the Reflex stage for improved gray zone classification.

Reference:
    Parankusham et al. (2025-2026). Resolving Amyloid Status Uncertainty
    in Intermediate Plasma p-tau217 Levels.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy import stats
from _grad_paths import (RESULTS, ADNI_DIR, A4_DIR, DATA_DIR, PROJECT_ROOT, SYNTHETIC)  # noqa: F401

warnings.filterwarnings('ignore')


def get_data_paths():
    """Get paths to data directories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    neuro_dir = os.path.dirname(current_dir)
    moirai_dir = os.path.dirname(neuro_dir)
    parent_dir = os.path.dirname(moirai_dir)

    # Navigate up to AlzheimersDisease_Research_Personal
    base_dir = current_dir
    for _ in range(5):
        base_dir = os.path.dirname(base_dir)
        candidate = str(A4_DIR)
        if os.path.exists(candidate):
            break
    return {
        'a4': candidate,
        'output': str(RESULTS)
    }


def load_a4_complete_data(a4_path: str) -> pd.DataFrame:
    """
    Load A4 data with plasma biomarkers, MRI, and PET.

    Returns DataFrame with complete cases for MRI enhancement analysis.
    """
    print("\n" + "="*60)
    print("LOADING A4 DATA WITH MRI")
    print("="*60)

    # Load pTau217
    ptau_path = os.path.join(a4_path, 'Clinical', 'External Data', 'biomarker_pTau217.csv')
    ptau = pd.read_csv(ptau_path)
    ptau['pTau217'] = pd.to_numeric(ptau['ORRES'], errors='coerce')
    ptau.loc[ptau['pTau217'].isna(), 'pTau217'] = pd.to_numeric(
        ptau.loc[ptau['pTau217'].isna(), 'ORRESRAW'], errors='coerce'
    )
    ptau_bl = ptau.groupby('BID')['pTau217'].first().reset_index()
    print(f"  pTau217: {len(ptau_bl)} participants")

    # Load Roche plasma biomarkers
    roche_path = os.path.join(a4_path, 'Clinical', 'External Data', 'biomarker_Plasma_Roche_Results.csv')
    roche = pd.read_csv(roche_path)
    roche['LABRESN'] = pd.to_numeric(roche['LABRESN'], errors='coerce')
    roche['LBTESTCD'] = roche['LBTESTCD'].str.strip()

    # Pivot Roche biomarkers
    roche_pivot = roche.pivot_table(
        index=['BID', 'VISCODE'],
        columns='LBTESTCD',
        values='LABRESN',
        aggfunc='first'
    ).reset_index()

    # Get baseline
    roche_bl = roche_pivot.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()

    # Rename columns
    rename_map = {'GFAP': 'GFAP', 'NF-L': 'NfL', 'AMYLB42': 'AB42', 'AMYLB40': 'AB40'}
    roche_bl = roche_bl.rename(columns=rename_map)

    # Calculate ratios
    if 'AB42' in roche_bl.columns and 'AB40' in roche_bl.columns:
        roche_bl['AB42_40_ratio'] = roche_bl['AB42'] / (roche_bl['AB40'] * 1000)  # AB40 in ng/mL

    if 'pTau217' not in roche_bl.columns:
        roche_bl = roche_bl.merge(ptau_bl, on='BID', how='outer')

    # Calculate pTau217/AB42 ratio
    if 'AB42' in roche_bl.columns:
        roche_bl['pTau217_AB42_ratio'] = roche_bl['pTau217'] / roche_bl['AB42']

    print(f"  Plasma biomarkers: {len(roche_bl)} participants")

    # Load MRI
    mri_path = os.path.join(a4_path, 'Clinical', 'External Data', 'imaging_volumetric_mri.csv')
    mri = pd.read_csv(mri_path)

    # Calculate average hippocampus and entorhinal
    mri['Hippocampus'] = (mri['LeftHippocampus'] + mri['RightHippocampus']) / 2
    mri['Entorhinal'] = (mri['LeftEntorhinal'] + mri['RightEntorhinal']) / 2
    mri['ICV'] = mri['IntraCranialVolume']

    # Normalize to ICV (multiply by 1000 for interpretability)
    mri['Hippocampus_norm'] = (mri['Hippocampus'] / mri['ICV']) * 1000
    mri['Entorhinal_norm'] = (mri['Entorhinal'] / mri['ICV']) * 1000

    # Get baseline MRI
    mri_bl = mri.sort_values(['BID', 'VISCODE']).groupby('BID').first().reset_index()
    mri_bl = mri_bl[['BID', 'Hippocampus', 'Entorhinal', 'ICV', 'Hippocampus_norm', 'Entorhinal_norm']]
    print(f"  MRI volumes: {len(mri_bl)} participants")

    # Load subject info (demographics and PET)
    subj_path = os.path.join(a4_path, 'Clinical', 'Derived Data', 'SUBJINFO.csv')
    subj = pd.read_csv(subj_path)
    subj = subj[['BID', 'TX', 'AGEYR', 'SEX', 'APOEGNPRSNFLG', 'AMYLCENT', 'SUVRCER']]
    subj = subj.rename(columns={
        'AGEYR': 'Age',
        'APOEGNPRSNFLG': 'APOE4_carrier',
        'AMYLCENT': 'Centiloid'
    })
    print(f"  Subject info: {len(subj)} participants")

    # Merge all data
    df = roche_bl.merge(mri_bl, on='BID', how='inner')
    df = df.merge(subj, on='BID', how='inner')

    # Define amyloid positivity (centiloid > 20)
    df['amyloid_positive'] = (df['Centiloid'] > 20).astype(int)

    print(f"\n  Merged dataset: {len(df)} participants")
    print(f"  With pTau217: {df['pTau217'].notna().sum()}")
    print(f"  With Hippocampus: {df['Hippocampus_norm'].notna().sum()}")
    print(f"  With Centiloid: {df['Centiloid'].notna().sum()}")

    # Complete cases
    required_cols = ['pTau217', 'Hippocampus_norm', 'Centiloid']
    complete = df.dropna(subset=required_cols)
    print(f"  Complete cases: {len(complete)}")
    print(f"  Amyloid positive: {complete['amyloid_positive'].sum()} ({complete['amyloid_positive'].mean()*100:.1f}%)")

    return complete


def define_gray_zone(df: pd.DataFrame, low_pct: float = 0.25, high_pct: float = 0.75):
    """
    Define gray zone based on pTau217 probability thresholds.

    Uses logistic regression to find pTau217 values corresponding to
    25% and 75% probability of amyloid positivity.
    """
    from sklearn.linear_model import LogisticRegression

    # Fit logistic regression
    X = np.log1p(df['pTau217'].values).reshape(-1, 1)
    y = df['amyloid_positive'].values

    valid_mask = ~np.isnan(X.flatten()) & ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    model = LogisticRegression(random_state=42)
    model.fit(X_valid, y_valid)

    # Find pTau217 thresholds
    # P = sigmoid(intercept + coef * log(1+pTau217))
    # logit(P) = intercept + coef * log(1+pTau217)
    # log(1+pTau217) = (logit(P) - intercept) / coef

    def find_threshold(prob):
        logit_p = np.log(prob / (1 - prob))
        log_ptau = (logit_p - model.intercept_[0]) / model.coef_[0, 0]
        return np.expm1(log_ptau)

    low_threshold = find_threshold(low_pct)
    high_threshold = find_threshold(high_pct)

    print(f"\nGray Zone Thresholds:")
    print(f"  25% probability: pTau217 = {low_threshold:.3f} pg/mL")
    print(f"  75% probability: pTau217 = {high_threshold:.3f} pg/mL")

    # Classify
    df = df.copy()
    df['zone'] = 'gray'
    df.loc[df['pTau217'] < low_threshold, 'zone'] = 'low'
    df.loc[df['pTau217'] > high_threshold, 'zone'] = 'high'

    # Predicted probability
    df['pTau217_prob'] = model.predict_proba(np.log1p(df['pTau217'].values).reshape(-1, 1))[:, 1]

    print(f"\nZone Distribution:")
    print(f"  Low zone (confident negative): {(df['zone'] == 'low').sum()} ({(df['zone'] == 'low').mean()*100:.1f}%)")
    print(f"  Gray zone (uncertain): {(df['zone'] == 'gray').sum()} ({(df['zone'] == 'gray').mean()*100:.1f}%)")
    print(f"  High zone (confident positive): {(df['zone'] == 'high').sum()} ({(df['zone'] == 'high').mean()*100:.1f}%)")

    return df, low_threshold, high_threshold


def prepare_features(df: pd.DataFrame, include_mri: bool = True):
    """Prepare feature matrix for modeling."""

    # Base plasma features
    plasma_features = []

    if 'pTau217' in df.columns and df['pTau217'].notna().sum() > 0:
        df['pTau217_log'] = np.log1p(df['pTau217'])
        plasma_features.append('pTau217_log')

    if 'pTau217_AB42_ratio' in df.columns and df['pTau217_AB42_ratio'].notna().sum() > len(df) * 0.3:
        df['pTau217_AB42_log'] = np.log1p(df['pTau217_AB42_ratio'])
        plasma_features.append('pTau217_AB42_log')

    if 'AB42_40_ratio' in df.columns and df['AB42_40_ratio'].notna().sum() > len(df) * 0.3:
        df['AB42_40_log'] = np.log1p(df['AB42_40_ratio'])
        plasma_features.append('AB42_40_log')

    if 'GFAP' in df.columns and df['GFAP'].notna().sum() > len(df) * 0.3:
        df['GFAP_log'] = np.log1p(df['GFAP'])
        plasma_features.append('GFAP_log')

    if 'NfL' in df.columns and df['NfL'].notna().sum() > len(df) * 0.3:
        df['NfL_log'] = np.log1p(df['NfL'])
        plasma_features.append('NfL_log')

    # MRI features
    mri_features = []
    if include_mri:
        if 'Hippocampus_norm' in df.columns:
            mri_features.append('Hippocampus_norm')
        if 'Entorhinal_norm' in df.columns:
            mri_features.append('Entorhinal_norm')

    all_features = plasma_features + mri_features

    return all_features, plasma_features, mri_features


def evaluate_model(df: pd.DataFrame, features: list, n_splits: int = 5):
    """
    Evaluate Random Forest model using stratified k-fold CV.

    Returns AUC scores and feature importances.
    """
    # Prepare data
    X = df[features].copy()
    y = df['amyloid_positive'].values

    # Handle missing values with median imputation
    for col in features:
        X[col] = X[col].fillna(X[col].median())

    X = X.values

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []
    importances = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)
        importances.append(model.feature_importances_)

    # Average importance
    mean_importance = np.mean(importances, axis=0)

    return {
        'auc_scores': auc_scores,
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores),
        'feature_importance': dict(zip(features, mean_importance))
    }


def bootstrap_comparison(df: pd.DataFrame, plasma_features: list, all_features: list, n_bootstrap: int = 1000):
    """
    Bootstrap comparison of plasma-only vs plasma+MRI models.
    """
    print(f"\nRunning bootstrap comparison ({n_bootstrap} iterations)...")

    X_plasma = df[plasma_features].copy()
    X_all = df[all_features].copy()
    y = df['amyloid_positive'].values

    # Impute missing
    for col in plasma_features:
        X_plasma[col] = X_plasma[col].fillna(X_plasma[col].median())
    for col in all_features:
        X_all[col] = X_all[col].fillna(X_all[col].median())

    X_plasma = X_plasma.values
    X_all = X_all.values

    auc_diffs = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(len(y), size=len(y), replace=True)
        X_plasma_boot = X_plasma[idx]
        X_all_boot = X_all[idx]
        y_boot = y[idx]

        # Out-of-bag indices
        oob_idx = list(set(range(len(y))) - set(idx))
        if len(oob_idx) < 10:
            continue

        X_plasma_oob = X_plasma[oob_idx]
        X_all_oob = X_all[oob_idx]
        y_oob = y[oob_idx]

        # Scale
        scaler_plasma = StandardScaler()
        scaler_all = StandardScaler()

        X_plasma_boot_scaled = scaler_plasma.fit_transform(X_plasma_boot)
        X_plasma_oob_scaled = scaler_plasma.transform(X_plasma_oob)

        X_all_boot_scaled = scaler_all.fit_transform(X_all_boot)
        X_all_oob_scaled = scaler_all.transform(X_all_oob)

        # Train models
        model_plasma = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model_all = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        model_plasma.fit(X_plasma_boot_scaled, y_boot)
        model_all.fit(X_all_boot_scaled, y_boot)

        # Evaluate on OOB
        try:
            auc_plasma = roc_auc_score(y_oob, model_plasma.predict_proba(X_plasma_oob_scaled)[:, 1])
            auc_all = roc_auc_score(y_oob, model_all.predict_proba(X_all_oob_scaled)[:, 1])
            auc_diffs.append(auc_all - auc_plasma)
        except:
            continue

        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap}")

    auc_diffs = np.array(auc_diffs)

    return {
        'mean_diff': np.mean(auc_diffs),
        'ci_low': np.percentile(auc_diffs, 2.5),
        'ci_high': np.percentile(auc_diffs, 97.5),
        'prop_favoring_mri': (auc_diffs > 0).mean()
    }


def hippocampal_stratification(df: pd.DataFrame):
    """Analyze amyloid positivity by hippocampal volume tertiles."""

    # Create tertiles
    df = df.copy()
    df['hipp_tertile'] = pd.qcut(df['Hippocampus_norm'], q=3, labels=['Small', 'Medium', 'Large'])

    results = []
    for tertile in ['Small', 'Medium', 'Large']:
        subset = df[df['hipp_tertile'] == tertile]
        results.append({
            'Tertile': tertile,
            'N': len(subset),
            'Amyloid_Positive_Rate': subset['amyloid_positive'].mean(),
            'Mean_pTau217': subset['pTau217'].mean(),
            'Mean_Centiloid': subset['Centiloid'].mean()
        })

    return pd.DataFrame(results)


def main():
    """Main analysis function."""

    print("\n" + "="*60)
    print("GRAY ZONE MRI ENHANCEMENT ANALYSIS")
    print("="*60)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get paths
    paths = get_data_paths()
    os.makedirs(paths['output'], exist_ok=True)

    # Load data
    df = load_a4_complete_data(paths['a4'])

    # Define gray zone
    df, low_thresh, high_thresh = define_gray_zone(df)

    # Filter to gray zone
    gray_df = df[df['zone'] == 'gray'].copy()
    print(f"\n" + "="*60)
    print(f"GRAY ZONE ANALYSIS (N={len(gray_df)})")
    print("="*60)

    # Prepare features
    all_features, plasma_features, mri_features = prepare_features(gray_df, include_mri=True)

    print(f"\nFeatures:")
    print(f"  Plasma: {plasma_features}")
    print(f"  MRI: {mri_features}")

    # Model 1: Plasma only
    print("\n" + "-"*40)
    print("MODEL 1: Plasma-only (Reflex baseline)")
    print("-"*40)
    plasma_results = evaluate_model(gray_df, plasma_features)
    print(f"  AUC: {plasma_results['mean_auc']:.3f} +/- {plasma_results['std_auc']:.3f}")
    print(f"  Feature importance:")
    for feat, imp in sorted(plasma_results['feature_importance'].items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.3f}")

    # Model 2: Plasma + MRI
    print("\n" + "-"*40)
    print("MODEL 2: Plasma + MRI (Enhanced)")
    print("-"*40)
    enhanced_results = evaluate_model(gray_df, all_features)
    print(f"  AUC: {enhanced_results['mean_auc']:.3f} +/- {enhanced_results['std_auc']:.3f}")
    print(f"  Feature importance:")
    for feat, imp in sorted(enhanced_results['feature_importance'].items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.3f}")

    # Model 3: MRI only
    print("\n" + "-"*40)
    print("MODEL 3: MRI-only (Benchmark)")
    print("-"*40)
    mri_results = evaluate_model(gray_df, mri_features)
    print(f"  AUC: {mri_results['mean_auc']:.3f} +/- {mri_results['std_auc']:.3f}")

    # AUC improvement
    delta_auc = enhanced_results['mean_auc'] - plasma_results['mean_auc']

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(enhanced_results['auc_scores'], plasma_results['auc_scores'])

    print("\n" + "="*60)
    print("AUC IMPROVEMENT WITH MRI")
    print("="*60)
    print(f"  Delta AUC: {delta_auc:+.3f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    # Bootstrap comparison
    bootstrap_results = bootstrap_comparison(gray_df, plasma_features, all_features, n_bootstrap=1000)
    print(f"\nBootstrap Validation (N=1000):")
    print(f"  Mean AUC improvement: {bootstrap_results['mean_diff']:+.3f}")
    print(f"  95% CI: [{bootstrap_results['ci_low']:.3f}, {bootstrap_results['ci_high']:.3f}]")
    print(f"  Proportion favoring MRI: {bootstrap_results['prop_favoring_mri']*100:.1f}%")

    # Hippocampal stratification
    print("\n" + "="*60)
    print("HIPPOCAMPAL VOLUME STRATIFICATION")
    print("="*60)
    hipp_strat = hippocampal_stratification(gray_df)
    print(hipp_strat.to_string(index=False))

    # High zone accuracy
    high_zone = df[df['zone'] == 'high']
    high_accuracy = high_zone['amyloid_positive'].mean()
    print(f"\nHigh zone accuracy (predict positive): {high_accuracy*100:.1f}%")

    # Save results
    results_summary = {
        'Analysis': 'Gray Zone MRI Enhancement',
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Total_N': len(df),
        'Gray_Zone_N': len(gray_df),
        'Plasma_Only_AUC': plasma_results['mean_auc'],
        'Plasma_MRI_AUC': enhanced_results['mean_auc'],
        'MRI_Only_AUC': mri_results['mean_auc'],
        'Delta_AUC': delta_auc,
        'p_value': p_value,
        'Bootstrap_Mean_Diff': bootstrap_results['mean_diff'],
        'Bootstrap_CI_Low': bootstrap_results['ci_low'],
        'Bootstrap_CI_High': bootstrap_results['ci_high'],
        'Prop_Favoring_MRI': bootstrap_results['prop_favoring_mri']
    }

    results_df = pd.DataFrame([results_summary])
    results_path = str(RESULTS / 'mri_enhancement_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'Feature': list(enhanced_results['feature_importance'].keys()),
        'Importance': list(enhanced_results['feature_importance'].values())
    }).sort_values('Importance', ascending=False)
    importance_path = str(RESULTS / 'mri_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)

    # Save hippocampal stratification
    hipp_path = str(RESULTS / 'hippocampal_stratification.csv')
    hipp_strat.to_csv(hipp_path, index=False)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return {
        'plasma_results': plasma_results,
        'enhanced_results': enhanced_results,
        'mri_results': mri_results,
        'bootstrap_results': bootstrap_results,
        'hippocampal_stratification': hipp_strat
    }


if __name__ == '__main__':
    main()
