#!/usr/bin/env python3
"""
Reflex Training Strategy Comparison
====================================
Compares gray zone performance when training Reflex on:
  A) Gray zone samples only (current approach, ~141 per fold)
  B) All samples (~319 per fold), predict only on gray zone
  C) All samples + logistic regression (simpler model, may generalize better)
  D) All samples + XGBoost (gradient boosting)
  E) Blended: average Gatekeeper prob + Reflex prob for gray zone cases

Uses identical LOOCV, harmonization, and Gatekeeper to ensure fair comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from _grad_paths import (RESULTS, ADNI_DIR, A4_DIR, DATA_DIR, PROJECT_ROOT, SYNTHETIC)  # noqa: F401
from data_loader import ADNIDataLoader
from harmonizer import AssayHarmonizer
from gatekeeper import GatekeeperModel
from reflex import ReflexModel

from _grad_paths import RESULTS


def engineer_features(df):
    """Replicate ReflexModel._engineer_features for standalone use."""
    result = df.copy()
    if 'pTau217_raw' in df.columns and 'AB42_40_ratio' in df.columns:
        result['tau_ab42_diff'] = np.log1p(df['pTau217_raw']) - np.log1p(df['AB42_40_ratio'])
    if 'NfL_Z' in df.columns and 'AGE' in df.columns:
        result['nfl_age_interaction'] = df['NfL_Z'] * df['AGE']
    if 'GFAP_Z' in df.columns and 'pTau217_Z' in df.columns:
        result['gfap_tau_interaction'] = df['GFAP_Z'] * df['pTau217_Z']
    # AGE passed through directly (no standardization needed for RF)
    return result


FEATURE_COLS = [
    'pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
    'APOE4_carrier', 'gfap_tau_interaction'
]


def prepare_features(df, feature_cols, scaler=None, fit=False):
    """Extract feature matrix, impute NaN, scale."""
    X = df[feature_cols].values.copy()
    for i in range(X.shape[1]):
        nan_mask = np.isnan(X[:, i])
        if nan_mask.any():
            X[nan_mask, i] = np.nanmedian(X[:, i])
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def run_comparison(df, target_col='amyloid_positive', gk_low=0.25, gk_high=0.75):
    """Run LOOCV with multiple Reflex training strategies."""
    n = len(df)

    # Storage for each strategy
    strategies = {
        'A_gz_only_rf': {'name': 'Gray-zone-only RF (current)', 'probs': np.zeros(n), 'stages': []},
        'B_all_data_rf': {'name': 'All-data RF', 'probs': np.zeros(n), 'stages': []},
        'C_all_data_lr': {'name': 'All-data Logistic Reg', 'probs': np.zeros(n), 'stages': []},
        'D_all_data_xgb': {'name': 'All-data GBM', 'probs': np.zeros(n), 'stages': []},
        'E_blended': {'name': 'Blended (GK+Reflex avg)', 'probs': np.zeros(n), 'stages': []},
    }

    for i in range(n):
        if i % 50 == 0:
            print(f"  Fold {i+1}/{n}")

        train_idx = list(range(n))
        train_idx.remove(i)

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[[i]].copy()
        train_y = train_df[target_col]

        # Harmonize within fold
        harmonizer = AssayHarmonizer()
        train_h = harmonizer.fit_transform(train_df)
        test_h = harmonizer.transform(test_df)

        # Gatekeeper (same for all strategies)
        gk = GatekeeperModel(low_threshold=gk_low, high_threshold=gk_high)
        gk.fit(train_h, train_y)
        gk_result = gk.classify(test_h)
        gk_prob = gk_result['probability'].values[0]
        in_gz = gk_result['in_gray_zone'].values[0]

        if not in_gz:
            # Gatekeeper resolved — same for all strategies
            for key in strategies:
                strategies[key]['probs'][i] = gk_prob
                strategies[key]['stages'].append('gatekeeper')
            continue

        # Gray zone case — test different training strategies
        train_gk = gk.classify(train_h)
        train_gz_idx = train_gk[train_gk['in_gray_zone']].index

        # Engineer features for all training data and test
        train_h_eng = engineer_features(train_h)
        test_h_eng = engineer_features(test_h)

        available_features = [f for f in FEATURE_COLS if f in train_h_eng.columns]

        # --- Strategy A: Train on gray zone only (current) ---
        if len(train_gz_idx) >= 10:
            train_gz = train_h_eng.loc[train_gz_idx]
            train_gz_y = train_y.loc[train_gz_idx]
            try:
                X_train, scaler_a = prepare_features(train_gz, available_features, fit=True)
                X_test, _ = prepare_features(test_h_eng, available_features, scaler=scaler_a)
                rf_a = RandomForestClassifier(
                    n_estimators=100, max_depth=5, min_samples_leaf=5,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                rf_a.fit(X_train, train_gz_y.values)
                prob_a = rf_a.predict_proba(X_test)[:, 1][0]
                strategies['A_gz_only_rf']['probs'][i] = prob_a
                strategies['A_gz_only_rf']['stages'].append('reflex')
            except Exception:
                strategies['A_gz_only_rf']['probs'][i] = gk_prob
                strategies['A_gz_only_rf']['stages'].append('gatekeeper_fallback')
        else:
            strategies['A_gz_only_rf']['probs'][i] = gk_prob
            strategies['A_gz_only_rf']['stages'].append('gatekeeper_fallback')

        # --- Strategy B: Train on ALL data, RF ---
        try:
            X_train_all, scaler_b = prepare_features(train_h_eng, available_features, fit=True)
            X_test_b, _ = prepare_features(test_h_eng, available_features, scaler=scaler_b)
            rf_b = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            rf_b.fit(X_train_all, train_y.values)
            prob_b = rf_b.predict_proba(X_test_b)[:, 1][0]
            strategies['B_all_data_rf']['probs'][i] = prob_b
            strategies['B_all_data_rf']['stages'].append('reflex')
        except Exception:
            strategies['B_all_data_rf']['probs'][i] = gk_prob
            strategies['B_all_data_rf']['stages'].append('gatekeeper_fallback')

        # --- Strategy C: Train on ALL data, Logistic Regression ---
        try:
            X_train_all, scaler_c = prepare_features(train_h_eng, available_features, fit=True)
            X_test_c, _ = prepare_features(test_h_eng, available_features, scaler=scaler_c)
            lr_c = LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs', max_iter=1000,
                class_weight='balanced', random_state=42
            )
            lr_c.fit(X_train_all, train_y.values)
            prob_c = lr_c.predict_proba(X_test_c)[:, 1][0]
            strategies['C_all_data_lr']['probs'][i] = prob_c
            strategies['C_all_data_lr']['stages'].append('reflex')
        except Exception:
            strategies['C_all_data_lr']['probs'][i] = gk_prob
            strategies['C_all_data_lr']['stages'].append('gatekeeper_fallback')

        # --- Strategy D: Train on ALL data, Gradient Boosting ---
        try:
            X_train_all, scaler_d = prepare_features(train_h_eng, available_features, fit=True)
            X_test_d, _ = prepare_features(test_h_eng, available_features, scaler=scaler_d)
            gbm_d = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=10, random_state=42
            )
            gbm_d.fit(X_train_all, train_y.values)
            prob_d = gbm_d.predict_proba(X_test_d)[:, 1][0]
            strategies['D_all_data_xgb']['probs'][i] = prob_d
            strategies['D_all_data_xgb']['stages'].append('reflex')
        except Exception:
            strategies['D_all_data_xgb']['probs'][i] = gk_prob
            strategies['D_all_data_xgb']['stages'].append('gatekeeper_fallback')

        # --- Strategy E: Blended (average GK prob + all-data RF prob) ---
        try:
            prob_blend = 0.5 * gk_prob + 0.5 * prob_b  # reuse strategy B's prediction
            strategies['E_blended']['probs'][i] = prob_blend
            strategies['E_blended']['stages'].append('reflex')
        except Exception:
            strategies['E_blended']['probs'][i] = gk_prob
            strategies['E_blended']['stages'].append('gatekeeper_fallback')

    return strategies


def compute_metrics(y_true, probs, stages):
    """Compute overall and gray-zone-specific metrics."""
    valid = ~np.isnan(y_true) & ~np.isnan(probs)

    # Overall
    auc = roc_auc_score(y_true[valid], probs[valid])
    acc = ((probs[valid] >= 0.5) == y_true[valid]).mean()
    brier = brier_score_loss(y_true[valid], probs[valid])

    preds = (probs[valid] >= 0.5).astype(int)
    tp = ((preds == 1) & (y_true[valid] == 1)).sum()
    fn = ((preds == 0) & (y_true[valid] == 1)).sum()
    tn = ((preds == 0) & (y_true[valid] == 0)).sum()
    fp = ((preds == 1) & (y_true[valid] == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Gray zone only
    gz_mask = np.array(stages) == 'reflex'
    gz_n = gz_mask.sum()
    if gz_n > 5 and len(np.unique(y_true[gz_mask])) > 1:
        gz_auc = roc_auc_score(y_true[gz_mask], probs[gz_mask])
        gz_acc = ((probs[gz_mask] >= 0.5) == y_true[gz_mask]).mean()

        gz_preds = (probs[gz_mask] >= 0.5).astype(int)
        gz_tp = ((gz_preds == 1) & (y_true[gz_mask] == 1)).sum()
        gz_fn = ((gz_preds == 0) & (y_true[gz_mask] == 1)).sum()
        gz_tn = ((gz_preds == 0) & (y_true[gz_mask] == 0)).sum()
        gz_fp = ((gz_preds == 1) & (y_true[gz_mask] == 0)).sum()
        gz_sens = gz_tp / (gz_tp + gz_fn) if (gz_tp + gz_fn) > 0 else np.nan
        gz_spec = gz_tn / (gz_tn + gz_fp) if (gz_tn + gz_fp) > 0 else np.nan
    else:
        gz_auc = gz_acc = gz_sens = gz_spec = np.nan

    return {
        'overall_auc': auc, 'overall_acc': acc, 'overall_brier': brier,
        'overall_sens': sens, 'overall_spec': spec,
        'gz_n': gz_n, 'gz_auc': gz_auc, 'gz_acc': gz_acc,
        'gz_sens': gz_sens, 'gz_spec': gz_spec,
    }


def main():
    print("=" * 70)
    print("REFLEX TRAINING STRATEGY COMPARISON")
    print("=" * 70)

    # Load ADNI data
    adni_path = str(ADNI_DIR)

    loader = ADNIDataLoader(adni_path)
    df = loader.merge_data(use_baseline_only=True)
    print(f"Loaded {len(df)} ADNI participants\n")

    print("Running LOOCV with 5 Reflex training strategies...")
    strategies = run_comparison(df)

    y_true = df['amyloid_positive'].values

    # Compute and display results
    results = []
    print("\n" + "=" * 120)
    print(f"{'Strategy':<35} {'OVR AUC':>8} {'OVR Acc':>8} {'OVR Sens':>9} {'OVR Spec':>9} "
          f"{'GZ N':>5} {'GZ AUC':>7} {'GZ Acc':>7} {'GZ Sens':>8} {'GZ Spec':>8} {'Brier':>7}")
    print("-" * 120)

    for key, strat in strategies.items():
        m = compute_metrics(y_true, strat['probs'], strat['stages'])
        m['strategy'] = strat['name']
        results.append(m)

        print(f"{strat['name']:<35} {m['overall_auc']:>8.4f} {m['overall_acc']:>8.4f} "
              f"{m['overall_sens']:>9.4f} {m['overall_spec']:>9.4f} "
              f"{m['gz_n']:>5d} {m['gz_auc']:>7.4f} {m['gz_acc']:>7.4f} "
              f"{m['gz_sens']:>8.4f} {m['gz_spec']:>8.4f} {m['overall_brier']:>7.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    col_order = ['strategy', 'overall_auc', 'overall_acc', 'overall_sens', 'overall_spec',
                 'overall_brier', 'gz_n', 'gz_auc', 'gz_acc', 'gz_sens', 'gz_spec']
    results_df = results_df[col_order]
    results_df.to_csv(RESULTS / 'reflex_training_comparison.csv', index=False)
    print(f"\nSaved results to results/reflex_training_comparison.csv")

    # Highlight best gray zone AUC
    best_idx = results_df['gz_auc'].idxmax()
    best = results_df.loc[best_idx]
    baseline = results_df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"BEST GRAY ZONE AUC: {best['strategy']} → {best['gz_auc']:.4f}")
    print(f"  vs baseline: {baseline['gz_auc']:.4f} (Δ = {best['gz_auc'] - baseline['gz_auc']:+.4f})")
    print(f"  GZ accuracy: {best['gz_acc']:.4f} vs {baseline['gz_acc']:.4f} (Δ = {best['gz_acc'] - baseline['gz_acc']:+.4f})")
    print(f"  Overall AUC: {best['overall_auc']:.4f} vs {baseline['overall_auc']:.4f} (Δ = {best['overall_auc'] - baseline['overall_auc']:+.4f})")


if __name__ == '__main__':
    main()
