#!/usr/bin/env python3
"""
NfL Ablation Analysis for GRAD Pipeline
========================================
Tests the incremental value of NfL-based features in the Reflex model
by running LOOCV with three feature configurations:

1. Base model (no NfL): pTau217_Z, tau_ab42_diff, GFAP_Z, AGE, APOE4_carrier, gfap_tau_interaction
2. +NfL_Z: Base + NfL_Z
3. +NfL_Z + nfl_age_interaction: Base + NfL_Z + nfl_age_interaction

Output: results/supp_table_nfl_ablation.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ADNIDataLoader
from harmonizer import AssayHarmonizer
from gatekeeper import GatekeeperModel
from reflex import ReflexModel

RESULTS = Path(__file__).parent / 'results'


def run_loocv_with_features(df, target_col, feature_set, gk_low=0.25, gk_high=0.75):
    """Run LOOCV with a specific forced feature set for the Reflex model."""
    n = len(df)
    all_probs = np.zeros(n)
    all_stages = []

    for i in range(n):
        train_idx = list(range(n))
        train_idx.remove(i)

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[[i]].copy()

        train_y = train_df[target_col]

        # Harmonize within fold
        harmonizer = AssayHarmonizer()
        train_h = harmonizer.fit_transform(train_df)
        test_h = harmonizer.transform(test_df)

        # Gatekeeper
        gk = GatekeeperModel(low_threshold=gk_low, high_threshold=gk_high)
        gk.fit(train_h, train_y)
        gk_result = gk.classify(test_h)
        gk_prob = gk_result['probability'].values[0]
        in_gz = gk_result['in_gray_zone'].values[0]

        if not in_gz:
            all_probs[i] = gk_prob
            all_stages.append('gatekeeper')
        else:
            # Train Reflex on gray zone subset
            train_gk = gk.classify(train_h)
            train_gz_idx = train_gk[train_gk['in_gray_zone']].index

            if len(train_gz_idx) >= 10:
                reflex = ReflexModel(n_estimators=100, max_depth=5)
                train_gz = train_h.loc[train_gz_idx]
                train_gz_y = train_y.loc[train_gz_idx]

                try:
                    reflex.fit(train_gz, train_gz_y, feature_cols=feature_set)
                    reflex_prob = reflex.predict_proba(test_h)[0]
                    all_probs[i] = reflex_prob
                    all_stages.append('reflex')
                except Exception:
                    all_probs[i] = gk_prob
                    all_stages.append('gatekeeper_fallback')
            else:
                all_probs[i] = gk_prob
                all_stages.append('gatekeeper_fallback')

    y_true = df[target_col].values
    valid = ~np.isnan(y_true) & ~np.isnan(all_probs)

    auc = roc_auc_score(y_true[valid], all_probs[valid])
    acc = ((all_probs[valid] >= 0.5) == y_true[valid]).mean()
    brier = brier_score_loss(y_true[valid], all_probs[valid])

    preds_binary = (all_probs[valid] >= 0.5).astype(int)
    tp = ((preds_binary == 1) & (y_true[valid] == 1)).sum()
    fn = ((preds_binary == 0) & (y_true[valid] == 1)).sum()
    tn = ((preds_binary == 0) & (y_true[valid] == 0)).sum()
    fp = ((preds_binary == 1) & (y_true[valid] == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Gray zone metrics
    gz_mask = np.array(all_stages) == 'reflex'
    gz_n = gz_mask.sum()
    if gz_n > 5 and len(np.unique(y_true[gz_mask])) > 1:
        gz_auc = roc_auc_score(y_true[gz_mask], all_probs[gz_mask])
    else:
        gz_auc = np.nan

    return {
        'auc': float(auc),
        'accuracy': float(acc),
        'brier_score': float(brier),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'n_reflex': int(gz_n),
        'gray_zone_auc': float(gz_auc) if not np.isnan(gz_auc) else None,
    }


def main():
    print("=" * 60)
    print("NfL ABLATION ANALYSIS")
    print("=" * 60)

    # Load ADNI data
    # Path: grayzone-classifier -> projects -> syntropi-ai-research -> Syntropi AI Group -> AlzheimersDisease_Research_Personal
    adni_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        'syntropi-ai-data', 'syntropi-ai-ADNI'
    )

    loader = ADNIDataLoader(adni_path)
    df = loader.merge_data(use_baseline_only=True)
    print(f"Loaded {len(df)} ADNI participants")

    # Check NfL availability
    nfl_avail = df['NfL_raw'].notna().sum() if 'NfL_raw' in df.columns else 0
    print(f"NfL available: {nfl_avail}/{len(df)} ({nfl_avail/len(df)*100:.1f}%)")

    # Define three feature configurations
    base_features = [
        'pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
        'APOE4_carrier', 'gfap_tau_interaction'
    ]

    configs = {
        'Base (no NfL)': base_features,
        'Base + NfL_Z': base_features + ['NfL_Z'],
        'Base + NfL_Z + nfl_age_interaction': base_features + ['NfL_Z', 'nfl_age_interaction'],
    }

    results = []
    for label, features in configs.items():
        print(f"\n{'─' * 50}")
        print(f"Running LOOCV: {label}")
        print(f"  Features: {features}")

        metrics = run_loocv_with_features(
            df, 'amyloid_positive', features
        )
        metrics['model'] = label
        metrics['n_features'] = len(features)
        metrics['features'] = ', '.join(features)
        results.append(metrics)

        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Brier: {metrics['brier_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

    # Format as supplementary table
    results_df = pd.DataFrame(results)
    col_order = ['model', 'n_features', 'auc', 'accuracy', 'sensitivity',
                 'specificity', 'brier_score', 'n_reflex', 'gray_zone_auc', 'features']
    results_df = results_df[col_order]

    # Compute deltas vs base
    base_auc = results_df.iloc[0]['auc']
    results_df['delta_auc'] = results_df['auc'] - base_auc

    results_df.to_csv(RESULTS / 'supp_table_nfl_ablation.csv', index=False)
    print(f"\nSaved supp_table_nfl_ablation.csv")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<38} {'AUC':>7} {'ΔAUC':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'Brier':>7}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(f"{row['model']:<38} {row['auc']:>7.4f} {row['delta_auc']:>+7.4f} "
              f"{row['accuracy']:>7.4f} {row['sensitivity']:>7.4f} {row['specificity']:>7.4f} "
              f"{row['brier_score']:>7.4f}")


if __name__ == '__main__':
    main()
