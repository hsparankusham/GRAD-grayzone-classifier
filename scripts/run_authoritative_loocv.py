#!/usr/bin/env python3
"""
Authoritative LOOCV with Locked 6-Feature Reflex
=================================================
Regenerates adni_loocv_predictions.csv and all derived metrics
with the EXACT 6 features described in the manuscript.

This replaces the run_analysis.py predictions which inadvertently
used 15 auto-selected features.

Locked features:
  1. pTau217_Z
  2. tau_ab42_diff
  3. GFAP_Z
  4. AGE_Z
  5. APOE4_carrier
  6. gfap_tau_interaction

Outputs:
  - results/adni_loocv_predictions.csv  (320 rows)
  - results/supp_table_by_stage.csv
  - results/supp_table_operating_points.csv
  - results/supp_table_bootstrap_ci.csv
  - results/supp_table_threshold_sensitivity.csv
  - results/feature_ablation_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.utils import resample
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ADNIDataLoader
from harmonizer import AssayHarmonizer
from gatekeeper import GatekeeperModel
from reflex import ReflexModel
from validation import LOOCVValidator

RESULTS = Path(__file__).parent / 'results'

# The locked 6-feature set
LOCKED_FEATURES = [
    'pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE_Z',
    'APOE4_carrier', 'gfap_tau_interaction'
]


def load_adni():
    """Load ADNI data."""
    adni_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        'syntropi-ai-data', 'syntropi-ai-ADNI'
    )
    loader = ADNIDataLoader(adni_path)
    return loader.merge_data(use_baseline_only=True)


def run_loocv_locked(df, target_col='amyloid_positive'):
    """Run LOOCV with locked 6 features."""
    print("Running LOOCV with locked 6-feature Reflex...")
    print(f"  Features: {LOCKED_FEATURES}")

    validator = LOOCVValidator(
        gatekeeper_low=0.25,
        gatekeeper_high=0.75,
        reflex_feature_cols=LOCKED_FEATURES
    )
    results = validator.validate_loocv(df, target_col, verbose=True)
    return results


def compute_stage_metrics(preds_df):
    """Compute per-stage metrics (Table: by_stage)."""
    y = preds_df['true_amyloid'].values
    p = preds_df['predicted_prob'].values

    rows = []
    for stage_name, mask_val in [('Gatekeeper', 'gatekeeper'), ('Reflex', 'reflex')]:
        mask = preds_df['stage'].values == mask_val
        if mask.sum() == 0:
            continue

        y_s = y[mask]
        p_s = p[mask]
        pred_s = (p_s >= 0.5).astype(int)

        tp = ((pred_s == 1) & (y_s == 1)).sum()
        fn = ((pred_s == 0) & (y_s == 1)).sum()
        tn = ((pred_s == 0) & (y_s == 0)).sum()
        fp = ((pred_s == 1) & (y_s == 0)).sum()

        n = len(y_s)
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        acc = (tp + tn) / n
        brier = brier_score_loss(y_s, p_s)

        if len(np.unique(y_s)) > 1:
            auc = roc_auc_score(y_s, p_s)
        else:
            auc = np.nan

        lr_pos = sens / (1 - spec) if spec < 1 else np.inf
        lr_neg = (1 - sens) / spec if spec > 0 else np.nan

        rows.append({
            'Stage': stage_name, 'N': n, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Sensitivity': sens, 'Specificity': spec, 'PPV': ppv, 'NPV': npv,
            'Accuracy': acc, 'AUC': auc, 'LR+': lr_pos, 'LR-': lr_neg, 'Brier': brier
        })

    return pd.DataFrame(rows)


def compute_operating_points(preds_df):
    """Compute metrics at key operating points."""
    y = preds_df['true_amyloid'].values
    p = preds_df['predicted_prob'].values
    valid = ~np.isnan(y) & ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]

    auc = roc_auc_score(y_v, p_v)
    brier = brier_score_loss(y_v, p_v)

    fpr, tpr, thresholds = roc_curve(y_v, p_v)

    def metrics_at_threshold(threshold, label):
        pred = (p_v >= threshold).astype(int)
        tp = ((pred == 1) & (y_v == 1)).sum()
        fn = ((pred == 0) & (y_v == 1)).sum()
        tn = ((pred == 0) & (y_v == 0)).sum()
        fp = ((pred == 1) & (y_v == 0)).sum()
        n = len(y_v)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        acc = (tp + tn) / n
        lr_pos = sens / (1 - spec) if spec < 1 else np.inf
        lr_neg = (1 - sens) / spec if spec > 0 else np.nan
        return {
            'Operating_Point': label, 'N': n, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Sensitivity': sens, 'Specificity': spec, 'PPV': ppv, 'NPV': npv,
            'Accuracy': acc, 'AUC': auc, 'LR+': lr_pos, 'LR-': lr_neg, 'Brier': brier
        }

    rows = [metrics_at_threshold(0.5, '50% threshold (standard)')]

    # Find threshold for 90% specificity (rule-in):
    # Want the LOWEST threshold that still gives >=90% specificity
    # (maximizes sensitivity while maintaining spec >= 0.90)
    spec_at = 1 - fpr  # specificity = 1 - fpr
    candidates_90spec = np.where(spec_at >= 0.90)[0]
    if len(candidates_90spec) > 0:
        # Among all points with spec >= 0.90, pick the one with highest sensitivity (tpr)
        best_spec_idx = candidates_90spec[np.argmax(tpr[candidates_90spec])]
        if best_spec_idx < len(thresholds):
            rows.append(metrics_at_threshold(
                thresholds[best_spec_idx], '90% Specificity (rule-in)'))

    # Find threshold for 90% sensitivity (rule-out):
    # Want the HIGHEST threshold that still gives >=90% sensitivity
    # (maximizes specificity while maintaining sens >= 0.90)
    candidates_90sens = np.where(tpr >= 0.90)[0]
    if len(candidates_90sens) > 0:
        # Among all points with sens >= 0.90, pick the one with highest specificity (lowest fpr)
        best_sens_idx = candidates_90sens[np.argmin(fpr[candidates_90sens])]
        if best_sens_idx < len(thresholds):
            rows.append(metrics_at_threshold(
                thresholds[best_sens_idx], '90% Sensitivity (rule-out)'))

    # Youden's J optimal
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    rows.append(metrics_at_threshold(best_thresh, "Youden's J optimal"))

    return pd.DataFrame(rows)


def compute_bootstrap_ci(preds_df, n_boot=2000, seed=42):
    """Compute bootstrap 95% CIs for key metrics."""
    rng = np.random.RandomState(seed)
    y = preds_df['true_amyloid'].values
    p = preds_df['predicted_prob'].values
    valid = ~np.isnan(y) & ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    n = len(y_v)

    boot_aucs, boot_accs, boot_sens, boot_specs = [], [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b, p_b = y_v[idx], p_v[idx]

        if len(np.unique(y_b)) < 2:
            continue

        boot_aucs.append(roc_auc_score(y_b, p_b))
        pred_b = (p_b >= 0.5).astype(int)
        boot_accs.append((pred_b == y_b).mean())

        tp = ((pred_b == 1) & (y_b == 1)).sum()
        fn = ((pred_b == 0) & (y_b == 1)).sum()
        tn = ((pred_b == 0) & (y_b == 0)).sum()
        fp = ((pred_b == 1) & (y_b == 0)).sum()

        boot_sens.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        boot_specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)

    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    point_auc = roc_auc_score(y_v, p_v)
    pred_v = (p_v >= 0.5).astype(int)
    point_acc = (pred_v == y_v).mean()
    tp = ((pred_v == 1) & (y_v == 1)).sum()
    fn = ((pred_v == 0) & (y_v == 1)).sum()
    tn = ((pred_v == 0) & (y_v == 0)).sum()
    fp = ((pred_v == 1) & (y_v == 0)).sum()
    point_sens = tp / (tp + fn)
    point_spec = tn / (tn + fp)

    rows = []
    for name, point, boots in [
        ('AUC', point_auc, boot_aucs),
        ('Accuracy', point_acc, boot_accs),
        ('Sensitivity', point_sens, boot_sens),
        ('Specificity', point_spec, boot_specs),
    ]:
        lo, hi = ci(boots)
        rows.append({
            'Metric': name, 'Estimate': point,
            '95% CI Lower': lo, '95% CI Upper': hi,
            '95% CI': f'[{lo:.3f}-{hi:.3f}]'
        })

    return pd.DataFrame(rows)


def compute_threshold_sensitivity(preds_df):
    """Compute resolution rates and accuracy across threshold pairs."""
    y = preds_df['true_amyloid'].values
    p = preds_df['predicted_prob'].values

    rows = []
    for low in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for high in [0.65, 0.70, 0.75, 0.80, 0.85]:
            neg_mask = p < low
            pos_mask = p > high
            resolved_mask = neg_mask | pos_mask
            gray_mask = ~resolved_mask

            n_neg = neg_mask.sum()
            n_pos = pos_mask.sum()
            n_resolved = resolved_mask.sum()
            resolution_rate = n_resolved / len(p)

            if n_resolved > 0:
                resolved_preds = np.where(p[resolved_mask] >= 0.5, 1, 0)
                resolved_acc = (resolved_preds == y[resolved_mask]).mean()
            else:
                resolved_acc = np.nan

            rows.append({
                'Low_Threshold': low, 'High_Threshold': high,
                'Resolution_Rate': resolution_rate,
                'Resolved_Accuracy': resolved_acc,
                'Gray_Zone_N': gray_mask.sum(),
                'Gray_Zone_%': gray_mask.mean() * 100,
                'N_Classified_Negative': n_neg,
                'N_Classified_Positive': n_pos,
            })

    return pd.DataFrame(rows)


def compute_feature_ablation(df, target_col='amyloid_positive'):
    """Drop-one feature ablation for the Reflex model."""
    print("\nRunning feature ablation (drop-one)...")

    rows = []
    for drop_feat in LOCKED_FEATURES:
        reduced = [f for f in LOCKED_FEATURES if f != drop_feat]
        validator = LOOCVValidator(
            gatekeeper_low=0.25, gatekeeper_high=0.75,
            reflex_feature_cols=reduced
        )
        results = validator.validate_loocv(df, target_col, verbose=False)

        y = results.predictions['true_amyloid'].values
        p = results.predictions['predicted_prob'].values
        valid = ~np.isnan(y) & ~np.isnan(p)
        pred = (p[valid] >= 0.5).astype(int)

        tp = ((pred == 1) & (y[valid] == 1)).sum()
        fn = ((pred == 0) & (y[valid] == 1)).sum()
        tn = ((pred == 0) & (y[valid] == 0)).sum()
        fp = ((pred == 1) & (y[valid] == 0)).sum()

        rows.append({
            'dropped_feature': drop_feat,
            'auc': results.overall_auc,
            'delta_auc': results.overall_auc - 0,  # will compute delta later
            'accuracy': (tp + tn) / valid.sum(),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        })
        print(f"  Dropped {drop_feat}: AUC = {results.overall_auc:.4f}")

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("AUTHORITATIVE LOOCV — LOCKED 6-FEATURE REFLEX")
    print("=" * 70)

    # Load data
    df = load_adni()
    print(f"Loaded {len(df)} ADNI participants\n")

    # 1. Run LOOCV
    results = run_loocv_locked(df)
    preds_df = results.predictions

    # Save predictions
    preds_df.to_csv(RESULTS / 'adni_loocv_predictions.csv', index=False)
    print(f"\nSaved adni_loocv_predictions.csv")

    # Verify
    y = preds_df['true_amyloid'].values
    p = preds_df['predicted_prob'].values
    valid = ~np.isnan(y) & ~np.isnan(p)
    auc = roc_auc_score(y[valid], p[valid])
    print(f"\n*** AUTHORITATIVE OVERALL AUC: {auc:.6f} ***")

    gz_mask = preds_df['stage'].values == 'reflex'
    if gz_mask.sum() > 0 and len(np.unique(y[gz_mask])) > 1:
        gz_auc = roc_auc_score(y[gz_mask], p[gz_mask])
        print(f"*** AUTHORITATIVE GRAY ZONE AUC: {gz_auc:.6f} ***")

    # 2. Stage metrics
    print("\nComputing stage metrics...")
    stage_df = compute_stage_metrics(preds_df)
    stage_df.to_csv(RESULTS / 'supp_table_by_stage.csv', index=False)
    print(stage_df.to_string(index=False))

    # 3. Operating points
    print("\nComputing operating points...")
    op_df = compute_operating_points(preds_df)
    op_df.to_csv(RESULTS / 'supp_table_operating_points.csv', index=False)
    print(op_df[['Operating_Point', 'Sensitivity', 'Specificity', 'AUC']].to_string(index=False))

    # 4. Bootstrap CIs
    print("\nComputing bootstrap CIs (2000 resamples)...")
    boot_df = compute_bootstrap_ci(preds_df)
    boot_df.to_csv(RESULTS / 'supp_table_bootstrap_ci.csv', index=False)
    print(boot_df.to_string(index=False))

    # 5. Threshold sensitivity
    print("\nComputing threshold sensitivity...")
    thresh_df = compute_threshold_sensitivity(preds_df)
    thresh_df.to_csv(RESULTS / 'supp_table_threshold_sensitivity.csv', index=False)

    # 6. Feature ablation
    ablation_df = compute_feature_ablation(df)
    base_auc = auc
    ablation_df['delta_auc'] = ablation_df['auc'] - base_auc
    ablation_df = ablation_df.sort_values('delta_auc')
    ablation_df.to_csv(RESULTS / 'feature_ablation_results.csv', index=False)
    print("\nFeature ablation:")
    print(ablation_df[['dropped_feature', 'auc', 'delta_auc']].to_string(index=False))

    # 7. Feature importance (from last fold's Reflex)
    if results.feature_importance is not None and len(results.feature_importance) > 0:
        print("\nFeature importance (averaged across folds):")
        print(results.feature_importance.to_string(index=False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — ALL NUMBERS FOR MANUSCRIPT")
    print("=" * 70)

    pred_v = (p[valid] >= 0.5).astype(int)
    tp = ((pred_v == 1) & (y[valid] == 1)).sum()
    fn = ((pred_v == 0) & (y[valid] == 1)).sum()
    tn = ((pred_v == 0) & (y[valid] == 0)).sum()
    fp = ((pred_v == 1) & (y[valid] == 0)).sum()

    print(f"Overall AUC: {auc:.3f} (95% CI: {boot_df.iloc[0]['95% CI']})")
    print(f"Accuracy: {(tp+tn)/valid.sum():.3f} (95% CI: {boot_df.iloc[1]['95% CI']})")
    print(f"Sensitivity: {tp/(tp+fn):.3f} (95% CI: {boot_df.iloc[2]['95% CI']})")
    print(f"Specificity: {tn/(tn+fp):.3f} (95% CI: {boot_df.iloc[3]['95% CI']})")
    print(f"PPV: {tp/(tp+fp):.3f}")
    print(f"NPV: {tn/(tn+fn):.3f}")
    print(f"LR+: {(tp/(tp+fn))/((fp/(fp+tn))):.2f}")
    print(f"LR-: {(fn/(tp+fn))/((tn/(tn+fp))):.2f}")
    print(f"Brier: {brier_score_loss(y[valid], p[valid]):.3f}")
    print(f"Gatekeeper resolved: {(preds_df['stage']=='gatekeeper').sum()}/320 ({(preds_df['stage']=='gatekeeper').mean()*100:.1f}%)")
    print(f"Gray zone: {gz_mask.sum()}/320 ({gz_mask.mean()*100:.1f}%)")
    print(f"Gray zone AUC: {gz_auc:.3f}")

    print("\nFiles regenerated:")
    for f in ['adni_loocv_predictions.csv', 'supp_table_by_stage.csv',
              'supp_table_operating_points.csv', 'supp_table_bootstrap_ci.csv',
              'supp_table_threshold_sensitivity.csv', 'feature_ablation_results.csv']:
        print(f"  results/{f}")


if __name__ == '__main__':
    main()
