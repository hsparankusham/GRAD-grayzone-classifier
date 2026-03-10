#!/usr/bin/env python3
"""
Subgroup Performance Analysis for GRAD Pipeline
================================================
Stratifies ADNI LOOCV performance by cognitive status, APOE4, sex, and age.

Output: results/supp_table_subgroup_performance.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ADNIDataLoader

RESULTS = Path(__file__).parent / 'results'


def compute_metrics(y_true, y_prob):
    """Compute AUC, accuracy, sensitivity, specificity for a subgroup."""
    n = len(y_true)
    if n < 10 or len(np.unique(y_true)) < 2:
        return {'n': n, 'n_positive': int(y_true.sum()),
                'auc': np.nan, 'accuracy': np.nan,
                'sensitivity': np.nan, 'specificity': np.nan}

    auc = roc_auc_score(y_true, y_prob)
    preds = (y_prob >= 0.5).astype(int)
    acc = (preds == y_true).mean()

    tp = ((preds == 1) & (y_true == 1)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        'n': n,
        'n_positive': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
        'auc': float(auc),
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
    }


def main():
    print("=" * 60)
    print("SUBGROUP PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Load LOOCV predictions
    preds = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')

    # Load ADNI data to get demographics
    # Path: grayzone-classifier -> projects -> syntropi-ai-research -> Syntropi AI Group -> AlzheimersDisease_Research_Personal
    adni_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        'syntropi-ai-data', 'syntropi-ai-ADNI'
    )

    loader = ADNIDataLoader(adni_path)
    adni_df = loader.merge_data(use_baseline_only=True)

    # Align predictions with demographics
    if len(adni_df) != len(preds):
        print(f"Warning: ADNI df ({len(adni_df)}) != predictions ({len(preds)})")
        # Use min length
        n = min(len(adni_df), len(preds))
        adni_df = adni_df.iloc[:n]
        preds = preds.iloc[:n]

    y_true = preds['true_amyloid'].values
    y_prob = preds['predicted_prob'].values

    results = []

    # Overall
    m = compute_metrics(y_true, y_prob)
    m['subgroup'] = 'Overall'
    m['category'] = 'All'
    results.append(m)

    # 1. By cognitive status
    if 'DX' in adni_df.columns:
        dx_col = 'DX'
    elif 'diagnosis' in adni_df.columns:
        dx_col = 'diagnosis'
    else:
        dx_col = None

    if dx_col:
        for dx_val in adni_df[dx_col].dropna().unique():
            mask = adni_df[dx_col].values == dx_val
            if mask.sum() >= 10:
                m = compute_metrics(y_true[mask[:len(y_true)]], y_prob[mask[:len(y_prob)]])
                m['subgroup'] = str(dx_val)
                m['category'] = 'Cognitive Status'
                results.append(m)
                print(f"  {dx_val}: n={m['n']}, AUC={m['auc']:.3f}" if not np.isnan(m['auc']) else f"  {dx_val}: n={m['n']}, insufficient")

    # 2. By APOE4
    if 'APOE4_carrier' in adni_df.columns:
        for val, label in [(1, 'APOE4 carrier'), (0, 'APOE4 non-carrier')]:
            mask = adni_df['APOE4_carrier'].values == val
            if mask.sum() >= 10:
                m = compute_metrics(y_true[mask[:len(y_true)]], y_prob[mask[:len(y_prob)]])
                m['subgroup'] = label
                m['category'] = 'APOE4 Status'
                results.append(m)
                print(f"  {label}: n={m['n']}, AUC={m['auc']:.3f}" if not np.isnan(m['auc']) else f"  {label}: n={m['n']}, insufficient")

    # 3. By sex
    sex_col = None
    for col in ['PTGENDER', 'SEX', 'sex', 'Gender']:
        if col in adni_df.columns:
            sex_col = col
            break

    if sex_col:
        for val in adni_df[sex_col].dropna().unique():
            mask = adni_df[sex_col].values == val
            if mask.sum() >= 10:
                m = compute_metrics(y_true[mask[:len(y_true)]], y_prob[mask[:len(y_prob)]])
                m['subgroup'] = str(val)
                m['category'] = 'Sex'
                results.append(m)
                print(f"  {val}: n={m['n']}, AUC={m['auc']:.3f}" if not np.isnan(m['auc']) else f"  {val}: n={m['n']}, insufficient")

    # 4. By age tertile
    if 'AGE' in adni_df.columns:
        age = adni_df['AGE'].values[:len(y_true)]
        tertiles = np.percentile(age[~np.isnan(age)], [33.3, 66.7])
        age_groups = np.where(age <= tertiles[0], 'Young tertile',
                     np.where(age <= tertiles[1], 'Middle tertile', 'Old tertile'))

        for group in ['Young tertile', 'Middle tertile', 'Old tertile']:
            mask = age_groups == group
            if mask.sum() >= 10:
                m = compute_metrics(y_true[mask], y_prob[mask])
                m['subgroup'] = f"{group} (≤{tertiles[0]:.0f} / ≤{tertiles[1]:.0f} / >{tertiles[1]:.0f})"
                m['category'] = 'Age Tertile'
                results.append(m)
                print(f"  {group}: n={m['n']}, AUC={m['auc']:.3f}" if not np.isnan(m['auc']) else f"  {group}: n={m['n']}, insufficient")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / 'supp_table_subgroup_performance.csv', index=False)
    print(f"\nSaved supp_table_subgroup_performance.csv ({len(results_df)} subgroups)")

    # Print summary
    print("\n" + "=" * 80)
    print(f"{'Category':<20} {'Subgroup':<25} {'N':>5} {'AUC':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        auc_str = f"{row['auc']:.3f}" if not np.isnan(row['auc']) else 'N/A'
        acc_str = f"{row['accuracy']:.3f}" if not np.isnan(row['accuracy']) else 'N/A'
        sens_str = f"{row['sensitivity']:.3f}" if not np.isnan(row['sensitivity']) else 'N/A'
        spec_str = f"{row['specificity']:.3f}" if not np.isnan(row['specificity']) else 'N/A'
        print(f"{row['category']:<20} {row['subgroup']:<25} {row['n']:>5} {auc_str:>7} {acc_str:>7} {sens_str:>7} {spec_str:>7}")


if __name__ == '__main__':
    main()
