#!/usr/bin/env python3
"""
Calibration Analysis for GRAD Pipeline
=======================================
Generates calibration curves, Hosmer-Lemeshow test, and Expected
Calibration Error (ECE) from LOOCV predictions.

Output:
  - results/calibration_analysis.csv (bin-level data)
  - results/calibration_metrics.json (HL chi-sq, ECE, calibration-in-the-large)
  - results/figure_s1_calibration.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.calibration import calibration_curve

RESULTS = Path(__file__).parent / 'results'


def hosmer_lemeshow(y_true, y_prob, n_bins=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    chi2 = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = bin_idx == i
        n_i = mask.sum()
        if n_i == 0:
            continue
        obs_i = y_true[mask].sum()
        exp_i = y_prob[mask].sum()
        # Avoid division by zero
        if exp_i > 0 and (n_i - exp_i) > 0:
            chi2 += (obs_i - exp_i) ** 2 / exp_i
            chi2 += ((n_i - obs_i) - (n_i - exp_i)) ** 2 / (n_i - exp_i)
        bin_data.append({
            'bin': i + 1,
            'bin_lower': bins[i],
            'bin_upper': bins[i + 1],
            'n': int(n_i),
            'observed_positive': int(obs_i),
            'expected_positive': float(exp_i),
            'observed_fraction': float(obs_i / n_i) if n_i > 0 else 0,
            'mean_predicted': float(y_prob[mask].mean()) if n_i > 0 else 0,
        })

    df_hl = n_bins - 2  # degrees of freedom
    p_value = 1 - sp_stats.chi2.cdf(chi2, df_hl)
    return chi2, df_hl, p_value, pd.DataFrame(bin_data)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        mask = bin_idx == i
        n_i = mask.sum()
        if n_i == 0:
            continue
        acc_i = y_true[mask].mean()
        conf_i = y_prob[mask].mean()
        ece += (n_i / n_total) * abs(acc_i - conf_i)
    return ece


def main():
    # Load LOOCV predictions
    preds = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
    y_true = preds['true_amyloid'].values
    y_prob = preds['predicted_prob'].values

    print("=" * 60)
    print("CALIBRATION ANALYSIS — GRAD LOOCV PREDICTIONS")
    print("=" * 60)
    print(f"N = {len(preds)}, prevalence = {y_true.mean():.3f}")

    # 1. Calibration curve (sklearn)
    prob_true_10, prob_pred_10 = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    prob_true_5, prob_pred_5 = calibration_curve(y_true, y_prob, n_bins=5, strategy='uniform')

    # 2. Hosmer-Lemeshow test
    hl_chi2, hl_df, hl_p, hl_bins = hosmer_lemeshow(y_true, y_prob, n_bins=10)
    print(f"\nHosmer-Lemeshow: chi2={hl_chi2:.3f}, df={hl_df}, p={hl_p:.4f}")
    if hl_p > 0.05:
        print("  -> No significant miscalibration (p > 0.05)")
    else:
        print("  -> Significant miscalibration detected (p < 0.05)")

    # 3. Calibration-in-the-large
    mean_pred = y_prob.mean()
    mean_obs = y_true.mean()
    cal_large = mean_pred - mean_obs
    print(f"\nCalibration-in-the-large:")
    print(f"  Mean predicted: {mean_pred:.4f}")
    print(f"  Mean observed:  {mean_obs:.4f}")
    print(f"  Difference:     {cal_large:.4f}")

    # 4. ECE
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    # 5. Stratified by stage
    gk_mask = preds['stage'] == 'gatekeeper'
    reflex_mask = preds['stage'] == 'reflex'

    print(f"\nBy Stage:")
    if gk_mask.sum() > 0:
        gk_ece = expected_calibration_error(y_true[gk_mask], y_prob[gk_mask], n_bins=5)
        print(f"  Gatekeeper (n={gk_mask.sum()}): ECE={gk_ece:.4f}")
    if reflex_mask.sum() > 0:
        reflex_ece = expected_calibration_error(y_true[reflex_mask], y_prob[reflex_mask], n_bins=5)
        print(f"  Reflex (n={reflex_mask.sum()}): ECE={reflex_ece:.4f}")

    # ── Save results ──
    hl_bins.to_csv(RESULTS / 'calibration_analysis.csv', index=False)

    metrics = {
        'hosmer_lemeshow': {
            'chi2': float(hl_chi2),
            'df': int(hl_df),
            'p_value': float(hl_p),
        },
        'calibration_in_the_large': {
            'mean_predicted': float(mean_pred),
            'mean_observed': float(mean_obs),
            'difference': float(cal_large),
        },
        'expected_calibration_error': float(ece),
        'ece_by_stage': {
            'gatekeeper': float(gk_ece) if gk_mask.sum() > 0 else None,
            'reflex': float(reflex_ece) if reflex_mask.sum() > 0 else None,
        },
        'n_samples': int(len(preds)),
    }

    with open(RESULTS / 'calibration_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Generate Figure S1 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(prob_pred_10, prob_true_10, 's-', color='#2196F3', markersize=8,
            label=f'GRAD pipeline (ECE={ece:.3f})')
    ax.set_xlabel('Mean predicted probability', fontsize=12)
    ax.set_ylabel('Observed frequency', fontsize=12)
    ax.set_title('A. Calibration Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    # Panel B: Histogram of predicted probabilities
    ax = axes[1]
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, color='#4CAF50',
            label='Amyloid negative', density=True)
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, color='#F44336',
            label='Amyloid positive', density=True)
    ax.axvline(0.25, color='gray', linestyle='--', alpha=0.7, label='Gatekeeper thresholds')
    ax.axvline(0.75, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Predicted probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('B. Prediction Distribution by True Class', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(RESULTS / 'figure_s1_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure_s1_calibration.png")
    print("Done.")


if __name__ == '__main__':
    main()
