"""
Generate Figure 2: Combined ROC curves from actual LOOCV predictions.
(A) Overall pipeline
(B) Gatekeeper-resolved cases
(C) Reflex model for gray zone
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def create_figure_2_combined():
    """Create combined 3-panel ROC figure from actual LOOCV predictions."""

    pred_path = os.path.join(OUTPUT_DIR, 'adni_loocv_predictions.csv')
    df = pd.read_csv(pred_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel A: Overall Pipeline ---
    ax1 = axes[0]
    fpr_all, tpr_all, _ = roc_curve(df['true_amyloid'], df['predicted_prob'])
    auc_all = auc(fpr_all, tpr_all)

    ax1.fill_between(fpr_all, 0, tpr_all, alpha=0.15, color='#1976D2')
    ax1.plot(fpr_all, tpr_all, 'b-', linewidth=2.5, label=f'AUC = {auc_all:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=10)
    ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10)
    ax1.set_title('A. Overall Pipeline\n(N=320)', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # --- Panel B: Gatekeeper-Resolved Cases ---
    ax2 = axes[1]
    gk = df[df['stage'] == 'gatekeeper']
    fpr_gk, tpr_gk, _ = roc_curve(gk['true_amyloid'], gk['predicted_prob'])
    auc_gk = auc(fpr_gk, tpr_gk)

    ax2.fill_between(fpr_gk, 0, tpr_gk, alpha=0.15, color='#388E3C')
    ax2.plot(fpr_gk, tpr_gk, 'g-', linewidth=2.5, label=f'AUC = {auc_gk:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=10)
    ax2.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10)
    n_gk = len(gk)
    pct_gk = n_gk / len(df) * 100
    ax2.set_title(f'B. Gatekeeper-Resolved Cases\n(N={n_gk}, {pct_gk:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    gk_acc = (gk['predicted_class'] == gk['true_amyloid']).mean()
    gk_neg = gk[gk['predicted_class'] == 0]
    gk_pos = gk[gk['predicted_class'] == 1]
    npv = (gk_neg['true_amyloid'] == 0).mean() if len(gk_neg) > 0 else float('nan')
    ppv = (gk_pos['true_amyloid'] == 1).mean() if len(gk_pos) > 0 else float('nan')
    ax2.text(0.55, 0.25, f'Accuracy: {gk_acc:.1%}\nNPV: {npv:.1%} | PPV: {ppv:.1%}',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Panel C: Reflex Model - Gray Zone ---
    ax3 = axes[2]
    gz = df[df['stage'] == 'reflex']
    fpr_gz, tpr_gz, _ = roc_curve(gz['true_amyloid'], gz['predicted_prob'])
    auc_gz = auc(fpr_gz, tpr_gz)

    ax3.fill_between(fpr_gz, 0, tpr_gz, alpha=0.15, color='#7B1FA2')
    ax3.plot(fpr_gz, tpr_gz, color='#7B1FA2', linewidth=2.5,
             label=f'AUC = {auc_gz:.3f}')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=10)
    ax3.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10)
    n_gz = len(gz)
    pct_gz = n_gz / len(df) * 100
    ax3.set_title(f'C. Reflex Model (Gray Zone)\n(N={n_gz}, {pct_gz:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)

    gz_acc = (gz['predicted_class'] == gz['true_amyloid']).mean()
    ax3.text(0.55, 0.25, f'Accuracy: {gz_acc:.1%}', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'figure_2_roc_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_figure_2_combined()
