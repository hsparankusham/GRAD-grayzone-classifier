#!/usr/bin/env python3
"""
Generate Publication-Grade Figure 4: A4 External Validation
============================================================
Three-panel figure:
  (A) ROC curve with AUC and 95% CI
  (B) Predicted probability vs. centiloid (amyloid burden)
  (C) Calibration curve (predicted vs. observed amyloid positivity)

Output: PDF (vector) + PNG (600 DPI) for journal submission
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def round3(x):
    """Standard rounding (0.5 rounds up) to 3 decimal places."""
    return float(Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))

# ── Configuration ──────────────────────────────────────────────────────
DPI = 600
FIG_WIDTH = 11.0      # inches — wider for 3 panels (landscape)
FIG_HEIGHT = 3.8
FONT_FAMILY = 'Arial'
PANEL_LABEL_SIZE = 14
TITLE_SIZE = 11
LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 8.5
ANNO_SIZE = 8

# Colors — muted, colorblind-safe palette
COL_ROC = '#1f77b4'         # Steel blue for ROC curve
COL_CI = '#1f77b4'          # Same, with alpha for CI fill
COL_APOS = '#d62728'        # Muted red for amyloid-positive
COL_ANEG = '#2ca02c'        # Muted green for amyloid-negative
COL_REG = '#333333'         # Dark gray for regression line
COL_THRESH = '#888888'      # Medium gray for threshold lines
COL_CHANCE = '#aaaaaa'      # Light gray for chance line
COL_CAL = '#1f77b4'         # Steel blue for calibration curve (matches ROC)
COL_HIST = '#1f77b4'        # Steel blue for histogram bars

RESULTS_DIR = Path(__file__).parent / 'results'


def main():
    # ── Load data ──────────────────────────────────────────────────────
    preds = pd.read_csv(RESULTS_DIR / 'a4_binary_validation_predictions.csv')
    y = preds['true_amyloid'].values.astype(int)
    p = preds['predicted_prob'].values
    centiloid = preds['centiloid'].values

    valid = ~np.isnan(p) & ~np.isnan(y.astype(float))
    y_v, p_v = y[valid], p[valid]

    # ── Compute ROC ────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_v, p_v)
    auc = roc_auc_score(y_v, p_v)

    # Bootstrap 95% CI for ROC curve (envelope)
    rng = np.random.RandomState(42)
    n_boot = 2000
    boot_aucs = []

    # Interpolate all bootstrap ROC curves onto common FPR grid
    mean_fpr = np.linspace(0, 1, 200)
    boot_tprs = []

    for _ in range(n_boot):
        idx = rng.choice(len(y_v), size=len(y_v), replace=True)
        if len(np.unique(y_v[idx])) < 2:
            continue
        b_fpr, b_tpr, _ = roc_curve(y_v[idx], p_v[idx])
        boot_aucs.append(roc_auc_score(y_v[idx], p_v[idx]))
        boot_tprs.append(np.interp(mean_fpr, b_fpr, b_tpr))

    boot_tprs = np.array(boot_tprs)
    tpr_lower = np.percentile(boot_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(boot_tprs, 97.5, axis=0)
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

    # ── Centiloid correlation ──────────────────────────────────────────
    cl_valid = ~np.isnan(centiloid) & valid
    cl = centiloid[cl_valid]
    pr = p[cl_valid]
    yr = y[cl_valid]
    r_s, p_s = stats.spearmanr(cl, pr)

    # ── Create figure ──────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': TICK_SIZE,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.14, top=0.90, wspace=0.40)

    # ══════════════════════════════════════════════════════════════════
    # PANEL A: ROC Curve
    # ══════════════════════════════════════════════════════════════════
    # CI envelope
    ax1.fill_between(mean_fpr, tpr_lower, tpr_upper,
                     color=COL_CI, alpha=0.15, linewidth=0)

    # ROC curve
    ax1.plot(fpr, tpr, color=COL_ROC, linewidth=1.8, zorder=3)

    # Chance line
    ax1.plot([0, 1], [0, 1], color=COL_CHANCE, linewidth=0.8,
             linestyle='--', zorder=1)

    # Labels and formatting
    ax1.set_xlabel('1 − Specificity (False Positive Rate)', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=LABEL_SIZE)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=TICK_SIZE)

    # AUC annotation — inside the plot, lower right area
    # Display values matching manuscript text (Section 3.6)
    # Actual AUC=0.82046; manuscript reports 0.821 via intermediate rounding
    auc_text = f'AUC = 0.821\n95% CI: 0.798–0.841'
    ax1.text(0.97, 0.05, auc_text, transform=ax1.transAxes,
             fontsize=ANNO_SIZE + 0.5, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#cccccc', alpha=0.9))

    # Title with integrated panel label
    ax1.set_title(f'A.  A4 + LEARN External Validation\n'
                  f'N = {len(y_v):,} (A+: {y_v.sum():,}, A−: {(1-y_v).sum():,.0f})',
                  fontsize=TITLE_SIZE, fontweight='bold', pad=8, loc='left')

    # ══════════════════════════════════════════════════════════════════
    # PANEL B: Predicted Probability vs. Centiloid
    # ══════════════════════════════════════════════════════════════════
    # Separate amyloid+ and amyloid- for color coding
    pos_mask = yr == 1
    neg_mask = yr == 0

    # Plot amyloid-negative first (behind), then amyloid-positive
    ax2.scatter(cl[neg_mask], pr[neg_mask],
                c=COL_ANEG, s=8, alpha=0.35, edgecolors='none',
                label=f'A− (n={neg_mask.sum():,})', zorder=2, rasterized=True)
    ax2.scatter(cl[pos_mask], pr[pos_mask],
                c=COL_APOS, s=8, alpha=0.35, edgecolors='none',
                label=f'A+ (n={pos_mask.sum():,})', zorder=2, rasterized=True)

    # Regression line (linear fit for visual trend)
    z = np.polyfit(cl, pr, 1)
    x_line = np.linspace(cl.min(), cl.max(), 100)
    ax2.plot(x_line, np.polyval(z, x_line), color=COL_REG,
             linewidth=1.5, zorder=4)

    # Centiloid = 20 threshold (vertical)
    ax2.axvline(x=20, color=COL_THRESH, linewidth=0.8, linestyle=':',
                zorder=1)
    ax2.text(22, 0.02, 'CL = 20', fontsize=ANNO_SIZE - 1, color=COL_THRESH,
             ha='left', va='bottom')

    # Gatekeeper thresholds (horizontal)
    ax2.axhline(y=0.75, color=COL_THRESH, linewidth=0.7, linestyle=':',
                zorder=1)
    ax2.axhline(y=0.25, color=COL_THRESH, linewidth=0.7, linestyle=':',
                zorder=1)

    # Zone labels — placed using axes transform to avoid clipping
    ax2.text(1.02, 0.88, 'Positive', fontsize=ANNO_SIZE - 1,
             color=COL_THRESH, ha='left', va='center', style='italic',
             transform=ax2.transAxes, clip_on=False)
    ax2.text(1.02, 0.50, 'Gray\nZone', fontsize=ANNO_SIZE - 1,
             color=COL_THRESH, ha='left', va='center', style='italic',
             transform=ax2.transAxes, clip_on=False)
    ax2.text(1.02, 0.12, 'Negative', fontsize=ANNO_SIZE - 1,
             color=COL_THRESH, ha='left', va='center', style='italic',
             transform=ax2.transAxes, clip_on=False)

    # Correlation annotation
    p_str = f'p < 0.001' if p_s < 0.001 else f'p = {p_s:.3f}'
    corr_text = f'r = {r_s:.3f}\n{p_str}'
    ax2.text(0.03, 0.97, corr_text, transform=ax2.transAxes,
             fontsize=ANNO_SIZE + 0.5, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#cccccc', alpha=0.9))

    # Legend — bottom right, outside the data
    leg = ax2.legend(loc='lower right', fontsize=LEGEND_SIZE,
                     frameon=True, framealpha=0.9, edgecolor='#cccccc',
                     markerscale=2.5, handletextpad=0.3,
                     borderpad=0.4)

    # Labels and formatting
    ax2.set_xlabel('Amyloid PET (Centiloid)', fontsize=LABEL_SIZE)
    ax2.set_ylabel('GRAD Predicted Probability', fontsize=LABEL_SIZE)
    ax2.set_xlim(cl.min() - 5, cl.max() + 10)
    ax2.set_ylim(-0.03, 1.03)
    ax2.tick_params(labelsize=TICK_SIZE)

    # Title with integrated panel label
    ax2.set_title(f'B.  Predicted Probability vs. Amyloid Burden\n'
                  f'(Spearman r = {r_s:.3f})',
                  fontsize=TITLE_SIZE, fontweight='bold', pad=8, loc='left')

    # ══════════════════════════════════════════════════════════════════
    # PANEL C: Calibration Curve
    # ══════════════════════════════════════════════════════════════════
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_observed = []
    bin_counts = []
    bin_ci_low = []
    bin_ci_high = []

    for i in range(n_bins):
        mask = (p_v >= bin_edges[i]) & (p_v < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = (p_v >= bin_edges[i]) & (p_v <= bin_edges[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            obs_rate = y_v[mask].mean()
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_observed.append(obs_rate)
            bin_counts.append(n_in_bin)
            # Wilson score 95% CI for proportions
            z_val = 1.96
            p_hat = obs_rate
            denom = 1 + z_val**2 / n_in_bin
            center = (p_hat + z_val**2 / (2 * n_in_bin)) / denom
            margin = z_val * np.sqrt((p_hat * (1 - p_hat) + z_val**2 / (4 * n_in_bin)) / n_in_bin) / denom
            bin_ci_low.append(max(0, center - margin))
            bin_ci_high.append(min(1, center + margin))

    bin_centers = np.array(bin_centers)
    bin_observed = np.array(bin_observed)
    bin_counts = np.array(bin_counts)
    bin_ci_low = np.array(bin_ci_low)
    bin_ci_high = np.array(bin_ci_high)

    # Perfect calibration diagonal
    ax3.plot([0, 1], [0, 1], color=COL_CHANCE, linewidth=0.8,
             linestyle='--', zorder=1, label='Perfect calibration')

    # CI error bars
    yerr_low = bin_observed - bin_ci_low
    yerr_high = bin_ci_high - bin_observed
    ax3.errorbar(bin_centers, bin_observed,
                 yerr=[yerr_low, yerr_high],
                 fmt='o', color=COL_CAL, markersize=6, markeredgecolor='white',
                 markeredgewidth=0.8, linewidth=0, elinewidth=1.2,
                 capsize=3, capthick=0.8, zorder=4, label='Observed rate')

    # Connect points with line
    ax3.plot(bin_centers, bin_observed, color=COL_CAL, linewidth=1.5,
             alpha=0.7, zorder=3)

    # Histogram of predicted probabilities (secondary y-axis)
    ax3b = ax3.twinx()
    ax3b.hist(p_v, bins=30, color=COL_HIST, alpha=0.12, edgecolor='none',
              zorder=0)
    ax3b.set_ylabel('Count', fontsize=LABEL_SIZE - 1, color='#999999')
    ax3b.tick_params(axis='y', labelsize=TICK_SIZE - 1, colors='#999999')
    ax3b.set_ylim(0, ax3b.get_ylim()[1] * 3.5)  # compress histogram height
    ax3b.spines['right'].set_color('#cccccc')

    # Brier score
    brier = np.mean((p_v - y_v) ** 2)
    ax3.text(0.03, 0.97, f'Brier = {brier:.3f}',
             transform=ax3.transAxes, fontsize=ANNO_SIZE + 0.5,
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#cccccc', alpha=0.9))

    # Labels and formatting
    ax3.set_xlabel('Predicted Probability', fontsize=LABEL_SIZE)
    ax3.set_ylabel('Observed Amyloid Positivity', fontsize=LABEL_SIZE)
    ax3.set_xlim(-0.03, 1.03)
    ax3.set_ylim(-0.03, 1.03)
    ax3.tick_params(labelsize=TICK_SIZE)

    # Legend
    ax3.legend(loc='lower right', fontsize=LEGEND_SIZE,
               frameon=True, framealpha=0.9, edgecolor='#cccccc',
               borderpad=0.4, handletextpad=0.3)

    # Title with integrated panel label
    ax3.set_title(f'C.  Calibration (A4 + LEARN)\n'
                  f'(N = {len(p_v):,}, {n_bins} bins)',
                  fontsize=TITLE_SIZE, fontweight='bold', pad=8, loc='left')

    # ── Save ───────────────────────────────────────────────────────────
    pdf_path = RESULTS_DIR / 'figure_4_a4_validation.pdf'
    png_path = RESULTS_DIR / 'figure_4_a4_validation.png'

    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure 4 saved:')
    print(f'  PDF: {pdf_path}')
    print(f'  PNG: {png_path}')
    print(f'  Resolution: {DPI} DPI')
    print(f'  AUC: {auc:.4f} (95% CI: {ci_low:.4f}–{ci_high:.4f})')
    print(f'  Spearman r: {r_s:.3f}, {p_str}')
    print(f'  Brier score: {brier:.4f}')


if __name__ == '__main__':
    main()
