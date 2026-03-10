#!/usr/bin/env python3
"""
Generate Publication-Grade Figure S3: Subgroup & Threshold Analysis
===================================================================
Four-panel figure:
  (A) Subgroup AUC forest plot (by cognitive status, APOE4, sex, age)
  (B) Sensitivity/Specificity tradeoff across probability thresholds
  (C) Gatekeeper threshold sensitivity (resolution vs accuracy)
  (D) Per-stage confusion matrices (Gatekeeper vs Reflex)

Output: PDF (vector) + PNG (600 DPI) for journal submission
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, confusion_matrix
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
DPI = 600
FIG_WIDTH = 7.5
FIG_HEIGHT = 8.5
FONT_FAMILY = 'Arial'
PANEL_LABEL_SIZE = 14
TITLE_SIZE = 10.5
LABEL_SIZE = 9.5
TICK_SIZE = 8.5
ANNO_SIZE = 8
LEGEND_SIZE = 7.5

# Colors
COL_PRIMARY = '#1565C0'
COL_SECONDARY = '#1E88E5'
COL_ACCENT = '#42A5F5'
COL_RED = '#E53935'
COL_GREEN = '#2E7D32'
COL_ORANGE = '#F57C00'
COL_PURPLE = '#7B1FA2'
COL_NEUTRAL = '#616161'
COL_LIGHT = '#BDBDBD'

RESULTS_DIR = Path(__file__).parent / 'results'


def main():
    # ── Load data ──────────────────────────────────────────────────────
    df = pd.read_csv(RESULTS_DIR / 'adni_loocv_predictions.csv')
    subgroup = pd.read_csv(RESULTS_DIR / 'supp_table_subgroup_performance.csv')
    threshold = pd.read_csv(RESULTS_DIR / 'supp_table_threshold_sensitivity.csv')

    y_true = df['true_amyloid'].values.astype(int)
    y_prob = df['predicted_prob'].values

    # ── Setup figure ──────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': TICK_SIZE,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
    })

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38,
                          left=0.10, right=0.96, bottom=0.06, top=0.95)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ══════════════════════════════════════════════════════════════════
    # PANEL A: Subgroup AUC Forest Plot
    # ══════════════════════════════════════════════════════════════════
    # Organize subgroups (exclude "Overall" — show as reference line)
    overall_auc = subgroup.loc[subgroup['subgroup'] == 'Overall', 'auc'].values[0]
    sub = subgroup[subgroup['subgroup'] != 'Overall'].copy()

    # Define display order (bottom to top in forest plot)
    order = [
        ('Age Tertile', 'Old tertile'),
        ('Age Tertile', 'Middle tertile'),
        ('Age Tertile', 'Young tertile'),
        ('Sex', 'Male'),
        ('Sex', 'Female'),
        ('APOE4 Status', 'APOE4 non-carrier'),
        ('APOE4 Status', 'APOE4 carrier'),
        ('Cognitive Status', 'Dementia'),
        ('Cognitive Status', 'MCI'),
        ('Cognitive Status', 'CN'),
    ]

    labels = []
    aucs = []
    ns = []
    colors = []
    category_colors = {
        'Cognitive Status': COL_PRIMARY,
        'APOE4 Status': COL_GREEN,
        'Sex': COL_ORANGE,
        'Age Tertile': COL_PURPLE,
    }

    for cat, name in order:
        row = sub[(sub['category'] == cat)].copy()
        for _, r in row.iterrows():
            # Exact match first, then substring for age tertiles
            subg = r['subgroup'].strip()
            matched = (subg.lower() == name.lower()) or \
                      (cat == 'Age Tertile' and name.lower() in subg.lower())
            if matched:
                clean = subg
                # Shorten age tertile labels
                if 'tertile' in clean.lower():
                    if 'young' in clean.lower():
                        clean = 'Age \u226470'
                    elif 'middle' in clean.lower():
                        clean = 'Age 71\u201375'
                    elif 'old' in clean.lower():
                        clean = 'Age >75'
                labels.append(f'{clean} (n={int(r["n"])})')
                aucs.append(r['auc'])
                ns.append(int(r['n']))
                colors.append(category_colors[cat])
                break

    y_pos = np.arange(len(labels))

    # Overall reference line
    ax_a.axvline(overall_auc, color=COL_LIGHT, linewidth=1.5,
                 linestyle='--', zorder=1)
    ax_a.text(overall_auc + 0.003, len(labels) - 0.3,
              f'Overall\n{overall_auc:.3f}',
              fontsize=ANNO_SIZE - 1, color=COL_NEUTRAL,
              ha='left', va='top', style='italic')

    # Plot points
    for i, (auc_val, col) in enumerate(zip(aucs, colors)):
        ax_a.plot(auc_val, i, 'D', color=col, markersize=7,
                  markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        # Horizontal whisker (visual only — no CI data, show as thin line to reference)
        ax_a.plot([auc_val, auc_val], [i, i], color=col,
                  linewidth=0, zorder=2)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels, fontsize=TICK_SIZE - 0.5)
    ax_a.set_xlabel('AUC', fontsize=LABEL_SIZE)
    ax_a.set_xlim(0.72, 0.95)
    ax_a.set_ylim(-0.5, len(labels) - 0.5)
    ax_a.set_title('Subgroup Performance',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=6)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.spines['left'].set_visible(False)
    ax_a.tick_params(axis='y', length=0)
    ax_a.xaxis.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax_a.set_axisbelow(True)

    # Category separators
    separators = [2.5, 4.5, 6.5]  # Between Age/Sex, Sex/APOE4, APOE4/Cognitive
    for sep in separators:
        ax_a.axhline(sep, color='#E0E0E0', linewidth=0.6, linestyle='-')

    # Category legend
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_PRIMARY,
               markersize=6, label='Cognitive Status'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_GREEN,
               markersize=6, label='APOE4'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_ORANGE,
               markersize=6, label='Sex'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_PURPLE,
               markersize=6, label='Age'),
    ]
    ax_a.legend(handles=legend_elements, loc='lower left', fontsize=LEGEND_SIZE,
                frameon=True, framealpha=0.9, edgecolor='#cccccc',
                handletextpad=0.3)

    ax_a.text(-0.16, 1.08, 'A', transform=ax_a.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # PANEL B: Threshold-Performance Tradeoff
    # ══════════════════════════════════════════════════════════════════
    thresholds = np.linspace(0.05, 0.95, 200)
    sens_arr = []
    spec_arr = []
    ppv_arr = []
    npv_arr = []
    acc_arr = []

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens_arr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_arr.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv_arr.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        npv_arr.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        acc_arr.append((tp + tn) / (tp + tn + fp + fn))

    ax_b.plot(thresholds, sens_arr, color=COL_RED, linewidth=1.8,
              label='Sensitivity', zorder=3)
    ax_b.plot(thresholds, spec_arr, color=COL_PRIMARY, linewidth=1.8,
              label='Specificity', zorder=3)
    ax_b.plot(thresholds, ppv_arr, color=COL_GREEN, linewidth=1.4,
              linestyle='--', label='PPV', zorder=2)
    ax_b.plot(thresholds, npv_arr, color=COL_ORANGE, linewidth=1.4,
              linestyle='--', label='NPV', zorder=2)
    ax_b.plot(thresholds, acc_arr, color=COL_NEUTRAL, linewidth=1.2,
              linestyle=':', label='Accuracy', alpha=0.7, zorder=2)

    # Mark the 0.50 default threshold
    ax_b.axvline(0.50, color='#999999', linewidth=0.8, linestyle=':',
                 zorder=1, alpha=0.6)
    ax_b.text(0.52, 0.03, 'p = 0.50', fontsize=ANNO_SIZE - 1.5,
              color='#999999', rotation=90, va='bottom')

    # Mark 90% sensitivity threshold
    ax_b.axvline(0.245, color=COL_RED, linewidth=0.7, linestyle=':',
                 zorder=1, alpha=0.5)
    ax_b.text(0.20, 0.40, '90% Sens\np = 0.245',
              fontsize=ANNO_SIZE - 1.5, color=COL_RED, ha='center',
              va='center', style='italic',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=0.8))

    # Mark 90% specificity threshold
    ax_b.axvline(0.729, color=COL_PRIMARY, linewidth=0.7, linestyle=':',
                 zorder=1, alpha=0.5)
    ax_b.text(0.78, 0.40, '90% Spec\np = 0.729',
              fontsize=ANNO_SIZE - 1.5, color=COL_PRIMARY, ha='center',
              va='center', style='italic',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=0.8))

    ax_b.set_xlabel('Classification Threshold', fontsize=LABEL_SIZE)
    ax_b.set_ylabel('Metric Value', fontsize=LABEL_SIZE)
    ax_b.set_xlim(0.05, 0.95)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_title('Threshold-Performance\nTradeoff',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=6)
    ax_b.legend(loc='center left', fontsize=LEGEND_SIZE,
                frameon=True, framealpha=0.9, edgecolor='#cccccc',
                bbox_to_anchor=(0.0, 0.65))
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(labelsize=TICK_SIZE)

    ax_b.text(-0.16, 1.08, 'B', transform=ax_b.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # PANEL C: Gatekeeper Threshold Sensitivity
    # ══════════════════════════════════════════════════════════════════
    # Reshape threshold data for heatmap
    low_thresholds = sorted(threshold['Low_Threshold'].unique())
    high_thresholds = sorted(threshold['High_Threshold'].unique())

    # Create resolution rate matrix
    res_matrix = np.zeros((len(low_thresholds), len(high_thresholds)))
    acc_matrix = np.zeros((len(low_thresholds), len(high_thresholds)))
    for i, lt in enumerate(low_thresholds):
        for j, ht in enumerate(high_thresholds):
            row = threshold[(threshold['Low_Threshold'] == lt) &
                            (threshold['High_Threshold'] == ht)]
            if len(row) > 0:
                res_matrix[i, j] = row['Resolution_Rate'].values[0] * 100
                acc_matrix[i, j] = row['Resolved_Accuracy'].values[0] * 100

    # Custom colormap (white to blue)
    cmap_res = mcolors.LinearSegmentedColormap.from_list(
        'res_cmap', ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1'])

    im = ax_c.imshow(res_matrix, cmap=cmap_res, aspect='auto',
                     origin='lower', vmin=25, vmax=85)

    # Annotate cells with both resolution rate and accuracy
    for i in range(len(low_thresholds)):
        for j in range(len(high_thresholds)):
            rate = res_matrix[i, j]
            acc = acc_matrix[i, j]
            text_color = 'white' if rate > 60 else COL_NEUTRAL
            # Highlight the actual thresholds used (0.25, 0.75)
            if low_thresholds[i] == 0.25 and high_thresholds[j] == 0.75:
                ax_c.text(j, i, f'{rate:.0f}%\n({acc:.0f}%)',
                          ha='center', va='center', fontsize=ANNO_SIZE - 1.5,
                          fontweight='bold', color='yellow')
                # Gold border
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='#FFD600',
                                     linewidth=2.5)
                ax_c.add_patch(rect)
            else:
                ax_c.text(j, i, f'{rate:.0f}%\n({acc:.0f}%)',
                          ha='center', va='center', fontsize=ANNO_SIZE - 2,
                          color=text_color)

    ax_c.set_xticks(range(len(high_thresholds)))
    ax_c.set_xticklabels([f'{h:.2f}' for h in high_thresholds],
                          fontsize=TICK_SIZE - 1)
    ax_c.set_yticks(range(len(low_thresholds)))
    ax_c.set_yticklabels([f'{l:.2f}' for l in low_thresholds],
                          fontsize=TICK_SIZE - 1)
    ax_c.set_xlabel('Upper Threshold (Rule-In)', fontsize=LABEL_SIZE)
    ax_c.set_ylabel('Lower Threshold (Rule-Out)', fontsize=LABEL_SIZE)
    ax_c.set_title('Gatekeeper Threshold\nSensitivity',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=6)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.8, pad=0.04)
    cbar.set_label('Resolution Rate (%)', fontsize=ANNO_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1)

    # Annotation: yellow box = current thresholds
    ax_c.text(0.02, 0.02, 'Gold box = current (0.25/0.75)\n(accuracy in parentheses)',
              transform=ax_c.transAxes, fontsize=ANNO_SIZE - 2,
              va='bottom', ha='left', color=COL_NEUTRAL, style='italic')

    ax_c.text(-0.22, 1.08, 'C', transform=ax_c.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # PANEL D: Per-Stage Confusion Matrices (side by side)
    # ══════════════════════════════════════════════════════════════════
    gk = df[df['stage'] == 'gatekeeper']
    rf = df[df['stage'] == 'reflex']
    cm_gk = confusion_matrix(gk['true_amyloid'], gk['predicted_class'])
    cm_rf = confusion_matrix(rf['true_amyloid'], rf['predicted_class'])

    # Create a combined visualization
    # Left half: Gatekeeper, Right half: Reflex
    # Use a 2x4 grid within the subplot
    ax_d.axis('off')

    # Gatekeeper mini-matrix (left)
    ax_gk = fig.add_axes([
        ax_d.get_position().x0 + 0.005,
        ax_d.get_position().y0 + 0.01,
        (ax_d.get_position().width - 0.03) / 2,
        ax_d.get_position().height - 0.06
    ])
    ax_rf = fig.add_axes([
        ax_d.get_position().x0 + ax_d.get_position().width / 2 + 0.025,
        ax_d.get_position().y0 + 0.01,
        (ax_d.get_position().width - 0.03) / 2,
        ax_d.get_position().height - 0.06
    ])

    def draw_cm(ax, cm, title, color_base):
        cm_pct = cm / cm.sum() * 100
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'cm', ['#FFFFFF', color_base])
        im = ax.imshow(cm_pct, cmap=cmap, aspect='equal',
                       vmin=0, vmax=cm_pct.max() * 1.15)
        labels_cm = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                tc = 'white' if cm_pct[i, j] > cm_pct.max() * 0.4 else COL_NEUTRAL
                ax.text(j, i - 0.10, labels_cm[i][j],
                        ha='center', va='center', fontsize=ANNO_SIZE - 1.5,
                        fontweight='bold', color=tc, alpha=0.6)
                ax.text(j, i + 0.10, f'{cm[i,j]}',
                        ha='center', va='center', fontsize=ANNO_SIZE + 0.5,
                        fontweight='bold', color=tc)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['A\u2212', 'A+'], fontsize=TICK_SIZE - 1)
        ax.set_yticklabels(['A\u2212', 'A+'], fontsize=TICK_SIZE - 1)
        ax.set_title(title, fontsize=ANNO_SIZE + 0.5, fontweight='bold', pad=4)

        # Thin white borders
        for i in range(2):
            for j in range(2):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='white', linewidth=1.5)
                ax.add_patch(rect)

        # Accuracy below
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
        ax.text(0.5, -0.22, f'Acc: {acc:.1f}%  N={cm.sum()}',
                ha='center', va='top', fontsize=ANNO_SIZE - 0.5,
                transform=ax.transAxes, color=COL_NEUTRAL, fontweight='bold')

    draw_cm(ax_gk, cm_gk, 'Stage 1: Gatekeeper', '#1565C0')
    draw_cm(ax_rf, cm_rf, 'Stage 2: Reflex', '#7B1FA2')

    # Panel label
    ax_d.text(-0.04, 1.08, 'D', transform=ax_d.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ── Save ───────────────────────────────────────────────────────────
    pdf_path = RESULTS_DIR / 'figure_s3_subgroup_threshold.pdf'
    png_path = RESULTS_DIR / 'figure_s3_subgroup_threshold.png'

    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure S3 saved:')
    print(f'  PDF: {pdf_path}')
    print(f'  PNG: {png_path}')
    print(f'  Resolution: {DPI} DPI')
    print(f'  Panels: (A) Subgroup forest plot, (B) Threshold tradeoff,')
    print(f'           (C) Gatekeeper sensitivity, (D) Per-stage confusion matrices')


if __name__ == '__main__':
    main()
