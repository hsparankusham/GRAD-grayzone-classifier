#!/usr/bin/env python3
"""
Generate Publication-Grade Figure 3: Comprehensive GRAD Characterization
=========================================================================
Eight-panel figure in 2×4 landscape layout with section headers:

  Row 1:  I. MODEL PERFORMANCE        | II. VALIDATION & STABILITY
          (A) Feature importance       | (C) Calibration curve
          (B) Classification matrix    | (D) Bootstrap AUC

  Row 2:  III. ROBUSTNESS & CLINICAL   | IV. STAGE-LEVEL ANALYSIS
          (E) Subgroup forest plot     | (G) Per-stage confusion matrices
          (F) Threshold tradeoff       | (H) Threshold sensitivity heatmap

Output: PDF (vector) + PNG (600 DPI) for journal submission
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
DPI = 600
FIG_WIDTH = 15
FIG_HEIGHT = 8.0
FONT_FAMILY = 'Arial'
PANEL_LABEL_SIZE = 13
TITLE_SIZE = 9
LABEL_SIZE = 8
TICK_SIZE = 7
ANNO_SIZE = 6.5
LEGEND_SIZE = 6
SECTION_SIZE = 9

# Color palette — unified, colorblind-safe
COL_DEEP = '#0D47A1'
COL_PRIMARY = '#1565C0'
COL_MED = '#1976D2'
COL_LIGHT_BLUE = '#1E88E5'
COL_ACCENT = '#42A5F5'
COL_PALE = '#90CAF9'
COL_RED = '#E53935'
COL_GREEN = '#2E7D32'
COL_ORANGE = '#F57C00'
COL_PURPLE = '#7B1FA2'
COL_NEUTRAL = '#616161'
COL_LIGHT = '#BDBDBD'

FEAT_COLORS = [COL_DEEP, COL_PRIMARY, COL_MED, COL_LIGHT_BLUE, COL_ACCENT, COL_PALE]

RESULTS_DIR = Path(__file__).parent / 'results'


def draw_section_header(fig, x0, x1, y, text):
    """Draw a section header bar spanning from x0 to x1 at y (figure coords)."""
    width = x1 - x0
    height = 0.026
    rect = mpatches.FancyBboxPatch(
        (x0, y - height / 2), width, height,
        boxstyle="round,pad=0.003", facecolor='#E3F2FD', edgecolor='#90CAF9',
        linewidth=0.6, transform=fig.transFigure, clip_on=False
    )
    fig.add_artist(rect)
    fig.text((x0 + x1) / 2, y, text, ha='center', va='center',
             fontsize=SECTION_SIZE, fontweight='bold', color=COL_PRIMARY,
             fontstyle='italic')


def draw_cm_panel(ax, cm, title, color_base):
    """Draw a confusion matrix heatmap on a given axis."""
    cm_pct = cm.astype(float) / cm.sum() * 100
    cmap = mcolors.LinearSegmentedColormap.from_list('cm', ['#FFFFFF', color_base])
    ax.imshow(cm_pct, cmap=cmap, aspect='equal', vmin=0, vmax=cm_pct.max() * 1.15)
    labs = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            tc = 'white' if cm_pct[i, j] > cm_pct.max() * 0.40 else COL_NEUTRAL
            ax.text(j, i - 0.12, labs[i][j], ha='center', va='center',
                    fontsize=ANNO_SIZE - 1, fontweight='bold', color=tc, alpha=0.6)
            ax.text(j, i + 0.08, f'n={cm[i,j]}', ha='center', va='center',
                    fontsize=ANNO_SIZE + 0.5, fontweight='bold', color=tc)
            ax.text(j, i + 0.30, f'({cm_pct[i,j]:.1f}%)', ha='center', va='center',
                    fontsize=ANNO_SIZE - 1, color=tc)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred A\u2212', 'Pred A+'], fontsize=TICK_SIZE)
    ax.set_yticklabels(['True A\u2212', 'True A+'], fontsize=TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE - 0.5, fontweight='bold', pad=4)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
    ax.text(0.5, -0.18, f'Accuracy: {acc:.1f}%  (N={cm.sum()})',
            ha='center', va='top', fontsize=ANNO_SIZE,
            transform=ax.transAxes, color=COL_NEUTRAL, fontweight='bold')


def main():
    # ── Load data ──────────────────────────────────────────────────────
    df = pd.read_csv(RESULTS_DIR / 'adni_loocv_predictions.csv')
    subgroup = pd.read_csv(RESULTS_DIR / 'supp_table_subgroup_performance.csv')
    threshold = pd.read_csv(RESULTS_DIR / 'supp_table_threshold_sensitivity.csv')

    y_true = df['true_amyloid'].values.astype(int)
    y_pred = df['predicted_class'].values.astype(int)
    y_prob = df['predicted_prob'].values

    # ── Setup figure ──────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': TICK_SIZE,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # ── 2×4 landscape grid ────────────────────────────────────────────
    gs = fig.add_gridspec(
        2, 4,
        left=0.055, right=0.965, bottom=0.08, top=0.88,
        hspace=0.58, wspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2])   # placeholder — will hold two sub-CMs
    ax_h = fig.add_subplot(gs[1, 3])

    # ── Section header bars ────────────────────────────────────────────
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    pos_e = ax_e.get_position()
    pos_f = ax_f.get_position()
    pos_g = ax_g.get_position()
    pos_h = ax_h.get_position()

    hdr_y_top = pos_a.y1 + 0.030
    hdr_y_bot = pos_e.y1 + 0.030

    draw_section_header(fig, pos_a.x0 - 0.008, pos_b.x1 + 0.008, hdr_y_top,
                        'I. Model Performance')
    draw_section_header(fig, pos_c.x0 - 0.008, pos_d.x1 + 0.008, hdr_y_top,
                        'II. Validation & Stability')
    draw_section_header(fig, pos_e.x0 - 0.008, pos_f.x1 + 0.008, hdr_y_bot,
                        'III. Robustness & Clinical Utility')
    draw_section_header(fig, pos_g.x0 - 0.008, pos_h.x1 + 0.008, hdr_y_bot,
                        'IV. Stage-Level Analysis')

    # ══════════════════════════════════════════════════════════════════
    # (A) Feature Importance
    # ══════════════════════════════════════════════════════════════════
    features = {
        'p-tau217':               0.3313,
        'Age':                    0.1672,
        'GFAP \u00d7 p-tau217':   0.1606,
        'Tau/A\u03b242 ratio':    0.1579,
        'APOE\u03b54 carrier':    0.1049,
        'GFAP':                   0.0781,
    }
    names = list(features.keys())
    importances = list(features.values())
    y_pos = np.arange(len(names))

    bars = ax_a.barh(y_pos, importances, color=FEAT_COLORS,
                     edgecolor='white', linewidth=0.4, height=0.58, zorder=3)
    for bar, imp in zip(bars, importances):
        w = bar.get_width()
        ax_a.text(w + 0.006, bar.get_y() + bar.get_height() / 2,
                  f'{imp:.1%}', ha='left', va='center',
                  fontsize=ANNO_SIZE, fontweight='bold', color=COL_NEUTRAL)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names, fontsize=LABEL_SIZE - 0.5)
    ax_a.invert_yaxis()
    ax_a.set_xlabel('Mean Decrease in Gini Impurity', fontsize=LABEL_SIZE)
    ax_a.set_xlim(0, max(importances) * 1.30)
    ax_a.set_title('Feature Importance\n(Reflex Random Forest)',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.xaxis.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.text(-0.18, 1.08, 'A', transform=ax_a.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (B) Overall Confusion Matrix
    # ══════════════════════════════════════════════════════════════════
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum()
    cmap_b = mcolors.LinearSegmentedColormap.from_list(
        'blue_cm', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1'])
    ax_b.imshow(cm_norm, cmap=cmap_b, aspect='equal', vmin=0, vmax=cm_norm.max() * 1.1)

    labs = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            tc = 'white' if cm_norm[i, j] > cm_norm.max() * 0.42 else COL_NEUTRAL
            ax_b.text(j, i - 0.12, labs[i][j], ha='center', va='center',
                      fontsize=ANNO_SIZE - 0.5, fontweight='bold', color=tc, alpha=0.65)
            ax_b.text(j, i + 0.08, f'n = {cm[i, j]}', ha='center', va='center',
                      fontsize=ANNO_SIZE + 1, fontweight='bold', color=tc)
            ax_b.text(j, i + 0.32, f'({cm_norm[i, j] * 100:.1f}%)', ha='center', va='center',
                      fontsize=ANNO_SIZE - 0.5, color=tc)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor='white', linewidth=2)
            ax_b.add_patch(rect)
    ax_b.set_xticks([0, 1])
    ax_b.set_yticks([0, 1])
    ax_b.set_xticklabels(['Predicted A\u2212', 'Predicted A+'], fontsize=TICK_SIZE)
    ax_b.set_yticklabels(['True A\u2212', 'True A+'], fontsize=TICK_SIZE)
    ax_b.set_title('Classification Matrix\n(N = 320, LOOCV)',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) * 100
    spec = tn / (tn + fp) * 100
    acc = (tp + tn) / cm.sum() * 100
    ax_b.text(0.5, -0.15, f'Acc {acc:.1f}%  |  Sens {sens:.1f}%  |  Spec {spec:.1f}%',
              ha='center', va='top', fontsize=ANNO_SIZE, fontweight='bold',
              transform=ax_b.transAxes, color=COL_NEUTRAL)
    ax_b.text(-0.15, 1.08, 'B', transform=ax_b.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (C) Calibration Curve
    # ══════════════════════════════════════════════════════════════════
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10,
                                             strategy='uniform')
    ax_c.plot([0, 1], [0, 1], color=COL_LIGHT, linewidth=1, linestyle='--', zorder=1,
              label='Perfect calibration')
    ax_c.plot(prob_pred, prob_true, color=COL_PRIMARY, linewidth=1.8,
              marker='s', markersize=4.5, markeredgecolor='white',
              markeredgewidth=0.8, zorder=3, label='GRAD pipeline')

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_cal = []
    for _ in range(500):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            bt, bp = calibration_curve(y_true[idx], y_prob[idx], n_bins=10,
                                       strategy='uniform')
            if len(bt) == len(prob_true):
                boot_cal.append(bt)
        except Exception:
            pass
    if len(boot_cal) > 50:
        boot_arr = np.array(boot_cal)
        ax_c.fill_between(prob_pred,
                          np.percentile(boot_arr, 5, axis=0),
                          np.percentile(boot_arr, 95, axis=0),
                          color=COL_PRIMARY, alpha=0.10, linewidth=0, zorder=2)

    # Mini prediction histogram
    ax_ch = ax_c.inset_axes([0, 0, 1, 0.15])
    bins_h = np.linspace(0, 1, 21)
    ax_ch.hist(y_prob[y_true == 0], bins=bins_h, alpha=0.50, color=COL_ACCENT,
               edgecolor='none', density=True)
    ax_ch.hist(y_prob[y_true == 1], bins=bins_h, alpha=0.50, color=COL_RED,
               edgecolor='none', density=True)
    ax_ch.set_xlim(0, 1)
    ax_ch.set_yticks([])
    ax_ch.set_xticks([])
    ax_ch.patch.set_alpha(0)
    for s in ax_ch.spines.values():
        s.set_visible(False)

    ax_c.text(0.05, 0.92, 'ECE = 0.078', transform=ax_c.transAxes,
              fontsize=ANNO_SIZE + 0.5, ha='left', va='top', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                        edgecolor='#cccccc', alpha=0.9))
    ax_c.set_xlabel('Mean Predicted Probability', fontsize=LABEL_SIZE)
    ax_c.set_ylabel('Observed Frequency', fontsize=LABEL_SIZE)
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_title('Calibration Curve\n(10-bin, LOOCV)',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax_c.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=True,
                framealpha=0.9, edgecolor='#cccccc')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.text(-0.15, 1.08, 'C', transform=ax_c.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (D) Bootstrap AUC Distribution
    # ══════════════════════════════════════════════════════════════════
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    auc_mean = boot_aucs.mean()

    _, bin_edges, patches = ax_d.hist(boot_aucs, bins=35, edgecolor='white',
                                      linewidth=0.3, zorder=3, alpha=0.9)
    for patch, left in zip(patches, bin_edges[:-1]):
        mid = left + (bin_edges[1] - bin_edges[0]) / 2
        if ci_lo <= mid <= ci_hi:
            patch.set_facecolor(COL_PRIMARY)
        else:
            patch.set_facecolor(COL_ACCENT)
            patch.set_alpha(0.45)

    ymax = ax_d.get_ylim()[1]
    ax_d.axvline(auc_mean, color=COL_RED, linewidth=1.6, zorder=4)
    ax_d.axvline(ci_lo, color=COL_NEUTRAL, linewidth=1, linestyle='--', zorder=4)
    ax_d.axvline(ci_hi, color=COL_NEUTRAL, linewidth=1, linestyle='--', zorder=4)
    ax_d.axvspan(ci_lo, ci_hi, alpha=0.05, color=COL_PRIMARY, zorder=1)

    ax_d.text(auc_mean, ymax * 0.95, f'AUC = {auc_mean:.3f}',
              ha='center', va='top', fontsize=ANNO_SIZE, fontweight='bold',
              color=COL_RED, bbox=dict(boxstyle='round,pad=0.15',
              facecolor='white', edgecolor='none', alpha=0.85))
    ax_d.text(ci_lo, ymax * 0.78, f'{ci_lo:.3f}', ha='center', va='top',
              fontsize=ANNO_SIZE - 0.5, color=COL_NEUTRAL, fontweight='bold')
    ax_d.text(ci_hi, ymax * 0.78, f'{ci_hi:.3f}', ha='center', va='top',
              fontsize=ANNO_SIZE - 0.5, color=COL_NEUTRAL, fontweight='bold')
    ax_d.annotate('', xy=(ci_lo, ymax * 0.85), xytext=(ci_hi, ymax * 0.85),
                  arrowprops=dict(arrowstyle='<->', color=COL_NEUTRAL, lw=1))
    ax_d.text((ci_lo + ci_hi) / 2, ymax * 0.88, '95% CI', ha='center', va='bottom',
              fontsize=ANNO_SIZE - 0.5, color=COL_NEUTRAL, fontweight='bold')

    ax_d.set_xlabel('AUC (Bootstrap Resamples)', fontsize=LABEL_SIZE)
    ax_d.set_ylabel('Frequency', fontsize=LABEL_SIZE)
    ax_d.set_title('Bootstrap Stability\n(2,000 resamples)',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.text(-0.15, 1.08, 'D', transform=ax_d.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (E) Subgroup AUC Forest Plot
    # ══════════════════════════════════════════════════════════════════
    overall_auc = subgroup.loc[subgroup['subgroup'] == 'Overall', 'auc'].values[0]
    sub = subgroup[subgroup['subgroup'] != 'Overall'].copy()

    order = [
        ('Age Tertile', 'Old tertile'), ('Age Tertile', 'Middle tertile'),
        ('Age Tertile', 'Young tertile'),
        ('Sex', 'Male'), ('Sex', 'Female'),
        ('APOE4 Status', 'APOE4 non-carrier'), ('APOE4 Status', 'APOE4 carrier'),
        ('Cognitive Status', 'Dementia'), ('Cognitive Status', 'MCI'),
        ('Cognitive Status', 'CN'),
    ]
    cat_colors = {'Cognitive Status': COL_PRIMARY, 'APOE4 Status': COL_GREEN,
                  'Sex': COL_ORANGE, 'Age Tertile': COL_PURPLE}
    labels_e, aucs_e, colors_e = [], [], []
    for cat, name in order:
        rows = sub[sub['category'] == cat]
        for _, r in rows.iterrows():
            subg = r['subgroup'].strip()
            matched = (subg.lower() == name.lower()) or \
                      (cat == 'Age Tertile' and name.lower() in subg.lower())
            if matched:
                clean = subg
                if 'young' in clean.lower():
                    clean = 'Age \u226470'
                elif 'middle' in clean.lower():
                    clean = 'Age 71\u201375'
                elif 'old' in clean.lower():
                    clean = 'Age >75'
                labels_e.append(f'{clean} (n={int(r["n"])})')
                aucs_e.append(r['auc'])
                colors_e.append(cat_colors[cat])
                break

    ax_e.axvline(overall_auc, color=COL_LIGHT, linewidth=1.3, linestyle='--', zorder=1)
    ax_e.text(overall_auc + 0.003, len(labels_e) - 0.3,
              f'Overall\n{overall_auc:.3f}',
              fontsize=ANNO_SIZE - 1, color=COL_NEUTRAL, ha='left', va='top',
              style='italic')
    for i, (auc_v, col) in enumerate(zip(aucs_e, colors_e)):
        ax_e.plot(auc_v, i, 'D', color=col, markersize=6,
                  markeredgecolor='white', markeredgewidth=0.7, zorder=3)

    ax_e.set_yticks(range(len(labels_e)))
    ax_e.set_yticklabels(labels_e, fontsize=TICK_SIZE - 0.5)
    ax_e.set_xlabel('AUC', fontsize=LABEL_SIZE)
    ax_e.set_xlim(0.72, 0.95)
    ax_e.set_ylim(-0.5, len(labels_e) - 0.5)
    ax_e.set_title('Subgroup Performance',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.spines['left'].set_visible(False)
    ax_e.tick_params(axis='y', length=0)
    ax_e.xaxis.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
    ax_e.set_axisbelow(True)
    for sep in [2.5, 4.5, 6.5]:
        ax_e.axhline(sep, color='#E0E0E0', linewidth=0.5)

    leg_e = [Line2D([0], [0], marker='D', color='w', markerfacecolor=c,
                    markersize=5, label=l)
             for l, c in [('Cognitive', COL_PRIMARY), ('APOE4', COL_GREEN),
                          ('Sex', COL_ORANGE), ('Age', COL_PURPLE)]]
    ax_e.legend(handles=leg_e, loc='lower left', fontsize=LEGEND_SIZE,
                frameon=True, framealpha=0.9, edgecolor='#cccccc',
                handletextpad=0.2)
    ax_e.text(-0.18, 1.08, 'E', transform=ax_e.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (F) Threshold-Performance Tradeoff
    # ══════════════════════════════════════════════════════════════════
    thresholds = np.linspace(0.05, 0.95, 200)
    metrics = {k: [] for k in ['sens', 'spec', 'ppv', 'npv', 'acc']}
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(
            y_true, pred, labels=[0, 1]).ravel()
        metrics['sens'].append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0)
        metrics['spec'].append(tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0)
        metrics['ppv'].append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0)
        metrics['npv'].append(tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0)
        metrics['acc'].append((tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t))

    ax_f.plot(thresholds, metrics['sens'], color=COL_RED, lw=1.6,
              label='Sensitivity', zorder=3)
    ax_f.plot(thresholds, metrics['spec'], color=COL_PRIMARY, lw=1.6,
              label='Specificity', zorder=3)
    ax_f.plot(thresholds, metrics['ppv'], color=COL_GREEN, lw=1.2, ls='--',
              label='PPV', zorder=2)
    ax_f.plot(thresholds, metrics['npv'], color=COL_ORANGE, lw=1.2, ls='--',
              label='NPV', zorder=2)
    ax_f.plot(thresholds, metrics['acc'], color=COL_NEUTRAL, lw=1, ls=':',
              label='Accuracy', alpha=0.6, zorder=2)

    # Operating point markers
    ax_f.axvline(0.50, color='#aaa', lw=0.7, ls=':', alpha=0.5, zorder=1)
    ax_f.axvline(0.245, color=COL_RED, lw=0.6, ls=':', alpha=0.4, zorder=1)
    ax_f.axvline(0.729, color=COL_PRIMARY, lw=0.6, ls=':', alpha=0.4, zorder=1)
    ax_f.text(0.195, 0.38, '90% Sens\np=0.245', fontsize=ANNO_SIZE - 1.5,
              color=COL_RED, ha='center', style='italic',
              bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                        edgecolor='none', alpha=0.8))
    ax_f.text(0.785, 0.38, '90% Spec\np=0.729', fontsize=ANNO_SIZE - 1.5,
              color=COL_PRIMARY, ha='center', style='italic',
              bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                        edgecolor='none', alpha=0.8))

    ax_f.set_xlabel('Classification Threshold', fontsize=LABEL_SIZE)
    ax_f.set_ylabel('Metric Value', fontsize=LABEL_SIZE)
    ax_f.set_xlim(0.05, 0.95)
    ax_f.set_ylim(-0.02, 1.05)
    ax_f.set_title('Threshold\u2013Performance Tradeoff',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax_f.legend(loc='center left', fontsize=LEGEND_SIZE, frameon=True,
                framealpha=0.9, edgecolor='#cccccc', bbox_to_anchor=(0.0, 0.65))
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.text(-0.15, 1.08, 'F', transform=ax_f.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (G) Per-Stage Confusion Matrices (two side-by-side in one panel)
    # ══════════════════════════════════════════════════════════════════
    gk = df[df['stage'] == 'gatekeeper']
    rf = df[df['stage'] == 'reflex']
    cm_gk = confusion_matrix(gk['true_amyloid'], gk['predicted_class'])
    cm_rf = confusion_matrix(rf['true_amyloid'], rf['predicted_class'])

    ax_g.axis('off')
    fig.canvas.draw()
    pos_g = ax_g.get_position()
    gap = 0.010
    cm_w = (pos_g.width - gap) / 2
    cm_h = pos_g.height * 0.82

    ax_gk = fig.add_axes([pos_g.x0,
                           pos_g.y0 + pos_g.height * 0.02,
                           cm_w, cm_h])
    ax_rf = fig.add_axes([pos_g.x0 + cm_w + gap,
                           pos_g.y0 + pos_g.height * 0.02,
                           cm_w, cm_h])

    draw_cm_panel(ax_gk, cm_gk, 'Stage 1: Gatekeeper', '#1565C0')
    draw_cm_panel(ax_rf, cm_rf, 'Stage 2: Reflex', '#7B1FA2')

    ax_g.text(-0.04, 1.08, 'G', transform=ax_g.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ══════════════════════════════════════════════════════════════════
    # (H) Gatekeeper Threshold Sensitivity Heatmap
    # ══════════════════════════════════════════════════════════════════
    low_th = sorted(threshold['Low_Threshold'].unique())
    high_th = sorted(threshold['High_Threshold'].unique())
    res_mat = np.zeros((len(low_th), len(high_th)))
    acc_mat = np.zeros((len(low_th), len(high_th)))
    for i, lt in enumerate(low_th):
        for j, ht in enumerate(high_th):
            row = threshold[(threshold['Low_Threshold'] == lt) &
                            (threshold['High_Threshold'] == ht)]
            if len(row) > 0:
                res_mat[i, j] = row['Resolution_Rate'].values[0] * 100
                acc_mat[i, j] = row['Resolved_Accuracy'].values[0] * 100

    cmap_h = mcolors.LinearSegmentedColormap.from_list(
        'res', ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1'])
    im_h = ax_h.imshow(res_mat, cmap=cmap_h, aspect='auto', origin='lower',
                       vmin=25, vmax=85)

    for i in range(len(low_th)):
        for j in range(len(high_th)):
            rate = res_mat[i, j]
            acc_v = acc_mat[i, j]
            tc = 'white' if rate > 60 else COL_NEUTRAL
            if low_th[i] == 0.25 and high_th[j] == 0.75:
                ax_h.text(j, i, f'{rate:.0f}%\n({acc_v:.0f}%)',
                          ha='center', va='center',
                          fontsize=ANNO_SIZE - 1.5, fontweight='bold', color='yellow')
                ax_h.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor='#FFD600', linewidth=2.2))
            else:
                ax_h.text(j, i, f'{rate:.0f}%\n({acc_v:.0f}%)',
                          ha='center', va='center',
                          fontsize=ANNO_SIZE - 2, color=tc)

    ax_h.set_xticks(range(len(high_th)))
    ax_h.set_xticklabels([f'{h:.2f}' for h in high_th], fontsize=TICK_SIZE - 1)
    ax_h.set_yticks(range(len(low_th)))
    ax_h.set_yticklabels([f'{l:.2f}' for l in low_th], fontsize=TICK_SIZE - 1)
    ax_h.set_xlabel('Upper Threshold (Rule-In)', fontsize=LABEL_SIZE)
    ax_h.set_ylabel('Lower Threshold\n(Rule-Out)', fontsize=LABEL_SIZE)
    ax_h.set_title('Gatekeeper Threshold\nSensitivity',
                    fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    cbar = fig.colorbar(im_h, ax=ax_h, shrink=0.70, pad=0.03)
    cbar.set_label('Resolution %', fontsize=ANNO_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1)

    ax_h.text(-0.20, 1.08, 'H', transform=ax_h.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top')

    # ── Save ───────────────────────────────────────────────────────────
    pdf_path = RESULTS_DIR / 'figure_3_model_characterization.pdf'
    png_path = RESULTS_DIR / 'figure_3_model_characterization.png'

    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # Backward compat
    import shutil
    shutil.copy(png_path, RESULTS_DIR / 'figure_3_feature_importance.png')

    print(f'Figure 3 saved:')
    print(f'  PDF: {pdf_path}')
    print(f'  PNG: {png_path}')
    print(f'  Layout: 2\u00d74 landscape (8 panels)')
    print(f'  Resolution: {DPI} DPI')
    print(f'  Panels: A-H across 4 thematic sections')
    print(f'  AUC: {auc_mean:.4f} (95% CI: {ci_lo:.4f}\u2013{ci_hi:.4f})')


if __name__ == '__main__':
    main()
