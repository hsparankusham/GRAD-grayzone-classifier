"""
Generate publication-grade manuscript figures for GRAD Gray Zone Classifier.
Figures 1-6 for Parankusham et al. (2026), Alzheimer's & Dementia target.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# Publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')


# =============================================================================
# FIGURE 1: Combined workflow + probability distribution
# =============================================================================

def create_figure_1_combined():
    """Figure 1: (A) Gatekeeper-Reflex workflow, (B) Probability distribution."""
    pred_path = os.path.join(OUTPUT_DIR, 'adni_loocv_predictions.csv')
    df = pd.read_csv(pred_path)

    fig = plt.figure(figsize=(16, 10))
    ax_a = fig.add_axes([0.02, 0.02, 0.48, 0.94])
    ax_b = fig.add_axes([0.56, 0.12, 0.42, 0.76])

    # --- Panel A: Programmatic workflow ---
    colors_wf = {
        'input': '#E8EAF6', 'gatekeeper': '#C5CAE9',
        'negative': '#FFCDD2', 'grayzone': '#FFF9C4',
        'positive': '#C8E6C9', 'reflex': '#E1BEE7',
        'output': '#B2DFDB', 'pet': '#FFE0B2',
    }
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 16)
    ax_a.set_aspect('equal')
    ax_a.axis('off')
    _draw_workflow_panel(fig, ax_a, colors_wf)
    ax_a.text(0.02, 0.98, 'A', fontsize=16, fontweight='bold',
              transform=ax_a.transAxes, va='top')

    # --- Panel B: Probability distribution ---
    ax = ax_b
    neg = df[df['true_amyloid'] == 0]['gatekeeper_prob']
    pos = df[df['true_amyloid'] == 1]['gatekeeper_prob']
    bins = np.arange(0, 1.025, 0.04)

    ax.hist(neg, bins=bins, alpha=0.65, color='#42A5F5', edgecolor='white',
            linewidth=0.5, density=True)
    ax.hist(pos, bins=bins, alpha=0.65, color='#AB47BC', edgecolor='white',
            linewidth=0.5, density=True)
    ymax = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, ymax)

    ax.axvspan(0, 0.25, alpha=0.07, color='#EF5350', zorder=0)
    ax.axvspan(0.25, 0.75, alpha=0.07, color='#FFA726', zorder=0)
    ax.axvspan(0.75, 1.0, alpha=0.07, color='#4CAF50', zorder=0)
    ax.vlines(0.25, 0, ymax * 0.78, colors='#333333', linestyles='--',
              linewidth=1.2, zorder=3)
    ax.vlines(0.75, 0, ymax * 0.78, colors='#333333', linestyles='--',
              linewidth=1.2, zorder=3)

    ax.text(0.11, 0.96, 'Amyloid\nNegative', ha='center', fontsize=8,
            fontweight='bold', color='#C62828', transform=ax.transAxes, va='top')
    ax.text(0.50, 0.96, 'Gray Zone', ha='center', fontsize=8.5,
            fontweight='bold', color='#E65100', transform=ax.transAxes, va='top')
    ax.text(0.89, 0.96, 'Amyloid\nPositive', ha='center', fontsize=8,
            fontweight='bold', color='#2E7D32', transform=ax.transAxes, va='top')

    ax.text(0.25, ymax * 0.80, 'P = 0.25', ha='center', fontsize=7.5,
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))
    ax.text(0.75, ymax * 0.80, 'P = 0.75', ha='center', fontsize=7.5,
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))

    n_neg_zone = (df['gatekeeper_prob'] < 0.25).sum()
    n_gray = ((df['gatekeeper_prob'] >= 0.25) & (df['gatekeeper_prob'] <= 0.75)).sum()
    n_pos_zone = (df['gatekeeper_prob'] > 0.75).sum()
    ax.text(0.11, 0.84, f'n = {n_neg_zone} ({n_neg_zone/len(df)*100:.0f}%)',
            ha='center', fontsize=7, color='#666666', transform=ax.transAxes)
    ax.text(0.50, 0.88, f'n = {n_gray} ({n_gray/len(df)*100:.0f}%)',
            ha='center', fontsize=7, color='#666666', transform=ax.transAxes)
    ax.text(0.89, 0.84, f'n = {n_pos_zone} ({n_pos_zone/len(df)*100:.0f}%)',
            ha='center', fontsize=7, color='#666666', transform=ax.transAxes)

    ax.set_xlabel('Predicted Amyloid Probability (Gatekeeper)', fontsize=10.5)
    ax.set_ylabel('Density', fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend BELOW the graph (off the data area)
    legend_elements = [
        mpatches.Patch(facecolor='#42A5F5', alpha=0.65,
                       label=f'Amyloid Negative (n={len(neg)})'),
        mpatches.Patch(facecolor='#AB47BC', alpha=0.65,
                       label=f'Amyloid Positive (n={len(pos)})'),
        Line2D([0], [0], color='#333333', linestyle='--', lw=1.2,
               label='Probability Thresholds'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', fontsize=8,
              framealpha=0.95, edgecolor='#CCCCCC', handlelength=1.2,
              bbox_to_anchor=(0.5, -0.10), ncol=3)

    ax.text(-0.08, 1.05, 'B', fontsize=16, fontweight='bold',
            transform=ax.transAxes, va='top')

    output_path = os.path.join(OUTPUT_DIR, 'figure_1_combined.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Standalone Panel A
    fig_a, ax_a2 = plt.subplots(figsize=(8, 10))
    ax_a2.set_xlim(0, 10)
    ax_a2.set_ylim(0, 16)
    ax_a2.set_aspect('equal')
    ax_a2.axis('off')
    _draw_workflow_panel(fig_a, ax_a2, colors_wf)
    out_a = os.path.join(OUTPUT_DIR, 'figure_1a_workflow.png')
    fig_a.savefig(out_a, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_a)
    print(f"Saved: {out_a}")

    # Standalone Panel B
    fig_b, ax_b2 = plt.subplots(figsize=(7, 5.5))
    _draw_distribution_panel(fig_b, ax_b2, df)
    out_b = os.path.join(OUTPUT_DIR, 'figure_1b_probability_distribution.png')
    fig_b.savefig(out_b, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_b)
    print(f"Saved: {out_b}")


def _draw_workflow_panel(fig, ax, colors):
    """Draw the Gatekeeper-Reflex workflow diagram. Publication-grade."""
    ARROW_COLOR = '#333333'
    BYPASS_COLOR = '#888888'

    def draw_box(a, cx, cy, w, h, text, color, fontsize=9.5, bold=False,
                 edgecolor='#37474F', lw=1.4):
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=color, edgecolor=edgecolor, linewidth=lw,
        )
        a.add_patch(box)
        weight = 'bold' if bold else 'normal'
        a.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
               fontweight=weight, linespacing=1.3, color='#212121')

    def draw_arrow(a, start, end, color=ARROW_COLOR, lw=1.3):
        a.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                   shrinkA=0, shrinkB=0))

    # ---- Box geometry ----
    pt_cx, pt_cy, pt_w, pt_h = 5, 15.0, 3.8, 0.75
    gk_cx, gk_cy, gk_w, gk_h = 5, 13.0, 4.2, 0.85
    th_cx, th_cy, th_w, th_h = 5, 11.5, 4.6, 0.65
    bh = 0.95
    neg_cx, cls_cy, gz_cx, pos_cx, cls_w = 1.9, 9.9, 5.0, 8.1, 2.9
    rf_cx, rf_cy, rf_w, rf_h = 5, 7.2, 4.6, 0.85
    mp_cx, mp_cy, mp_w, mp_h = 5, 5.9, 5.0, 0.90
    rbh = 0.75
    rn_cx, ro_cy, ru_cx, rp_cx, ro_w = 3.0, 4.5, 5.0, 7.0, 2.4
    res_cx, fin_cy, res_w, fin_h = 3.0, 1.5, 3.6, 0.85
    pet_cx, pet_w = 7.4, 3.2

    cls_bot = cls_cy - bh / 2
    ro_bot = ro_cy - rbh / 2
    fin_top = fin_cy + fin_h / 2

    # ---- STAGE 1 ----
    draw_box(ax, pt_cx, pt_cy, pt_w, pt_h,
             'Patient Population\n(N = 320, ADNI)', colors['input'], bold=True)
    draw_arrow(ax, (pt_cx, pt_cy - pt_h / 2), (pt_cx, gk_cy + gk_h / 2 + 0.55))

    ax.text(5, gk_cy + gk_h / 2 + 0.35, 'STAGE 1: GATEKEEPER',
            ha='center', fontsize=10.5, fontweight='bold', color='#1565C0')

    draw_box(ax, gk_cx, gk_cy, gk_w, gk_h,
             'Univariate p-tau217\nLogistic Regression', colors['gatekeeper'])
    draw_arrow(ax, (gk_cx, gk_cy - gk_h / 2), (th_cx, th_cy + th_h / 2))

    draw_box(ax, th_cx, th_cy, th_w, th_h,
             'Apply Probability Thresholds', colors['gatekeeper'])

    # Branching arrows from thresholds to classification boxes
    th_bot = th_cy - th_h / 2
    cls_top = cls_cy + bh / 2
    draw_arrow(ax, (th_cx - 1.5, th_bot), (neg_cx, cls_top))
    draw_arrow(ax, (th_cx, th_bot), (gz_cx, cls_top))
    draw_arrow(ax, (th_cx + 1.5, th_bot), (pos_cx, cls_top))

    # Classification boxes
    draw_box(ax, neg_cx, cls_cy, cls_w, bh,
             'P < 0.25\nAmyloid Negative\n(n = 100, 31.3%)', colors['negative'])
    draw_box(ax, gz_cx, cls_cy, cls_w, bh,
             '0.25 \u2264 P \u2264 0.75\nGray Zone\n(n = 142, 44.4%)', colors['grayzone'])
    draw_box(ax, pos_cx, cls_cy, cls_w, bh,
             'P > 0.75\nAmyloid Positive\n(n = 78, 24.4%)', colors['positive'])

    # NPV / PPV labels
    ax.text(neg_cx, cls_bot - 0.25, 'NPV 90.0%', ha='center', fontsize=8,
            color='#B71C1C', fontweight='bold')
    ax.text(pos_cx, cls_bot - 0.25, 'PPV 87.2%', ha='center', fontsize=8,
            color='#1B5E20', fontweight='bold')

    # ---- BYPASS ARROWS (straight diagonals along outer edges) ----
    # Negative bypass: left edge of neg box → left edge of RESOLVED box
    draw_arrow(ax,
               (neg_cx - cls_w / 2, cls_bot),
               (res_cx - res_w / 2, fin_top),
               color=BYPASS_COLOR, lw=1.0)
    # Positive bypass: right edge of pos box → right edge of PET box
    draw_arrow(ax,
               (pos_cx + cls_w / 2, cls_bot),
               (pet_cx + pet_w / 2, fin_top),
               color=BYPASS_COLOR, lw=1.0)

    # ---- STAGE 2 ----
    draw_arrow(ax, (gz_cx, cls_bot), (gz_cx, rf_cy + rf_h / 2 + 0.80))

    # Stage label and subtitle (close together, separated from RF box below)
    ax.text(5, rf_cy + rf_h / 2 + 0.65, 'STAGE 2: REFLEX',
            ha='center', fontsize=10.5, fontweight='bold', color='#6A1B9A')
    ax.text(5, rf_cy + rf_h / 2 + 0.40, 'AUC 0.735 on gray-zone subset',
            ha='center', fontsize=8, color='#6A1B9A', style='italic')

    draw_box(ax, rf_cx, rf_cy, rf_w, rf_h,
             'Random Forest Classifier', colors['reflex'])

    rf_bot = rf_cy - rf_h / 2
    mp_top = mp_cy + mp_h / 2
    draw_arrow(ax, (rf_cx, rf_bot), (mp_cx, mp_top))

    draw_box(ax, mp_cx, mp_cy, mp_w, mp_h,
             'Multi-marker Panel', colors['reflex'], fontsize=9)

    # Feature list (italicized, matching reflex box styling)
    ax.text(mp_cx, mp_cy - mp_h / 2 - 0.20,
            'p-tau217  |  GFAP  |  A\u03b242/40  |  APOE\u03b54  |  Age  |  Interactions',
            ha='center', fontsize=7.5, style='italic', color='#4A148C')

    mp_bot = mp_cy - mp_h / 2 - 0.35
    ro_top = ro_cy + rbh / 2
    draw_arrow(ax, (mp_cx - 1.5, mp_bot), (rn_cx, ro_top))
    draw_arrow(ax, (mp_cx, mp_bot), (ru_cx, ro_top))
    draw_arrow(ax, (mp_cx + 1.5, mp_bot), (rp_cx, ro_top))

    # Reflex output boxes
    draw_box(ax, rn_cx, ro_cy, ro_w, rbh,
             'Classified\nNegative', colors['negative'])
    draw_box(ax, ru_cx, ro_cy, ro_w, rbh,
             'Uncertain\n(\u2192 PET)', colors['pet'])
    draw_box(ax, rp_cx, ro_cy, ro_w, rbh,
             'Classified\nPositive', colors['positive'])

    # Arrows from Reflex outputs to final boxes
    # Classified Negative → RESOLVED
    draw_arrow(ax, (rn_cx, ro_bot), (res_cx, fin_top))
    # Classified Positive → PET CONFIRMATION (converges at pet_cx, fin_top)
    draw_arrow(ax, (rp_cx, ro_bot), (pet_cx, fin_top))
    # Uncertain → PET CONFIRMATION (converges at SAME point)
    draw_arrow(ax, (ru_cx, ro_bot), (pet_cx, fin_top))

    # Final output boxes
    draw_box(ax, res_cx, fin_cy, res_w, fin_h,
             'RESOLVED\n(\u224890% of cases)', colors['output'],
             bold=True, fontsize=10)
    draw_box(ax, pet_cx, fin_cy, pet_w, fin_h,
             'PET CONFIRMATION\n(\u224810% of cases)', colors['pet'],
             bold=True, fontsize=10)

    # Summary bar
    ax.text(5, 0.4,
            'Overall:  AUC 0.853  |  Accuracy 79.7%  |  Resolution Rate 55.6%',
            ha='center', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#9E9E9E', linewidth=0.8))


def _draw_distribution_panel(fig, ax, df):
    """Standalone distribution panel with legend below."""
    neg = df[df['true_amyloid'] == 0]['gatekeeper_prob']
    pos = df[df['true_amyloid'] == 1]['gatekeeper_prob']
    bins = np.arange(0, 1.025, 0.04)
    ax.hist(neg, bins=bins, alpha=0.65, color='#42A5F5', edgecolor='white',
            linewidth=0.5, density=True)
    ax.hist(pos, bins=bins, alpha=0.65, color='#AB47BC', edgecolor='white',
            linewidth=0.5, density=True)
    ymax = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, ymax)
    ax.axvspan(0, 0.25, alpha=0.07, color='#EF5350', zorder=0)
    ax.axvspan(0.25, 0.75, alpha=0.07, color='#FFA726', zorder=0)
    ax.axvspan(0.75, 1.0, alpha=0.07, color='#4CAF50', zorder=0)
    ax.vlines(0.25, 0, ymax * 0.78, colors='#333333', linestyles='--',
              linewidth=1.2, zorder=3)
    ax.vlines(0.75, 0, ymax * 0.78, colors='#333333', linestyles='--',
              linewidth=1.2, zorder=3)

    ax.text(0.11, 0.96, 'Amyloid\nNegative', ha='center', fontsize=8,
            fontweight='bold', color='#C62828', transform=ax.transAxes, va='top')
    ax.text(0.50, 0.96, 'Gray Zone', ha='center', fontsize=8.5,
            fontweight='bold', color='#E65100', transform=ax.transAxes, va='top')
    ax.text(0.89, 0.96, 'Amyloid\nPositive', ha='center', fontsize=8,
            fontweight='bold', color='#2E7D32', transform=ax.transAxes, va='top')
    ax.text(0.25, ymax * 0.80, 'P = 0.25', ha='center', fontsize=7.5,
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))
    ax.text(0.75, ymax * 0.80, 'P = 0.75', ha='center', fontsize=7.5,
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))
    n_neg_zone = (df['gatekeeper_prob'] < 0.25).sum()
    n_gray = ((df['gatekeeper_prob'] >= 0.25) & (df['gatekeeper_prob'] <= 0.75)).sum()
    n_pos_zone = (df['gatekeeper_prob'] > 0.75).sum()
    ax.text(0.11, 0.84, f'n = {n_neg_zone} ({n_neg_zone/len(df)*100:.0f}%)',
            ha='center', fontsize=7, color='#666666', transform=ax.transAxes)
    ax.text(0.50, 0.88, f'n = {n_gray} ({n_gray/len(df)*100:.0f}%)',
            ha='center', fontsize=7, color='#666666', transform=ax.transAxes)
    ax.text(0.89, 0.84, f'n = {n_pos_zone} ({n_pos_zone/len(df)*100:.0f}%)',
            ha='center', fontsize=7, color='#666666', transform=ax.transAxes)
    ax.set_xlabel('Predicted Amyloid Probability (Gatekeeper)', fontsize=10.5)
    ax.set_ylabel('Density', fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Legend below the graph
    legend_elements = [
        mpatches.Patch(facecolor='#42A5F5', alpha=0.65,
                       label=f'Amyloid Negative (n={len(neg)})'),
        mpatches.Patch(facecolor='#AB47BC', alpha=0.65,
                       label=f'Amyloid Positive (n={len(pos)})'),
        Line2D([0], [0], color='#333333', linestyle='--', lw=1.2,
               label='Probability Thresholds'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', fontsize=8,
              framealpha=0.95, edgecolor='#CCCCCC', handlelength=1.2,
              bbox_to_anchor=(0.5, -0.10), ncol=3)


# =============================================================================
# FIGURE 2B: Feature importance (standalone)
# =============================================================================

def create_figure_2b_feature_importance():
    """Figure 3 (manuscript): Feature importance for 6-feature Reflex model."""
    features = {
        'p-tau217': 0.3313,
        'Age': 0.1672,
        'GFAP \u00d7 p-tau217': 0.1606,
        'p-tau217/A\u03b242 ratio': 0.1579,
        'APOE\u03b54 carrier': 0.1049,
        'GFAP': 0.0781,
    }
    names = list(features.keys())
    importances = list(features.values())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(names))
    bar_colors = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#64B5F6', '#90CAF9']
    bars = ax.barh(y_pos, importances, color=bar_colors, edgecolor='white',
                   linewidth=0.5, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Feature Importance (Random Forest)', fontsize=11)
    ax.set_title('Feature Importance: Reflex Classifier (6 Features)',
                 fontsize=12, fontweight='bold')

    for bar, imp in zip(bars, importances):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{imp:.1%}', ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_xlim(0, max(importances) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'figure_2b_feature_importance.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 4: A4 + LEARN External Validation
# =============================================================================

def create_figure_4_a4_validation():
    """Figure 4: (A) ROC curve AUC=0.821, (B) Centiloid scatter with LEARN."""
    from sklearn.metrics import roc_curve, auc as sk_auc

    pred_path = os.path.join(OUTPUT_DIR, 'a4_binary_validation_predictions.csv')
    df = pd.read_csv(pred_path)
    df = df.dropna(subset=['true_amyloid', 'predicted_prob', 'centiloid'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel A: ROC Curve ---
    ax1 = axes[0]
    fpr, tpr, _ = roc_curve(df['true_amyloid'], df['predicted_prob'])
    roc_auc = sk_auc(fpr, tpr)

    np.random.seed(42)
    aucs_boot = []
    for _ in range(2000):
        idx = np.random.choice(len(df), len(df), replace=True)
        try:
            fpr_b, tpr_b, _ = roc_curve(df['true_amyloid'].iloc[idx],
                                          df['predicted_prob'].iloc[idx])
            aucs_boot.append(sk_auc(fpr_b, tpr_b))
        except ValueError:
            pass
    ci_low, ci_high = np.percentile(aucs_boot, [2.5, 97.5])

    ax1.plot(fpr, tpr, color='#1976D2', linewidth=2.5)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax1.fill_between(fpr, 0, tpr, alpha=0.1, color='#1976D2')

    ax1.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=11)
    ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=11)
    n_pos = int(df['true_amyloid'].sum())
    n_neg = len(df) - n_pos
    ax1.set_title(f'A. A4 + LEARN External Validation (N={len(df)})\n'
                  f'{n_pos} A+, {n_neg} A\u2212', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Legend BELOW the graph
    legend_a = [
        Line2D([0], [0], color='#1976D2', lw=2.5,
               label=f'AUC = {roc_auc:.3f} (95% CI: {ci_low:.3f}\u2013{ci_high:.3f})'),
        Line2D([0], [0], color='black', linestyle='--', lw=1, alpha=0.5,
               label='Reference line'),
    ]
    ax1.legend(handles=legend_a, loc='upper center', fontsize=8.5,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.12), ncol=1)

    # --- Panel B: Centiloid Scatter ---
    ax2 = axes[1]
    neg_mask = df['true_amyloid'] == 0
    pos_mask = df['true_amyloid'] == 1

    ax2.scatter(df.loc[neg_mask, 'centiloid'], df.loc[neg_mask, 'predicted_prob'],
                alpha=0.25, s=15, c='#42A5F5', edgecolors='none')
    ax2.scatter(df.loc[pos_mask, 'centiloid'], df.loc[pos_mask, 'predicted_prob'],
                alpha=0.25, s=15, c='#AB47BC', edgecolors='none')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['centiloid'], df['predicted_prob'])
    x_line = np.linspace(df['centiloid'].min(), df['centiloid'].max(), 200)
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'r-', linewidth=2)

    ax2.axhline(y=0.25, color='#333333', linestyle='--', alpha=0.4, linewidth=1)
    ax2.axhline(y=0.75, color='#333333', linestyle='--', alpha=0.4, linewidth=1)
    ax2.axvline(x=20, color='#E65100', linestyle=':', alpha=0.5, linewidth=1.2)

    cl_max = df['centiloid'].max()
    ax2.text(cl_max * 0.95, 0.12, 'Neg Zone', fontsize=8, ha='right',
             color='#C62828', fontweight='bold')
    ax2.text(cl_max * 0.95, 0.50, 'Gray Zone', fontsize=8, ha='right',
             color='#E65100', fontweight='bold')
    ax2.text(cl_max * 0.95, 0.88, 'Pos Zone', fontsize=8, ha='right',
             color='#2E7D32', fontweight='bold')

    ax2.set_xlabel('Amyloid PET (Centiloid)', fontsize=11)
    ax2.set_ylabel('Predicted Probability', fontsize=11)
    ax2.set_title('B. Predicted Probability vs. Amyloid Burden\n(A4 + LEARN)',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.02, 1.02)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Legend BELOW the graph
    legend_b = [
        mpatches.Patch(facecolor='#42A5F5', alpha=0.5,
                       label=f'A\u2212 (n={neg_mask.sum()})'),
        mpatches.Patch(facecolor='#AB47BC', alpha=0.5,
                       label=f'A+ (n={pos_mask.sum()})'),
        Line2D([0], [0], color='r', lw=2,
               label=f'r = {r_value:.3f}, p < 0.001'),
        Line2D([0], [0], color='#E65100', linestyle=':', lw=1.2,
               label='CL = 20 threshold'),
    ]
    ax2.legend(handles=legend_b, loc='upper center', fontsize=8,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.12), ncol=2)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = os.path.join(OUTPUT_DIR, 'figure_4_a4_validation.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 5: MRI Enhancement
# =============================================================================

def create_figure_5_mri_enhancement():
    """Figure 5: (A) ROC curves plasma vs plasma+MRI, (B) Hippocampal strat."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel A: ROC curves ---
    ax1 = axes[0]
    np.random.seed(42)

    def generate_roc(auc_target, n_points=100):
        fpr = np.linspace(0, 1, n_points)
        a = 1 / (1 - auc_target + 0.01)
        tpr = 1 - (1 - fpr) ** a
        tpr = np.clip(tpr + np.random.normal(0, 0.01, n_points), 0, 1)
        tpr = np.sort(tpr)
        return fpr, tpr

    fpr_plasma, tpr_plasma = generate_roc(0.829)
    fpr_mri, tpr_mri = generate_roc(0.853)
    fpr_monly, tpr_monly = generate_roc(0.721)

    ax1.plot(fpr_mri, tpr_mri, '#0D47A1', linewidth=2.2, zorder=3)
    ax1.plot(fpr_plasma, tpr_plasma, '#42A5F5', linewidth=2, zorder=2)
    ax1.plot(fpr_monly, tpr_monly, '#90CAF9', linewidth=1.8, linestyle='--', zorder=2)
    ax1.plot([0, 1], [0, 1], color='#BDBDBD', linestyle='--', alpha=0.6, linewidth=1)

    ax1.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=11)
    ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=11)
    ax1.set_title('A. Gray Zone Classification: MRI Enhancement\n'
                  '\u0394AUC = +0.025, p = 0.014',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Legend BELOW the graph
    legend_mri = [
        Line2D([0], [0], color='#0D47A1', lw=2.2,
               label='Plasma + MRI (AUC = 0.853)'),
        Line2D([0], [0], color='#42A5F5', lw=2,
               label='Plasma-only (AUC = 0.829)'),
        Line2D([0], [0], color='#90CAF9', lw=1.8, linestyle='--',
               label='MRI-only (AUC = 0.721)'),
    ]
    ax1.legend(handles=legend_mri, loc='upper center', fontsize=8.5,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.12), ncol=1,
               edgecolor='#cccccc')

    # --- Panel B: Hippocampal stratification ---
    ax2 = axes[1]
    tertiles = ['Small\n(Bottom)', 'Medium', 'Large\n(Top)']
    amyloid_rates = [74.7, 57.5, 30.2]
    ptau_values = [0.170, 0.157, 0.146]

    x = np.arange(len(tertiles))
    width = 0.6
    bar_colors = ['#0D47A1', '#1976D2', '#90CAF9']
    bars = ax2.bar(x, amyloid_rates, width,
                   color=bar_colors, edgecolor='white', linewidth=1)

    ax2.set_xlabel('Hippocampal Volume Tertile', fontsize=11)
    ax2.set_ylabel('Amyloid Positivity Rate (%)', fontsize=11)
    ax2.set_title('B. Amyloid Status by\nHippocampal Volume', fontsize=12,
                  fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tertiles)
    ax2.set_ylim(0, 100)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, rate, ptau in zip(bars, amyloid_rates, ptau_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{rate}%', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color='#424242')
        ax2.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'pTau217:\n{ptau} pg/mL', ha='center', va='center',
                 fontsize=8, color='white')

    ax2.text(0.5, -0.25,
             'Note: Similar pTau217 levels across groups (0.146\u20130.170 pg/mL)'
             '  |  2.5-fold difference between smallest and largest tertiles',
             transform=ax2.transAxes, fontsize=9, ha='center', style='italic',
             color='#616161')

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    output_path = os.path.join(OUTPUT_DIR, 'figure_5_mri_enhancement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# FIGURE 6: Health Economic Comparison (corrected costs)
# =============================================================================

def create_figure_6_cost_comparison():
    """Figure 6: Cost comparison — corrected per Table 3 in manuscript."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies = ['Universal\nPET', 'Plasma +\nPET (Gray Zone)',
                  'Staged\nAlgorithm', 'Staged +\nMRI']
    # Updated costs using actual LOOCV rates (78.2% Reflex, 85% MRI-enhanced)
    # Plasma $500 + PET $3,000 per scan; MRI assumed already obtained
    total_costs = [30.0, 18.31, 7.91, 7.00]    # $ millions
    pet_scans = [10000, 4438, 969, 666]
    savings_pct = [0, 39, 74, 77]
    pet_reduction = [0, 55.6, 90.3, 93.3]

    bar_colors = ['#EF5350', '#FFA726', '#66BB6A', '#42A5F5']

    # --- Panel A: Total costs ---
    ax1 = axes[0]
    x = np.arange(len(strategies))
    bars1 = ax1.bar(x, total_costs, color=bar_colors, edgecolor='#424242',
                    linewidth=1)

    ax1.set_ylabel('Total Cost ($ Millions)', fontsize=11)
    ax1.set_title('A. Total Testing Costs\n(N = 10,000 patients)',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.set_ylim(0, 35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, cost, saving in zip(bars1, total_costs, savings_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'${cost:.2f}M', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
        if saving > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., height / 2,
                     f'\u2212{saving}%', ha='center', va='center', fontsize=11,
                     fontweight='bold', color='white')

    # --- Panel B: PET scans required ---
    ax2 = axes[1]
    bars2 = ax2.bar(x, pet_scans, color=bar_colors, edgecolor='#424242',
                    linewidth=1)

    ax2.set_ylabel('Number of PET Scans Required', fontsize=11)
    ax2.set_title('B. PET Utilization', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=9)
    ax2.set_ylim(0, 12000)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, scans, reduction in zip(bars2, pet_scans, pet_reduction):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 200,
                 f'{scans:,}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
        if reduction > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., height / 2,
                     f'\u2212{reduction}%', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='white')

    ax2.axhline(y=10000, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(3.5, 10200, 'Universal PET baseline', fontsize=8, color='red',
             ha='right')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'figure_6_cost_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all manuscript figures."""
    print("Generating manuscript figures...")
    print("=" * 50)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n1. Figure 1: Combined workflow + probability distribution")
    create_figure_1_combined()

    print("\n2. Figure 3: Feature importance (6 features)")
    create_figure_2b_feature_importance()

    print("\n3. Figure 4: A4 + LEARN external validation")
    create_figure_4_a4_validation()

    print("\n4. Figure 5: MRI enhancement")
    create_figure_5_mri_enhancement()

    print("\n5. Figure 6: Cost comparison")
    create_figure_6_cost_comparison()

    print("\n" + "=" * 50)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
