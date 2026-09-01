# =============================================================================
# GRAD Manuscript Figure Generation
# =============================================================================
# Paste each cell (separated by # %% markers) into a Jupyter notebook.
# Outputs: individual PNG/PDF files for compositing in Inkscape.
#
# Figures:
#   2A: Overall ROC curve
#   2B: Gatekeeper-resolved ROC
#   2C: Reflex gray zone ROC
#   3A: Feature importance (horizontal bar)
#   3B: Overall confusion matrix
#   3C: Subgroup AUC forest plot
#   3D: Threshold operating point curves
#   4A: A4 ROC curve
#   4B: Centiloid scatter
#   4C: A4 calibration curve
#   5A: MRI enhancement ROC (placeholder — needs MRI data)
#   5B: Hippocampal tertile bar chart
#   6:  Cost simulation bar chart
# =============================================================================

# %% [markdown]
# # Setup & Data Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ── Color palette (Nature-style muted tones) ─────────────────────────────────
C = {
    'blue':    '#2166AC',
    'red':     '#B2182B',
    'green':   '#1B7837',
    'orange':  '#E08214',
    'purple':  '#7B3294',
    'grey':    '#636363',
    'ltblue':  '#D1E5F0',
    'ltred':   '#FDDBC7',
    'ltgreen': '#D9F0D3',
    'gold':    '#DFC27D',
    'bg':      '#FAFAFA',
}

# ── Load data ─────────────────────────────────────────────────────────────────
from _grad_paths import RESULTS, PANELS
RESULTS = str(RESULTS)
OUTDIR = str(PANELS)

adni = pd.read_csv(f'{RESULTS}/adni_loocv_predictions.csv')
a4   = pd.read_csv(f'{RESULTS}/a4_binary_validation_predictions.csv')

y_adni = adni['true_amyloid'].values
p_adni = adni['predicted_prob'].values
stage  = adni['stage'].values

y_a4 = a4['true_amyloid'].values
p_a4 = a4['predicted_prob'].values

print(f"ADNI: {len(adni)} samples, {int(y_adni.sum())} amyloid+")
print(f"A4:   {len(a4)} samples,  {int(y_a4.sum())} amyloid+")


# %% [markdown]
# # Figure 2: ROC Curves (3 panels)

# %% Figure 2A — Overall ROC
fig, ax = plt.subplots(figsize=(4.5, 4.5))

fpr, tpr, _ = roc_curve(y_adni, p_adni)
auc_val = roc_auc_score(y_adni, p_adni)

# Bootstrap CI band
rng = np.random.RandomState(42)
tpr_boot = []
mean_fpr = np.linspace(0, 1, 200)
for _ in range(500):
    idx = rng.choice(len(y_adni), len(y_adni), replace=True)
    if len(np.unique(y_adni[idx])) < 2:
        continue
    f, t, _ = roc_curve(y_adni[idx], p_adni[idx])
    tpr_boot.append(np.interp(mean_fpr, f, t))
tpr_boot = np.array(tpr_boot)
tpr_lo = np.percentile(tpr_boot, 2.5, axis=0)
tpr_hi = np.percentile(tpr_boot, 97.5, axis=0)

ax.fill_between(mean_fpr, tpr_lo, tpr_hi, alpha=0.15, color=C['blue'], linewidth=0)
ax.plot(fpr, tpr, color=C['blue'], linewidth=2.2)
ax.plot([0, 1], [0, 1], '--', color=C['grey'], linewidth=0.8, alpha=0.6)

ax.set_xlabel('1 − Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

# AUC annotation
ax.text(0.55, 0.15, f'AUC = {auc_val:.3f}\n95% CI: 0.813–0.897',
        fontsize=11, fontweight='bold', color=C['blue'],
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C['blue'], alpha=0.9, linewidth=1.2))

fig.savefig(f'{OUTDIR}/fig2a_overall_roc.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig2a_overall_roc.pdf', transparent=True)
plt.show()


# %% Figure 2B — Gatekeeper-resolved ROC
fig, ax = plt.subplots(figsize=(4.5, 4.5))

gk_mask = stage == 'gatekeeper'
y_gk = y_adni[gk_mask]
p_gk = adni.loc[gk_mask, 'gatekeeper_prob'].values

fpr_gk, tpr_gk, _ = roc_curve(y_gk, p_gk)
auc_gk = roc_auc_score(y_gk, p_gk)

# Bootstrap CI
tpr_boot = []
for _ in range(500):
    idx = rng.choice(len(y_gk), len(y_gk), replace=True)
    if len(np.unique(y_gk[idx])) < 2:
        continue
    f, t, _ = roc_curve(y_gk[idx], p_gk[idx])
    tpr_boot.append(np.interp(mean_fpr, f, t))
tpr_boot = np.array(tpr_boot)
ax.fill_between(mean_fpr, np.percentile(tpr_boot, 2.5, axis=0),
                np.percentile(tpr_boot, 97.5, axis=0),
                alpha=0.15, color=C['green'], linewidth=0)

ax.plot(fpr_gk, tpr_gk, color=C['green'], linewidth=2.2)
ax.plot([0, 1], [0, 1], '--', color=C['grey'], linewidth=0.8, alpha=0.6)

ax.set_xlabel('1 − Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

ax.text(0.55, 0.15, f'AUC = {auc_gk:.3f}',
        fontsize=11, fontweight='bold', color=C['green'],
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C['green'], alpha=0.9, linewidth=1.2))

fig.savefig(f'{OUTDIR}/fig2b_gatekeeper_roc.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig2b_gatekeeper_roc.pdf', transparent=True)
plt.show()


# %% Figure 2C — Reflex (gray zone) ROC
fig, ax = plt.subplots(figsize=(4.5, 4.5))

gz_mask = stage == 'reflex'
y_gz = y_adni[gz_mask]
p_gz = p_adni[gz_mask]

fpr_gz, tpr_gz, _ = roc_curve(y_gz, p_gz)
auc_gz = roc_auc_score(y_gz, p_gz)

# Bootstrap CI
tpr_boot = []
for _ in range(500):
    idx = rng.choice(len(y_gz), len(y_gz), replace=True)
    if len(np.unique(y_gz[idx])) < 2:
        continue
    f, t, _ = roc_curve(y_gz[idx], p_gz[idx])
    tpr_boot.append(np.interp(mean_fpr, f, t))
tpr_boot = np.array(tpr_boot)
ax.fill_between(mean_fpr, np.percentile(tpr_boot, 2.5, axis=0),
                np.percentile(tpr_boot, 97.5, axis=0),
                alpha=0.15, color=C['orange'], linewidth=0)

ax.plot(fpr_gz, tpr_gz, color=C['orange'], linewidth=2.2)
ax.plot([0, 1], [0, 1], '--', color=C['grey'], linewidth=0.8, alpha=0.6)

ax.set_xlabel('1 − Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

ax.text(0.55, 0.15, f'AUC = {auc_gz:.3f}',
        fontsize=11, fontweight='bold', color=C['orange'],
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C['orange'], alpha=0.9, linewidth=1.2))

fig.savefig(f'{OUTDIR}/fig2c_reflex_roc.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig2c_reflex_roc.pdf', transparent=True)
plt.show()


# %% [markdown]
# # Figure 3: Model Characterization (4 panels: A, B, C, D)

# %% Figure 3A — Feature Importance (horizontal bar chart)
fig, ax = plt.subplots(figsize=(5, 3.5))

features = ['p-Tau217', 'Age', 'GFAP × p-Tau217', 'Tau/Aβ42 ratio',
            'APOE ε4', 'GFAP']
importance = [0.331, 0.167, 0.161, 0.158, 0.105, 0.078]

# Sort ascending for horizontal bars (bottom to top)
order = np.argsort(importance)
features_sorted = [features[i] for i in order]
imp_sorted = [importance[i] for i in order]

colors = [C['blue'] if imp > 0.15 else C['ltblue'] for imp in imp_sorted]
# Make top feature darker
colors[-1] = '#0D47A1'

bars = ax.barh(range(len(features_sorted)), imp_sorted, height=0.6,
               color=colors, edgecolor='white', linewidth=0.5)

# Value labels
for i, (bar, val) in enumerate(zip(bars, imp_sorted)):
    ax.text(val + 0.005, i, f'{val:.1%}', va='center', fontsize=9.5,
            fontweight='bold' if val > 0.15 else 'normal',
            color='#333333')

ax.set_yticks(range(len(features_sorted)))
ax.set_yticklabels(features_sorted, fontsize=10)
ax.set_xlabel('Mean Decrease in Gini Impurity', fontsize=10.5)
ax.set_xlim(0, 0.40)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

fig.savefig(f'{OUTDIR}/fig3a_feature_importance.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig3a_feature_importance.pdf', transparent=True)
plt.show()


# %% Figure 3B — Overall Confusion Matrix
fig, ax = plt.subplots(figsize=(4, 3.8))

pred_adni = (p_adni >= 0.5).astype(int)
cm = confusion_matrix(y_adni, pred_adni)
# cm = [[TN, FP], [FN, TP]]

labels = np.array([['TN', 'FP'], ['FN', 'TP']])
colors_cm = np.array([[C['ltblue'], C['ltred']],
                       [C['ltred'], C['ltblue']]])

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        label = labels[i, j]
        rect = FancyBboxPatch((j - 0.45, i - 0.45), 0.9, 0.9,
                               boxstyle='round,pad=0.05',
                               facecolor=colors_cm[i, j],
                               edgecolor='#999999', linewidth=1.0)
        ax.add_patch(rect)
        ax.text(j, i + 0.05, f'{val}', ha='center', va='center',
                fontsize=22, fontweight='bold', color='#1a1a1a')
        ax.text(j, i - 0.30, label, ha='center', va='center',
                fontsize=9, color='#666666')

ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.6, 1.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Predicted Aβ−', 'Predicted Aβ+'], fontsize=10)
ax.set_yticks([0, 1])
ax.set_yticklabels(['True Aβ−', 'True Aβ+'], fontsize=10)
ax.invert_yaxis()
ax.set_aspect('equal')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(length=0)

# Annotation
tp, fn = cm[1, 1], cm[1, 0]
tn, fp = cm[0, 0], cm[0, 1]
sens = tp / (tp + fn)
spec = tn / (tn + fp)
acc = (tp + tn) / cm.sum()
ax.text(0.5, -0.55, f'Accuracy {acc:.1%}  |  Sens {sens:.1%}  |  Spec {spec:.1%}',
        ha='center', fontsize=9, color=C['grey'], transform=ax.transData)

fig.savefig(f'{OUTDIR}/fig3b_confusion_matrix.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig3b_confusion_matrix.pdf', transparent=True)
plt.show()


# %% Figure 3C — Subgroup AUC Forest Plot
fig, ax = plt.subplots(figsize=(5, 4.5))

# Subgroup data (from supp_table_subgroup_performance.csv)
subgroups = pd.read_csv(f'{RESULTS}/tables/supp_table_subgroup_performance.csv')
subgroups = subgroups[subgroups['subgroup'] != 'Overall']

# Order: cognitive status, APOE4, sex, age tertile
cat_order = ['Cognitive Status', 'APOE4 Status', 'Sex', 'Age Tertile']
rows = []
for cat in cat_order:
    subset = subgroups[subgroups['category'] == cat]
    for _, row in subset.iterrows():
        rows.append(row)
plot_df = pd.DataFrame(rows).reset_index(drop=True)

n_rows = len(plot_df)
y_pos = np.arange(n_rows)

# Compute bootstrap CIs per subgroup (approximate with +/- 1.96*SE)
# For a more precise version, you'd bootstrap from predictions.
# Using the normal approximation: SE ≈ sqrt(AUC*(1-AUC)/min(n+,n-))
cis = []
for _, row in plot_df.iterrows():
    auc = row['auc']
    n = row['n']
    prev = row['prevalence']
    n_pos = max(int(n * prev), 1)
    n_neg = max(n - n_pos, 1)
    se = np.sqrt(auc * (1 - auc) / min(n_pos, n_neg))
    cis.append(1.96 * se)
cis = np.array(cis)

# Colors by category
cat_colors = {
    'Cognitive Status': C['blue'],
    'APOE4 Status': C['green'],
    'Sex': C['purple'],
    'Age Tertile': C['orange'],
}
colors = [cat_colors[c] for c in plot_df['category']]

# Plot
ax.errorbar(plot_df['auc'], y_pos, xerr=cis, fmt='o',
            color='none', ecolor='#aaaaaa', elinewidth=1.2, capsize=3)
for i, (auc, c) in enumerate(zip(plot_df['auc'], colors)):
    ax.plot(auc, i, 'o', color=c, markersize=8, zorder=5)

# Overall AUC reference line
overall_auc = roc_auc_score(y_adni, p_adni)
ax.axvline(overall_auc, color=C['grey'], linestyle='--', linewidth=1, alpha=0.5)
ax.text(overall_auc + 0.003, n_rows - 0.3, f'Overall\n{overall_auc:.3f}',
        fontsize=8, color=C['grey'], va='top')

# Clean labels
display_labels = []
for _, row in plot_df.iterrows():
    sg = row['subgroup']
    # Shorten age tertile labels
    if 'Young' in sg:
        sg = 'Age ≤70'
    elif 'Middle' in sg:
        sg = 'Age 70–75'
    elif 'Old' in sg:
        sg = 'Age >75'
    display_labels.append(f"{sg}  (n={int(row['n'])})")

ax.set_yticks(y_pos)
ax.set_yticklabels(display_labels, fontsize=9)
ax.set_xlabel('AUC', fontsize=10.5)
ax.set_xlim(0.70, 0.96)
ax.invert_yaxis()

# Category separators
prev_cat = None
for i, (_, row) in enumerate(plot_df.iterrows()):
    if prev_cat is not None and row['category'] != prev_cat:
        ax.axhline(i - 0.5, color='#dddddd', linewidth=0.8)
    prev_cat = row['category']

fig.savefig(f'{OUTDIR}/fig3c_subgroup_forest.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig3c_subgroup_forest.pdf', transparent=True)
plt.show()


# %% Figure 3D — Threshold Operating Point Curves
fig, ax = plt.subplots(figsize=(5.5, 4))

thresholds = np.linspace(0.01, 0.99, 200)
sens_arr, spec_arr, ppv_arr, npv_arr, acc_arr = [], [], [], [], []

for t in thresholds:
    pred = (p_adni >= t).astype(int)
    tp = ((pred == 1) & (y_adni == 1)).sum()
    fn = ((pred == 0) & (y_adni == 1)).sum()
    tn = ((pred == 0) & (y_adni == 0)).sum()
    fp = ((pred == 1) & (y_adni == 0)).sum()
    sens_arr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    spec_arr.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    ppv_arr.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    npv_arr.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
    acc_arr.append((tp + tn) / len(y_adni))

ax.plot(thresholds, sens_arr, color=C['red'], linewidth=2, label='Sensitivity')
ax.plot(thresholds, spec_arr, color=C['blue'], linewidth=2, label='Specificity')
ax.plot(thresholds, ppv_arr, color=C['green'], linewidth=1.5, linestyle='--', label='PPV')
ax.plot(thresholds, npv_arr, color=C['orange'], linewidth=1.5, linestyle='--', label='NPV')
ax.plot(thresholds, acc_arr, color=C['grey'], linewidth=1.5, linestyle=':', label='Accuracy')

# 90% operating points
ax.axvline(0.240, color=C['red'], linewidth=0.8, alpha=0.5, linestyle=':')
ax.axvline(0.674, color=C['blue'], linewidth=0.8, alpha=0.5, linestyle=':')
ax.text(0.240, 0.03, '0.24', ha='center', fontsize=8, color=C['red'])
ax.text(0.674, 0.03, '0.67', ha='center', fontsize=8, color=C['blue'])

# Annotations
ax.annotate('90% Sens\nrule-out', xy=(0.240, 0.90), fontsize=7.5,
            color=C['red'], ha='right',
            xytext=(0.14, 0.78), arrowprops=dict(arrowstyle='->', color=C['red'], lw=0.8))
ax.annotate('90% Spec\nrule-in', xy=(0.674, 0.90), fontsize=7.5,
            color=C['blue'], ha='left',
            xytext=(0.76, 0.78), arrowprops=dict(arrowstyle='->', color=C['blue'], lw=0.8))

ax.set_xlabel('Classification Threshold', fontsize=10.5)
ax.set_ylabel('Metric Value', fontsize=10.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc='center left', fontsize=8.5, frameon=True, framealpha=0.9,
          edgecolor='#cccccc')

fig.savefig(f'{OUTDIR}/fig3d_threshold_curves.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig3d_threshold_curves.pdf', transparent=True)
plt.show()


# %% [markdown]
# # Figure 4: A4 External Validation (3 panels)

# %% Figure 4A — A4 ROC
fig, ax = plt.subplots(figsize=(4.5, 4.5))

fpr_a4, tpr_a4, _ = roc_curve(y_a4, p_a4)
auc_a4 = roc_auc_score(y_a4, p_a4)

# Bootstrap CI
tpr_boot = []
for _ in range(500):
    idx = rng.choice(len(y_a4), len(y_a4), replace=True)
    if len(np.unique(y_a4[idx])) < 2:
        continue
    f, t, _ = roc_curve(y_a4[idx], p_a4[idx])
    tpr_boot.append(np.interp(mean_fpr, f, t))
tpr_boot = np.array(tpr_boot)
ax.fill_between(mean_fpr, np.percentile(tpr_boot, 2.5, axis=0),
                np.percentile(tpr_boot, 97.5, axis=0),
                alpha=0.15, color=C['red'], linewidth=0)

ax.plot(fpr_a4, tpr_a4, color=C['red'], linewidth=2.2)
ax.plot([0, 1], [0, 1], '--', color=C['grey'], linewidth=0.8, alpha=0.6)

ax.set_xlabel('1 − Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

ax.text(0.55, 0.15, f'AUC = {auc_a4:.3f}\n95% CI: 0.806–0.849',
        fontsize=11, fontweight='bold', color=C['red'],
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C['red'], alpha=0.9, linewidth=1.2))

fig.savefig(f'{OUTDIR}/fig4a_a4_roc.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig4a_a4_roc.pdf', transparent=True)
plt.show()


# %% Figure 4B — Centiloid Scatter
fig, ax = plt.subplots(figsize=(5, 4.5))

cl = a4['centiloid'].values
valid = ~np.isnan(cl) & ~np.isnan(p_a4)
cl_v = cl[valid]
p_v = p_a4[valid]
y_v = y_a4[valid]

# Split by amyloid status for coloring
neg_mask = y_v == 0
pos_mask = y_v == 1

ax.scatter(p_v[neg_mask], cl_v[neg_mask], s=8, alpha=0.35,
           color=C['green'], edgecolors='none', label='Aβ−', rasterized=True)
ax.scatter(p_v[pos_mask], cl_v[pos_mask], s=8, alpha=0.35,
           color=C['red'], edgecolors='none', label='Aβ+', rasterized=True)

# Reference lines
ax.axhline(20, color='#999999', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(0.5, color='#999999', linewidth=0.8, linestyle='--', alpha=0.5)

# Fit line
z = np.polyfit(p_v, cl_v, 1)
x_fit = np.linspace(0, 1, 100)
ax.plot(x_fit, np.polyval(z, x_fit), color='#333333', linewidth=1.5, linestyle='-')

from scipy.stats import spearmanr
r, pval = spearmanr(p_v, cl_v)

ax.set_xlabel('GRAD Predicted Probability', fontsize=10.5)
ax.set_ylabel('Centiloid', fontsize=10.5)
ax.set_xlim(-0.05, 1.05)

ax.text(0.05, 0.92, f'Spearman r = {r:.3f}\np < 0.001',
        fontsize=10, fontweight='bold', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.9,
          markerscale=2.5, edgecolor='#cccccc')

fig.savefig(f'{OUTDIR}/fig4b_centiloid_scatter.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig4b_centiloid_scatter.pdf', transparent=True)
plt.show()


# %% Figure 4C — A4 Calibration Curve
fig, ax = plt.subplots(figsize=(4.5, 4.5))

prob_true, prob_pred = calibration_curve(y_a4, p_a4, n_bins=10, strategy='uniform')

ax.plot(prob_pred, prob_true, 's-', color=C['red'], linewidth=2, markersize=7,
        markeredgecolor='white', markeredgewidth=1, zorder=5)
ax.plot([0, 1], [0, 1], '--', color=C['grey'], linewidth=0.8, alpha=0.6)

# Error bars via bootstrap
boot_trues = []
for _ in range(500):
    idx = rng.choice(len(y_a4), len(y_a4), replace=True)
    try:
        pt, pp = calibration_curve(y_a4[idx], p_a4[idx], n_bins=10, strategy='uniform')
        # Interpolate to same x-positions
        pt_interp = np.interp(prob_pred, pp, pt, left=np.nan, right=np.nan)
        boot_trues.append(pt_interp)
    except:
        pass
boot_trues = np.array(boot_trues)
ci_lo = np.nanpercentile(boot_trues, 5, axis=0)
ci_hi = np.nanpercentile(boot_trues, 95, axis=0)
valid_ci = ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
ax.fill_between(prob_pred[valid_ci], ci_lo[valid_ci], ci_hi[valid_ci],
                alpha=0.12, color=C['red'], linewidth=0)

# Prediction density histogram (bottom)
ax_hist = ax.inset_axes([0, 0, 1, 0.08], transform=ax.transAxes)
ax_hist.hist(p_a4[y_a4 == 0], bins=30, alpha=0.5, color=C['green'], density=True)
ax_hist.hist(p_a4[y_a4 == 1], bins=30, alpha=0.5, color=C['red'], density=True)
ax_hist.set_xlim(0, 1)
ax_hist.axis('off')

from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_a4, p_a4)

ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Fraction Aβ+')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

ax.text(0.05, 0.90, f'Brier = {brier:.3f}',
        fontsize=10, fontweight='bold', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

fig.savefig(f'{OUTDIR}/fig4c_a4_calibration.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig4c_a4_calibration.pdf', transparent=True)
plt.show()


# %% [markdown]
# # Figure 5: MRI Enhancement (2 panels)

# %% Figure 5B — Hippocampal Tertile Bar Chart
# (5A requires MRI-specific predictions; generate from run_a4_binary_validation
#  or use the mri_enhancement_results.csv if available)

fig, ax = plt.subplots(figsize=(4.5, 4))

tertiles = ['Small', 'Medium', 'Large']
rates = [74.7, 57.5, 30.2]
ptau_labels = ['0.170', '0.157', '0.146']

bars = ax.bar(tertiles, rates, width=0.55,
              color=[C['red'], C['gold'], C['green']],
              edgecolor='white', linewidth=1.5)

# Value labels
for bar, rate, ptau in zip(bars, rates, ptau_labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
            f'p-Tau217\n{ptau} pg/mL', ha='center', fontsize=8,
            color='white', fontweight='bold')

ax.set_ylabel('Aβ Positivity Rate (%)', fontsize=10.5)
ax.set_xlabel('Hippocampal Volume Tertile', fontsize=10.5)
ax.set_ylim(0, 90)

# Fold-change annotation
ax.annotate('', xy=(0, 76), xytext=(2, 76),
            arrowprops=dict(arrowstyle='<->', color=C['grey'], lw=1.5))
ax.text(1, 79, '2.5× variation', ha='center', fontsize=9, color=C['grey'],
        fontweight='bold')

fig.savefig(f'{OUTDIR}/fig5b_hippocampal_tertiles.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig5b_hippocampal_tertiles.pdf', transparent=True)
plt.show()


# %% [markdown]
# # Figure 6: Cost Simulation

# %%
fig, ax = plt.subplots(figsize=(7, 5.2))

strategies = ['Universal\nPET', 'p-Tau217 +\nPET (GZ)', 'GRAD\nStaged', 'GRAD +\nMRI']
# Differentiated plasma costs: p-tau217 alone $350 (CMS 2024 CLFS);
# full GRAD panel (p-tau217 + GFAP + Abeta42/40) $600; PET $3,000
costs = [30.0, 16.8, 10.0, 8.7]  # in millions
pet_scans = [10000, 4440, 1331, 887]

# Color scheme consistent with manuscript palette (Nature-style muted tones)
bar_colors = [C['red'], C['gold'], C['blue'], C['green']]

bars = ax.bar(strategies, costs, width=0.55, color=bar_colors,
              edgecolor='white', linewidth=1.5)

# Cost labels above bars
for bar, cost in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f'${cost:.1f}M', ha='center', fontsize=11, fontweight='bold',
            color='#1a1a1a')

# PET scan counts below x-axis labels
for bar, n_pet in zip(bars, pet_scans):
    ax.text(bar.get_x() + bar.get_width() / 2, -5.5,
            f'{n_pet:,} PET scans', ha='center', fontsize=8.5,
            color=C['grey'], clip_on=False)

ax.set_ylabel('Total Cost ($ millions)', fontsize=10.5)
ax.set_ylim(0, 37)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0fM'))

# Title — bolded black
ax.set_title('Projected 10,000-patient cohort', fontsize=12,
             fontweight='bold', color='black', pad=12)

fig.subplots_adjust(bottom=0.2)

fig.savefig(f'{OUTDIR}/fig6_cost_simulation.png', transparent=True)
fig.savefig(f'{OUTDIR}/fig6_cost_simulation.pdf', transparent=True)
plt.show()


# %% [markdown]
# # Supplementary Figures (moved from old Figure 3)

# %% Supp: Calibration Curve (ADNI LOOCV)
fig, ax = plt.subplots(figsize=(4.5, 4.5))

calib = pd.read_csv(f'{RESULTS}/tables/calibration_analysis.csv')

ax.plot(calib['mean_predicted'], calib['observed_fraction'], 's-',
        color=C['blue'], linewidth=2, markersize=7,
        markeredgecolor='white', markeredgewidth=1, zorder=5)
ax.plot([0, 1], [0, 1], '--', color=C['grey'], linewidth=0.8, alpha=0.6)

# Histogram inset
ax_hist = ax.inset_axes([0, 0, 1, 0.08], transform=ax.transAxes)
ax_hist.hist(p_adni[y_adni == 0], bins=30, alpha=0.5, color=C['green'], density=True)
ax_hist.hist(p_adni[y_adni == 1], bins=30, alpha=0.5, color=C['red'], density=True)
ax_hist.set_xlim(0, 1)
ax_hist.axis('off')

ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Fraction Aβ+')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
ax.text(0.05, 0.90, 'ECE = 0.060', fontsize=10, fontweight='bold',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

fig.savefig('supp_calibration_adni.png', transparent=True)
fig.savefig('supp_calibration_adni.pdf', transparent=True)
plt.show()


# %% Supp: Bootstrap AUC Distribution
fig, ax = plt.subplots(figsize=(5, 3.5))

boot_aucs = []
for _ in range(2000):
    idx = rng.choice(len(y_adni), len(y_adni), replace=True)
    if len(np.unique(y_adni[idx])) > 1:
        boot_aucs.append(roc_auc_score(y_adni[idx], p_adni[idx]))
boot_aucs = np.array(boot_aucs)

ax.hist(boot_aucs, bins=50, color=C['ltblue'], edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(boot_aucs), color=C['red'], linewidth=2, label=f'Mean = {np.mean(boot_aucs):.3f}')
ax.axvline(np.percentile(boot_aucs, 2.5), color=C['grey'], linewidth=1, linestyle='--')
ax.axvline(np.percentile(boot_aucs, 97.5), color=C['grey'], linewidth=1, linestyle='--')

ax.set_xlabel('AUC', fontsize=10.5)
ax.set_ylabel('Count', fontsize=10.5)
ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor='#cccccc')

lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
ax.text(0.95, 0.85, f'95% CI: [{lo:.3f}–{hi:.3f}]',
        fontsize=9, ha='right', transform=ax.transAxes, color=C['grey'])

fig.savefig('supp_bootstrap_auc.png', transparent=True)
fig.savefig('supp_bootstrap_auc.pdf', transparent=True)
plt.show()


# %% Supp: Per-Stage Confusion Matrices (Gatekeeper + Reflex side by side)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

for ax_idx, (mask_val, title, color) in enumerate([
    ('gatekeeper', 'Stage 1: Gatekeeper (N=178)', C['green']),
    ('reflex', 'Stage 2: Reflex (N=142)', C['orange'])
]):
    ax = axes[ax_idx]
    m = stage == mask_val
    y_s = y_adni[m]
    if mask_val == 'gatekeeper':
        p_s = adni.loc[m, 'gatekeeper_prob'].values
    else:
        p_s = p_adni[m]
    pred_s = (p_s >= 0.5).astype(int)
    cm_s = confusion_matrix(y_s, pred_s)

    labels = np.array([['TN', 'FP'], ['FN', 'TP']])
    for i in range(2):
        for j in range(2):
            val = cm_s[i, j]
            lbl = labels[i, j]
            fcolor = C['ltblue'] if lbl in ['TN', 'TP'] else C['ltred']
            rect = FancyBboxPatch((j - 0.42, i - 0.42), 0.84, 0.84,
                                   boxstyle='round,pad=0.05',
                                   facecolor=fcolor,
                                   edgecolor='#999999', linewidth=0.8)
            ax.add_patch(rect)
            ax.text(j, i + 0.05, f'{val}', ha='center', va='center',
                    fontsize=18, fontweight='bold')
            ax.text(j, i - 0.25, lbl, ha='center', va='center',
                    fontsize=8, color='#666666')

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pred Aβ−', 'Pred Aβ+'], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['True Aβ−', 'True Aβ+'], fontsize=9)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)

    tp_s = cm_s[1, 1]
    acc_s = (cm_s[0, 0] + cm_s[1, 1]) / cm_s.sum()
    ax.text(0.5, -0.52, f'Accuracy: {acc_s:.1%}', ha='center', fontsize=9,
            color=C['grey'], transform=ax.transData)

plt.tight_layout()
fig.savefig('supp_perstage_confusion.png', transparent=True)
fig.savefig('supp_perstage_confusion.pdf', transparent=True)
plt.show()


# %% Supp: Threshold Sensitivity Heatmap
fig, ax = plt.subplots(figsize=(5.5, 4.5))

thresh_df = load_threshold_sweep()

# Pivot for heatmap
pivot_res = thresh_df.pivot(index='Low_Threshold', columns='High_Threshold',
                             values='Resolution_Rate')
pivot_acc = thresh_df.pivot(index='Low_Threshold', columns='High_Threshold',
                             values='Resolved_Accuracy')

import matplotlib.colors as mcolors
from _grad_paths import (RESULTS, ADNI_DIR, A4_DIR, DATA_DIR, PROJECT_ROOT, SYNTHETIC, load_threshold_sweep)  # noqa: F401

im = ax.imshow(pivot_res.values, cmap='YlOrRd', aspect='auto',
               vmin=0.2, vmax=0.9)

# Annotate
for i in range(pivot_res.shape[0]):
    for j in range(pivot_res.shape[1]):
        res = pivot_res.values[i, j]
        acc = pivot_acc.values[i, j]
        text_color = 'white' if res > 0.65 else 'black'
        ax.text(j, i, f'{res:.0%}\n{acc:.0%}',
                ha='center', va='center', fontsize=7.5, color=text_color)

# Highlight selected 0.25/0.75
sel_i = list(pivot_res.index).index(0.25)
sel_j = list(pivot_res.columns).index(0.75)
rect = plt.Rectangle((sel_j - 0.5, sel_i - 0.5), 1, 1,
                       linewidth=2.5, edgecolor='gold', facecolor='none')
ax.add_patch(rect)

ax.set_xticks(range(len(pivot_res.columns)))
ax.set_xticklabels([f'{x:.2f}' for x in pivot_res.columns], fontsize=9)
ax.set_yticks(range(len(pivot_res.index)))
ax.set_yticklabels([f'{x:.2f}' for x in pivot_res.index], fontsize=9)
ax.set_xlabel('Upper Threshold (Rule-In)', fontsize=10)
ax.set_ylabel('Lower Threshold (Rule-Out)', fontsize=10)

cb = fig.colorbar(im, ax=ax, shrink=0.8, label='Resolution Rate')

fig.savefig('supp_threshold_heatmap.png', transparent=True)
fig.savefig('supp_threshold_heatmap.pdf', transparent=True)
plt.show()

print("\n✓ All figures generated.")
