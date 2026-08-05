"""
New figure panels for the GRAD manuscript revision.

  panel_grayzone_by_dx      -> Figure 1: where the gray zone concentrates
  panel_ptau_vs_grad        -> Figure 2: what Stage 2 adds over p-Tau217 alone
  panel_a4_reflex_roc       -> Figure 3: Stage 2 performance in the external cohort

Outputs to results/figures/panels/ as PNG + PDF.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from _grad_paths import PANELS, ADNI_DIR, RESULTS          # noqa: F401
from _grad_style import (apply_style, save_panel, ABETA_NEG, ABETA_POS,
                         INK, GUIDE, GRID, NAVY, AMBER)
from data_loader import ADNIDataLoader

apply_style(base=8.5)
# Abeta-encoded series use the frozen pair; the gray zone is a neutral amber
# so it never competes with the Abeta+/- reading.
NEG, GRAY, POS = ABETA_NEG, '#D9C77E', ABETA_POS
# method comparison (p-Tau217 alone vs GRAD) is NOT an Abeta contrast, so it
# uses the accent ramp instead
BLUE, ORANGE = NAVY, AMBER
LOW, HIGH = 0.25, 0.75

def save(fig, name):
    save_panel(fig, str(PANELS / name))
    print(f'  {name}')

# ---------------------------------------------------------------- load
df = ADNIDataLoader(str(ADNI_DIR)).merge_data(use_baseline_only=True).reset_index(drop=True)
p = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
df['stage'] = p.stage.values; df['gk'] = p.gatekeeper_prob.values
df['final'] = p.predicted_prob.values
df['route'] = np.where(df.stage != 'reflex',
                       np.where(df.gk < LOW, 'ruled out', 'ruled in'), 'gray zone')

# ---------------------------------------------- 1. gray zone by diagnosis
order = ['CN', 'MCI', 'Dementia']
ct = pd.crosstab(df.DX, df.route).reindex(order)[['ruled out', 'gray zone', 'ruled in']]
pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(3.4, 2.1))
left = np.zeros(len(order))
for col, c in zip(['ruled out', 'gray zone', 'ruled in'], [NEG, GRAY, POS]):
    ax.barh(order, pct[col], left=left, color=c, height=.62,
            edgecolor='white', linewidth=.8,
            label={'ruled out': 'Stage 1: Aβ−', 'gray zone': 'Gray zone (to Stage 2)',
                   'ruled in': 'Stage 1: Aβ+'}[col])
    for i, (v, l) in enumerate(zip(pct[col], left)):
        if v > 7:
            ax.text(l + v/2, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')
    left += pct[col].values
ax.set_xlim(0, 100); ax.set_xlabel('Participants (%)')
ax.set_yticklabels([f'{d}\n(n={int(ct.loc[d].sum())})' for d in order], fontsize=7.5)
ax.invert_yaxis()
ax.legend(fontsize=6.5, frameon=False, ncol=1, loc='center left',
          bbox_to_anchor=(1.01, .5), handlelength=1.1)
ax.set_title('Routing by cognitive status', fontsize=8.5, pad=6, loc='left')
save(fig, 'panel_grayzone_by_dx')
print(ct.assign(**{'gray %': pct['gray zone'].round(1)}).to_string())

# ------------------------------------- 2. p-Tau217 alone vs GRAD (gray zone)
gz = df[df.stage == 'reflex']; y = gz.amyloid_positive.values
def mets(v):
    pr = v >= .5
    return dict(Sensitivity=(pr & (y == 1)).sum()/(y == 1).sum(),
                Specificity=((~pr) & (y == 0)).sum()/(y == 0).sum(),
                Accuracy=(pr == y).mean())
m1, m2 = mets(gz.gk.values), mets(gz.final.values)
labels = ['Sensitivity', 'Specificity', 'Accuracy']
x = np.arange(3); w = .36
fig, ax = plt.subplots(figsize=(3.2, 2.2))
b1 = ax.bar(x - w/2, [m1[k]*100 for k in labels], w, color='#B9C6D6',
            edgecolor=INK, linewidth=.5, label='p-Tau217 alone')
b2 = ax.bar(x + w/2, [m2[k]*100 for k in labels], w, color=BLUE,
            edgecolor=INK, linewidth=.5, label='GRAD Stage 2')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
            f'{b.get_height():.1f}', ha='center', fontsize=6.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel('%'); ax.set_ylim(0, 100)
ax.legend(fontsize=6.8, frameon=False, loc='upper center',
          bbox_to_anchor=(.5, 1.22), ncol=2, handlelength=1.1)
ax.set_title(f'Within the gray zone (n={len(gz)})', fontsize=8.5, pad=22, loc='left')
ax.grid(axis='y', lw=.4, color=GRID); ax.set_axisbelow(True)
save(fig, 'panel_ptau_vs_grad')
print(f"  p-Tau217 alone : sens {m1['Sensitivity']:.1%} spec {m1['Specificity']:.1%} acc {m1['Accuracy']:.1%}  AUC {roc_auc_score(y, gz.gk):.3f}")
print(f"  GRAD Stage 2   : sens {m2['Sensitivity']:.1%} spec {m2['Specificity']:.1%} acc {m2['Accuracy']:.1%}  AUC {roc_auc_score(y, gz.final):.3f}")

# ------------------------------------------------ 3. A4 Reflex ROC
a4 = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
rf = a4[a4.stage == 'reflex']
fpr, tpr, _ = roc_curve(rf.true_amyloid, rf.predicted_prob)
auc = roc_auc_score(rf.true_amyloid, rf.predicted_prob)
rng = np.random.default_rng(42); grid = np.linspace(0, 1, 101); boot = []
yv, sv = rf.true_amyloid.values, rf.predicted_prob.values
for _ in range(500):
    i = rng.integers(0, len(yv), len(yv))
    if len(np.unique(yv[i])) > 1:
        f, t, _ = roc_curve(yv[i], sv[i]); boot.append(np.interp(grid, f, t))
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.fill_between(grid, lo, hi, color=ORANGE, alpha=.18, lw=0)
ax.plot(fpr, tpr, color=ORANGE, lw=1.6)
ax.plot([0, 1], [0, 1], ls='--', lw=.7, color='#9A9A9A')
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
ax.text(.97, .06, f'AUC = {auc:.3f}\nn = {len(rf):,}', ha='right', fontsize=7,
        bbox=dict(boxstyle='round,pad=.35', fc='white', ec='#C8C8C8', lw=.5))
ax.set_title('Stage 2 — A4 + LEARN', fontsize=8.5, pad=6, loc='left')
save(fig, 'panel_a4_reflex_roc')
print(f"  A4 Reflex AUC {auc:.4f} (n={len(rf)})   ADNI Reflex AUC 0.751 (n=142)")
