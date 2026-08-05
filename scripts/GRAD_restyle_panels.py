"""
Unified restyle of the ROC panels + rework of the two panels flagged for
aesthetics (hippocampal tertiles, centiloid scatter).

All ROC panels now share one treatment: same figure size, same axis style,
light bootstrap band (alpha .18) so the curve stays legible at n=142, and an
identical annotation box carrying AUC, 95% CI and n. Stage is encoded by hue
only; everything else is constant.

Outputs -> results/figures/panels/  (prefix rs_)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from _grad_paths import PANELS, RESULTS                       # noqa: F401
from _grad_style import (apply_style, save_panel, ABETA_NEG, ABETA_POS,
                         INK, GUIDE, GRID, ROC_BLUE, NAVY, TEAL, AMBER, CRIMSON, GREEN)

apply_style(base=9)
BLUE, CORAL = ABETA_NEG, ABETA_POS

def save(fig, name):
    save_panel(fig, str(PANELS / name)); print(f'  {name}')

def roc_panel(y, s, colour, title, name, n_boot=600):
    y, s = np.asarray(y), np.asarray(s)
    fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
    rng = np.random.default_rng(42); grid = np.linspace(0, 1, 101); boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            f, t, _ = roc_curve(y[i], s[i]); boot.append(np.interp(grid, f, t))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    aucs = np.percentile([roc_auc_score(y[i], s[i]) for i in
                          (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
                          if len(np.unique(y[i])) > 1] or [auc], [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(2.45, 2.45))
    # chance line: thin solid black, drawn under the curve
    ax.plot([0, 1], [0, 1], lw=.6, color=INK, zorder=1)
    ax.fill_between(grid, lo, hi, color=ROC_BLUE, alpha=.16, lw=0, zorder=2)
    ax.plot(fpr, tpr, color=ROC_BLUE, lw=1.7, solid_capstyle='round', zorder=3)
    ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_title(title, fontsize=9, pad=5, loc='left')
    # unboxed, black annotation
    ax.text(.97, .04, f'AUC {auc:.3f}\n95% CI {aucs[0]:.3f}–{aucs[1]:.3f}\nn = {len(y):,}',
            ha='right', va='bottom', fontsize=7, linespacing=1.5, color=INK)
    save(fig, name)

adni = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
a4 = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
gk = adni[adni.stage != 'reflex']; gz = adni[adni.stage == 'reflex']
a4r = a4[a4.stage == 'reflex']
a4gk = a4[a4.stage != 'reflex']          # Stage-1-resolved, mirrors ADNI's gk

print('ROC panels (unified treatment):')
roc_panel(adni.true_amyloid, adni.predicted_prob, ROC_BLUE,  'Full GRAD pipeline',      'rs_roc_overall')
roc_panel(gk.true_amyloid,   gk.gatekeeper_prob,  ROC_BLUE,  'Stage 1 — Gatekeeper',    'rs_roc_gatekeeper')
roc_panel(gz.true_amyloid,   gz.predicted_prob,   ROC_BLUE, 'Stage 2 — Reflex',        'rs_roc_reflex')
roc_panel(a4.true_amyloid,   a4.predicted_prob,   ROC_BLUE, 'Full GRAD pipeline',      'rs_roc_a4_overall')
roc_panel(a4gk.true_amyloid, a4gk.gatekeeper_prob, ROC_BLUE, 'Stage 1 — Gatekeeper',    'rs_roc_a4_gatekeeper')
roc_panel(a4r.true_amyloid,  a4r.predicted_prob,  ROC_BLUE, 'Stage 2 — Reflex',        'rs_roc_a4_reflex')

# ---------------------------------------- hippocampal tertiles (reworked)
h = pd.read_csv(RESULTS / 'hippocampal_stratification.csv')
print('\nhippocampal stratification source:'); print(h.to_string(index=False))
# select by NAME -- positional indexing grabbed N instead of the rate
rate = h['Amyloid_Positive_Rate'].values * 100
ptau = h['Mean_pTau217'].values
nper = h['N'].values
labels = list(h['Tertile'].values)
fig, ax = plt.subplots(figsize=(3.1, 2.5))
bars = ax.bar(labels, rate, width=.62, color=[CORAL, '#E5C77C', GREEN],
              edgecolor='white', linewidth=.8, zorder=3)
for b, r in zip(bars, rate):
    ax.text(b.get_x()+b.get_width()/2, r+1.8, f'{r:.1f}%', ha='center',
            fontsize=9, fontweight='bold')
ax.set_ylim(0, 92); ax.set_ylabel('Aβ positivity (%)')
ax.set_xlabel('Hippocampal volume tertile (ICV-normalised)', labelpad=8)
ax.grid(axis='y', lw=.4, color=GRID, zorder=0); ax.set_axisbelow(True)
# p-tau217 as a quiet second row beneath the axis, not inside the bars
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f'{l}\n(n={int(n)})\np-Tau217 {p:.3f}'
                    for l, n, p in zip(labels, nper, ptau)], fontsize=7)
ax.annotate('', xy=(1.98, 82), xytext=(0.02, 82),
            arrowprops=dict(arrowstyle='<->', color=GUIDE, lw=.8))
ax.text(1, 84.5, '2.5-fold difference', ha='center', fontsize=7.2, color=GUIDE)
fig.subplots_adjust(bottom=.30)
save(fig, 'rs_hippocampal_tertiles')

# ---------------------------------------- centiloid scatter (reworked)
# Continuous amyloid burden against the GRAD probability: shows the output is a
# graded biological readout, not just a thresholded label.
d = a4.dropna(subset=['centiloid', 'predicted_prob'])
from scipy.stats import spearmanr
rho, pv = spearmanr(d.predicted_prob, d.centiloid)
fig, ax = plt.subplots(figsize=(3.0, 2.5))

# CL = 20 positivity cut, drawn as a thin solid black rule under the points
ax.axhline(20, lw=.6, color=INK, zorder=1)
for lab, m, c in ((0, d.true_amyloid == 0, BLUE), (1, d.true_amyloid == 1, CORAL)):
    ax.scatter(d.predicted_prob[m], d.centiloid[m], s=4.5, color=c, alpha=.30,
               linewidths=0, rasterized=True, zorder=2)
z = np.polyfit(d.predicted_prob, d.centiloid, 1)
xs = np.linspace(d.predicted_prob.min(), d.predicted_prob.max(), 50)
ax.plot(xs, np.polyval(z, xs), color=INK, lw=1.2, zorder=3)

ax.set_xlabel('GRAD predicted probability'); ax.set_ylabel('Amyloid PET (Centiloid)')
ax.set_xlim(0, 1.02)
# unboxed, black annotation
ax.text(.03, .97, f'Spearman ρ = {rho:.3f}\nP < .001', transform=ax.transAxes,
        va='top', ha='left', fontsize=7, linespacing=1.5, color=INK)
ax.text(1.0, 23, 'CL = 20', ha='right', va='bottom', fontsize=6.3, color=INK)

# square swatches, black label text -- no coloured type anywhere in the set
handles = [plt.Line2D([], [], marker='s', ls='', ms=4.2, color=c)
           for c in (BLUE, CORAL)]
ax.legend(handles, ['Aβ−', 'Aβ+'], fontsize=6.8, frameon=False,
          loc='lower right', handletextpad=.4, labelcolor=INK,
          borderpad=.2, labelspacing=.3)
save(fig, 'rs_centiloid_scatter')
