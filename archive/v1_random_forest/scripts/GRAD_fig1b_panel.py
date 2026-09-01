"""
Figure 1B — where the gray zone falls, by true amyloid status.

Plots the STAGE 1 (Gatekeeper) probability, not the pooled two-stage output.
The 25%/75% bands are Gatekeeper routing thresholds, so the distribution they
are drawn on must be the Gatekeeper's. Using the final probability would show
gray-zone cases already rescored by Stage 2, and the bands would no longer
correspond to the routing they depict.

Output: results/figures/panels/panel_fig1b_gatekeeper.png / .pdf
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _grad_paths import PANELS, RESULTS                      # noqa: F401

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.linewidth': .7,
    'axes.edgecolor': '#3C3C42', 'text.color': '#3C3C42',
    'axes.labelcolor': '#3C3C42', 'xtick.color': '#3C3C42',
    'ytick.color': '#3C3C42', 'axes.spines.top': False, 'axes.spines.right': False,
})
BLUE, CORAL, BAND = '#6FA3DB', '#E8836F', '#EDEDED'
LOW, HIGH = 25, 75

adni = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
a4 = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
a4 = a4[a4.gatekeeper_prob.notna()]

cohorts = [('ADNI (Development)', adni, adni.gatekeeper_prob.values * 100),
           ('A4 + LEARN (Validation)', a4, a4.gatekeeper_prob.values * 100)]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
rng = np.random.default_rng(7)
for ax, (title, d, prob) in zip(axes, cohorts):
    y_true = d.true_amyloid.values
    ax.axhspan(LOW, HIGH, color=BAND, zorder=0)
    ax.axhline(LOW, ls=':', lw=.8, color='#8C8C8C', zorder=1)
    ax.axhline(HIGH, ls=':', lw=.8, color='#8C8C8C', zorder=1)
    for xc, grp, col in ((0, 0, BLUE), (1, 1, CORAL)):
        v = prob[y_true == grp]
        jitter = rng.normal(0, .085, len(v))
        ax.scatter(xc + jitter, v, s=9, color=col, alpha=.55,
                   linewidths=0, zorder=3)
        ax.hlines(np.median(v), xc - .28, xc + .28, color=col, lw=3.2, zorder=4)
    n_neg, n_pos = int((y_true == 0).sum()), int((y_true == 1).sum())
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Aβ-PET neg\n(n={n_neg:,})', f'Aβ-PET pos\n(n={n_pos:,})'])
    ax.set_xlim(-.55, 1.55); ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10, pad=8)
    ax.text(.03, .975, f'N = {len(d):,}', transform=ax.transAxes, va='top',
            fontsize=8, color='#6E6E6E',
            bbox=dict(boxstyle='round,pad=.28', fc='white', ec='#D6D6D6', lw=.5))
    gz = ((prob >= LOW) & (prob <= HIGH)).mean() * 100
    ax.text(1.51, (LOW + HIGH) / 2, f'Gray Zone\n{gz:.0f}%', ha='right', va='center',
            fontsize=8.5, color='#8C8C8C', style='italic')
    ax.text(1.51, 92, 'Aβ+', ha='right', fontsize=9, color=CORAL, style='italic')
    ax.text(1.51, 6, 'Aβ−', ha='right', fontsize=9, color=BLUE, style='italic')

axes[0].set_ylabel('Stage 1 (Gatekeeper) Aβ+ Probability (%)')
axes[0].set_yticks([0, 25, 50, 75, 100])
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(PANELS / f'panel_fig1b_gatekeeper.{ext}', dpi=400,
                bbox_inches='tight', facecolor='white')
print('  wrote panel_fig1b_gatekeeper.png / .pdf')
for title, d, prob in cohorts:
    y = d.true_amyloid.values
    print(f"  {title:26s} gray zone {((prob>=LOW)&(prob<=HIGH)).mean():5.1%} | "
          f"median Aβ− {np.median(prob[y==0]):4.1f}%  Aβ+ {np.median(prob[y==1]):4.1f}%")
