"""
Figure 1B — Stage 1 probability by true amyloid status, ADNI and A4.

Styling is inherited from _grad_style (frozen). PROB_SOURCE selects the
probability plotted:
  'gatekeeper'  Stage 1 output. The 25/75 bands are Gatekeeper routing
                thresholds, so this is the distribution they describe.
  'final'       pooled two-stage output; gray-zone cases have been rescored by
                Stage 2, so the bands no longer match the routing shown.

Participant counts are deliberately omitted from the panel -- they belong in
the caption. Output: results/figures/panels/fig1b_strip_probability.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from _grad_paths import PANELS, RESULTS                       # noqa: F401
from _grad_style import (apply_style, save_panel, ABETA_NEG, ABETA_POS,
                         BAND, GUIDE, INK)

PROB_SOURCE = 'gatekeeper'
LOW, HIGH = 25, 75
apply_style(base=9)

adni = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
a4 = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
col = 'gatekeeper_prob' if PROB_SOURCE == 'gatekeeper' else 'predicted_prob'
a4 = a4[a4[col].notna()]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.7), sharey=True)
rng = np.random.default_rng(7)
for ax, (title, d) in zip(axes, [('ADNI (Development)', adni),
                                 ('A4 + LEARN (Validation)', a4)]):
    prob = np.clip(d[col].values * 100, 0, 100)
    y = d.true_amyloid.values
    ax.axhspan(LOW, HIGH, color=BAND, zorder=0, lw=0)
    for t in (LOW, HIGH):
        ax.axhline(t, ls=(0, (2, 2)), lw=.7, color=GUIDE, zorder=1)
    for xc, grp, c in ((0, 0, ABETA_NEG), (1, 1, ABETA_POS)):
        v = prob[y == grp]
        ax.scatter(xc + rng.normal(0, .075, len(v)), v, s=7, color=c,
                   alpha=.55, linewidths=0, zorder=3, clip_on=False)
        ax.hlines(np.median(v), xc - .27, xc + .27, color=c, lw=3, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Aβ-PET\nnegative', 'Aβ-PET\npositive'])
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(-1.5, 101.5)          # headroom so markers at 0 / 100 render whole
    ax.set_title(title, pad=7)
    ax.tick_params(axis='x', length=0, pad=5)
    ax.text(1.46, (LOW + HIGH) / 2, 'Gray zone', ha='right', va='center',
            fontsize=7.5, color=GUIDE, rotation=90)
    ax.text(1.46, 94, 'Aβ+', ha='right', fontsize=8.5, color=INK)
    ax.text(1.46, 6, 'Aβ−', ha='right', fontsize=8.5, color=INK)

axes[0].set_ylabel('Predicted Aβ+ probability (%)')
axes[0].set_yticks([0, 25, 50, 75, 100])
fig.subplots_adjust(wspace=.10)
fig.tight_layout(pad=.4)
save_panel(fig, str(PANELS / 'fig1b_strip_probability'))
print(f"  fig1b_strip_probability  [{PROB_SOURCE}]")
for t, d in [('ADNI', adni), ('A4', a4)]:
    p = d[col].values * 100; y = d.true_amyloid.values
    print(f"    {t:5s} n={len(d):5d}  median Aβ− {np.median(p[y==0]):4.1f}%  "
          f"Aβ+ {np.median(p[y==1]):4.1f}%   at 100%: {int((p>=99.95).sum())}")
