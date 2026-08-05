#!/usr/bin/env python3
"""
Figure 1 panel — gray-zone burden across populations, one common cutoff rule.

Three cohorts, each split three ways by plasma p-tau217 under the SAME rule
(cutoffs at 95% sensitivity and 95% specificity, per cohort):

    ruled out  |  gray zone  |  ruled in

Using one rule everywhere is what makes the panels comparable. Note this is NOT
GRAD's own routing, which uses 0.25/0.75 Gatekeeper probabilities and gives a
slightly different split (e.g. ADNI CN 49% rather than 53%); that belongs in the
pipeline panel, not here.

Giacomucci et al. values are PUBLISHED (Alzheimers Dement Amst 2026;18:e70285,
Section 3.2) and drawn with hatching to mark them as external:
    SCD          58.1% below | 22.6% gray | 19.4% above
    MCI          38.8% below | 21.6% gray | 39.7% above
    AD dementia   0.0% below |  6.8% gray | 93.2% above

Caveats for the legend: their "SCD" is a help-seeking clinic population rather
than research-volunteer CN, and their AD-dementia group is Core1+ by definition,
so its small gray zone is partly definitional.

Output: results/figures/panels/panel_fig1_grayzone_stacked.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from _grad_paths import PANELS, TABLES
from _grad_style import apply_style, save_panel, INK, ABETA_NEG, ABETA_POS
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features

apply_style(base=9)
GRAY = '#D9C77E'          # indeterminate, same amber used throughout the set
LW = 0.8


def split_9595(d):
    """Fractions below / between / above the 95%-sens and 95%-spec cutoffs."""
    y = d.amyloid_positive.values.astype(int)
    z = d.pTau217_Z.values
    ok = ~np.isnan(z)
    fpr, tpr, thr = roc_curve(y[ok], z[ok])
    lo = thr[np.where(tpr >= .95)[0][0]]
    hi = thr[np.where((1 - fpr) >= .95)[0][-1]]
    return lo, hi


adni, a4 = load_adni(), load_a4()
ah, bh = harmonize(adni, a4)
ah = engineer_features(ah).reset_index(drop=True)
bh = engineer_features(bh).reset_index(drop=True)
lo_a, hi_a = split_9595(ah)
lo_b, hi_b = split_9595(bh)


def frac(d, lo, hi):
    z = d.pTau217_Z.values
    return (z < lo).mean(), ((z >= lo) & (z <= hi)).mean(), (z > hi).mean()


PANELS_SPEC = [
    ('ADNI', False, [
        ('CN',       frac(ah[ah.DX.isin(['CN', 'NL', 'Normal'])], lo_a, hi_a), 175),
        ('MCI',      frac(ah[ah.DX.isin(['MCI', 'EMCI', 'LMCI'])], lo_a, hi_a), 117),
        ('Dementia', frac(ah[ah.DX.isin(['Dementia', 'AD'])], lo_a, hi_a), 28),
    ]),
    ('A4 + LEARN', False, [
        ('CU', frac(bh, lo_b, hi_b), 1644),
    ]),
    ('Giacomucci et al.', True, [
        ('SCD',      (.5806, .2258, .1935), 124),
        ('MCI',      (.3879, .2155, .3965), 188),
        ('Dementia', (.0000, .0677, .9322), 136),
    ]),
]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2),
                         gridspec_kw={'width_ratios': [3, 1.15, 3]})
rows = []
for ax, (title, published, bars) in zip(axes, PANELS_SPEC):
    for i, (grp, (neg, gz, pos), n) in enumerate(bars):
        left = 0
        for val, colour in ((neg, ABETA_NEG), (gz, GRAY), (pos, ABETA_POS)):
            ax.barh(i, val * 100, left=left * 100, height=.62, color=colour,
                    edgecolor=INK, lw=LW, zorder=3,
                    hatch='///' if published else None)
            left += val
        # only the gray-zone share is labelled; it is the quantity of interest
        if gz > .05:
            ax.text((neg + gz / 2) * 100, i, f'{gz:.0%}', ha='center',
                    va='center', fontsize=7, color=INK, zorder=4)
        rows.append(dict(cohort=title, group=grp, n=n, ruled_out=neg,
                         gray_zone=gz, ruled_in=pos,
                         source='published' if published else 'this study'))
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([f'{g}\n({n:,})' for g, _, n in bars], fontsize=6.8,
                       linespacing=1.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.set_ylim(len(bars) - .45, -.55)
    ax.set_title(title, fontsize=8.5, pad=4, loc='left')
    ax.set_xlabel('Participants (%)')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=INK, lw=LW)
           for c in (ABETA_NEG, GRAY, ABETA_POS)]
fig.legend(handles, ['Ruled out (Aβ−)', 'Gray zone', 'Ruled in (Aβ+)'],
           loc='lower center', ncol=3, frameon=False, fontsize=7.2,
           handlelength=1.1)
fig.tight_layout(pad=.35)
fig.subplots_adjust(bottom=.32, wspace=.42)
save_panel(fig, PANELS / 'panel_fig1_grayzone_stacked')

t = pd.DataFrame(rows)
t.to_csv(TABLES / 'grayzone_stacked_by_population.csv', index=False)
print(t.to_string(index=False))
print('  panel_fig1_grayzone_stacked')
