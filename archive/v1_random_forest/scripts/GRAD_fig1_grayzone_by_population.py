#!/usr/bin/env python3
"""
Figure 1 panel — the gray zone is concentrated in early disease.

Establishes the problem the paper addresses: under a fixed cutoff rule, the
proportion of patients p-tau217 cannot classify rises as you move earlier in
the disease continuum, which is exactly where prevention trials recruit and
where early treatment decisions are made.

Two independent datasets, the same 95%-sensitivity / 95%-specificity cutoff
rule, and the same monotone gradient:

    this study     ADNI CN 53.1% -> MCI 42.7% -> dementia 35.7%
    Giacomucci     SCD    22.6% -> MCI 21.6% -> dementia  6.8%

Our estimates sit higher throughout because p-tau217 separates less well in
these populations (AUC 0.87 in ADNI and A4 versus 0.95 in a cohort that is 34%
AD dementia). Under this rule the gray-zone fraction is a deterministic
function of that AUC -- simulation gives 52% at AUC 0.85 and 13% at 0.96 -- so
the size of the gray zone is a property of the population under test, not of
the threshold rule.

Giacomucci values are PUBLISHED figures (Alzheimers Dement Amst 2026;18:e70285),
drawn open to distinguish them from values computed here.

Output: results/figures/panels/panel_fig1_grayzone_by_population.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from _grad_paths import PANELS, TABLES, RESULTS
from _grad_style import apply_style, save_panel, INK
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features

apply_style(base=9)
GRAY = '#D9C77E'          # the amber that marks "indeterminate" throughout the set
LW = 0.9


def gz_9595(d):
    """Giacomucci's rule: cutoffs at 95% sensitivity and 95% specificity."""
    y = d.amyloid_positive.values.astype(int)
    z = d.pTau217_Z.values
    ok = ~np.isnan(z)
    fpr, tpr, thr = roc_curve(y[ok], z[ok])
    lo = thr[np.where(tpr >= .95)[0][0]]
    hi = thr[np.where((1 - fpr) >= .95)[0][-1]]
    return (z >= lo) & (z <= hi), roc_auc_score(y[ok], z[ok])


adni, a4 = load_adni(), load_a4()
ah, bh = harmonize(adni, a4)
ah = engineer_features(ah).reset_index(drop=True)
bh = engineer_features(bh).reset_index(drop=True)
ah['gz'], auc_a = gz_9595(ah)
bh['gz'], auc_b = gz_9595(bh)

# severity blocks, earliest disease on the left
BLOCKS = [
    ('Cognitively\nunimpaired', [
        ('ADNI',       ah[ah.DX.isin(['CN', 'NL', 'Normal'])].gz.mean(), 175, False),
        ('A4',         bh.gz.mean(),                                     1644, False),
        ('Giacomucci', .2258,                                             124, True),
    ]),
    ('MCI', [
        ('ADNI',         ah[ah.DX.isin(['MCI', 'EMCI', 'LMCI'])].gz.mean(), 117, False),
        ('Giacomucci',   .2155,                                             188, True),
    ]),
    ('Dementia', [
        ('ADNI',         ah[ah.DX.isin(['Dementia', 'AD'])].gz.mean(),       28, False),
        ('Giacomucci',   .0677,                                             136, True),
    ]),
]

fig, ax = plt.subplots(figsize=(4.5, 2.5))
x, ticks, labels, rows = 0., [], [], []
block_centres = []
for block, bars in BLOCKS:
    start = x
    for name, val, n, published in bars:
        # published values drawn open; values computed here drawn solid
        ax.bar(x, val * 100, width=.72, zorder=3,
               facecolor='white' if published else GRAY,
               edgecolor=INK, lw=LW, hatch='///' if published else None)
        ax.text(x, val * 100 + 1.4, f'{val:.0%}', ha='center', va='bottom',
                fontsize=6.8, color=INK)
        ticks.append(x); labels.append(name)
        rows.append(dict(block=block, series=name, gray_zone=val, n=n,
                         source='published' if published else 'this study'))
        x += 1
    block_centres.append(((start + x - 1) / 2, block))
    x += .8

ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=6.8)
ax.set_ylabel('Patients in the p-tau217\ngray zone (%)')
ax.set_ylim(0, 66)
ax.set_yticks(range(0, 61, 20))
ax.set_xlim(-.7, x - 1.1)
ax.tick_params(axis='x', length=0, pad=3)
ax.spines['bottom'].set_color(INK); ax.spines['bottom'].set_linewidth(.8)

# severity labels beneath the group, outside the data area
for cx, block in block_centres:
    ax.text(cx, -.13, block, transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=7.5, color=INK)

handles = [plt.Rectangle((0, 0), 1, 1, fc=GRAY, ec=INK, lw=LW),
           plt.Rectangle((0, 0), 1, 1, fc='white', ec=INK, lw=LW, hatch='///')]
ax.legend(handles, ['This study', 'Giacomucci et al. (published)'],
          loc='upper right', frameon=False, fontsize=6.8, handlelength=1.1)
fig.tight_layout(pad=.3)
fig.subplots_adjust(bottom=.22)
save_panel(fig, PANELS / 'panel_fig1_grayzone_by_population')

t = pd.DataFrame(rows)
t.to_csv(TABLES / 'grayzone_by_population.csv', index=False)
print(t.to_string(index=False))
print(f'\np-tau217 AUC: ADNI {auc_a:.3f}, A4 {auc_b:.3f}, Giacomucci 0.95 (published)')
print('  panel_fig1_grayzone_by_population')
