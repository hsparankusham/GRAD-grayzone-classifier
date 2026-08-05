#!/usr/bin/env python3
"""
Reclassification panel — does Stage 2 fix more than it breaks?

Two questions the AUC panel cannot answer, both in the field's own language:

  (a) MATCHED-BAND RESOLUTION. The 0.40-0.60 indeterminate band is applied to
      BOTH scores, not just Stage 2. Without this the claim "Stage 2 resolved
      76.8% of gray-zone cases" is unfalsifiable -- p-tau217 under the same band
      resolves cases too, and in ADNI it resolves slightly more.

  (b) PAIRED RECLASSIFICATION. Among gray-zone patients, how many does Stage 2
      move wrong -> right, versus right -> wrong? McNemar's test on the
      discordant pairs, which is the statistic Giacomucci et al. report.

Stage 2 here is the regularized logistic model (C = 0.1), scored leave-one-out
in ADNI and applied once to A4.

Output: results/figures/panels/panel_reclassification.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

from _grad_paths import PANELS, TABLES, RESULTS
from _grad_style import (apply_style, save_panel, INK, ABETA_NEG,
                         ABETA_POS, ROC_BLUE)
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features

apply_style(base=9)
GRAY = '#D9C77E'
LW = 0.9
LOCKED = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
          'APOE4_carrier', 'gfap_tau_interaction']
BAND = (.40, .60)


def build():
    adni, a4 = load_adni(), load_a4()
    ah, bh = harmonize(adni, a4)
    ah = engineer_features(ah).reset_index(drop=True)
    bh = engineer_features(bh).reset_index(drop=True)
    for df, f in ((ah, 'adni_loocv_predictions.csv'),
                  (bh, 'a4_binary_validation_predictions.csv')):
        p = pd.read_csv(RESULTS / f)
        df['stage'] = p['stage'].values
        df['gk'] = p['gatekeeper_prob'].values
        df['y'] = p['true_amyloid'].values
    tr, te = ah[ah.stage == 'reflex'], bh[bh.stage == 'reflex']

    X = tr[LOCKED].astype(float).values.copy()
    MED = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = MED[j]
    y = tr.y.values.astype(int)
    LOG = lambda: LogisticRegression(C=.1, max_iter=5000, class_weight='balanced')

    loo = np.empty(len(y))                       # ADNI scored out-of-fold
    for i in range(len(y)):
        k = np.ones(len(y), bool); k[i] = False
        sc = StandardScaler().fit(X[k])
        loo[i] = LOG().fit(sc.transform(X[k]), y[k]) \
                      .predict_proba(sc.transform(X[i:i + 1]))[0, 1]

    Z = te[LOCKED].astype(float).values.copy()
    for j in range(Z.shape[1]):
        Z[np.isnan(Z[:, j]), j] = MED[j]         # training medians
    sc = StandardScaler().fit(X)
    pte = LOG().fit(sc.transform(X), y).predict_proba(sc.transform(Z))[:, 1]
    return [('ADNI', y, tr.gk.values, loo),
            ('A4 + LEARN', te.y.values.astype(int), te.gk.values, pte)]


COHORTS = build()
rows = []
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5),
                         gridspec_kw={'width_ratios': [1, 1.05]})

# ---- (a) outcome of every gray-zone patient, same band on both scores ------
# Resolution rate alone is not the clinically meaningful quantity: a score that
# resolves many cases badly is worse than one that resolves fewer well. What
# matters is how many patients get a CORRECT answer without a scan, so each bar
# is split into resolved-correct / resolved-wrong / still needs PET.
ax = axes[0]
xs, ticks, labels = 0., [], []
centres = []
for nm, y, pa, pb in COHORTS:
    start = xs
    for lab, p in [('p-tau217', pa), ('Stage 2', pb)]:
        res = (p < BAND[0]) | (p > BAND[1])
        correct = res & (((p >= .5).astype(int)) == y)
        wrong = res & ~correct
        f_ok, f_bad, f_pet = correct.mean(), wrong.mean(), (~res).mean()
        bottom = 0
        for val, colour in ((f_ok, ROC_BLUE), (f_bad, ABETA_POS), (f_pet, GRAY)):
            ax.bar(xs, val * 100, bottom=bottom * 100, width=.72, zorder=3,
                   color=colour, edgecolor=INK, lw=LW)
            bottom += val
        ax.text(xs, f_ok * 50, f'{f_ok:.0%}', ha='center', va='center',
                fontsize=7.2, color='white', zorder=4)
        ticks.append(xs); labels.append(lab)
        rows.append(dict(panel='outcome', cohort=nm, score=lab,
                         resolved_correct=f_ok, resolved_wrong=f_bad,
                         needs_pet=f_pet, n=len(y)))
        xs += 1
    centres.append(((start + xs - 1) / 2, nm)); xs += .8
ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6.8)
ax.set_ylabel('Gray-zone patients (%)')
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 25))
ax.set_xlim(-.7, xs - 1.1)
ax.tick_params(axis='x', length=0, pad=3)
ax.spines['bottom'].set_color(INK); ax.spines['bottom'].set_linewidth(.8)
ax.set_title('Outcome without PET', fontsize=8, pad=5, loc='left')
for cx, nm in centres:
    ax.text(cx, -.16, nm, transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=7.5, color=INK)
h0 = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=INK, lw=LW)
      for c in (ROC_BLUE, ABETA_POS, GRAY)]
ax.legend(h0, ['Resolved correctly', 'Resolved incorrectly', 'Still needs PET'],
          loc='upper center', bbox_to_anchor=(.5, -.30), ncol=1, frameon=False,
          fontsize=6.6, handlelength=1.1)

# ---- (b) paired reclassification -------------------------------------------
ax = axes[1]
for i, (nm, y, pa, pb) in enumerate(COHORTS):
    ca = (pa >= .5).astype(int) == y
    cb = (pb >= .5).astype(int) == y
    gain = int((~ca & cb).sum())          # p-tau217 wrong, Stage 2 right
    loss = int((ca & ~cb).sum())          # p-tau217 right, Stage 2 wrong
    tab = [[int((ca & cb).sum()), loss], [gain, int((~ca & ~cb).sum())]]
    pv = mcnemar(tab, exact=(gain + loss) < 25).pvalue
    ax.barh(i, gain, height=.42, color=ABETA_NEG, edgecolor=INK, lw=LW, zorder=3)
    ax.barh(i, -loss, height=.42, color=ABETA_POS, edgecolor=INK, lw=LW, zorder=3)
    ax.text(gain + 4, i, f'{gain}', va='center', fontsize=7, color=INK)
    ax.text(-loss - 4, i, f'{loss}', va='center', ha='right', fontsize=7, color=INK)
    st = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    ax.text(158, i, st, ha='right', va='center', fontsize=8, color=INK)
    rows.append(dict(panel='reclassification', cohort=nm, score='Stage 2',
                     wrong_to_right=gain, right_to_wrong=loss, p=pv, n=len(y)))

ax.axvline(0, color=INK, lw=.8, zorder=4)
ax.set_yticks(range(len(COHORTS)))
ax.set_yticklabels([f'{n}\n(n={len(y):,})' for n, y, _, _ in COHORTS],
                   fontsize=7, linespacing=1.3)
ax.invert_yaxis()
ax.set_xlabel('Patients reclassified')
ax.set_xlim(-58, 165)
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_title('Paired reclassification by Stage 2', fontsize=8, pad=5, loc='left')
h = [plt.Rectangle((0, 0), 1, 1, fc=ABETA_NEG, ec=INK, lw=LW),
     plt.Rectangle((0, 0), 1, 1, fc=ABETA_POS, ec=INK, lw=LW)]
ax.legend(h, ['wrong to right', 'right to wrong'], loc='upper center',
          bbox_to_anchor=(.5, -.30), ncol=2, frameon=False, fontsize=6.8,
          handlelength=1.1)

fig.tight_layout(pad=.35)
fig.subplots_adjust(bottom=.34, wspace=.36)
save_panel(fig, PANELS / 'panel_reclassification')
t = pd.DataFrame(rows)
t.to_csv(TABLES / 'reclassification.csv', index=False)
print(t.to_string(index=False))
print('  panel_reclassification')
