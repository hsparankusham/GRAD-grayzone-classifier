#!/usr/bin/env python3
"""
Figure 3D — gray-zone comparison on MATCHED operating points.

Same three-bar-per-cohort layout as the previous panel, but every bar is now a
valid comparison. The old version read sensitivity and specificity off each
model at its own 0.5 cut -- and 0.5 is not the same place on the two ROC curves,
because Stage 1 emits a calibrated logistic probability while Stage 2 emits a
raw forest vote share. That is why the old panel showed sensitivity rising
51.5% -> 65.1% while, at matched specificity, p-tau217 was actually better.

The three comparisons drawn here:

  Sensitivity @ matched spec   p-tau217 is read at its own 0.5 cut; GRAD's
                               threshold is moved until its specificity equals
                               p-tau217's. Same false-alarm rate, so the
                               question becomes who catches more positives.
  Specificity @ matched sens   the mirror image: GRAD moved to p-tau217's
                               sensitivity, then compare false-alarm rates.
  AUC                          threshold-free; no matching needed.

Boxes are paired bootstrap distributions (2,000 resamples of the same
participants). P values are bootstrap two-sided for the matched points and
DeLong for AUC, which is the correct test for two correlated ROC curves.

Output: results/figures/panels/panel_fig3d_matched.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from _grad_paths import PANELS, TABLES
from _grad_style import apply_style, save_panel, INK
from GRAD_band_splitsample import adni_h, a4_h, LOCKED, RF_KW
from GRAD_mri_gray_zone import delong_p

apply_style(base=9)
N_BOOT = 2000
RNG = np.random.default_rng(41)
LW = 0.9


# ---- fit once on all 142 ADNI gray-zone cases, medians from training --------
tr = adni_h[adni_h.stage == 'reflex']
Xtr = tr[LOCKED].astype(float).values.copy()
MED = np.nanmedian(Xtr, axis=0)
for j in range(Xtr.shape[1]):
    Xtr[np.isnan(Xtr[:, j]), j] = MED[j]
y_tr = tr.y.values.astype(int)
SC = StandardScaler().fit(Xtr)
RF = RandomForestClassifier(**RF_KW).fit(SC.transform(Xtr), y_tr)


def score(df):
    Z = df[LOCKED].astype(float).values.copy()
    for j in range(Z.shape[1]):
        Z[np.isnan(Z[:, j]), j] = MED[j]        # training medians, never test's
    return RF.predict_proba(SC.transform(Z))[:, 1]


# ADNI must be scored out-of-fold or it is reading its own training labels
loo = np.empty(len(y_tr))
for i in range(len(y_tr)):
    k = np.ones(len(y_tr), bool); k[i] = False
    s = StandardScaler().fit(Xtr[k])
    m = RandomForestClassifier(**RF_KW).fit(s.transform(Xtr[k]), y_tr[k])
    loo[i] = m.predict_proba(s.transform(Xtr[i:i + 1]))[0, 1]

a4 = a4_h[a4_h.stage == 'reflex'].copy()
a4 = a4[a4[LOCKED].notna().all(axis=1)]         # complete panel only
COHORTS = [('ADNI (Development)', y_tr, tr.gk.values, loo),
           ('A4 (External Validation)', a4.y.values.astype(int),
            a4.gk.values, score(a4))]


# ---- the three comparisons -------------------------------------------------
def sens_spec(y, p, thr=.5):
    pred = p >= thr
    return (pred & (y == 1)).sum() / (y == 1).sum(), \
           (~pred & (y == 0)).sum() / (y == 0).sum()


def sens_at_spec(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = np.where((1 - fpr) >= target)[0]
    return tpr[ok[-1]] if len(ok) else np.nan


def spec_at_sens(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return (1 - fpr)[ok[0]] if len(ok) else np.nan


def triple(y, pa, pb):
    """(p-tau217 value, GRAD value) for each of the three comparisons."""
    s_a, q_a = sens_spec(y, pa)                       # p-tau217 at its own cut
    return [(s_a, sens_at_spec(y, pb, q_a)),          # GRAD moved to match spec
            (q_a, spec_at_sens(y, pb, s_a)),          # GRAD moved to match sens
            (roc_auc_score(y, pa), roc_auc_score(y, pb))]


METRICS = ['Sensitivity\n@ matched spec', 'Specificity\n@ matched sens', 'AUC']
rows = []
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)

for ax, (title, y, pa, pb) in zip(axes, COHORTS):
    obs = triple(y, pa, pb)
    boots = [([], []) for _ in METRICS]
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for k, (va, vb) in enumerate(triple(y[i], pa[i], pb[i])):
            if not (np.isnan(va) or np.isnan(vb)):
                boots[k][0].append(va * 100); boots[k][1].append(vb * 100)

    for k, m in enumerate(METRICS):
        for j, vals in enumerate(boots[k]):
            filled = (j == 1)
            ax.boxplot([vals], positions=[k + (-.18 if j == 0 else .18)],
                       widths=.28, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=INK if filled else 'white',
                                     edgecolor=INK, lw=LW),
                       medianprops=dict(color='white' if filled else INK, lw=LW),
                       whiskerprops=dict(color=INK, lw=LW),
                       capprops=dict(color=INK, lw=LW))
        # AUC uses DeLong; the matched points use a two-sided bootstrap
        if m == 'AUC':
            _, _, pv = delong_p(y, pb, pa)
        else:
            d = np.array(boots[k][1]) - np.array(boots[k][0])
            pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
        rows.append(dict(cohort=title, metric=m.replace('\n', ' '),
                         ptau=obs[k][0], grad=obs[k][1],
                         delta=obs[k][1] - obs[k][0], p=pv, n=len(y)))

        def cap(v):
            q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
            return max(x for x in v if x <= q3 + 1.5 * iqr)
        h = max(cap(boots[k][0]), cap(boots[k][1])) + 2.0
        ax.plot([k - .18, k - .18, k + .18, k + .18],
                [h, h + 1.4, h + 1.4, h], lw=.7, color=INK)
        st = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
        ax.text(k, h + 1.9, st, ha='center', va='bottom', fontsize=8, color=INK)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS, fontsize=6.8, linespacing=1.35)
    ax.set_xlim(-.5, len(METRICS) - .5)
    ax.set_title(f'{title}   n = {len(y)}', fontsize=8.5, pad=4)
    ax.tick_params(axis='x', length=0, pad=3)

axes[0].set_ylabel('Gray-zone performance (%; AUC × 100)')
handles = [plt.Rectangle((0, 0), 1, 1, fc='white', ec=INK, lw=LW),
           plt.Rectangle((0, 0), 1, 1, fc=INK, ec=INK, lw=LW)]
fig.legend(handles, ['p-tau217 alone', 'GRAD Stage 2'], loc='lower center',
           ncol=2, frameon=False, fontsize=7.5, handlelength=1.1)
fig.tight_layout(pad=.3)
fig.subplots_adjust(bottom=.24, wspace=.07)
save_panel(fig, PANELS / 'panel_fig3d_matched')

t = pd.DataFrame(rows)
t.to_csv(TABLES / 'fig3d_matched.csv', index=False)
print(f"{'cohort':26s}{'metric':30s}{'p-tau217':>10s}{'GRAD':>9s}{'delta':>9s}{'p':>9s}")
for r in t.itertuples():
    v = 100 if 'AUC' not in r.metric else 1
    print(f'{r.cohort:26s}{r.metric:30s}{r.ptau*v:>10.3f}{r.grad*v:>9.3f}'
          f'{(r.grad-r.ptau)*v:>+9.3f}{r.p:>9.4f}')
