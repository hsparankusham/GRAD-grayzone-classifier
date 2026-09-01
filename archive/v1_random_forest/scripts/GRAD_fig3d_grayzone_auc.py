#!/usr/bin/env python3
"""
Figure 3D — gray-zone discrimination: p-tau217 alone vs GRAD Stage 2.

Replaces the earlier sensitivity/specificity/accuracy panel, which read both
scores at a fixed 0.5 cut. That was invalid: Stage 1 emits a calibrated logistic
probability and Stage 2 a raw forest vote share, so 0.5 sits at different points
on the two ROC curves. It showed sensitivity rising 51.5% -> 65.1% while, at
MATCHED specificity, p-tau217 was actually better. AUC is threshold-free and
immune to that artefact.

Configuration:
    ADNI  all 142 gray-zone cases, missing Stage 2 analytes median-imputed,
          scored out-of-fold by leave-one-out
    A4    all 727 gray-zone cases, imputed with TRAINING medians -- never the
          test cohort's own, since a deployed model has no test cohort to
          borrow a median from

Boxes are paired bootstrap distributions (2,000 resamples of the same
participants); the bracket carries DeLong's test on the observed data, which is
the correct test for two correlated ROC curves.

Self-contained by design: importing the analysis modules re-ran their whole
pipelines just to obtain a data frame.

Output: results/figures/panels/panel_fig3d_grayzone_auc.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm


def auc_to_d(a):
    """Binormal Cohen's d implied by an AUC: d = sqrt(2) * Phi^-1(AUC).

    The standardised separation between the Abeta+ and Abeta- score
    distributions. Reported alongside AUC because d is the effect-size
    convention readers expect; the underlying test is still DeLong on
    the AUC difference.
    """
    return np.sqrt(2) * norm.ppf(a)

from _grad_paths import PANELS, TABLES, RESULTS
from _grad_style import apply_style, save_panel, INK
from run_a4_binary_validation import load_adni, load_a4, harmonize, engineer_features

apply_style(base=9)
LOCKED = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
          'APOE4_carrier', 'gfap_tau_interaction']
RF_KW = dict(n_estimators=100, max_depth=5, min_samples_leaf=5,
             random_state=42, n_jobs=-1, class_weight='balanced')
N_BOOT = 2000
RNG = np.random.default_rng(41)
LW = 0.9


def delong_p(y, p1, p2):
    """DeLong test for two correlated ROC curves on the same participants."""
    y = np.asarray(y).astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    m, n = len(pos), len(neg)

    def structural(p):
        pp, pn = p[pos][:, None], p[neg][None, :]
        psi = (pp > pn) + .5 * (pp == pn)
        return psi.mean(1), psi.mean(0), psi.mean()

    v10a, v01a, aa = structural(p1)
    v10b, v01b, ab = structural(p2)
    S = np.cov(np.vstack([v10a, v10b])) / m + np.cov(np.vstack([v01a, v01b])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    return float(2 * norm.sf(abs((aa - ab) / np.sqrt(var)))) if var > 0 else 1.0


# ---- data ------------------------------------------------------------------
adni, a4 = load_adni(), load_a4()
adni_h, a4_h = harmonize(adni, a4)
adni_h = engineer_features(adni_h).reset_index(drop=True)
a4_h = engineer_features(a4_h).reset_index(drop=True)
for df, f in ((adni_h, 'adni_loocv_predictions.csv'),
              (a4_h, 'a4_binary_validation_predictions.csv')):
    p = pd.read_csv(RESULTS / f)
    df['stage'] = p['stage'].values
    df['gk'] = p['gatekeeper_prob'].values
    df['y'] = p['true_amyloid'].values

tr = adni_h[adni_h.stage == 'reflex']
te = a4_h[a4_h.stage == 'reflex']

X = tr[LOCKED].astype(float).values.copy()
MED = np.nanmedian(X, axis=0)                      # training medians
for j in range(X.shape[1]):
    X[np.isnan(X[:, j]), j] = MED[j]
y_tr = tr.y.values.astype(int)

# ADNI scored out-of-fold; scoring it in-sample would read its own labels
loo = np.empty(len(y_tr))
for i in range(len(y_tr)):
    k = np.ones(len(y_tr), bool); k[i] = False
    sc = StandardScaler().fit(X[k])
    rf = RandomForestClassifier(**RF_KW).fit(sc.transform(X[k]), y_tr[k])
    loo[i] = rf.predict_proba(sc.transform(X[i:i + 1]))[0, 1]

SC = StandardScaler().fit(X)
RF = RandomForestClassifier(**RF_KW).fit(SC.transform(X), y_tr)
Z = te[LOCKED].astype(float).values.copy()
for j in range(Z.shape[1]):
    Z[np.isnan(Z[:, j]), j] = MED[j]
p_te = RF.predict_proba(SC.transform(Z))[:, 1]

COHORTS = [('ADNI (Development)', y_tr, tr.gk.values, loo),
           ('A4 + LEARN (External)', te.y.values.astype(int), te.gk.values, p_te)]

# ---- draw ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.3), sharey=True)
rows = []
tops = []
for ax, (title, y, pa, pb) in zip(axes, COHORTS):
    auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
    pv = delong_p(y, pb, pa)
    ba, bb = [], []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        ba.append(roc_auc_score(y[i], pa[i])); bb.append(roc_auc_score(y[i], pb[i]))

    for k, (vals, filled) in enumerate([(ba, False), (bb, True)]):
        ax.boxplot([vals], positions=[k], widths=.5, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=INK if filled else 'white',
                                 edgecolor=INK, lw=LW),
                   medianprops=dict(color='white' if filled else INK, lw=LW),
                   whiskerprops=dict(color=INK, lw=LW),
                   capprops=dict(color=INK, lw=LW))

    def cap(v):
        q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
        return max(x for x in v if x <= q3 + 1.5 * iqr)
    h = max(cap(ba), cap(bb)) + .014
    tops.append(h)
    ax.plot([0, 0, 1, 1], [h, h + .011, h + .011, h], lw=.8, color=INK)
    st = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    ax.text(.5, h + .014, st, ha='center', va='bottom',
            fontsize=9 if st != 'ns' else 7.5, color=INK)

    # effect size in the panel corner, clear of the data
    d_a, d_b = auc_to_d(auc_a), auc_to_d(auc_b)
    ax.text(.03, .97, f'$d$ = {d_b - d_a:+.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=7.5, color=INK)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['p-tau217\nalone', 'GRAD\nStage 2'], fontsize=7.2,
                       linespacing=1.4)
    ax.set_title(f'{title}   n = {len(y):,}', fontsize=8.5, pad=5, loc='left')
    ax.set_xlim(-.6, 1.6)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.spines['bottom'].set_color(INK); ax.spines['bottom'].set_linewidth(.8)
    rows.append(dict(cohort=title, n=len(y), auc_ptau=auc_a, auc_stage2=auc_b,
                     delta_auc=auc_b - auc_a, d_ptau=d_a, d_stage2=d_b,
                     delta_d=d_b - d_a, delong_p=pv))
    print(f'{title:24s} n={len(y):>4d}   AUC {auc_a:.3f} -> {auc_b:.3f}'
          f'   dAUC {auc_b-auc_a:+.3f}   d {d_a:.2f} -> {d_b:.2f}'
          f'   delta_d {d_b-d_a:+.2f}   P = {pv:.3f}')

axes[0].set_ylabel('Gray-zone AUC')
axes[0].set_ylim(.58, max(tops) + .075)
axes[0].set_yticks(np.arange(.60, .91, .10))
for a in axes: a.set_xlim(-.55, 1.55)
fig.tight_layout(pad=.25)
fig.subplots_adjust(wspace=.06)
save_panel(fig, PANELS / 'panel_fig3d_grayzone_auc')
pd.DataFrame(rows).to_csv(TABLES / 'fig3d_grayzone_auc.csv', index=False)
print('  panel_fig3d_grayzone_auc')
