#!/usr/bin/env python3
"""
Reflex feature importance + subgroup performance panels.
========================================================

Replaces two panels that were off-style and, in one case, off-data:

  panel_feature_importance  (was fig3a) -- the old panel hard-coded its six
      importance values, so it could silently drift from the fitted model.
      This version reads them off the Reflex forest itself.

  panel_subgroup_auc  (was fig3c, a forest plot) -- redrawn as a grouped bar
      chart anchored at chance (0.50), which is the meaningful floor for an
      AUC. Bars carry percentile-bootstrap CIs.

Both follow the frozen style: one muted blue, black unboxed text, Helvetica.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from _grad_paths import RESULTS, ADNI_DIR, TABLES, PANELS
from _grad_style import apply_style, save_panel, INK, ROC_BLUE, RULE
from data_loader import ADNIDataLoader
from harmonizer import AssayHarmonizer
from reflex import ReflexModel

apply_style(base=8)

GK_LOW, GK_HIGH = 0.25, 0.75
N_BOOT = 2000        # AUC bootstrap replicates
N_BOOT_IMP = 300     # forest refits for the importance bootstrap
RNG = np.random.default_rng(20260201)

LOCKED_FEATURES = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
                   'APOE4_carrier', 'gfap_tau_interaction']

# One shade per stratum, light -> dark in a muted green ramp. Green rather
# than blue so this panel is not mistaken for the (blue) importance and ROC
# panels. Lightness increases monotonically, so the blocks stay separable in
# greyscale and under colour-vision deficiency.
SHADES = {'Cognitive status': '#9DBFA8', 'APOE ε4': '#7BA68A',
          'Sex': '#5C8A6E', 'Age': '#3F6B52'}

# Display names, keyed to the locked feature columns.
PRETTY = {
    'pTau217_Z':            'p-tau217',
    'AGE':                  'Age',
    'gfap_tau_interaction': 'GFAP × p-tau217',
    'tau_ab42_diff':        'Tau/Aβ42 ratio',
    'APOE4_carrier':        'APOE ε4',
    'GFAP_Z':               'GFAP',
}


# ---------------------------------------------------------------- data
def load_cohort():
    """ADNI baseline cohort, harmonised exactly as the pipeline does."""
    df = ADNIDataLoader(str(ADNI_DIR)).merge_data(use_baseline_only=True)
    return AssayHarmonizer().fit_transform(df)


def attach_predictions(df):
    """Join LOOCV predictions onto the cohort and mark the gray zone."""
    preds = pd.read_csv(RESULTS / 'adni_loocv_predictions.csv')
    # predictions are written in cohort row order, so align positionally
    if len(preds) != len(df):
        raise SystemExit(f'row mismatch: {len(preds)} predictions vs {len(df)} cohort rows')
    out = df.reset_index(drop=True).copy()
    for c in ('true_amyloid', 'predicted_prob', 'gatekeeper_prob', 'stage'):
        out[c] = preds[c].values
    out['gray_zone'] = out['gatekeeper_prob'].between(GK_LOW, GK_HIGH)
    return out


def boot_auc_ci(y, p, n_boot=N_BOOT):
    """Percentile bootstrap CI for AUC; resamples participants."""
    y, p = np.asarray(y), np.asarray(p)
    n, aucs = len(y), []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:       # skip degenerate resamples
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan)


# --------------------------------------------------- A: feature importance
def panel_importance(df):
    """
    Mean decrease in Gini for the Reflex forest, fitted on the gray zone.

    Whiskers are a participant-level bootstrap: each replicate resamples the
    gray-zone participants and refits the whole forest, so the interval answers
    "would this ranking hold in another sample of this size". Across-tree SD --
    the usual default -- only describes disagreement inside one fitted forest
    and is far wider, which overstates the instability that matters here.
    """
    gz = df[df['gray_zone']].reset_index(drop=True)
    y = gz['true_amyloid']

    def fit_importance(frame, target):
        m = ReflexModel()
        m.fit(frame, target, feature_cols=LOCKED_FEATURES)
        return list(m.feature_names), m.model.feature_importances_

    names, mean = fit_importance(gz, y)

    # participant-level bootstrap
    reps = []
    for _ in range(N_BOOT_IMP):
        idx = RNG.integers(0, len(gz), len(gz))
        if y.iloc[idx].nunique() < 2:
            continue
        _, imp = fit_importance(gz.iloc[idx].reset_index(drop=True),
                                y.iloc[idx].reset_index(drop=True))
        reps.append(imp)
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)

    order = np.argsort(mean)                 # ascending -> largest on top
    labels = [PRETTY.get(names[i], names[i]) for i in order]
    m, l, h = mean[order] * 100, lo[order] * 100, hi[order] * 100

    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    ypos = np.arange(len(m))
    # one fill colour: bar length already encodes importance, so shading it
    # by the same quantity restates it and implies a cut-off that isn't there
    ax.barh(ypos, m, height=.6, color=ROC_BLUE, zorder=2)
    ax.errorbar(m, ypos, xerr=[m - l, h - m], fmt='none', ecolor=INK,
                elinewidth=.7, capsize=2, capthick=.7, zorder=3)

    xmax = h.max() * 1.16
    # values in a fixed right-hand column so they never collide with whiskers
    for yi, mi in zip(ypos, m):
        ax.text(xmax, yi, f'{mi:.1f}', va='center', ha='right',
                fontsize=7, color=INK)

    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlabel('Mean decrease in Gini impurity (%)')
    ax.set_xlim(0, xmax)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    save_panel(fig, PANELS / 'panel_feature_importance')

    return pd.DataFrame({'feature': [names[i] for i in order], 'label': labels,
                         'importance_pct': m, 'ci_lo_pct': l, 'ci_hi_pct': h})


# ------------------------------------------------------ B: subgroup AUC
def panel_subgroup(df):
    """
    Subgroup AUC as a grouped bar chart anchored at chance.

    Strata are kept in separate blocks so the comparison a reader makes is
    within-stratum (CN vs MCI vs Dementia), not across unrelated splits.
    """
    age_t = df['AGE'].quantile([1/3, 2/3]).values
    strata = [
        ('Cognitive status', [
            ('CN',  df['DX'].isin(['CN', 'NL', 'Normal'])),
            ('MCI', df['DX'].isin(['MCI', 'EMCI', 'LMCI'])),
            ('AD',  df['DX'].isin(['Dementia', 'AD'])),
        ]),
        ('APOE ε4', [
            ('ε4+', df['APOE4_carrier'] == 1),
            ('ε4−', df['APOE4_carrier'] == 0),
        ]),
        ('Sex', [
            ('Female', df['PTGENDER'].astype(str).str.upper().str.startswith('F')),
            ('Male',   df['PTGENDER'].astype(str).str.upper().str.startswith('M')),
        ]),
        ('Age', [
            (f'≤{age_t[0]:.0f}',            df['AGE'] <= age_t[0]),
            (f'{age_t[0]:.0f}–{age_t[1]:.0f}', df['AGE'].between(age_t[0], age_t[1], inclusive='right')),
            (f'>{age_t[1]:.0f}',            df['AGE'] > age_t[1]),
        ]),
    ]

    # compute AUC + CI per subgroup, laying out x positions with a gap
    # between strata so the blocks read as separate comparisons
    rows, xpos, x = [], [], 0.0
    for stratum, levels in strata:
        for label, mask in levels:
            sub = df[mask]
            if sub['true_amyloid'].nunique() < 2 or len(sub) < 15:
                continue
            auc = roc_auc_score(sub['true_amyloid'], sub['predicted_prob'])
            lo, hi = boot_auc_ci(sub['true_amyloid'], sub['predicted_prob'])
            rows.append(dict(stratum=stratum, label=label, n=len(sub),
                             auc=auc, lo=lo, hi=hi))
            xpos.append(x); x += 1
        x += 0.6                             # blank slot between strata
    res = pd.DataFrame(rows); res['x'] = xpos

    overall = roc_auc_score(df['true_amyloid'], df['predicted_prob'])

    fig, ax = plt.subplots(figsize=(4.3, 2.4))
    # bars start at 0.50: an AUC of 0 is not a meaningful origin, and a
    # zero-anchored bar would squeeze every difference into the top tenth
    base = 0.50
    # one shade per stratum -- distinguishes the blocks without adding a
    # second colour family; within a block every bar is identical
    colours = [SHADES[st] for st in res['stratum']]
    ax.bar(res['x'], res['auc'] - base, bottom=base, width=.86,
           color=colours, zorder=2)
    ax.errorbar(res['x'], res['auc'],
                yerr=[res['auc'] - res['lo'], res['hi'] - res['auc']],
                fmt='none', ecolor=INK, elinewidth=.7, capsize=2,
                capthick=.7, zorder=3)

    ax.set_xticks(res['x'])
    ax.set_xticklabels(res['label'], fontsize=6.5)
    ax.set_ylabel('AUC')
    ax.set_ylim(base, 1.0)
    ax.set_yticks([.5, .6, .7, .8, .9, 1.0])
    ax.set_xlim(-.7, res['x'].max() + .7)
    ax.tick_params(axis='x', length=0)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(INK)
    ax.spines['bottom'].set_linewidth(.8)

    save_panel(fig, PANELS / 'panel_subgroup_auc')
    return res.drop(columns='x')


if __name__ == '__main__':
    df = attach_predictions(load_cohort())
    print(f'cohort {len(df)}, gray zone {int(df.gray_zone.sum())}')

    imp = panel_importance(df)
    imp.to_csv(TABLES / 'supp_table_reflex_importance.csv', index=False)
    print('\nfeature importance (%):')
    print(imp[['label', 'importance_pct', 'ci_lo_pct', 'ci_hi_pct']].to_string(index=False))

    sub = panel_subgroup(df)
    sub.to_csv(TABLES / 'supp_table_subgroup_auc.csv', index=False)
    print('\nsubgroup AUC:')
    print(sub.to_string(index=False))
