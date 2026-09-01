#!/usr/bin/env python3
"""
Figure 2 panels A-D: GRAD model performance.
==============================================

    A   ROC, ADNI development cohort (full pipeline / Stage 1 / Stage 2)
    B   ROC, A4 + LEARN validation cohort (same three)
    C   Stage 1 probability distribution and the 0.25-0.75 routing band,
        both cohorts, by Aβ status
    D   Predicted probability against Centiloid, A4 + LEARN

Panels are written separately as RGB PNG at 600 dpi for downstream assembly.
Reads the prediction files written by run_grad_impaired_logistic.py.

    python3 scripts/GRAD_fig2_panels.py
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

from _grad_paths import RESULTS, PROJECT_ROOT  # noqa: F401
from _grad_style import (apply_style, save_panel, PANEL_LETTER,
                         ABETA_NEG, ABETA_POS, ROC_BLUE, NAVY, BAND, GUIDE,
                         INK, SLATE)

SLATE = '#6E7A85'    # Stage 2 ROC; distinct from the chance diagonal
CHANCE = '#CBCBCB'   # chance diagonal only

GK_LOW, GK_HIGH = 0.25, 0.75
PANEL_W, PANEL_H = 3.35, 2.95      # inches; 85 mm wide, two per column width

OUT = PROJECT_ROOT / 'results' / 'figures' / 'panels'
OUT.mkdir(parents=True, exist_ok=True)


def load():
    adni = pd.read_csv(RESULTS / 'adni_loocv_predictions_v2.csv')
    a4 = pd.read_csv(RESULTS / 'a4_predictions_v2.csv')
    return adni, a4


def letter(ax, ch):
    ax.text(-0.20, 1.10, ch, transform=ax.transAxes, **PANEL_LETTER)


# ------------------------------------------------------------------ A, B ---

def roc_panel(df, title, stem, ch):
    """Three ROCs: full pipeline, Stage 1 on the cases it resolves, Stage 2."""
    y = df['y'].values
    p = df['final_prob'].values
    g = df['gk_prob'].values
    res = (df['stage'] == 'gatekeeper').values
    gz = ~res

    series = [
        ('Full pipeline', y, p, '-', ROC_BLUE, 1.6),
        ('Stage 1 (resolved)', y[res], g[res], '--', NAVY, 1.2),
        ('Stage 2 (gray zone)', y[gz], p[gz], (0, (1.2, 1.4)), SLATE, 1.3),
    ]

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.plot([0, 1], [0, 1], color=CHANCE, lw=.5, ls=(0, (2, 3)), zorder=1)
    for label, yy, ss, ls, colour, lw in series:
        fpr, tpr, _ = roc_curve(yy, ss)
        auc = roc_auc_score(yy, ss)
        ax.plot(fpr, tpr, ls=ls, color=colour, lw=lw, solid_capstyle='round',
                label=f'{label}  {auc:.3f}  (n = {len(yy):,})', zorder=3)

    ax.set_xlim(-.02, 1.02)
    ax.set_ylim(-.02, 1.02)
    ax.set_xlabel('1 − specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(title, pad=6)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.legend(loc='lower right', handlelength=1.9, borderpad=.2,
              labelspacing=.35)
    ax.set_aspect('equal', adjustable='box')
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


# --------------------------------------------------------------------- C ---

def band_panel(adni, a4, stem, ch):
    """Stage 1 probability by Aβ status, with the routing band marked."""
    fig, axes = plt.subplots(2, 1, figsize=(PANEL_W, PANEL_H), sharex=True,
                             gridspec_kw=dict(hspace=.42))
    bins = np.linspace(0, 1, 31)

    for ax, df, name in ((axes[0], adni, 'ADNI (n = 145)'),
                         (axes[1], a4, 'A4 + LEARN (n = 1,644)')):
        g = df['gk_prob'].values
        y = df['y'].values
        ax.axvspan(GK_LOW, GK_HIGH, color=BAND, lw=0, zorder=0)
        ax.hist(g[y == 0], bins=bins, color=ABETA_NEG, alpha=.85, lw=0,
                label='Aβ−', zorder=2)
        ax.hist(g[y == 1], bins=bins, color=ABETA_POS, alpha=.75, lw=0,
                label='Aβ+', zorder=3)
        for x in (GK_LOW, GK_HIGH):
            ax.axvline(x, color=INK, lw=.6, ls=(0, (2, 2)), zorder=4)
        ax.set_ylabel('Participants')
        ax.set_title(name, pad=4, loc='left')
        ax.margins(x=0)

    gz_a = ((adni['stage'] == 'reflex').mean()) * 100
    gz_b = ((a4['stage'] == 'reflex').mean()) * 100
    axes[0].text(.5, .95, f'gray zone\n{gz_a:.1f}%', transform=axes[0].transAxes,
                 ha='center', va='top', fontsize=10, color=INK)
    axes[1].text(.5, .95, f'gray zone\n{gz_b:.1f}%', transform=axes[1].transAxes,
                 ha='center', va='top', fontsize=10, color=INK)
    axes[0].legend(loc='upper right', handlelength=1.1, borderpad=.2)
    axes[1].set_xlabel('Stage 1 predicted probability of Aβ positivity')
    axes[1].set_xticks([0, .25, .5, .75, 1])
    letter(axes[0], ch)
    return save_panel(fig, str(OUT / stem))


# --------------------------------------------------------------------- D ---

def centiloid_panel(a4, stem, ch):
    """Predicted probability against continuous amyloid burden.

    Deliberately bare: the only type in the panel is the correlation statistic.
    Axis labels, legend and the Centiloid-20 annotation are added by hand at
    assembly, so nothing here competes with them.
    """
    d = a4.dropna(subset=['centiloid'])
    y = d['y'].values
    p = d['final_prob'].values
    c = d['centiloid'].values
    rho = spearmanr(p, c).statistic

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.axvline(20, color=GUIDE, lw=.7, ls=(0, (3, 3)), zorder=1)
    for mask, colour in ((y == 0, ABETA_NEG), (y == 1, ABETA_POS)):
        ax.scatter(c[mask], p[mask], s=5, color=colour, alpha=.55,
                   linewidths=0, zorder=2)

    edges = np.percentile(c, np.linspace(0, 100, 13))
    mids, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c <= hi)
        if m.sum() >= 10:
            mids.append(np.median(c[m]))
            med.append(np.median(p[m]))
    ax.plot(mids, med, color=NAVY, lw=1.3, zorder=4)

    ax.set_ylim(-.03, 1.03)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.text(.97, .06, f'ρ = {rho:.3f}, P < .001', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color=INK)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


# --------------------------------------------------------------------- E ---

def feature_weights(stem, ch=''):
    """Stage 2 predictive contribution: absolute weight per SD.

    Magnitude only. A signed plot invites the wrong reading - Aβ42/40 carries a
    negative coefficient because a *lower* ratio indicates more amyloid, which
    a reader can mistake for the marker being unhelpful. Direction belongs in
    the legend, not the bars.
    """
    m = json.load(open(RESULTS / 'grad_v2_numbers.json'))['stage2_model']
    pretty = {'APOE4_carrier': 'APOE ε4 carrier',
              'pTau217_Z': 'p-Tau217',
              'AB4240_log': 'Aβ42/40',
              'AGE': 'Age',
              'GFAP_Z': 'GFAP',
              'tau_ab42_diff': 'tau−Aβ42/40 ratio',
              'gfap_tau_interaction': 'GFAP × p-Tau217'}
    items = sorted(m['coefficients'].items(), key=lambda kv: abs(kv[1]))
    names = [pretty[k] for k, _ in items]
    vals = np.array([abs(v) for _, v in items])

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    y = np.arange(len(vals))
    ax.barh(y, vals, .62, lw=0, color=ROC_BLUE, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, .50)
    ax.set_xticks([0, .1, .2, .3, .4, .5])
    ax.set_ylim(-.68, len(vals) - .32)
    ax.set_xlabel('Predictive weight per SD')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    if ch:
        letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


def main():
    apply_style()
    adni, a4 = load()
    written = [
        band_panel(adni, a4, 'fig2A_routing_band', 'A'),
        roc_panel(adni, 'ADNI development cohort', 'fig2B_roc_adni', 'B'),
        roc_panel(a4, 'A4 + LEARN validation cohort', 'fig2C_roc_a4', 'C'),
        centiloid_panel(a4, 'fig2D_centiloid', 'D'),
        feature_weights('fig2E_feature_weights'),
    ]
    for w in written:
        print('wrote', w)


if __name__ == '__main__':
    main()
