#!/usr/bin/env python3
"""
Figure 3 panels A-D: benchmarking and model characterisation.
==============================================================

    A   ROC in the A4 gray zone participants with p-Tau181 available (n = 256):
        GRAD Stage 2 vs p-Tau217 alone vs p-Tau181 integration
    B   What happens to those patients when no scan is ordered - resolved
        correctly, resolved wrongly, referred for PET - for the three methods
    C   Full-pipeline AUC by clinical subgroup, ADNI development cohort
    D   Calibration of the full pipeline in A4 + LEARN

Panels are written separately as RGB PNG at 600 dpi for downstream assembly.

    python3 scripts/GRAD_fig3_panels.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

from _grad_paths import RESULTS, PROJECT_ROOT, A4_DIR
from _grad_style import (apply_style, save_panel, PANEL_LETTER, ABETA_NEG,
                         ABETA_POS, ROC_BLUE, NAVY, GUIDE, INK, BAND,
                         GREEN, AMBER, CRIMSON, SLATE, CHANCE)
from run_grad_impaired_logistic import (load_adni, load_a4, IMPAIRED_DX,
                                        MCI_DX, DEM_DX,
                                        fit_full_and_score_a4, BAND_LOW,
                                        BAND_HIGH, delong_p as delong, boot_ci)

PANEL_W, PANEL_H = 3.35, 2.95

OUT = PROJECT_ROOT / 'results' / 'figures' / 'panels'
OUT.mkdir(parents=True, exist_ok=True)


def letter(ax, ch):
    ax.text(-0.20, 1.10, ch, transform=ax.transAxes, **PANEL_LETTER)


def ptau181_subset():
    """The 256 A4 gray-zone participants with a p-Tau181 measurement."""
    adni, a4 = load_adni(), load_a4()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]
    out, _ = fit_full_and_score_a4(adni, idx, a4)
    gz = (out['stage'] == 'reflex').values

    roche = pd.read_csv(os.path.join(str(A4_DIR), 'Clinical', 'External Data',
                                     'biomarker_Plasma_Roche_Results.csv'))
    roche['LABRESN'] = pd.to_numeric(roche['LABRESN'], errors='coerce')
    roche['LBTESTCD'] = roche['LBTESTCD'].str.strip()
    p181 = (roche[roche['LBTESTCD'] == 'TPP181']
            .sort_values(['BID', 'VISCODE']).groupby('BID').first()
            .reset_index()[['BID', 'LABRESN']])
    j = a4[['BID']].merge(p181, on='BID', how='left')
    has = j['LABRESN'].notna().values & gz
    return pd.DataFrame({
        'y': out['y'].values[has],
        'grad': out['final_prob'].values[has],
        'ptau217': out['gk_prob'].values[has],
        'ptau181': j['LABRESN'].values[has],
    })


# --------------------------------------------------------------------- A ---

def _box(ax, vals, pos, colour, width=.55):
    ax.boxplot([vals], positions=[pos], widths=width, patch_artist=True,
               showfliers=False,
               medianprops=dict(color=INK, lw=1.0),
               whiskerprops=dict(color=colour, lw=.8),
               capprops=dict(color=colour, lw=.8),
               boxprops=dict(facecolor=colour, edgecolor=colour, lw=.8,
                             alpha=.55))
    rng = np.random.RandomState(7)
    ax.scatter(pos + rng.uniform(-.15, .15, len(vals)), vals, s=9,
               facecolor='white', edgecolor=colour, linewidths=.7, zorder=3)


def _stars(p):
    return 'ns' if p >= .05 else '*' if p >= .01 else '**' if p >= .001 else '***'


def _bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=.8,
            clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom',
            fontsize=10, color=INK, clip_on=False)


def _boot_auc(y, score, rng, n_boot=2000):
    n = len(y)
    out = []
    for _ in range(n_boot):
        i = rng.choice(n, n, replace=True)
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], score[i]))
    return np.array(out)


def separation_boxes(d, stem, ch):
    """Benchmarking: each comparison shown at the sample size it applies to.

    Left  - full A4 gray zone (n = 630): GRAD Stage 2 vs p-Tau217 alone.
    Right - the subset with p-Tau181 measured (n = 256): GRAD vs p-Tau181.
    """
    adni, a4 = load_adni(), load_a4()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]
    out, _ = fit_full_and_score_a4(adni, idx, a4)
    gz = (out['stage'] == 'reflex').values
    y_full = out['y'].values[gz]
    grad_full = out['final_prob'].values[gz]
    p217_full = out['gk_prob'].values[gz]

    rng = np.random.RandomState(42)
    blocks = [
        ('Full gray zone\n(n = %d)' % len(y_full),
         [('GRAD\nStage 2', grad_full, ROC_BLUE), ('p-Tau217\nalone', p217_full, SLATE)],
         y_full, delong(y_full, grad_full, p217_full)[2]),
        ('With p-Tau181\n(n = %d)' % len(d),
         [('GRAD\nStage 2', d['grad'].values, ROC_BLUE), ('p-Tau181', d['ptau181'].values, CRIMSON)],
         d['y'].values, delong(d['y'].values, d['grad'].values, d['ptau181'].values)[2]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_W * 1.30, PANEL_H))
    for ax, (title, series, yy, pv) in zip(axes, blocks):
        tops = []
        for i, (label, score, colour) in enumerate(series):
            v = _boot_auc(yy, score, rng)
            _box(ax, v[rng.choice(len(v), 90, replace=False)], i, colour)
            tops.append(v.max())
        top = max(tops)
        _bracket(ax, 0, 1, top + .015, .020, _stars(pv))
        ax.set_xticks([0, 1])
        ax.set_xticklabels([s[0] for s in series])
        ax.set_xlim(-.62, 1.62)
        ax.set_ylim(.44, top + .10)
        ax.set_ylabel('AUC')
        ax.set_title(title, pad=8)
    fig.subplots_adjust(wspace=.45)
    axes[0].text(-0.34, 1.14, ch, transform=axes[0].transAxes, **PANEL_LETTER)
    return save_panel(fig, str(OUT / stem))


# --------------------------------------------------------------------- B ---

def coverage_accuracy(d, stem, ch):
    """Accuracy against coverage: how often each method is right, as a function
    of how many patients it is willing to answer for."""
    y = d['y'].values
    fpr, tpr, thr = roc_curve(y, d['ptau181'])
    cut181 = thr[np.argmax(tpr - fpr)]

    methods = [('GRAD Stage 2', d['grad'].values, .5, ROC_BLUE, '-', 1.7,
                .805, .728),
               ('p-Tau217 alone', d['ptau217'].values, .5, SLATE, '--', 1.3,
                .641, .689),
               ('p-Tau181', d['ptau181'].values, cut181, CRIMSON,
                (0, (1.2, 1.4)), 1.4, 1.0, .598)]

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    for label, score, thr_, colour, ls, lw, op_cov, op_acc in methods:
        conf = np.abs(score - thr_)
        order = np.argsort(-conf)
        call = (score >= thr_).astype(int)
        n = len(y)
        cov, acc = [], []
        for k in range(int(.5 * n), n + 1):
            i = order[:k]
            cov.append(100 * k / n)
            acc.append(100 * (call[i] == y[i]).mean())
        ax.plot(cov, acc, ls=ls, color=colour, lw=lw, label=label, zorder=3)
        ax.plot(100 * op_cov, 100 * op_acc, 'o', ms=6, color=colour,
                markeredgecolor='white', markeredgewidth=.9, zorder=5)

    ax.set_xlim(50, 103)
    ax.set_xlabel('Patients given an answer (%)')
    ax.set_ylabel('Accuracy of those answers (%)')
    ax.set_title('Accuracy versus coverage', pad=8)
    ax.margins(y=.12)
    ax.legend(loc='upper right', handlelength=1.9, borderpad=.2,
              labelspacing=.35)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


# --------------------------------------------------------------------- C ---

def _forest(rows, overall, olo, ohi, title, xlim, xticks, stem, ch):
    """Shared forest-plot renderer, so every cohort's panel looks identical."""
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    yy = np.arange(len(rows))[::-1]
    ax.axvspan(olo, ohi, color=BAND, lw=0, zorder=0)
    ax.axvline(overall, color=GUIDE, lw=.8, ls=(0, (3, 3)), zorder=1)
    for i, (name, n, auc, lo, hi) in enumerate(rows):
        ax.plot([lo, min(hi, xlim[1])], [yy[i], yy[i]], color=SLATE, lw=1.0,
                solid_capstyle='round', zorder=2)
        ax.plot(auc, yy[i], 'o', ms=4, color=ROC_BLUE, zorder=3)
        ax.text(1.02, yy[i], f'{auc:.3f}', va='center', fontsize=10, color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(yy)
    ax.set_yticklabels([f'{n} (n = {c:,})' for n, c, *_ in rows])
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-.75, len(rows) - 1 + .75)
    ax.set_xlabel('Full-pipeline AUC')
    ax.set_title(title, pad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


def subgroup_forest(stem, ch):
    """A4 + LEARN validation cohort."""
    adni, a4 = load_adni(), load_a4()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]
    out, _ = fit_full_and_score_a4(adni, idx, a4)
    y, p = out['y'].values, out['final_prob'].values

    age = a4['AGE'].values
    t1, t2 = np.percentile(age, [100 / 3, 200 / 3])
    strata = [('APOE ε4+', (a4['APOE4_carrier'] == 1).values),
              ('APOE ε4−', (a4['APOE4_carrier'] == 0).values),
              ('Female', (a4['SEX'] == 1).values),
              ('Male', (a4['SEX'] == 2).values),
              (f'Age ≤{t1:.0f}', age <= t1),
              (f'Age {t1:.0f}–{t2:.0f}', (age > t1) & (age <= t2)),
              (f'Age >{t2:.0f}', age > t2)]
    rows = [(nm, int(m.sum()), roc_auc_score(y[m], p[m]),
             *boot_ci(y[m], p[m], roc_auc_score)) for nm, m in strata]
    return _forest(rows, roc_auc_score(y, p), *boot_ci(y, p, roc_auc_score),
                   'A4 + LEARN subgroups', (.72, 1.0),
                   [.75, .80, .85, .90, .95, 1.0], stem, ch)


def subgroup_forest_adni(stem, ch):
    """ADNI development cohort - supplementary companion to the A4 panel."""
    adni = load_adni()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]
    imp = adni.loc[idx].reset_index(drop=True)
    pred = pd.read_csv(RESULTS / 'adni_loocv_predictions_v2.csv')
    order = {v: i for i, v in enumerate(idx)}
    pred = pred.sort_values('idx', key=lambda s: s.map(order))
    y, p = pred['y'].values, pred['final_prob'].values

    age = pd.to_numeric(imp['AGE'], errors='coerce').values
    ok = ~np.isnan(age)
    t1, t2 = np.nanpercentile(age, [100 / 3, 200 / 3])
    strata = [('MCI', imp['DX'].isin(MCI_DX).values),
              ('AD dementia', imp['DX'].isin(DEM_DX).values),
              ('APOE ε4+', (imp['APOE4_carrier'] == 1).values),
              ('APOE ε4−', (imp['APOE4_carrier'] == 0).values),
              ('Female', (imp['PTGENDER'] == 'Female').values),
              ('Male', (imp['PTGENDER'] == 'Male').values),
              (f'Age ≤{t1:.0f}', ok & (age <= t1)),
              (f'Age {t1:.0f}–{t2:.0f}', ok & (age > t1) & (age <= t2)),
              (f'Age >{t2:.0f}', ok & (age > t2))]
    rows = [(nm, int(m.sum()), roc_auc_score(y[m], p[m]),
             *boot_ci(y[m], p[m], roc_auc_score)) for nm, m in strata
            if m.sum() >= 10 and len(np.unique(y[m])) > 1]
    return _forest(rows, roc_auc_score(y, p), *boot_ci(y, p, roc_auc_score),
                   'ADNI subgroups', (.55, 1.03),
                   [.6, .7, .8, .9, 1.0], stem, ch)


# --------------------------------------------------------------------- D ---

def calibration_panel(stem, ch):
    """Calibration with binomial error bars and the score distribution below."""
    a4 = pd.read_csv(RESULTS / 'a4_predictions_v2.csv')
    cal = json.load(open(RESULTS / 'grad_v2_numbers.json'))['a4_calibration']
    p = a4['final_prob'].values
    y = a4['y'].values

    edges = np.percentile(p, np.linspace(0, 100, 9))
    edges[0], edges[-1] = 0, 1
    xs, ys, lo, hi = [], [], [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (p >= a_) & (p <= b_)
        if m.sum() < 20:
            continue
        k, n = y[m].sum(), m.sum()
        f = k / n
        se = np.sqrt(max(f * (1 - f), 1e-9) / n)
        xs.append(p[m].mean())
        ys.append(f)
        lo.append(max(0, f - 1.96 * se))
        hi.append(min(1, f + 1.96 * se))
    xs, ys = np.array(xs), np.array(ys)

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(PANEL_W, PANEL_H), sharex=True,
        gridspec_kw=dict(height_ratios=[4.2, 1], hspace=.08))

    ax.plot([0, 1], [0, 1], color=INK, lw=.7, zorder=1)
    ax.errorbar(xs, ys, yerr=[ys - np.array(lo), np.array(hi) - ys],
                fmt='o-', color=ROC_BLUE, ecolor=ROC_BLUE, lw=1.4, ms=5,
                elinewidth=.9, capsize=2.5, capthick=.9, zorder=3)
    ax.set_xlim(-.02, 1.03)
    ax.set_ylim(-.02, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel('Observed fraction Aβ+')
    ax.text(.03, .97, f"Brier {cal['brier']:.3f}\nSlope {cal['slope']:.2f}\n"
                      f"Intercept {cal['intercept']:+.2f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=10, color=INK)

    bins = np.linspace(0, 1, 31)
    axh.hist(p[y == 0], bins=bins, color=ABETA_NEG, alpha=.85, lw=0)
    axh.hist(p[y == 1], bins=bins, color=ABETA_POS, alpha=.75, lw=0)
    axh.set_xlim(-.02, 1.03)
    axh.set_xticks([0, .25, .5, .75, 1])
    axh.set_xlabel('Predicted probability')
    axh.set_yticks([])
    for s in ('left', 'right', 'top'):
        axh.spines[s].set_visible(False)
    ax.text(-0.24, 1.06, ch, transform=ax.transAxes, **PANEL_LETTER)
    return save_panel(fig, str(OUT / stem))


def main():
    apply_style()
    d = ptau181_subset()
    written = [
        subgroup_forest('fig3A_subgroups', 'A'),
        separation_boxes(d, 'fig3B_benchmark_boxes', 'B'),
        coverage_accuracy(d, 'fig3C_coverage_accuracy', 'C'),
        subgroup_forest_adni('figS1_subgroups_adni', ''),
        calibration_panel('fig3D_calibration', 'D'),
    ]
    for w in written:
        print('wrote', w)


if __name__ == '__main__':
    main()
