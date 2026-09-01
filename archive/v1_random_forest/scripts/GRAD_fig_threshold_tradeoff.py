#!/usr/bin/env python3
"""
Threshold trade-off curves — the continuous view behind the operating configs.

Companion to GRAD_fig_operating_configs.py. That panel shows three chosen
operating points; this one shows the surface they are chosen from, so a reader
can see that the configurations are cuts on a continuum rather than three
separate models. The two share CONFIGS, so the guides drawn here are literally
the cuts the bar panel uses.

Colour follows the rest of the figure set:
    rule-out arm  (sensitivity, NPV)  -> ABETA_NEG, the colour of an Abeta-
                                         call in every other panel
    rule-in arm   (specificity, PPV)  -> ABETA_POS
    referred band                     -> the same amber as the config bars

Accuracy is not plotted. It is a prevalence-weighted blend of the two arms and
tells a clinician nothing that sensitivity and specificity do not already say.

Output: results/figures/panels/panel_threshold_tradeoff.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _grad_paths import PANELS, RESULTS
from _grad_style import apply_style, save_panel, ABETA_NEG, ABETA_POS, INK, GUIDE
from GRAD_fig_operating_configs import CONFIGS, split, PET

apply_style(base=9)

GRID_T = np.linspace(0.005, 0.995, 400)
# the configuration whose band is shaded; the other two are listed in the
# companion panel rather than crowding six guides onto one axis
HEADLINE = 'Balanced triage\n90% / 90%'
MIN_DENOM = 20          # smallest PPV/NPV denominator worth plotting


def sweep(y, p):
    """Sensitivity, specificity, PPV and NPV across every classification cut."""
    rows = []
    for t in GRID_T:
        pred = p >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        tn, fn = (~pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
        rows.append(dict(
            t=t,
            sens=tp / (tp + fn) if tp + fn else np.nan,
            spec=tn / (tn + fp) if tn + fp else np.nan,
            # PPV/NPV are undefined when no one is called positive/negative.
            # The old panel plotted those as 0, which drew a false cliff at the
            # low end; leaving them NaN simply ends the line.
            ppv=tp / (tp + fp) if tp + fp >= MIN_DENOM else np.nan,
            npv=tn / (tn + fn) if tn + fn >= MIN_DENOM else np.nan,
        ))
    return pd.DataFrame(rows)


def panel(ax, y, p, title, label_curves=False):
    s = sweep(y, p)
    st, sp = [(a, b) for lab, a, b in CONFIGS if lab == HEADLINE][0]
    cuts = split(y, p, st, sp)

    # shaded band = participants referred for PET under the headline config;
    # this is the same amber segment that the operating-config bars show
    ax.axvspan(cuts['t_out'], cuts['t_in'], color=PET, alpha=.22, lw=0, zorder=1)
    for t in (cuts['t_out'], cuts['t_in']):
        ax.axvline(t, color=GUIDE, lw=.6, zorder=2)

    series = [('sens', ABETA_NEG, '-',  1.7, 'Sensitivity'),
              ('npv',  ABETA_NEG, '--', 1.0, 'NPV'),
              ('spec', ABETA_POS, '-',  1.7, 'Specificity'),
              ('ppv',  ABETA_POS, '--', 1.0, 'PPV')]
    for col, colour, ls, lw, name in series:
        ax.plot(s['t'], s[col], color=colour, ls=ls, lw=lw,
                solid_capstyle='round', zorder=4)

    # Direct labels at the right edge instead of a legend box: at t = 1 the four
    # curves are far apart, so they self-separate and never cover the data.
    # Only the right-hand panel is labelled -- both panels carry the same four
    # series, and repeating the key would just add ink.
    if label_curves:
        ends = [(s[col].dropna().iloc[-1], name, colour)
                for col, colour, ls, lw, name in series]
        # Specificity and PPV both finish near 1.0, so nudge labels apart:
        # walk down from the top and push each one below the last if it is
        # within MIN_GAP. Leader lines are not needed at this separation.
        MIN_GAP = .075
        ends.sort(key=lambda e: -e[0])
        placed, prev = [], None
        for yv, name, colour in ends:
            yt = yv if prev is None else min(yv, prev - MIN_GAP)
            placed.append((yv, yt, name, colour)); prev = yt
        for yv, yt, name, colour in placed:
            ax.text(1.03, yt, name, color=colour, fontsize=6.8, ha='left',
                    va='center', clip_on=False, zorder=5)
            if abs(yt - yv) > .01:            # short tick back to the curve
                ax.plot([1.005, 1.025], [yv, yt], color=colour, lw=.5,
                        clip_on=False, zorder=5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.06)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Classification threshold')
    ax.set_title(title, fontsize=9, pad=5, loc='left')

    # cut values sit inside the axes; below it they collided with the ticks
    for t, ha in ((cuts['t_out'], 'right'), (cuts['t_in'], 'left')):
        ax.text(t + (.012 if ha == 'left' else -.012), .015, f'{t:.2f}',
                ha=ha, va='bottom', fontsize=6.4, color=INK)

    return cuts


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (title, path) in zip(axes, [
            ('ADNI', RESULTS / 'adni_loocv_predictions.csv'),
            ('A4 + LEARN', RESULTS / 'a4_binary_validation_predictions.csv')]):
        d = pd.read_csv(path)
        c = panel(ax, d.true_amyloid.values, d.predicted_prob.values, title,
                  label_curves=(title == 'A4 + LEARN'))
        print(f'{title:12s} rule-out < {c["t_out"]:.3f}   rule-in > {c["t_in"]:.3f}'
              f'   referred {c["mid"]:.1%}')
    axes[0].set_ylabel('Metric value')
    fig.subplots_adjust(wspace=.14, right=.90)   # margin for the curve labels
    save_panel(fig, PANELS / 'panel_threshold_tradeoff')
