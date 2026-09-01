#!/usr/bin/env python3
"""
Figure 4 panels A-D: health-economic model.
============================================

    A   Total cost per 10,000 patients, split into plasma and imaging spend
    B   Ab-PET scans required per 10,000 patients
    C   Cost per correct diagnosis, counting PET referrals as correct
    D   Sensitivity of total cost to the assumed Ab-PET unit price

Colour is shared with Figures 2 and 3 through _grad_style: GRAD is ROC_BLUE,
the first comparator SLATE, the second PALE. Values are read from the axes -
no printed labels on the bars - and all type is a single size in ink.

    python3 scripts/GRAD_fig4_panels.py
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _grad_paths import RESULTS, PROJECT_ROOT
from _grad_style import (apply_style, save_panel, PANEL_LETTER, ROC_BLUE,
                         SLATE, PALE, INK)

STRAT = [PALE, SLATE, ROC_BLUE]
LABELS = ['Universal\nAβ-PET', 'p-Tau217\n+ PET', 'GRAD\nstaged']
PANEL_W, PANEL_H = 2.72, 2.52
OUT = PROJECT_ROOT / 'results' / 'figures' / 'panels'
OUT.mkdir(parents=True, exist_ok=True)


def letter(ax, ch):
    ax.text(-0.30, 1.13, ch, transform=ax.transAxes, **PANEL_LETTER)


def frame(ax):
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(LABELS)
    ax.set_xlim(-.66, 2.66)
    ax.tick_params(axis='x', length=0)


def load():
    return json.load(open(RESULTS / 'grad_v2_numbers.json'))['cost']


def cost_breakdown(c, stem, ch):
    u, a4 = c['unit_costs'], c['a4']
    n, gz = u['n'], 0.3832
    plasma = np.array([0, n * u['plasma'], n * u['plasma'] + n * gz * u['addon']])
    total = np.array([a4[k]['cost'] for k in ('universal', 'ptau_first', 'grad')])

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    x = np.arange(3)
    ax.bar(x, plasma / 1e6, .62, color=ROC_BLUE, lw=0)
    ax.bar(x, (total - plasma) / 1e6, .62, bottom=plasma / 1e6, color=PALE, lw=0)
    ax.set_ylabel('Total cost per 10,000 ($M)')
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_title('Cost of each strategy', pad=7)
    ax.legend(handles=[Patch(facecolor=ROC_BLUE, label='Plasma'),
                       Patch(facecolor=PALE, label='Aβ-PET')],
              loc='upper right', handlelength=.9, handleheight=.9,
              borderpad=.15, labelspacing=.25, borderaxespad=.2)
    frame(ax)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


def scans_needed(c, stem, ch):
    vals = [c['a4'][k]['scans'] for k in ('universal', 'ptau_first', 'grad')]
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.bar(np.arange(3), vals, .62, color=STRAT, lw=0)
    ax.set_ylabel('Aβ-PET scans per 10,000')
    ax.set_ylim(0, 10600)
    ax.set_yticks([0, 2500, 5000, 7500, 10000])
    ax.set_yticklabels(['0', '2,500', '5,000', '7,500', '10,000'])
    ax.set_title('Imaging burden', pad=7)
    frame(ax)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


def cost_per_correct(c, stem, ch):
    vals = [c['a4'][k]['cost_per_correct'] for k in ('universal', 'ptau_first', 'grad')]
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.bar(np.arange(3), vals, .62, color=STRAT, lw=0)
    ax.set_ylabel('Cost per correct diagnosis ($)')
    ax.set_ylim(0, 3200)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(['0', '1,000', '2,000', '3,000'])
    ax.set_title('Cost per correct diagnosis', pad=7)
    frame(ax)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


def price_sensitivity(c, stem, ch):
    u = c['unit_costs']
    n, gz, resid = u['n'], 0.3832, 0.0931
    prices = np.linspace(1000, 5000, 200)
    series = [('Universal Aβ-PET', n * prices, PALE),
              ('p-Tau217 + PET', n * u['plasma'] + n * gz * prices, SLATE),
              ('GRAD staged',
               n * u['plasma'] + n * gz * u['addon'] + n * resid * prices, ROC_BLUE)]

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    for _, ycost, colour in series:
        ax.plot(prices, ycost / 1e6, color=colour, lw=2.0, solid_capstyle='round')
    ax.axvline(u['pet'], color=PALE, lw=.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(1000, 5000)
    ax.set_ylim(0, 52)
    ax.set_xticks([1000, 3000, 5000])
    ax.set_xticklabels(['$1k', '$3k', '$5k'])
    ax.set_yticks([0, 20, 40])
    ax.set_xlabel('Assumed Aβ-PET unit price')
    ax.set_ylabel('Total cost per 10,000 ($M)')
    ax.set_title('Sensitivity to scan price', pad=7)
    ax.legend(handles=[Line2D([], [], color=col, lw=2.0, label=lab)
                       for lab, _, col in series[::-1]],
              loc='upper left', handlelength=1.2, borderpad=.15,
              labelspacing=.25, borderaxespad=.2)
    letter(ax, ch)
    return save_panel(fig, str(OUT / stem))


def main():
    apply_style()
    c = load()
    for w in [cost_breakdown(c, 'fig4a_cost_breakdown', 'A'),
              scans_needed(c, 'fig4b_scans', 'B'),
              cost_per_correct(c, 'fig4c_cost_per_correct', 'C'),
              price_sensitivity(c, 'fig4d_price_sensitivity', 'D')]:
        print('wrote', w)


if __name__ == '__main__':
    main()
