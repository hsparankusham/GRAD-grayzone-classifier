#!/usr/bin/env python3
"""
External calibration of the GRAD probability in A4 + LEARN.

TRIPOD asks an external validation to report calibration alongside
discrimination, and here the two answers differ, which is the point of the
panel. A4 + LEARN is ~70% amyloid-positive against ADNI's ~48%, so a model
carrying ADNI's prevalence under-predicts risk in a trial-screening cohort.
Discrimination transports; calibration-in-the-large does not, and would be
corrected at deployment by a one-parameter intercept shift.

Reported:
  Brier score            overall accuracy of the probabilities
  Calibration slope      logistic refit on logit(p); 1.0 = correct spread
  Calibration intercept  refit with the slope offset; 0 = correct average risk

Output: results/figures/panels/panel_a4_calibration.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import beta

from _grad_paths import PANELS, RESULTS
from _grad_style import apply_style, save_panel, INK, ABETA_NEG, ABETA_POS, ROC_BLUE

apply_style(base=9)

N_BINS = 10
EPS = 1e-6


def calibration_stats(y, p):
    """Brier, plus the standard calibration slope and intercept."""
    brier = np.mean((p - y) ** 2)
    logit = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    slope = sm.Logit(y, sm.add_constant(logit)).fit(disp=0).params[1]
    # intercept with the linear predictor held at slope 1 (calibration-in-the-large)
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                       offset=logit).fit().params[0]
    return brier, slope, intercept


def binned(y, p, n_bins=N_BINS):
    """Equal-count bins; Jeffreys interval on each observed fraction."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:
            continue
        k, n = int(y[m].sum()), int(m.sum())
        rows.append(dict(x=p[m].mean(), obs=k / n, n=n,
                         lo=beta.ppf(.025, k + .5, n - k + .5),
                         hi=beta.ppf(.975, k + .5, n - k + .5)))
    return pd.DataFrame(rows)


d = pd.read_csv(RESULTS / 'a4_binary_validation_predictions.csv')
y, p = d.true_amyloid.values.astype(float), d.predicted_prob.values
brier, slope, intercept = calibration_stats(y, p)
b = binned(y, p)

fig, ax = plt.subplots(figsize=(2.9, 2.9))

# perfect calibration: thin solid black, drawn under the data
ax.plot([0, 1], [0, 1], lw=.6, color=INK, zorder=1)
ax.errorbar(b.x, b.obs, yerr=[b.obs - b.lo, b.hi - b.obs], fmt='none',
            ecolor=INK, elinewidth=.7, capsize=2, capthick=.7, zorder=3)
ax.plot(b.x, b.obs, '-o', color=ROC_BLUE, lw=1.4, ms=3.6, zorder=4)

# marginal distribution of the predictions, split by true status; the Abeta
# colours are the same ones every other panel uses for these two groups
base, height = -.11, .085
for m, colour in ((y == 0, ABETA_NEG), (y == 1, ABETA_POS)):
    h, edges = np.histogram(p[m], bins=30, range=(0, 1))
    ax.bar(edges[:-1], h / h.max() * height, bottom=base, width=np.diff(edges),
           align='edge', color=colour, alpha=.55, lw=0, zorder=2)

ax.set_xlabel('Predicted probability')
ax.set_ylabel('Observed fraction Aβ+')
ax.set_xlim(-.02, 1.02); ax.set_ylim(base - .015, 1.02)
ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
ax.spines['left'].set_bounds(0, 1)

# unboxed, black annotation
ax.text(.04, .97, f'Brier {brier:.3f}\nSlope {slope:.2f}\nIntercept {intercept:+.2f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=7,
        linespacing=1.5, color=INK)

save_panel(fig, PANELS / 'panel_a4_calibration')
print(f'  panel_a4_calibration   Brier {brier:.3f}  slope {slope:.3f} '
      f' intercept {intercept:+.3f}')
print(f'  observed prevalence {y.mean():.1%}   mean predicted {p.mean():.1%}')
