"""
Gray-zone comparison: p-tau217 alone vs GRAD Stage 2, both cohorts.

Boxes are bootstrap distributions (2,000 resamples of the gray-zone
participants) and therefore show SAMPLING uncertainty in each metric, not
between-participant variability.

Significance is NOT taken from the bootstrap. Both methods are evaluated on the
same participants, so the correct test is McNemar's on the discordant pairs --
computed once on the observed data and annotated above each bracket.

Drawn in black and white: this panel contrasts two METHODS, not two patient
groups, so it deliberately stays out of the Abeta colour palette. Filled black
= GRAD Stage 2, open = p-tau217 alone.

Output: results/figures/panels/panel_gz_comparison.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from _grad_paths import PANELS, RESULTS                      # noqa: F401
from _grad_style import apply_style, save_panel, INK

apply_style(base=9)

N_BOOT = 2000
METRICS = ['Sensitivity', 'Specificity', 'Accuracy']
LW = 0.9                 # one weight for box, whisker and cap alike
COHORTS = [('ADNI (Development)', 'adni_loocv_predictions.csv'),
           ('A4 + LEARN (Validation)', 'a4_binary_validation_predictions.csv')]


def masks(y):
    """Which participants each metric is computed over."""
    return {'Sensitivity': y, 'Specificity': ~y,
            'Accuracy': np.ones(len(y), bool)}


def mcnemar(correct_a, correct_b):
    """Continuity-corrected McNemar on the discordant pairs."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return 1.0
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


def cohort(path):
    d = pd.read_csv(path)
    gz = d[d.stage.astype(str).str.contains('reflex')]
    y = gz.true_amyloid.values.astype(bool)
    ca = (gz.gatekeeper_prob.values >= .5) == y      # p-tau217 alone
    cb = (gz.predicted_prob.values >= .5) == y       # GRAD Stage 2
    return y, ca, cb


def bootstrap(y, ca, cb, rng):
    """Resample participants; recompute every metric for both methods."""
    out = {m: ([], []) for m in METRICS}
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        yb, ab, bb = y[i], ca[i], cb[i]
        for m, msk in masks(yb).items():
            if msk.sum() == 0:
                continue
            out[m][0].append(ab[msk].mean() * 100)
            out[m][1].append(bb[msk].mean() * 100)
    return out


def whisker_span(v):
    """Where matplotlib will actually draw the caps (1.5 x IQR rule)."""
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    inside = [x for x in v if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
    return min(inside), max(inside)


# ---- pass 1: bootstrap both cohorts first, so the shared y-axis can be sized
# from the whiskers that actually get drawn plus bracket headroom
rng = np.random.default_rng(42)
data = []
for title, fname in COHORTS:
    y, ca, cb = cohort(RESULTS / fname)
    data.append((title, y, ca, cb, bootstrap(y, ca, cb, rng)))

spans = [whisker_span(v) for _, _, _, _, b in data for m in METRICS for v in b[m]]
lo_data, hi_data = min(s[0] for s in spans), max(s[1] for s in spans)

BRACKET_GAP, BRACKET_H, STAR_PAD = 2.0, 1.4, 0.5
y_top = hi_data + BRACKET_GAP + BRACKET_H + STAR_PAD + 4.0   # room for the stars
y_bot = lo_data - 3

# ---- pass 2: draw
fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), sharey=True)

for ax, (title, y, ca, cb, boots) in zip(axes, data):
    for k, m in enumerate(METRICS):
        # open box = p-tau217 alone, filled box = GRAD; the median flips to
        # white on the filled box so it stays readable
        for j, vals in enumerate(boots[m]):
            filled = (j == 1)
            pos = k + (-.18 if j == 0 else .18)
            ax.boxplot(
                [vals], positions=[pos], widths=.28, showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=INK if filled else 'white',
                              edgecolor=INK, lw=LW),
                medianprops=dict(color='white' if filled else INK, lw=LW),
                whiskerprops=dict(color=INK, lw=LW),
                capprops=dict(color=INK, lw=LW))

        # bracket sits above the taller of the two drawn whiskers, so it can
        # never land inside a box
        msk = masks(y)[m]
        p = mcnemar(ca[msk], cb[msk])
        h = max(whisker_span(v)[1] for v in boots[m]) + BRACKET_GAP
        ax.plot([k - .18, k - .18, k + .18, k + .18],
                [h, h + BRACKET_H, h + BRACKET_H, h], lw=.7, color=INK)
        ax.text(k, h + BRACKET_H + STAR_PAD, stars(p), ha='center',
                va='bottom', fontsize=8, color=INK)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_xlim(-.5, len(METRICS) - .5)
    ax.set_ylim(y_bot, y_top)
    ax.set_title(title, pad=4, fontsize=8.5)
    ax.tick_params(axis='x', length=0, pad=3)

axes[0].set_ylabel('Performance in gray zone (%)')
handles = [plt.Rectangle((0, 0), 1, 1, fc='white', ec=INK, lw=LW),
           plt.Rectangle((0, 0), 1, 1, fc=INK, ec=INK, lw=LW)]
fig.legend(handles, ['p-tau217 alone', 'GRAD Stage 2'], loc='lower center',
           ncol=2, frameon=False, fontsize=7.5, handlelength=1.1)
fig.tight_layout(pad=.3)
fig.subplots_adjust(bottom=.24, wspace=.07)
save_panel(fig, str(PANELS / 'panel_gz_comparison'))
print('  panel_gz_comparison')

for title, fname in COHORTS:
    y, ca, cb = cohort(RESULTS / fname)
    for m, msk in masks(y).items():
        p = mcnemar(ca[msk], cb[msk])
        print(f"    {title[:4]:5s} {m:12s} {ca[msk].mean():6.1%} -> "
              f"{cb[msk].mean():6.1%}   McNemar p={p:.2e} {stars(p)}")
