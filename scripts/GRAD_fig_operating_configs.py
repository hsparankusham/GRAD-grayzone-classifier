"""
Clinical operating configurations.

GRAD returns a probability, so a site can place the rule-out and rule-in cuts
wherever the clinical question demands. This panel shows what three defensible
configurations cost and buy, in both cohorts:

  Balanced triage   90% sensitivity / 90% specificity
  Screening         95% sensitivity  -- do not miss amyloid-positive candidates
  DMT initiation    95% specificity  -- do not expose Abeta-negative patients
                                        to anti-amyloid therapy and ARIA risk

Cuts are derived from the ROC of the final GRAD probability in each cohort.
Patients between the two cuts are referred for confirmatory Abeta-PET.

Output: results/figures/panels/panel_operating_configs.{png,pdf}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from _grad_paths import PANELS, RESULTS                      # noqa: F401
from _grad_style import (apply_style, save_panel, ABETA_NEG, ABETA_POS,
                         INK, GUIDE, GRID)

apply_style(base=9)
PET = '#D9C77E'          # neutral amber: neither an Abeta+ nor an Abeta- call

CONFIGS = [
    ('DMT initiation\n95% specificity',   .90, .95),
    ('Screening\n95% sensitivity',        .95, .90),
    ('Balanced triage\n90% / 90%',        .90, .90),
]


def split(y, p, sens_t, spec_t):
    """Place rule-out / rule-in cuts on the ROC, then partition the cohort."""
    fpr, tpr, thr = roc_curve(y, p)
    t_out = thr[np.where(tpr >= sens_t)[0][0]]
    t_in = thr[np.where((1 - fpr) >= spec_t)[0][-1]]
    out, mid, inn = p < t_out, (p >= t_out) & (p <= t_in), p > t_in
    return dict(
        t_out=t_out, t_in=t_in,
        out=out.mean(), mid=mid.mean(), inn=inn.mean(),
        npv=(y[out] == 0).mean() if out.sum() else np.nan,
        ppv=(y[inn] == 1).mean() if inn.sum() else np.nan,
    )


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, (title, path) in zip(axes, [
            ('ADNI (Development)', RESULTS / 'adni_loocv_predictions.csv'),
            ('A4 + LEARN (Validation)', RESULTS / 'a4_binary_validation_predictions.csv')]):
        d = pd.read_csv(path)
        y, p = d.true_amyloid.values, d.predicted_prob.values

        for i, (label, st, sp) in enumerate(CONFIGS):
            r = split(y, p, st, sp)
            left = 0
            for frac, colour in ((r['out'], ABETA_NEG), (r['mid'], PET), (r['inn'], ABETA_POS)):
                ax.barh(i, frac * 100, left=left * 100, height=.58, color=colour,
                        edgecolor='white', linewidth=.9, zorder=3)
                left += frac
            # segment labels, only where a label fits
            if r['out'] > .09:
                ax.text(r['out'] * 50, i, f"{r['out']:.0%}", ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold', zorder=4)
            if r['mid'] > .09:
                ax.text((r['out'] + r['mid'] / 2) * 100, i, f"{r['mid']:.0%}",
                        ha='center', va='center', fontsize=7.5, color=INK, zorder=4)
            if r['inn'] > .09:
                ax.text((r['out'] + r['mid'] + r['inn'] / 2) * 100, i, f"{r['inn']:.0%}",
                        ha='center', va='center', fontsize=7.5, color='white',
                        fontweight='bold', zorder=4)
            # predictive values sit outside the bar so they never crowd it
            ax.text(101, i, f"NPV {r['npv']:.0%}   PPV {r['ppv']:.0%}", va='center',
                    fontsize=6.8, color=GUIDE)

        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels([c[0] for c in CONFIGS], fontsize=8)
        ax.set_xlim(0, 100); ax.set_ylim(-.6, len(CONFIGS) - .4)
        ax.set_xlabel('Patients (%)')
        ax.set_title(title, pad=7)
        ax.grid(axis='x', color=GRID, lw=.5); ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='white')
               for c in (ABETA_NEG, PET, ABETA_POS)]
    axes[0].legend(handles, ['Ruled out (Aβ−)', 'Referred for PET', 'Ruled in (Aβ+)'],
                   loc='upper center', bbox_to_anchor=(1.05, -.22), ncol=3,
                   fontsize=7.5, handlelength=1.2)
    fig.tight_layout(pad=.5)
    fig.subplots_adjust(right=.86, bottom=.30)
    save_panel(fig, str(PANELS / 'panel_operating_configs'))
    print('  panel_operating_configs')

    for nm, path in [('ADNI', RESULTS / 'adni_loocv_predictions.csv'),
                     ('A4', RESULTS / 'a4_binary_validation_predictions.csv')]:
        d = pd.read_csv(path); y, p = d.true_amyloid.values, d.predicted_prob.values
        print(f"  {nm}")
        for label, st, sp in CONFIGS:
            r = split(y, p, st, sp)
            print(f"    {label.replace(chr(10),' '):34s} no PET {r['out']+r['inn']:5.1%}"
                  f"   PET {r['mid']:5.1%}   NPV {r['npv']:.1%}  PPV {r['ppv']:.1%}")
