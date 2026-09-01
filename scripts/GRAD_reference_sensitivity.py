#!/usr/bin/env python3
"""
Sensitivity of the external validation to how the A4 reference set is defined.
==============================================================================

The primary analysis anchors each biomarker to the cognitively unimpaired,
Aβ-negative participants *within* A4 + LEARN. Those participants are also in the
evaluation set, so the transformation applied to the validation cohort is
estimated using validation-set outcome labels. A reviewer can reasonably call
that outcome-informed preprocessing.

This script re-scores A4 under reference definitions that use progressively less
outcome information, holding the ADNI-fitted model completely fixed:

    cu_abneg      primary - mean/SD of log1p among the 499 Aβ-negative
    low_tercile   label-free - mean/SD among the bottom third of the p-Tau217
                  distribution, a presumed-normal anchor a laboratory could
                  define without any PET result
    robust_all    label-free - median and IQR/1.349 among all 1,644
    mean_all      label-free - mean/SD among all 1,644
    split_half    reference from a random half of the Aβ-negative group,
                  repeated, to show sampling stability of the estimate itself

Two of the headline numbers are invariant by construction and act as a check on
the implementation: Stage 1 AUC is a monotone function of raw p-Tau217 whatever
mu and sigma are chosen, and ADNI never uses A4 at all. What can move is the
routing - the gray zone is an interval on the probability scale, so mu and sigma
decide which raw values land inside it - and everything downstream of it.

    python3 scripts/GRAD_reference_sensitivity.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _grad_paths import RESULTS
from run_grad_impaired_logistic import (
    load_adni, load_a4, harmonise, engineer, fit_stage1, fit_stage2,
    score_stage2, HARMONISED, IMPAIRED_DX, FEATURES, GK_LOW, GK_HIGH,
)

NSPLIT = 200
SEED = 42


# ------------------------------------------------------- reference variants ---

def ref_stats(a4, how, rng=None):
    """{biomarker: {assay: (mu, sigma)}} under one reference definition."""
    if how == 'cu_abneg':
        mask = a4['amyloid_positive'] == 0
    elif how in ('robust_all', 'mean_all'):
        mask = pd.Series(True, index=a4.index)
    elif how == 'low_tercile':
        # presumed-normal anchor from the screening marker alone, no labels
        cut = a4['pTau217_raw'].quantile(1 / 3)
        mask = a4['pTau217_raw'] <= cut
    elif how == 'split_half':
        neg = a4.index[a4['amyloid_positive'] == 0]
        keep = rng.choice(neg, size=len(neg) // 2, replace=False)
        mask = pd.Series(a4.index.isin(keep), index=a4.index)
    else:
        raise ValueError(how)

    stats = {}
    for bm in HARMONISED:
        if bm not in a4.columns:
            continue
        per_assay = {}
        for assay in a4['assay'].dropna().unique():
            v = np.log1p(a4.loc[mask & (a4['assay'] == assay), bm].dropna())
            if len(v) < 5:
                v = np.log1p(a4.loc[mask, bm].dropna())
            if how == 'robust_all':
                mu = v.median()
                sd = (v.quantile(.75) - v.quantile(.25)) / 1.349
            else:
                mu, sd = v.mean(), v.std()
            per_assay[assay] = (mu, sd if sd and np.isfinite(sd) and sd > 0 else 1.0)
        stats[bm] = per_assay
    return stats


def apply_ref(a4, stats):
    out = a4.copy()
    for bm, per_assay in stats.items():
        z = bm.replace('_raw', '_Z')
        out[z] = np.nan
        for assay, (mu, sd) in per_assay.items():
            m = a4['assay'] == assay
            out.loc[m, z] = (np.log1p(a4.loc[m, bm]) - mu) / sd
    return engineer(out)


# -------------------------------------------------------------- evaluation ---

def fit_adni(adni, idx):
    """The ADNI-fitted model, identical to the primary analysis."""
    adni_h = engineer(harmonise(adni, adni))
    train = adni_h.loc[idx]
    y = train['amyloid_positive'].values
    gk = fit_stage1(train, y)
    p = gk.predict_proba(train[['pTau217_Z']].values)[:, 1]
    gz = (p >= GK_LOW) & (p <= GK_HIGH)
    m2, sc, med, used = fit_stage2(train[gz], y[gz], features=FEATURES)
    return gk, (m2, sc, med, used)


def score(a4_h, gk, s2):
    m2, sc, med, used = s2
    y = a4_h['amyloid_positive'].values
    p1 = gk.predict_proba(a4_h[['pTau217_Z']].values)[:, 1]
    gz = (p1 >= GK_LOW) & (p1 <= GK_HIGH)
    final = p1.copy()
    if gz.sum():
        final[gz] = score_stage2(m2, sc, med, used, a4_h[gz])
    return dict(
        stage1_auc_all=roc_auc_score(y, p1),
        stage1_auc=roc_auc_score(y[~gz], p1[~gz]),
        gz_n=int(gz.sum()),
        gz_pct=100 * gz.mean(),
        stage2_auc=roc_auc_score(y[gz], final[gz]) if gz.sum() else np.nan,
        pipeline_auc=roc_auc_score(y, final),
        accuracy=100 * ((final >= .5).astype(int) == y).mean(),
    )


def main():
    adni, a4 = load_adni(), load_a4()
    idx = adni.index[adni['DX'].isin(IMPAIRED_DX)]
    gk, s2 = fit_adni(adni, idx)

    rows = []
    for how in ('cu_abneg', 'low_tercile', 'robust_all', 'mean_all'):
        r = score(apply_ref(a4, ref_stats(a4, how)), gk, s2)
        r['variant'] = how
        rows.append(r)
        print(f"{how:12s} s1all {r['stage1_auc_all']:.4f}  "
              f"stage1 {r['stage1_auc']:.4f}  gz {r['gz_pct']:5.1f}%  "
              f"stage2 {r['stage2_auc']:.4f}  pipeline {r['pipeline_auc']:.4f}  "
              f"acc {r['accuracy']:.1f}")

    rng = np.random.RandomState(SEED)
    splits = [score(apply_ref(a4, ref_stats(a4, 'split_half', rng)), gk, s2)
              for _ in range(NSPLIT)]
    sp = pd.DataFrame(splits)
    summary = {k: dict(median=float(sp[k].median()),
                       lo=float(sp[k].quantile(.025)),
                       hi=float(sp[k].quantile(.975)))
               for k in sp.columns}
    print(f"split_half   stage1 {summary['stage1_auc']['median']:.4f} "
          f"[{summary['stage1_auc']['lo']:.4f}, {summary['stage1_auc']['hi']:.4f}]  "
          f"gz {summary['gz_pct']['median']:5.1f}% "
          f"[{summary['gz_pct']['lo']:.1f}, {summary['gz_pct']['hi']:.1f}]  "
          f"pipeline {summary['pipeline_auc']['median']:.4f} "
          f"[{summary['pipeline_auc']['lo']:.4f}, {summary['pipeline_auc']['hi']:.4f}]")

    out = dict(fixed=rows, split_half=summary, n_splits=NSPLIT)
    path = RESULTS / 'reference_sensitivity.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print('\nwrote', path)


if __name__ == '__main__':
    main()
