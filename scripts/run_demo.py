#!/usr/bin/env python3
"""
GRAD Demo: Run pipeline on synthetic data
==========================================
Demonstrates the full GRAD two-stage algorithm using the synthetic
demo dataset (no ADNI/A4 access required).

Usage:
    python scripts/run_demo.py

Expected runtime: ~2 minutes on a standard desktop
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

SEED = 42
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def main():
    print("=" * 65)
    print("  GRAD Demo: Gatekeeper-Reflex for Alzheimer's Diagnostics")
    print("  Using synthetic data (no ADNI/A4 access required)")
    print("=" * 65)

    # ── Load synthetic data ───────────────────────────────────────────
    data_path = Path(__file__).parent.parent / 'data' / 'synthetic' / 'synthetic_cohort.csv'
    if not data_path.exists():
        print(f"\nERROR: Synthetic data not found at {data_path}")
        print("Run: python scripts/generate_synthetic_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nLoaded synthetic cohort: N={len(df)} "
          f"({df.amyloid_positive.sum()} A+, {(1-df.amyloid_positive).sum():.0f} A-)")

    # ── Harmonization (Z-score) ───────────────────────────────────────
    print("\n--- Stage 0: Harmonization ---")
    # Reference population: CN amyloid-negative
    ref_mask = (df['DX'] == 'CN') & (df['amyloid_positive'] == 0)
    ref = df[ref_mask]
    print(f"Reference population (CN, A-): N={len(ref)}")

    biomarkers = ['pTau217_raw', 'GFAP_raw', 'NfL_raw']
    for bm in biomarkers:
        log_vals = np.log1p(ref[bm].dropna())
        mu, sigma = log_vals.mean(), log_vals.std()
        z_col = bm.replace('_raw', '_Z')
        df[z_col] = (np.log1p(df[bm]) - mu) / sigma

    # Feature engineering
    df['tau_ab42_diff'] = np.log1p(df['pTau217_raw']) - np.log1p(df['AB42_40_ratio'])
    df['gfap_tau_interaction'] = df['GFAP_Z'] * df['pTau217_Z']
    # AGE is passed through directly (RF is scale-invariant)

    # ── LOOCV ─────────────────────────────────────────────────────────
    print("\n--- Running Leave-One-Out Cross-Validation ---")
    N = len(df)
    y = df['amyloid_positive'].values
    predictions = np.zeros(N)
    stages = []

    FEATURES = ['pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
                'APOE4_carrier', 'gfap_tau_interaction']

    for i in range(N):
        if i % 50 == 0:
            print(f"  Fold {i+1}/{N}...")

        # Train/test split
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False
        train = df[train_mask]
        test = df.iloc[[i]]

        # Re-harmonize within fold (prevent leakage)
        ref_fold = train[(train['DX'] == 'CN') & (train['amyloid_positive'] == 0)]
        for bm in biomarkers:
            log_vals = np.log1p(ref_fold[bm].dropna())
            mu, sigma = log_vals.mean(), log_vals.std()
            z_col = bm.replace('_raw', '_Z')
            train = train.copy()
            test = test.copy()
            train[z_col] = (np.log1p(train[bm]) - mu) / sigma
            test[z_col] = (np.log1p(test[bm]) - mu) / sigma

        # Re-engineer features
        train['tau_ab42_diff'] = np.log1p(train['pTau217_raw']) - np.log1p(train['AB42_40_ratio'])
        test['tau_ab42_diff'] = np.log1p(test['pTau217_raw']) - np.log1p(test['AB42_40_ratio'])
        train['gfap_tau_interaction'] = train['GFAP_Z'] * train['pTau217_Z']
        test['gfap_tau_interaction'] = test['GFAP_Z'] * test['pTau217_Z']
        # AGE passed through directly (no standardization needed for RF)

        # Stage 1: Gatekeeper
        gk = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                max_iter=1000, random_state=SEED)
        gk.fit(train[['pTau217_Z']].values, train['amyloid_positive'].values)
        gk_prob = gk.predict_proba(test[['pTau217_Z']].values)[0, 1]

        if gk_prob < 0.25 or gk_prob > 0.75:
            predictions[i] = gk_prob
            stages.append('gatekeeper')
        else:
            # Stage 2: Reflex
            X_train_gz = train[(train['pTau217_Z'].between(
                *np.percentile(train['pTau217_Z'], [10, 90])))]
            # Use all training data for Reflex if gray zone too small
            if len(X_train_gz) < 20:
                X_train_gz = train

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_gz[FEATURES].fillna(0).values)
            y_tr = X_train_gz['amyloid_positive'].values
            X_te = scaler.transform(test[FEATURES].fillna(0).values)

            rf = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=5,
                class_weight='balanced', random_state=SEED
            )
            rf.fit(X_tr, y_tr)
            predictions[i] = rf.predict_proba(X_te)[0, 1]
            stages.append('reflex')

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DEMO RESULTS (Synthetic Data)")
    print("=" * 65)

    auc = roc_auc_score(y, predictions)
    pred_class = (predictions >= 0.5).astype(int)
    acc = accuracy_score(y, pred_class)
    tn, fp, fn, tp = confusion_matrix(y, pred_class).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    gk_count = stages.count('gatekeeper')
    gk_rate = gk_count / N * 100

    print(f"\n  AUC:          {auc:.4f}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Sensitivity:  {sens:.4f}")
    print(f"  Specificity:  {spec:.4f}")
    print(f"  Gatekeeper:   {gk_count}/{N} resolved ({gk_rate:.1f}%)")
    print(f"  Reflex:       {N - gk_count}/{N} gray zone ({100-gk_rate:.1f}%)")

    print(f"\n  NOTE: These results are from SYNTHETIC data and will")
    print(f"  differ from the manuscript values (ADNI AUC = 0.853).")
    print(f"  Use real ADNI/A4 data to reproduce manuscript results.")
    print("=" * 65)


if __name__ == '__main__':
    main()
