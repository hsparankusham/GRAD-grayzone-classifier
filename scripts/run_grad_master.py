#!/usr/bin/env python3
"""
================================================================================
GRAD: Gatekeeper-Reflex Algorithm for Amyloid Determination
================================================================================
Master Script — Complete End-to-End Pipeline in One File

This script runs the ENTIRE GRAD process from raw data to final metrics:
  1. Load ADNI + A4 data
  2. Z-score harmonize biomarkers (cross-platform normalization)
  3. Engineer biologically motivated features
  4. Stage 1: Gatekeeper (univariate p-tau217 logistic regression)
  5. Stage 2: Reflex (6-feature Random Forest for gray zone)
  6. LOOCV internal validation on ADNI (320 iterations)
  7. External validation on A4+LEARN (frozen ADNI parameters)
  8. Compute all metrics, operating points, bootstrap CIs

All algorithm logic is inlined. No imports from harmonizer.py, gatekeeper.py,
reflex.py, pipeline.py, or validation.py. Data loading uses data_loader.py
(that's just CSV parsing, not algorithm logic).

Reference:
    Parankusham et al. (2026). Resolving Amyloid Status Uncertainty in
    Intermediate Plasma p-tau217 Levels. Alzheimer's & Dementia.

Author: Harthik S. Parankusham
================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix, accuracy_score
from scipy import stats

warnings.filterwarnings('ignore')

# Add current directory to path for data_loader import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ADNIDataLoader, A4DataLoader


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Amyloid PET thresholds
AV45_THRESHOLD = 1.11          # ADNI: AV45 SUVR > 1.11 = amyloid-positive
CENTILOID_THRESHOLD = 20       # A4: Centiloid >= 20 = amyloid-positive

# Gatekeeper probability thresholds (pre-specified, not optimized)
GK_LOW = 0.25                  # Below this = confident amyloid-negative
GK_HIGH = 0.75                 # Above this = confident amyloid-positive
                               # Between = gray zone -> routed to Reflex

# The locked 6-feature set for the Reflex model
# Selected via domain knowledge, NOT by data-driven search
REFLEX_FEATURES = [
    'pTau217_Z',               # Primary biomarker (harmonized p-tau217)
    'tau_ab42_diff',           # Tau-amyloid divergence: log(pTau217) - log(AB42/40)
    'GFAP_Z',                 # Astrogliosis marker (harmonized GFAP)
    'AGE',                    # Raw age in years (no standardization needed for RF)
    'APOE4_carrier',          # Binary APOE epsilon-4 carrier status
    'gfap_tau_interaction',   # Inflammation x pathology: GFAP_Z * pTau217_Z
]

# Biomarkers to harmonize
BIOMARKERS = ['pTau217_raw', 'NfL_raw', 'GFAP_raw']

# Random Forest hyperparameters (all set a priori, not tuned)
RF_N_ESTIMATORS = 100          # Number of trees
RF_MAX_DEPTH = 5               # Hard cap on tree depth (prevents overfitting)
RF_MIN_SAMPLES_LEAF = 5        # Each leaf needs >= 5 samples
RF_RANDOM_STATE = 42           # Reproducibility seed

# Bootstrap parameters
N_BOOTSTRAP = 2000

# Output directory
RESULTS_DIR = Path(__file__).parent / 'results'


# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================
# Uses data_loader.py for CSV parsing. This is mechanical file I/O,
# not algorithm logic. See data_loader.py for column mappings.

def load_data():
    """
    Load ADNI and A4 datasets.

    ADNI sources:
      - UPENN plasma biomarkers: pTau217, AB42, AB40, NfL, GFAP
      - Janssen plasma p-tau217 (different assay platform)
      - ADNIMERGE: demographics, APOE4, AV45 amyloid PET

    A4 sources:
      - Lilly pTau217 (third assay platform)
      - Roche GFAP, NfL, AB42, AB40
      - SUBJINFO: demographics, APOE4, centiloid amyloid PET
    """
    print("=" * 70)
    print("SECTION 1: DATA LOADING")
    print("=" * 70)

    # Resolve paths: 4 levels up from grayzone-classifier to project root
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    data_dir = os.path.join(base, 'syntropi-ai-data')

    # Load ADNI
    print("\nLoading ADNI...")
    adni_loader = ADNIDataLoader(os.path.join(data_dir, 'syntropi-ai-ADNI'))
    adni_df = adni_loader.merge_data(use_baseline_only=True)

    # Load A4
    print("\nLoading A4...")
    a4_loader = A4DataLoader(os.path.join(data_dir, 'syntropi-ai-A4'))
    a4_df = a4_loader.merge_data(use_baseline_only=True)

    # Filter A4 to subjects with amyloid labels (need ground truth for validation)
    a4_df = a4_df[a4_df['amyloid_positive'].notna()].copy()
    a4_df['amyloid_positive'] = a4_df['amyloid_positive'].astype(int)

    print(f"\n  ADNI: {len(adni_df)} participants, "
          f"{int(adni_df['amyloid_positive'].sum())} amyloid+ "
          f"({adni_df['amyloid_positive'].mean()*100:.1f}%)")
    print(f"  A4:   {len(a4_df)} participants, "
          f"{int(a4_df['amyloid_positive'].sum())} amyloid+ "
          f"({a4_df['amyloid_positive'].mean()*100:.1f}%)")

    return adni_df, a4_df


# ==============================================================================
# SECTION 2: HARMONIZATION (Z-Score Normalization)
# ==============================================================================
# Purpose: Convert raw biomarker concentrations (pg/mL) into platform-agnostic
# Z-scores so that UPENN, Janssen, and Lilly assays are comparable.
#
# Formula: Z = (log(1 + raw_value) - mu_ref) / sigma_ref
#
# Reference population: Cognitively normal (CN) + amyloid-negative subjects
# within each assay platform. This anchors Z=0 to "healthy normal" regardless
# of which assay was used.
#
# CRITICAL FOR LEAKAGE PREVENTION:
#   - During LOOCV: mu/sigma are recomputed from 319 training subjects each fold
#   - During A4 validation: mu/sigma are frozen from ADNI, applied to A4

def fit_harmonizer(train_df):
    """
    Compute harmonization parameters from training data.

    Identifies the CN amyloid-negative reference population within each assay,
    log-transforms their biomarker values, and computes mean/std.

    Also computes a pooled (overall) reference for use when applying to
    external cohorts with a different assay platform (e.g., A4's Lilly assay).

    Args:
        train_df: Training DataFrame (e.g., 319 ADNI subjects in LOOCV)

    Returns:
        Dictionary: {assay: {biomarker: (mean, std)}, '_pooled': {biomarker: (mean, std)}}
    """
    params = {}

    # First: compute pooled reference across ALL assays (for external validation)
    pooled_ref_mask = pd.Series(True, index=train_df.index)
    if 'DX' in train_df.columns:
        cn_mask = (train_df['DX'].isin(['CN', 'NL', 'Normal']) |
                   train_df.get('DX_bl', pd.Series(False, index=train_df.index))
                   .isin(['CN', 'NL', 'Normal']))
        pooled_ref_mask = pooled_ref_mask & cn_mask
    if 'amyloid_positive' in train_df.columns:
        pooled_ref_mask = pooled_ref_mask & (train_df['amyloid_positive'] == 0)

    pooled_ref = train_df[pooled_ref_mask]
    if len(pooled_ref) < 5:
        pooled_ref = train_df

    params['_pooled'] = {}
    for biomarker in BIOMARKERS:
        if biomarker not in train_df.columns:
            continue
        values = pooled_ref[biomarker].dropna()
        if len(values) > 0:
            log_values = np.log1p(values)
            mu = log_values.mean()
            sigma = log_values.std() if log_values.std() > 0 else 1.0
            params['_pooled'][biomarker] = (mu, sigma, len(values))

    # Then: compute per-assay reference (for within-cohort harmonization)
    for assay in train_df['assay'].unique():
        params[assay] = {}
        assay_mask = train_df['assay'] == assay

        # Reference population: CN + amyloid-negative within this assay
        ref_mask = assay_mask.copy()

        # Filter to cognitively normal
        if 'DX' in train_df.columns:
            cn_mask = (train_df['DX'].isin(['CN', 'NL', 'Normal']) |
                       train_df.get('DX_bl', pd.Series(False, index=train_df.index))
                       .isin(['CN', 'NL', 'Normal']))
            ref_mask = ref_mask & cn_mask

        # Filter to amyloid-negative
        if 'amyloid_positive' in train_df.columns:
            ref_mask = ref_mask & (train_df['amyloid_positive'] == 0)

        ref_pop = train_df[ref_mask]

        # Fallback: if < 5 reference subjects, use all subjects from this assay
        if len(ref_pop) < 5:
            ref_pop = train_df[assay_mask]

        # Compute log-space mean and std for each biomarker
        for biomarker in BIOMARKERS:
            if biomarker not in train_df.columns:
                continue
            values = ref_pop[biomarker].dropna()
            if len(values) > 0:
                log_values = np.log1p(values)  # log(1 + x) handles zeros gracefully
                mu = log_values.mean()
                sigma = log_values.std() if log_values.std() > 0 else 1.0
                params[assay][biomarker] = (mu, sigma, len(values))

    return params


def apply_harmonizer(df, params):
    """
    Apply Z-score harmonization using pre-computed parameters.

    This is the "transform" step. For each subject, their raw biomarker
    value is converted to a Z-score using the reference population's
    mean and std from their assay platform.

    Args:
        df: DataFrame to harmonize
        params: Output from fit_harmonizer()

    Returns:
        DataFrame with new columns: pTau217_Z, NfL_Z, GFAP_Z
    """
    result = df.copy()

    for biomarker in BIOMARKERS:
        if biomarker not in df.columns:
            continue

        z_col = biomarker.replace('_raw', '_Z')  # e.g., pTau217_raw -> pTau217_Z
        result[z_col] = np.nan

        for assay in df['assay'].unique():
            mask = df['assay'] == assay

            if assay in params and biomarker in params[assay]:
                # Use assay-specific reference parameters
                mu, sigma, _ = params[assay][biomarker]
            else:
                # Assay not in training params (e.g., A4's Lilly assay).
                # Use pooled ADNI reference: overall CN amyloid-negative
                # mean/std across all training assays. This matches how
                # run_a4_binary_validation.py harmonizes external cohorts.
                if '_pooled' in params and biomarker in params['_pooled']:
                    mu, sigma, _ = params['_pooled'][biomarker]
                else:
                    # Last resort: use current data's own stats
                    values = df.loc[mask, biomarker]
                    log_values = np.log1p(values)
                    mu = log_values.mean()
                    sigma = log_values.std() if log_values.std() > 0 else 1.0

            # Apply: Z = (log(1 + raw_value) - mu) / sigma
            result.loc[mask, z_col] = (np.log1p(df.loc[mask, biomarker]) - mu) / sigma

    return result


# ==============================================================================
# SECTION 3: FEATURE ENGINEERING
# ==============================================================================
# Creates biologically motivated features from harmonized biomarkers.
# Each feature captures a specific aspect of AD pathophysiology.
#
# LEAKAGE NOTE:
#   - tau_ab42_diff: pure algebra on raw values, no population parameters
#   - gfap_tau_interaction: product of already-harmonized Z-scores
#   - AGE: uses mean/std from whatever df is passed in (training fold in LOOCV)

def engineer_features(df):
    """
    Create the 6 features used by the Reflex model.

    Args:
        df: DataFrame with harmonized biomarkers (pTau217_Z, GFAP_Z, etc.)

    Returns:
        DataFrame with additional engineered feature columns
    """
    result = df.copy()

    # Feature 1: tau_ab42_diff — Tau-amyloid divergence
    # BIOLOGICAL RATIONALE: In amyloid-positive patients, p-tau217 rises while
    # AB42/40 ratio drops. The difference between their log-transformed values
    # captures this divergence pattern. No population parameters needed.
    if 'pTau217_raw' in df.columns and 'AB42_40_ratio' in df.columns:
        result['tau_ab42_diff'] = (
            np.log1p(df['pTau217_raw']) - np.log1p(df['AB42_40_ratio'])
        )

    # Feature 2: gfap_tau_interaction — Inflammation x Pathology synergy
    # BIOLOGICAL RATIONALE: GFAP (astrocyte reactivity) x pTau217 (tau pathology).
    # High values on BOTH markers together are more predictive than either alone.
    # A patient with high GFAP but low pTau217 gets ~0 (inflammation without tau).
    if 'GFAP_Z' in df.columns and 'pTau217_Z' in df.columns:
        result['gfap_tau_interaction'] = df['GFAP_Z'] * df['pTau217_Z']

    # Feature 3: AGE — Raw age in years
    # BIOLOGICAL RATIONALE: Age is the #1 risk factor for AD.
    # No standardization needed: Random Forest splits are rank-based and
    # invariant to monotonic transformations. Using raw age avoids the
    # cross-cohort distortion that within-cohort z-scoring introduces.
    # AGE column is passed through directly from df.

    return result


# ==============================================================================
# SECTION 4: GATEKEEPER (Stage 1) — Univariate p-tau217 Classifier
# ==============================================================================
# A single-feature logistic regression that produces P(amyloid+) from pTau217_Z.
# Patients with high confidence (p < 0.25 or p > 0.75) are resolved here.
# Patients in the gray zone (0.25 <= p <= 0.75) are routed to the Reflex.
#
# Model: LogisticRegression(penalty='l2', C=1.0)
# Input: pTau217_Z (single feature)
# Output: P(amyloid+) for each subject

def fit_gatekeeper(X_train_z, y_train):
    """
    Fit the Gatekeeper logistic regression on training data.

    Args:
        X_train_z: Array of pTau217_Z values for training subjects
        y_train: Array of amyloid status (0/1) for training subjects

    Returns:
        Fitted LogisticRegression model
    """
    gatekeeper = LogisticRegression(
        penalty='l2',       # L2 regularization (ridge)
        C=1.0,              # Regularization strength (default, mild)
        solver='lbfgs',     # Quasi-Newton optimizer
        max_iter=1000,      # Convergence iterations
        random_state=42
    )

    # Filter out any NaN values
    valid = ~np.isnan(X_train_z.flatten()) & ~np.isnan(y_train)
    gatekeeper.fit(X_train_z[valid].reshape(-1, 1), y_train[valid])

    return gatekeeper


def apply_gatekeeper(gatekeeper, X_z):
    """
    Apply Gatekeeper to produce probabilities and classifications.

    Args:
        gatekeeper: Fitted LogisticRegression model
        X_z: Array of pTau217_Z values

    Returns:
        probs: Array of P(amyloid+)
        classifications: Array of 'negative', 'positive', or 'gray_zone'
    """
    probs = gatekeeper.predict_proba(X_z.reshape(-1, 1))[:, 1]

    classifications = np.full(len(probs), 'gray_zone', dtype=object)
    classifications[probs < GK_LOW] = 'negative'     # Confident negative
    classifications[probs > GK_HIGH] = 'positive'    # Confident positive
    # Anything between GK_LOW and GK_HIGH stays 'gray_zone'

    return probs, classifications


# ==============================================================================
# SECTION 5: REFLEX (Stage 2) — Multi-marker Random Forest
# ==============================================================================
# Applied ONLY to gray zone patients. Uses 6 biologically motivated features
# to resolve diagnostic ambiguity that a single biomarker cannot handle.
#
# Model: RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced')
# The RF learns feature importance via Gini impurity — features that produce
# better splits get used more often across the 100 trees.
#
# StandardScaler is applied before the RF. This is technically redundant for a
# Random Forest (trees are scale-invariant), but is included for pipeline
# generality in case a scale-sensitive model is substituted.

def fit_reflex(X_train, y_train):
    """
    Fit the Reflex Random Forest on gray zone training data.

    Args:
        X_train: Feature matrix (N_gray_zone x 6 features)
        y_train: Amyloid status (0/1) for gray zone training subjects

    Returns:
        (fitted_rf, fitted_scaler, feature_importances)
    """
    # StandardScaler: centers each feature to mean=0, std=1
    # fit_transform learns mu/sigma from training data and applies it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Random Forest with constrained complexity
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,     # 100 trees vote on each prediction
        max_depth=RF_MAX_DEPTH,           # Max 5 sequential splits per tree
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,  # Each leaf needs >= 5 samples
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,                        # Parallel tree building
        class_weight='balanced'           # Upweight minority class
    )

    valid = ~np.isnan(y_train)
    rf.fit(X_scaled[valid], y_train[valid])

    return rf, scaler, rf.feature_importances_


def apply_reflex(rf, scaler, X_test):
    """
    Apply Reflex to predict amyloid probability for gray zone subjects.

    Uses the FROZEN scaler (trained on training gray zone) to transform
    the test subject's features, then the FROZEN RF to predict.

    Args:
        rf: Fitted RandomForestClassifier
        scaler: Fitted StandardScaler (from training data)
        X_test: Feature matrix for test subjects

    Returns:
        Array of P(amyloid+)
    """
    # transform (NOT fit_transform) — uses training mu/sigma, not test data
    X_scaled = scaler.transform(X_test)
    return rf.predict_proba(X_scaled)[:, 1]


def prepare_reflex_features(df, feature_names=REFLEX_FEATURES):
    """
    Extract and impute the 6 Reflex features from a DataFrame.

    Missing values are replaced with the column median (computed from
    the data passed in — training data during LOOCV).

    Args:
        df: DataFrame with harmonized + engineered features
        feature_names: List of feature column names

    Returns:
        Feature matrix (N x 6)
    """
    X = df[feature_names].values.copy()

    # Median imputation per feature
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, i] = np.nanmedian(col)

    return X


# ==============================================================================
# SECTION 6: LOOCV (ADNI Internal Validation)
# ==============================================================================
# Leave-One-Out Cross-Validation: for each of 320 ADNI participants, train on
# 319 and predict the 1 held-out subject.
#
# CRITICAL: Every preprocessing step is re-computed within each fold:
#   - Fresh AssayHarmonizer (new mu/sigma from 319 training subjects)
#   - Fresh feature engineering (new AGE mean/std from training fold)
#   - Fresh Gatekeeper (new logistic regression coefficients)
#   - Fresh Reflex (new RF weights, new StandardScaler)
#
# The held-out subject NEVER contaminates any training computation.

def run_adni_loocv(adni_df):
    """
    Run full LOOCV validation on ADNI.

    Args:
        adni_df: Full ADNI DataFrame (320 participants)

    Returns:
        predictions_df: Per-subject predictions with true labels
        importance_dict: Averaged feature importances across folds
    """
    print("\n" + "=" * 70)
    print("SECTION 6: LOOCV — ADNI INTERNAL VALIDATION")
    print("=" * 70)
    print(f"  N = {len(adni_df)} subjects")
    print(f"  Gatekeeper thresholds: [{GK_LOW}, {GK_HIGH}]")
    print(f"  Reflex features: {REFLEX_FEATURES}")

    n = len(adni_df)
    loo = LeaveOneOut()
    y_all = adni_df['amyloid_positive'].values

    # Storage
    all_probs = np.zeros(n)            # Final predicted probability
    all_gk_probs = np.zeros(n)         # Gatekeeper probability (for all subjects)
    all_stages = []                     # 'gatekeeper' or 'reflex'
    importance_accum = {f: [] for f in REFLEX_FEATURES}

    print(f"  Running {n} LOOCV iterations...")

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(adni_df)):
        if fold_idx % 50 == 0:
            print(f"    Fold {fold_idx + 1}/{n}")

        # ---- SPLIT ----
        train_df = adni_df.iloc[train_idx].copy()   # 319 subjects
        test_df = adni_df.iloc[test_idx].copy()      # 1 subject
        train_y = y_all[train_idx]
        test_y = y_all[test_idx]

        # ---- HARMONIZE (within-fold) ----
        # Compute mu/sigma from training CN amyloid-negative reference
        harm_params = fit_harmonizer(train_df)
        # Apply to both train and test using TRAINING params only
        train_h = apply_harmonizer(train_df, harm_params)
        test_h = apply_harmonizer(test_df, harm_params)

        # ---- GATEKEEPER ----
        gatekeeper = fit_gatekeeper(train_h['pTau217_Z'].values, train_y)
        test_gk_prob = gatekeeper.predict_proba(
            test_h[['pTau217_Z']].values
        )[:, 1][0]
        all_gk_probs[test_idx[0]] = test_gk_prob

        # ---- CLASSIFY: is this subject in the gray zone? ----
        in_gray_zone = (test_gk_prob >= GK_LOW) and (test_gk_prob <= GK_HIGH)

        if not in_gray_zone:
            # Resolved by Gatekeeper — use Gatekeeper probability directly
            all_probs[test_idx[0]] = test_gk_prob
            all_stages.append('gatekeeper')
        else:
            # Gray zone — need Reflex model

            # First: identify gray zone subjects in the TRAINING data
            # (re-classify all 319 training subjects through the Gatekeeper)
            train_gk_probs = gatekeeper.predict_proba(
                train_h[['pTau217_Z']].values
            )[:, 1]
            train_gray_mask = (train_gk_probs >= GK_LOW) & (train_gk_probs <= GK_HIGH)

            if train_gray_mask.sum() >= 10:
                # Enough gray zone training subjects to fit Reflex
                # IMPORTANT: engineer features on gray zone subset only
                # (AGE mean/std comes from gray zone training, not all 319)
                gray_train_h = train_h.iloc[np.where(train_gray_mask)[0]]
                gray_train = engineer_features(gray_train_h)
                gray_train_y = train_y[train_gray_mask]

                # Engineer features for test subject separately
                # (single-row AGE = NaN, which RF handles as missing)
                test_eng = engineer_features(test_h)

                # Prepare features
                X_gray_train = prepare_reflex_features(gray_train)
                X_test_reflex = prepare_reflex_features(test_eng)

                try:
                    # Fit Reflex on training gray zone
                    rf, scaler, importances = fit_reflex(X_gray_train, gray_train_y)

                    # Apply to test subject (using FROZEN scaler and RF)
                    reflex_prob = apply_reflex(rf, scaler, X_test_reflex)[0]
                    all_probs[test_idx[0]] = reflex_prob

                    # Accumulate feature importances
                    for i, feat in enumerate(REFLEX_FEATURES):
                        importance_accum[feat].append(importances[i])

                    all_stages.append('reflex')

                except Exception:
                    # Fallback to Gatekeeper probability if Reflex fails
                    all_probs[test_idx[0]] = test_gk_prob
                    all_stages.append('gatekeeper')
            else:
                # Not enough gray zone subjects — fallback
                all_probs[test_idx[0]] = test_gk_prob
                all_stages.append('gatekeeper')

    # ---- BUILD PREDICTIONS DATAFRAME ----
    predictions = pd.DataFrame({
        'patient_id': range(n),
        'true_amyloid': y_all,
        'predicted_prob': all_probs,
        'gatekeeper_prob': all_gk_probs,
        'stage': all_stages,
        'predicted_class': (all_probs >= 0.5).astype(int)
    })

    # ---- AVERAGE FEATURE IMPORTANCES ----
    avg_importance = {f: np.mean(v) for f, v in importance_accum.items() if v}

    print(f"    Done. {n} folds completed.\n")

    return predictions, avg_importance


# ==============================================================================
# SECTION 7: A4 EXTERNAL VALIDATION
# ==============================================================================
# Train on ALL of ADNI (not LOOCV), then apply the FROZEN pipeline to A4.
# No recomputation of any parameters on A4 data.
#
# Frozen components:
#   1. Harmonizer mu/sigma (from ADNI CN amyloid-negative)
#   2. Gatekeeper coefficients (from ADNI logistic regression)
#   3. Reflex RF weights (from ADNI gray zone)
#   4. StandardScaler mu/sigma (from ADNI gray zone)

def run_a4_validation(adni_df, a4_df):
    """
    External validation: train on ADNI, test on A4+LEARN.

    Args:
        adni_df: Full ADNI DataFrame (320 participants)
        a4_df: Full A4 DataFrame (1,644 participants with amyloid labels)

    Returns:
        a4_predictions: Per-subject A4 predictions
        summary: Dictionary of validation metrics
    """
    print("\n" + "=" * 70)
    print("SECTION 7: A4 EXTERNAL VALIDATION")
    print("=" * 70)

    adni_y = adni_df['amyloid_positive'].values.astype(int)
    a4_y = a4_df['amyloid_positive'].values.astype(int)

    # ---- STEP 1: HARMONIZE ----
    # Fit harmonizer on ADNI (learn mu/sigma from ADNI reference population)
    print("  Step 1: Harmonizing biomarkers...")
    harm_params = fit_harmonizer(adni_df)

    # Apply to ADNI using per-assay params
    adni_h = apply_harmonizer(adni_df, harm_params)

    # Apply to A4 using FROZEN ADNI params (no recomputation on A4)
    a4_h = apply_harmonizer(a4_df, harm_params)

    print(f"    ADNI pTau217_Z: mean={adni_h['pTau217_Z'].mean():.3f}, std={adni_h['pTau217_Z'].std():.3f}")
    print(f"    A4 pTau217_Z:   mean={a4_h['pTau217_Z'].mean():.3f}, std={a4_h['pTau217_Z'].std():.3f}")

    # ---- STEP 2: FIT GATEKEEPER ON ALL ADNI ----
    # No feature engineering needed here — Gatekeeper only uses pTau217_Z
    print("  Step 2: Fitting Gatekeeper on all ADNI...")
    gatekeeper = fit_gatekeeper(adni_h['pTau217_Z'].values, adni_y)
    print(f"    Intercept: {gatekeeper.intercept_[0]:.4f}")
    print(f"    Coefficient: {gatekeeper.coef_[0, 0]:.4f}")

    # ---- STEP 3: FIT REFLEX ON ADNI GRAY ZONE ----
    print("  Step 3: Fitting Reflex on ADNI gray zone...")

    # Identify ADNI gray zone
    adni_gk_probs, adni_gk_class = apply_gatekeeper(
        gatekeeper, adni_h['pTau217_Z'].values
    )
    adni_gray_mask = adni_gk_class == 'gray_zone'

    # Engineer features on gray zone subset ONLY
    # AGE uses gray zone mean/std, matching run_a4_binary_validation.py
    gray_train_h = adni_h[adni_gray_mask]
    gray_train = engineer_features(gray_train_h)
    gray_train_y = adni_y[adni_gray_mask]

    X_gray_train = prepare_reflex_features(gray_train)
    rf, scaler, importances = fit_reflex(X_gray_train, gray_train_y)

    print(f"    Trained on {adni_gray_mask.sum()} gray zone subjects")
    print(f"    Feature importance: {dict(zip(REFLEX_FEATURES, [f'{x:.3f}' for x in importances]))}")

    # ---- STEP 4: APPLY FROZEN PIPELINE TO A4 ----
    print("  Step 4: Applying frozen pipeline to A4...")

    # Gatekeeper on A4
    a4_gk_probs, a4_gk_class = apply_gatekeeper(
        gatekeeper, a4_h['pTau217_Z'].values
    )

    n_neg = (a4_gk_class == 'negative').sum()
    n_pos = (a4_gk_class == 'positive').sum()
    n_gray = (a4_gk_class == 'gray_zone').sum()
    print(f"    Gatekeeper: {n_neg} negative, {n_pos} positive, {n_gray} gray zone")

    # Final predictions: start with Gatekeeper probabilities
    a4_final_probs = a4_gk_probs.copy()
    a4_stages = a4_gk_class.copy()

    # Apply Reflex to gray zone (using FROZEN RF and FROZEN scaler)
    gray_mask = a4_gk_class == 'gray_zone'
    if gray_mask.sum() > 0:
        # Engineer features on A4 gray zone ONLY
        # AGE uses A4 gray zone mean/std (same approach as authoritative script)
        a4_gray_h = a4_h[gray_mask]
        a4_gray = engineer_features(a4_gray_h)
        X_a4_gray = prepare_reflex_features(a4_gray)
        reflex_probs = apply_reflex(rf, scaler, X_a4_gray)
        a4_final_probs[gray_mask] = reflex_probs
        a4_stages[gray_mask] = 'reflex'
        print(f"    Reflex applied to {gray_mask.sum()} gray zone subjects")

    # ---- STEP 5: COMPUTE METRICS ----
    print("\n  Computing A4 metrics...")

    valid = ~np.isnan(a4_final_probs)
    y_true = a4_y[valid]
    y_prob = a4_final_probs[valid]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    brier = brier_score_loss(y_true, y_prob)

    # Bootstrap CI for AUC
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

    # Centiloid correlation
    if 'AMYLCENT' in a4_df.columns:
        cl_valid = a4_df['AMYLCENT'].notna().values & valid
        r_spearman, p_spearman = stats.spearmanr(
            a4_df.loc[a4_df.index[cl_valid], 'AMYLCENT'].values,
            a4_final_probs[cl_valid]
        )
    else:
        r_spearman, p_spearman = np.nan, np.nan

    # Build predictions DataFrame
    a4_predictions = pd.DataFrame({
        'true_amyloid': a4_y,
        'predicted_prob': a4_final_probs,
        'predicted_class': (a4_final_probs >= 0.5).astype(int),
        'stage': a4_stages,
    })

    summary = {
        'n': int(valid.sum()),
        'n_positive': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
        'auc': float(auc),
        'auc_ci': [float(ci_low), float(ci_high)],
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'ppv': float(ppv),
        'npv': float(npv),
        'brier': float(brier),
        'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
        'gatekeeper_neg': int(n_neg),
        'gatekeeper_pos': int(n_pos),
        'gatekeeper_gray': int(n_gray),
        'gatekeeper_resolution_rate': float((n_neg + n_pos) / len(a4_df)),
        'centiloid_spearman_r': float(r_spearman),
        'centiloid_spearman_p': float(p_spearman),
        'reflex_importance': dict(zip(REFLEX_FEATURES, [float(x) for x in importances])),
    }

    return a4_predictions, summary


# ==============================================================================
# SECTION 8: METRICS COMPUTATION
# ==============================================================================
# Computes all metrics reported in the manuscript from LOOCV predictions.

def compute_all_metrics(preds_df):
    """
    Compute comprehensive metrics from LOOCV predictions.

    Args:
        preds_df: DataFrame with true_amyloid, predicted_prob, stage columns

    Returns:
        Dictionary with all metrics
    """
    y = preds_df['true_amyloid'].values
    p = preds_df['predicted_prob'].values
    valid = ~np.isnan(y) & ~np.isnan(p)
    y_v, p_v = y[valid], p[valid]
    pred_v = (p_v >= 0.5).astype(int)

    # Confusion matrix
    tp = ((pred_v == 1) & (y_v == 1)).sum()
    fn = ((pred_v == 0) & (y_v == 1)).sum()
    tn = ((pred_v == 0) & (y_v == 0)).sum()
    fp = ((pred_v == 1) & (y_v == 0)).sum()

    # Core metrics
    auc = roc_auc_score(y_v, p_v)
    acc = (tp + tn) / len(y_v)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    lr_pos = sens / (1 - spec) if spec < 1 else np.inf
    lr_neg = (1 - sens) / spec if spec > 0 else np.nan
    brier = brier_score_loss(y_v, p_v)

    # Gray zone metrics
    gz_mask = preds_df['stage'].values == 'reflex'
    if gz_mask.sum() > 5 and len(np.unique(y[gz_mask])) > 1:
        gz_auc = roc_auc_score(y[gz_mask], p[gz_mask])
    else:
        gz_auc = np.nan

    # Gatekeeper metrics
    gk_mask = preds_df['stage'].values == 'gatekeeper'
    gk_acc = np.nan
    if gk_mask.sum() > 0:
        gk_pred = (preds_df.loc[gk_mask, 'gatekeeper_prob'].values >= 0.5).astype(int)
        gk_true = y[gk_mask]
        gk_acc = (gk_pred == gk_true).mean()

    # Bootstrap 95% CIs
    rng = np.random.RandomState(42)
    boot_aucs, boot_accs, boot_sens, boot_specs = [], [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_v), size=len(y_v), replace=True)
        yb, pb = y_v[idx], p_v[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))
        predb = (pb >= 0.5).astype(int)
        boot_accs.append((predb == yb).mean())
        tp_b = ((predb == 1) & (yb == 1)).sum()
        fn_b = ((predb == 0) & (yb == 1)).sum()
        tn_b = ((predb == 0) & (yb == 0)).sum()
        fp_b = ((predb == 1) & (yb == 0)).sum()
        boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else np.nan)
        boot_specs.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else np.nan)

    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    # Operating points
    fpr, tpr, thresholds = roc_curve(y_v, p_v)
    spec_at = 1 - fpr

    # Rule-out: >= 90% sensitivity
    sens_90_idx = np.where(tpr >= 0.90)[0]
    rule_out = {}
    if len(sens_90_idx) > 0:
        best = sens_90_idx[np.argmin(fpr[sens_90_idx])]
        if best < len(thresholds):
            rule_out = {
                'threshold': float(thresholds[best]),
                'sensitivity': float(tpr[best]),
                'specificity': float(1 - fpr[best]),
                'LR-': float((1 - tpr[best]) / (1 - fpr[best])) if fpr[best] < 1 else np.nan
            }

    # Rule-in: >= 90% specificity
    spec_90_idx = np.where(spec_at >= 0.90)[0]
    rule_in = {}
    if len(spec_90_idx) > 0:
        best = spec_90_idx[np.argmax(tpr[spec_90_idx])]
        if best < len(thresholds):
            rule_in = {
                'threshold': float(thresholds[best]),
                'sensitivity': float(tpr[best]),
                'specificity': float(1 - fpr[best]),
                'LR+': float(tpr[best] / fpr[best]) if fpr[best] > 0 else np.inf
            }

    return {
        'n': int(len(y_v)),
        'tp': int(tp), 'fn': int(fn), 'tn': int(tn), 'fp': int(fp),
        'auc': float(auc),
        'auc_ci': ci(boot_aucs),
        'accuracy': float(acc),
        'accuracy_ci': ci(boot_accs),
        'sensitivity': float(sens),
        'sensitivity_ci': ci(boot_sens),
        'specificity': float(spec),
        'specificity_ci': ci(boot_specs),
        'ppv': float(ppv),
        'npv': float(npv),
        'lr_pos': float(lr_pos),
        'lr_neg': float(lr_neg),
        'brier': float(brier),
        'gray_zone_n': int(gz_mask.sum()),
        'gray_zone_auc': float(gz_auc) if not np.isnan(gz_auc) else None,
        'gatekeeper_resolved': int(gk_mask.sum()),
        'gatekeeper_accuracy': float(gk_acc) if not np.isnan(gk_acc) else None,
        'rule_out_90': rule_out,
        'rule_in_90': rule_in,
    }


# ==============================================================================
# SECTION 9: MAIN — RUN EVERYTHING
# ==============================================================================

def main():
    print("\n" + "=" * 70)
    print("GRAD: Gatekeeper-Reflex Algorithm for Amyloid Determination")
    print("Complete End-to-End Pipeline")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- LOAD DATA ----
    adni_df, a4_df = load_data()

    # ---- ADNI LOOCV ----
    adni_preds, importance = run_adni_loocv(adni_df)

    # Save ADNI predictions
    adni_preds.to_csv(RESULTS_DIR / 'adni_loocv_predictions_master.csv', index=False)

    # ---- COMPUTE ADNI METRICS ----
    print("=" * 70)
    print("ADNI LOOCV RESULTS")
    print("=" * 70)
    metrics = compute_all_metrics(adni_preds)

    print(f"\n  Overall AUC:      {metrics['auc']:.4f}  (95% CI: {metrics['auc_ci'][0]:.3f}-{metrics['auc_ci'][1]:.3f})")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}  (95% CI: {metrics['accuracy_ci'][0]:.3f}-{metrics['accuracy_ci'][1]:.3f})")
    print(f"  Sensitivity:      {metrics['sensitivity']:.4f}  (95% CI: {metrics['sensitivity_ci'][0]:.3f}-{metrics['sensitivity_ci'][1]:.3f})")
    print(f"  Specificity:      {metrics['specificity']:.4f}  (95% CI: {metrics['specificity_ci'][0]:.3f}-{metrics['specificity_ci'][1]:.3f})")
    print(f"  PPV:              {metrics['ppv']:.4f}")
    print(f"  NPV:              {metrics['npv']:.4f}")
    print(f"  LR+:              {metrics['lr_pos']:.2f}")
    print(f"  LR-:              {metrics['lr_neg']:.2f}")
    print(f"  Brier Score:      {metrics['brier']:.4f}")
    print(f"\n  Confusion Matrix: TP={metrics['tp']}, FN={metrics['fn']}, TN={metrics['tn']}, FP={metrics['fp']}")
    print(f"  Gatekeeper resolved: {metrics['gatekeeper_resolved']}/320 ({metrics['gatekeeper_resolved']/320*100:.1f}%)")
    print(f"  Gray zone (Reflex):  {metrics['gray_zone_n']}/320 ({metrics['gray_zone_n']/320*100:.1f}%)")
    print(f"  Gatekeeper accuracy: {metrics['gatekeeper_accuracy']:.3f}")
    print(f"  Gray zone AUC:       {metrics['gray_zone_auc']:.3f}")

    if metrics['rule_out_90']:
        ro = metrics['rule_out_90']
        print(f"\n  Rule-Out (>=90% sens): Sens={ro['sensitivity']:.3f}, Spec={ro['specificity']:.3f}, LR-={ro['LR-']:.2f}")
    if metrics['rule_in_90']:
        ri = metrics['rule_in_90']
        print(f"  Rule-In  (>=90% spec): Sens={ri['sensitivity']:.3f}, Spec={ri['specificity']:.3f}, LR+={ri['LR+']:.2f}")

    # Feature importance
    print(f"\n  Feature Importance (Reflex RF, averaged across LOOCV folds):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.3f} ({imp*100:.1f}%)")

    # ---- A4 EXTERNAL VALIDATION ----
    a4_preds, a4_summary = run_a4_validation(adni_df, a4_df)

    # Save A4 predictions
    a4_preds.to_csv(RESULTS_DIR / 'a4_validation_predictions_master.csv', index=False)

    # Print A4 results
    print("\n" + "=" * 70)
    print("A4 EXTERNAL VALIDATION RESULTS")
    print("=" * 70)
    s = a4_summary
    print(f"\n  N:                {s['n']}")
    print(f"  Prevalence:       {s['prevalence']:.1%}")
    print(f"  AUC:              {s['auc']:.4f}  (95% CI: {s['auc_ci'][0]:.3f}-{s['auc_ci'][1]:.3f})")
    print(f"  Accuracy:         {s['accuracy']:.4f}")
    print(f"  Sensitivity:      {s['sensitivity']:.4f}")
    print(f"  Specificity:      {s['specificity']:.4f}")
    print(f"  PPV:              {s['ppv']:.4f}")
    print(f"  NPV:              {s['npv']:.4f}")
    print(f"  Brier Score:      {s['brier']:.4f}")
    cm = s['confusion_matrix']
    print(f"  Confusion Matrix: TP={cm['TP']}, FN={cm['FN']}, TN={cm['TN']}, FP={cm['FP']}")
    print(f"\n  Gatekeeper: {s['gatekeeper_neg']} neg, {s['gatekeeper_pos']} pos, {s['gatekeeper_gray']} gray zone")
    print(f"  Resolution rate:  {s['gatekeeper_resolution_rate']:.1%}")
    print(f"  Centiloid r:      {s['centiloid_spearman_r']:.3f} (p={s['centiloid_spearman_p']:.2e})")

    # ---- SAVE COMBINED SUMMARY ----
    summary = {
        'timestamp': datetime.now().isoformat(),
        'adni_loocv': metrics,
        'a4_external': a4_summary,
        'feature_importance': importance,
    }

    summary_path = RESULTS_DIR / 'grad_master_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nResults saved to:")
    print(f"  {RESULTS_DIR / 'adni_loocv_predictions_master.csv'}")
    print(f"  {RESULTS_DIR / 'a4_validation_predictions_master.csv'}")
    print(f"  {summary_path}")

    print("\n" + "=" * 70)
    print("GRAD MASTER PIPELINE COMPLETE")
    print("=" * 70)

    return metrics, a4_summary


if __name__ == '__main__':
    main()
