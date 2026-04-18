"""
Validation Module for Gray Zone Classifier
==========================================

Implements Leave-One-Out Cross-Validation (LOOCV) for the two-stage
Gatekeeper + Reflex diagnostic framework.

All preprocessing steps are performed within each fold to prevent
data leakage:
    - Log-transformation
    - Assay harmonization
    - Feature engineering
    - Model fitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

try:
    from .harmonizer import AssayHarmonizer
    from .gatekeeper import GatekeeperModel
    from .reflex import ReflexModel
except ImportError:
    from harmonizer import AssayHarmonizer
    from gatekeeper import GatekeeperModel
    from reflex import ReflexModel


@dataclass
class ValidationResults:
    """Complete results from cross-validation."""
    # Overall metrics
    overall_auc: float
    overall_accuracy: float
    overall_brier: float

    # Stage 1 metrics
    gatekeeper_resolution_rate: float
    gatekeeper_accuracy: float

    # Stage 2 metrics
    gray_zone_size: int
    gray_zone_auc: float
    gray_zone_accuracy: float

    # Per-sample predictions
    predictions: pd.DataFrame

    # ROC curve data
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray

    # Feature importance (from Reflex)
    feature_importance: pd.DataFrame = None


class LOOCVValidator:
    """
    Leave-One-Out Cross-Validation for the two-stage system.

    Ensures complete prevention of data leakage by re-computing
    all preprocessing within each fold.
    """

    # Locked 6-feature set for the Reflex model.
    # This ensures reproducibility and manuscript-code consistency.
    REFLEX_FEATURES = [
        'pTau217_Z', 'tau_ab42_diff', 'GFAP_Z', 'AGE',
        'APOE4_carrier', 'gfap_tau_interaction'
    ]

    def __init__(
        self,
        gatekeeper_low: float = 0.25,
        gatekeeper_high: float = 0.75,
        reflex_n_estimators: int = 100,
        reflex_max_depth: int = 5,
        reflex_feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize LOOCV validator.

        Args:
            gatekeeper_low: Low probability threshold for Gatekeeper
            gatekeeper_high: High probability threshold for Gatekeeper
            reflex_n_estimators: Number of trees for Reflex RF
            reflex_max_depth: Max depth for Reflex RF
            reflex_feature_cols: Explicit feature list for Reflex.
                Defaults to REFLEX_FEATURES (6 features).
        """
        self.gatekeeper_low = gatekeeper_low
        self.gatekeeper_high = gatekeeper_high
        self.reflex_n_estimators = reflex_n_estimators
        self.reflex_max_depth = reflex_max_depth
        self.reflex_feature_cols = reflex_feature_cols or self.REFLEX_FEATURES

    def validate_loocv(
        self,
        df: pd.DataFrame,
        target_col: str = 'amyloid_positive',
        verbose: bool = True
    ) -> ValidationResults:
        """
        Run full LOOCV validation.

        Args:
            df: DataFrame with features and target
            target_col: Column name for amyloid status
            verbose: Print progress updates

        Returns:
            ValidationResults with all metrics and predictions
        """
        n_samples = len(df)
        loo = LeaveOneOut()

        # Storage for predictions
        all_probs = np.zeros(n_samples)
        all_stages = []  # Track which stage resolved each patient
        all_gatekeeper_probs = np.zeros(n_samples)

        # Feature importance accumulator
        importance_accum = {}

        if verbose:
            print(f"Running LOOCV on {n_samples} samples...")
            print(f"  Gatekeeper thresholds: [{self.gatekeeper_low}, {self.gatekeeper_high}]")

        for fold_idx, (train_idx, test_idx) in enumerate(loo.split(df)):
            if verbose and fold_idx % 50 == 0:
                print(f"  Fold {fold_idx + 1}/{n_samples}")

            # Split data
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            train_y = train_df[target_col]
            test_y = test_df[target_col]

            # Step 1: Harmonize within fold
            harmonizer = AssayHarmonizer()
            train_harmonized = harmonizer.fit_transform(train_df)
            test_harmonized = harmonizer.transform(test_df)

            # Step 2: Fit Gatekeeper on training data
            gatekeeper = GatekeeperModel(
                low_threshold=self.gatekeeper_low,
                high_threshold=self.gatekeeper_high
            )
            gatekeeper.fit(train_harmonized, train_y)

            # Step 3: Apply Gatekeeper to test sample
            gk_result = gatekeeper.classify(test_harmonized)
            gk_prob = gk_result['probability'].values[0]
            all_gatekeeper_probs[test_idx[0]] = gk_prob

            # Step 4: Check if in gray zone
            in_gray_zone = gk_result['in_gray_zone'].values[0]

            if not in_gray_zone:
                # Resolved by Gatekeeper
                all_probs[test_idx[0]] = gk_prob
                all_stages.append('gatekeeper')
            else:
                # Need Reflex model
                # Find gray zone in training data
                train_gk_result = gatekeeper.classify(train_harmonized)
                train_gray_idx = train_gk_result[train_gk_result['in_gray_zone']].index

                if len(train_gray_idx) >= 10:
                    # Enough gray zone samples to train Reflex
                    reflex = ReflexModel(
                        n_estimators=self.reflex_n_estimators,
                        max_depth=self.reflex_max_depth
                    )

                    train_gray_df = train_harmonized.loc[train_gray_idx]
                    train_gray_y = train_y.loc[train_gray_idx]

                    try:
                        reflex.fit(train_gray_df, train_gray_y,
                                   feature_cols=self.reflex_feature_cols)
                        reflex_prob = reflex.predict_proba(test_harmonized)[0]
                        all_probs[test_idx[0]] = reflex_prob

                        # Accumulate feature importance
                        importance = reflex.get_feature_importance()
                        for _, row in importance.iterrows():
                            feat = row['feature']
                            if feat not in importance_accum:
                                importance_accum[feat] = []
                            importance_accum[feat].append(row['importance'])

                    except Exception as e:
                        # Fall back to Gatekeeper probability
                        all_probs[test_idx[0]] = gk_prob
                        all_stages.append('gatekeeper_fallback')
                        continue

                    all_stages.append('reflex')
                else:
                    # Not enough gray zone samples, use Gatekeeper prob
                    all_probs[test_idx[0]] = gk_prob
                    all_stages.append('gatekeeper_fallback')

        # Compute metrics
        y_true = df[target_col].values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(all_probs)

        # Overall AUC
        overall_auc = roc_auc_score(y_true[valid_mask], all_probs[valid_mask])

        # Overall accuracy (using 0.5 threshold)
        overall_acc = ((all_probs[valid_mask] >= 0.5) == y_true[valid_mask]).mean()

        # Brier score
        overall_brier = brier_score_loss(y_true[valid_mask], all_probs[valid_mask])

        # Gatekeeper metrics
        gk_resolved = np.array(all_stages) == 'gatekeeper'
        gk_resolution_rate = gk_resolved.sum() / len(all_stages)

        if gk_resolved.sum() > 0:
            gk_preds = all_gatekeeper_probs[gk_resolved] >= 0.5
            gk_true = y_true[gk_resolved]
            gk_acc = (gk_preds == gk_true).mean()
        else:
            gk_acc = np.nan

        # Gray zone metrics
        gray_zone_mask = np.array(all_stages) == 'reflex'
        gray_zone_size = gray_zone_mask.sum()

        if gray_zone_size > 5 and len(np.unique(y_true[gray_zone_mask])) > 1:
            gray_auc = roc_auc_score(y_true[gray_zone_mask], all_probs[gray_zone_mask])
            gray_acc = ((all_probs[gray_zone_mask] >= 0.5) == y_true[gray_zone_mask]).mean()
        else:
            gray_auc = np.nan
            gray_acc = np.nan

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true[valid_mask], all_probs[valid_mask])

        # Feature importance
        if importance_accum:
            feat_imp = pd.DataFrame({
                'feature': list(importance_accum.keys()),
                'importance': [np.mean(v) for v in importance_accum.values()],
                'std': [np.std(v) for v in importance_accum.values()]
            }).sort_values('importance', ascending=False)
        else:
            feat_imp = pd.DataFrame()

        # Predictions DataFrame
        predictions = pd.DataFrame({
            'patient_id': df.index if df.index.name else range(len(df)),
            'true_amyloid': y_true,
            'predicted_prob': all_probs,
            'gatekeeper_prob': all_gatekeeper_probs,
            'stage': all_stages,
            'predicted_class': (all_probs >= 0.5).astype(int)
        })

        if verbose:
            print(f"\nResults:")
            print(f"  Overall AUC: {overall_auc:.3f}")
            print(f"  Overall Accuracy: {overall_acc:.3f}")
            print(f"  Gatekeeper Resolution: {gk_resolution_rate*100:.1f}%")
            print(f"  Gatekeeper Accuracy: {gk_acc:.3f}" if not np.isnan(gk_acc) else "  Gatekeeper Accuracy: N/A")
            print(f"  Gray Zone Size: {gray_zone_size} ({gray_zone_size/len(df)*100:.1f}%)")
            print(f"  Gray Zone AUC: {gray_auc:.3f}" if not np.isnan(gray_auc) else "  Gray Zone AUC: N/A")

        return ValidationResults(
            overall_auc=overall_auc,
            overall_accuracy=overall_acc,
            overall_brier=overall_brier,
            gatekeeper_resolution_rate=gk_resolution_rate,
            gatekeeper_accuracy=gk_acc,
            gray_zone_size=gray_zone_size,
            gray_zone_auc=gray_auc,
            gray_zone_accuracy=gray_acc,
            predictions=predictions,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            feature_importance=feat_imp
        )


class ExternalValidator:
    """
    External validation of trained models on independent cohort.

    Used for validating ADNI-trained models on A4 data.
    """

    def __init__(self):
        """Initialize external validator."""
        self.gatekeeper = None
        self.reflex = None
        self.harmonizer = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = 'amyloid_positive',
        gatekeeper_low: float = 0.25,
        gatekeeper_high: float = 0.75,
        reflex_feature_cols: Optional[List[str]] = None
    ) -> 'ExternalValidator':
        """
        Fit models on training data.

        Args:
            train_df: Training DataFrame (e.g., ADNI)
            target_col: Column name for amyloid status
            gatekeeper_low: Low probability threshold
            gatekeeper_high: High probability threshold
            reflex_feature_cols: Explicit feature list for Reflex.
                Defaults to LOOCVValidator.REFLEX_FEATURES (6 features).

        Returns:
            self (fitted validator)
        """
        if reflex_feature_cols is None:
            reflex_feature_cols = LOOCVValidator.REFLEX_FEATURES

        train_y = train_df[target_col]

        # Harmonize training data
        self.harmonizer = AssayHarmonizer()
        train_harmonized = self.harmonizer.fit_transform(train_df)

        # Fit Gatekeeper
        self.gatekeeper = GatekeeperModel(
            low_threshold=gatekeeper_low,
            high_threshold=gatekeeper_high
        )
        self.gatekeeper.fit(train_harmonized, train_y)

        # Identify gray zone in training
        train_gk_result = self.gatekeeper.classify(train_harmonized)
        train_gray_idx = train_gk_result[train_gk_result['in_gray_zone']].index

        # Fit Reflex on gray zone
        if len(train_gray_idx) >= 10:
            self.reflex = ReflexModel()
            train_gray_df = train_harmonized.loc[train_gray_idx]
            train_gray_y = train_y.loc[train_gray_idx]
            self.reflex.fit(train_gray_df, train_gray_y,
                            feature_cols=reflex_feature_cols)

        return self

    def validate(
        self,
        test_df: pd.DataFrame,
        target_col: str = 'amyloid_positive'
    ) -> Dict[str, Any]:
        """
        Validate on external test data.

        Args:
            test_df: Test DataFrame (e.g., A4)
            target_col: Column name for amyloid status

        Returns:
            Dictionary with validation metrics
        """
        if self.gatekeeper is None:
            raise ValueError("Must fit before validation")

        # Harmonize test data using training parameters
        test_harmonized = self.harmonizer.transform(test_df)

        # Apply Gatekeeper
        gk_result = self.gatekeeper.classify(test_harmonized)

        # Initialize predictions
        all_probs = gk_result['probability'].values.copy()
        all_stages = gk_result['classification'].values.copy()

        # Apply Reflex to gray zone
        if self.reflex is not None:
            gray_mask = gk_result['in_gray_zone'].values
            if gray_mask.sum() > 0:
                gray_df = test_harmonized[gray_mask]
                reflex_probs = self.reflex.predict_proba(gray_df)
                all_probs[gray_mask] = reflex_probs
                all_stages[gray_mask] = 'reflex'

        # Compute metrics
        y_true = test_df[target_col].values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(all_probs)

        results = {
            'n_test': len(test_df),
            'n_gray_zone': (all_stages == 'reflex').sum() if isinstance(all_stages[0], str) else gk_result['in_gray_zone'].sum(),
            'predictions': pd.DataFrame({
                'true_amyloid': y_true,
                'predicted_prob': all_probs,
                'stage': all_stages
            })
        }

        if valid_mask.sum() > 0 and len(np.unique(y_true[valid_mask])) > 1:
            results['auc'] = roc_auc_score(y_true[valid_mask], all_probs[valid_mask])
            results['accuracy'] = ((all_probs[valid_mask] >= 0.5) == y_true[valid_mask]).mean()
            results['brier'] = brier_score_loss(y_true[valid_mask], all_probs[valid_mask])
        else:
            results['auc'] = np.nan
            results['accuracy'] = ((all_probs[valid_mask] >= 0.5) == y_true[valid_mask]).mean() if valid_mask.sum() > 0 else np.nan
            results['brier'] = np.nan

        return results


def run_loocv(
    df: pd.DataFrame,
    target_col: str = 'amyloid_positive',
    gatekeeper_thresholds: Tuple[float, float] = (0.25, 0.75),
    verbose: bool = True
) -> ValidationResults:
    """
    Convenience function to run LOOCV validation.

    Args:
        df: DataFrame with features and target
        target_col: Column name for amyloid status
        gatekeeper_thresholds: (low, high) probability thresholds
        verbose: Print progress

    Returns:
        ValidationResults
    """
    validator = LOOCVValidator(
        gatekeeper_low=gatekeeper_thresholds[0],
        gatekeeper_high=gatekeeper_thresholds[1]
    )
    return validator.validate_loocv(df, target_col, verbose)
