"""
GRAD: The Two-Stage Clinical Pipeline for Gray Zone Classification
================================================

Integrates the Gatekeeper and Reflex models into a unified
diagnostic ML pipeline with proper data flow and result aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from .harmonizer import AssayHarmonizer
    from .gatekeeper import GatekeeperModel, GatekeeperMetrics
    from .reflex import ReflexModel, ReflexMetrics
except ImportError:
    from harmonizer import AssayHarmonizer
    from gatekeeper import GatekeeperModel, GatekeeperMetrics
    from reflex import ReflexModel, ReflexMetrics


@dataclass
class PipelineResults:
    """Results from the complete two-stage pipeline."""
    n_total: int
    n_resolved_stage1: int
    n_gray_zone: int

    # Predictions
    predictions: pd.DataFrame

    # Metrics
    gatekeeper_metrics: GatekeeperMetrics
    reflex_metrics: Optional[ReflexMetrics]
    overall_auc: float
    overall_accuracy: float


class GrayZonePipeline:
    """
    Complete two-stage Gatekeeper + Reflex diagnostic pipeline.

    Stage 1: Univariate p-tau217 Gatekeeper
        - Classifies high-confidence cases
        - Routes ambiguous cases to Stage 2

    Stage 2: Multi-marker Reflex
        - Random Forest with biological features
        - Applied only to gray zone cases
    """

    def __init__(
        self,
        gatekeeper_low: float = 0.25,
        gatekeeper_high: float = 0.75,
        reflex_n_estimators: int = 100,
        reflex_max_depth: int = 5
    ):
        """
        Initialize pipeline.

        Args:
            gatekeeper_low: Low probability threshold for Gatekeeper
            gatekeeper_high: High probability threshold for Gatekeeper
            reflex_n_estimators: Number of trees for Reflex RF
            reflex_max_depth: Max depth for Reflex RF
        """
        self.gatekeeper_low = gatekeeper_low
        self.gatekeeper_high = gatekeeper_high
        self.reflex_n_estimators = reflex_n_estimators
        self.reflex_max_depth = reflex_max_depth

        self.harmonizer = AssayHarmonizer()
        self.gatekeeper = GatekeeperModel(
            low_threshold=gatekeeper_low,
            high_threshold=gatekeeper_high
        )
        self.reflex = ReflexModel(
            n_estimators=reflex_n_estimators,
            max_depth=reflex_max_depth
        )

        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'amyloid_positive'
    ) -> 'GrayZonePipeline':
        """
        Fit the complete pipeline.

        Args:
            df: Training DataFrame
            target_col: Column name for amyloid status

        Returns:
            self (fitted pipeline)
        """
        y = df[target_col]

        # Step 1: Harmonize
        print("Step 1: Harmonizing biomarkers...")
        df_harmonized = self.harmonizer.fit_transform(df)

        # Step 2: Fit Gatekeeper
        print("Step 2: Fitting Gatekeeper model...")
        self.gatekeeper.fit(df_harmonized, y)

        # Step 3: Identify gray zone
        gk_result = self.gatekeeper.classify(df_harmonized)
        gray_zone_idx = gk_result[gk_result['in_gray_zone']].index

        print(f"  Gray zone identified: {len(gray_zone_idx)} samples " +
              f"({len(gray_zone_idx)/len(df)*100:.1f}%)")

        # Step 4: Fit Reflex on gray zone
        if len(gray_zone_idx) >= 10:
            print("Step 3: Fitting Reflex model on gray zone...")
            gray_df = df_harmonized.loc[gray_zone_idx]
            gray_y = y.loc[gray_zone_idx]
            self.reflex.fit(gray_df, gray_y)
            print(f"  Features used: {self.reflex.feature_names}")
        else:
            print("  Warning: Insufficient gray zone samples for Reflex model")

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for new data.

        Args:
            df: DataFrame with required features

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")

        # Harmonize
        df_harmonized = self.harmonizer.transform(df)

        # Stage 1: Gatekeeper
        gk_result = self.gatekeeper.classify(df_harmonized)

        # Initialize results
        results = pd.DataFrame({
            'gatekeeper_prob': gk_result['probability'],
            'gatekeeper_class': gk_result['classification'],
            'in_gray_zone': gk_result['in_gray_zone'],
            'final_prob': gk_result['probability'].copy(),
            'final_class': gk_result['classification'].copy(),
            'resolved_by': 'gatekeeper'
        }, index=df.index)

        # Stage 2: Reflex for gray zone
        gray_mask = results['in_gray_zone']
        if gray_mask.sum() > 0 and self.reflex.is_fitted:
            gray_df = df_harmonized[gray_mask]
            reflex_result = self.reflex.classify(gray_df)

            results.loc[gray_mask, 'final_prob'] = reflex_result['probability'].values
            results.loc[gray_mask, 'final_class'] = reflex_result['classification'].values
            results.loc[gray_mask, 'resolved_by'] = 'reflex'

        # Add predicted binary class
        results['predicted_positive'] = (results['final_prob'] >= 0.5).astype(int)

        return results

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = 'amyloid_positive'
    ) -> PipelineResults:
        """
        Evaluate pipeline on labeled data.

        Args:
            df: DataFrame with features and target
            target_col: Column name for amyloid status

        Returns:
            PipelineResults with all metrics
        """
        from sklearn.metrics import roc_auc_score, accuracy_score

        y_true = df[target_col]

        # Get predictions
        predictions = self.predict(df)
        predictions['true_amyloid'] = y_true.values

        # Overall metrics
        valid_mask = ~predictions['final_prob'].isna() & ~y_true.isna()
        y_valid = y_true[valid_mask].values
        probs_valid = predictions.loc[valid_mask, 'final_prob'].values

        if len(np.unique(y_valid)) > 1:
            overall_auc = roc_auc_score(y_valid, probs_valid)
        else:
            overall_auc = np.nan

        overall_acc = accuracy_score(y_valid, (probs_valid >= 0.5).astype(int))

        # Gatekeeper metrics
        df_harmonized = self.harmonizer.transform(df)
        gk_metrics = self.gatekeeper.evaluate(df_harmonized, y_true)

        # Reflex metrics (if applicable)
        gray_mask = predictions['in_gray_zone'] & valid_mask
        if gray_mask.sum() >= 5 and self.reflex.is_fitted:
            gray_df = df_harmonized[gray_mask]
            gray_y = y_true[gray_mask]
            reflex_metrics = self.reflex.evaluate(gray_df, gray_y)
        else:
            reflex_metrics = None

        return PipelineResults(
            n_total=len(df),
            n_resolved_stage1=gk_metrics.n_resolved,
            n_gray_zone=gk_metrics.n_gray_zone,
            predictions=predictions,
            gatekeeper_metrics=gk_metrics,
            reflex_metrics=reflex_metrics,
            overall_auc=overall_auc,
            overall_accuracy=overall_acc
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Reflex model."""
        if self.reflex.is_fitted:
            return self.reflex.get_feature_importance()
        return pd.DataFrame()

    def get_decision_boundaries(self) -> Dict[str, float]:
        """
        Get decision boundaries in p-tau217_Z space.

        Returns:
            Dictionary with Z-score values at key probability thresholds
        """
        if not self.gatekeeper.is_fitted:
            return {}

        return {
            'z_at_25pct': self.gatekeeper.get_decision_boundary_z(0.25),
            'z_at_50pct': self.gatekeeper.get_decision_boundary_z(0.50),
            'z_at_75pct': self.gatekeeper.get_decision_boundary_z(0.75),
        }


def run_pipeline(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = 'amyloid_positive',
    gatekeeper_thresholds: Tuple[float, float] = (0.25, 0.75)
) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline.

    Args:
        train_df: Training DataFrame (e.g., ADNI)
        test_df: Optional test DataFrame (e.g., A4)
        target_col: Column name for amyloid status
        gatekeeper_thresholds: (low, high) probability thresholds

    Returns:
        Dictionary with training and optional test results
    """
    # Initialize and fit pipeline
    pipeline = GrayZonePipeline(
        gatekeeper_low=gatekeeper_thresholds[0],
        gatekeeper_high=gatekeeper_thresholds[1]
    )

    pipeline.fit(train_df, target_col)

    # Evaluate on training data
    train_results = pipeline.evaluate(train_df, target_col)

    results = {
        'pipeline': pipeline,
        'train_results': train_results,
        'feature_importance': pipeline.get_feature_importance(),
        'decision_boundaries': pipeline.get_decision_boundaries()
    }

    # Evaluate on test data if provided
    if test_df is not None:
        test_results = pipeline.evaluate(test_df, target_col)
        results['test_results'] = test_results

    return results
