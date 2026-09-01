"""
Stage 1: Gatekeeper Model
=========================

Univariate logistic regression using harmonized plasma p-tau217_Z
as the sole predictor of amyloid positivity.

Probability Thresholds:
    - <25%: Classified as amyloid-negative (high confidence)
    - >75%: Classified as amyloid-positive (high confidence)
    - 25-75%: Gray Zone -> routed to Stage 2 Reflex model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class GatekeeperResult:
    """Result of Gatekeeper classification for a single patient."""
    probability: float
    classification: str  # 'negative', 'positive', or 'gray_zone'
    confidence: str  # 'high' or 'low'


@dataclass
class GatekeeperMetrics:
    """Aggregate metrics for Gatekeeper performance."""
    n_total: int
    n_resolved: int
    n_gray_zone: int
    resolution_rate: float
    accuracy_resolved: float
    sensitivity_positive: float
    specificity_negative: float


class GatekeeperModel:
    """
    Stage 1 Gatekeeper: Univariate p-tau217 classifier.

    Uses pre-specified probability thresholds to classify high-confidence
    cases and route ambiguous cases to the Reflex model.
    """

    def __init__(
        self,
        low_threshold: float = 0.25,
        high_threshold: float = 0.75,
        feature_col: str = 'pTau217_Z'
    ):
        """
        Initialize Gatekeeper model.

        Args:
            low_threshold: Probability below which -> amyloid-negative
            high_threshold: Probability above which -> amyloid-positive
            feature_col: Column name for harmonized p-tau217
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.feature_col = feature_col
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> 'GatekeeperModel':
        """
        Fit the Gatekeeper model.

        Args:
            X: Feature DataFrame (must contain feature_col)
            y: Binary amyloid status (0=negative, 1=positive)

        Returns:
            self (fitted model)
        """
        # Extract feature
        if self.feature_col not in X.columns:
            raise ValueError(f"Feature column '{self.feature_col}' not found in X")

        features = X[[self.feature_col]].values

        # Handle missing values
        valid_mask = ~np.isnan(features.flatten()) & ~np.isnan(y.values)
        features_valid = features[valid_mask]
        y_valid = y.values[valid_mask]

        if len(y_valid) < 10:
            raise ValueError(f"Insufficient valid samples for fitting: {len(y_valid)}")

        # Fit logistic regression
        self.model.fit(features_valid, y_valid)
        self.is_fitted = True

        # Store training statistics
        self.train_n = len(y_valid)
        self.train_prevalence = y_valid.mean()

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of amyloid positivity.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features = X[[self.feature_col]].values

        # Handle missing values
        probs = np.full(len(X), np.nan)
        valid_mask = ~np.isnan(features.flatten())
        probs[valid_mask] = self.model.predict_proba(features[valid_mask])[:, 1]

        return probs

    def classify(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Classify patients into negative, positive, or gray_zone.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with columns: probability, classification, confidence
        """
        probs = self.predict_proba(X)

        results = pd.DataFrame({
            'probability': probs,
            'classification': np.nan,
            'confidence': np.nan,
            'in_gray_zone': False
        }, index=X.index)

        # Classify based on thresholds
        results.loc[probs < self.low_threshold, 'classification'] = 'negative'
        results.loc[probs < self.low_threshold, 'confidence'] = 'high'

        results.loc[probs > self.high_threshold, 'classification'] = 'positive'
        results.loc[probs > self.high_threshold, 'confidence'] = 'high'

        gray_mask = (probs >= self.low_threshold) & (probs <= self.high_threshold)
        results.loc[gray_mask, 'classification'] = 'gray_zone'
        results.loc[gray_mask, 'confidence'] = 'low'
        results.loc[gray_mask, 'in_gray_zone'] = True

        return results

    def get_gray_zone_indices(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get indices of patients in the gray zone.

        Args:
            X: Feature DataFrame

        Returns:
            Array of indices for gray zone patients
        """
        results = self.classify(X)
        return results[results['in_gray_zone']].index.values

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> GatekeeperMetrics:
        """
        Evaluate Gatekeeper performance.

        Args:
            X: Feature DataFrame
            y: True amyloid status

        Returns:
            GatekeeperMetrics with performance statistics
        """
        results = self.classify(X)

        # Total
        valid_mask = ~results['probability'].isna()
        n_total = valid_mask.sum()

        # Resolved (high confidence)
        resolved_mask = valid_mask & (results['confidence'] == 'high')
        n_resolved = resolved_mask.sum()

        # Gray zone
        gray_mask = valid_mask & results['in_gray_zone']
        n_gray_zone = gray_mask.sum()

        # Resolution rate
        resolution_rate = n_resolved / n_total if n_total > 0 else 0

        # Accuracy in resolved cases
        if n_resolved > 0:
            resolved_df = results[resolved_mask]
            y_resolved = y[resolved_mask]

            predicted_positive = resolved_df['classification'] == 'positive'
            correct = (predicted_positive.values == y_resolved.values)
            accuracy_resolved = correct.mean()

            # Sensitivity for positive classification
            truly_positive = y_resolved == 1
            if truly_positive.sum() > 0:
                sensitivity = (predicted_positive.values & truly_positive.values).sum() / truly_positive.sum()
            else:
                sensitivity = np.nan

            # Specificity for negative classification
            truly_negative = y_resolved == 0
            if truly_negative.sum() > 0:
                specificity = ((~predicted_positive.values) & truly_negative.values).sum() / truly_negative.sum()
            else:
                specificity = np.nan
        else:
            accuracy_resolved = np.nan
            sensitivity = np.nan
            specificity = np.nan

        return GatekeeperMetrics(
            n_total=n_total,
            n_resolved=n_resolved,
            n_gray_zone=n_gray_zone,
            resolution_rate=resolution_rate,
            accuracy_resolved=accuracy_resolved,
            sensitivity_positive=sensitivity,
            specificity_negative=specificity
        )

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_fitted:
            return {}

        return {
            'intercept': self.model.intercept_[0],
            'coefficient': self.model.coef_[0, 0]
        }

    def get_decision_boundary_z(self, probability: float = 0.5) -> float:
        """
        Calculate the p-tau217_Z value corresponding to a given probability.

        Args:
            probability: Target probability

        Returns:
            Z-score value at that probability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # logit(p) = intercept + coef * z
        # z = (logit(p) - intercept) / coef
        logit_p = np.log(probability / (1 - probability))
        z = (logit_p - self.model.intercept_[0]) / self.model.coef_[0, 0]
        return z


def create_gatekeeper(
    low_threshold: float = 0.25,
    high_threshold: float = 0.75
) -> GatekeeperModel:
    """
    Factory function to create a Gatekeeper model with specified thresholds.

    Args:
        low_threshold: Probability threshold for negative classification
        high_threshold: Probability threshold for positive classification

    Returns:
        Configured GatekeeperModel
    """
    return GatekeeperModel(
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
