"""
Stage 2: Reflex Model
=====================

Random Forest classifier applied exclusively to Gray Zone participants.
Integrates multiple biomarkers and engineered features to resolve
diagnostic ambiguity.

Features:
    - pTau217_Z: Harmonized p-tau217
    - NfL_Z: Harmonized neurofilament light
    - GFAP_Z: Harmonized glial fibrillary acidic protein
    - AB42_40_ratio_Z: Harmonized amyloid beta ratio (if available)
    - AGE: Raw age in years
    - APOE4_carrier: APOE epsilon 4 carrier status
    - tau_ab42_diff: log(pTau217) - log(AB42/40) ratio
    - gfap_tau_interaction: GFAP_Z * pTau217_Z
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ReflexResult:
    """Result of Reflex classification for a gray zone patient."""
    probability: float
    classification: str  # 'negative' or 'positive'
    confidence: float  # Model confidence


@dataclass
class ReflexMetrics:
    """Aggregate metrics for Reflex model performance."""
    n_gray_zone: int
    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    brier_score: float


class ReflexModel:
    """
    Stage 2 Reflex: Multi-marker Random Forest for gray zone resolution.

    Applied only to patients in the diagnostic gray zone identified by
    the Gatekeeper model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        """
        Initialize Reflex model.

        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth (constrained to limit overfitting)
            min_samples_leaf: Minimum samples per leaf
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create biologically motivated engineered features.

        Args:
            df: Input DataFrame with raw and harmonized features

        Returns:
            DataFrame with additional engineered features
        """
        result = df.copy()

        # 1. tau/amyloid divergence ratio (main driver in paper)
        # log(pTau217) - log(AB42/40)
        if 'pTau217_raw' in df.columns and 'AB42_40_ratio' in df.columns:
            result['tau_ab42_diff'] = (
                np.log1p(df['pTau217_raw']) -
                np.log1p(df['AB42_40_ratio'])
            )

        # 2. NfL:Age interaction (neurodegeneration vulnerability)
        if 'NfL_Z' in df.columns and 'AGE' in df.columns:
            age_z = (df['AGE'] - df['AGE'].mean()) / df['AGE'].std()
            result['nfl_age_interaction'] = df['NfL_Z'] * age_z

        # 3. GFAP:pTau217 interaction (inflammation x pathology)
        if 'GFAP_Z' in df.columns and 'pTau217_Z' in df.columns:
            result['gfap_tau_interaction'] = df['GFAP_Z'] * df['pTau217_Z']

        # 4. Raw age is passed through directly (RF is scale-invariant)
        # AGE column already exists in df, no engineering needed.

        # 5. pTau217/NfL ratio (tau pathology vs general neurodegeneration)
        if 'pTau217_raw' in df.columns and 'NfL_raw' in df.columns:
            result['tau_nfl_ratio'] = np.log1p(df['pTau217_raw']) - np.log1p(df['NfL_raw'])

        # 6. Age x APOE4 interaction (age-dependent genetic risk)
        if 'AGE' in df.columns and 'APOE4_carrier' in df.columns:
            age_z = (df['AGE'] - df['AGE'].mean()) / df['AGE'].std()
            result['age_apoe4_interaction'] = age_z * df['APOE4_carrier']

        # 7. Age x pTau217 interaction (age-dependent tau pathology)
        if 'pTau217_Z' in df.columns and 'AGE' in df.columns:
            age_z = (df['AGE'] - df['AGE'].mean()) / df['AGE'].std()
            result['age_ptau_interaction'] = age_z * df['pTau217_Z']

        # 8. GFAP / AB42_40 ratio (inflammation relative to amyloid)
        if 'GFAP_Z' in df.columns and 'AB42_40_ratio' in df.columns:
            result['gfap_ab42_ratio'] = df['GFAP_Z'] / (np.log1p(df['AB42_40_ratio']) + 0.01)

        # 9. pTau217 squared (capture nonlinearity)
        if 'pTau217_Z' in df.columns:
            result['ptau217_squared'] = df['pTau217_Z'] ** 2

        return result

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select available features for modeling.

        Prioritizes features based on biological relevance and availability.

        Args:
            df: DataFrame with all features

        Returns:
            List of feature column names to use
        """
        # Priority order of features
        feature_priority = [
            'pTau217_Z',           # Primary biomarker
            'tau_ab42_diff',       # Key divergence feature
            'GFAP_Z',              # Astrogliosis
            'NfL_Z',               # Neurodegeneration
            'AGE',                 # Raw age in years
            'APOE4_carrier',       # Genetic risk
            'nfl_age_interaction', # Interaction term
            'gfap_tau_interaction',# Interaction term
            'tau_nfl_ratio',       # Ratio feature
            'Hippocampus_norm',    # MRI: hippocampal atrophy
            'Entorhinal_norm',     # MRI: entorhinal thinning
            'age_apoe4_interaction',  # Interaction term
            'age_ptau_interaction',   # Interaction term
            'gfap_ab42_ratio',     # Ratio feature
            'ptau217_squared',     # Nonlinear term
        ]

        # Select features that are available and not all NaN
        available = []
        for feat in feature_priority:
            if feat in df.columns:
                if df[feat].notna().sum() > len(df) * 0.5:  # >50% non-missing
                    available.append(feat)

        return available

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None
    ) -> 'ReflexModel':
        """
        Fit the Reflex model on gray zone data.

        Args:
            X: Feature DataFrame (gray zone patients only)
            y: Binary amyloid status
            feature_cols: Optional specific feature columns to use

        Returns:
            self (fitted model)
        """
        # Engineer features
        X_eng = self._engineer_features(X)

        # Select features
        if feature_cols is None:
            self.feature_names = self._select_features(X_eng)
        else:
            self.feature_names = [f for f in feature_cols if f in X_eng.columns]

        if len(self.feature_names) < 2:
            raise ValueError(f"Insufficient features available: {self.feature_names}")

        # Prepare feature matrix
        X_features = X_eng[self.feature_names].values

        # Handle missing values with median imputation
        for i, col in enumerate(self.feature_names):
            col_values = X_features[:, i]
            nan_mask = np.isnan(col_values)
            if nan_mask.any():
                median_val = np.nanmedian(col_values)
                X_features[nan_mask, i] = median_val

        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        # Fit Random Forest
        valid_mask = ~np.isnan(y.values)
        self.model.fit(X_scaled[valid_mask], y.values[valid_mask])

        self.is_fitted = True
        self.train_n = valid_mask.sum()

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

        # Engineer features
        X_eng = self._engineer_features(X)

        # Prepare feature matrix
        X_features = X_eng[self.feature_names].values

        # Handle missing values
        for i, col in enumerate(self.feature_names):
            col_values = X_features[:, i]
            nan_mask = np.isnan(col_values)
            if nan_mask.any():
                median_val = np.nanmedian(col_values)
                X_features[nan_mask, i] = median_val

        # Scale
        X_scaled = self.scaler.transform(X_features)

        # Predict
        return self.model.predict_proba(X_scaled)[:, 1]

    def classify(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Classify gray zone patients.

        Args:
            X: Feature DataFrame
            threshold: Probability threshold for positive classification

        Returns:
            DataFrame with probability and classification
        """
        probs = self.predict_proba(X)

        return pd.DataFrame({
            'probability': probs,
            'classification': np.where(probs >= threshold, 'positive', 'negative'),
            'confidence': np.abs(probs - 0.5) * 2  # Confidence: 0 at threshold, 1 at extremes
        }, index=X.index)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            return pd.DataFrame()

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ReflexMetrics:
        """
        Evaluate Reflex model performance.

        Args:
            X: Feature DataFrame (gray zone)
            y: True amyloid status

        Returns:
            ReflexMetrics with performance statistics
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        valid_mask = ~np.isnan(y.values) & ~np.isnan(probs)
        y_valid = y.values[valid_mask]
        probs_valid = probs[valid_mask]
        preds_valid = preds[valid_mask]

        # AUC
        if len(np.unique(y_valid)) > 1:
            auc = roc_auc_score(y_valid, probs_valid)
        else:
            auc = np.nan

        # Accuracy
        accuracy = accuracy_score(y_valid, preds_valid)

        # Sensitivity
        truly_positive = y_valid == 1
        if truly_positive.sum() > 0:
            sensitivity = (preds_valid[truly_positive] == 1).mean()
        else:
            sensitivity = np.nan

        # Specificity
        truly_negative = y_valid == 0
        if truly_negative.sum() > 0:
            specificity = (preds_valid[truly_negative] == 0).mean()
        else:
            specificity = np.nan

        # Brier score
        brier = brier_score_loss(y_valid, probs_valid)

        return ReflexMetrics(
            n_gray_zone=len(y_valid),
            auc=auc,
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            brier_score=brier
        )


def create_reflex_model(
    n_estimators: int = 100,
    max_depth: int = 5
) -> ReflexModel:
    """
    Factory function to create a Reflex model.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth

    Returns:
        Configured ReflexModel
    """
    return ReflexModel(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
