"""
Assay Harmonization Module for Gray Zone Classifier
====================================================

Implements reference-based Z-scoring for cross-platform harmonization
of plasma biomarkers. Harmonization is performed within the cross-validation
framework to prevent data leakage.

Reference Population:
    Cognitively normal, amyloid-negative participants (AV45 SUVR <= 1.11)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class HarmonizationParams:
    """Parameters for harmonization from reference population."""
    mean: float
    std: float
    n_reference: int
    assay: str


class AssayHarmonizer:
    """
    Reference-based Z-score harmonization for plasma biomarkers.

    Uses cognitively normal, amyloid-negative participants as reference
    population for each assay platform.
    """

    def __init__(
        self,
        biomarkers: List[str] = None,
        reference_dx: List[str] = None,
        amyloid_threshold: float = 1.11
    ):
        """
        Initialize harmonizer.

        Args:
            biomarkers: List of biomarker columns to harmonize
            reference_dx: Diagnosis values for reference population (CN)
            amyloid_threshold: AV45 SUVR threshold for amyloid negativity
        """
        self.biomarkers = biomarkers or ['pTau217_raw', 'NfL_raw', 'GFAP_raw']
        self.reference_dx = reference_dx or ['CN', 'NL', 'Normal']
        self.amyloid_threshold = amyloid_threshold
        self.params: Dict[str, Dict[str, HarmonizationParams]] = {}

    def _identify_reference_population(
        self,
        df: pd.DataFrame,
        assay: str
    ) -> pd.DataFrame:
        """
        Identify reference population for a given assay.

        Reference: Cognitively normal AND amyloid-negative

        Args:
            df: DataFrame with clinical data
            assay: Assay platform name

        Returns:
            DataFrame subset of reference population
        """
        # Filter by assay
        assay_mask = df['assay'] == assay

        # Filter by cognitive status (CN)
        if 'DX' in df.columns:
            dx_mask = df['DX'].isin(self.reference_dx) | df['DX_bl'].isin(self.reference_dx)
        elif 'DX_bl' in df.columns:
            dx_mask = df['DX_bl'].isin(self.reference_dx)
        else:
            # If no DX available, use all participants
            dx_mask = pd.Series(True, index=df.index)

        # Filter by amyloid status
        if 'amyloid_positive' in df.columns:
            amyloid_mask = df['amyloid_positive'] == 0
        elif 'AV45' in df.columns:
            amyloid_mask = df['AV45'] <= self.amyloid_threshold
        else:
            amyloid_mask = pd.Series(True, index=df.index)

        return df[assay_mask & dx_mask & amyloid_mask]

    def fit(self, df: pd.DataFrame) -> 'AssayHarmonizer':
        """
        Compute harmonization parameters from training data.

        Args:
            df: Training DataFrame with biomarkers and clinical data

        Returns:
            self (fitted harmonizer)
        """
        self.params = {}

        # Get unique assays
        assays = df['assay'].unique()

        for assay in assays:
            self.params[assay] = {}
            ref_pop = self._identify_reference_population(df, assay)

            if len(ref_pop) < 5:
                print(f"  Warning: Only {len(ref_pop)} reference subjects for {assay}")
                # Fall back to all subjects from this assay
                ref_pop = df[df['assay'] == assay]

            for biomarker in self.biomarkers:
                if biomarker not in df.columns:
                    continue

                # Log-transform for right-skewed biomarkers
                values = ref_pop[biomarker].dropna()
                if len(values) > 0:
                    log_values = np.log1p(values)
                    self.params[assay][biomarker] = HarmonizationParams(
                        mean=log_values.mean(),
                        std=log_values.std() if log_values.std() > 0 else 1.0,
                        n_reference=len(values),
                        assay=assay
                    )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply harmonization to data.

        Args:
            df: DataFrame to harmonize

        Returns:
            DataFrame with harmonized Z-score columns
        """
        result = df.copy()

        for biomarker in self.biomarkers:
            if biomarker not in df.columns:
                continue

            z_col = biomarker.replace('_raw', '_Z')
            result[z_col] = np.nan

            for assay in df['assay'].unique():
                if assay not in self.params or biomarker not in self.params[assay]:
                    # If no params, use global standardization
                    mask = df['assay'] == assay
                    values = df.loc[mask, biomarker]
                    log_values = np.log1p(values)
                    result.loc[mask, z_col] = (
                        (log_values - log_values.mean()) / log_values.std()
                    )
                else:
                    params = self.params[assay][biomarker]
                    mask = df['assay'] == assay
                    values = df.loc[mask, biomarker]
                    log_values = np.log1p(values)
                    result.loc[mask, z_col] = (
                        (log_values - params.mean) / params.std
                    )

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class CrossValidationHarmonizer:
    """
    Harmonizer that respects cross-validation splits.

    Ensures no data leakage by re-computing harmonization parameters
    within each CV fold using only training data.
    """

    def __init__(self, base_harmonizer: AssayHarmonizer = None):
        """
        Initialize CV harmonizer.

        Args:
            base_harmonizer: Base harmonizer to use (will be cloned per fold)
        """
        self.base_harmonizer = base_harmonizer or AssayHarmonizer()

    def harmonize_loocv(
        self,
        df: pd.DataFrame,
        fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Harmonize data using LOOCV-aware procedure.

        For each fold:
        1. Compute harmonization params using only training indices
        2. Apply to test index

        Args:
            df: Full DataFrame
            fold_indices: List of (train_idx, test_idx) tuples

        Returns:
            DataFrame with harmonized values (computed without leakage)
        """
        result = df.copy()

        # Initialize Z-score columns
        for biomarker in self.base_harmonizer.biomarkers:
            if biomarker in df.columns:
                z_col = biomarker.replace('_raw', '_Z')
                result[z_col] = np.nan

        # Process each fold
        for train_idx, test_idx in fold_indices:
            # Clone harmonizer
            harmonizer = AssayHarmonizer(
                biomarkers=self.base_harmonizer.biomarkers,
                reference_dx=self.base_harmonizer.reference_dx,
                amyloid_threshold=self.base_harmonizer.amyloid_threshold
            )

            # Fit on training data only
            train_data = df.iloc[train_idx]
            harmonizer.fit(train_data)

            # Transform test data
            test_data = df.iloc[test_idx]
            transformed = harmonizer.transform(test_data)

            # Store results
            for biomarker in self.base_harmonizer.biomarkers:
                if biomarker in df.columns:
                    z_col = biomarker.replace('_raw', '_Z')
                    result.loc[result.index[test_idx], z_col] = transformed[z_col].values

        return result


def harmonize_for_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    biomarkers: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonize train and test sets for external validation.

    Fits harmonization on training data (ADNI) and applies to test data (A4).

    Args:
        train_df: Training DataFrame (e.g., ADNI)
        test_df: Test DataFrame (e.g., A4)
        biomarkers: List of biomarkers to harmonize

    Returns:
        Tuple of (harmonized_train, harmonized_test)
    """
    harmonizer = AssayHarmonizer(biomarkers=biomarkers)

    # Fit on training data
    harmonizer.fit(train_df)

    # Transform both
    train_harmonized = harmonizer.transform(train_df)
    test_harmonized = harmonizer.transform(test_df)

    return train_harmonized, test_harmonized
