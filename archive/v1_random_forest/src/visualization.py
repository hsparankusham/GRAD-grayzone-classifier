"""
Visualization Module for Gray Zone Classifier
==============================================

Generates publication-quality figures for the two-stage
Gatekeeper + Reflex diagnostic framework.

Figures:
    - Figure 1A: Table 1 (Baseline characteristics)
    - Figure 1B: Distribution of p-tau217 probabilities with thresholds
    - Figure 2A: ROC curve for Reflex model in gray zone
    - Figure 2B: Feature importance analysis
    - Figure 3A: Cross-platform consistency
    - Figure 4A: Calibration curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from typing import Dict, Optional, Any, Tuple
import warnings


class ResultsVisualizer:
    """Generate publication figures for Gray Zone Classifier results."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 150):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'negative': '#2E86AB',  # Blue
            'positive': '#A23B72',  # Magenta
            'gray_zone': '#F18F01', # Orange
            'primary': '#1a1a2e',   # Dark
            'secondary': '#4a4a6a', # Medium
        }

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_probability_distribution(
        self,
        predictions: pd.DataFrame,
        low_threshold: float = 0.25,
        high_threshold: float = 0.75,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 1B: Distribution of p-tau217-based probability estimates.

        Args:
            predictions: DataFrame with 'gatekeeper_prob' and 'true_amyloid'
            low_threshold: Low probability threshold
            high_threshold: High probability threshold
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        probs = predictions['gatekeeper_prob'].dropna()
        true_labels = predictions.loc[probs.index, 'true_amyloid']

        # Separate by true amyloid status
        probs_neg = probs[true_labels == 0]
        probs_pos = probs[true_labels == 1]

        # Plot histograms
        bins = np.linspace(0, 1, 30)
        ax.hist(probs_neg, bins=bins, alpha=0.6, color=self.colors['negative'],
                label=f'Amyloid Negative (n={len(probs_neg)})', density=True)
        ax.hist(probs_pos, bins=bins, alpha=0.6, color=self.colors['positive'],
                label=f'Amyloid Positive (n={len(probs_pos)})', density=True)

        # Add threshold lines
        ax.axvline(low_threshold, color='gray', linestyle='--', linewidth=2,
                   label=f'Low threshold ({low_threshold})')
        ax.axvline(high_threshold, color='gray', linestyle='--', linewidth=2,
                   label=f'High threshold ({high_threshold})')

        # Shade gray zone
        ax.axvspan(low_threshold, high_threshold, alpha=0.2, color=self.colors['gray_zone'],
                   label='Gray Zone')

        ax.set_xlabel('Predicted Amyloid Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Figure 1B: Distribution of p-tau217-based Amyloid Probability Estimates\nand Gatekeeper Thresholds',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_gray_zone_roc(
        self,
        predictions: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 2A: ROC curve for Reflex model in Gray Zone.

        Args:
            predictions: DataFrame with predictions
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)

        # Filter to gray zone
        gray_mask = predictions['stage'] == 'reflex'
        gray_preds = predictions[gray_mask]

        if len(gray_preds) < 10:
            ax.text(0.5, 0.5, 'Insufficient gray zone samples',
                    ha='center', va='center', fontsize=14)
            return fig

        y_true = gray_preds['true_amyloid'].values
        y_prob = gray_preds['predicted_prob'].values

        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2,
                label=f'Reflex Model (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1,
                label='Chance (AUC = 0.50)')

        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('Figure 2A: Receiver Operating Characteristic Curve\nReflex Model in Gray Zone Participants',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 2B: Feature importance from Reflex model.

        Args:
            feature_importance: DataFrame with 'feature' and 'importance'
            top_n: Number of top features to show
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        if len(feature_importance) == 0:
            ax.text(0.5, 0.5, 'No feature importance data available',
                    ha='center', va='center', fontsize=14)
            return fig

        # Get top features
        top_features = feature_importance.head(top_n)

        # Create bar plot
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['importance'].values,
                       color=self.colors['primary'], alpha=0.8)

        # Add error bars if available
        if 'std' in top_features.columns:
            ax.errorbar(top_features['importance'].values, y_pos,
                        xerr=top_features['std'].values,
                        fmt='none', color='black', capsize=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Figure 2B: Feature Importance Analysis\nReflex Model Key Biological Drivers',
                     fontsize=14, fontweight='bold')

        # Add value labels
        for i, (imp, bar) in enumerate(zip(top_features['importance'].values, bars)):
            ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_calibration_curve(
        self,
        predictions: pd.DataFrame,
        n_bins: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 4A: Calibration curve of integrated system.

        Args:
            predictions: DataFrame with predictions
            n_bins: Number of bins for calibration
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)

        y_true = predictions['true_amyloid'].values
        y_prob = predictions['predicted_prob'].values

        # Remove NaN
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Plot
        ax.plot(prob_pred, prob_true, 's-', color=self.colors['primary'],
                label='Integrated System', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        ax.set_xlabel('Predicted Amyloid Probability', fontsize=12)
        ax.set_ylabel('Actual Amyloid Proportion', fontsize=12)
        ax.set_title('Figure 4A: Calibration Curve of the Integrated Diagnostic System',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_cross_platform_consistency(
        self,
        predictions: pd.DataFrame,
        assay_col: str = 'assay',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 3A: Consistency across assay platforms.

        Args:
            predictions: DataFrame with predictions and assay info
            assay_col: Column name for assay platform
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)

        if assay_col not in predictions.columns:
            for ax in axes:
                ax.text(0.5, 0.5, 'Assay information not available',
                        ha='center', va='center', fontsize=14)
            return fig

        assays = predictions[assay_col].unique()

        # Left panel: Probability distributions by assay
        ax = axes[0]
        for i, assay in enumerate(assays):
            mask = predictions[assay_col] == assay
            probs = predictions.loc[mask, 'predicted_prob'].dropna()
            ax.hist(probs, bins=20, alpha=0.6, label=f'{assay} (n={len(probs)})',
                    density=True)

        ax.set_xlabel('Predicted Amyloid Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Probability Distribution by Assay', fontsize=12)
        ax.legend()

        # Right panel: Accuracy by assay
        ax = axes[1]
        accuracies = []
        assay_names = []
        for assay in assays:
            mask = predictions[assay_col] == assay
            assay_preds = predictions[mask]
            valid = ~assay_preds['true_amyloid'].isna() & ~assay_preds['predicted_prob'].isna()
            if valid.sum() > 0:
                acc = ((assay_preds.loc[valid, 'predicted_prob'] >= 0.5) ==
                       assay_preds.loc[valid, 'true_amyloid']).mean()
                accuracies.append(acc)
                assay_names.append(f'{assay}\n(n={valid.sum()})')

        bars = ax.bar(assay_names, accuracies, color=self.colors['primary'], alpha=0.8)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Classification Accuracy by Assay', fontsize=12)
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2f}', ha='center', fontsize=10)

        fig.suptitle('Figure 3A: Consistency of Integrated System Predictions\nAcross Assay Platforms',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_overall_roc(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot overall ROC curve for integrated system.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)

        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2,
                label=f'Integrated System (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1,
                label='Chance (AUC = 0.50)')

        ax.fill_between(fpr, tpr, alpha=0.2, color=self.colors['primary'])

        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('Overall ROC Curve: Integrated Gatekeeper + Reflex System',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def generate_cohort_table(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate Table 1: Baseline Demographic and Clinical Characteristics.

        Args:
            df: DataFrame with demographic and clinical data
            save_path: Optional path to save CSV

        Returns:
            DataFrame with summary statistics
        """
        table_data = []

        # Sample size
        table_data.append({
            'Characteristic': 'N',
            'Value': str(len(df))
        })

        # Age
        if 'AGE' in df.columns:
            age = df['AGE'].dropna()
            table_data.append({
                'Characteristic': 'Age, years (mean +/- SD)',
                'Value': f'{age.mean():.1f} +/- {age.std():.1f}'
            })

        # Sex
        if 'SEX_binary' in df.columns:
            male_pct = df['SEX_binary'].mean() * 100
            table_data.append({
                'Characteristic': 'Male, n (%)',
                'Value': f'{df["SEX_binary"].sum()} ({male_pct:.1f}%)'
            })

        # Education
        if 'PTEDUCAT' in df.columns:
            edu = df['PTEDUCAT'].dropna()
            table_data.append({
                'Characteristic': 'Education, years (mean +/- SD)',
                'Value': f'{edu.mean():.1f} +/- {edu.std():.1f}'
            })

        # APOE4
        if 'APOE4_carrier' in df.columns:
            apoe_pct = df['APOE4_carrier'].mean() * 100
            table_data.append({
                'Characteristic': 'APOE e4 carrier, n (%)',
                'Value': f'{df["APOE4_carrier"].sum()} ({apoe_pct:.1f}%)'
            })

        # Amyloid status
        if 'amyloid_positive' in df.columns:
            amy_pct = df['amyloid_positive'].mean() * 100
            table_data.append({
                'Characteristic': 'Amyloid positive, n (%)',
                'Value': f'{df["amyloid_positive"].sum()} ({amy_pct:.1f}%)'
            })

        # MMSE
        if 'MMSE' in df.columns:
            mmse = df['MMSE'].dropna()
            table_data.append({
                'Characteristic': 'MMSE (mean +/- SD)',
                'Value': f'{mmse.mean():.1f} +/- {mmse.std():.1f}'
            })

        # p-tau217
        if 'pTau217_raw' in df.columns:
            ptau = df['pTau217_raw'].dropna()
            table_data.append({
                'Characteristic': 'p-tau217 (median [IQR])',
                'Value': f'{ptau.median():.3f} [{ptau.quantile(0.25):.3f}-{ptau.quantile(0.75):.3f}]'
            })

        table = pd.DataFrame(table_data)

        if save_path:
            table.to_csv(save_path, index=False)

        return table

    def generate_all_figures(
        self,
        validation_results,
        df: pd.DataFrame,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate all publication figures.

        Args:
            validation_results: ValidationResults object
            df: Original DataFrame with demographics
            output_dir: Directory to save figures

        Returns:
            Dictionary mapping figure names to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        figures = {}

        # Figure 1B: Probability distribution
        fig1b_path = os.path.join(output_dir, 'figure_1b_probability_distribution.png')
        self.plot_probability_distribution(
            validation_results.predictions,
            save_path=fig1b_path
        )
        figures['Figure 1B'] = fig1b_path

        # Figure 2A: Gray zone ROC
        fig2a_path = os.path.join(output_dir, 'figure_2a_gray_zone_roc.png')
        self.plot_gray_zone_roc(
            validation_results.predictions,
            save_path=fig2a_path
        )
        figures['Figure 2A'] = fig2a_path

        # Figure 2B: Feature importance
        if validation_results.feature_importance is not None and len(validation_results.feature_importance) > 0:
            fig2b_path = os.path.join(output_dir, 'figure_2b_feature_importance.png')
            self.plot_feature_importance(
                validation_results.feature_importance,
                save_path=fig2b_path
            )
            figures['Figure 2B'] = fig2b_path

        # Figure 4A: Calibration curve
        fig4a_path = os.path.join(output_dir, 'figure_4a_calibration.png')
        self.plot_calibration_curve(
            validation_results.predictions,
            save_path=fig4a_path
        )
        figures['Figure 4A'] = fig4a_path

        # Overall ROC
        roc_path = os.path.join(output_dir, 'overall_roc.png')
        self.plot_overall_roc(
            validation_results.fpr,
            validation_results.tpr,
            validation_results.overall_auc,
            save_path=roc_path
        )
        figures['Overall ROC'] = roc_path

        # Table 1
        table_path = os.path.join(output_dir, 'table_1_demographics.csv')
        self.generate_cohort_table(df, save_path=table_path)
        figures['Table 1'] = table_path

        print(f"Generated {len(figures)} figures in {output_dir}")
        return figures
