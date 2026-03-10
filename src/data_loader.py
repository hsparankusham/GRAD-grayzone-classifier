"""
Data Loading Module for Gray Zone Classifier
=============================================

Loads and preprocesses ADNI and A4 datasets for the two-stage
Gatekeeper + Reflex diagnostic framework.

ADNI provides:
    - UPENN plasma biomarkers: pT217_F, AB42_F, AB40_F, NfL_Q, GFAP_Q
    - Janssen plasma p-tau217: DILUTION_CORRECTED_CONC
    - ADNIMERGE: Demographics, APOE4, AV45 (amyloid PET)

A4 provides:
    - pTau217 from Lilly assay
    - Roche plasma: GFAP, NfL, AB42, AB40
    - SUBJINFO: Demographics, APOE4, AMYLCENT (amyloid centiloid)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


class ADNIDataLoader:
    """Load and merge ADNI biomarker and clinical data."""

    def __init__(self, base_dir: str):
        """
        Initialize ADNI data loader.

        Args:
            base_dir: Path to syntropi-ai-ADNI directory
        """
        self.base_dir = base_dir
        self.paths = {
            'upenn': os.path.join(base_dir, 'Pathology', 'UPENN_PlasmaBiomarkers.csv'),
            'janssen': os.path.join(base_dir, 'Pathology', 'JANSSEN_PLASMA_P217_TAU_18Dec2025.csv'),
            'merge': os.path.join(base_dir, 'ADNIMERGE2025.csv'),
            'apoe': os.path.join(base_dir, 'Reserve', 'Genetics', 'APOERES_26Dec2025.csv'),
        }

    def load_upenn_biomarkers(self) -> pd.DataFrame:
        """Load UPENN plasma biomarkers."""
        df = pd.read_csv(self.paths['upenn'])

        # Rename columns for consistency
        df = df.rename(columns={
            'pT217_F': 'pTau217_raw',
            'AB42_F': 'AB42_raw',
            'AB40_F': 'AB40_raw',
            'AB42_AB40_F': 'AB42_40_ratio',
            'NfL_Q': 'NfL_raw',
            'GFAP_Q': 'GFAP_raw'
        })

        # Filter out missing/invalid values (coded as -4)
        numeric_cols = ['pTau217_raw', 'AB42_raw', 'AB40_raw', 'AB42_40_ratio', 'NfL_raw', 'GFAP_raw']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < 0, col] = np.nan

        df['assay'] = 'UPENN'
        df['PTID'] = df['PTID'].astype(str)

        return df[['PTID', 'RID', 'VISCODE', 'VISCODE2', 'pTau217_raw', 'AB42_raw',
                   'AB40_raw', 'AB42_40_ratio', 'NfL_raw', 'GFAP_raw', 'assay']]

    def load_janssen_biomarkers(self) -> pd.DataFrame:
        """Load Janssen plasma p-tau217."""
        df = pd.read_csv(self.paths['janssen'])

        df = df.rename(columns={
            'DILUTION_CORRECTED_CONC': 'pTau217_raw'
        })

        df['pTau217_raw'] = pd.to_numeric(df['pTau217_raw'], errors='coerce')
        df['assay'] = 'Janssen'
        df['PTID'] = df['PTID'].astype(str)

        return df[['PTID', 'RID', 'VISCODE2', 'pTau217_raw', 'assay']]

    def load_adnimerge(self) -> pd.DataFrame:
        """Load ADNIMERGE for demographics and amyloid PET."""
        df = pd.read_csv(self.paths['merge'])

        # Select relevant columns
        cols = ['RID', 'PTID', 'VISCODE', 'DX', 'DX_bl', 'AGE', 'PTGENDER',
                'PTEDUCAT', 'APOE4', 'AV45', 'AV45_bl', 'MMSE',
                'Hippocampus', 'Entorhinal', 'ICV']
        df = df[[c for c in cols if c in df.columns]]

        # Convert AV45 to numeric
        df['AV45'] = pd.to_numeric(df['AV45'], errors='coerce')
        df['AV45_bl'] = pd.to_numeric(df['AV45_bl'], errors='coerce')

        # Define amyloid positivity: AV45 SUVR > 1.11
        df['amyloid_positive'] = (df['AV45'] > 1.11).astype(int)
        df.loc[df['AV45'].isna(), 'amyloid_positive'] = np.nan

        # Use baseline AV45 if current visit AV45 is missing
        df.loc[df['AV45'].isna() & df['AV45_bl'].notna(), 'amyloid_positive'] = (
            df.loc[df['AV45'].isna() & df['AV45_bl'].notna(), 'AV45_bl'] > 1.11
        ).astype(int)

        # ICV-normalized MRI features (matching A4 MRI enhancement approach)
        if 'Hippocampus' in df.columns and 'ICV' in df.columns:
            df['Hippocampus'] = pd.to_numeric(df['Hippocampus'], errors='coerce')
            df['Entorhinal'] = pd.to_numeric(df['Entorhinal'], errors='coerce')
            df['ICV'] = pd.to_numeric(df['ICV'], errors='coerce')
            df['Hippocampus_norm'] = (df['Hippocampus'] / df['ICV']) * 1000
            df['Entorhinal_norm'] = (df['Entorhinal'] / df['ICV']) * 1000

        df['PTID'] = df['PTID'].astype(str)

        return df

    def merge_data(self, use_baseline_only: bool = True) -> pd.DataFrame:
        """
        Merge UPENN and Janssen biomarkers with ADNIMERGE.

        Args:
            use_baseline_only: If True, use only baseline visits

        Returns:
            Merged DataFrame with all biomarkers and amyloid status
        """
        # Load all data sources
        upenn = self.load_upenn_biomarkers()
        janssen = self.load_janssen_biomarkers()
        merge = self.load_adnimerge()

        # For UPENN: merge on PTID and VISCODE2
        upenn_merged = upenn.merge(
            merge,
            on=['PTID', 'RID'],
            how='inner',
            suffixes=('', '_merge')
        )

        # Match visits - prefer exact VISCODE2 match
        upenn_merged = upenn_merged[
            (upenn_merged['VISCODE2'] == upenn_merged['VISCODE']) |
            (upenn_merged['VISCODE2'].str.contains('bl', case=False, na=False) &
             upenn_merged['VISCODE'].str.contains('bl', case=False, na=False))
        ].copy()

        # For Janssen: merge on PTID and VISCODE2
        janssen_merged = janssen.merge(
            merge,
            left_on=['PTID', 'RID'],
            right_on=['PTID', 'RID'],
            how='inner'
        )

        # Match visits
        janssen_merged = janssen_merged[
            (janssen_merged['VISCODE2'] == janssen_merged['VISCODE']) |
            (janssen_merged['VISCODE2'].str.contains('bl', case=False, na=False) &
             janssen_merged['VISCODE'].str.contains('bl', case=False, na=False))
        ].copy()

        # Combine UPENN and Janssen
        # Prefer UPENN as it has more biomarkers
        combined = upenn_merged.copy()

        # Add Janssen patients not in UPENN
        janssen_only = janssen_merged[~janssen_merged['PTID'].isin(upenn_merged['PTID'])]
        if len(janssen_only) > 0:
            combined = pd.concat([combined, janssen_only], ignore_index=True)

        # If baseline only, keep first visit per patient
        if use_baseline_only:
            combined = combined.sort_values(['PTID', 'VISCODE2'])
            combined = combined.groupby('PTID').first().reset_index()

        # Filter to patients with amyloid PET and p-tau217
        combined = combined[
            combined['amyloid_positive'].notna() &
            combined['pTau217_raw'].notna()
        ].copy()

        # Create APOE4 carrier flag
        combined['APOE4_carrier'] = (combined['APOE4'] > 0).astype(int)
        combined.loc[combined['APOE4'].isna(), 'APOE4_carrier'] = np.nan

        # Create sex binary (Male=1, Female=0)
        combined['SEX_binary'] = (combined['PTGENDER'] == 'Male').astype(int)

        print(f"ADNI merged dataset: {len(combined)} participants")
        print(f"  - UPENN assay: {(combined['assay'] == 'UPENN').sum()}")
        print(f"  - Janssen assay: {(combined['assay'] == 'Janssen').sum()}")
        print(f"  - Amyloid positive: {combined['amyloid_positive'].sum()} ({combined['amyloid_positive'].mean()*100:.1f}%)")

        return combined


class A4DataLoader:
    """Load and merge A4 Study biomarker and clinical data."""

    def __init__(self, base_dir: str):
        """
        Initialize A4 data loader.

        Args:
            base_dir: Path to syntropi-ai-A4 directory
        """
        self.base_dir = base_dir
        self.paths = {
            'ptau217': os.path.join(base_dir, 'Clinical', 'External Data', 'biomarker_pTau217.csv'),
            'roche': os.path.join(base_dir, 'Clinical', 'External Data', 'biomarker_Plasma_Roche_Results.csv'),
            'subjinfo': os.path.join(base_dir, 'Clinical', 'Derived Data', 'SUBJINFO.csv'),
            'demog': os.path.join(base_dir, 'Clinical', 'Raw Data', 'ptdemog.csv'),
        }

    def load_ptau217(self) -> pd.DataFrame:
        """Load Lilly pTau217 measurements."""
        df = pd.read_csv(self.paths['ptau217'])

        # Handle <LLOQ values - use ORRESRAW if available
        df['pTau217_raw'] = pd.to_numeric(df['ORRES'], errors='coerce')
        df.loc[df['pTau217_raw'].isna(), 'pTau217_raw'] = pd.to_numeric(
            df.loc[df['pTau217_raw'].isna(), 'ORRESRAW'], errors='coerce'
        )

        df['assay'] = 'Lilly'

        return df[['BID', 'VISCODE', 'pTau217_raw', 'assay']]

    def load_roche_biomarkers(self) -> pd.DataFrame:
        """Load Roche plasma biomarkers (GFAP, NfL, AB42, AB40)."""
        df = pd.read_csv(self.paths['roche'])

        # Pivot to wide format
        df['LABRESN'] = pd.to_numeric(df['LABRESN'], errors='coerce')

        # Clean test codes
        df['LBTESTCD'] = df['LBTESTCD'].str.strip()

        # Pivot
        pivot = df.pivot_table(
            index=['BID', 'VISCODE'],
            columns='LBTESTCD',
            values='LABRESN',
            aggfunc='first'
        ).reset_index()

        # Rename columns
        rename_map = {
            'GFAP': 'GFAP_raw',
            'NF-L': 'NfL_raw',
            'AMYLB42': 'AB42_raw',
            'AMYLB40': 'AB40_raw'
        }
        pivot = pivot.rename(columns=rename_map)

        # Calculate AB42/40 ratio (convert AB40 from ng/mL to pg/mL)
        if 'AB42_raw' in pivot.columns and 'AB40_raw' in pivot.columns:
            pivot['AB42_40_ratio'] = pivot['AB42_raw'] / (pivot['AB40_raw'] * 1000)

        # Convert GFAP from ng/mL to pg/mL for comparability
        if 'GFAP_raw' in pivot.columns:
            pivot['GFAP_raw'] = pivot['GFAP_raw'] * 1000

        return pivot

    def load_subjinfo(self) -> pd.DataFrame:
        """Load subject info including demographics and amyloid status."""
        df = pd.read_csv(self.paths['subjinfo'])

        # Select relevant columns
        cols = ['BID', 'TX', 'AGEYR', 'SEX', 'EDCCNTU', 'APOEGNPRSNFLG',
                'SUVRCER', 'AMYLCENT', 'MMSETSV1']
        df = df[[c for c in cols if c in df.columns]]

        # Rename for consistency
        df = df.rename(columns={
            'AGEYR': 'AGE',
            'EDCCNTU': 'PTEDUCAT',
            'APOEGNPRSNFLG': 'APOE4_carrier',
            'MMSETSV1': 'MMSE'
        })

        # A4 includes both treatment arm (amyloid-positive, SUVR > 1.15)
        # AND LEARN observational arm (screen-negative, amyloid-negative)
        # Use centiloid >= 20 as amyloid-positive threshold (standard cutoff)
        df['AMYLCENT'] = pd.to_numeric(df['AMYLCENT'], errors='coerce')
        df['amyloid_positive'] = (df['AMYLCENT'] >= 20).astype(int)
        df.loc[df['AMYLCENT'].isna(), 'amyloid_positive'] = np.nan

        # Sex: 1=Male, 2=Female in A4
        df['SEX_binary'] = (df['SEX'] == 1).astype(int)

        return df

    def merge_data(self, use_baseline_only: bool = True) -> pd.DataFrame:
        """
        Merge A4 biomarkers with subject info.

        Args:
            use_baseline_only: If True, use only baseline visits

        Returns:
            Merged DataFrame with all biomarkers
        """
        # Load all data sources
        ptau217 = self.load_ptau217()
        roche = self.load_roche_biomarkers()
        subjinfo = self.load_subjinfo()

        # Get baseline pTau217
        ptau217_bl = ptau217.sort_values(['BID', 'VISCODE'])
        if use_baseline_only:
            ptau217_bl = ptau217_bl.groupby('BID').first().reset_index()

        # Get baseline Roche biomarkers
        roche_bl = roche.sort_values(['BID', 'VISCODE'])
        if use_baseline_only:
            roche_bl = roche_bl.groupby('BID').first().reset_index()

        # Merge
        combined = subjinfo.merge(ptau217_bl[['BID', 'pTau217_raw', 'assay']], on='BID', how='inner')
        combined = combined.merge(
            roche_bl[['BID', 'GFAP_raw', 'NfL_raw', 'AB42_raw', 'AB40_raw', 'AB42_40_ratio']],
            on='BID',
            how='left'
        )

        # Filter to patients with p-tau217
        combined = combined[combined['pTau217_raw'].notna()].copy()

        n_with_label = combined['amyloid_positive'].notna().sum()
        n_pos = int(combined['amyloid_positive'].sum())
        n_neg = int(n_with_label - n_pos)
        print(f"A4 merged dataset: {len(combined)} participants")
        print(f"  - With GFAP: {combined['GFAP_raw'].notna().sum()}")
        print(f"  - With NfL: {combined['NfL_raw'].notna().sum()}")
        print(f"  - With AB42/40: {combined['AB42_40_ratio'].notna().sum()}")
        print(f"  - Amyloid positive (CL>=20): {n_pos} ({n_pos/n_with_label*100:.1f}%)")
        print(f"  - Amyloid negative (CL<20): {n_neg} ({n_neg/n_with_label*100:.1f}%)")

        return combined


def load_adni_data(base_dir: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load ADNI data."""
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'syntropi-ai-ADNI'
        )
    loader = ADNIDataLoader(base_dir)
    return loader.merge_data()


def load_a4_data(base_dir: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load A4 data."""
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'syntropi-ai-A4'
        )
    loader = A4DataLoader(base_dir)
    return loader.merge_data()
