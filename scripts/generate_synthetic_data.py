#!/usr/bin/env python3
"""
Generate Synthetic Demo Dataset
================================
Creates a synthetic cohort that mimics the statistical properties of the
ADNI training data without containing any real participant information.

This allows reviewers and users to test the GRAD pipeline without
requiring ADNI or A4 data access.

Output: data/synthetic/synthetic_cohort.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible
rng = np.random.RandomState(42)

N = 320  # Match ADNI cohort size
N_POS = 155  # Amyloid-positive
N_NEG = N - N_POS

# ── Demographics ──────────────────────────────────────────────────────
# Based on published ADNI cohort characteristics (no individual data used)
ages = np.concatenate([
    rng.normal(73.5, 7.2, N_NEG),   # Amyloid-negative: slightly younger
    rng.normal(75.1, 6.8, N_POS),   # Amyloid-positive: slightly older
])
ages = np.clip(ages, 55, 95).round(1)

sex = rng.binomial(1, 0.52, N)  # ~52% male in ADNI

apoe4 = np.concatenate([
    rng.binomial(1, 0.22, N_NEG),   # ~22% APOE4 carriers in amyloid-negative
    rng.binomial(1, 0.58, N_POS),   # ~58% APOE4 carriers in amyloid-positive
])

dx = []
for i in range(N):
    if i < N_NEG:
        dx.append(rng.choice(['CN', 'MCI', 'Dementia'], p=[0.55, 0.38, 0.07]))
    else:
        dx.append(rng.choice(['CN', 'MCI', 'Dementia'], p=[0.25, 0.42, 0.33]))

amyloid = np.array([0] * N_NEG + [1] * N_POS)

# ── Biomarkers ────────────────────────────────────────────────────────
# Synthetic biomarker values based on published population distributions
# (Ashton et al. 2024, Palmqvist et al. 2023)

# pTau217 (pg/mL) — key discriminator
ptau217 = np.concatenate([
    rng.lognormal(np.log(0.15), 0.5, N_NEG),    # Amyloid-negative: low
    rng.lognormal(np.log(0.55), 0.45, N_POS),   # Amyloid-positive: high
])
ptau217 = np.clip(ptau217, 0.01, 5.0)

# AB42/40 ratio — lower in amyloid-positive
ab42_40 = np.concatenate([
    rng.normal(0.082, 0.012, N_NEG),
    rng.normal(0.058, 0.010, N_POS),
])
ab42_40 = np.clip(ab42_40, 0.02, 0.15)

# GFAP (pg/mL) — elevated in AD
gfap = np.concatenate([
    rng.lognormal(np.log(120), 0.45, N_NEG),
    rng.lognormal(np.log(180), 0.50, N_POS),
])
gfap = np.clip(gfap, 20, 800)

# NfL (pg/mL) — elevated in neurodegeneration
nfl = np.concatenate([
    rng.lognormal(np.log(18), 0.40, N_NEG),
    rng.lognormal(np.log(24), 0.45, N_POS),
])
nfl = np.clip(nfl, 3, 150)

# AB42 and AB40 raw (derive from ratio)
ab40 = rng.lognormal(np.log(280), 0.25, N)
ab42 = ab42_40 * ab40

# Assay platform
assay = rng.choice(['UPENN', 'Janssen'], N, p=[0.75, 0.25])

# MMSE
mmse = np.concatenate([
    rng.normal(28.5, 1.5, N_NEG),
    rng.normal(25.8, 4.2, N_POS),
])
mmse = np.clip(mmse, 10, 30).round(0).astype(int)

# Shuffle to avoid ordered blocks
idx = rng.permutation(N)

df = pd.DataFrame({
    'participant_id': [f'SYNTH_{i:04d}' for i in range(N)],
    'AGE': ages[idx],
    'SEX_binary': sex[idx],
    'DX': [dx[i] for i in idx],
    'APOE4_carrier': apoe4[idx],
    'MMSE': mmse[idx],
    'pTau217_raw': ptau217[idx].round(4),
    'AB42_raw': ab42[idx].round(2),
    'AB40_raw': ab40[idx].round(2),
    'AB42_40_ratio': ab42_40[idx].round(5),
    'NfL_raw': nfl[idx].round(2),
    'GFAP_raw': gfap[idx].round(2),
    'assay': assay[idx],
    'amyloid_positive': amyloid[idx],
})

out_dir = Path(__file__).parent.parent / 'data' / 'synthetic'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'synthetic_cohort.csv'
df.to_csv(out_path, index=False)

print(f'Synthetic cohort saved: {out_path}')
print(f'  N = {len(df)} ({df.amyloid_positive.sum()} amyloid+, {(1-df.amyloid_positive).sum():.0f} amyloid-)')
print(f'  Age: {df.AGE.mean():.1f} +/- {df.AGE.std():.1f}')
print(f'  Male: {df.SEX_binary.mean()*100:.1f}%')
print(f'  APOE4 carriers: {df.APOE4_carrier.mean()*100:.1f}%')
print(f'  pTau217 range: {df.pTau217_raw.min():.3f} - {df.pTau217_raw.max():.3f} pg/mL')
print(f'  Diagnosis: {df.DX.value_counts().to_dict()}')
