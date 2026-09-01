"""
GRAD: Gatekeeper-Reflex for Alzheimer's Diagnostics
=====================================================
A two-stage machine learning algorithm that resolves the plasma p-tau217
diagnostic gray zone for amyloid PET prediction.

Modules:
    data_loader   - Load and merge ADNI and A4 Study datasets
    harmonizer    - Cross-platform Z-score normalization
    gatekeeper    - Stage 1: Univariate logistic regression screen
    reflex        - Stage 2: Multi-marker random forest classifier
    pipeline      - Orchestrates the two-stage workflow
    validation    - Leave-one-out cross-validation framework
    visualization - Publication-grade figure generation
"""

__version__ = "1.0.0"
