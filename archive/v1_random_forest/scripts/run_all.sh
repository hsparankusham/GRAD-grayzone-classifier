#!/bin/bash
# ============================================================
# GRAD: Reproduce All Manuscript Results
# ============================================================
# Prerequisites:
#   1. ADNI and A4 data access (see data/README_data.md)
#   2. Python environment: pip install -r requirements.txt
#
# Data location is resolved by scripts/_grad_paths.py. Override with:
#   export GRAD_DATA_DIR=/path/to/data
#
# Usage:
#   bash scripts/run_all.sh          # full reproduction
#   bash scripts/run_all.sh --demo   # synthetic data, no access needed
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ "$1" = "--demo" ]; then
    echo "Running GRAD demo with synthetic data..."
    python3 scripts/run_demo.py
    exit 0
fi

echo "============================================================"
echo "  GRAD: Full Manuscript Reproduction Pipeline"
echo "============================================================"
echo ""

# ---- Core analyses (must run first; every figure reads their output) ----
echo "[1/4] ADNI leave-one-out cross-validation..."
python3 scripts/run_authoritative_loocv.py
echo ""

echo "[2/4] A4 + LEARN external validation, and MRI enhancement..."
python3 scripts/run_a4_binary_validation.py
python3 scripts/gray_zone_mri_enhancement.py
echo ""

# ---- Supplementary analyses ----
echo "[3/4] Supplementary analyses..."
python3 scripts/run_nfl_ablation.py
python3 scripts/run_subgroup_analysis.py
python3 scripts/run_calibration_analysis.py
python3 scripts/run_reflex_training_comparison.py
echo ""

# ---- Figures ----
# generate_all_figures.py is the primary generator (Figures 2, 3, 4, 5, S1, S2, S3).
# generate_manuscript_figures.py adds Figures 1 and 6.
echo "[4/4] Generating figures..."
python3 scripts/generate_all_figures.py
python3 scripts/generate_manuscript_figures.py
python3 scripts/generate_figure6_cost.py
python3 scripts/generate_figure_panels.py
echo ""

echo "============================================================"
echo "  Done. Outputs:"
echo "    results/            prediction CSVs"
echo "    results/figures/    manuscript figures (+ panels/)"
echo "    results/tables/     supplementary tables"
echo "  Compare against expected values in config/config.yaml"
echo "============================================================"
