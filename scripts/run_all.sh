#!/bin/bash
# ============================================================
# GRAD: Reproduce All Manuscript Results
# ============================================================
# Prerequisites:
#   1. ADNI and A4 data access (see data/README_data.md)
#   2. Python environment: pip install -r requirements.txt
#
# Usage:
#   bash scripts/run_all.sh --adni-dir /path/to/adni --a4-dir /path/to/a4
#
# Or to run the demo with synthetic data (no data access needed):
#   bash scripts/run_all.sh --demo
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

# Step 1: ADNI Internal Validation (LOOCV)
echo "[1/4] Running ADNI Leave-One-Out Cross-Validation..."
python3 scripts/run_authoritative_loocv.py
echo "       Done. Output: results/adni_loocv_predictions.csv"
echo ""

# Step 2: A4 External Validation
echo "[2/4] Running A4 + LEARN External Validation..."
python3 scripts/run_a4_binary_validation.py
echo "       Done. Output: results/a4_binary_validation_predictions.csv"
echo ""

# Step 3: Supplementary Analyses
echo "[3/4] Running supplementary analyses..."
python3 scripts/run_nfl_ablation.py
python3 scripts/run_subgroup_analysis.py
python3 scripts/run_calibration_analysis.py
echo "       Done. Outputs in results/tables/"
echo ""

# Step 4: Generate Figures
echo "[4/4] Generating manuscript figures..."
python3 scripts/generate_figure2_combined.py
python3 scripts/generate_figure3_final.py
python3 scripts/generate_figure4.py
python3 scripts/generate_manuscript_figures.py
python3 scripts/generate_figure_s3_subgroup.py
python3 scripts/generate_stard_diagram.py
echo "       Done. Outputs in results/figures/"
echo ""

echo "============================================================"
echo "  All results generated successfully."
echo "  Compare against expected values in config/config.yaml"
echo "============================================================"
