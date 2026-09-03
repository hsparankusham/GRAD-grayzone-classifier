"""
Central path resolution for every GRAD script.
==============================================

This module exposes the project's data and output locations. Every script
resolves paths through here, so moving or renaming the project directory cannot
break them — nothing counts directory levels.

The restricted-access cohorts are not in the repository. Point at them either
with environment variables:

    export GRAD_DATA_DIR=/path/to/data        # parent of the two cohort folders
    export GRAD_ADNI_DIR=ADNI                 # optional, if named differently
    export GRAD_A4_DIR=A4                     # optional

or by creating ``scripts/_local_paths.py`` (gitignored) with the same names as
module-level variables. See data/README_data.md.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- restricted-access data (not in the repo; see data/README_data.md) -----
try:
    import _local_paths as _local            # gitignored, machine specific
except ImportError:
    _local = None


def _resolve(env, attr, default):
    if os.environ.get(env):
        return os.environ[env]
    return getattr(_local, attr, default) if _local else default


DATA_DIR = Path(_resolve('GRAD_DATA_DIR', 'GRAD_DATA_DIR', ''))
ADNI_DIR = DATA_DIR / _resolve('GRAD_ADNI_DIR', 'GRAD_ADNI_DIR', 'ADNI')
A4_DIR = DATA_DIR / _resolve('GRAD_A4_DIR', 'GRAD_A4_DIR', 'A4')
SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic' / 'synthetic_cohort.csv'

# --- outputs --------------------------------------------------------------
_FIGURE_SUFFIXES = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.tif', '.tiff', '.eps'}
_TABLE_SUFFIXES = {'.csv', '.json', '.txt', '.tsv', '.docx', '.xlsx'}


class _Results(os.PathLike):
    """``results/`` with automatic routing.

    ``RESULTS / 'figure_2_roc_combined.png'`` -> ``results/figures/...``
    ``RESULTS / 'supp_table_by_stage.csv'``   -> ``results/tables/...``
    ``RESULTS / 'adni_loocv_predictions.csv'``-> ``results/`` (root)

    Routing applies to reads as well as writes, so a script that writes a
    table and a later script that reads it always agree on the location.
    """

    def __init__(self, root):
        self.root = Path(root)

    def _route(self, name):
        p = Path(str(name))
        suffix = p.suffix.lower()
        if 'prediction' in p.name.lower():
            sub = self.root
        elif suffix in _FIGURE_SUFFIXES:
            sub = self.root / 'figures'
        elif suffix in _TABLE_SUFFIXES:
            sub = self.root / 'tables'
        else:
            sub = self.root
        sub.mkdir(parents=True, exist_ok=True)
        return sub / p.name

    def __truediv__(self, name):
        return self._route(name)

    def __fspath__(self):
        return str(self.root)

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return f'_Results({self.root!r})'


RESULTS = _Results(PROJECT_ROOT / 'results')
FIGURES = PROJECT_ROOT / 'results' / 'figures'
TABLES = PROJECT_ROOT / 'results' / 'tables'
PANELS = PROJECT_ROOT / 'results' / 'figures' / 'panels'

for _d in (FIGURES, TABLES, PANELS):
    _d.mkdir(parents=True, exist_ok=True)


def load_threshold_sweep():
    """
    Gatekeeper threshold sweep, with legacy column names.

    Replaces the retired supp_table_threshold_sensitivity.csv, which applied the
    probability bands to the FINAL two-stage score rather than to the Gatekeeper
    probability -- so it described post-hoc banding of the output, not the
    routing thresholds it was labelled with, and its 0.25/0.75 row contradicted
    the main text. Regenerate with scripts/GRAD_threshold_sweep.py.

    Columns are renamed to the legacy names so existing figure code is unchanged.
    """
    import pandas as pd
    path = TABLES / 'supp_table_gatekeeper_threshold_sweep.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found. Run: python3 scripts/GRAD_threshold_sweep.py')
    return pd.read_csv(path).rename(columns={
        'low': 'Low_Threshold',
        'high': 'High_Threshold',
        'resolution_rate': 'Resolution_Rate',
        'resolved_accuracy': 'Resolved_Accuracy',
        'gray_zone_n': 'Gray_Zone_N',
        'gray_zone_auc': 'Gray_Zone_AUC',
        'overall_auc': 'Overall_AUC',
    })


def require_data():
    """Raise a clear error if the restricted-access data is not reachable."""
    missing = [str(p) for p in (ADNI_DIR, A4_DIR) if not p.is_dir()]
    if missing:
        raise SystemExit(
            'GRAD data not found:\n  ' + '\n  '.join(missing) +
            '\n\nSet GRAD_DATA_DIR (and optionally GRAD_ADNI_DIR / '
            'GRAD_A4_DIR) to point at the two cohort directories, or create '
            'scripts/_local_paths.py. See data/README_data.md.'
        )
