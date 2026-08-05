"""
FROZEN figure style for the GRAD manuscript.

Every figure script imports from here. Do not override these values locally --
if a figure needs something different, change it here so all figures move
together.

    from _grad_style import apply_style, ABETA_NEG, ABETA_POS, save_panel
    apply_style()

Locked 2026-08-04.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── palette (frozen) ──────────────────────────────────────────────────────
# Any reference to individual Abeta-negative / Abeta-positive participants uses
# these two colours, in every figure, without exception.
ABETA_NEG = '#6FA3DB'   # blue
ABETA_POS = '#E8836F'   # coral
# Blue/coral rather than green/coral: red-green pairings are a common
# accessibility objection at journal review, and blue/coral stays legible under
# all three common colour-vision deficiencies as well as in greyscale.

# supporting neutrals
INK       = '#1A1A1A'   # all text
RULE      = '#3C3C42'   # axis lines
BAND      = '#EDEDED'   # gray-zone shading
GUIDE     = '#9A9A9A'   # reference lines, secondary labels
GRID      = '#ECECEC'

# All ROC curves use ONE muted blue. Panels are distinguished by their titles,
# not by hue -- three different colours for three ROCs reads as an encoding that
# does not exist.
ROC_BLUE  = '#4A7395'

# sequential accents for non-Abeta series (tertiles, cost)
NAVY, TEAL, AMBER, CRIMSON, GREEN = (
    '#2F5C8A', '#3E8E7E', '#D99A3C', '#B4444E', '#5AA469')

FONT_STACK = ['Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans']


def apply_style(base=9):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_STACK,
        'font.size': base,
        'axes.titlesize': base + 1.5,
        'axes.labelsize': base,
        'xtick.labelsize': base - .5,
        'ytick.labelsize': base - .5,
        'legend.fontsize': base - 1.5,
        'text.color': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'axes.edgecolor': RULE,
        'axes.linewidth': .7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': .7,
        'ytick.major.width': .7,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'grid.color': GRID,
        'grid.linewidth': .5,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,      # embed as TrueType so text stays editable
        'ps.fonttype': 42,
    })


def save_panel(fig, path_stem, dpi=400):
    """Write PNG + PDF with consistent margins."""
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_stem}.{ext}', dpi=dpi, bbox_inches='tight',
                    pad_inches=0.02, facecolor='white')
    plt.close(fig)
