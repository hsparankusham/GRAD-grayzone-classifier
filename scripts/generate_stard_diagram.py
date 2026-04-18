#!/usr/bin/env python3
"""
STARD 2015 Flow Diagram for GRAD Paper
========================================
Generates a participant flow diagram compliant with STARD 2015
reporting guidelines for diagnostic accuracy studies.

Output: results/figure_s2_stard_flow.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path(__file__).parent / 'results'


def draw_box(ax, x, y, text, width=2.4, height=0.55, color='#E3F2FD',
             edgecolor='#1565C0', fontsize=8.5):
    """Draw a rounded box with centered text."""
    box = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.08", facecolor=color, edgecolor=edgecolor,
        linewidth=1.2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='normal', wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#424242'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')

    # Title
    ax.text(5, 10.1, 'STARD 2015 Flow Diagram — GRAD Algorithm',
            ha='center', fontsize=14, fontweight='bold')

    # ── ADNI ARM (left) ──
    ax.text(2.5, 9.6, 'Development Cohort (ADNI)', ha='center',
            fontsize=11, fontweight='bold', color='#1565C0')

    draw_box(ax, 2.5, 8.8, 'ADNI participants with\nplasma biomarkers & amyloid PET\n(N=320)')
    draw_arrow(ax, 2.5, 8.5, 2.5, 7.85)

    draw_box(ax, 2.5, 7.55, 'LOOCV: Stage 1 Gatekeeper\n(univariate p-tau217)')
    draw_arrow(ax, 2.5, 7.25, 2.5, 6.65)

    # Split into three
    draw_box(ax, 0.5, 6.2, 'Amyloid Negative\nP < 0.25\nn=100\n(NPV 90.0%)',
             width=2.0, height=0.75, color='#C8E6C9', edgecolor='#2E7D32')
    draw_box(ax, 2.5, 6.2, 'Gray Zone\n0.25 ≤ P ≤ 0.75\nn=142 (44.4%)',
             width=2.0, height=0.75, color='#FFF9C4', edgecolor='#F9A825')
    draw_box(ax, 4.5, 6.2, 'Amyloid Positive\nP > 0.75\nn=78\n(PPV 87.2%)',
             width=2.0, height=0.75, color='#FFCDD2', edgecolor='#C62828')

    draw_arrow(ax, 1.5, 7.25, 0.5, 6.6)
    draw_arrow(ax, 2.5, 7.25, 2.5, 6.6)
    draw_arrow(ax, 3.5, 7.25, 4.5, 6.6)

    # Gray zone to Reflex
    draw_arrow(ax, 2.5, 5.8, 2.5, 5.2)
    draw_box(ax, 2.5, 4.85, 'Stage 2 Reflex\n(6-feature Random Forest)\nn=142')
    draw_arrow(ax, 2.5, 4.55, 2.5, 3.95)

    # Reflex outputs
    draw_box(ax, 1.2, 3.55, 'Classified\nNegative/Positive\nn=109 (70.4% acc)',
             width=2.0, height=0.65, color='#E1F5FE', edgecolor='#0277BD')
    draw_box(ax, 3.8, 3.55, 'Indeterminate\n(0.40–0.60)\nRefer to PET\n~10%',
             width=2.0, height=0.65, color='#F3E5F5', edgecolor='#7B1FA2')
    draw_arrow(ax, 1.8, 3.95, 1.2, 3.9)
    draw_arrow(ax, 3.2, 3.95, 3.8, 3.9)

    # Overall ADNI result
    draw_box(ax, 2.5, 2.4, 'Overall ADNI Pipeline\nAUC 0.857 (95% CI: 0.813–0.897)\nAccuracy 80.6%',
             width=3.2, height=0.65, color='#BBDEFB', edgecolor='#1565C0', fontsize=9)

    draw_arrow(ax, 2.5, 3.2, 2.5, 2.75)

    # ── A4 ARM (right) ──
    ax.text(7.5, 9.6, 'External Validation (A4 + LEARN)', ha='center',
            fontsize=11, fontweight='bold', color='#C62828')

    draw_box(ax, 7.5, 8.8, 'A4 treatment arm (n=1,145; amyloid+)\n+ LEARN arm (n=499; amyloid−)\nTotal N=1,644')
    draw_arrow(ax, 7.5, 8.5, 7.5, 7.85)

    draw_box(ax, 7.5, 7.55, 'ADNI-trained Gatekeeper\napplied to A4')
    draw_arrow(ax, 7.5, 7.25, 7.5, 6.65)

    # A4 splits
    draw_box(ax, 5.7, 6.2, 'Negative\nn=2',
             width=1.5, height=0.55, color='#C8E6C9', edgecolor='#2E7D32')
    draw_box(ax, 7.5, 6.2, 'Gray Zone\nn=1,293\n(78.6%)',
             width=1.8, height=0.55, color='#FFF9C4', edgecolor='#F9A825')
    draw_box(ax, 9.3, 6.2, 'Positive\nn=349',
             width=1.5, height=0.55, color='#FFCDD2', edgecolor='#C62828')

    draw_arrow(ax, 6.5, 7.25, 5.7, 6.5)
    draw_arrow(ax, 7.5, 7.25, 7.5, 6.5)
    draw_arrow(ax, 8.5, 7.25, 9.3, 6.5)

    # A4 Reflex
    draw_arrow(ax, 7.5, 5.9, 7.5, 5.2)
    draw_box(ax, 7.5, 4.85, 'ADNI-trained Reflex\napplied to A4 gray zone\nn=1,293')
    draw_arrow(ax, 7.5, 4.55, 7.5, 3.95)

    # MRI subset
    draw_box(ax, 7.5, 3.55, 'MRI subset (n=1,044)\nPlasma+MRI AUC: 0.853\nΔAUC: +0.025 (p=0.014)',
             width=2.8, height=0.65, color='#FFF3E0', edgecolor='#E65100')

    # Overall A4 result
    draw_box(ax, 7.5, 2.4, 'Overall A4 Pipeline\nAUC 0.828 (95% CI: 0.806–0.849)\nCentiloid r=0.642',
             width=3.2, height=0.65, color='#FFCDD2', edgecolor='#C62828', fontsize=9)

    draw_arrow(ax, 7.5, 3.2, 7.5, 2.75)

    # Divider
    ax.plot([5, 5], [1.5, 9.5], 'k--', alpha=0.2, linewidth=1)

    # Reference
    ax.text(5, 1.2, 'Reporting follows STARD 2015 guidelines [Bossuyt et al., BMJ 2015]',
            ha='center', fontsize=8, style='italic', alpha=0.6)

    plt.tight_layout()
    fig.savefig(RESULTS / 'figure_s2_stard_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figure_s2_stard_flow.png")


if __name__ == '__main__':
    main()
