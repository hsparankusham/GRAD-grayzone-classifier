"""
Generate publication-quality Figure 6 for GRAD manuscript.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

COLORS = {
    'pet_0':    '#C4626A',   # Universal PET — richer muted rose/burgundy
    'plasma_0': '#C4626A',

    'pet_1':    '#BF8F00',
    'plasma_1': '#E8D9A8',

    'pet_2':    '#2166AC',
    'plasma_2': '#92C5DE',

    'pet_3':    '#1B7837',
    'plasma_3': '#7FBF7B',

    'text':     '#1a1a1a',
    'subtext':  '#777777',
    'errbar':   '#333333',
    'grid':     '#E5E5E5',
    'navy':     '#0D1B4A',
    'spine':    '#444444',
}

# ── Data ──────────────────────────────────────────────────────────────
strategies = ['Universal\nPET', 'p-Tau217 +\nPET (GZ)', 'GRAD\nStaged', 'GRAD +\nMRI']

plasma_costs = [0.0, 3.5, 6.0, 6.0]
pet_costs    = [30.0, 13.32, 3.993, 2.661]
total_costs  = [p + t for p, t in zip(plasma_costs, pet_costs)]

pet_scans    = [10000, 4440, 1331, 887]
per_capita   = ['$3,000', '$1,682', '$999', '$866']
savings_pct  = [0, 43.9, 66.7, 71.1]

ci_lo = [30.0, 15.2, 8.8, 7.6]
ci_hi = [30.0, 18.4, 11.2, 9.8]
err_lo = [t - lo for t, lo in zip(total_costs, ci_lo)]
err_hi = [hi - t for t, hi in zip(total_costs, ci_hi)]
err_lo[0] = 0
err_hi[0] = 0

pet_colors    = [COLORS[f'pet_{i}'] for i in range(4)]
plasma_colors = [COLORS[f'plasma_{i}'] for i in range(4)]

# ── Figure ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.0, 6.5))
fig.patch.set_facecolor('white')

x = np.arange(len(strategies))
width = 0.52

# Gridlines
ax.yaxis.grid(True, alpha=0.25, linewidth=0.4, color=COLORS['grid'], zorder=0)
ax.set_axisbelow(True)

# Stacked bars — slight offset so bottom spine shows
bar_bottom = 0.35
bars_pet = ax.bar(x, pet_costs, width, bottom=bar_bottom,
                  color=pet_colors, edgecolor='white', linewidth=1.8, zorder=3)
bars_plasma = ax.bar(x, plasma_costs, width,
                     bottom=[p + bar_bottom for p in pet_costs],
                     color=plasma_colors, edgecolor='white',
                     linewidth=1.8, zorder=3)

display_tops = [t + bar_bottom for t in total_costs]

# Error bars (skip Universal PET)
ax.errorbar(x[1:], display_tops[1:],
            yerr=[[err_lo[i] for i in range(1, 4)],
                  [err_hi[i] for i in range(1, 4)]],
            fmt='none', ecolor=COLORS['errbar'], elinewidth=1.3,
            capsize=6, capthick=1.3, zorder=5)

# ── Cost labels — tight to bars/CIs ──────────────────────────────────
for i, (xi, total, pc) in enumerate(zip(x, total_costs, per_capita)):
    if i == 0:
        ci_top = display_tops[i]
    else:
        ci_top = ci_hi[i] + bar_bottom
    ax.text(xi, ci_top + 2.0,
            f'${total:.1f}M', ha='center', fontsize=10.5, fontweight='normal',
            color=COLORS['text'])
    ax.text(xi, ci_top + 0.8,
            f'{pc}/patient', ha='center', fontsize=7,
            color=COLORS['subtext'])

# ── X-axis labels ─────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=10.5, fontweight='bold',
                   linespacing=1.15)
ax.tick_params(axis='x', pad=8)

# ── n = counts directly below x-axis labels ───────────────────────────
for xi, n in zip(x, pet_scans):
    ax.text(xi, -5.5, f'n = {n:,}', ha='center', fontsize=8,
            color=COLORS['text'], clip_on=False)

# ── X-axis title below n= statements ──────────────────────────────────
ax.text(1.5, -8.5, 'Modality', ha='center', fontsize=11.5,
        fontweight='extra bold', fontfamily='Arial',
        color=COLORS['text'], clip_on=False)

# ── Y-axis label: bold title, normal units (black, not grey) ──────────
ax.set_ylabel('')
ax.text(-0.15, 0.5, 'Projected Healthcare\nExpenditure',
        fontsize=10.5, fontweight='extra bold', fontfamily='Arial',
        color=COLORS['text'],
        ha='center', va='center', rotation=90,
        transform=ax.transAxes)
ax.text(-0.09, 0.5, '($ USD, Millions)',
        fontsize=9, fontweight='normal', fontfamily='Arial',
        color=COLORS['text'],
        ha='center', va='center', rotation=90,
        transform=ax.transAxes)

ax.set_ylim(0, 42)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0fM'))
ax.tick_params(axis='y', pad=4)
ax.yaxis.set_label_coords(-0.08, 0.5)

# Box all spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color(COLORS['spine'])

# ── Title — BOLDED ────────────────────────────────────────────────────
ax.set_title('Cost Impact Simulation: Projected 10,000-Patient Cohort',
             fontsize=14, fontweight='extra bold', fontfamily='Arial',
             color='black', pad=18)

# ── Legend ─────────────────────────────────────────────────────────────
pet_patch = mpatches.Patch(facecolor=COLORS['pet_2'],
                           edgecolor='#CCCCCC', linewidth=0.8,
                           label='PET Cost (bottom)')
plasma_patch = mpatches.Patch(facecolor=COLORS['plasma_2'],
                              edgecolor='#CCCCCC', linewidth=0.8,
                              label='Plasma Panel Cost (top)')
ci_handle = Line2D([0], [0], color=COLORS['errbar'], linewidth=0,
                   marker='|', markersize=10, markeredgewidth=1.3,
                   label='95% CI')

legend = ax.legend(handles=[pet_patch, plasma_patch, ci_handle],
                   loc='upper left', bbox_to_anchor=(0.01, 0.98),
                   fontsize=8.5, frameon=True, framealpha=0.95,
                   edgecolor='#CCCCCC', handlelength=1.6,
                   ncol=1, borderpad=0.8)
legend.get_frame().set_linewidth(0.8)

# ── Inset: % Savings ─────────────────────────────────────────────────
fig.canvas.draw()
ax_pos = ax.get_position()

inset_w = 0.30
inset_h = 0.30
inset_right = ax_pos.x1
inset_left = inset_right - inset_w
# Top of inset title aligns with top spine of bar chart
inset_bottom = ax_pos.y0 + ax_pos.height * 0.55

ax_inset = fig.add_axes([inset_left, inset_bottom, inset_w, inset_h])
ax_inset.patch.set_facecolor('#FAFAFA')
ax_inset.patch.set_alpha(0.95)

ax_inset.plot(range(4), savings_pct, 's-', color=COLORS['navy'],
              markersize=7, markeredgecolor='white', markeredgewidth=1.3,
              linewidth=2.2, zorder=3)

# Labels — no "Ref" on first point
for i, sv in enumerate(savings_pct):
    if sv > 0:
        ax_inset.text(i, sv + 4.5, f'{sv:.0f}%', ha='center', fontsize=8,
                      fontweight='bold', color=COLORS['navy'])

ax_inset.set_xticks(range(4))
ax_inset.set_xticklabels(['Univ.\nPET', 'p-Tau\n+PET', 'GRAD', 'GRAD\n+MRI'],
                          fontsize=6.5, fontweight='bold', linespacing=1.0)
ax_inset.set_ylabel('Savings (%)', fontsize=8, fontweight='bold',
                     color=COLORS['navy'], labelpad=4)
ax_inset.set_ylim(-5, 88)
ax_inset.set_xlim(-0.4, 3.4)
ax_inset.yaxis.set_major_formatter(mticker.PercentFormatter())
ax_inset.tick_params(axis='both', labelsize=7, pad=2)
ax_inset.yaxis.grid(True, alpha=0.3, linewidth=0.3, color='#CCCCCC')
ax_inset.set_axisbelow(True)

ax_inset.set_title('Savings vs. Universal PET', fontsize=8.5,
                    fontweight='bold', color=COLORS['navy'], pad=6)

for spine in ax_inset.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color(COLORS['spine'])

fig.subplots_adjust(bottom=0.22, left=0.18)

# ── Save ──────────────────────────────────────────────────────────────
from _grad_paths import RESULTS
out = str(RESULTS / 'figure_6_cost_comparison.png').rsplit('.png', 1)[0]
fig.savefig(f'{out}.png', facecolor='white')
fig.savefig(f'{out}.pdf', facecolor='white')
plt.close(fig)
print(f'Saved: {out}.png and .pdf')
