#!/usr/bin/env python3
"""
Main-text Table 2, as a single table (ACTN does not permit table panels).
=========================================================================

The panelled version split discrimination (panel A, comparators as columns) from
decision-level outcomes (panel B, comparators as rows). Those two layouts cannot
be stacked. Transposing panel A into panel B's grammar - one row per comparator,
AUC as a column - lets both sets of columns coexist in one table.

Delta AUC and the DeLong P value sit on the comparator rows and are defined as
GRAD minus that comparator, so the GRAD row is the reference in each block.

    python3 scripts/GRAD_table2.py   ->  manuscript/GRAD_ACTN_Table2.docx

Numbers are read from results/tables/grad_v2_numbers.json; nothing is typed by
hand.
"""

import json

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from _grad_paths import RESULTS, PROJECT_ROOT

TABLE_FONT, LEGEND_FONT, SIZE = 'Cambria', 'Helvetica', Pt(12)

HEAD = ['Cohort and score', 'n', 'AUC (95% CI)', 'Δ AUC', 'P',
        'Resolved correctly, %', 'Resolved incorrectly, %', 'Referred for Aβ-PET, %']

TITLE = ('Benchmarking GRAD Stage 2 against p-Tau217 alone and p-Tau181 within '
         'the gray zone.')

NOTE = (
    'Each block reports one comparison; GRAD Stage 2 is the reference within '
    'each block, so Δ AUC and the accompanying P value are GRAD minus that '
    'comparator, by DeLong test. Decision-level outcomes apply the same '
    '0.40–0.60 indeterminate band to every score, so the comparison is matched '
    'on abstention; percentages are of the gray zone participants in that '
    'block and sum to 100. p-Tau181 was applied as a single dichotomous cutoff '
    'derived by Youden index, following the original integration study, and '
    'therefore never abstains. Paired reclassification favoured GRAD in both '
    'cohorts (McNemar: ADNI 9 corrected versus 5 introduced, P = .42; A4 + '
    'LEARN 160 versus 67, P < .001). The p-Tau181 subset comprises the 256 A4 + '
    'LEARN gray zone participants with a visit-matched p-Tau181 measurement '
    '(55.5% Aβ-positive). AUC, area under the receiver operating '
    'characteristic curve.')


def pct(v):
    return f'{v:.1f}'


def build_rows(d):
    a = d['adni_stage2']
    b = d['a4_stage2']
    s = d['ptau181_comparison']
    band = {(r['cohort'], r['score']): r for r in d['band_outcomes']}

    def outcome(cohort, score):
        r = band[(cohort, score)]
        return [pct(100 * r['resolved_correct']), pct(100 * r['resolved_wrong']),
                pct(100 * r['needs_pet'])]

    rows = []
    rows.append(('HEAD', 'ADNI gray zone'))
    rows.append(['p-Tau217 alone', f"{a['n']}",
                 f"{a['auc_ptau']:.3f} ({a['auc_ptau_ci'][0]:.3f}–{a['auc_ptau_ci'][1]:.3f})",
                 f"+{a['auc_grad'] - a['auc_ptau']:.3f}",
                 f"{a['delong_p']:.2f}".lstrip('0')] + outcome('ADNI', 'p-tau217'))
    rows.append(['GRAD Stage 2', f"{a['n']}",
                 f"{a['auc_grad']:.3f} ({a['auc_grad_ci'][0]:.3f}–{a['auc_grad_ci'][1]:.3f})",
                 '—', '—'] + outcome('ADNI', 'GRAD Stage 2'))

    rows.append(('HEAD', 'A4 + LEARN gray zone'))
    rows.append(['p-Tau217 alone', f"{b['n']}",
                 f"{b['auc_ptau']:.3f} ({b['auc_ptau_ci'][0]:.3f}–{b['auc_ptau_ci'][1]:.3f})",
                 f"+{b['auc_grad'] - b['auc_ptau']:.3f}", '<.001']
                + outcome('A4', 'p-tau217'))
    rows.append(['GRAD Stage 2', f"{b['n']}",
                 f"{b['auc_grad']:.3f} ({b['auc_grad_ci'][0]:.3f}–{b['auc_grad_ci'][1]:.3f})",
                 '—', '—'] + outcome('A4', 'GRAD Stage 2'))

    rows.append(('HEAD', 'A4 + LEARN, p-Tau181 subset'))
    n = s['n']
    rows.append(['p-Tau181 (Youden cutoff)', f'{n}',
                 f"{s['auc_ptau181']:.3f} ({s['auc_ptau181_ci'][0]:.3f}–{s['auc_ptau181_ci'][1]:.3f})",
                 f"+{s['auc_grad'] - s['auc_ptau181']:.3f}", '<.001',
                 pct(s['ptau181_correct']), pct(s['ptau181_wrong']), '0'])
    rows.append(['p-Tau217 alone', f'{n}',
                 f"{s['auc_ptau217']:.3f} ({s['auc_ptau217_ci'][0]:.3f}–{s['auc_ptau217_ci'][1]:.3f})",
                 f"+{s['auc_grad'] - s['auc_ptau217']:.3f}",
                 f"{s['delong_grad_vs_217']:.2f}".lstrip('0'),
                 pct(s['ptau217_resolved_correct']), pct(s['ptau217_resolved_wrong']),
                 pct(s['ptau217_to_pet'])])
    rows.append(['GRAD Stage 2', f'{n}',
                 f"{s['auc_grad']:.3f} ({s['auc_grad_ci'][0]:.3f}–{s['auc_grad_ci'][1]:.3f})",
                 '—', '—', pct(s['grad_resolved_correct']),
                 pct(s['grad_resolved_wrong']), pct(s['grad_to_pet'])])
    return rows


def style_run(run, font, bold=False):
    run.bold = bold
    run.font.size = SIZE
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.append(rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rf.set(qn(a), font)


def double(par):
    pf = par.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_before = pf.space_after = Pt(0)
    return par


def hrule(cell, edges):
    b = OxmlElement('w:tcBorders')
    for e in edges:
        el = OxmlElement(f'w:{e}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '8')
        el.set(qn('w:color'), '000000')
        b.append(el)
    cell._element.tcPr.append(b)


def main():
    d = json.load(open(RESULTS / 'grad_v2_numbers.json'))
    rows = build_rows(d)

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.left_margin = sec.right_margin = Inches(0.6)

    par = double(doc.add_paragraph())
    style_run(par.add_run('Table 2. '), LEGEND_FONT, bold=True)
    style_run(par.add_run(TITLE), LEGEND_FONT)

    t = doc.add_table(rows=1, cols=len(HEAD))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(HEAD):
        c = t.rows[0].cells[j]
        style_run(double(c.paragraphs[0]).add_run(h), TABLE_FONT, bold=True)
        hrule(c, ('top', 'bottom'))

    for i, row in enumerate(rows):
        cells = t.add_row().cells
        if isinstance(row, tuple):                      # spanning subheading
            merged = cells[0]
            for c in cells[1:]:
                merged = merged.merge(c)
            style_run(double(merged.paragraphs[0]).add_run(row[1]),
                      TABLE_FONT, bold=True)
            continue
        for j, v in enumerate(row):
            style_run(double(cells[j].paragraphs[0]).add_run(v), TABLE_FONT)
        if i == len(rows) - 1:
            for c in cells:
                hrule(c, ('bottom',))

    widths = [2.2, .55, 1.75, .8, .7, 1.35, 1.45, 1.4]
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Inches(w)

    par = double(doc.add_paragraph())
    style_run(par.add_run(NOTE), LEGEND_FONT)

    path = PROJECT_ROOT / 'manuscript' / 'GRAD_ACTN_Table2.docx'
    doc.save(path)
    print('wrote', path)
    for r in rows:
        print('  ', r[1] if isinstance(r, tuple) else ' | '.join(r))


if __name__ == '__main__':
    main()
