"""Generate submission-ready table files for ART upload."""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os
from _grad_paths import (RESULTS, ADNI_DIR, A4_DIR, DATA_DIR, PROJECT_ROOT, SYNTHETIC)  # noqa: F401

OUTPUT_DIR = str(RESULTS / 'tables')


def set_cell_shading(cell, color_hex):
    shading = cell._element.get_or_add_tcPr()
    shading_elem = shading.makeelement(qn('w:shd'), {
        qn('w:fill'): color_hex, qn('w:val'): 'clear'})
    shading.append(shading_elem)


def style_header_row(table, n_cols):
    for j in range(n_cols):
        cell = table.rows[0].cells[j]
        set_cell_shading(cell, '2166AC')
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(10)
                run.font.name = 'Arial'


def style_data_rows(table, n_rows, n_cols):
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table.rows[i].cells[j]
            if i % 2 == 0:
                set_cell_shading(cell, 'F5F5F5')
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'
        # Left-align first column
        cell0 = table.rows[i].cells[0]
        for p in cell0.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def create_table1():
    """Table 1: Baseline Participant Characteristics"""
    doc = Document()
    for section in doc.sections:
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    p = doc.add_paragraph()
    run = p.add_run('Table 1.')
    run.bold = True
    run.font.size = Pt(11)
    run = p.add_run(' Baseline Participant Characteristics')
    run.font.size = Pt(11)
    p.space_after = Pt(8)

    data = [
        ['Characteristics', 'ADNI (N=320)', 'A4 (N=1,644)'],
        ['Age, years (mean \u00b1 SD)', '72.5 \u00b1 6.8', '71.8 \u00b1 4.7'],
        ['Female, n (%)', '152 (47.5%)', '953 (58.0%)'],
        ['Education, years (mean \u00b1 SD)', '16.3 \u00b1 2.6', '16.8 \u00b1 2.4'],
        ['White race, n (%)', '285 (89.1%)', '1,479 (89.9%)'],
        ['APOE \u03b54 carrier, n (%)', '113 (35.3%)', '598 (36.4%)'],
        ['A\u03b2 positive, n (%)', '155 (48.4%)', '1,145 (69.6%)'],
        ['p-Tau217, pg/mL (median [IQR])', '0.110 [0.064\u20130.228]', '0.152 [0.098\u20130.234]'],
        ['Cognitive status: CN / MCI / AD', '175 / 117 / 28', '1,644 / 0 / 0'],
    ]

    table = doc.add_table(rows=len(data), cols=3)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            cell = table.rows[i].cells[j]
            cell.text = val
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'

    style_header_row(table, 3)
    style_data_rows(table, len(data) - 1, 3)

    # Left-align header first col too
    cell = table.rows[0].cells[0]
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    path = str(RESULTS / 'Table_1_Baseline_Characteristics.docx')
    doc.save(path)
    print(f'Saved: {path}')


def create_table2():
    """Table 2: Complete Model Performance Metrics"""
    doc = Document()
    for section in doc.sections:
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    p = doc.add_paragraph()
    run = p.add_run('Table 2.')
    run.bold = True
    run.font.size = Pt(11)
    run = p.add_run(' Complete Model Performance Metrics')
    run.font.size = Pt(11)
    p.space_after = Pt(8)

    data = [
        ['Metric', 'Value', '95% CI'],
        ['AUC', '0.857', '[0.813\u20130.897]'],
        ['AUPRC', '0.827', '[0.757\u20130.897]'],
        ['Accuracy', '80.6%', '[76.2%\u201384.7%]'],
        ['Sensitivity', '78.7%', '[72.3%\u201384.8%]'],
        ['Specificity', '82.4%', '[76.4%\u201388.0%]'],
        ['PPV', '80.8%', '[74.5%\u201386.7%]'],
        ['NPV', '80.5%', '[74.6%\u201386.2%]'],
        ['LR+', '4.48', '[3.28\u20136.50]'],
        ['LR\u2212', '0.26', '[0.18\u20130.35]'],
        ['Brier Score', '0.148', '\u2014'],
    ]

    table = doc.add_table(rows=len(data), cols=3)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            cell = table.rows[i].cells[j]
            cell.text = val

    style_header_row(table, 3)
    style_data_rows(table, len(data) - 1, 3)

    path = str(RESULTS / 'Table_2_Model_Performance.docx')
    doc.save(path)
    print(f'Saved: {path}')


def create_table3():
    """Table 3: Cost-Impact Simulation"""
    doc = Document()
    for section in doc.sections:
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    p = doc.add_paragraph()
    run = p.add_run('Table 3.')
    run.bold = True
    run.font.size = Pt(11)
    run = p.add_run(' Cost-Impact Simulation by Diagnostic Method')
    run.font.size = Pt(11)
    p.space_after = Pt(8)

    data = [
        ['Strategy', 'Total Cost', 'Per Capita', 'PET Scans', 'Savings vs. PET'],
        ['Universal PET', '$30,000,000', '$3,000', '10,000', 'Reference'],
        ['p-Tau217 + PET (Gray Zone)', '$16,820,000', '$1,682', '4,440', '44%'],
        ['Staged Algorithm (GRAD)', '$9,993,000', '$999', '1,331', '67%'],
        ['Staged Algorithm + MRI', '$8,661,000', '$866', '887', '71%'],
    ]

    table = doc.add_table(rows=len(data), cols=5)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            cell = table.rows[i].cells[j]
            cell.text = val

    style_header_row(table, 5)
    style_data_rows(table, len(data) - 1, 5)

    path = str(RESULTS / 'Table_3_Cost_Simulation.docx')
    doc.save(path)
    print(f'Saved: {path}')


if __name__ == '__main__':
    create_table1()
    create_table2()
    create_table3()
