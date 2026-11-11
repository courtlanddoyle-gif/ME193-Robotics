"""Builds V_data_gradient_descent.xlsx: 5 iterations of gradient descent
on m and b for one neuron with a ReLU.  One sheet per iteration."""

import shutil
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, PatternFill

HERE = Path(__file__).parent
if not (HERE / "annacode-v.tsv").exists():
    shutil.copy(HERE.parent / "v-data" / "annacode-v.tsv", HERE / "annacode-v.tsv")
x, y = np.loadtxt(HERE / "annacode-v.tsv", skiprows=1, usecols=(1, 2)).T
N = len(x)
FIRST, LAST = 8, 8 + N - 1   # data rows on each iteration sheet

bold = Font(bold=True)
yellow = PatternFill("solid", fgColor="FFF2CC")
blue = PatternFill("solid", fgColor="DDEBF7")

wb = Workbook()
summary = wb.active
summary.title = "Summary"
summary["A1"] = "Gradient descent on one neuron + ReLU (Anna's V data)"
summary["A1"].font = Font(bold=True, size=14)
summary["A3"], summary["B3"] = "learning rate α", 0.2
summary["A4"], summary["B4"] = "starting m", 1
summary["A5"], summary["B5"] = "starting b", 1
for c in ("B3", "B4", "B5"):
    summary[c].fill = yellow
summary["A6"] = "(change the yellow cells and everything updates)"

for j, h in enumerate(["iteration", "m", "b", "loss"]):
    summary.cell(8, 1 + j, h).font = bold

for k in range(1, 6):
    ws = wb.create_sheet(f"Iteration {k}")
    prev = "Summary" if k == 1 else f"'Iteration {k-1}'"
    ws["A1"] = f"Iteration {k}"
    ws["A1"].font = Font(bold=True, size=14)

    ws["A2"], ws["B2"] = "m", "=Summary!B4" if k == 1 else f"={prev}!B5"
    ws["A3"], ws["B3"] = "b", "=Summary!B5" if k == 1 else f"={prev}!B6"
    ws["D2"], ws["E2"] = "loss = average of error²", f"=SUMSQ(G{FIRST}:G{LAST})/{N}"
    ws["D3"], ws["E3"] = "∂L/∂m = average of column H", f"=AVERAGE(H{FIRST}:H{LAST})"
    ws["D4"], ws["E4"] = "∂L/∂b = average of column I", f"=AVERAGE(I{FIRST}:I{LAST})"
    ws["A5"], ws["B5"] = "new m", "=B2-Summary!$B$3*E3"
    ws["A6"], ws["B6"] = "new b", "=B3-Summary!$B$3*E4"
    for c in ("A2", "A3", "A5", "A6", "D2", "D3", "D4"):
        ws[c].font = bold
    for c in ("B5", "B6"):
        ws[c].fill = blue

    heads = ["x", "y", "x scaled\n(x−60)/20", "z = m·xs + b", "a = ReLU(z)",
             "ReLU'(z)", "error = y − a", "−2·error·ReLU'·xs", "−2·error·ReLU'"]
    for j, h in enumerate(heads):
        ws.cell(FIRST - 1, 1 + j, h).font = bold
        ws.column_dimensions["ABCDEFGHI"[j]].width = 16
    ws.column_dimensions["D"].width = 28

    for i in range(N):
        r = FIRST + i
        ws[f"A{r}"] = float(x[i])
        ws[f"B{r}"] = float(y[i])
        ws[f"C{r}"] = f"=(A{r}-60)/20"
        ws[f"D{r}"] = f"=$B$2*C{r}+$B$3"
        ws[f"E{r}"] = f"=MAX(0,D{r})"
        ws[f"F{r}"] = f"=IF(D{r}>=0,1,0)"
        ws[f"G{r}"] = f"=B{r}-E{r}"
        ws[f"H{r}"] = f"=-2*G{r}*F{r}*C{r}"
        ws[f"I{r}"] = f"=-2*G{r}*F{r}"
    ws.freeze_panes = f"A{FIRST}"

    s = f"'Iteration {k}'"
    summary.cell(8 + k, 1, k)
    summary.cell(8 + k, 2, f"={s}!B2")
    summary.cell(8 + k, 3, f"={s}!B3")
    summary.cell(8 + k, 4, f"={s}!E2")

summary.cell(14, 1, "after 5")
summary.cell(14, 2, "='Iteration 5'!B5")
summary.cell(14, 3, "='Iteration 5'!B6")
summary.column_dimensions["A"].width = 18

chart = LineChart()
chart.title = "Loss goes down each iteration"
chart.x_axis.title = "iteration"
chart.y_axis.title = "loss"
chart.add_data(Reference(summary, min_col=4, min_row=8, max_row=13), titles_from_data=True)
chart.set_categories(Reference(summary, min_col=1, min_row=9, max_row=13))
chart.legend = None
chart.x_axis.delete = False
chart.y_axis.delete = False
summary.add_chart(chart, "F3")

wb.save(HERE / "V_data_gradient_descent.xlsx")
print("saved V_data_gradient_descent.xlsx")
