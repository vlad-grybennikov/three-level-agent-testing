"""Regenerate the per-task result matrices (paper Fig. 3 panels)
from the exported task-matrix CSVs.

Usage:
    python regen_matrices.py task_matrix-qwen.csv matrix_qwen.png
    python regen_matrices.py task_matrix-opus.csv matrix_opus.png

Column *display labels* follow the paper's Table 2 / tab:live naming
(S3a renames: "slot match", "recall@5"). CSV column keys are unchanged.
Same palette and layout as export_assets.py figures, no title band:
model attribution lives in the LaTeX caption (fig:matrices).
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Column layout mirroring the original figures, minus the title band.
GROUPS = [
    ("runs",    [("L1",        "L1 runs"),   ("judge",     "judge runs")]),
    ("Level 1", [("pass^1",    "pass$^1$"),  ("pass^3",    "pass$^3$")]),
    ("Level 2", [("node_f1",   "node F1"),   ("edge_f1",   "edge F1"),
                 ("order",     "order"),     ("redundancy","redundancy"),
                 ("policies",  "P asserts")]),
    ("Level 3", [("intent",    "intent"),    ("slots",     "slot match"),
                 ("recall@k",  "recall@5"),  ("binding",   "binding")]),
]

GREEN, RED = "#1e8a4c", "#c0392b"

def cell_value_color(col, raw):
    """Return (display_text, color, bold) for a cell."""
    if col in ("L1", "judge"):                      # fraction like 0/3
        num = int(raw.split("/")[0])
        frac = num / 3.0
        return raw, frac_color(frac), num < 3
    if col == "intent":                             # 'ok' flag
        return raw, GREEN if raw == "ok" else RED, raw != "ok"
    v = float(raw)
    if col == "redundancy":                          # 0 is ideal
        return fmt(v), GREEN if v == 0 else frac_color(1 - v), v > 0
    if col == "policies":                            # binary clean flag
        return fmt(v), GREEN if v == 1 else RED, v < 1
    return fmt(v), frac_color(v), v < 1

def frac_color(v):
    """0 -> red, 0.5 -> orange, 1 -> green (matching original palette)."""
    stops = [(0.0, (0.753, 0.224, 0.169)),
             (0.5, (0.902, 0.494, 0.133)),
             (0.75, (0.62, 0.55, 0.16)),
             (1.0, (0.118, 0.541, 0.298))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = 0 if b == a else (v - a) / (b - a)
            return tuple(x + t * (y - x) for x, y in zip(ca, cb))
    return stops[-1][1]

def fmt(v):
    if v in (0.0, 1.0):
        return str(int(v))
    return f"{v:.4g}"

def render(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path)))
    cols = [(c, lbl) for _, group in GROUPS for c, lbl in group]
    n_r, n_c = len(rows), len(cols)

    fig, ax = plt.subplots(figsize=(0.98 * n_c + 3.4, 0.62 * n_r + 1.1))
    ax.set_xlim(0, n_c); ax.set_ylim(0, n_r)
    ax.axis("off")

    for i, row in enumerate(rows):
        y = n_r - 1 - i
        ax.text(-0.15, y + 0.5, row["task"], ha="right", va="center",
                fontsize=10.5, family="monospace")
        for j, (col, _) in enumerate(cols):
            text, color, bold = cell_value_color(col, row[col])
            ax.add_patch(plt.Rectangle((j + 0.03, y + 0.04), 0.94, 0.92,
                                       color=color, ec="white", lw=1.5))
            ax.text(j + 0.5, y + 0.5, text, ha="center", va="center",
                    fontsize=10.5, color="white",
                    fontweight="bold" if bold else "normal")

    # column labels
    for j, (_, lbl) in enumerate(cols):
        ax.text(j + 0.5, n_r + 0.12, lbl, ha="center", va="bottom",
                fontsize=11)
    # group headers + separators
    x = 0
    for name, group in GROUPS:
        w = len(group)
        ax.text(x + w / 2, n_r + 0.55, name, ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")
        if x > 0:
            ax.plot([x, x], [0, n_r], color="black", lw=1.6,
                    solid_capstyle="butt", zorder=5)
        x += w

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        render(sys.argv[1], sys.argv[2])
    else:
        render("exports/task_matrix-qwen.csv", "media/matrix_qwen.png")
        render("exports/task_matrix-opus.csv", "media/matrix_opus.png")
