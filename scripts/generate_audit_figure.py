#!/usr/bin/env python3
"""Generate code-level audit graphics; never plots projected model outcomes."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
cap_path = ROOT / "experiments/expected/neighbor_cap_sweep.csv"
out = ROOT / "paper/figures/Figure_audit_controls.png"

caps = []
with cap_path.open(newline="") as f:
    for row in csv.DictReader(f):
        if row["L"] == "32" and row["cap_sampling"] == "input" and row["credited_fraction"]:
            caps.append((row["dataset"], float(row["credited_fraction"].split("_")[0]) * 100,
                         "upper bound" in row["notes"] or "upper_bound" in row["credited_fraction"]))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
labels = ["G1 edge band", "G2 channel mix", "G3 weight range", "contrastive loss", "proxy loss"]
values = [3.0, 1.0, 1.0, 0.5, 1.0]
axes[0].barh(labels[::-1], values[::-1], color="#3977a9")
axes[0].set_xlabel("Nominal maximum magnitude (%)")
axes[0].set_title("Configured mechanism magnitudes")
axes[0].grid(axis="x", alpha=.25)

names = [x[0] for x in caps]
vals = [x[1] for x in caps]
colors = ["#d98c3f" if x[2] else "#4b9b68" for x in caps]
axes[1].bar(names, vals, color=colors)
axes[1].set_ylabel("Edges with non-zero G1 target (%)")
axes[1].set_title("L=32 target coverage")
axes[1].tick_params(axis="x", rotation=30)
axes[1].grid(axis="y", alpha=.25)
axes[1].set_ylim(0, 100)
fig.text(.5, .01, "CODE-LEVEL VALUES ONLY", ha="center",
         fontsize=9, color="#8b1a1a", weight="bold")
fig.tight_layout(rect=(0, .06, 1, 1))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220)
print(out)
