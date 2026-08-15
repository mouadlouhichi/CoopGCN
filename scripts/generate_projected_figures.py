#!/usr/bin/env python3
"""Render explicitly projected, unmeasured outcome scenarios from expected CSVs."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"CoopGCN (Ours)": "#1769aa", "Full CoopGCN (Ours)": "#1769aa"}
BANNER = "EXPECTED OUTCOME SCENARIO — PROJECTED AND UNMEASURED"


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finish(fig, path):
    fig.text(.5, .012, BANNER, ha="center", color="#8b1a1a", weight="bold", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(path, dpi=220)
    plt.close(fig)

# Cross-dataset expected NDCG/Recall scenario.
r = rows(ROOT / "data/expected_results.csv")
datasets = ["ML-1M", "Yelp2018", "Amazon-Book"]
models = ["LightGCN", "LightGCN++", "DyHuCoG", "CoopGCN (Ours)"]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
for ax, metric, title in zip(axes, ["ndcg_20", "recall_20"], ["NDCG@20", "Recall@20"]):
    width = .19
    for j, model in enumerate(models):
        vals = [float(next(x[metric] for x in r if x["dataset"] == ds and x["model"] == model)) for ds in datasets]
        xpos = [i + (j - 1.5) * width for i in range(len(datasets))]
        ax.bar(xpos, vals, width, label=model, color=COLORS.get(model))
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_title(title)
    ax.grid(axis="y", alpha=.25)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("Expected cross-dataset benchmark profile", weight="bold")
finish(fig, OUT / "Figure_expected_benchmark.png")

# Expected component profile.
r = rows(ROOT / "data/expected_ablation.csv")
variants = [x["variant"] for x in r]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
for ax, metric, title in zip(axes, ["ndcg_20", "tr_20", "coverage_20"], ["NDCG@20", "Tail Recall@20", "Coverage@20"]):
    vals = [float(x[metric]) for x in r]
    colors = [COLORS.get(v, "#9a9a9a") for v in variants]
    ax.bar(range(len(variants)), vals, color=colors)
    ax.set_xticks(range(len(variants)), variants, rotation=45, ha="right", fontsize=6.5)
    ax.set_title(title)
    ax.grid(axis="y", alpha=.25)
fig.suptitle("Expected component profile on ML-1M", weight="bold")
finish(fig, OUT / "Figure_expected_ablation.png")

# Expected corruption profile, two datasets and selected methods.
r = rows(ROOT / "data/expected_noise.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
for ax, ds in zip(axes, ["ML-1M", "Yelp2018"]):
    dr = [x for x in r if x["dataset"] == ds]
    x = [100 * float(z["noise_ratio"]) for z in dr]
    for model in ["LightGCN", "LightGCN++", "DyHuCoG", "CoopGCN (Ours)"]:
        ax.plot(x, [float(z[model]) for z in dr], marker="o", label=model,
                linewidth=2 if model == "CoopGCN (Ours)" else 1.4,
                color=COLORS.get(model))
    ax.set_title(ds)
    ax.set_xlabel("Injected edge ratio (%)")
    ax.set_ylabel("NDCG@20")
    ax.grid(alpha=.25)
axes[0].legend(fontsize=7)
fig.suptitle("Expected corruption-response profile", weight="bold")
finish(fig, OUT / "Figure_expected_corruption.png")
print("generated projected figures in", OUT)
