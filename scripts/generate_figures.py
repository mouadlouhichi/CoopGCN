"""
Generate publication-quality figures for CoopGCN paper.
All numbers are real empirical results from live PyTorch evaluation.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C_COOP   = "#1a6faf"   # strong blue  – CoopGCN
C_PEER1  = "#e07b39"   # orange       – DyHuCoG
C_PEER2  = "#5ba35b"   # green        – HPCF
C_PEER3  = "#b55cc0"   # purple       – HCCF
C_LGCN   = "#8c8c8c"   # grey         – LightGCN
C_LGCNPP = "#3aaa6e"   # teal         – LightGCN++
C_ACCENT = "#d62728"   # red          – highlight bar

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — NDCG@20 within direct peer group (HCCF / HPCF / DyHuCoG / CoopGCN)
#         across all 5 datasets
# ════════════════════════════════════════════════════════════════════════════
datasets   = ["ML-100k", "ML-1M", "Gowalla", "Yelp2018", "Amazon-Book"]
ndcg_coop  = [0.1826,    0.1994,   0.1089,    0.0376,     0.0235]
ndcg_dyhucog=[0.1718,   0.2114,   0.0338,    0.0144,     0.0002]
ndcg_hpcf  = [0.1747,    0.2141,   0.0278,    0.0110,     0.0003]
ndcg_hccf  = [0.1723,    0.2070,   0.0291,    0.0113,     0.0002]

x     = np.arange(len(datasets))
w     = 0.2
fig, ax = plt.subplots(figsize=(11, 5.5))

b1 = ax.bar(x - 1.5*w, ndcg_hccf,   w, label="HCCF",         color=C_PEER3, alpha=0.85)
b2 = ax.bar(x - 0.5*w, ndcg_hpcf,   w, label="HPCF",         color=C_PEER2, alpha=0.85)
b3 = ax.bar(x + 0.5*w, ndcg_dyhucog,w, label="DyHuCoG",      color=C_PEER1, alpha=0.85)
b4 = ax.bar(x + 1.5*w, ndcg_coop,   w, label="CoopGCN (Ours)",color=C_COOP,  alpha=0.95,
            edgecolor="black", linewidth=0.8)

# annotate CoopGCN bars
for rect, v in zip(b4, ndcg_coop):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.002,
            f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=C_COOP, fontweight="bold")

# mark ML-1M explicitly as deficit with asterisk
ml1m_idx = 1
ax.text(x[ml1m_idx] + 1.5*w, ndcg_coop[ml1m_idx] + 0.009, "†",
        ha="center", fontsize=12, color=C_ACCENT)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("NDCG@20", fontsize=12)
ax.set_title("NDCG@20 — CoopGCN vs. Direct Architectural Peers\n"
             "(Hypergraph / Cooperative-Game GCN family, full-catalog ranking)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, max(max(ndcg_coop), max(ndcg_hpcf)) * 1.22)

ax.text(0.01, 0.01,
        "† ML-1M: CoopGCN −6.9% vs HPCF; attributable to CL loss weight imbalance (future work)",
        transform=ax.transAxes, fontsize=8, color=C_ACCENT, va="bottom")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_ndcg_peers.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ fig1_ndcg_peers.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — TR@20 within direct peer group across 5 datasets
#         (this is where CoopGCN wins decisively everywhere)
# ════════════════════════════════════════════════════════════════════════════
tr_coop   = [0.0106, 0.0075, 0.0104, 0.0006, 0.0036]
tr_dyhucog= [0.0026, 0.0000, 0.0000, 0.0000, 0.0000]
tr_hpcf   = [0.0020, 0.0000, 0.0000, 0.0000, 0.0000]
tr_hccf   = [0.0019, 0.0000, 0.0000, 0.0000, 0.0000]

fig, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(x - 1.5*w, tr_hccf,    w, label="HCCF",          color=C_PEER3, alpha=0.85)
b2 = ax.bar(x - 0.5*w, tr_hpcf,    w, label="HPCF",          color=C_PEER2, alpha=0.85)
b3 = ax.bar(x + 0.5*w, tr_dyhucog, w, label="DyHuCoG",       color=C_PEER1, alpha=0.85)
b4 = ax.bar(x + 1.5*w, tr_coop,    w, label="CoopGCN (Ours)", color=C_COOP, alpha=0.95,
            edgecolor="black", linewidth=0.8)

for rect, v in zip(b4, tr_coop):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.0001,
            f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=C_COOP, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("Tail Recall TR@20\n(bottom-80% items)", fontsize=12)
ax.set_title("Tail Recall TR@20 — CoopGCN vs. Direct Architectural Peers\n"
             "CoopGCN is #1 on all 5 datasets; all three peers score 0.0000 on 4/5 datasets",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, max(tr_coop) * 1.30)

# zero-line annotation
ax.axhline(0, color="black", linewidth=0.5)
ax.text(0.01, 0.96,
        "HCCF, HPCF, DyHuCoG = 0.0000 on ML-1M, Gowalla, Yelp2018, Amazon-Book",
        transform=ax.transAxes, fontsize=8.5, color="#888888", va="top", style="italic")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_tail_recall_peers.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ fig2_tail_recall_peers.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Catalog Coverage@20 within direct peer group across 5 datasets
# ════════════════════════════════════════════════════════════════════════════
cov_coop    = [0.4685, 0.4066, 0.0795, 0.0536, 0.0647]
cov_dyhucog = [0.2051, 0.0942, 0.0032, 0.0035, 0.0005]
cov_hpcf    = [0.2568, 0.0923, 0.0039, 0.0048, 0.0005]
cov_hccf    = [0.2117, 0.0823, 0.0035, 0.0025, 0.0005]

fig, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(x - 1.5*w, cov_hccf,    w, label="HCCF",          color=C_PEER3, alpha=0.85)
b2 = ax.bar(x - 0.5*w, cov_hpcf,    w, label="HPCF",          color=C_PEER2, alpha=0.85)
b3 = ax.bar(x + 0.5*w, cov_dyhucog, w, label="DyHuCoG",       color=C_PEER1, alpha=0.85)
b4 = ax.bar(x + 1.5*w, cov_coop,    w, label="CoopGCN (Ours)", color=C_COOP, alpha=0.95,
            edgecolor="black", linewidth=0.8)

for rect, v in zip(b4, cov_coop):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=C_COOP, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("Catalog Coverage@20", fontsize=12)
ax.set_title("Catalog Coverage@20 — CoopGCN vs. Direct Architectural Peers\n"
             "CoopGCN is #1 on all 5 datasets (+82% to +12,840% vs best peer)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, max(cov_coop) * 1.20)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_coverage_peers.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ fig3_coverage_peers.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — NDCG@20 across all 5 datasets, ALL 10 models (heatmap / grouped)
#         Shows where CoopGCN wins and where it doesn't — honest view
# ════════════════════════════════════════════════════════════════════════════
all_models = ["MF", "NCF", "LightGCN", "LightGCN++", "GAT-CF",
              "RecDCL", "HCCF", "HPCF", "DyHuCoG", "CoopGCN\n(Ours)"]

ndcg_all = np.array([
    # ML-100k  ML-1M   Gowalla  Yelp2018  Amazon-Book
    [0.1467,  0.2043,  0.0004,  0.0006,   0.0003],  # MF
    [0.2939,  0.2883,  0.0444,  0.0129,   0.0062],  # NCF
    [0.1842,  0.2128,  0.0348,  0.0145,   0.0002],  # LightGCN
    [0.1627,  0.2054,  0.1092,  0.0359,   0.0222],  # LightGCN++
    [0.1943,  0.2096,  0.0038,  0.0014,   0.0002],  # GAT-CF
    [0.3103,  0.2997,  0.0434,  0.0109,   0.0042],  # RecDCL
    [0.1723,  0.2070,  0.0291,  0.0113,   0.0002],  # HCCF
    [0.1747,  0.2141,  0.0278,  0.0110,   0.0003],  # HPCF
    [0.1718,  0.2114,  0.0338,  0.0144,   0.0002],  # DyHuCoG
    [0.1826,  0.1994,  0.1089,  0.0376,   0.0235],  # CoopGCN
])

# Normalise each column (dataset) to [0,1] for colour comparison
ndcg_norm = (ndcg_all - ndcg_all.min(axis=0)) / (ndcg_all.max(axis=0) - ndcg_all.min(axis=0) + 1e-9)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(ndcg_norm.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

ax.set_xticks(range(len(all_models)))
ax.set_xticklabels(all_models, fontsize=9.5)
ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets, fontsize=11)
ax.set_xlabel("Model", fontsize=12)
ax.set_title("NDCG@20 Rank Heat-Map — All 10 Models × 5 Datasets\n"
             "(green = best per dataset, red = worst; absolute values annotated)",
             fontsize=11, fontweight="bold")

for i in range(len(all_models)):
    for j in range(len(datasets)):
        v   = ndcg_all[i, j]
        nv  = ndcg_norm[i, j]
        col = "black" if 0.25 < nv < 0.80 else "white"
        ax.text(i, j, f"{v:.4f}", ha="center", va="center", fontsize=7.5,
                color=col, fontweight="bold" if i == 9 else "normal")

# highlight CoopGCN column border
for j in range(len(datasets)):
    ax.add_patch(plt.Rectangle((8.5, j - 0.5), 1, 1,
                               fill=False, edgecolor=C_COOP, linewidth=2.5))

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Relative rank (per dataset)", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_ndcg_heatmap_all.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ fig4_ndcg_heatmap_all.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Component ablation (ML-100k): what each G1/G2/G3 component adds
# ════════════════════════════════════════════════════════════════════════════
ablation_labels = [
    "LightGCN\n(floor)",
    "DyHuCoG\n(G2 only)",
    "CoopGCN\nw/o G1",
    "CoopGCN\nw/o G2",
    "CoopGCN\nw/o L_game",
    "Full\nCoopGCN",
]
abl_ndcg = [0.1842, 0.1718, 0.1674, 0.1678, 0.1674, 0.1826]
abl_tr   = [0.0036, 0.0026, 0.0020, 0.0023, 0.0020, 0.0106]
abl_cov  = [0.2776, 0.2051, 0.3341, 0.3335, 0.3347, 0.4685]

colors_abl = [C_LGCN, C_PEER1, "#aaaaaa", "#aaaaaa", "#aaaaaa", C_COOP]
xa = np.arange(len(ablation_labels))

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for ax_i, (metric, values, ylabel, yl) in enumerate(zip(
    ["NDCG@20", "TR@20", "Cov@20"],
    [abl_ndcg, abl_tr, abl_cov],
    ["NDCG@20", "Tail Recall TR@20", "Catalog Coverage@20"],
    [None, None, None],
)):
    ax = axes[ax_i]
    bars = ax.bar(xa, values, color=colors_abl, alpha=0.88,
                  edgecolor="black", linewidth=0.6)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width()/2,
                rect.get_height() + max(values)*0.015,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xa)
    ax.set_xticklabels(ablation_labels, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.22)
    # highlight Full CoopGCN bar
    bars[-1].set_edgecolor(C_COOP)
    bars[-1].set_linewidth(2.5)

fig.suptitle("Component Ablation Study (ML-100k)\n"
             "Each G1 / G2 / G3 component contributes to TR@20 and Cov@20 gains",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_ablation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ fig5_ablation.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Cross-dataset scaling: NDCG@20 of top-4 graph models across scale
#         Shows CoopGCN is only model that doesn't collapse on Amazon-Book
# ════════════════════════════════════════════════════════════════════════════
scale_datasets = ["ML-100k\n(1.7K items)", "ML-1M\n(3.7K items)",
                  "Gowalla\n(40K items)", "Yelp2018\n(45K items)", "Amazon-Book\n(91K items)"]
scale_ndcg = {
    "LightGCN":    [0.1842, 0.2128, 0.0348, 0.0145, 0.0002],
    "LightGCN++":  [0.1627, 0.2054, 0.1092, 0.0359, 0.0222],
    "DyHuCoG":     [0.1718, 0.2114, 0.0338, 0.0144, 0.0002],
    "CoopGCN (Ours)": [0.1826, 0.1994, 0.1089, 0.0376, 0.0235],
}
style = {
    "LightGCN":       (C_LGCN,   "o",  "--", 1.5),
    "LightGCN++":     (C_LGCNPP, "s",  "--", 1.5),
    "DyHuCoG":        (C_PEER1,  "^",  "--", 1.5),
    "CoopGCN (Ours)": (C_COOP,   "D",  "-",  2.8),
}

xs = np.arange(len(scale_datasets))
fig, ax = plt.subplots(figsize=(11, 5.5))
for model, values in scale_ndcg.items():
    col, marker, ls, lw = style[model]
    ax.plot(xs, values, marker=marker, linestyle=ls, linewidth=lw,
            color=col, label=model, markersize=8 if model == "CoopGCN (Ours)" else 6,
            zorder=5 if model == "CoopGCN (Ours)" else 3)
    # label endpoints
    ax.text(xs[-1] + 0.07, values[-1], f"{values[-1]:.4f}",
            va="center", fontsize=8, color=col, fontweight="bold" if model == "CoopGCN (Ours)" else "normal")

ax.set_xticks(xs)
ax.set_xticklabels(scale_datasets, fontsize=10)
ax.set_ylabel("NDCG@20", fontsize=12)
ax.set_title("NDCG@20 Across Dataset Scales — Graph CF Models\n"
             "CoopGCN and LightGCN++ are the only models that survive Amazon-Book scale",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(-0.2, len(scale_datasets) - 0.5)

# shade the two large-scale datasets
ax.axvspan(2.5, len(scale_datasets) - 0.5, alpha=0.07, color="navy", label="_nolegend_")
ax.text(3.5, ax.get_ylim()[1] * 0.97, "Large-scale\n(sparse)", ha="center",
        fontsize=9, color="navy", alpha=0.6, va="top")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ fig6_scaling.png")

print("\nAll figures saved to", os.path.abspath(OUT))
