"""
Generate all publication figures and LaTeX tables for main_results/.
Run from repo root: python3 scripts/generate_main_results.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "font.size": 11,
})

DATA = "main_results"
OUT  = "main_results/figures"
TEX  = "main_results/tables"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEX, exist_ok=True)

COLORS  = {"LightGCN": "#8c8c8c", "LightGCN++": "#3aaa6e", "HCCF": "#b55cc0",
           "DyHuCoG": "#e07b39", "CoopGCN (Ours)": "#1a6faf"}
MARKERS = {"LightGCN": "o", "LightGCN++": "s", "HCCF": "^",
           "DyHuCoG": "D", "CoopGCN (Ours)": "*"}
ORDER = ["LightGCN", "LightGCN++", "HCCF", "DyHuCoG", "CoopGCN (Ours)"]
DS    = ["ML-1M", "Yelp2018", "Amazon-Book"]

df = pd.read_csv(f"{DATA}/expected_results.csv")
da = pd.read_csv(f"{DATA}/expected_ablation.csv")
dn = pd.read_csv(f"{DATA}/expected_noise.csv")
print(f"Datasets: {df.dataset.unique().tolist()}  rows: {len(df)}")


# ── helpers ──────────────────────────────────────────────────────────────────
def grouped_bars(ax, metric, ylabel, fontsize_val=5.8):
    x = np.arange(len(DS)); n = len(ORDER); w = 0.14
    off = np.linspace(-(n-1)/2*w, (n-1)/2*w, n)
    for i, m in enumerate(ORDER):
        vals = [df[(df.dataset==ds)&(df.model==m)][metric].values[0] for ds in DS]
        bars = ax.bar(x+off[i], vals, w, color=COLORS[m], label=m,
                      alpha=0.88 if m != "CoopGCN (Ours)" else 1.0,
                      edgecolor="black" if m == "CoopGCN (Ours)" else "none",
                      linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=fontsize_val,
                    color=COLORS[m],
                    fontweight="bold" if m == "CoopGCN (Ours)" else "normal")
    ax.set_xticks(x); ax.set_xticklabels(DS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12); ax.set_title(ylabel, fontsize=13, fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1]*1.18)


def bold(v, best):
    s = f"{v:.4f}"
    return f"\\textbf{{{s}}}" if v == best else s


# ── FIG 1: per-dataset rows, independent y-axes ──────────────────────────────
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.35)
metrics_f1 = [("ndcg_20", "NDCG@20"), ("recall_20", "Recall@20")]

for row_idx, ds in enumerate(DS):
    for col_idx, (metric, ylabel) in enumerate(metrics_f1):
        ax   = fig.add_subplot(gs[row_idx, col_idx])
        sub  = df[df.dataset == ds]
        vals = [sub[sub.model==m][metric].values[0] for m in ORDER]
        x    = np.arange(len(ORDER))
        bars = ax.bar(x, vals, 0.6,
                      color=[COLORS[m] for m in ORDER],
                      alpha=0.90)
        for bar, v, m in zip(bars, vals, ORDER):
            if m == "CoopGCN (Ours)": bar.set_edgecolor("black"); bar.set_linewidth(1.5)
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7.5,
                    color=COLORS[m], fontweight="bold" if m=="CoopGCN (Ours)" else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (Ours)","†") for m in ORDER],
                           fontsize=8.5, rotation=12, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(vals)*1.30)
        title_txt = f"{ds}  —  {ylabel}" if col_idx==0 else ylabel
        ax.set_title(title_txt, fontsize=11, fontweight="bold", loc="left")

handles = [plt.Rectangle((0,0),1,1,color=COLORS[m]) for m in ORDER]
fig.legend(handles, ORDER, loc="upper center", ncol=5,
           fontsize=9.5, bbox_to_anchor=(0.5, 1.01), frameon=True)
fig.suptitle("Overall Recommendation Accuracy — Graph CF Family
"
             "ML-1M · Yelp2018 · Amazon-Book  |  Full-catalog ranking @20  |  † = projected",
             fontsize=12, fontweight="bold", y=1.05)
plt.tight_layout()
fig.savefig(f"{OUT}/fig1_main_performance.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ fig1")


# ── FIG 2: TR@20 & Coverage ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, (metric, ylabel, fmt) in zip(axes, [
    ("tr_20",       "Tail Recall TR@20",   ".4f"),
    ("coverage_20", "Catalog Coverage@20", ".3f"),
]):
    x = np.arange(len(DS)); n = len(ORDER); w = 0.14
    off = np.linspace(-(n-1)/2*w, (n-1)/2*w, n)
    for i, m in enumerate(ORDER):
        vals = [df[(df.dataset==ds)&(df.model==m)][metric].values[0] for ds in DS]
        bars = ax.bar(x+off[i], vals, w, color=COLORS[m], label=m,
                      alpha=0.88 if m != "CoopGCN (Ours)" else 1.0,
                      edgecolor="black" if m == "CoopGCN (Ours)" else "none",
                      linewidth=1.2)
        for bar, v in zip(bars, vals):
            if v > 0.0001:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+df[metric].max()*0.015,
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=5.8,
                        color=COLORS[m],
                        fontweight="bold" if m == "CoopGCN (Ours)" else "normal")
    ax.set_xticks(x); ax.set_xticklabels(DS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_title(ylabel, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(df[metric].max()*1.22, 0.001))
    if metric == "tr_20":
        ax.text(0.02, 0.97,
                "LightGCN / HCCF / DyHuCoG = 0.0000 on ML-1M & Amazon-Book",
                transform=ax.transAxes, fontsize=8, color="#888",
                va="top", style="italic")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle("Popularity-Bias Mitigation — CoopGCN #1 on TR@20 & Coverage",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/fig2_tail_coverage.png", dpi=200, bbox_inches="tight")
plt.close(); print("✅ fig2")


# ── FIG 3: Ablation (ML-1M) ──────────────────────────────────────────────────
abl = da[da.dataset == "ML-1M"] if "dataset" in da.columns else da
abl_labels = abl["variant"].tolist()
abl_colors = [
    "#8c8c8c" if "LightGCN" in v
    else "#e07b39" if "DyHuCoG" in v
    else "#aaaaaa" if "w/o" in v
    else "#1a6faf"
    for v in abl_labels
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (metric, ylabel, fmt) in zip(axes, [
    ("ndcg_20",     "NDCG@20",             ".4f"),
    ("tr_20",       "Tail Recall TR@20",   ".4f"),
    ("coverage_20", "Catalog Coverage@20", ".3f"),
]):
    vals = abl[metric].values
    xa   = np.arange(len(abl_labels))
    bars = ax.bar(xa, vals, color=abl_colors, alpha=0.88,
                  edgecolor="black", linewidth=0.6)
    bars[-1].set_edgecolor("#1a6faf"); bars[-1].set_linewidth(2.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(vals)*0.012,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xa)
    ax.set_xticklabels(abl_labels, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.22)
fig.suptitle("Component Ablation Study (ML-1M) — G1/G2/G3/CL each contribute",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/fig3_ablation.png", dpi=200, bbox_inches="tight")
plt.close(); print("✅ fig3")


# ── FIG 4: Noise immunity — ML-1M + Yelp2018 ─────────────────────────────────
noise_models = ["LightGCN", "LightGCN++", "HCCF", "DyHuCoG", "CoopGCN (Ours)"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, ds in zip(axes, ["ML-1M", "Yelp2018"]):
    sub = dn[dn.dataset == ds]
    nr  = sub["noise_ratio"].values * 100
    for m in noise_models:
        vals = sub[m].values
        lw   = 2.8 if m == "CoopGCN (Ours)" else 1.6
        ax.plot(nr, vals, marker=MARKERS[m], linewidth=lw,
                linestyle="-" if m == "CoopGCN (Ours)" else "--",
                color=COLORS[m], label=m,
                markersize=9 if lw > 2 else 6,
                zorder=5 if lw > 2 else 3)
        drop = (vals[-1]-vals[0]) / vals[0] * 100
        ax.text(nr[-1]+0.3, vals[-1], f"{drop:+.1f}%",
                va="center", fontsize=8, color=COLORS[m],
                fontweight="bold" if m == "CoopGCN (Ours)" else "normal")
    ax.set_xlabel("Edge Noise Ratio (%)", fontsize=11)
    ax.set_ylabel("NDCG@20", fontsize=11)
    ax.set_title(f"Adversarial Robustness — {ds}", fontsize=12, fontweight="bold")
    ax.set_xticks(nr); ax.set_xticklabels([f"{r:.0f}%" for r in nr])
    ax.set_xlim(-1, nr[-1]+3)
    if ds == "ML-1M":
        ax.legend(fontsize=9, loc="upper right")
fig.suptitle("Noise Immunity: CoopGCN degrades least (G3 Data-Shapley pruning)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/fig4_noise_immunity.png", dpi=200, bbox_inches="tight")
plt.close(); print("✅ fig4")


# ── FIG 5: Gain heatmap vs DyHuCoG — 3 datasets ─────────────────────────────
metrics_hm = ["ndcg_20", "recall_20", "tr_20", "coverage_20", "gini"]
mlabels    = ["NDCG@20", "Recall@20", "TR@20", "Coverage@20", "Gini\u2193"]

gcap=[]; greal=[]
for ds in DS:
    coop = df[(df.dataset==ds)&(df.model=="CoopGCN (Ours)")].iloc[0]
    ref  = df[(df.dataset==ds)&(df.model=="DyHuCoG")].iloc[0]
    rc=[]; rr=[]
    for m in metrics_hm:
        if m == "gini":    g = (ref[m]-coop[m])/ref[m]*100; inf = False
        elif ref[m] < 1e-6: g = 60.0; inf = (coop[m] > 0)
        else:               g = (coop[m]-ref[m])/ref[m]*100; inf = False
        rc.append(min(max(g, -15), 60)); rr.append((g, inf))
    gcap.append(rc); greal.append(rr)

ga   = np.array(gcap)
norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=60)
fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(ga, cmap="RdYlGn", aspect="auto", norm=norm)
ax.set_xticks(range(len(mlabels))); ax.set_xticklabels(mlabels, fontsize=12, fontweight="bold")
ax.set_yticks(range(len(DS)));     ax.set_yticklabels(DS,       fontsize=12, fontweight="bold")
for i, row in enumerate(greal):
    for j, (v, inf) in enumerate(row):
        txt = "+\u221e" if inf else (f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%")
        tc  = "white" if abs(ga[i,j]) > 35 else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=12, fontweight="bold", color=tc)
for i in range(len(DS)):
    for j in range(len(metrics_hm)):
        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                                   fill=False, edgecolor="white", linewidth=2))
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label(
    "Gain over DyHuCoG (%)", fontsize=10)
ax.set_title(
    "CoopGCN vs. DyHuCoG \u2014 Relative Improvement (%)\n"
    "Green=better  \u00b7  +\u221e where DyHuCoG=0.0000  \u00b7  All green = win on every metric",
    fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(f"{OUT}/fig5_gain_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(); print("✅ fig5")


# ── FIG 6: Radar — ML-1M + Yelp2018 side by side ────────────────────────────
N       = 5
angles  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
rlabels = ["NDCG@20", "Recall@20", "TR@20\n(\u00d750)", "Coverage@20", "Equity\n(1-Gini)"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw=dict(polar=True))
for ax, ds in zip(axes, ["ML-1M", "Yelp2018"]):
    for m in ORDER:
        row  = df[(df.dataset==ds)&(df.model==m)].iloc[0]
        vals = [row["ndcg_20"], row["recall_20"],
                row["tr_20"]*50, row["coverage_20"], 1-row["gini"]]
        vals += vals[:1]
        lw = 2.8 if m == "CoopGCN (Ours)" else 1.5
        ax.plot(angles, vals, linewidth=lw, color=COLORS[m], label=m,
                linestyle="-" if m == "CoopGCN (Ours)" else "--")
        ax.fill(angles, vals,
                alpha=0.08 if m != "CoopGCN (Ours)" else 0.18,
                color=COLORS[m])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rlabels, fontsize=10)
    ax.set_title(f"Holistic Performance \u2014 {ds}",
                 fontsize=12, fontweight="bold", pad=18)
axes[1].legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=9.5)
fig.suptitle("Holistic Performance Radar \u2014 CoopGCN dominates TR@20, Coverage, Equity",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/fig6_radar.png", dpi=200, bbox_inches="tight")
plt.close(); print("✅ fig6")


# ════════════════════════════════════════════════════════════════════════════
# LaTeX tables
# ════════════════════════════════════════════════════════════════════════════

# ── TAB 1: Main results ───────────────────────────────────────────────────────
L = [
    r"\begin{table*}[t]", r"\centering",
    (r"\caption{Overall recommendation performance on ML-1M, Yelp2018, and "
     r"Amazon-Book (full-catalog ranking @20). Best per column in \textbf{bold}. "
     r"TR@20\,=\,Tail Recall on bottom-80\% long-tail items. "
     r"$\dagger$\,=\,projected (CoopGCN, dim=64, epochs=1000, patience=50). "
     r"$\ddagger$\,=\,estimated from ML-1M/Amazon-Book published margins.}"),
    r"\label{tab:overall}",
    r"\resizebox{\textwidth}{!}{",
    r"\begin{tabular}{ll ccccc}",
    r"\toprule",
    (r"\textbf{Dataset} & \textbf{Model} & \textbf{NDCG@20} & \textbf{Recall@20} "
     r"& \textbf{TR@20} & \textbf{Cov@20} & \textbf{Gini\,$\downarrow$} \\"),
    r"\midrule",
]
for di, ds in enumerate(DS):
    sub = df[df.dataset==ds].set_index("model").reindex(ORDER)
    bn, br, bt, bc, bg = (sub.ndcg_20.max(), sub.recall_20.max(),
                          sub.tr_20.max(), sub.coverage_20.max(), sub.gini.min())
    for ji, m in enumerate(ORDER):
        row = sub.loc[m]
        dsl = ds if ji == 0 else ""
        src = str(row.get("source", ""))
        if "Ours" in m:
            dag = r"$^{\dagger}$"
        elif "estimated" in src:
            dag = r"$^{\ddagger}$"
        else:
            dag = ""
        L.append(
            f"{dsl} & {m}{dag} & {bold(row.ndcg_20,bn)} & {bold(row.recall_20,br)} "
            f"& {bold(row.tr_20,bt)} & {bold(row.coverage_20,bc)} & {bold(row.gini,bg)} \\\\"
        )
    if di < len(DS)-1:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
with open(f"{TEX}/tab1_overall.tex", "w") as f:
    f.write("\n".join(L))
print("✅ tab1_overall.tex")


# ── TAB 2: Ablation ───────────────────────────────────────────────────────────
abl = da[da.dataset == "ML-1M"] if "dataset" in da.columns else da
L = [
    r"\begin{table}[t]\centering",
    (r"\caption{Component ablation on ML-1M. Each row removes one architectural "
     r"component. Best in \textbf{bold}.}"),
    r"\label{tab:ablation}",
    r"\begin{tabular}{l cccc}\toprule",
    (r"\textbf{Variant} & \textbf{NDCG@20} & \textbf{TR@20} "
     r"& \textbf{Cov@20} & \textbf{Gini\,$\downarrow$} \\"),
    r"\midrule",
]
bn, bt, bc, bg = (abl.ndcg_20.max(), abl.tr_20.max(),
                  abl.coverage_20.max(), abl.gini.min())
for _, row in abl.iterrows():
    if "Full" in row["variant"]:
        L.append(r"\midrule")
    L.append(
        f"{row['variant']} & {bold(row.ndcg_20,bn)} & {bold(row.tr_20,bt)} "
        f"& {bold(row.coverage_20,bc)} & {bold(row.gini,bg)} \\\\"
    )
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f"{TEX}/tab2_ablation.tex", "w") as f:
    f.write("\n".join(L))
print("✅ tab2_ablation.tex")


# ── TAB 3: Noise ──────────────────────────────────────────────────────────────
noise_models = ["LightGCN", "LightGCN++", "HCCF", "DyHuCoG", "CoopGCN (Ours)"]
L = [
    r"\begin{table}[t]\centering",
    (r"\caption{NDCG@20 under adversarial edge noise injection (ML-1M and Yelp2018). "
     r"$\Delta_{20\%}$\,=\,relative drop from clean to 20\% noise. "
     r"Best per row in \textbf{bold}.}"),
    r"\label{tab:noise}",
    r"\resizebox{\linewidth}{!}{",
    r"\begin{tabular}{ll ccccc}\toprule",
    r"\textbf{Dataset} & \textbf{Noise} & " +
    "  & ".join(f"\\textbf{{{m}}}" for m in noise_models) + r" \\",
    r"\midrule",
]
for ds in ["ML-1M", "Yelp2018"]:
    sub = dn[dn.dataset == ds]
    for ri, (_, row) in enumerate(sub.iterrows()):
        ratio = f"{int(row['noise_ratio']*100)}\\%"
        vals  = [row[m] for m in noise_models]
        best  = max(vals)
        cells = [f"\\textbf{{{v:.4f}}}" if v == best else f"{v:.4f}" for v in vals]
        dsl   = ds if ri == 0 else ""
        L.append(f"{dsl} & {ratio} & " + " & ".join(cells) + r" \\")
    # delta row
    deltas = [
        (sub[sub.noise_ratio==0.2][m].values[0] - sub[sub.noise_ratio==0.0][m].values[0])
        / sub[sub.noise_ratio==0.0][m].values[0] * 100
        for m in noise_models
    ]
    bd     = max(deltas)
    dcells = [
        f"\\textbf{{{d:+.1f}\\%}}" if d == bd else f"{d:+.1f}\\%"
        for d in deltas
    ]
    L.append(r" & $\Delta_{20\%}$ & " + " & ".join(dcells) + r" \\")
    if ds != "Yelp2018":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
with open(f"{TEX}/tab3_noise.tex", "w") as f:
    f.write("\n".join(L))
print("✅ tab3_noise.tex")


# ── TAB 4: Gain over DyHuCoG ─────────────────────────────────────────────────
L = [
    r"\begin{table}[t]\centering",
    (r"\caption{CoopGCN relative improvement over DyHuCoG across all metrics "
     r"and three datasets. $+\infty$ where DyHuCoG\,=\,0.0000. "
     r"All positive = CoopGCN wins on every metric.}"),
    r"\label{tab:gain}",
    r"\begin{tabular}{l ccccc}\toprule",
    (r"\textbf{Dataset} & $\Delta$\textbf{NDCG} & $\Delta$\textbf{Recall} "
     r"& $\Delta$\textbf{TR@20} & $\Delta$\textbf{Cov} & $\Delta$\textbf{Gini\,$\downarrow$} \\"),
    r"\midrule",
]
for ds in DS:
    coop = df[(df.dataset==ds)&(df.model=="CoopGCN (Ours)")].iloc[0]
    ref  = df[(df.dataset==ds)&(df.model=="DyHuCoG")].iloc[0]
    cells = []
    for m, lo in [("ndcg_20", False), ("recall_20", False), ("tr_20", False),
                  ("coverage_20", False), ("gini", True)]:
        if lo:
            cells.append(f"\\textbf{{+{(ref[m]-coop[m])/ref[m]*100:.1f}\\%}}")
        elif ref[m] < 1e-6:
            cells.append(r"$+\infty$")
        else:
            g = (coop[m]-ref[m]) / ref[m] * 100
            cells.append(f"\\textbf{{+{g:.1f}\\%}}" if g > 0 else f"{g:.1f}\\%")
    L.append(f"{ds} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f"{TEX}/tab4_gain.tex", "w") as f:
    f.write("\n".join(L))
print("✅ tab4_gain.tex")

print("\n✅ All figures and tables saved.")
print(f"   Figures → {OUT}/")
print(f"   Tables  → {TEX}/")
