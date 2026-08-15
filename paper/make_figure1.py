"""
Figure 1 (architecture) — monochrome fallback renderer.

The authoritative figure is the TikZ picture embedded in coopgcn_array.tex
(black only, per the DyHuCoG house style). This script reproduces the same
diagram with matplotlib so the no-LaTeX preview build has an equivalent
image. If you compile with pdflatex, the TikZ version is used and this file
is not needed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({
    "font.family": "DejaVu Serif",
    "text.color": "black",
})

K = "black"
fig, ax = plt.subplots(figsize=(13.6, 8.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, lines, dashed=False, tfs=9.0, fs=7.8, lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0.5",
        fc="white", ec=K, lw=lw,
        linestyle=(0, (4, 2)) if dashed else "solid"))
    ty = y + h - 2.2
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=K)
        ty -= 3.6
    for ln in lines:
        ax.text(x + 2.0, ty, ln, ha="left", va="top", fontsize=fs, color=K)
        ty -= 3.3


def arrow(x1, y1, x2, y2, dashed=False, lw=0.9):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        lw=lw, color=K, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 2)) if dashed else "solid"))


def elbow(x1, y1, x2, y2, ymid, dashed=False):
    ax.plot([x1, x1], [y1, ymid], color=K, lw=0.9,
            linestyle="--" if dashed else "-")
    ax.plot([x1, x2], [ymid, ymid], color=K, lw=0.9,
            linestyle="--" if dashed else "-")
    arrow(x2, ymid, x2, y2, dashed=dashed)


# input
box(28, 91.5, 44, 6.5,
    r"Input bipartite graph  $\mathcal{G}=(\mathcal{U},\mathcal{I},\mathcal{E})$ — training split only",
    [], tfs=9.4)

# three channels
box(1, 65, 31, 21, r"Channel A — edge game  $\mathbf{G_1}$", [
    r"1. Coalitions $S\subseteq\mathcal{N}(u)$,  $|S|\leq L$",
    r"2. $v^{\rm cons}_u(S)=-\|\frac{1}{|S|}\sum_{i\in S}e_i-\bar{e}_u\|^2$",
    r"3. MC-Shapley $\hat{\phi}_{ui}$ over $T$ perms",
    r"4. $w_{ui}=\frac{1}{\sqrt{d_ud_i}}[(1-\lambda)+\lambda\sigma(\hat{\phi}_{ui}/\tau)]$"])

box(34.5, 65, 31, 21, r"Channel B — group game  $\mathbf{G_2}$", [
    r"1. Hyperedges $h\in\mathcal{E}_H$ (train only)",
    r"2. $v_h(S)=\alpha\,$aff$\,+\beta\,$tail$\,+\gamma\,$div",
    r"3. Group Shapley $\hat{\phi}_j(h)$",
    r"4. $\beta_h=\sigma(\frac{1}{|h|}\sum_{j\in h}\hat{\phi}_j(h)/\tau_h)$"])

box(68, 65, 31, 21, "Channel C — contrastive", [
    r"1. Global adjacency $A$",
    r"2. Truncated SVD, rank $q$",
    r"3. Low-rank reconstructed graph",
    r"4. Propagate $e^g_u,\; e^g_i$"])

for x in (16.5, 50, 83.5):
    elbow(50, 91.5, x, 86, 89)

# pooling
box(16, 53, 68, 7,
    r"Multi-layer propagation and layer pooling:   $\bar{e}_u=\sum_k\alpha_k e^{(k)}_u$,   $\bar{e}_i=\sum_k\alpha_k e^{(k)}_i$",
    [], tfs=9.2)
for x in (16.5, 50, 83.5):
    elbow(x, 65, min(max(x, 20), 80), 60, 62.5)

# G3 + bridge
box(3, 27, 42, 20, r"$\mathbf{G_3}$  data valuation game", [
    r"Players: training interactions $(u,i)$",
    r"Utility: holdout NDCG@20 (temporal val.)",
    r"TMC-Shapley every $M$ epochs",
    r"$\gamma_{ui}=1+\kappa\,$rank$(\hat{\phi}^{\rm data}_{ui})$;  prune bottom $p\%$"],
    dashed=True)

box(55, 27, 42, 20, r"$\mathcal{L}_{\rm game}$  inference bridge", [
    r"Learnable attention $a_{ui}$",
    r"EMA target $\bar{\hat{\phi}}_{ui}\leftarrow\eta\bar{\hat{\phi}}_{ui}+(1-\eta)\hat{\phi}_{ui}$",
    r"$\mathcal{L}_{\rm game}=\|\sigma(a_{ui})-{\rm sg}(\bar{\hat{\phi}}_{ui})\|^2$",
    r"Inference: $\sigma(a_{ui})$ only $\Rightarrow$ zero cost"],
    dashed=True)

elbow(50, 53, 24, 47, 50)
elbow(50, 53, 76, 47, 50)

# objective
box(16, 13, 68, 7,
    r"$\mathcal{L}=\mathcal{L}_{\rm rank}$ (gBCE, 256 neg, $\gamma_{ui}$-weighted) $+\ \lambda_1\mathcal{L}_{\rm cl}$ (InfoNCE) $+\ \lambda_2\mathcal{L}_{\rm game}+\lambda_3\|\Theta\|^2$",
    [], tfs=9.2)
elbow(24, 27, 34, 20, 23.5)
elbow(76, 27, 66, 20, 23.5)

# feedback path: distilled attention returns to Channel A at inference
#   right of the bridge -> down under the objective -> left -> up into Channel A
ax.plot([97, 99.3], [37, 37], color=K, lw=0.9, linestyle="--")
ax.plot([99.3, 99.3], [37, 9.0], color=K, lw=0.9, linestyle="--")
ax.plot([99.3, 0.7], [9.0, 9.0], color=K, lw=0.9, linestyle="--")
ax.plot([0.7, 0.7], [9.0, 75.5], color=K, lw=0.9, linestyle="--")
ax.add_patch(FancyArrowPatch((0.7, 74.5), (0.7, 75.6), arrowstyle="-|>",
                             mutation_scale=11, lw=0.9, color=K))
ax.text(50, 10.2, r"$\sigma(a_{ui})$ used at inference — no Shapley computation",
        ha="center", fontsize=7.6, style="italic", color=K)

plt.tight_layout()
fig.savefig("figures/Figure_1.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("saved figures/Figure_1.png")
