import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams.update({"font.family":"DejaVu Sans"})

fig, ax = plt.subplots(figsize=(13.5, 8.6))
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

C_IN="#37474F"; C_A="#1a6faf"; C_B="#e07b39"; C_C="#3aaa6e"; C_G3="#b55cc0"; C_OUT="#455A64"

def box(x,y,w,h,title,lines,fc,ec,fs=8.4,tfs=9.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=1.2",
                 fc=fc,ec=ec,lw=1.9,alpha=0.97))
    ax.text(x+w/2,y+h-2.6,title,ha="center",va="top",fontsize=tfs,fontweight="bold",color=ec)
    for i,l in enumerate(lines):
        ax.text(x+2.4,y+h-6.4-i*3.5,l,ha="left",va="top",fontsize=fs,color="#263238")

def arrow(x1,y1,x2,y2,color="#546E7A",style="-|>",lw=1.9,rad=0.0,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=15,
                 lw=lw,color=color,connectionstyle=f"arc3,rad={rad}",linestyle=ls))

# input
box(30,89,40,9,"Input Bipartite Graph  G = (U, I, E)   [train split only]",
    [],"#ECEFF1",C_IN,tfs=10.2)

# three channels
box(1.5,60,31,25,"Channel A — Edge Shapley GCN  (G1)",
    ["1. Coalitions S ⊆ N(u), |S| ≤ 32",
     "2. v_cons(S) = −|| mean(e_S) − ē_u ||²",
     "3. MC-Shapley  φ̂_ui   (T permutations)",
     "4. w_ui = (1/√(d_u d_i))·[(1−λ)+λσ(φ̂/τ)]"],"#E3F2FD",C_A)

box(34.5,60,31,25,"Channel B — Hyperedge GCN  (G2)",
    ["1. Hyperedges h ∈ E_H (train-only)",
     "2. v_h(S) = α·aff + β·tail + γ·div",
     "3. Group Shapley φ̂_j(h)",
     "4. β_h = σ( mean_j φ̂_j(h) / τ_h )"],"#FFF3E0",C_B)

box(67.5,60,31,25,"Channel C — SVD Contrastive View",
    ["1. Global adjacency A",
     "2. Truncated SVD (top-q)",
     "3. Low-rank reconstructed graph",
     "4. Propagate → e^g_u , e^g_i"],"#E8F5E9",C_C)

for x in (17,50,83):
    arrow(50,89,x,85.4,rad=0.0 if x==50 else (0.10 if x<50 else -0.10))

# pooling
box(18,46.5,64,9.5,"Multi-Layer Propagation & Layer Pooling      ē_u = Σ_k α_k e_u^(k)   ,   ē_i = Σ_k α_k e_i^(k)",
    [],"#ECEFF1",C_OUT,tfs=10.0)
arrow(17,60,34,56)
arrow(50,60,50,56)
arrow(83,60,66,56)

# G3 + consistency
box(3,24.5,42,18,"G3 — Data-Level Valuation Game",
    ["Players: training interactions (u,i)",
     "Utility: holdout NDCG@20 (temporal val.)",
     "TMC-Shapley every M epochs",
     "γ_ui = 1 + κ·rank(φ̂_data);  prune bottom p%"],"#F3E5F5",C_G3)

box(55,24.5,42,18,"L_game — Zero-Overhead Inference Bridge",
    ["Learnable attention a_ui",
     "EMA target  φ̄_ui ← η φ̄_ui + (1−η) φ̂_ui",
     "L_game = || σ(a_ui) − sg(φ̄_ui) ||²",
     "Inference: σ(a_ui) replaces Shapley → 0 cost"],"#E0F7FA","#00838F")

arrow(35,46.5,26,42.8,rad=0.10)
arrow(65,46.5,74,42.8,rad=-0.10)
# feedback from L_game to channel A
arrow(97.2,33,99.4,71.0,color=C_A,rad=-0.30,ls="--",lw=1.6)
ax.text(99.6,52,"a_ui used\nat inference",ha="right",va="center",fontsize=7.6,
        color=C_A,style="italic",fontweight="bold")

# loss
box(18,10,64,10.5,"L = L_rank (gBCE, 256 neg, γ_ui-weighted)  +  λ₁ L_cl (InfoNCE)  +  λ₂ L_game  +  λ₃ ||Θ||²",
    [],"#FAFAFA","#263238",tfs=10.4)
arrow(24,24.5,32,20.5,rad=0.08)
arrow(76,24.5,68,20.5,rad=-0.08)

ax.text(50,4.2,"Shapley estimation is confined to training; inference uses distilled attention weights only.",
        ha="center",fontsize=8.8,style="italic",color="#546E7A")

plt.tight_layout()
fig.savefig("figures/Figure_1.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
