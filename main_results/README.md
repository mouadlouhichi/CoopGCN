# CoopGCN — Main Results

This folder is self-contained. All files needed to generate publication figures and LaTeX tables are here.

## Contents

| File | Description |
|---|---|
| `coopgcn_expected_figures.ipynb` | Main notebook — run this to generate all figures & tables |
| `expected_results.csv` | 5 models × 2 datasets (ML-1M, Amazon-Book) — NDCG, Recall, TR@20, Coverage, Gini |
| `expected_ablation.csv` | G1/G2/G3/CL component ablation results (ML-1M) |
| `expected_noise.csv` | NDCG@20 under 0%, 5%, 10%, 20% injected edge noise (ML-1M) |

## How to run

```bash
cd main_results
jupyter notebook coopgcn_expected_figures.ipynb
# Kernel → Restart & Run All
```

Figures are saved to `main_results/figures/`  
LaTeX tables are saved to `main_results/tables/`

## How to update with real training results

After running the full training benchmark:

1. Replace `expected_results.csv` with your real test metrics
2. Replace `expected_ablation.csv` with real ablation results
3. Replace `expected_noise.csv` with real noise immunity results
4. Re-run the notebook → all 6 figures and 4 tables regenerate automatically

## CSV format

### expected_results.csv
```
dataset,model,recall_20,ndcg_20,tr_20,coverage_20,gini,source
ML-1M,LightGCN,0.2741,0.2169,...
```
- `source`: use `published_*` for literature numbers, `real_*` for your trained results

### expected_ablation.csv
```
variant,recall_20,ndcg_20,tr_20,coverage_20,gini,note
LightGCN (floor),0.2741,0.2169,...
```

### expected_noise.csv
```
noise_ratio,LightGCN,LightGCN++,HCCF,DyHuCoG,CoopGCN (Ours)
0.00,0.2169,...
```

## Figures generated

| Figure | Content |
|---|---|
| `fig1_main_performance.png` | NDCG@20 & Recall@20 — grouped bar chart |
| `fig2_tail_coverage.png` | TR@20 & Catalog Coverage@20 |
| `fig3_ablation.png` | 3-panel component ablation |
| `fig4_noise_immunity.png` | Adversarial noise robustness |
| `fig5_gain_heatmap.png` | Relative gain over DyHuCoG |
| `fig6_radar.png` | Holistic performance radar |

## LaTeX tables generated

| Table | Usage in paper |
|---|---|
| `tab1_overall.tex` | Table I — Main results |
| `tab2_ablation.tex` | Table II — Ablation study |
| `tab3_noise.tex` | Table III — Noise immunity |
| `tab4_gain.tex` | Table IV — Gain over DyHuCoG |

Include in LaTeX with: `\input{main_results/tables/tab1_overall.tex}`
