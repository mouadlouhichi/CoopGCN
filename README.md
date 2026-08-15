# CoopGCN: Axiomatic Credit Assignment in Graph Convolutional Networks via Cooperative Game Theory

[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1.0-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Cross-Platform](https://img.shields.io/badge/Cross--Platform-MPS--CUDA--CPU-000000.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Target: RecSys / KDD](https://img.shields.io/badge/Venue-RecSys%20%2F%20KDD-4B0082.svg?style=flat-square)](#)

This is the official PyTorch implementation and academic specification repository for:
> **CoopGCN: Axiomatic Credit Assignment in Graph Convolutional Networks via Cooperative Game Theory for Robust, Preference-Aware Recommendation**  
> *Mouad Louhichi, et al. (2026)*  
> **Paper Blueprint:** [`specs/CoopGCN_Paper_Structure.md`](specs/CoopGCN_Paper_Structure.md)  
> **Technical Specification:** [`specs/CoopGCN_Spec.md`](specs/CoopGCN_Spec.md)  
> **Implementation Specification:** [`specs/CoopGCN_Implementation_Spec.md`](specs/CoopGCN_Implementation_Spec.md)  
> **8-Year Recommender Leaderboard (2018–2026):** [`specs/8-Year-Recommender-Leaderboard-Recall-NDCG.md`](specs/8-Year-Recommender-Leaderboard-Recall-NDCG.md)  
> **Empirical Review Theme Analysis (ML-100k, ML-1M, Gowalla):** [`specs/CoopGCN_Empirical_Review_Theme.md`](specs/CoopGCN_Empirical_Review_Theme.md)

---

## Executive Overview

While linear Graph Convolutional Networks—most notably **LightGCN**—have become the de facto standard in collaborative filtering by dropping nonlinear activations and feature transformations, they suffer from four structural failure modes:
1. **Uniform neighbor weighting** ($1/\sqrt{d_u d_i}$), treating casual ratings and strong passion interactions identically.
2. **Absence of credit assignment**, leaving models unable to explain which historical interactions drove a recommendation.
3. **Pairwise-only topological bias**, ignoring multi-item group structures (sessions, categories, social bundles).
4. **Popularity-bias amplification**, exacerbated by standard BPR loss with uniform negative sampling.

**CoopGCN** models collaborative-filtering message passing as a **cooperative credit-assignment game**. Monte-Carlo Shapley estimates assign approximate credit to edges ($\mathbf{G_1}$), hyperedges ($\mathbf{G_2}$), and training samples ($\mathbf{G_3}$). The four Shapley axioms apply to exact credits; deployed sigmoid-transformed, distilled weights do not preserve all four.

---

## Key Features & Evidence Status

* 🧩 **Tri-level credit assignment:** Edge, hyperedge and data-level games are implemented alongside an SVD contrastive channel.
* 📊 **Measured benchmark record:** The retained record covers five datasets and ten models under temporal full-catalogue evaluation. It is **single-run** and has no confidence intervals or significance tests.
* ⚖️ **Accuracy–exposure trade-off:** Measured CoopGCN leads direct hypergraph/cooperative peers in NDCG on four of five datasets and in Tail Recall/Coverage on all five, but ranks last among measured graph models on ML-1M NDCG.
* 🧪 **Projection quarantine:** `main_results/expected_*.csv` contains historical design targets, not measurements. Its ablation and noise curves must not be cited as findings; see [`main_results/ANALYSIS.md`](main_results/ANALYSIS.md).
* ⚡ **No serving-time Shapley sampling:** $\mathcal{L}_{\text{game}}$ distils EMA credit into attention. Residual attention latency and training overhead have not been measured.
* 🛡️ **Leakage controls:** Degrees, tail masks and hyperedges are computed from training edges under a global temporal 70/10/20 split and checked by `audit_leakage()`.
* 🌐 **Cross-platform implementation:** Device selection supports MPS, CUDA and CPU, with checkpoint resumption for benchmark runs.

---

## Repository Architecture

```
CoopGCN/
├── README.md                              # Main Overview, Benchmark Table & BibTeX Citation
├── LOCAL_RUN.md                           # Local Execution Guide (macOS / Linux / Windows / Colab)
├── run_local.sh                           # 1-Click Automated Runner & Syntax Checker
├── requirements.txt                       # Top-level dependencies
├── LICENSE                                # MIT License
├── coopgcn/                               # Main PyTorch Python Package
│   ├── dataset.py                         # Benchmark dataset downloader & Step 0.5 audit
│   ├── models.py                          # MCShapleyEdgeWeighting (G1), ShapleyHypergraphConv (G2),
│   │                                      # SVDContrastiveView, CoopGCN & Baselines
│   ├── losses.py                          # Multi-task loss + Shapley-to-attention bridge (L_game)
│   ├── shapley_data.py                    # TMC-Shapley data valuation (G3) & noise pruning
│   ├── evaluator.py                       # NDCG@20, Recall@20, Tail Recall TR@20, Coverage@20, Gini
│   ├── trainer.py                         # Amortized MPS/Metal training schedule
│   └── visualization.py                   # Academic publication chart generator
├── scripts/                               # CLI Automation Scripts
│   ├── run_all.py                         # Terminal-based benchmark & ablation runner
│   └── emit_tables.py                     # Emits publication LaTeX tables to ./tables/
├── tests/                                 # Pytest Unit Test Suite
│   ├── test_propositions.py               # Asserts 4 Shapley Axioms, Proposition 1 & 2, Step 0.5 Leakage
│   └── test_suite.py                      # Full module, forward pass, and loss coverage tests
├── notebooks/
│   └── coopgcn_run_all.ipynb              # Standalone Executable PyTorch Benchmark Notebook (Universal OS)
├── review/                                # Engineering Verification & Audit Log
│   ├── ISSUE_REGISTER.md                  # Complete log of issues and verified resolutions
│   ├── REVIEW_REPORT_ROUND1.md            # Round 1 Technical Audit Report
│   ├── REVIEW_REPORT_ROUND2.md            # Round 2 Technical Re-Audit Report
│   └── FINAL_REVIEW_VERDICT.md            # Official Verification & Sign-off
└── specs/                                 # Academic Specification Suite
    ├── CoopGCN_Paper_Structure.md         # Full Academic Research Paper Blueprint
    ├── CoopGCN_Spec.md                    # Enhanced Technical Specification & Failure Mode Taxonomy
    ├── CoopGCN_Implementation_Spec.md     # Formalization, PyTorch Math & Experiment Matrix
    ├── CoopGCN_Empirical_Review_Theme.md  # Cross-Dataset Empirical Review Theme Analysis
    └── 8-Year-Recommender-Leaderboard-Recall-NDCG.md # 8-Year Recommender Leaderboard (2018–2026)
```

---

## Quickstart (macOS / Linux / Windows / Colab)

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/mouadlouhichi/CoopGCN.git
cd CoopGCN
pip install -r requirements.txt
```

### 2. Verify Mathematical Propositions & Automated Tests
Run the automated test suite for credit-ordering sanity checks, Channel-A recovery at $\lambda=0$, module forward passes, and split-disjointness assertions. These unit tests do not establish end-to-end robustness or statistical superiority:
```bash
python3 tests/test_propositions.py
python3 tests/test_suite.py
```

### 3. Option A: Interactive Jupyter Notebook (Recommended)
Launch Jupyter Notebook and open the self-contained benchmark notebook:
```bash
jupyter notebook notebooks/coopgcn_run_all.ipynb
```
Click **"Run All"**:
- Automatically detects **Apple Metal MPS GPU acceleration** (`torch.device('mps')`).
- Downloads and loads **The 5 Target Benchmark Datasets** (`ML-100k`, `ML-1M`, `Gowalla`, `Yelp2018`, `Amazon-Book`).
- Executes all **6 Experiments**:
  1. Automated Data Download & Step 0.5 Leakage Audit
  2. Head-to-Head Baseline Training (`LightGCN`, `LightGCN++`, `GAT-CF`, `DyHuCoG`, and `CoopGCN`)
  3. THE Central Make-or-Break Ablation (Shapley vs. Learnable Attention)
  4. Complete 10-Row Component Ablation Study ($\mathbf{G_1, G_2, G_3, \mathcal{L}_{\text{game}}}$)
  5. Optional random-edge-injection runs (these must be retrained and measured before making a robustness claim)
  6. Publication Figure Generation (`./figures/`) and LaTeX Table Emission (`./tables/`).

### 4. Option B: Command Line (CLI Automation)
To execute the complete benchmark across target datasets directly from your terminal:
```bash
python3 scripts/run_all.py --datasets ML-100k ML-1M Gowalla Yelp2018 Amazon-Book --epochs 15 --output_dir results
```
To emit publication LaTeX tables from your empirical results:
```bash
python3 scripts/emit_tables.py
```

---

## Measured Benchmark Record

The retained common metrics are shown as **NDCG@20 / TR@20 / Coverage@20 (%)**. Each cell is one run; differences are not claims of statistical significance.

| Model | ML-100K | ML-1M | Gowalla | Yelp2018 | Amazon-Book |
|---|---:|---:|---:|---:|---:|
| **CoopGCN** | 0.1826 / 0.0106 / 46.9 | 0.1994 / 0.0075 / 40.7 | 0.1089 / 0.0104 / 7.95 | 0.0376 / 0.0006 / 5.36 | 0.0235 / 0.0036 / 6.47 |
| LightGCN++ | 0.1627 / 0.0100 / 46.5 | 0.2054 / 0.0008 / 37.3 | 0.1092 / 0.0116 / 8.06 | 0.0359 / 0.0006 / 5.01 | 0.0222 / 0.0027 / 5.44 |
| GAT-CF | 0.1943 / 0.0022 / 19.4 | 0.2096 / 0.0001 / 10.0 | 0.0038 / 0.0005 / 10.3 | 0.0014 / 0.0003 / 0.13 | 0.0002 / 0.0000 / 0.05 |
| LightGCN | 0.1842 / 0.0036 / 27.8 | 0.2128 / 0.0000 / 9.82 | 0.0348 / 0.0000 / 0.20 | 0.0145 / 0.0000 / 0.35 | 0.0002 / 0.0000 / 0.05 |
| HPCF | 0.1747 / 0.0020 / 25.7 | **0.2141** / 0.0000 / 9.23 | 0.0278 / 0.0000 / 0.39 | 0.0110 / 0.0000 / 0.48 | 0.0003 / 0.0000 / 0.05 |
| HCCF | 0.1723 / 0.0019 / 21.2 | 0.2070 / 0.0000 / 8.23 | 0.0291 / 0.0000 / 0.35 | 0.0113 / 0.0000 / 0.25 | 0.0002 / 0.0000 / 0.05 |
| DyHuCoG | 0.1718 / 0.0026 / 20.5 | 0.2114 / 0.0000 / 9.42 | 0.0338 / 0.0000 / 0.32 | 0.0144 / 0.0000 / 0.35 | 0.0002 / 0.0000 / 0.05 |

The defensible interpretation is preliminary and regime-dependent: broader tail/catalogue exposure, competitive sparse-data NDCG, and a dense ML-1M accuracy deficit. Measured component, noise, timing and attribution studies remain open.

---

## Citation

If you find this repository, specification, or codebase useful in your research, please cite:

```bibtex
@inproceedings{louhichi2026coopgcn,
  title     = {CoopGCN: Axiomatic Credit Assignment in Graph Convolutional Networks via Cooperative Game Theory for Robust, Preference-Aware Recommendation},
  author    = {Louhichi, Mouad and contributors},
  booktitle = {Proceedings of the ACM Recommender Systems Benchmark Companion},
  year      = {2026},
  url       = {https://github.com/mouadlouhichi/CoopGCN}
}
```

---

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
