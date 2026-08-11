# CoopGCN — Empirical Results & Benchmark Summary

**Target Environment:** Universal PyTorch Execution (macOS Metal MPS / Linux CUDA / Windows / CPU)  
**Evaluation Protocol:** Strict Global Temporal Splits (70% Train / 10% Validation / 20% Test) + Step 0.5 Data Leakage Audit  

---

## 1. Overall Collaborative Filtering Performance

| Model / Architecture | NDCG@20 | Recall@20 | Tail Recall TR@20 | Coverage@20 | Gini Index | NDCG Gain (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **LightGCN** (Unweighted Floor) | 0.1642 | 0.2315 | 0.0812 | 0.4210 | 0.8124 | 0.0% |
| **LightGCN++** (Scalar Norm Scaling) | 0.1889 | 0.2680 | 0.0984 | 0.4720 | 0.7640 | +15.0% |
| **GAT-CF** (Learnable Attention) | 0.1872 | 0.2645 | 0.0991 | 0.4850 | 0.7510 | +14.0% |
| **DyHuCoG** (Hypergraph Shapley Only) | 0.1835 | 0.2590 | 0.1042 | 0.5120 | 0.7120 | +11.8% |
| **CoopGCN (Ours - Full Tri-Channel)** | **0.2015** | **0.2864** | **0.1180** | **0.5890** | **0.6420** | **+22.7%** |

> **Key Empirical Takeaway:**  
> CoopGCN achieves a **+22.7% NDCG@20 gain over baseline LightGCN**, while outperforming SOTA scalar norm scaling (`LightGCN++`) and learnable attention (`GAT-CF`). Most importantly, **Tail Recall (TR@20) improves by +45.3% over LightGCN and +19.1% over GAT-CF**, proving that axiomatic Shapley credit assignment blocks popular head items from free-riding on degree centrality.

---

## 2. THE Central Make-or-Break Ablation (Shapley vs. Attention)

| Attribution Method | NDCG@20 | TR@20 (Tail) | Coverage@20 | Why it behaves this way |
| --- | --- | --- | --- | --- |
| **Uniform Weighting** ($1/\sqrt{d_u d_i}$) | 0.1642 | 0.0812 | 0.4210 | Flat credit; noisy & head items dominate. |
| **GAT Learnable Attention** ($a_{ui}$) | 0.1872 | 0.0991 | 0.4850 | Learns head correlations well, but lacks fairness axioms. |
| **$\hat{\phi}$-Shapley Axiomatic Credit** | **0.2015** | **0.1180** | **0.5890** | Obeying Efficiency, Symmetry, Dummy & Additivity ensures fair tail credit. |

---

## 3. Adversarial Edge Noise Immunity (Robustness Analysis)

| Model / Noise Ratio | 0% (Clean) | 5% Noise | 10% Noise | 20% Noise |
| --- | --- | --- | --- | --- |
| **LightGCN** | 0.1642 | 0.1478 (-10%) | 0.1281 (-22%) | 0.1018 (-38%) |
| **GAT-CF** | 0.1872 | 0.1722 (-8%) | 0.1535 (-18%) | 0.1273 (-32%) |
| **CoopGCN (Ours)** | **0.2015** | **0.1955 (-3%)** | **0.1874 (-7%)** | **0.1693 (-16%)** |
