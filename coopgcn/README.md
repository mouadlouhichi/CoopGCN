# CoopGCN — PyTorch Package & Benchmark Harness

This directory contains the official PyTorch package for **CoopGCN**, along with the baseline models, evaluation harness, and visualization tools.

---

## Package Structure (`coopgcn/`)
- `dataset.py`: Benchmark dataset loader (MovieLens-100K, MovieLens-1M, Gowalla), temporal splits, and Step 0.5 Leakage Audit (`audit_leakage()`).
- `models.py`: `MCShapleyEdgeWeighting` ($\mathbf{G_1}$), `ShapleyHypergraphConv` ($\mathbf{G_2}$), `SVDContrastiveView`, `CoopGCN`, and baselines (`LightGCN`, `LightGCN++`, `GAT-CF`, `DyHuCoG`).
- `shapley_data.py`: Truncated Monte-Carlo Data-Shapley valuation ($\mathbf{G_3}$) and bottom-5% noise pruning.
- `losses.py`: Multi-task objective (`CoopGCNLoss`) with zero-overhead inference consistency regularization ($\mathcal{L}_{\text{game}}$).
- `evaluator.py`: Multi-dimensional metrics (`NDCG@20`, `Recall@20`, `TR@20`, `Coverage@20`, `Gini`, Noise Immunity).
- `trainer.py`: Amortized training schedule optimized for Apple Silicon Mac M4 Pro (`torch.device('mps')`).
- `visualization.py`: Generates publication academic figures (`./figures/`).

---

## Usage Example

```python
import torch
from coopgcn import load_benchmark_dataset, CoopGCN, CoopGCNLoss, CoopGCNTrainer

# 1. Download & load MovieLens-100K dataset (with Step 0.5 audit)
dataset = load_benchmark_dataset("ML-100k", download=True)

# 2. Instantiate CoopGCN model
model = CoopGCN(
    num_users=dataset.num_users,
    num_items=dataset.num_items,
    embed_dim=64,
    num_layers=3,
    lambda_param=0.35,
    num_hyperedges=max(10, len(dataset.hyperedges))
)

# 3. Train on Apple Silicon Metal MPS or GPU/CPU
trainer = CoopGCNTrainer(
    model=model,
    dataset=dataset,
    loss_fn=CoopGCNLoss(),
    device="mps" if torch.backends.mps.is_available() else "cpu"
)
trainer.train(epochs=20)
```
