# CoopGCN — PyTorch Package & Benchmark Harness

This directory contains the official PyTorch package for **CoopGCN**, along with the baseline models, evaluation harness, and visualization tools.

---

## Package Structure (`coopgcn/`)
- `dataset.py`: loaders for five benchmarks. MovieLens uses temporal splits; Gowalla/Yelp/Amazon use a pinned upstream split. All graph-derived quantities use train edges.
- `models.py`: legacy-named deterministic edge/group credit proxies, `SVDContrastiveView`, `CoopGCN`, and baselines. The retained proxy methods are not MC Shapley estimators.
- `shapley_data.py`: retained score-and-tail sample-weight proxy ($\mathbf{G_3}$); a bottom-5% mask is computed but not applied as pruning.
- `losses.py`: retained weighted-BPR objective and edge-proxy-to-attention consistency regularization; attention latency is unmeasured.
- `evaluator.py`: NDCG, Recall, TR, Coverage and Gini. Its noise helper is a template, not measured evidence.
- `trainer.py`: Amortized training schedule optimized for universal hardware acceleration (`MPS`, `CUDA`, or `CPU`).
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

# 3. Train universally across GPU (CUDA/MPS) or CPU
trainer = CoopGCNTrainer(
    model=model,
    dataset=dataset,
    loss_fn=CoopGCNLoss(),
    device="mps" if torch.backends.mps.is_available() else "cpu"
)
trainer.train(epochs=20)
```
