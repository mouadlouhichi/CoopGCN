"""
CoopGCN: Axiomatic Credit Assignment in Graph Convolutional Networks
via Cooperative Game Theory for Robust, Preference-Aware Recommendation.

Modules:
- dataset: Benchmark dataset loader (MovieLens, Gowalla) & Step 0.5 Leakage Audit.
- models: MCShapleyEdgeWeighting (G1), ShapleyHypergraphConv (G2), SVDContrastiveView,
          CoopGCN full architecture, and baselines (LightGCN, LightGCN++, GAT-CF, DyHuCoG).
- losses: Multi-task objective combining L_rank (gBCE), L_cl (InfoNCE), and L_game (consistency).
- shapley_data: Offline TMC-Shapley data valuation (G3) & noise pruning.
- evaluator: Multi-dimensional evaluation (NDCG@K, Recall@K, TR@K, Coverage@K, Gini, Noise Immunity).
- trainer: Amortized training loop with MPS/CUDA/CPU support and periodic Shapley refresh.
- visualization: Publication-quality figure generation.
"""

from .dataset import BenchmarkDataset, load_benchmark_dataset, load_or_generate_dataset
from .models import (
    MCShapleyEdgeWeighting,
    ShapleyHypergraphConv,
    SVDContrastiveView,
    CoopGCN,
    LightGCN,
    LightGCNPlusPlus,
    GATCF,
    DyHuCoGBaseline,
)
from .losses import CoopGCNLoss
from .shapley_data import TMCShapleyDataValuator
from .evaluator import Evaluator, compute_all_metrics, evaluate_noise_immunity
from .trainer import CoopGCNTrainer
from .visualization import plot_benchmark_results

__all__ = [
    "BenchmarkDataset",
    "load_benchmark_dataset",
    "load_or_generate_dataset",
    "MCShapleyEdgeWeighting",
    "ShapleyHypergraphConv",
    "SVDContrastiveView",
    "CoopGCN",
    "LightGCN",
    "LightGCNPlusPlus",
    "GATCF",
    "DyHuCoGBaseline",
    "CoopGCNLoss",
    "TMCShapleyDataValuator",
    "Evaluator",
    "compute_all_metrics",
    "evaluate_noise_immunity",
    "CoopGCNTrainer",
    "plot_benchmark_results",
]

__version__ = "1.0.0"
