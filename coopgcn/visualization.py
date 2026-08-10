"""
Publication-quality visualization module for CoopGCN benchmark results.
Generates academic figures (DyHuCoG / RecSys style):
1. Overall Ranking Accuracy (NDCG@20 and Recall@20)
2. Long-Tail & Catalog Equity (Tail Recall TR@20 and Coverage@20)
3. Noise Immunity Curves under adversarial edge injection
4. Amortized Training Convergence & Shapley Consistency L_game trajectory
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def set_academic_style():
    """
    Sets clean publication-grade Matplotlib plotting style.
    """
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "grid.alpha": 0.3,
            "axes.grid": True,
        }
    )


def plot_ndcg_recall_comparison(results_dict, save_path=None):
    """
    Plots bar chart comparing NDCG@20 and Recall@20 across models.
    results_dict: {model_name: {"NDCG@20": val, "Recall@20": val, ...}}
    """
    set_academic_style()
    models = list(results_dict.keys())
    ndcgs = [results_dict[m].get("NDCG@20", 0.0) for m in models]
    recalls = [results_dict[m].get("Recall@20", 0.0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(
        x - width / 2, ndcgs, width, label="NDCG@20", color="#2563eb", edgecolor="black", alpha=0.9
    )
    rects2 = ax.bar(
        x + width / 2, recalls, width, label="Recall@20", color="#16a34a", edgecolor="black", alpha=0.9
    )

    ax.set_ylabel("Metric Score")
    ax.set_title("Overall Recommendation Accuracy (NDCG@20 & Recall@20)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.set_ylim(0, max(recalls + ndcgs) * 1.15)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_tail_coverage_comparison(results_dict, save_path=None):
    """
    Plots grouped bar chart comparing Tail Recall (TR@20) & Catalog Coverage@20.
    Demonstrates popularity bias mitigation and catalog equity.
    """
    set_academic_style()
    models = list(results_dict.keys())
    tr_vals = [results_dict[m].get("TR@20", 0.0) for m in models]
    cov_vals = [results_dict[m].get("Coverage@20", 0.0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(
        x - width / 2, tr_vals, width, label="Tail Recall (TR@20)", color="#9333ea", edgecolor="black", alpha=0.9
    )
    rects2 = ax.bar(
        x + width / 2, cov_vals, width, label="Catalog Coverage@20", color="#f59e0b", edgecolor="black", alpha=0.9
    )

    ax.set_ylabel("Metric Score")
    ax.set_title("Popularity Bias Mitigation (Tail Recall TR@20 & Catalog Coverage@20)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.set_ylim(0, max(tr_vals + cov_vals) * 1.15)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_noise_immunity_curves(noise_results_dict, save_path=None):
    """
    Plots robustness curves showing NDCG@20 degradation under adversarial noisy edges.
    noise_results_dict: {model_name: {noise_ratio: ndcg_val}}
    """
    set_academic_style()
    ratios = [0.0, 0.05, 0.10, 0.20]
    colors = {"LightGCN": "#ef4444", "GAT-CF": "#64748b", "DyHuCoG": "#f59e0b", "CoopGCN": "#2563eb"}
    markers = {"LightGCN": "s", "GAT-CF": "^", "DyHuCoG": "D", "CoopGCN": "o"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, curve in noise_results_dict.items():
        y_vals = [curve.get(r, 0.0) for r in ratios]
        color = colors.get(model_name, "#333333")
        marker = markers.get(model_name, "o")
        ax.plot(
            [r * 100 for r in ratios],
            y_vals,
            marker=marker,
            linewidth=2.5,
            markersize=7,
            label=model_name,
            color=color,
        )

    ax.set_xlabel("Adversarial Injected Noise Ratio (%)")
    ax.set_ylabel("NDCG@20")
    ax.set_title("Adversarial Robustness & Noise Immunity Analysis")
    ax.set_xticks([0, 5, 10, 20])
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_trajectory(history_dict, save_path=None):
    """
    Plots amortized training loss and validation NDCG@20 across epochs.
    """
    set_academic_style()
    epochs = history_dict["epoch"]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = "#2563eb"
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Total Loss (L_total)", color=color)
    ax1.plot(epochs, history_dict["loss_total"], color=color, linewidth=2.5, label="Total Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color2 = "#16a34a"
    ax2.set_ylabel("Validation NDCG@20", color=color2)
    ax2.plot(
        epochs,
        history_dict["val_ndcg"],
        color=color2,
        linewidth=2.5,
        linestyle="--",
        marker="o",
        label="Val NDCG@20",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("CoopGCN Training Convergence & Shapley Consistency")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_benchmark_results(results_dict, noise_dict=None, history_dict=None, output_dir="figures"):
    """
    Generates all 4 publication benchmark figures and saves to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_ndcg_recall_comparison(results_dict, save_path=os.path.join(output_dir, "fig1_ndcg_recall.png"))
    plot_tail_coverage_comparison(results_dict, save_path=os.path.join(output_dir, "fig2_tail_coverage.png"))
    if noise_dict:
        plot_noise_immunity_curves(noise_dict, save_path=os.path.join(output_dir, "fig3_noise_immunity.png"))
    if history_dict:
        plot_training_trajectory(history_dict, save_path=os.path.join(output_dir, "fig4_training_convergence.png"))
