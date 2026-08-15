"""
Multi-dimensional evaluation harness for CoopGCN:
- NDCG@K and Recall@K (Overall Accuracy)
- Tail Recall TR@K (Long-tail recommendation accuracy on bottom 80% items)
- Catalog Coverage@K and Gini Index (Catalog equity & diversity)
- A placeholder random-edge-injection helper (not a measured robustness result)
Bounds-safe on Apple Metal MPS / CUDA.
"""

import numpy as np
import torch
from collections import defaultdict


class Evaluator:
    """
    Computes ranking, tail, coverage, and robustness metrics for a trained model.
    """

    def __init__(self, k_list=[10, 20]):
        self.k_list = k_list

    @staticmethod
    def _compute_ndcg_at_k(ranked_items, ground_truth_set, k):
        top_k = ranked_items[:k]
        dcg = 0.0
        idcg = 0.0
        for idx, item in enumerate(top_k):
            if item in ground_truth_set:
                dcg += 1.0 / np.log2(idx + 2.0)
        for idx in range(min(k, len(ground_truth_set))):
            idcg += 1.0 / np.log2(idx + 2.0)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _compute_recall_at_k(ranked_items, ground_truth_set, k):
        top_k = set(ranked_items[:k])
        hits = len(top_k.intersection(ground_truth_set))
        return hits / float(len(ground_truth_set)) if len(ground_truth_set) > 0 else 0.0

    @staticmethod
    def _compute_tail_recall_at_k(ranked_items, ground_truth_set, tail_mask, k):
        top_k = set(ranked_items[:k])
        tail_gt = {item for item in ground_truth_set if item < len(tail_mask) and tail_mask[item]}
        if len(tail_gt) == 0:
            return None
        hits = len(top_k.intersection(tail_gt))
        return hits / float(len(tail_gt))

    @staticmethod
    def compute_gini_index(item_counts, num_items):
        counts = np.zeros(num_items, dtype=np.float64)
        for idx, c in enumerate(item_counts):
            if idx < num_items:
                counts[idx] = c
        counts = np.sort(counts)
        n = len(counts)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts) + 1e-8)


def compute_all_metrics(
    model,
    dataset,
    eval_dict,
    k=20,
    device="cpu",
    batch_size=512,
    edge_index=None,
    topo_norm=None,
):
    """
    Evaluates model on eval_dict (user_id -> list of positive item_ids).
    Returns dict containing NDCG@K, Recall@K, TR@K, Coverage@K, and Gini Index.
    100% bounds-safe on Apple Metal MPS.
    """
    model.eval()
    if edge_index is None or topo_norm is None:
        edge_index, topo_norm, _, _ = dataset.get_sparse_adjacency(device=device)

    with torch.no_grad():
        final_u, final_i, _ = model(
            edge_index, topo_norm, getattr(dataset, "hyperedges", None)
        )

    num_users = dataset.num_users
    num_items = dataset.num_items
    tail_mask = dataset.tail_item_mask.cpu().numpy()

    ndcg_list = []
    recall_list = []
    tail_recall_list = []
    rec_item_counts = np.zeros(num_items, dtype=np.int64)

    users_with_gt = [u for u, items in eval_dict.items() if len(items) > 0 and u < num_users]
    for start_idx in range(0, len(users_with_gt), batch_size):
        batch_u_ids = users_with_gt[start_idx : start_idx + batch_size]
        u_tensor = torch.tensor(batch_u_ids, device=device, dtype=torch.long)
        u_tensor = torch.clamp(u_tensor, 0, num_users - 1)
        batch_u_emb = final_u[u_tensor]  # shape: (B, d)

        scores = torch.matmul(batch_u_emb, final_i.T)  # shape: (B, num_items)

        for b_idx, u_id in enumerate(batch_u_ids):
            train_items = [i for i in dataset.user_train_dict.get(u_id, []) if i < num_items]
            if len(train_items) > 0:
                scores[b_idx, train_items] = -float("inf")

        top_k_items = torch.topk(scores, k=k, dim=-1).indices.cpu().numpy()

        for b_idx, u_id in enumerate(batch_u_ids):
            gt_set = {i for i in eval_dict[u_id] if i < num_items}
            ranked = top_k_items[b_idx]

            ndcg_list.append(Evaluator._compute_ndcg_at_k(ranked, gt_set, k))
            recall_list.append(Evaluator._compute_recall_at_k(ranked, gt_set, k))

            tr_val = Evaluator._compute_tail_recall_at_k(ranked, gt_set, tail_mask, k)
            if tr_val is not None:
                tail_recall_list.append(tr_val)

            for rec_item in ranked:
                if rec_item < num_items:
                    rec_item_counts[rec_item] += 1

    ndcg_mean = float(np.mean(ndcg_list)) if len(ndcg_list) > 0 else 0.0
    recall_mean = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0
    tail_recall_mean = (
        float(np.mean(tail_recall_list)) if len(tail_recall_list) > 0 else 0.0
    )
    coverage_val = float(np.count_nonzero(rec_item_counts)) / float(num_items)
    gini_val = float(Evaluator.compute_gini_index(rec_item_counts, num_items))

    return {
        f"NDCG@{k}": ndcg_mean,
        f"Recall@{k}": recall_mean,
        f"TR@{k}": tail_recall_mean,
        f"Coverage@{k}": coverage_val,
        f"Gini": gini_val,
    }


def evaluate_noise_immunity(model_class, dataset, epochs=30, noise_ratios=[0.0, 0.05, 0.10, 0.20], device="cpu"):
    """
    Placeholder API for future random-edge-injection evaluation.

    The current implementation returns the requested ratios, not trained NDCG
    values, and must not be cited as robustness evidence.
    """
    results = {}
    for ratio in noise_ratios:
        if ratio == 0.0:
            noisy_ds = dataset
        else:
            noisy_ds = dataset.inject_noisy_edges(noise_ratio=ratio)
        results[ratio] = ratio
    return results
