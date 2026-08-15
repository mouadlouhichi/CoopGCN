"""
Data-level credit proxy used by the retained CoopGCN checkpoints.

The legacy class/API uses TMC-Shapley terminology, but the executable method is
an EMA of score-based credits with a tail multiplier and quantile reweighting.
It computes a bottom-percentile mask; the retained trainer does not apply that
mask to remove graph edges. This distinction is documented in the manuscript.
"""

import numpy as np
import torch


class TMCShapleyDataValuator:
    """
    Computes score-and-tail interaction credits for quantile reweighting.

    The name is retained for compatibility; this implementation does not
    retrain coalitions or evaluate validation NDCG per TMC permutation.
    """

    def __init__(self, num_train_edges, prune_cutoff_percentile=5.0, kappa=0.01):
        self.num_train_edges = num_train_edges
        self.prune_cutoff_percentile = prune_cutoff_percentile
        self.kappa = kappa
        self.sample_credits = np.ones(num_train_edges, dtype=np.float32)
        self.sample_weights = np.ones(num_train_edges, dtype=np.float32)
        self.pruned_mask = np.zeros(num_train_edges, dtype=bool)

    def evaluate_sample_shapley(self, model, dataset, edge_index, topo_norm, num_mc_steps=5):
        """
        Refresh the deterministic score-and-tail proxy.

        ``num_mc_steps`` is retained for API compatibility and is unused.
        """
        device = edge_index.device
        model.eval()
        with torch.no_grad():
            final_u, final_i, _ = model(edge_index, topo_norm, getattr(dataset, "hyperedges", None))

            # Retrieve only the first num_train_edges (u -> i edges)
            u_indices = edge_index[0, : self.num_train_edges]
            i_indices_raw = edge_index[1, : self.num_train_edges]

            # Safely map to item index space (0 to num_items-1)
            i_indices = torch.clamp(
                i_indices_raw - dataset.num_users, 0, dataset.num_items - 1
            )

            scores = (final_u[u_indices] * final_i[i_indices]).sum(dim=-1)

            # Vectorized lookup of tail bonus
            tail_mask_dev = dataset.tail_item_mask.to(device)
            tail_bonus = torch.where(tail_mask_dev[i_indices], 1.02, 0.99)

            credits = (torch.sigmoid(scores) * tail_bonus).cpu().numpy()

        self.sample_credits = 0.80 * self.sample_credits + 0.20 * credits

        ranks = np.argsort(np.argsort(self.sample_credits)) / float(
            max(1, len(self.sample_credits) - 1)
        )
        self.sample_weights = 1.0 + self.kappa * ranks

        cutoff = np.percentile(self.sample_credits, self.prune_cutoff_percentile)
        self.pruned_mask = self.sample_credits <= cutoff

    def get_sample_weights_tensor(self, device="cpu"):
        """
        Returns PyTorch tensor of sample weights gamma_ui for ranking loss L_rank.
        """
        return torch.tensor(self.sample_weights, dtype=torch.float32, device=device)
