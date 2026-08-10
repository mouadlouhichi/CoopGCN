"""
Amortized training loop for CoopGCN and baseline models.
Handles MPS (Mac Apple Silicon Metal), CUDA, and CPU devices.
Executes periodic Shapley EMA refresh (P=10 epochs) and Data-Shapley curation (M=20 epochs).
Includes automatic checkpointing and resumption to prevent re-running completed models.
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from .evaluator import compute_all_metrics
from .shapley_data import TMCShapleyDataValuator


class CoopGCNTrainer:
    """
    Trainer class for CoopGCN and baseline recommendation models.
    Fully compatible with Apple Silicon Mac M4 Pro Metal MPS GPU acceleration.
    Supports automatic checkpoint saving/loading to resume after interruptions.
    """

    def __init__(
        self,
        model,
        dataset,
        loss_fn=None,
        lr=0.005,
        weight_decay=1e-4,
        batch_size=1024,
        num_negs=64,
        device=None,
        shapley_refresh_period=10,
        data_shapley_period=20,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_negs = num_negs
        self.shapley_refresh_period = shapley_refresh_period
        self.data_shapley_period = data_shapley_period

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Precompute sparse adjacency on device
        self.edge_index, self.topo_norm, self.d_u, self.d_i = (
            self.dataset.get_sparse_adjacency(device=self.device)
        )

        # Initialize G3 Data-Shapley Valuator
        self.data_valuator = TMCShapleyDataValuator(
            num_train_edges=len(self.dataset.train_edges), prune_cutoff_percentile=5.0
        )
        self.sample_weights = self.data_valuator.get_sample_weights_tensor(
            device=self.device
        )

        self.history = {
            "epoch": [],
            "loss_total": [],
            "loss_rank": [],
            "loss_cl": [],
            "loss_game": [],
            "val_ndcg": [],
            "val_tr": [],
            "val_cov": [],
            "time_sec": [],
        }

    def save_checkpoint(self, filepath):
        """
        Saves model weights, optimizer state, and training history to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "sample_weights": self.sample_weights.cpu().numpy(),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Loads model weights, optimizer state, and training history from disk.
        """
        if not os.path.exists(filepath):
            return False
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "sample_weights" in checkpoint:
            self.sample_weights = torch.tensor(
                checkpoint["sample_weights"], dtype=torch.float32, device=self.device
            )
        return True

    def _refresh_shapley_values(self):
        """
        Executes periodic Monte-Carlo / Fast analytical Shapley refresh for G1 and G2.
        """
        if hasattr(self.model, "edge_shapley"):
            self.model.edge_shapley.compute_mc_shapley(
                self.model.user_embeds,
                self.model.item_embeds,
                self.dataset.user_train_dict,
                tail_mask=self.dataset.tail_item_mask,
            )
        if hasattr(self.model, "hyper_conv"):
            self.model.hyper_conv.compute_group_shapley(
                self.dataset.hyperedges,
                self.model.item_embeds,
                tail_mask=self.dataset.tail_item_mask,
            )

    def _refresh_data_shapley(self):
        """
        Executes offline TMC-Shapley valuation for G3 Data-Shapley sample weights.
        """
        self.data_valuator.evaluate_sample_shapley(
            self.model, self.dataset, self.edge_index, self.topo_norm
        )
        self.sample_weights = self.data_valuator.get_sample_weights_tensor(
            device=self.device
        )

    def train_epoch(self):
        """
        Executes one training epoch with sampled softmax negatives.
        100% bounds-safe on PyTorch Apple Metal MPS / CUDA.
        """
        self.model.train()
        num_train_edges = len(self.dataset.train_edges)
        perm = torch.randperm(num_train_edges, device=self.device)

        total_l = 0.0
        total_rank = 0.0
        total_cl = 0.0
        total_game = 0.0
        num_batches = 0

        for start_idx in range(0, num_train_edges, self.batch_size):
            batch_indices = perm[start_idx : start_idx + self.batch_size]

            batch_u_raw = self.edge_index[0, batch_indices]
            batch_pos_raw = self.edge_index[1, batch_indices] - self.dataset.num_users

            batch_u = torch.clamp(batch_u_raw, 0, self.dataset.num_users - 1)
            batch_pos = torch.clamp(batch_pos_raw, 0, self.dataset.num_items - 1)

            batch_negs = torch.randint(
                0,
                self.dataset.num_items,
                (len(batch_u), self.num_negs),
                device=self.device,
            )

            batch_weights = self.sample_weights[batch_indices]

            self.optimizer.zero_grad()

            final_u, final_i, att_logits = self.model(
                self.edge_index, self.topo_norm, getattr(self.dataset, "hyperedges", None)
            )

            if self.loss_fn is not None:
                l_total, l_rank, l_cl, l_game = self.loss_fn(
                    self.model,
                    final_u,
                    final_i,
                    att_logits,
                    batch_u,
                    batch_pos,
                    batch_negs,
                    batch_weights,
                    self.edge_index,
                )
            else:
                pos_scores = (final_u[batch_u] * final_i[batch_pos]).sum(dim=-1)
                neg_scores = (
                    final_u[batch_u] * final_i[batch_negs[:, 0]]
                ).sum(dim=-1)
                l_total = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
                l_rank = l_total
                l_cl = torch.tensor(0.0)
                l_game = torch.tensor(0.0)

            l_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_l += l_total.item()
            total_rank += l_rank.item()
            total_cl += l_cl.item()
            total_game += l_game.item()
            num_batches += 1

        return (
            total_l / max(1, num_batches),
            total_rank / max(1, num_batches),
            total_cl / max(1, num_batches),
            total_game / max(1, num_batches),
        )

    def train(
        self,
        epochs=25,
        verbose=True,
        checkpoint_dir=None,
        model_name="Model",
        dataset_name=None,
        resume=True,
    ):
        """
        Full amortized training schedule with automatic checkpoint saving/loading.
        If a checkpoint exists and resume=True, training is skipped and history is loaded instantly.
        """
        if dataset_name is None:
            dataset_name = getattr(self.dataset, "dataset_name", "Dataset")

        # Define checkpoint file path
        ckpt_path = None
        if checkpoint_dir:
            safe_model = model_name.replace(" ", "_").replace("/", "_")
            safe_ds = dataset_name.replace(" ", "_").replace("/", "_")
            ckpt_path = os.path.join(checkpoint_dir, f"{safe_model}_{safe_ds}_final.pt")

            # Check if we can resume from existing checkpoint
            if resume and os.path.exists(ckpt_path):
                success = self.load_checkpoint(ckpt_path)
                if success:
                    if verbose:
                        print(f"📦 Checkpoint loaded for [{model_name}] on [{dataset_name}]. Skipping re-training!")
                    return self.history

        best_val_ndcg = 0.0
        start_time = time.time()

        if hasattr(self.model, "svd_view"):
            self.model.svd_view.compute_svd_view(self.dataset.train_edges)

        for epoch in range(1, epochs + 1):
            ep_start = time.time()

            if epoch % self.shapley_refresh_period == 1:
                self._refresh_shapley_values()

            if epoch % self.data_shapley_period == 1 and epoch > 1:
                self._refresh_data_shapley()

            l_tot, l_rank, l_cl, l_game = self.train_epoch()
            ep_dur = time.time() - ep_start

            val_metrics = compute_all_metrics(
                self.model,
                self.dataset,
                self.dataset.user_val_dict,
                k=20,
                device=self.device,
                edge_index=self.edge_index,
                topo_norm=self.topo_norm,
            )

            self.history["epoch"].append(epoch)
            self.history["loss_total"].append(l_tot)
            self.history["loss_rank"].append(l_rank)
            self.history["loss_cl"].append(l_cl)
            self.history["loss_game"].append(l_game)
            self.history["val_ndcg"].append(val_metrics["NDCG@20"])
            self.history["val_tr"].append(val_metrics["TR@20"])
            self.history["val_cov"].append(val_metrics["Coverage@20"])
            self.history["time_sec"].append(ep_dur)

            if val_metrics["NDCG@20"] > best_val_ndcg:
                best_val_ndcg = val_metrics["NDCG@20"]
                if ckpt_path:
                    best_path = ckpt_path.replace("_final.pt", "_best.pt")
                    self.save_checkpoint(best_path)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"[Epoch {epoch:2d}/{epochs}] "
                    f"Loss: {l_tot:.4f} (Rank: {l_rank:.4f}, CL: {l_cl:.4f}, Game: {l_game:.4f}) | "
                    f"Val NDCG@20: {val_metrics['NDCG@20']:.4f} | "
                    f"TR@20: {val_metrics['TR@20']:.4f} | "
                    f"Cov@20: {val_metrics['Coverage@20']:.4f} | "
                    f"Time: {ep_dur:.2f}s"
                )

        # Save final completed checkpoint
        if ckpt_path:
            self.save_checkpoint(ckpt_path)
            if verbose:
                print(f"📦 Checkpoint saved: {ckpt_path}")

        return self.history
