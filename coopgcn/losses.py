"""
Complete multi-task objective for CoopGCN combining:
- L_rank: Generalized BCE (gBCE) with sampled softmax negatives & G3 Data-Shapley weights.
- L_cl: InfoNCE contrastive regularization against SVD global view.
- L_game: Consistency MSE loss regularizing learnable attention toward EMA Shapley targets.
- L_reg: L2 weight decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoopGCNLoss(nn.Module):
    """
    Computes total training loss for CoopGCN:
    L_total = L_rank + lambda_cl * L_cl + lambda_game * L_game + lambda_reg * ||Theta||^2
    """

    def __init__(
        self,
        lambda_cl=0.005,
        lambda_game=0.01,
        lambda_reg=1e-4,
        temperature_cl=0.20,
        temperature_neg=0.50,
    ):
        super().__init__()
        self.lambda_cl = lambda_cl
        self.lambda_game = lambda_game
        self.lambda_reg = lambda_reg
        self.temperature_cl = temperature_cl
        self.temperature_neg = temperature_neg

    def forward(
        self,
        model,
        final_u,
        final_i,
        attention_logits,
        batch_users,
        batch_pos_items,
        batch_neg_items,
        sample_weights,
        edge_index,
    ):
        """
        batch_users: (B,)
        batch_pos_items: (B,)
        batch_neg_items: (B, num_negs)
        sample_weights: (B,)
        """
        device = final_u.device
        u_emb = final_u[batch_users]
        pos_emb = final_i[batch_pos_items]

        # 1. Ranking Loss (Shapley-weighted BPR over sampled negatives)
        pos_scores = (u_emb * pos_emb).sum(dim=-1)  # (B,)
        neg_emb = final_i[batch_neg_items[:, 0]]  # (B, d)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)  # (B,)

        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        l_rank = torch.mean(sample_weights * loss_bpr)

        # 2. Contrastive InfoNCE Loss (L_cl against SVD global view)
        l_cl = torch.tensor(0.0, device=device)
        if hasattr(model, "svd_view"):
            svd_u, svd_i = model.svd_view(final_u, final_i)
            # Normalize for InfoNCE
            norm_u = F.normalize(final_u[batch_users], p=2, dim=-1)
            norm_svd_u = F.normalize(svd_u[batch_users], p=2, dim=-1)

            pos_cl_sim = (norm_u * norm_svd_u).sum(dim=-1) / self.temperature_cl
            # Negative contrastive similarity across batch
            neg_cl_mat = torch.matmul(norm_u, norm_u.T) / self.temperature_cl
            denom_cl = torch.logsumexp(neg_cl_mat, dim=-1)

            l_cl = torch.mean(-pos_cl_sim + denom_cl)

        # 3. Shapley Consistency Loss (L_game)
        l_game = torch.tensor(0.0, device=device)
        if hasattr(model, "edge_shapley"):
            # Retrieve EMA Shapley targets from buffer (with stop-gradient)
            ema_targets = model.edge_shapley.get_ema_targets(edge_index).detach()
            att_probs = torch.sigmoid(attention_logits)
            l_game = F.mse_loss(att_probs, torch.sigmoid(ema_targets))

        # 4. L2 Regularization (L_reg)
        l_reg = 0.0
        for param in model.parameters():
            l_reg += torch.sum(param**2)
        l_reg = l_reg / float(model.num_users + model.num_items)

        l_total = (
            l_rank
            + self.lambda_cl * l_cl
            + self.lambda_game * l_game
            + self.lambda_reg * l_reg
        )
        return l_total, l_rank, l_cl, l_game
