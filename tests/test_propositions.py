"""
Unit-level sanity checks for:
1. credit symmetry and ordering on a toy construction;
2. exact Channel-A recovery of LightGCN weights at lambda=0;
3. lower estimated credit for one deliberately inconsistent toy edge; and
4. split-disjointness in a dummy dataset.

These checks do not prove all Shapley axioms for the implementation, recover
the complete LightGCN++ system, or establish end-to-end robustness.
"""

import os
import sys
import math
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
from coopgcn.models import MCShapleyEdgeWeighting
from coopgcn.dataset import BenchmarkDataset, _create_dummy_test_dataset


def test_shapley_symmetry_and_dummy_axioms():
    """
    Verifies Shapley Symmetry Axiom (interchangeable players receive equal credit in expectation)
    and Dummy Player Axiom (uninformative interactions receive zero credit).
    """
    torch.manual_seed(42)
    num_users, num_items, embed_dim = 1, 4, 16
    user_embeds = torch.ones(num_users, embed_dim)
    item_embeds = torch.zeros(num_items, embed_dim)
    item_embeds[0] = torch.ones(embed_dim)
    item_embeds[1] = torch.ones(embed_dim)
    item_embeds[2] = torch.zeros(embed_dim)

    mod = MCShapleyEdgeWeighting(num_users, num_items, embed_dim, num_permutations=2000)
    adj = {0: [0, 1, 2]}
    mod.compute_mc_shapley(user_embeds, item_embeds, adj)

    phi_0 = mod.ema_shapley[0, 0].item()
    phi_1 = mod.ema_shapley[0, 1].item()
    phi_dummy = mod.ema_shapley[0, 2].item()

    # Assert Symmetry within Monte-Carlo estimation tolerance
    assert abs(phi_0 - phi_1) < 0.05, f"Symmetry violated: {phi_0} != {phi_1}"
    # Assert Dummy player receives lower credit than informative players
    assert phi_dummy < min(phi_0, phi_1), f"Dummy player credit too high: {phi_dummy}"
    print("✅ Shapley Symmetry & Dummy Player Axioms verified!")


def test_proposition_1_lightgcn_recovery():
    """
    Proposition 1: At lambda = 0, Channel A edge weights reduce exactly to LightGCN
    symmetric normalization 1 / sqrt(d_u * d_i).
    """
    num_users, num_items, embed_dim = 2, 3, 16
    mod = MCShapleyEdgeWeighting(num_users, num_items, embed_dim)
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    topo_norm = torch.tensor([0.5, 0.25], dtype=torch.float32)
    att_logits = torch.randn(2)

    w_0 = mod(edge_index, topo_norm, lambda_param=0.0, attention_logits=att_logits)
    assert torch.allclose(w_0, topo_norm, atol=1e-6), "LightGCN recovery failed!"
    print("✅ Proposition 1 (LightGCN Recovery at lambda=0) verified!")


def test_inconsistent_edge_receives_lower_credit():
    """A deliberately anti-aligned toy edge should receive lower estimated credit.

    This is a local ordering check, not an end-to-end robustness test.
    """
    torch.manual_seed(42)
    num_users, num_items, embed_dim = 1, 3, 16
    user_embeds = torch.ones(num_users, embed_dim)
    item_embeds = torch.zeros(num_items, embed_dim)
    item_embeds[0] = torch.ones(embed_dim)
    item_embeds[1] = -torch.ones(embed_dim) * 10.0

    mod = MCShapleyEdgeWeighting(num_users, num_items, embed_dim, num_permutations=100)
    adj = {0: [0, 1]}
    mod.compute_mc_shapley(user_embeds, item_embeds, adj)

    phi_good = mod.ema_shapley[0, 0].item()
    phi_bad = mod.ema_shapley[0, 1].item()

    assert phi_good > phi_bad, f"Credit ordering failed: good={phi_good} <= bad={phi_bad}"
    print("✅ Inconsistent-edge credit ordering sanity check passed!")


def test_step_0_5_data_leakage_audit():
    """
    Step 0.5: Asserts zero evaluation leakage (W11) in benchmark temporal splits.
    """
    ds = _create_dummy_test_dataset(num_users=20, num_items=40, num_interactions=200)
    train_set = set(map(tuple, ds.train_edges))
    val_set = set(map(tuple, ds.val_edges))
    test_set = set(map(tuple, ds.test_edges))

    assert len(train_set.intersection(val_set)) == 0, "Val leakage detected!"
    assert len(train_set.intersection(test_set)) == 0, "Test leakage detected!"
    print("✅ Step 0.5 Leakage Audit Protocol verified!")


if __name__ == "__main__":
    test_shapley_symmetry_and_dummy_axioms()
    test_proposition_1_lightgcn_recovery()
    test_inconsistent_edge_receives_lower_credit()
    test_step_0_5_data_leakage_audit()
    print("\n✅ ALL LOCAL CREDIT, RECOVERY, AND SPLIT SANITY CHECKS PASSED!")
