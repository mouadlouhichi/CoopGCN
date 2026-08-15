"""
Comprehensive unit test suite for CoopGCN modules, baselines, and evaluation harness.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
from coopgcn.dataset import _create_dummy_test_dataset
from coopgcn.models import (
    MCShapleyEdgeWeighting,
    ShapleyHypergraphConv,
    SVDContrastiveView,
    CoopGCN,
    LightGCN,
    LightGCNPlusPlus,
    GATCF,
    DyHuCoGBaseline,
    aggregate_symmetric_edges,
)
from coopgcn.losses import CoopGCNLoss
from coopgcn.shapley_data import TMCShapleyDataValuator
from coopgcn.evaluator import compute_all_metrics, candidate_exclusions


def test_symmetric_edge_index_is_not_double_counted():
    x = torch.tensor([[1.0], [2.0]])
    # Both directed entries are already present.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    weights = torch.ones(2)
    out = aggregate_symmetric_edges(x, edge_index, weights)
    assert torch.allclose(out, torch.tensor([[2.0], [1.0]]))


def test_mc_shapley_edge_weighting_forward():
    mod = MCShapleyEdgeWeighting(10, 20, 16)
    edge_index = torch.tensor([[0, 1], [10, 11]], dtype=torch.long)
    topo_norm = torch.tensor([0.5, 0.25], dtype=torch.float32)
    att_logits = torch.randn(2)
    out = mod(edge_index, topo_norm, 0.35, att_logits)
    assert out.shape == topo_norm.shape


def test_shapley_hypergraph_conv_forward():
    conv = ShapleyHypergraphConv(num_hyperedges=10)
    x = torch.randn(30, 16)
    hyperedges = [[0, 1, 2], [3, 4, 5]]
    out = conv(x, hyperedges, 10, 20)
    assert out.shape == x.shape


def test_svd_contrastive_view():
    svd = SVDContrastiveView(10, 20, rank=8)
    edges = np.array([[0, 0], [1, 5], [2, 10], [0, 15]])
    svd.compute_svd_view(edges)
    u_emb = torch.randn(10, 16)
    i_emb = torch.randn(20, 16)
    u_proj, i_proj = svd(u_emb, i_emb)
    assert u_proj.shape == u_emb.shape
    assert i_proj.shape == i_emb.shape


def test_full_coopgcn_forward_and_predict():
    model = CoopGCN(10, 20, embed_dim=16, num_layers=2, num_hyperedges=10)
    edge_index = torch.tensor([[0, 1, 2], [10, 11, 12]], dtype=torch.long)
    topo_norm = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    hyperedges = [[0, 1], [2, 3]]

    u_emb, i_emb, att = model(edge_index, topo_norm, hyperedges)
    assert u_emb.shape == (10, 16)
    assert i_emb.shape == (20, 16)
    assert att.shape == (3,)

    scores = model.predict([0, 1], [0, 1], edge_index, topo_norm, hyperedges)
    assert scores.shape == (2,)


def test_all_baselines_forward():
    edge_index = torch.tensor([[0, 1], [10, 11]], dtype=torch.long)
    topo_norm = torch.tensor([0.5, 0.5], dtype=torch.float32)

    for name, cls in [
        ("LightGCN", LightGCN),
        ("LightGCN++", LightGCNPlusPlus),
        ("GAT-CF", GATCF),
        ("DyHuCoG", DyHuCoGBaseline),
    ]:
        m = cls(10, 20, embed_dim=16, num_layers=2)
        u, i, _ = m(edge_index, topo_norm)
        assert u.shape == (10, 16)
        assert i.shape == (20, 16)


def test_coopgcn_loss_calculation():
    model = CoopGCN(10, 20, embed_dim=16, num_layers=1, num_hyperedges=5)
    loss_fn = CoopGCNLoss()
    edge_index = torch.tensor([[0, 1], [10, 11]], dtype=torch.long)
    topo_norm = torch.tensor([0.5, 0.5], dtype=torch.float32)
    u, i, att = model(edge_index, topo_norm)

    batch_u = torch.tensor([0, 1], dtype=torch.long)
    batch_pos = torch.tensor([0, 1], dtype=torch.long)
    batch_negs = torch.tensor([[2, 3], [4, 5]], dtype=torch.long)
    weights = torch.ones(2)

    l_tot, l_rank, l_cl, l_game = loss_fn(
        model, u, i, att, batch_u, batch_pos, batch_negs, weights, edge_index
    )
    assert not torch.isnan(l_tot)
    assert l_tot.item() >= 0.0


def test_tmc_shapley_data_valuator():
    ds = _create_dummy_test_dataset(num_users=20, num_items=40, num_interactions=100)
    model = LightGCN(ds.num_users, ds.num_items, embed_dim=16, num_layers=1)
    edge_index, topo_norm, _, _ = ds.get_sparse_adjacency()

    valuator = TMCShapleyDataValuator(len(ds.train_edges))
    valuator.evaluate_sample_shapley(model, ds, edge_index, topo_norm)
    weights = valuator.get_sample_weights_tensor()
    assert len(weights) == len(ds.train_edges)
    assert not torch.isnan(weights).any()


def test_candidate_cross_filtering():
    """Other held-out positives are masked, but overlapping current GT is not."""
    class SplitStub:
        user_train_dict = {0: [1]}
        user_val_dict = {0: [2, 3]}
        user_test_dict = {0: [3, 4]}

    ds = SplitStub()
    assert candidate_exclusions(ds, ds.user_test_dict, 0, 10) == {1, 2}
    assert candidate_exclusions(ds, ds.user_val_dict, 0, 10) == {1, 4}
    assert candidate_exclusions(ds, ds.user_test_dict, 0, 10, False) == {1}


def test_compute_all_metrics():
    ds = _create_dummy_test_dataset(num_users=20, num_items=40, num_interactions=100)
    model = LightGCN(ds.num_users, ds.num_items, embed_dim=16, num_layers=1)
    metrics = compute_all_metrics(model, ds, ds.user_test_dict, k=10)
    assert "NDCG@10" in metrics
    assert "Recall@10" in metrics
    assert "TR@10" in metrics
    assert "Coverage@10" in metrics


if __name__ == "__main__":
    test_symmetric_edge_index_is_not_double_counted()
    test_mc_shapley_edge_weighting_forward()
    test_shapley_hypergraph_conv_forward()
    test_svd_contrastive_view()
    test_full_coopgcn_forward_and_predict()
    test_all_baselines_forward()
    test_coopgcn_loss_calculation()
    test_tmc_shapley_data_valuator()
    test_candidate_cross_filtering()
    test_compute_all_metrics()
    print("\n🏆 ALL COOPGCN UNIT TESTS PASSED SUCCESSFULLY!")
