"""
100% REAL benchmark dataset loader, automated downloader & caching for CoopGCN.
Loads MovieLens-100K/1M with temporal splits and a pinned LightGCN snapshot for
Gowalla, Yelp2018 and Amazon-Book (which has no timestamps). Enforces train vs
validation/test pair disjointness; the dummy generator is reserved for tests.
"""

import os
import urllib.request
import zipfile
import random
import numpy as np
import torch
from collections import defaultdict


class BenchmarkDataset:
    """
    Holds a user-item interaction graph, dataset-specific train/val/test splits,
    training-degree tail labels, and training-only hyperedges.
    Enforces strict Data Leakage Audit protocol (Step 0.5).
    """

    def __init__(
        self,
        num_users,
        num_items,
        train_edges,
        val_edges,
        test_edges,
        hyperedges,
        item_degrees=None,
        dataset_name="Dataset",
    ):
        self.dataset_name = dataset_name
        self.num_users = num_users
        self.num_items = num_items
        self.train_edges = np.array(train_edges, dtype=np.int64)
        self.val_edges = np.array(val_edges, dtype=np.int64)
        self.test_edges = np.array(test_edges, dtype=np.int64)
        self.hyperedges = hyperedges  # list of lists of item_ids

        # Build adjacency lists for fast neighborhood lookup
        self.user_train_dict = defaultdict(list)
        self.item_train_dict = defaultdict(list)
        for u, i in self.train_edges:
            self.user_train_dict[int(u)].append(int(i))
            self.item_train_dict[int(i)].append(int(u))

        self.user_test_dict = defaultdict(list)
        for u, i in self.test_edges:
            self.user_test_dict[int(u)].append(int(i))

        self.user_val_dict = defaultdict(list)
        for u, i in self.val_edges:
            self.user_val_dict[int(u)].append(int(i))

        # Compute degrees strictly from train_edges to avoid W11 evaluation leakage
        if item_degrees is None:
            self.item_degrees = np.zeros(num_items, dtype=np.int64)
            for _, i in self.train_edges:
                self.item_degrees[int(i)] += 1
        else:
            self.item_degrees = item_degrees

        self.user_degrees = np.zeros(num_users, dtype=np.int64)
        for u, _ in self.train_edges:
            self.user_degrees[int(u)] += 1

        # Determine tail items (bottom 80% degree threshold calculated on train degrees ONLY)
        cutoff_val = np.percentile(self.item_degrees, 80)
        self.tail_item_mask = torch.tensor(
            self.item_degrees <= cutoff_val, dtype=torch.bool
        )

        # Audit data leakage
        self.audit_leakage()

    def audit_leakage(self):
        """
        Step 0.5 — Leakage Audit Protocol.
        Verifies that test/val interactions do not leak into training graph.
        """
        train_set = set(map(tuple, self.train_edges))
        val_set = set(map(tuple, self.val_edges))
        test_set = set(map(tuple, self.test_edges))

        val_leak = train_set.intersection(val_set)
        test_leak = train_set.intersection(test_set)
        assert len(val_leak) == 0, f"Leakage detected: {len(val_leak)} val edges in train!"
        assert len(test_leak) == 0, f"Leakage detected: {len(test_leak)} test edges in train!"

    def get_sparse_adjacency(self, device="cpu"):
        """
        Returns symmetric normalized adjacency matrix in PyTorch sparse format for GCN propagation.
        """
        u_indices = self.train_edges[:, 0]
        i_indices = self.train_edges[:, 1] + self.num_users

        row = np.concatenate([u_indices, i_indices])
        col = np.concatenate([i_indices, u_indices])
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)

        num_nodes = self.num_users + self.num_items
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        deg.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float32))

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm_values = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        return edge_index, norm_values, deg[: self.num_users], deg[self.num_users :]

    def inject_noisy_edges(self, noise_ratio=0.10, seed=42):
        """
        Injects random edges into the training graph for a future corruption study.
        Returns a new BenchmarkDataset with injected noisy edges.
        """
        np.random.seed(seed)
        num_noise = int(len(self.train_edges) * noise_ratio)
        noisy_u = np.random.randint(0, self.num_users, size=num_noise)
        noisy_i = np.random.randint(0, self.num_items, size=num_noise)
        noisy_edges = np.stack([noisy_u, noisy_i], axis=1)

        val_test_set = set(map(tuple, self.val_edges)).union(set(map(tuple, self.test_edges)))
        filtered_noisy = np.array([e for e in noisy_edges if tuple(e) not in val_test_set])
        new_train_edges = np.vstack([self.train_edges, filtered_noisy])
        return BenchmarkDataset(
            self.num_users,
            self.num_items,
            new_train_edges,
            self.val_edges,
            self.test_edges,
            self.hyperedges,
            dataset_name=f"{self.dataset_name}-{int(noise_ratio*100)}Noise",
        )


def _download_movielens_100k(cache_dir="./data"):
    """
    Downloads and parses MovieLens-100K dataset from official GroupLens repository.
    Constructs genre-based hyperedges for Channel B (G2).
    All ratings are treated as implicit collaborative filtering interaction edges.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "ml-100k.zip")
    extract_dir = os.path.join(cache_dir, "ml-100k")

    if not os.path.exists(extract_dir):
        print(f"📥 Downloading MovieLens-100K from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        print("✅ Download and extraction complete.")

    data_path = os.path.join(cache_dir, "ml-100k", "u.data")
    item_path = os.path.join(cache_dir, "ml-100k", "u.item")

    interactions = []
    with open(data_path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, r, ts = map(int, line.strip().split())
            interactions.append((u - 1, i - 1, ts))

    interactions.sort(key=lambda x: x[2])
    edges = np.array([[x[0], x[1]] for x in interactions], dtype=np.int64)

    num_users = int(np.max(edges[:, 0]) + 1)
    num_items = int(np.max(edges[:, 1]) + 1)

    num_edges = len(edges)
    train_end = int(num_edges * 0.70)
    val_end = int(num_edges * 0.80)

    train_edges = edges[:train_end]
    val_edges = edges[train_end:val_end]
    test_edges = edges[val_end:]

    train_set = set(map(tuple, train_edges))
    val_edges = np.array([e for e in val_edges if tuple(e) not in train_set])
    test_edges = np.array([e for e in test_edges if tuple(e) not in train_set])

    items_in_train = set(train_edges[:, 1])
    genre_hyperedges = defaultdict(list)
    if os.path.exists(item_path):
        with open(item_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                item_id = int(parts[0]) - 1
                if item_id in items_in_train and item_id < num_items:
                    genres = [idx for idx, val in enumerate(parts[5:]) if val == "1"]
                    for g in genres:
                        genre_hyperedges[g].append(item_id)
    hyperedges = [items for items in genre_hyperedges.values() if len(items) >= 2]

    return BenchmarkDataset(
        num_users=num_users,
        num_items=num_items,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        hyperedges=hyperedges,
        dataset_name="ML-100k",
    )


def _download_movielens_1m(cache_dir="./data"):
    """
    Downloads and parses MovieLens-1M dataset from official GroupLens repository.
    Constructs genre-based hyperedges for Channel B (G2).
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "ml-1m.zip")
    extract_dir = os.path.join(cache_dir, "ml-1m")

    if not os.path.exists(extract_dir):
        print(f"📥 Downloading MovieLens-1M from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        print("✅ Download and extraction complete.")

    data_path = os.path.join(cache_dir, "ml-1m", "ratings.dat")
    item_path = os.path.join(cache_dir, "ml-1m", "movies.dat")

    interactions = []
    with open(data_path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, r, ts = map(int, line.strip().split("::"))
            interactions.append((u - 1, i - 1, ts))

    unique_items = {item_id: idx for idx, item_id in enumerate(sorted(set(x[1] for x in interactions)))}
    interactions = [(x[0], unique_items[x[1]], x[2]) for x in interactions]

    interactions.sort(key=lambda x: x[2])
    edges = np.array([[x[0], x[1]] for x in interactions], dtype=np.int64)

    num_users = int(np.max(edges[:, 0]) + 1)
    num_items = int(np.max(edges[:, 1]) + 1)

    num_edges = len(edges)
    train_end = int(num_edges * 0.70)
    val_end = int(num_edges * 0.80)

    train_edges = edges[:train_end]
    val_edges = edges[train_end:val_end]
    test_edges = edges[val_end:]

    train_set = set(map(tuple, train_edges))
    val_edges = np.array([e for e in val_edges if tuple(e) not in train_set])
    test_edges = np.array([e for e in test_edges if tuple(e) not in train_set])

    items_in_train = set(train_edges[:, 1])
    genre_hyperedges = defaultdict(list)
    if os.path.exists(item_path):
        with open(item_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                orig_id = int(parts[0]) - 1
                if orig_id in unique_items:
                    new_id = unique_items[orig_id]
                    if new_id in items_in_train and new_id < num_items:
                        genres = parts[2].split("|")
                        for g in genres:
                            genre_hyperedges[g].append(new_id)
    hyperedges = [items for items in genre_hyperedges.values() if len(items) >= 2]

    return BenchmarkDataset(
        num_users=num_users,
        num_items=num_items,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        hyperedges=hyperedges,
        dataset_name="ML-1M",
    )


def _download_lightgcn_text_benchmark(dataset_name, cache_dir="./data"):
    """
    Downloads standard RecSys benchmark splits (Gowalla, Yelp2018, Amazon-Book)
    from official LightGCN repository.
    Constructs co-occurrence hyperedges strictly from training interactions.
    """
    name_lower = dataset_name.lower()
    # Pin the upstream benchmark snapshot used by the retained protocol.
    source_commit = "947ca2b3b1d2d3545b114145710cb06c4e57b3d2"
    base_url = (
        "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/"
        f"{source_commit}/data/{name_lower}"
    )
    os.makedirs(os.path.join(cache_dir, name_lower), exist_ok=True)
    train_file = os.path.join(cache_dir, name_lower, "train.txt")
    test_file = os.path.join(cache_dir, name_lower, "test.txt")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"📥 Downloading benchmark dataset [{dataset_name}] from {base_url} ...")
        urllib.request.urlretrieve(f"{base_url}/train.txt", train_file)
        urllib.request.urlretrieve(f"{base_url}/test.txt", test_file)
        print("✅ Download complete.")

    train_edges = []
    num_users = 0
    num_items = 0

    with open(train_file, "r") as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            u = parts[0]
            num_users = max(num_users, u + 1)
            for i in parts[1:]:
                train_edges.append((u, i))
                num_items = max(num_items, i + 1)

    test_edges_list = []
    with open(test_file, "r") as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            u = parts[0]
            num_users = max(num_users, u + 1)
            for i in parts[1:]:
                test_edges_list.append((u, i))
                num_items = max(num_items, i + 1)

    train_edges = np.array(train_edges, dtype=np.int64)
    test_edges_all = np.array(test_edges_list, dtype=np.int64)

    val_end = int(len(test_edges_all) * 0.33)
    val_edges = test_edges_all[:val_end]
    test_edges = test_edges_all[val_end:]

    train_set = set(map(tuple, train_edges))
    val_edges = np.array([e for e in val_edges if tuple(e) not in train_set])
    test_edges = np.array([e for e in test_edges if tuple(e) not in train_set])

    print(f"--> Constructing co-occurrence hyperedges strictly from train_edges [{dataset_name}]...")
    item_users_map = defaultdict(set)
    for u, i in train_edges:
        item_users_map[int(i)].add(int(u))

    hyperedges = []
    items = list(item_users_map.keys())
    random.seed(42)
    for _ in range(250):
        seed_item = random.choice(items)
        seed_users = item_users_map[seed_item]
        cluster = [seed_item]
        for _ in range(min(50, len(items))):
            cand = random.choice(items)
            if cand != seed_item and len(seed_users.intersection(item_users_map[cand])) >= 2:
                cluster.append(cand)
                if len(cluster) >= 12:
                    break
        if len(cluster) >= 3:
            hyperedges.append(cluster)

    return BenchmarkDataset(
        num_users=num_users,
        num_items=num_items,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        hyperedges=hyperedges,
        dataset_name=dataset_name,
    )


def _create_dummy_test_dataset(
    num_users=20, num_items=40, num_interactions=200, num_hyperedges=10, seed=42
):
    """
    Internal helper strictly for unit testing in memory.
    """
    np.random.seed(seed)
    train_edges = np.column_stack([
        np.random.randint(0, num_users, size=int(num_interactions * 0.7)),
        np.random.randint(0, num_items, size=int(num_interactions * 0.7))
    ])
    val_edges_raw = np.column_stack([
        np.random.randint(0, num_users, size=int(num_interactions * 0.1)),
        np.random.randint(0, num_items, size=int(num_interactions * 0.1))
    ])
    test_edges_raw = np.column_stack([
        np.random.randint(0, num_users, size=int(num_interactions * 0.2)),
        np.random.randint(0, num_items, size=int(num_interactions * 0.2))
    ])

    train_set = set(map(tuple, train_edges))
    val_edges = np.array([e for e in val_edges_raw if tuple(e) not in train_set])
    test_edges = np.array([e for e in test_edges_raw if tuple(e) not in train_set])

    hyperedges = [
        random.sample(range(num_items), min(5, num_items)) for _ in range(num_hyperedges)
    ]
    return BenchmarkDataset(
        num_users=num_users,
        num_items=num_items,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        hyperedges=hyperedges,
        dataset_name="DummyTest",
    )


def load_benchmark_dataset(
    dataset_name="ML-100k", seed=42, cache_dir="./data", download=True
):
    """
    Downloads and loads benchmark dataset across all 5 target datasets:
    ML-100k, ML-1M, Gowalla, Yelp2018, Amazon-Book.
    """
    name_lower = dataset_name.lower()
    if name_lower in ["ml-100k", "movielens-100k"]:
        return _download_movielens_100k(cache_dir=cache_dir)
    elif name_lower in ["ml-1m", "movielens-1m"]:
        return _download_movielens_1m(cache_dir=cache_dir)
    elif name_lower in ["gowalla", "yelp2018", "amazon-book"]:
        return _download_lightgcn_text_benchmark(dataset_name, cache_dir=cache_dir)
    else:
        print(f"ℹ️ Unrecognized dataset '{dataset_name}', loading MovieLens-100K...")
        return _download_movielens_100k(cache_dir=cache_dir)


# Alias for backward compatibility
load_or_generate_dataset = load_benchmark_dataset
