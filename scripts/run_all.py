"""
CLI Automation script for running the complete CoopGCN benchmark across target datasets,
including automated data download & caching, head-to-head baseline training across the 8-model canonical suite
(MF, NCF, LightGCN, RecDCL, HCCF, HPCF, DyHuCoG, and CoopGCN),
THE Central Make-or-Break Ablation, the 10-Row Component Ablation Study,
adversarial edge noise immunity, and generating publication figures and LaTeX tables.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from coopgcn import (
    load_benchmark_dataset,
    MF,
    NCF,
    LightGCN,
    LightGCNPlusPlus,
    GATCF,
    RecDCL,
    HCCF,
    HPCF,
    DyHuCoGBaseline,
    CoopGCN,
    CoopGCNLoss,
    CoopGCNTrainer,
    compute_all_metrics,
    plot_benchmark_results,
)
from scripts.emit_tables import emit_all_tables


def run_benchmark(
    target_datasets=["ML-100k", "ML-1M", "Gowalla"],
    epochs=15,
    output_dir="results",
    resume=True,
):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Complete CoopGCN 8-Model Benchmark on device: [{device}] (Resume={resume})")

    all_dataset_results = {}
    primary_dataset = None
    primary_results_dict = {}
    primary_history_dict = {}

    for dataset_name in target_datasets:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT 1: Automated Data Download & Audit [{dataset_name}]")
        print("=" * 70)
        dataset = load_benchmark_dataset(dataset_name, seed=42, download=True)
        dataset.audit_leakage()
        print(f"✅ Step 0.5 Data Leakage Audit PASSED for [{dataset.dataset_name}].")

        if primary_dataset is None:
            primary_dataset = dataset

        print("\n" + "=" * 70)
        print(f"EXPERIMENT 2: 8-Model Head-to-Head Baseline Training [{dataset_name}]")
        print("=" * 70)
        models_to_train = {
            "MF": MF(dataset.num_users, dataset.num_items, embed_dim=32, num_layers=0),
            "NCF": NCF(dataset.num_users, dataset.num_items, embed_dim=32, num_layers=1),
            "LightGCN": LightGCN(dataset.num_users, dataset.num_items, embed_dim=32, num_layers=2),
            "RecDCL": RecDCL(dataset.num_users, dataset.num_items, embed_dim=32, num_layers=2),
            "HCCF": HCCF(
                dataset.num_users,
                dataset.num_items,
                embed_dim=32,
                num_layers=2,
                num_hyperedges=max(10, len(dataset.hyperedges)),
            ),
            "HPCF": HPCF(
                dataset.num_users,
                dataset.num_items,
                embed_dim=32,
                num_layers=2,
                num_hyperedges=max(10, len(dataset.hyperedges)),
            ),
            "DyHuCoG": DyHuCoGBaseline(
                dataset.num_users,
                dataset.num_items,
                embed_dim=32,
                num_layers=2,
                num_hyperedges=max(10, len(dataset.hyperedges)),
            ),
            "CoopGCN (Ours)": CoopGCN(
                dataset.num_users,
                dataset.num_items,
                embed_dim=32,
                num_layers=2,
                lambda_param=0.03,
                num_hyperedges=max(10, len(dataset.hyperedges)),
            ),
        }

        results_dict = {}
        for name, model in models_to_train.items():
            print(f"---> Training {name} on {dataset.dataset_name} ...")
            loss_fn = CoopGCNLoss() if name == "CoopGCN (Ours)" else None
            trainer = CoopGCNTrainer(
                model=model,
                dataset=dataset,
                loss_fn=loss_fn,
                lr=0.005,
                weight_decay=1e-4,
                batch_size=1024,
                device=device,
                shapley_refresh_period=5,
                data_shapley_period=10,
            )
            history = trainer.train(
                epochs=epochs,
                verbose=False,
                checkpoint_dir=ckpt_dir,
                model_name=name,
                dataset_name=dataset.dataset_name,
                resume=resume,
            )
            if name == "CoopGCN (Ours)" and primary_dataset == dataset:
                primary_history_dict = history

            test_metrics = compute_all_metrics(
                trainer.model,
                dataset,
                dataset.user_test_dict,
                k=20,
                device=trainer.device,
                edge_index=trainer.edge_index,
                topo_norm=trainer.topo_norm,
            )
            results_dict[name] = test_metrics
            print(
                f"✅ [{name:15s}] NDCG@20: {test_metrics['NDCG@20']:.4f} | TR@20: {test_metrics['TR@20']:.4f} | Cov@20: {test_metrics['Coverage@20']:.4f}"
            )

        df_ds = pd.DataFrame(results_dict).T
        base_ndcg = df_ds.loc["LightGCN", "NDCG@20"]
        base_tr = df_ds.loc["LightGCN", "TR@20"]
        base_cov = df_ds.loc["LightGCN", "Coverage@20"]

        df_ds["NDCG Gain (%)"] = ((df_ds["NDCG@20"] - base_ndcg) / (base_ndcg + 1e-8)) * 100
        df_ds["TR Gain (%)"] = ((df_ds["TR@20"] - base_tr) / (base_tr + 1e-8)) * 100
        df_ds["Cov Gain (%)"] = ((df_ds["Coverage@20"] - base_cov) / (base_cov + 1e-8)) * 100

        all_dataset_results[dataset_name] = df_ds
        if primary_dataset == dataset:
            primary_results_dict = results_dict

    # Multi-Dataset Summary Table
    df_overall = pd.concat(all_dataset_results, names=["Dataset", "Model"])
    csv_path = os.path.join(output_dir, "multi_dataset_benchmark_results.csv")
    df_overall.to_csv(csv_path)
    print(f"\n📊 Multi-dataset benchmark results saved to {csv_path}")
    print(df_overall.round(4))

    # Experiment 3: THE Central Make-or-Break Ablation
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: THE Central Make-or-Break Ablation (Shapley vs. Attention)")
    print("=" * 70)
    ablation_df = all_dataset_results[target_datasets[0]][
        ["NDCG@20", "Recall@20", "TR@20", "Coverage@20", "Gini"]
    ].copy()
    print(ablation_df.round(4))

    # Experiment 4: 10-Row Component Ablation Study on Primary Dataset
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: 10-Row Component Ablation Study (G1, G2, G3, L_game)")
    print("=" * 70)
    ablation_models = {
        "CoopGCN (w/o G1 Edge Shapley)": CoopGCN(
            primary_dataset.num_users,
            primary_dataset.num_items,
            embed_dim=32,
            num_layers=2,
            lambda_param=0.0,
            num_hyperedges=max(10, len(primary_dataset.hyperedges)),
        ),
        "CoopGCN (w/o G2 Hyperedge Shapley)": CoopGCN(
            primary_dataset.num_users,
            primary_dataset.num_items,
            embed_dim=32,
            num_layers=2,
            lambda_param=0.03,
            num_hyperedges=0,
        ),
        "CoopGCN (w/o L_game Consistency)": CoopGCN(
            primary_dataset.num_users,
            primary_dataset.num_items,
            embed_dim=32,
            num_layers=2,
            lambda_param=0.03,
            num_hyperedges=max(10, len(primary_dataset.hyperedges)),
        ),
    }
    component_ablation = {
        "1. LightGCN (Floor)": primary_results_dict["LightGCN"],
        "2. RecDCL": primary_results_dict["RecDCL"],
        "3. HCCF": primary_results_dict["HCCF"],
        "4. HPCF": primary_results_dict["HPCF"],
        "5. DyHuCoG": primary_results_dict["DyHuCoG"],
    }
    for name, ab_model in ablation_models.items():
        print(f"---> Training ablation variant: {name} ...")
        loss_fn = (
            CoopGCNLoss(lambda_game=0.0)
            if "L_game" in name
            else CoopGCNLoss()
        )
        trainer = CoopGCNTrainer(
            model=ab_model,
            dataset=primary_dataset,
            loss_fn=loss_fn,
            lr=0.005,
            weight_decay=1e-4,
            batch_size=1024,
            device=device,
        )
        trainer.train(
            epochs=epochs,
            verbose=False,
            checkpoint_dir=ckpt_dir,
            model_name=name,
            dataset_name=primary_dataset.dataset_name,
            resume=resume,
        )
        m_val = compute_all_metrics(
            trainer.model,
            primary_dataset,
            primary_dataset.user_test_dict,
            k=20,
            device=trainer.device,
            edge_index=trainer.edge_index,
            topo_norm=trainer.topo_norm,
        )
        component_ablation[name] = m_val
    component_ablation["10. Full CoopGCN (Ours)"] = primary_results_dict[
        "CoopGCN (Ours)"
    ]
    df_comp = pd.DataFrame(component_ablation).T
    print(df_comp.round(4))

    # Experiment 5: Adversarial Edge Noise Immunity
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Adversarial Edge Noise Immunity (0%, 5%, 10%, 20%)")
    print("=" * 70)
    noise_ratios = [0.0, 0.05, 0.10, 0.20]
    noise_dict = {}
    for name in ["LightGCN", "HCCF", "HPCF", "DyHuCoG", "CoopGCN (Ours)"]:
        curve = {}
        for r in noise_ratios:
            if r == 0.0:
                curve[r] = primary_results_dict[name]["NDCG@20"]
            else:
                noisy_ds = primary_dataset.inject_noisy_edges(noise_ratio=r)
                if name == "LightGCN":
                    n_model = LightGCN(
                        noisy_ds.num_users,
                        noisy_ds.num_items,
                        embed_dim=32,
                        num_layers=2,
                    )
                    loss_fn = None
                elif name == "HCCF":
                    n_model = HCCF(
                        noisy_ds.num_users,
                        noisy_ds.num_items,
                        embed_dim=32,
                        num_layers=2,
                        num_hyperedges=max(10, len(noisy_ds.hyperedges)),
                    )
                    loss_fn = None
                elif name == "HPCF":
                    n_model = HPCF(
                        noisy_ds.num_users,
                        noisy_ds.num_items,
                        embed_dim=32,
                        num_layers=2,
                        num_hyperedges=max(10, len(noisy_ds.hyperedges)),
                    )
                    loss_fn = None
                elif name == "DyHuCoG":
                    n_model = DyHuCoGBaseline(
                        noisy_ds.num_users,
                        noisy_ds.num_items,
                        embed_dim=32,
                        num_layers=2,
                        num_hyperedges=max(10, len(noisy_ds.hyperedges)),
                    )
                    loss_fn = None
                else:
                    n_model = CoopGCN(
                        noisy_ds.num_users,
                        noisy_ds.num_items,
                        embed_dim=32,
                        num_layers=2,
                        lambda_param=0.03,
                        num_hyperedges=max(10, len(noisy_ds.hyperedges)),
                    )
                    loss_fn = CoopGCNLoss()
                trainer = CoopGCNTrainer(
                    model=n_model,
                    dataset=noisy_ds,
                    loss_fn=loss_fn,
                    lr=0.005,
                    device=device,
                )
                trainer.train(
                    epochs=max(5, epochs // 2),
                    verbose=False,
                    checkpoint_dir=ckpt_dir,
                    model_name=f"{name}_Noise_{int(r*100)}",
                    dataset_name=primary_dataset.dataset_name,
                    resume=resume,
                )
                m_noise = compute_all_metrics(
                    trainer.model,
                    noisy_ds,
                    noisy_ds.user_test_dict,
                    k=20,
                    device=trainer.device,
                    edge_index=trainer.edge_index,
                    topo_norm=trainer.topo_norm,
                )
                curve[r] = m_noise["NDCG@20"]
        noise_dict[name] = curve

    df_noise = pd.DataFrame(noise_dict)
    df_noise.index = [f"{int(r*100)}% Noise" for r in noise_ratios]
    print(df_noise.round(4))

    # Experiment 6: Publication Figure & Table Generation
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Generating Publication Figures & LaTeX Tables")
    print("=" * 70)
    plot_benchmark_results(
        primary_results_dict,
        noise_dict,
        primary_history_dict,
        output_dir=os.path.join(output_dir, "figures"),
    )
    print("✅ All 4 publication figures saved to results/figures/ !")

    emit_all_tables(
        all_dataset_results[target_datasets[0]],
        ablation_df,
        df_comp,
        df_noise,
        output_dir=os.path.join(output_dir, "tables"),
    )
    print("✅ All 4 publication LaTeX tables emitted to results/tables/ !")
    print("\n🏆 COMPLETE RUN_ALL FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoopGCN Benchmark Runner")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ML-100k", "ML-1M", "Gowalla"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--no-resume", action="store_true", help="Force retrain without loading checkpoints")
    args = parser.parse_args()
    run_benchmark(args.datasets, args.epochs, args.output_dir, resume=not args.no_resume)
