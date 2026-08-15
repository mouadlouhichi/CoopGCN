#!/usr/bin/env python3
"""Calibrate LightGCN before any cross-model experiment is accepted."""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coopgcn import load_benchmark_dataset, LightGCN, CoopGCNTrainer, compute_all_metrics

REFERENCES = {"Gowalla": 0.1554, "Yelp2018": 0.0530, "Amazon-Book": 0.0315}
OUT = ROOT / "experiments" / "expected" / "baseline_calibration.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REFERENCES))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold-ratio", type=float, default=2/3)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rows=[]
    for name in args.datasets:
        ds=load_benchmark_dataset(name, seed=args.seed, cache_dir=str(ROOT/"data"), download=True)
        model=LightGCN(ds.num_users,ds.num_items,embed_dim=64,num_layers=3)
        trainer=CoopGCNTrainer(model,ds,loss_fn=None,lr=0.001,weight_decay=1e-4,
                               batch_size=2048,device=args.device)
        history=trainer.train(epochs=args.epochs,patience=50,resume=False,
                              checkpoint_dir=str(ROOT/"results"/"calibration"),
                              model_name="LightGCN-calibration",dataset_name=name)
        m=compute_all_metrics(trainer.model,ds,ds.user_test_dict,k=20,device=trainer.device,
                              edge_index=trainer.edge_index,topo_norm=trainer.topo_norm)
        ref=REFERENCES[name];ratio=m["NDCG@20"]/ref;passed=ratio>=args.threshold_ratio
        rows.append({"dataset":name,"model":"LightGCN","seed":args.seed,"split_id":"retained",
                     "retained_ndcg_20":f'{m["NDCG@20"]:.6f}',
                     "published_reference_ndcg_20":ref,
                     "retained_to_reference_ratio":f"{ratio:.4f}",
                     "calibration_threshold_ratio":args.threshold_ratio,
                     "calibration_pass":str(passed).lower(),
                     "status":"complete" if passed else "failed_calibration",
                     "notes":f"epochs={len(history.get('epoch',[]))};same_evaluator=true"})
    with OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(f"wrote {OUT}")
    if not all(r["calibration_pass"]=="true" for r in rows):
        raise SystemExit("Calibration failed; do not run comparative claims.")

if __name__=="__main__":main()
