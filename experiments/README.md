# Confirmatory experiment records

This directory is CSV-first: schemas are created before experiments, and
artifacts are generated only from recorded rows.

```bash
python scripts/build_confirmatory_artifacts.py
```

## Status semantics

- `pending`: experiment has not been run;
- `requires_rerun` / `failed_calibration`: existing result is not accepted;
- `diagnostic_complete`: arithmetic/code audit complete, but not a confirmatory
  model result;
- `complete`: confirmatory row is eligible for a scientific table after audit.

The builder treats only `complete` as completed evidence. Generated SVGs for
pending records are workflow status figures, not paper figures.

## Required execution order

1. Run `python scripts/run_baseline_calibration.py`. It exits non-zero unless
   every sparse LightGCN ratio reaches the calibration threshold and writes
   `baseline_calibration.csv`.
2. Complete `component_ablation.csv`, starting with `frozen_norm`.
3. Complete `multiseed_main.csv`.
4. Complete `neighbor_cap_sweep.csv`.
5. Complete `proxy_shapley_validation.csv`.
6. Run runtime, corruption, and attribution studies only if those claims are
   retained.

Never replace blank metrics with expected or projected values, and never
promote legacy diagnostic rows to `complete` by hand.
