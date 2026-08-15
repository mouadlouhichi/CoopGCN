# CoopGCN — Measured Result Summary

**Protocol:** MovieLens uses a global temporal 70/10/20 split. Gowalla,
Yelp2018 and Amazon-Book use a pinned LightGCN train/test snapshot, with its
test interaction stream divided 33/67 for validation/test. Ranking is over all
items unseen in training at cut-off 20. Degrees, hyperedges and tail masks use
training edges only.

## Evidence status

The table below is an **uncalibrated single-run audit record**, not a
leaderboard. Retained LightGCN NDCG is 4–150× below published references on the
three sparse datasets. Values are **NDCG@20 / Tail Recall@20 / Coverage@20
(%)** and must not support comparative claims until calibration reruns pass.

| Model | ML-100K | ML-1M | Gowalla | Yelp2018 | Amazon-Book |
|---|---:|---:|---:|---:|---:|
| **CoopGCN** | 0.1826 / 0.0106 / 46.9 | 0.1994 / 0.0075 / 40.7 | 0.1089 / 0.0104 / 7.95 | 0.0376 / 0.0006 / 5.36 | 0.0235 / 0.0036 / 6.47 |
| LightGCN++ | 0.1627 / 0.0100 / 46.5 | 0.2054 / 0.0008 / 37.3 | 0.1092 / 0.0116 / 8.06 | 0.0359 / 0.0006 / 5.01 | 0.0222 / 0.0027 / 5.44 |
| GAT-CF | 0.1943 / 0.0022 / 19.4 | 0.2096 / 0.0001 / 10.0 | 0.0038 / 0.0005 / 10.3 | 0.0014 / 0.0003 / 0.13 | 0.0002 / 0.0000 / 0.05 |
| LightGCN | 0.1842 / 0.0036 / 27.8 | 0.2128 / 0.0000 / 9.82 | 0.0348 / 0.0000 / 0.20 | 0.0145 / 0.0000 / 0.35 | 0.0002 / 0.0000 / 0.05 |
| HPCF | 0.1747 / 0.0020 / 25.7 | 0.2141 / 0.0000 / 9.23 | 0.0278 / 0.0000 / 0.39 | 0.0110 / 0.0000 / 0.48 | 0.0003 / 0.0000 / 0.05 |
| HCCF | 0.1723 / 0.0019 / 21.2 | 0.2070 / 0.0000 / 8.23 | 0.0291 / 0.0000 / 0.35 | 0.0113 / 0.0000 / 0.25 | 0.0002 / 0.0000 / 0.05 |
| DyHuCoG | 0.1718 / 0.0026 / 20.5 | 0.2114 / 0.0000 / 9.42 | 0.0338 / 0.0000 / 0.32 | 0.0144 / 0.0000 / 0.35 | 0.0002 / 0.0000 / 0.05 |

## Audit findings

- Sparse LightGCN calibration fails against same-dataset literature references.
- CoopGCN and LightGCN++ share a learnable norm scale and similar exposure profiles.
- Cooperative proxy ranges are small relative to reported exposure differences.
- The first-32-neighbour cap leaves most ML-1M edges at a neutral target.
- No retained row establishes model superiority, component causality, robustness, explanation, or efficiency.

## Unsupported claims

The following historical values came from projections and must not be reported
as empirical findings:

- `+22.7%` NDCG superiority;
- `+45.3%` Tail Recall superiority;
- the `62.2%` G1 component attribution;
- the `5.4–5.7%` random-noise degradation curves;
- zero wall-clock inference overhead.

`main_results/expected_*.csv` is design history. See
[`main_results/ANALYSIS.md`](../main_results/ANALYSIS.md) and the manuscript's
limitations for the required confirmatory experiments.
