# Response to revised-manuscript reviews R1 and R2

We thank the reviewers for distinguishing editorial corrections from missing
experiments. The manuscript has been revised under one rule: unsupported
numbers are removed or explicitly left open; no value is inferred, projected,
or fabricated to fill an empirical gap.

## Resolved in the manuscript

| Review issue | Resolution |
|---|---|
| Projection-led results contradict measurement | Removed projected leaderboards, ablation, noise, gain table and derived figures from the results section. Main evidence is now the retained measured record only. |
| “Five datasets / ten models” but table showed three / seven | Replaced the partial table with the full measured 5-dataset × 10-model table. |
| Abstract foregrounded projected gains | Rewritten around measured point estimates and the dense accuracy–exposure trade-off. No projected gain appears as a result. |
| Stale 12.1% and projection-dependent 26.3% Amazon claims | Removed. The measured Amazon-Book point estimate is reported directly; its 5.9% margin over LightGCN++ is explicitly non-significant single-run evidence. |
| Projected 62.2% G1 attribution | Withdrawn. RQ3 is explicitly open pending measured component ablation. |
| Projected 5.4–5.7% noise degradation | Withdrawn. RQ4 is explicitly open pending independently retrained noise runs. |
| GAT-CF simultaneously “unreported” and “answered” | Harmonized throughout. GAT-CF clean-data metrics are measured; the three-part comparison is only partially answered because noise and tuning-audit evidence are missing. |
| Unexplained GAT-CF collapse | Added a direct diagnostic caveat: its tenfold Yelp deficit to LightGCN is anomalous and may be a tuning/implementation artifact. The paper requests LR sweeps and training curves rather than generalizing from the collapse. |
| Proposition 2 overstated full-system robustness | Narrowed to a per-edge, per-layer Channel-A deviation from the zero-credit reference. It is explicitly not an embedding, multi-channel or NDCG theorem. |
| Degree normalization / equation reference | Uses `sqrt(d_u d_i)` consistently and cites the consistency utility by label. |
| Invalid random-injection corollary | Removed. The text now states the distributional assumptions that would be required and makes no corollary to random or targeted attacks. |
| Undefined `v(empty)` and small coalitions | Defined `v(empty)=0` for G1/G2/G3 and `div(S)=0` for `|S|<2`; utility targets are stop-gradient snapshots. |
| Axioms overstated for deployed weights | Guarantees are restricted to exact Shapley credit. The manuscript now distinguishes credit modulation from full degree-normalized propagation weights and from distilled attention. |
| LightGCN / LightGCN++ recovery overstated | Exact recovery is restricted to Channel A at `lambda=0` with other channels disabled. The LightGCN++ relation is described as a scalar-modulated functional family, not full implementation equivalence. |
| “Zero-overhead inference” | Replaced by “no serving-time Monte-Carlo Shapley sampling.” Residual attention latency is explicitly unmeasured. |
| Algorithm inconsistencies | Uses modulo notation, the correct total-objective equation, a valid section reference, distinct raw/EMA notation, and recomputes graph-derived structures after pruning. |
| Projection-based conclusion and ethics text | Rewritten to describe only measured evidence, single-run limits, partial GAT comparison, and unvalidated attribution. |
| Coverage units | Standardized as percentages in the measured table and caption. |
| Title overclaimed noise resilience | Removed the noise-resilience claim from the title because no measured robustness result is retained. |

## Not claimed as resolved

The following require new computation or missing logs. They cannot be repaired
by manuscript editing:

1. multi-seed means, standard deviations, confidence intervals and tests;
2. a matched tuning audit and training curves, especially for GAT-CF;
3. measured `w/o G1/G2/G3/CL/L_game` ablations;
4. measured 0/5/10/20% random-edge-injection experiments;
5. training time, serving latency, memory and distillation-gap measurements;
6. deletion, insertion, stability and explainer-comparison experiments.

The abstract, Results, Conclusion, Limitations and Ethics sections all identify
these as unavailable. RQ3–RQ5 remain open. The pre-specified attribution
protocol is retained as a falsifiable future protocol, not an XAI result.

## Resulting scientific claim

The revised claim is deliberately narrower:

> Single-run measurements provide preliminary evidence that CoopGCN shifts the
> accuracy–exposure frontier relative to direct hypergraph/cooperative peers,
> with broad tail/catalogue exposure and competitive sparse-data NDCG, while
> sacrificing NDCG on dense ML-1M. This pattern requires multi-seed and
> tuning-controlled confirmation.

This replaces all earlier claims of universal accuracy, measured robustness,
component causality, validated explainability, and zero-cost deployment.
