# Response to Third-Iteration Review (R3)

We thank the reviewer for recognizing the manuscript's correction of its
projection/measurement conflict and theoretical scope. We addressed every
issue that can be resolved from the retained evidence and did not manufacture
multi-seed or attribution results that were never run.

## Changes made

| Reviewer concern | Revision |
|---|---|
| Abstract says “two” sparse benchmarks | Corrected to **three**: Gowalla, Yelp2018 and Amazon-Book. |
| NCF and RecDCL lack citations | Added He et al. (WWW 2017) for NCF and Zhang et al. (The Web Conference 2024) for RecDCL, with DOI metadata. |
| RecDCL/NCF dense-data dominance is ignored | Added an explicit RQ1 discussion. Their NDCG point estimates are roughly 35–60% above the best graph-model values on ML-100K/ML-1M, while TR@20 is zero and coverage is low. The manuscript now states that graph credit assignment is not the strongest dense-NDCG lever in this record. |
| MF anomaly receives less scrutiny than GAT-CF | Added the same epistemic warning: MF's near-floor Gowalla/Amazon values may reflect sparse-regime failure or tuning/implementation artifacts, and cannot support an architecture-wide conclusion. |
| Coverage may rise mechanically as NDCG falls | Added a reproducible descriptive sensitivity analysis. `data/measured_results.csv` is the machine-readable measured table; `scripts/analyze_measured_tradeoff.py` computes per-dataset Spearman correlation for all models and graph models; `paper/measured_tradeoff.csv` records the generated output. The manuscript reports mixed signs and explicitly says the check is neither causal nor a substitute for ablation/seeds. |
| Theory/validation imbalance | Reframed the manuscript explicitly as a methods paper with Stage-1-style preliminary validation rather than a completed confirmatory empirical study. |
| “Credit assignment” may be read as validated explanation | Added an Introduction-level definition: credit assignment means allocating propagation/training influence, not a faithful human-facing explanation. The unexecuted faithfulness protocol remains clearly separate. |

## Experiments still unavailable

The reviewer correctly identifies two principal empirical blockers:

1. at least five independent seeds with uncertainty estimates; and
2. at least one deletion/necessity attribution experiment.

The repository contains neither multi-seed outputs nor a completed attribution
evaluation. They cannot be supplied by rewording the paper or deriving values
from the single-run table. We therefore retain them as explicit requirements,
make no significance or explanation-faithfulness claim, and keep RQ3–RQ5 open.
The same applies to measured component ablation, GAT-CF tuning traces,
random-edge injection and wall-clock timing.

## Resulting scope

The manuscript's claim remains preliminary:

> Single-run measurements motivate an accuracy–exposure hypothesis for
> Shapley-derived graph credit assignment. A reproducible cross-model
> sensitivity check shows that high coverage is not universally associated
> with low NDCG, but only multi-seed, tuning-controlled component and
> attribution experiments can validate the mechanism.
