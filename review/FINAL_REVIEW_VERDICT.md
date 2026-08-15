# Historical Engineering Review — Superseded

**Status:** Superseded by the empirical audit in
[`main_results/ANALYSIS.md`](../main_results/ANALYSIS.md), the revised
manuscript, and [`RESPONSE_TO_R1_R2.md`](RESPONSE_TO_R1_R2.md).

The earlier document at this path described the implementation as “perfect”
and certified benchmark superiority. That conclusion was not supported by the
available evidence:

- several headline tables were projections rather than completed runs;
- measured results contradict the projected accuracy ranking;
- component and noise studies were not measured in the retained pipeline;
- no multi-seed uncertainty, timing, or attribution-faithfulness study exists;
- unit tests provide local sanity checks, not proofs of end-to-end robustness
  or empirical superiority.

## Current verdict

The repository contains a substantive Shapley-derived graph-CF implementation
and a coherent theoretical design, but the empirical study remains
**under-validated**. The retained single-run record suggests strong tail and
catalogue exposure relative to direct hypergraph/cooperative peers, competitive
sparse-data NDCG, and a dense ML-1M accuracy deficit. These are preliminary
point estimates, not statistically established conclusions.

The following remain required before an empirical superiority or robustness
claim can be certified:

1. matched tuning logs and at least five seeds;
2. measured component and `w/o L_game` ablations;
3. measured random-edge-injection experiments;
4. wall-clock and memory benchmarks;
5. attribution deletion/insertion/stability tests.

This file is retained to make the correction explicit rather than silently
removing the earlier audit history.
