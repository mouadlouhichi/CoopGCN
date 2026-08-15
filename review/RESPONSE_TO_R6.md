# Response to major-revision review (R6)

## Summary

We agree that the retained graph results cannot support publication claims. The
aggregation audit identified a twofold message-counting defect; all affected
checkpoints remain invalidated. No corrected metric has been invented. The
manuscript remains an audit-first report pending post-fix reruns.

## Mandatory issues

1. **Post-fix LightGCN calibration.** The corrected propagation helper and
   `scripts/run_baseline_calibration.py` are present. The factor-1.5 gate is now
   described as a permissive engineering screen, not equivalence. Execution is
   still pending and the manuscript says so.
2. **Candidate protocol.** The evaluator now masks opposite-held-out positives
   while retaining current ground truth, including overlap cases. A regression
   test covers validation, test, overlap, and the opt-out control.
3. **Frozen norm ablation.** The model and scheduled ablation already support a
   fixed norm scale of 1. No result is claimed.
4. **Tail-bonus circularity.** G1, G2, and G3 tail coefficients are now
   independently configurable. The scheduled variants set G1=0, G2=0, and G3
   tail/head multipliers to 1/1. No result is claimed.
5. **Neighbour cap.** Seeded uniform random cap sampling and L=16/32/64/128
   variants are executable. The manuscript now states that the retained prefix
   was exact upstream adjacency order.
6. **Implementation appendix.** The former and corrected aggregation operations
   are shown algebraically, with the regression-test invariant.

## Clarifications and additions

- Added a negative-contribution audit explaining when down-modulation could be
  meaningful and why the retained proxy cannot establish harmfulness.
- Added a specified synthetic G2 sanity check (tail-share contrast, zero-tail
  equality, and gradient-path test). It is explicitly not reported as run.
- Added machine-readable expected values only as design settings and acceptance
  thresholds under `experiments/projected/`. They are not projected outcomes.
- Extended component and cap CSV schemas and regenerated all pending-safe tables
  and figures.

## Still required before empirical submission

Corrected calibration; multi-seed baseline and CoopGCN runs; frozen-norm and all
zero-tail ablations; randomized cap sweep; proxy-versus-exact-Shapley validation;
attribution faithfulness; corruption controls; and runtime/memory reporting.
Expected values cannot substitute for these measurements.
