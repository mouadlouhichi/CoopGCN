# Response to Baseline-Calibration and Mechanism-Audit Review

We agree that a hedged faulty run is not evidence. This revision therefore
withdraws the retained table from comparative use and makes calibration the
first validity gate.

## CSV-first confirmatory workflow

Before editing claims, we created schema-controlled records in
`experiments/expected/` for:

- sparse LightGCN calibration;
- multi-seed main results;
- frozen norm and G1/G2/G3/contrastive/game-loss/tail ablations;
- L=16/32/64/128 sweep;
- proxy-versus-50-permutation-Shapley validation;
- corruption, attribution, runtime/memory, and user-degree diagnostics;
- mechanism magnitude, hyperedge footprint, and norm-scale exposure audit.

`scripts/build_confirmatory_artifacts.py` produces LaTeX tables, SVG completion
figures and `experiments/ARTIFACT_STATUS.md`. Incomplete rows are rendered as
pending and cannot become manuscript claims.

## Sparse baseline calibration

The paper now compares retained LightGCN NDCG with He et al. references:

- Gowalla: 0.0348 vs 0.1554 (ratio 0.224);
- Yelp2018: 0.0145 vs 0.0530 (ratio 0.274);
- Amazon-Book: 0.0002 vs 0.0315 (ratio 0.006).

All fail a pre-specified factor-1.5 calibration threshold. Audit then found a
concrete harness bug: `get_sparse_adjacency` already emits both directions,
while each graph model added a second reverse `index_add`, double-counting every
message. A shared `aggregate_symmetric_edges` helper now counts each directed
entry once and has a regression test. Corrected checkpoints do not yet exist;
all retained graph comparisons remain invalid. Sparse comparative claims and
the earlier accuracy–coverage association RQ/table were removed. The 5×10 table remains
only as an unbolded diagnostic audit record.

## Mechanism magnitude and norm-scale confound

The manuscript now audits the code-level ranges:

- G1: [0.97, 1.03] multiplier;
- G2: 0.01 channel mixture and under-1% sparse catalogue footprint;
- G3: [1, 1.01] sample range, no pruning;
- contrastive: 0.005 coefficient;
- proxy loss: 0.01 coefficient;
- learned norm scalar: unbounded.

CoopGCN and LightGCN++ exposure profiles are shown side by side. The manuscript
identifies frozen-norm CoopGCN as the first causal ablation and makes no
cooperative-mechanism claim until it is complete.

## Neighbour-cap audit

At L=32, non-zero G1 proxy targets cover:

- at most 43.1% of ML-100K edges;
- at most 27.6% of ML-1M edges;
- exactly 68.8%/64.4%/56.4% on Gowalla/Yelp/Amazon.

Edges beyond the cap are not wholly unsupervised: the released consistency
loss gives them a zero proxy target, supervising neutral modulation. This
creates a dominant neutral-target class, while inference sharpens logits using
temperature 0.5. The paper no longer attributes the ML-1M deficit only to
contrastive loss; L and temperature mismatch are explicit alternatives.

## Taxonomy and tail circularity

- W7 is now marked **not addressed**: retained weighted BPR uses one negative
  and discards 63 generated IDs.
- W5 is marked partial: sparse G2 hyperedges cover below 1% of catalogue slots
  at 0.01 mixture.
- W10 is marked not evaluated.
- The abstract and results state that every proxy explicitly encodes tail
  preference. G1 z-standardization makes the bonus depend on each capped
  neighbourhood's tail fraction and can cancel it for tail-heavy users.
- Zero-tail-bonus and neighbourhood-tail-distribution rows are included in the
  confirmatory schemas.

## Protocol and presentation

- The paper requests zero/<5-training-interaction counts for global MovieLens
  test users; sparse pinned test-user counts are already recorded.
- It explains downward false-negative bias from validation/test positives not
  being cross-filtered.
- Coverage bolding was removed because high coverage can accompany a failed
  ranker.
- Cross-dataset coverage percentages are not interpreted as method effects.
- The cross-model Spearman table/RQ was deleted; its script remains an audit
  artifact only.

## Theory positioning

Attention is no longer said to “violate” Shapley axioms; it is not an
allocation of a characteristic function, so those axioms do not apply.
Symmetry is explicitly restricted to players identical in every coalition and
is not claimed to close popularity bias. The ideal theory is retained only as
design rationale, with a pending ML-100K 50-permutation experiment to test
whether q correlates with ideal credit and to measure the distillation gap.

## Remaining work

No new training result is fabricated. LightGCN calibration, frozen norm,
component ablations, three seeds, L sweep, proxy-vs-Shapley validation and
runtime remain pending. The manuscript's conclusion now states that its
retained empirical interpretation is invalid until those gates pass.
