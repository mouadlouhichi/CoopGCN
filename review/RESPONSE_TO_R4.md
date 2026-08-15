# Response to Fourth-Iteration Reviews

We took the reviewers' **narrow-scope route**. The manuscript is now a
preliminary methods study with two descriptive questions, not a broad empirical
paper claiming component causality, robustness, explanation, or efficiency.
No missing experiment has been replaced by a projection.

## Front matter and scope

- Title now says **cooperative-game-inspired credit signals**, **long-tail
  exposure**, and **preliminary methods study**.
- The first abstract sentence states that every benchmark cell is one run and
  lists the unavailable confirmatory studies.
- Robustness was removed from keywords and measured outcomes.
- Formal RQ3–RQ5 and the unmeasured ablation matrix were removed. The paper now
  asks only (RQ1) where retained point estimates lie and (RQ2) whether
  cross-model NDCG/Coverage association is uniformly negative.
- “Stage-1-style” was replaced by “preliminary measured evaluation”; no formal
  preregistration is claimed.
- Internal credit is defined as a training/propagation signal, not explanation.
  The long unexecuted XAI section was removed; faithfulness tests are future
  work and explanation is conditional on passing them.

## Most important implementation correction

An audit of the checkpoint-generation code found that legacy class/method names
were stronger than the executable behavior. The paper and package now disclose
that the retained checkpoints use deterministic proxies, not permutation
Shapley estimators:

- G1: centered edge–user alignment + 0.05 tail bonus, standardization, EMA;
- G2: off-diagonal cosine affinity + 0.5 tail share;
- G3: sigmoid score × tail factor, EMA and quantile reweighting;
- the G3 5% mask is computed but **not applied** as graph pruning;
- no G1/G2 permutations and no G3 TMC retraining occur;
- weighted BPR uses one effective negative although 64 IDs are generated;
- consistency training is untempered while propagation uses temperature 0.5.

The manuscript now separates ideal cooperative-game theory from this measured
proxy implementation. The exact retained configuration is tabulated and
archived in `paper/retained_run_manifest.yaml`. Package docstrings retain
legacy names only for checkpoint/API compatibility and explicitly state their
actual behavior.

## Mathematical and algorithm corrections

- Edge modulation now matches released code:
  `1 + 2 lambda (sigmoid(credit/tau) - 1/2)`.
- Zero credit is neutral, not minimum, modulation.
- Proposition 2's bound is consequently
  `lambda epsilon / (2 tau sqrt(d_u d_i))` and cites the consistency utility by
  label.
- Strict monotonicity is restricted to `lambda>0, tau>0`; at `lambda=0` the map
  is constant.
- Ideal permutation unbiasedness is explicitly conditional on a fixed capped
  player set and finite variance. Samplewise vs expectation-level axiom
  behavior is distinguished.
- Exact, estimated, EMA, deterministic-proxy and attention symbols are separate.
- Algorithm 1 was replaced by the actual proxy training path; no broken
  equation/section references, TMC, or pruning remain.
- Complexity now reports symbolic proxy costs and memory, with no invented
  percentage overhead.
- The dummy-player/pruning equivalence claim was deleted.

## Data and evaluation reproducibility

- MovieLens uses global temporal 70/10/20 splits.
- Gowalla/Yelp/Amazon use LightGCN-PyTorch commit
  `947ca2b3b1d2d3545b114145710cb06c4e57b3d2`; these files have no timestamps.
- Split interaction counts, candidate construction, eligible users, seen-item
  filtering, score-tie behavior, TR averaging and coverage denominator are now
  explicit.
- Hyperedge generation is specified for all five datasets, including accepted
  counts and size ranges. Sparse-data hyperedges are seeded co-occurrence
  clusters—not geographic/category metadata.
- The authoritative comparison set is nine baselines plus CoopGCN. SGL and
  SimGCL are related work only because no same-protocol record survives.
- GAT-CF is identified as an in-house MLP attention adaptation, not a published
  recommender reproduction.
- The measured table now has separate NDCG, TR and Coverage panels. Rounded
  CoopGCN coverage percentages are accompanied by approximate distinct-item
  counts.

## Descriptive association

Table 9 was renamed “descriptive cross-model association.” Ties receive average
ranks. Because model identities are not a random sample, bootstrap confidence
intervals would be misleading; the paper instead reports complete
leave-one-model-out ranges. It explicitly discusses Amazon-Book's bimodal
collapsed cluster and a post-hoc non-collapsed diagnostic, and states that a
third sparsity-handling factor could improve both metrics without implicating
credit.

## References, ethics, and presentation

- Added recommendation-specific explainability literature.
- Replaced fragile diacritics in the GAT reference and marked incomplete
  DyHuCoG metadata as a preprint rather than inventing publication fields.
- Ethics now covers relevance loss from increased coverage, provider/user
  trade-offs, sensitive history leakage and false-explanation trust.
- Data and code provenance point to pinned sources and checkpoint lineage.

## Experiments that remain unavailable

The repository still lacks five-seed reruns, controlled component ablations,
random-corruption retraining, attribution faithfulness, timing/memory, and
matched baseline tuning traces. These cannot be fixed by prose. The narrow
manuscript makes no affirmative claim in those areas and lists them only as
pre-specified confirmatory work.
