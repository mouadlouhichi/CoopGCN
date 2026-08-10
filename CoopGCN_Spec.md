# CoopGCN — Why LightGCN stops improving & the Shapley upgrade path (Spec)

# Why LightGCN stops improving — and how cooperative game theory (Shapley) can get us further

A systematic analysis of LightGCN’s failure modes, the highest-ROI places to build a stronger model, and a concrete blueprint — **CoopGCN** — that injects Shapley values from cooperative game theory into message passing, hypergraph structure, and training. Companion document to the *Recommender Systems Benchmark 2026* report.

- 7 weakness categories
- 8 game-theory integration points
- concrete architecture blueprint
- evidence-grounded gain estimates

> **TL;DR — the argument in one paragraph:**
> LightGCN’s success comes from one insight —
> *drop the nonlinearities, keep the graph propagation*
> . But its remaining weaknesses are exactly the things its simplicity stripped away:
> **uniform neighbor weighting**
> (all interactions treated equally),
> **no sense of which interaction deserves how much credit**
> ,
> **no group-level (beyond-pairwise) structure**
> , and
> **a training recipe that amplifies popularity bias**
> . Cooperative game theory is a natural fit because message passing
> *is*
> credit assignment: Shapley values tell us, fairly and with axioms to back it, how much each interaction/neighbor/hyperedge contributes to a user’s predicted preference. Used as
> **edge weights in aggregation**
> ,
> **hyperedge weights in a hypergraph channel**
> , and
> **sample weights in training**
> , Shapley-based credit can fix LightGCN’s main failure modes — and the evidence base (DyHuCoG +9.8–16.2%, LightGCN++ +17.8%, LightGCL +10–23%, LightGNN +44% on Gowalla) says the headroom is real.

### 1 LightGCN in one equation — and what it silently assumes
$$
**E^(k+1)^ = (D^-½^  A D^-½^) E^(k)^** *# symmetric-normalized propagation, no weights, no nonlinearity*  
  
**E = α₀E⁽⁰⁾ + α₁E⁽¹⁾ + α₂E⁽²⁾ + α₃E⁽³⁾** *# mean pooling of layers*  
  
**ℒ = Σ −log σ(êᵤ·êᵢ⁺ − êᵤ·êᵢ⁻) + λ‖Θ‖²** *# BPR loss, one negative per positive*
$$

Every neighbor is aggregated with weight **1/√(dᵤ·dᵢ)** — a purely topological constant. Every layer is pooled equally. Every negative is sampled uniformly. These three “simplifications” are exactly where the weaknesses live.

### 2 The weaknesses — systematically

#### 2.1 Representational weaknesses
| # | Weakness | Severity | What it means | Evidence |
| --- | --- | --- | --- | --- |
| W1 | **Uniform neighbor weighting** — all interactions carry equal weight 1/√(dᵤdᵢ) | HIGH | A “misclick” and a “passion” interaction propagate identically; noisy/irrelevant neighbors dilute the signal for everyone they touch. | LightGCN++ analysis (RecSys’24): inflexible norm scaling & neighbor weighting are core defects; fixing them gives up to +17.8% NDCG@20. |
| W2 | **Embedding-norm inflexibility** — propagation preserves embedding norms by construction | HIGH | Norm carries no information, yet the model can’t use it; layer-0 vs layer≥1 norms are inconsistent, so layer pooling is miscalibrated. | LightGCN++ (RecSys’24); “LightGCN: Evaluated and Enhanced” (2023) shows results swing with normalization choices. |
| W3 | **Dimensional collapse** — repeated propagation is dominated by a few largest eigenvalues of the graph | HIGH | Embeddings collapse onto few directions; the space becomes “head-heavy”, which empirically shows up as popularity bias and worse long-tail recall. | LogDet/DC analysis (ICLR’23): LightGCN provably prone to dimensional collapse; GCF_logdet beats it and improves unpopular-item performance. |
| W4 | **Over-smoothing with depth** — performance peaks at 2–3 layers, then degrades | MED | Cannot exploit genuinely deep structure; limits modeling of multi-hop relations. | LightGCN paper itself; DAP study (2023): deeper convolution keeps Recall up but TR@20 (tail recall) collapses. |
| W5 | **No beyond-pairwise (group) structure** — edges connect exactly two nodes | MED | Sessions, categories, tags, social groups, bundles are relational units, not pairs — LightGCN must approximate them with cliques. | Hypergraph line of work: HCCF +30% R@20 (Yelp), DHCN beats all session GNNs, DHLCF +15–26% NDCG@10. |
| W6 | **Static, non-temporal embeddings** — no sequence/recency signal | MED | Preference drift, recency, and session context are invisible; identical to a user who interacted a year ago vs yesterday. | Sequential models (SASRec/gSASRec, HSTU) gain +7–30% over CF baselines when order matters. |

#### 2.2 Training & data weaknesses
| # | Weakness | Severity | What it means | Evidence |
| --- | --- | --- | --- | --- |
| W7 | **BPR loss with one uniform negative** | HIGH | Uniform negatives under-sample hard negatives; overconfidence inflates head-item scores; loss is the same failure mode diagnosed for SASRec. | gSASRec (2023): overconfidence under BCE/BPR; gBCE + many negatives gives +9.5% NDCG@10 (ML-1M), +47% (Gowalla). |
| W8 | **Popularity-bias amplification** — high-degree nodes dominate propagation | HIGH | Graph convolution moves users closer to popular items; tail items get recommended less the deeper the model. LightGCN shows the worst bias amplification among GNNs in comparative tests. | Comparative bias study (2023): LightGCN highest popularity bias sensitivity; DAP (2023): TR@20 drops sharply with layers. |
| W9 | **Cold start — random-initialized embeddings, no side information** | MED | New users/items have nothing to propagate; pure CF cannot generalize from metadata. | LLM-augmented hybrids gain +20–30% in cold-start regimes (GenAIRecP@KDD’25); MARec-style metadata alignment. |
| W10 | **No robustness to noisy/poisoned interactions** | MED | All interactions trusted equally; one malicious or erroneous rating ripples through the graph. | Data-Shapley line (Ghorbani 2019): valuing and pruning low-value samples improves robustness; HCDV (2026) scales it. |
| W11 | **Evaluation leakage in the standard protocol** | HIGH | The classic LightGCN protocol leaks test interactions into the training graph; reported numbers are inflated, and a popularity baseline closes most of the gap on cleansed data. | CUISINART (KDD’24): LightGCN’s apparent SOTA partially an artifact of leakage. |
| W12 | **Replication fragility** — results swing with normalization, layer count, and splits | MED | Gowalla R@20 ranges 0.152–0.181 across documented configurations; hard to trust numbers across papers. | “LightGCN: Evaluated and Enhanced” (2023): replication interval analysis. |

- ****🔑 Root cause #1 — credit is flat**** The aggregation weight 1/√(dᵤdᵢ) is the same for a decade-old rating and a last-night purchase, for a niche favorite and a viral hit. LightGCN never asks: *how much did this specific interaction contribute?* That is precisely a cooperative-game question.
- ****🔑 Root cause #2 — structure is pairwise-only**** Real preference lives in groups (sessions, categories, circles). Pairwise edges approximate groups poorly, especially in sparse data where group signals are the strongest evidence available.
- ****🔑 Root cause #3 — the recipe amplifies the head**** Uniform negatives + uniform weights + mean pooling + no norm information = a system that gets “safer” by recommending popular items. Every recent improvement (LightGCL, SimGCL, LightGNN, LightGCN++) is essentially a patch on one of these three.

### 3 Opportunity map — where the headroom is
| # | Fix (already proven) | Weaknesses addressed | Reported gain | Source |
| --- | --- | --- | --- | --- |
| 1 | Norm-aware scaling + learnable neighbor weighting + tuned layer pooling | W1, W2, W12 | up to +17.8% NDCG@20 | LightGCN++ (RecSys’24) |
| 2 | Global SVD-truncated contrastive view (robust augmentation) | W3, W4, W8, W9 | +10–23% R@20 over SimGCL | LightGCL (ICLR’23) |
| 3 | Edge pruning + knowledge distillation (denoise the graph) | W1, W10, W8 | Gowalla R@20 0.181 → 0.261 (+44%) | LightGNN (WSDM’25) |
| 4 | Hypergraph channel for group structure | W5, W9 | +30% R@20 (Yelp); +15–26% NDCG@10 | HCCF (SIGIR’22); DHLCF (CIKM’22) |
| 5 | Training-recipe fixes: gBCE / sampled softmax, 256 negatives | W7, W8 | +9.5% NDCG@10 (ML-1M); +47% (Gowalla) | gSASRec (RecSys’23) |
| 6 | Contrastive + decorrelation (dimensional-collapse fix) | W3, W8 | +16–50% on sparse sets | XSimGCL (TOIS’23); GCF_logdet (ICLR’23) |
| 7 | Shapley-weighted dynamic hypergraph (game theory!) | W1, W5, W8, W10 | +9.8–16.2% over SOTA HPCF; +coverage, +diversity | DyHuCoG (2025) |

![Reported gains by fix](chart_gain_evidence.png)
Evidence base for the opportunity map — all values from the cited papers (see Sources). The “protocol” bar is deliberately different in kind: it is a warning that a share of published gains evaporates under leakage-free evaluation.

### 4 Cooperative game theory — the Shapley toolset

#### 4.1 The idea in 30 seconds
Cooperative game theory studies how to **fairly split the value created by a team**. A **coalition** S is any subset of “players” (interactions, neighbors, hyperedges, data samples…). A **value function** v(S) measures what the coalition achieves. The **Shapley value** φⱼ of player j is its average marginal contribution over all possible coalitions:

$$
φⱼ = Σ_{S ⊆ N∖{j}} **|S|!(|N|−|S|−1)! / |N|!** · [ v(S ∪ {j}) − v(S) ]  
  
*# the only allocation satisfying: efficiency (Σφⱼ = v(N)), symmetry, dummy, additivity — Shapley 1953*
$$

The axioms are what make it attractive: efficiency guarantees the credits add up to the total value; symmetry means two interchangeable players get equal credit; the dummy axiom zeroes out irrelevant players. That is exactly the property a recommender needs when asking “which interactions made this user like this item?”.

> **Key insight from the XAI literature**
> (Beyond Shapley Values, 2025): the
> **choice of value function v(·) is the real design decision**
> — Shapley only aggregates. In a recommender, v(S) can be defined per use case: ranking utility, preference alignment, diversity, robustness… and the same Shapley machinery then yields different, purpose-built credits.

#### 4.2 Eight places Shapley plugs into a LightGCN-class model
| # | Integration point | Game setup (players → v(S)) | Weakness fixed | Status in literature |
| --- | --- | --- | --- | --- |
| G1 | **Edge weighting in message passing** | Players = neighbors/interactions of a node; v(S) = quality of the representation built from S (e.g. downstream ranking loss). Edge weight w(u,i) ∝ φ̂(u,i). | W1 (uniform weighting), W10 (noise) | Novel direction; analogous to attention but axiomatically grounded |
| G2 | **Hyperedge weighting in a hypergraph channel** | Players = members of a hyperedge (a session, a category, a social circle); v(S) = preference-consistent utility of that group. Hyperedge weight ∝ aggregated φ̂. | W5 (group structure), W8 (bias) | **Proven** — DyHuCoG (2025): preference-aware Monte-Carlo Shapley → dynamic hyperedge weights → +9.8–16.2% over HPCF, +coverage & diversity |
| G3 | **Training-data valuation / denoising** | Players = training interactions; v(S) = validation ranking metric of a model trained on S. Low-φ interactions are down-weighted or pruned. | W10 (noise/poisoning), W8 (bias) | **Proven in ML broadly** — Data-Shapley (2019), truncated/hashed/stratified accelerations; HCDV (2026) 10–100× faster |
| G4 | **Popularity-bias correction via coalition credit** | Players = items in a user’s candidate set; v(S) = user preference for S. Tail items get credit when they genuinely drive utility; the model stops “free-riding” on popularity. | W8 (bias), W3 (collapse) | Conceptually supported by DyHuCoG’s long-tail gains; no dedicated study yet |
| G5 | **Cold-start embedding construction** | Players = side features (categories, text, images); v(S) = predictive utility of the embedding built from S. φ gives a principled feature-attribution initialization for new nodes. | W9 (cold start) | Feature-attribution Shapley is mature (SHAP); application to CF cold-start is open |
| G6 | **Multi-behavior / multi-view fusion** | Players = behavior channels (view, cart, buy, favorite); v(S) = utility of fusing channels S. Channel weights ∝ φ. | W1 (flat credit) applied to channels | Analogous to Shapley-based federated client weighting (2025) |
| G7 | **Negative sampling reweighting** | Players = candidate negatives; v(S) = ranking loss improvement of S. Hard negatives get higher Shapley importance → smarter sampling than uniform. | W7 (BPR negatives) | Open; complements gBCE/sampled-softmax recipes |
| G8 | **Explainability by-product** | Same games as G1/G2; φ values double as “why this item” attributions. | Trust/transparency (non-accuracy) | Mature in XAI; rare inside CF models |

The highest-value, lowest-risk starting points are **G1 + G2 + G3**: they directly patch LightGCN’s three root causes (flat credit, pairwise-only structure, noisy training), reuse machinery that already works (DyHuCoG for G2, Data-Shapley for G3), and compose cleanly with the proven non-game fixes (LightGCN++ norms, LightGCL contrastive view, gBCE loss).

#### 4.3 The cost problem — and the standard answers
Exact Shapley values are #P-hard (2^\|N\|^ coalitions). In a recommender, \|N\| is the size of a neighborhood (tens), a hyperedge (tens), or a training set (millions) — so exact computation is impossible. The toolbox:

| Approximation | Idea | Cost | Used by |
| --- | --- | --- | --- |
| **Monte-Carlo permutation sampling** | Random permutations of players; φ̂ⱼ = average marginal contribution over T permutations | O(T·\|N\|) evaluations | Data-Shapley; DyHuCoG (preference-aware MC) |
| **Truncated MC** | Stop a permutation once marginal gains vanish (convergence threshold) | ~2–5× cheaper than plain MC | Jia et al. 2019 (TMC-Shapley) |
| **Stratified / hashed sampling** | Group players by estimated value, sample within groups; hash-based dedup | 10–100× cheaper | Wu et al. 2023; Kwon & Zou 2021 (Owen sampling, AME) |
| **Hierarchical coalition trees** | Coarse-to-fine clusters of players, propagate budgets downward | 10–100× cheaper with guarantees | HCDV (2026) |
| **Restricted games** | Only coalitions within a node’s neighborhood (≤ degree, e.g. ≤ 32) — tractable and semantically meaningful | O(2^k^) with k small | Suggested for G1/G2 in rec graphs |

> **Practical takeaway:**
> for neighborhood-scale games (G1/G2) restrict coalitions to ≤ 20–30 players and use MC with ~50–200 permutations; for data valuation (G3) use truncated/stratified MC or the hierarchical method. In all cases compute φ̂
> **periodically**
> (every few epochs), not per step — the graph changes slowly.

### 5 Blueprint — “CoopGCN”, a concrete better model

_[Architecture diagram — see the HTML version]_  

#### 5.1 Design decisions (each maps to evidence)
| Component | Choice | Why (evidence) |
| --- | --- | --- |
| Backbone | LightGCN-style propagation, L=3, with **LightGCN++ norm scaling** (α, β, γ on embedding norms, neighbor weighting, layer pooling) | +17.8% NDCG@20 with a near-drop-in change (RecSys’24) |
| Channel A — graph GCN | Messages weighted by **φ̂(u,i)** from neighborhood coalition games (G1), softmax-normalized inside the node’s neighborhood | Fixes flat credit (W1); principled alternative to uniform weights / heuristic attention |
| Channel B — hypergraph GCN | Hyperedges from sessions/categories/tags; edge weights from **φ̂ over hyperedge coalitions** (G2) with an interaction-level attention gate | DyHuCoG proof: +9.8–16.2% over HPCF; captures group structure (W5) |
| Channel C — global view | SVD-truncated graph for local–global InfoNCE contrast (LightGCL recipe) | +10–23% R@20; fights collapse & bias (W3, W8) |
| Training loss | **gBCE or sampled softmax with 256 negatives**, temperature τ; + λ₁ℒ_cl + λ₂ℒ_game (consistency between φ̂ and learned scores) | gSASRec: +9.5% NDCG@10 ML-1M, +47% Gowalla; kills overconfidence (W7) |
| Data curation | Offline **Data-Shapley scan** (G3) — prune/weight the bottom-φ interactions; re-run on drift | Robustness (W10); HCDV makes this cheap at scale |
| Evaluation | **Global temporal split**, no test leakage; report Recall@20, NDCG@20 **plus TR@20, coverage, Gini, diversity** | CUISINART: leakage inflates LightGCN numbers; bias metrics expose W8 |

#### 5.2 CoopGCN vs DyHuCoG — the differences that matter
**DyHuCoG is not CoopGCN.** They share one ingredient (Shapley-valued hyperedge weights) and little else. The comparison:

| Dimension | DyHuCoG (2025) | CoopGCN (this blueprint) |
| --- | --- | --- |
| **Where game theory applies** | One place: hyperedge weighting (players = user–item–context interaction triples, Monte-Carlo preference-aware Shapley → hyperedge weights) | **Three levels:** G1 edge weights in pairwise message passing + G2 hyperedge weights + G3 training-data valuation — all in one model |
| **Backbone** | Hypergraph NN only (lightweight HGNN + interaction-level attention gate) | **Dual-channel:** LightGCN-style pairwise GCN as the main channel, hypergraph channel as auxiliary, plus an SVD global view |
| **Other components** | None (no contrastive learning, no norm scaling, standard loss) | LightGCN++ norm scaling, LightGCL-style contrastive view, gBCE + 256 negatives, Data-Shapley denoising, ℒ_game consistency between φ̂ and scores |
| **Shapley engineering** | Preference-aware Monte Carlo | Restricted coalitions (≤32 players), periodic recompute, hierarchical/stratified options, and an explicit **φ̂ vs learned-attention ablation** |
| **Evaluation** | ML-1M + Amazon-Book; Recall/NDCG@20, coverage, diversity | 5 datasets, **global temporal splits**, TR@20 (tail recall), Gini, popularity gap, 3 seeds + significance tests |
| **Reported gains** | +9.8–16.2% over HPCF (hypergraph SOTA) | +15–30% over tuned LightGCN — an **estimate**, see evidence caveat below |

> **Honest positioning (read this before writing the paper):**
> **G2 (Shapley-weighted hyperedge channel) is DyHuCoG’s idea.** If CoopGCN includes it, DyHuCoG must be cited as the direct source and the component ablated against it. There is no novelty claim available at that single point. / **What would genuinely be new:** (a) **edge-level Shapley in pairwise message passing (G1)** — DyHuCoG never weights plain graph edges; (b) **Shapley as data valuation inside the same training loop (G3)**; (c) the **multi-level consistency loss** tying φ̂ across edges/hyperedges/samples; (d) the **system-level combination** with contrastive view, norm scaling and loss recipe; (e) the **rigorous evaluation protocol** (temporal splits + bias metrics) that DyHuCoG-style papers typically lack. / **Evidence correction:** DyHuCoG’s +9.8–16.2% validates *only the G2 component*, on 2 datasets, against *hypergraph* baselines. It does **not** validate edge-level Shapley (G1), data valuation (G3), or the full stack. In the expected-gain table below, treat the G2 row as borrowed with caution and G1/G3 as hypotheses to be tested — that is exactly what experiment steps 3–5 are designed to do.

#### 5.3 Why this will beat LightGCN — expected gains, evidence-grounded
| Source of gain | Reported effect | Expected contribution in CoopGCN |
| --- | --- | --- |
| Norm-aware weighting + tuned pooling (LightGCN++) | +17.8% NDCG@20 | +8–12% (base uplift, reliably) |
| Shapley edge weights (G1) replacing uniform weights | Analogous denoising in LightGNN: +44% R@20 Gowalla (with KD) | +3–8% on top (on noisy datasets) |
| Shapley hyperedge channel (G2) | DyHuCoG +9.8–16.2% over SOTA hypergraph baseline; +coverage/diversity | +5–10% where groups exist (sessions, categories) |
| Training recipe (gBCE, 256 negs) | +9.5% NDCG@10 (ML-1M), +47% (Gowalla) | +5–10% |
| Contrastive global view (LightGCL) | +10–23% R@20 over SimGCL | +5–10% on sparse sets |
| Data-Shapley denoising (G3) | Robustness/efficiency gains in data valuation literature | +2–5%, mainly robustness & tail recall |

**Realistic combined expectation: +15–30% NDCG@20 over a well-tuned LightGCN** on standard benchmarks — in line with what the strongest 2024–25 graph models achieve individually, with the added bonus of coverage/diversity/fairness metrics that LightGCN specifically fails at. **Evidence provenance matters:** the LightGCN++ / LightGCL / LightGNN / gSASRec / XSimGCL rows are from peer-reviewed studies on graph-CF and transfer directly; the DyHuCoG row validates only the hyperedge-Shapley component (G2) and transfers to the full stack only as a hypothesis (§5.2); the G1 and G3 rows are hypotheses until the step 3–5 ablations run. Interaction effects can be sub-additive — treat the range as a planning estimate, not a promise.)

#### 5.4 Experiment plan (first 8 weeks)
**1.** 1

**2.** 2

**3.** 3

**4.** 4

**5.** 5

**6.** 6

#### 5.5 Risks and pitfalls
> **Read before building:**
> **Shapley cost.** MC estimation on every edge is infeasible — compute φ̂ periodically, restrict coalitions, or use hierarchical approximations. Budget: φ̂ must cost < 20% of training time or the model dies in review. / **Is Shapley better than a learned weight?** Attention weights are trained end-to-end and often match. Shapley’s selling points are *axiomatic fairness, robustness to noise, and interpretability* — the ablation in step 3 must be honest about when it wins. / **Value-function choice is everything.** A badly chosen v(S) (e.g. pure accuracy) can reintroduce popularity bias through the back door. Define v(S) to include tail/coverage terms (as DyHuCoG does). / **Leakage will inflate your numbers.** Temporal splits only. If you report random-split numbers alongside, label them clearly. / **Baseline tuning asymmetry** is the #1 way reviewers reject a paper: tune baselines as hard as your own model (RecSys’22 lesson). / **Complexity creep.** Every module added must beat its own ablation; the “no-game version” of CoopGCN is already a strong model — keep it that way.

### 6 External critical review — response & refinements
A peer-style critical review of this blueprint graded it **A− as a blueprint, incomplete as a paper**, and flagged concrete gaps. All points are addressed — the detailed formal fixes live in the companion document ***coopgcn_formalization.html*** (games, value functions, losses, PyTorch sketch, experiment matrix). Summary of the response:

| Review point | Response / action taken |
| --- | --- |
| **“v(S) is never defined — the biggest missing piece.”** | Now formalized: **v^cons^**(S) = −‖(1/\|S\|)Σ_i∈S_ e_i_ − ē_u_‖² (default G1), **v^pref^**(S) = α·avg affinity + β·tail + γ·diversity (default G2, blocks popularity free-riding), **v^ndcg^** = holdout NDCG (offline G3). Full math in §2 of the companion. |
| **“Is Shapley just expensive attention?”** | Accepted as **the central ablation**: φ̂ vs uniform vs degree-norm vs LightGCN++ scalar vs GAT attention — judged on R@20, N@20, **TR@20, coverage, robustness**, 3 seeds, paired t-test. If φ̂ wins only on tail/robustness, the paper is reframed as *trustworthy + robust graph recs via axiomatic credit*. |
| **“Scope creep: G4–G8 read as unfocused.”** | Adopted: core paper = **G1 + G2 + G3** only. G4–G8 explicitly demoted to future-work section (§8 of companion). |
| **“Gains must be non-additive, vs Baseline+, not vanilla.”** | Reframed: every ablation reports delta from **Baseline+** (LightGCN++ norms + gBCE/256 + LightGCL view, no game). Expectation: **+5–15% over Baseline+** (not +15–30% over vanilla), and temporal splits shrink absolute numbers 10–20% (CUISINART) — stated explicitly. |
| **“Add a leakage audit.”** | **Step 0.5** added: random-vs-temporal split inflation measured and reported per dataset, for every model. |
| **“8 technical open questions.”** | All answered in companion §7: coalition sampling (top-recent + stratified, k≤32), MC variance (antithetic, std reported), efficiency renormalization after truncation, symmetry under timestamped duplicates, EMA + stop-grad for non-stationarity, complexity O(T·k·d) with wall-clock budget <20%. |
| **“Hypergraph construction is dataset-dependent / can leak.”** | Per-dataset construction rules defined (Yelp/Amazon categories, Gowalla spatial clusters, ML-1M genres, LastFM tags), built from **train-only** data. |
| **“Add robustness test & statistical rigor.”** | Injected-noise test (5/10/20%) added — G3’s payoff; 3 seeds + paired t-test mandated in the matrix. |

> **Publishability bar (from the review, adopted as the go/no-go test):**
> G1+G2 must beat uniform
> *and*
> learned attention on ≥3 datasets under temporal splits — on tail/robustness metrics if not NDCG. If only G2 works, the submission is repositioned (or not submitted) as incremental over DyHuCoG.

### 7 Roadmap summary
| Phase | Goal | Deliverable | Time |
| --- | --- | --- | --- |
| **0 — Floor** | Reproduce 5 baselines on 5 datasets (temporal splits) | Calibrated benchmark harness | 2–3 wk |
| **1 — Free gains** | LightGCN++ + gBCE + contrastive view | Baseline+ model, +10–20% | 1–2 wk |
| **2 — Game module** | Shapley edge weights (G1) | Ablation: φ̂ vs uniform vs attention | 2–3 wk |
| **3 — Hypergraph + Shapley** | Shapley-weighted hyperedge channel (G2) | Group-structure gains, DyHuCoG-comparable | 2–3 wk |
| **4 — Data valuation** | Data-Shapley denoising (G3) | Robustness study + tail-recall gains | 1–2 wk |
| **5 — Rigor** | Full metrics, seeds, stats, fairness analysis | Paper-ready results + repository | 1–2 wk |

### 8 Sources
1. LightGCN — [SIGIR’20](https://dl.acm.org/doi/10.1145/3397271.3401063)
2. LightGCN++ (inflexibility/inconsistency analysis + remedy) — [RecSys’24](http://dmlab.kaist.ac.kr/~kijungs/papers/lightgcnppRecSys2024.pdf)
3. LightGCN: Evaluated and Enhanced (replication) — [arXiv:2312.16183](https://arxiv.org/abs/2312.16183)
4. CUISINART (evaluation leakage in graph-CF benchmarks) — [KDD’24](https://arxiv.org/abs/2308.03953)
5. Dimensional collapse of graph CF (LogDet) — [ICLR’23](https://openreview.net/pdf?id=MvCq52yt9Y)
6. Popularity-bias amplification in GCN recommenders (DAP) — [Front. Comput. Sci. 2023](https://arxiv.org/pdf/2305.14886); comparative bias study — [arXiv:2301.07639](https://arxiv.org/pdf/2301.07639)
7. LightGCL (SVD global contrastive view) — [ICLR’23](https://arxiv.org/abs/2302.08191)
8. LightGNN (edge pruning + distillation) — [WSDM’25](https://arxiv.org/abs/2501.03228)
9. XSimGCL — [TOIS’23](https://arxiv.org/abs/2209.02544); SimGCL — [SIGIR’22](https://arxiv.org/abs/2203.02516)
10. HCCF (hypergraph contrastive CF) — [SIGIR’22](https://arxiv.org/abs/2204.12200); DHCN (session hypergraph) — [AAAI’21](https://arxiv.org/abs/2012.06852); DHLCF (dynamic hypergraph learning) — [CIKM’22](https://dl.acm.org/doi/10.1145/3511808.3557301)
11. DyHuCoG (dynamic hypergraph cooperative game, preference-aware Shapley) — [2025](https://oaji.net/articles/2025/3603-1768748355.pdf)
12. Data-Shapley — [Ghorbani & Zou, ICML’19](https://arxiv.org/abs/1904.02868); TMC-Shapley — [Jia et al.’19](https://arxiv.org/abs/1907.02846); HCDV (hierarchical contrastive valuation) — [arXiv:2512.19363](https://arxiv.org/html/2512.19363)
13. Beyond Shapley Values (value-function design for cooperative-game ML) — [arXiv:2506.13900](https://arxiv.org/pdf/2506.13900)
14. gSASRec (overconfidence + gBCE) — [RecSys’23](https://arxiv.org/abs/2308.09756); Turning Dross Into Gold (negatives recipe) — [RecSys’23](https://dl.acm.org/doi/10.1145/3604915.3610644)
15. Are We Really Making Much Progress? (baseline tuning) — [RecSys’22](https://dl.acm.org/doi/10.1145/3523227.3546767)

_ Companion to *Recommender Systems Benchmark 2026* (same folder) · Compiled Aug 2026 · Gain estimates are engineering extrapolations from single-factor studies — validate per dataset. _

