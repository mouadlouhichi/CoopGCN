# CoopGCN — Formalization & Implementation Sketch

# CoopGCN — Formalization & Implementation Sketch

Technical companion to *specs/CoopGCN_Spec.md*. This document answers the critical review’s open questions: **exact game definitions (players, coalitions, value functions v(S))** for the three game levels, **loss equations**, **complexity analysis**, **PyTorch pseudo-code** for MC-Shapley edge weighting, and the **ablation/experiment matrix** that decides whether the paper lives or dies.

- G1 edges · G2 hyperedges · G3 data
- value functions defined
- PyTorch sketch
- 8 open questions answered

> **TL;DR — what changed after the critical review:**
> **v(S) is now concrete** (§2–3): a consistency utility (distance to the full-model embedding), a preference-aware utility with tail/diversity terms (DyHuCoG-style), and an offline holdout-NDCG variant. One instantiation per game level. / **Scope tightened:** the paper = G1+G2+G3. G4–G8 (popularity, cold-start, multi-behavior, negatives, explainability) move to future work. / **The "Shapley vs attention" question becomes the central ablation** (§7): φ̂-weighting vs uniform vs degree-norm vs GAT vs LightGCN++ scalar, judged on tail/robustness metrics — not just NDCG. / **Gains reframed as non-additive**, reported as per-ablation deltas from a strong no-game baseline (Baseline+), on leakage-free temporal splits.

### 1 Notation
| Symbol | Meaning |
| --- | --- |
| G = (U, I, E) | Bipartite user–item interaction graph |
| N(u) ⊆ I | Item-neighborhood of user u (players of u’s game) |
| d_u_, d_i_ | Node degrees |
| e_u_^(k)^ | Layer-k embedding; ē_u_ = Σ_k_ α_k_ e_u_^(k)^ final (pooled) |
| s_ui_ = ē_u_·ē_i_ | Predicted score |
| H = (V, E_H_) | Hypergraph (sessions, categories, tags); D_v_, D_e_ degree matrices |
| Γ = (N, v) | Cooperative game: players N, value function v: 2^N^ → ℝ |
| φ_j_, φ̂_j_ | Shapley value (exact / Monte-Carlo estimate) |

### 2 The three games — formal definitions

#### 2.1 G1 Edge-level game (pairwise message weights) *— main novelty claim*
**Game.** For each user u, define the local game Γ_u_ = (N(u), v_u_): players are u’s neighbors (interactions); a coalition S ⊆ N(u) is a subset of interactions; v_u_(S) measures how good the representation built from S is.

**Value functions (the review’s key ask).** Three instantiations, to be selected per level:

| # | v_u_(S) | Intuition | Cost | Use |
| --- | --- | --- | --- | --- |
| v^cons^ | −‖ (1/\|S\|) Σ_i∈S_ e_i_ − ē_u_ ‖² | Consistency: a coalition is valuable if it reconstructs the full-model user embedding | O(\|S\|·d), free | **Default (G1)**, recomputed with period |
| v^pref^ | α·(1/\|S\|)Σ_i∈S_ s_ui_ + β·tail(S) + γ·div(S) | Preference-aware (DyHuCoG-style): accuracy + tail-share + diversity — prevents popularity free-riding | O(\|S\|·d) | **Default (G2)** and when fairness is the goal |
| v^ndcg^ | NDCG@K(u; model trained on S) | Ground-truth utility | Full retraining — offline only | G3 (data valuation), audit |

$$
φ_i_(u) = Σ_S ⊆ N(u)∖{i}_ **|S|!(|N(u)|−|S|−1)! / |N(u)|!** · [ v_u_(S ∪ {i}) − v_u_(S) ]  
*— the unique allocation satisfying efficiency, symmetry, dummy, additivity (Shapley 1953)*
$$

**Edge weights.** Inject φ̂ as a modulation of the LightGCN normalization, with an interpolation λ that makes the game optional:

$$
w_ui_ = (1/√(d_u_ d_i_)) · [ (1−λ) + λ·σ(φ̂_ui_/τ) ]    *λ ∈ [0,1], σ = sigmoid, τ = temperature*
$$

> **Proposition 1 (recovery).**
> For λ = 0, CoopGCN’s Channel A reduces exactly to LightGCN. Moreover, if all neighbors are symmetric in φ̂ (as the symmetry axiom forces for interchangeable players), the modulation is uniform — so
> **φ̂-weighted aggregation is a generalization of LightGCN++-style scalar weighting**
> , with the theory telling us when weighting is unnecessary. This is the “φ̂ vs learned scalar” bridge reviewers will probe.

#### 2.2 G2 Hyperedge-level game (group channel) *— DyHuCoG’s idea, extended*
**Game.** For each hyperedge h ∈ E_H_ (a session, category, tag — constructed from train data only), players = member interactions; coalition game Γ_h_ = (h, v_h_) with preference-aware utility:

$$
v_h_(S) = α·(1/|S|²) Σ_j,k∈S_  aff(j,k) + β·tail(S) + γ·div(S) *— aff = pairwise affinity (co-occurrence or embedding similarity)*
$$

**Hyperedge weight** (aggregated Shapley credit of the group):

$$
β_h_ = σ( (1/|h|) Σ_j∈h_  φ̂_j_(h) / τ_h_ )   ⇒    
e_v_^(k+1)^ = Σ_h∋v_ (β_h_/√(D_v_ D_h_)) · Σ_v′∈h_ (1/√D_h_) e_v′_^(k)^
$$

β_h_ ≡ 1 recovers standard hypergraph CF (DHCN/HCCF-style) — the ablation that separates “hypergraph helps” from “Shapley-weighted hypergraph helps”, which is the actual novelty over DyHuCoG at this level.

#### 2.3 G3 Data-level game (training-set curation)
**Game.** Players = training interactions; v(S) = ranking metric of a model trained on S, evaluated on a *temporal* validation set (v^ndcg^ above). Estimated with **TMC-Shapley** (truncated MC) or the hierarchical HCDV method. Sample weight:

$$
γ_ui_ = 1 + κ · rank(φ̂_ui_)  ∈ [1, 1+κ] *— reweights the loss; optionally prune the bottom p% (e.g. p=5)*
$$

### 3 Loss functions
$$
ℒ_rank_ = −(1/|B|) Σ_(u,i)∈B_ [ log σ(s_ui_) + Σ_j∈Neg(u)_  w_j_  log(1 − σ(s_uj_)) ] *— gBCE, 256 negatives, sample weights w_j_ ∝ γ (G3)*  
  
ℒ_cl_ = −log [ exp(sim(ē_u_, e^g^_u_)/τ) / Σ_u′_  exp(sim(ē_u_, ē_u′_)/τ) ] *— InfoNCE, local (Channel A/B) vs global (Channel C, SVD view)*  
  
ℒ_game_ = ‖ σ(a_ui_) − φ̂̄_ui_ ‖² *— consistency: learned attention a_ui_ ← EMA of φ̂ (stop-grad), non-stationarity fix*  
  
ℒ = ℒ_rank_ + λ_1_ ℒ_cl_ + λ_2_ ℒ_game_ + λ_3_‖Θ‖²
$$

The ℒ_game_ term is the “consistency loss” the review flagged as missing: it makes the model learn to reproduce the Shapley credit end-to-end (so inference needs no game computation), with an EMA target for stability and stop-gradients on φ̂.

### 4 Complexity & feasibility
| Game | Per-refresh cost | Typical numbers | Budget |
| --- | --- | --- | --- |
| G1 (edges) | O(T · k · d) per user | T=100 permutations, k=32 coalition, d=64 → ~200K FLOPs/user | < 20% wall-clock (refresh every 5 epochs) |
| G2 (hyperedges) | O(T · \|h\| · d) per hyperedge | \|h\| ≤ 50; #hyperedges ≪ #edges | negligible |
| G3 (data) | TMC-Shapley: O(T′·n·train) | Offline every 10 epochs; HCDV variant 10–100× faster | offline |

- **Restricted coalitions:** \|N(u)\| capped at k=32 via stratified sampling (top-recent + random) — bounds G1 exactly.
- **Variance control:** report φ̂ std over 5 runs; use antithetic permutation pairing (pair π with reverse(π)) or Owen/stratified sampling if std > 0.1·mean.
- **Efficiency restoration:** after truncation, renormalize so Σφ̂ = v(N) (the code does this) — preserves the efficiency axiom approximately.
- **Non-stationarity:** φ̂ refreshed every 5 epochs; EMA (rate 0.9) + stop-grad in ℒ_game_.

### 5 PyTorch pseudo-code

#### 5.1 MC-Shapley edge weighting (G1)
```python
# ============================================================
# G1 — MC Shapley edge weighting (restricted coalitions)
# ============================================================

def value_fn_consistency(s_emb, e_full):
    """v(S): negative L2 between the coalition-built profile and the
    full-model embedding (stop-grad). Constant offsets cancel in
    marginals, so v(empty)=0 w.l.o.g."""
    return -((s_emb - e_full.detach()).pow(2).sum(-1))

@torch.no_grad()
def mc_shapley_edge(e_u, e_nbrs, T=100, max_k=32, eps=1e-3):
    """Players = neighbors of user u (<= max_k). Returns phi (k,).

    phi[i] = avg over T permutations of the marginal contribution
    of neighbor i to the coalition that precedes it (MC-Shapley),
    with truncation once marginals vanish.
    """
    k = e_nbrs.shape[0]
    if k > max_k:                          # restricted coalition
        idx = torch.randperm(k)[:max_k]
        e_nbrs, k = e_nbrs[idx], max_k

    phi = torch.zeros(k, device=e_nbrs.device)
    for _ in range(T):
        order = torch.randperm(k)
        s = torch.zeros_like(e_u)          # coalition accumulator
        for pos, i in enumerate(order):
            v_before = value_fn_consistency(s, e_u)
            s = s + e_nbrs[i]              # add player i
            v_after  = value_fn_consistency(s, e_u)
            phi[i]  += (v_after - v_before) / T
            if pos > k // 2 and abs(v_after - v_before) < eps:
                break                      # truncation
    # restore efficiency: sum(phi) = v(full coalition) = -||sum(e_i) - e_u||^2
    phi *= value_fn_consistency(e_nbrs.sum(0), e_u) / (phi.sum() + 1e-9)
    return phi

def edge_weights(phi, deg_u, deg_i, lam=0.5, tau=0.1):
    """w_ui = LightGCN weight x (1-lam + lam*sigmoid(phi/tau)).
    lam=0  ->  exactly LightGCN (recovery property, see Prop. 1)."""
    base = 1.0 / torch.sqrt(deg_u * deg_i)
    return base * (1.0 - lam + lam * torch.sigmoid(phi / tau))
```

#### 5.2 CoopGCN training loop
```python
# ============================================================
# CoopGCN — training loop (channels A + B + C, three games)
# ============================================================

class CoopGCN(nn.Module):
    def __init__(self, n_user, n_item, dim=64, L=3,
                 lam=0.5, tau=0.1, period=5, ema_rate=0.9):
        super().__init__()
        self.E_u = nn.Embedding(n_user, dim)   # user embeddings
        self.E_i = nn.Embedding(n_item, dim)   # item embeddings
        self.attn = nn.Linear(2 * dim, 1)      # learned attention (for L_game)
        self.lam, self.tau = lam, tau
        self.period, self.ema_rate = period, ema_rate
        self.phi_bar = None                    # EMA of Shapley weights

    def propagate(self, w_edges, w_hyper):
        # Channel A: norm-aware LightGCN w/ Shapley edge weights (G1)
        # Channel B: hypergraph GCN w/ Shapley hyperedge weights (G2)
        # returns pooled embeddings e_u, e_i (dim,)
        ...

    def forward(self, u, i, neg, w_edges, w_hyper):
        e_u, e_i = self.propagate(w_edges, w_hyper)
        return (e_u[u] * e_i[i]).sum(-1), (e_u[u] * e_i[neg]).sum(-1)

# --- refresh Shapley weights every `period` epochs, with EMA ---
def refresh_shapley(model, graph, hypergraph):
    phi_edges = {}
    for u, nbrs in graph.neighbors.items():
        e_u   = model.E_u(torch.tensor(u))
        e_nbrs = model.E_i(torch.tensor(nbrs))
        phi_edges[u] = mc_shapley_edge(e_u, e_nbrs)          # G1
    phi_hyper = mc_shapley_hyper(model, hypergraph)           # G2
    if model.phi_bar is None:
        model.phi_bar = phi_edges
    else:
        model.phi_bar = {u: model.ema_rate * model.phi_bar[u]
                            + (1 - model.ema_rate) * phi_edges[u]
                         for u in phi_edges}                  # non-stationary fix
    w_edges = {u: edge_weights(model.phi_bar[u], deg_u[u], deg_i)
               for u in model.phi_bar}
    return w_edges, phi_hyper

# --- one epoch ---
for epoch in range(EPOCHS):
    if epoch % model.period == 0:
        w_edges, w_hyper = refresh_shapley(model, graph, hypergraph)

    for batch in loader:
        s_pos, s_neg = model(batch.u, batch.i, batch.neg, w_edges, w_hyper)
        L_rank = gbce(s_pos, s_neg, n_neg=256)                # W7 fix
        L_cl   = info_nce(model.E_u.weight, e_svd)            # Channel C
        a = torch.sigmoid(model.attn(torch.cat([e_u, e_i], -1)))
        L_game = F.mse_loss(a, model.phi_bar[u, i].detach())  # consistency
        L = L_rank + lam1 * L_cl + lam2 * L_game + lam3 * reg(model)
        L.backward(); opt.step()

# --- G3: data valuation (offline, every M epochs) ---
if epoch % 10 == 0:
    gamma = tmc_shapley_data(model, train_loader, val_loader) # TMC-Shapley
    apply_sample_weights(loader, gamma)                       # reweight / prune
```

Full code would also include the hypergraph channel forward (message in §2.2) and the SVD view of Channel C (LightGCL recipe: truncated SVD of the normalized adjacency, rank q=5, precomputed once).

### 6 The experiment matrix (harness design)

#### 6.1 Datasets & hyperedge construction (leakage-safe: build from train only)
| Dataset | Hyperedges | Construction rule | Why it fits |
| --- | --- | --- | --- |
| Yelp2018 | Business categories | Category membership per business | Natural group signal; strong category structure |
| Amazon-Book | Book categories / genres | Category membership | Sparse — group signal matters most |
| Gowalla | Spatial clusters | k-means on check-in coordinates (k≈500) | Geo groups proxy social circles |
| MovieLens-1M | Genres | Genre membership | Dense control; timestamps enable clean temporal split |
| LastFM | Tags | Item tags | Tag co-occurrence as hyperedges |

#### 6.2 Leakage audit (Step 0.5 — the review’s addition)
For every dataset, train LightGCN under **(a) random split** (classic protocol) and **(b) global temporal split**. Report ΔR@20 — the inflation factor. All CoopGCN results are reported under (b); (a) is reported only as a labeled diagnostic. This is the CUISINART (KDD’24) lesson, baked into the protocol.

#### 6.3 Configurations (each is a separate ablation row)
| # | Config | Purpose |
| --- | --- | --- |
| C0 | LightGCN (vanilla, tuned hard) | Floor; baseline-tuning symmetry check |
| C1 | **Baseline+** = LightGCN++ norms + gBCE/256 + LightGCL view (no game) | The “no-game CoopGCN” — strong reference; every game module must beat its ablation against C1 |
| C2 | C1 + G1 (Shapley edges) | Main novelty claim |
| C3 | C1 + G2 (Shapley hyperedges) | DyHuCoG-diff ablation (also vs β≡1 hypergraph) |
| C4 | C1 + G1 + G2 | Joint effect |
| C5 | C1 + G1 + G2 + G3 | Data valuation adds |
| C6 | **CoopGCN full** (C5 + ℒ_game_ consistency) | Submission model |

#### 6.4 THE ablation (the review’s make-or-break): φ̂ vs attention
| Weighting for Channel A | Question answered |
| --- | --- |
| Uniform (LightGCN) | Floor |
| Degree-norm only (LightGCN default) | Topological baseline |
| LightGCN++ scalar (learned α,β,γ) | Learned, non-game weighting |
| GAT-style attention | Learned, per-neighbor attention |
| **φ̂ (Shapley, G1)** | Game-theoretic weighting |

Verdict metrics: R@20, N@20, **TR@20 (tail recall)**, Coverage@20, Gini, popularity gap, diversity; plus a **noise-robustness test** (inject 5/10/20% random interactions, measure degradation — G3’s payoff). 3 seeds, paired t-test, not just means. If φ̂ wins NDCG — great. If it wins only tail/robustness, that is still a publishable story: *trustworthy, robust graph recs via axiomatic credit*.

### 7 The review’s 8 open questions — answered
| Review question | Answer |
| --- | --- |
| 1. Exact v(S) for G1/G2 with math? | §2.1–2.2: v^cons^ (default G1), v^pref^ (default G2, with tail+diversity terms), v^ndcg^ (offline G3). The tail/diversity terms block popularity free-riding. |
| 2. Coalition restriction strategy for hubs (100+ neighbors)? | Cap at k=32: top-most-recent + stratified random (diversity of interaction types). Sensitivity analysis over k ∈ {16, 32, 64}. |
| 3. MC estimator variance? | Report std over 5 runs; antithetic pairing; T=100–200; Owen/stratified if needed. Truncation ε=1e-3 with renormalization. |
| 4. Axiom preservation under approximation? | Efficiency restored by renormalization (Σφ̂ = v(N)); symmetry holds by MC construction; dummy players get ≈0 φ̂ (their marginals vanish). |
| 5. Symmetry for duplicate interactions (same item, different time)? | Symmetry is desirable only for truly interchangeable players. Timestamps break symmetry: interactions are dated, v^pref^ includes recency weight; duplicates become distinct players with distinct φ̂. |
| 6. Training stability (φ̂ recomputed → non-stationary targets)? | EMA of φ̂ (rate 0.9) as the target of ℒ_game_, stop-grad on the target; refresh every 5 epochs, not every step. |
| 7. Cold-start G5 — how are side features players? | Moved to future work (§8). Sketch: feature groups (categories, text clusters) as players of a feature game; φ̂ initializes new-node embeddings. Not in v1. |
| 8. Complexity budget formal? | O(T·k·d) per user refresh + O(E·L·d) propagation; measured wall-clock % per component reported in every table. Budget <20% or the design changes. |

### 8 Scope control (the review’s recommendation, adopted)
**Core paper (v1):** G1 + G2 + G3, Channels A+B+C, ℒ_game_, leakage-free protocol, the φ̂-vs-attention ablation, robustness study.

**Future work (discussion section only):** G4 popularity-bias correction as a standalone game, G5 cold-start feature games, G6 multi-behavior fusion, G7 Shapley-driven negative sampling, G8 explanation outputs, temporal/sequential extension (SASRec/HSTU-class backbone).

> **Publishability bar (from the review):**
> if G1+G2 beat uniform AND learned attention on ≥3 datasets under temporal splits on tail/robustness metrics → SIGIR’27 / RecSys’27 material. If only G2 works → incremental over DyHuCoG; reposition as “edge+data-level Shapley” or don’t submit.

### 9 Sources
1. Shapley (1953) — *A Value for n-Person Games*
2. Owen (2014) — Monte-Carlo Shapley estimation; [Ghorbani & Zou (ICML’19)](https://arxiv.org/abs/1904.02868) Data-Shapley; [Jia et al. (2019)](https://arxiv.org/abs/1907.02846) TMC-Shapley; [HCDV (2026)](https://arxiv.org/html/2512.19363) hierarchical valuation
3. [DyHuCoG (2025)](https://oaji.net/articles/2025/3603-1768748355.pdf) — preference-aware MC Shapley hyperedge weights
4. [LightGCN++ (RecSys’24)](http://dmlab.kaist.ac.kr/~kijungs/papers/lightgcnppRecSys2024.pdf) — norm scaling & neighbor weighting
5. [LightGCL (ICLR’23)](https://arxiv.org/abs/2302.08191) — SVD global contrastive view
6. [gSASRec (RecSys’23)](https://arxiv.org/abs/2308.09756) — gBCE, overconfidence fix
7. [CUISINART (KDD’24)](https://arxiv.org/abs/2308.03953) — evaluation leakage in graph CF
8. [HCCF (SIGIR’22)](https://arxiv.org/abs/2204.12200), [DHCN (AAAI’21)](https://arxiv.org/abs/2012.06852) — hypergraph CF propagation
9. [Beyond Shapley Values (2025)](https://arxiv.org/pdf/2506.13900) — value-function design is the real decision
10. [Are We Really Making Much Progress? (RecSys’22)](https://dl.acm.org/doi/10.1145/3523227.3546767) — baseline tuning symmetry

_ CoopGCN project docs: *specs/CoopGCN_Spec.md* (analysis & blueprint) → this file (formalization) → experiment harness (next) · Compiled Aug 2026. _

