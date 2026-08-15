# CoopGCN — Array (Elsevier) manuscript

Submission-ready manuscript for **Array** (Elsevier, open access; CiteScore 10.1,
Impact Factor 5.3), prepared against the journal's
[Guide for Authors](https://www.sciencedirect.com/journal/array/publish/guide-for-authors).

## Contents

| File | Description |
|---|---|
| `coopgcn_array.tex` | Manuscript source (`elsarticle.cls`, `final,5p,times,twocolumn` — published Elsevier two-column layout) |
| `references.bib` | 20 references, BibTeX |
| `figures/Figure_1.png` | Monochrome fallback of the TikZ architecture figure (preview builds only) |
| `figures/Figure_2..7.png` | Result figures, copied verbatim from `main_results/figures/` |
| `make_figure1.py` | Regenerates the Figure 1 fallback PNG |
| `build_preview_pdf.py` | Renders a preview PDF without a TeX installation |
| `CoopGCN_Array_preview.pdf` | Rendered preview (13 pp., two-column) |

**Figure 1 is drawn in TikZ, black only**, inline in `coopgcn_array.tex`
(matching the DyHuCoG house style). Compiling with `pdflatex` uses the TikZ
picture; `figures/Figure_1.png` exists only so the no-LaTeX preview build has
an equivalent image.

## Building the authoritative PDF

`coopgcn_array.tex` is the artefact to submit. With a TeX distribution:

```bash
pdflatex coopgcn_array
bibtex   coopgcn_array
pdflatex coopgcn_array
pdflatex coopgcn_array
```

`elsarticle.cls` and `elsarticle-num.bst` ship with TeX Live
(`texlive-publishers`) and are also downloadable from Elsevier.

### About the preview PDF

The sandbox this was prepared in has no reachable TeX mirror, so
`CoopGCN_Array_preview.pdf` was produced by `build_preview_pdf.py`, which parses
`coopgcn_array.tex` directly and lays it out with ReportLab (display equations
rendered via matplotlib mathtext). It **parses the `.tex` rather than
duplicating its content**, so the preview cannot silently drift from the source.

It is a faithful preview, not a substitute for LaTeX output: line breaking,
hyphenation, float placement and math typesetting will differ from `pdflatex`.
Regenerate it after editing the source with:

```bash
python build_preview_pdf.py
```

## Structure

The manuscript follows `specs/CoopGCN_Paper_Structure.md`,
`specs/CoopGCN_Spec.md` and `specs/CoopGCN_Implementation_Spec.md`:

| Spec section | Paper |
|---|---|
| Abstract (Background / Problem / Insight / Method / Results / Significance) | Abstract |
| 1.1 The LightGCN paradigm and its core trade-off | §1.1 |
| 1.2 Four structural failure modes | §1.2 |
| 1.3 The game-theoretic perspective | §1.3 |
| 1.4 Summary of contributions | §1.4 |
| 2.1 GCNs in collaborative filtering | §2.1 |
| 2.2 Hypergraph recommender systems | §2.2 |
| 2.3 XAI and Shapley values | §2.3 |
| 2.4 Popularity bias and robustness | §2.4 |
| — (added) 12-weakness taxonomy W1–W12 | §2.5, Table 1 |
| 3.1 Cooperative games and Shapley definition | §3.1, Eq. (2) |
| 3.2 Four axioms in CF | §3.2, Table 4 |
| 3.3 Proposition 1 (LightGCN/LightGCN++ recovery) | §3.3 + proof |
| 3.4 Proposition 2 (robustness) | §3.4 + proof + Corollary 1 |
| 4.1 Tri-channel overview | §4.1, Fig. 1 (TikZ) |
| 4.2 G1 edge-level game | §4.2, Eqs. (4)–(6) |
| 4.3 G2 hyperedge-level game | §4.3, Eqs. (7)–(9) |
| 4.4 G3 data-level game | §4.4, Eq. (10) |
| 4.5 Multi-task objective | §4.5, Eqs. (11)–(15) |
| 4.6 Zero-overhead inference | §4.6, Eq. (16) |
| 5.1 Restricted coalitions and MC sampling | §5.1, Algorithm 1 |
| 5.2 Amortised complexity table | §5.2, Table 6 |
| 6.1 Leakage audit and temporal protocol | §6.1 |
| 6.2 Benchmark datasets | §6.2, Table 7 |
| 6.3 Baseline competitors | §6.3 |
| 6.4 Multi-dimensional metrics | §6.4 |
| 6.5 Research questions RQ1–RQ5 | §6.5, answered §7.1–7.5 |
| — (added) 10-row ablation matrix + victory condition | §6.6, Table 8 |
| 7 Results and discussion | §7 |
| 8 Conclusion, limitations and future work | §8.1–8.3 |

## Journal compliance

- Published two-column Elsevier layout (`5p`)
- Numbered sections (1, 1.1, 1.1.1); abstract excluded from numbering
- Highlights list (max 85 characters per bullet target)
- Structured abstract, keywords, highlights-compatible contribution list
- CRediT authorship statement, competing-interest and generative-AI declarations
- Data availability statement with repository link
- Editable text tables (no vertical rules, no cell shading), captions above
- Figures as separate numbered files (`Figure_1.png` …) at 200–300 dpi
- Equations displayed, numbered consecutively, variables italicised
- `elsarticle-num` numbered reference style

## Data provenance

All reported values come from `main_results/` in this repository
(`expected_results.csv`, `expected_ablation.csv`, `expected_noise.csv`).
Provenance markers are carried through into the manuscript exactly as recorded
in those files and are explained in §5.4 (*Provenance of reported numbers*):

- `†` — CoopGCN projections at the target configuration (d=64, 1000 epochs, patience 50)
- `*` — estimated from published margins on adjacent datasets
- unmarked — values reported in the cited publications

§8 (*Limitations*) states plainly that the results are single point estimates
without seed variance, that the 6.7% ML-1M margin may fall within run-to-run
variation, and that a learned-attention (GAT-CF) baseline is absent. See
`main_results/ANALYSIS.md` for the full audit of the underlying result records.

## Compliance with Elsevier Structured Peer Review prompts

The manuscript is audited against every prompt in `PeerReviewGuidance.pdf`
(Elsevier, *Structured Peer Review prompts for Original Research article
type*, April 2026). Section numbers refer to `coopgcn_array.tex`.

| Prompt | Where it is addressed |
|---|---|
| **Abstract** — reflects all essential aspects incl. all major results *and limitations* | Structured abstract with `Background / Problem / Insight / Proposed method / Key results / **Limitations** / Significance`. The Limitations sentence names the single-run projections, the absence of CIs/significance tests, and the unresolved central ablation. |
| **Introduction** — background and literature up to date; scientific rationale explained | §1.1–§1.3. Opens with the concrete empirical finding (six of seven baselines have TR@20 exactly 0.0000), then derives all three symptoms from one root cause; §1.3 argues why the axiomatic route differs from attention. References span 1953–2026, with 7 entries from 2023 onward. |
| **Introduction** — (primary and secondary) objectives clearly stated *at the end* | §1.4, `\paragraph{Objectives}`, immediately before the roadmap: one primary objective and four numbered secondary objectives. |
| **Methods** — theory, applicability, modelling described in enough detail to replicate | §3 (theory + two propositions with proofs), §4 (architecture, explicit characteristic functions, Algorithm 1), §5 (complexity), and **§6.4 Implementation and reproducibility settings** — $d$, layers, init, optimiser, LR, weight decay, batch size, epochs, patience, negatives, $\lambda$, $L$, $R$, $P$, $M$, $\eta$, and which values are validation-tuned. |
| **Methods** — experimental design identified; population described | §6.1 leakage-audited temporal protocol; §6.2 dataset table with users/items/density; §6.3 baselines with equal tuning budget. Sample-size estimation and MCID are clinical-trial prompts and are not applicable to offline benchmark evaluation. |
| **Methods** — statistical analyses, controls, sampling, reporting described | §6.4 (sampling, controls: shared codebase so backbone/sampler/evaluator are identical across conditions) and §6.8 `Statistical reporting`. |
| **Results** — number of tables/figures appropriate to visualise findings | 12 tables and 7 figures, one per research question plus taxonomy/notation/axioms/complexity support; Table 8 carries the headline comparison, Figs. 2–7 the per-RQ evidence. |
| **Results** — whether additional sub-analyses / CIs / effect sizes / sensitivity are needed | §6.8 `Statistical reporting` states that no significance claim is made anywhere, distinguishes the suggestive small-margin regime (6.7 %) from the load-bearing large-effect regime (4.2×–56.7×), and specifies the powered $n \ge 5$ seed study as future work §8.3. |
| **Discussion** — interpretation supported by the data and design | §7.7 `Discussion: what the axioms buy` is scoped to the ablation and robustness evidence; Amazon-Book claims are deliberately framed against LightGCN/DyHuCoG only, because the SGL/SimGCL cells are flagged unreconcilable. |
| **Discussion** — limitations of theory, methods *and argument* clearly emphasised | §8.2, six labelled limitations: unresolved central comparison; provenance and variance; ablation nesting; unmeasured efficiency; metric interpretation against small denominators; scope. |
| **References** — most up-to-date and relevant | 20 entries, 1953–2026; every entry cited, no orphans (audited). |
| **Overall** — novelty/importance; reproducibility; educational goals | §1.4 contributions; artefacts, seeds and result records released (Data availability); the taxonomy in §2.5 and the axiom treatment in §3.2 are written to be usable as teaching material. |
| **Overall** — ethical / integrity concerns | **`Ethics and research integrity`** section before the AI declaration: public non-identifying datasets, no human subjects, explicit statement of which cells are projected vs estimated, unreconcilable inherited baselines reported rather than dropped, and a societal-impact note on credit scores as sensitive artefacts. |
| **Title** (optional prompt) — accurate, sufficient, discoverable | Names the method, the mechanism (axiomatic credit assignment / cooperative game theory), the model family (GCN) and the two claimed properties (robust, preference-aware). |

Audit command (all green): numeric integrity 142/142 literals traceable to
`main_results/*.csv`, 0 missing citations, 0 uncited entries, 0 dangling
refs, 0 unbalanced environments, 0 sparse pages, no colour packages.
