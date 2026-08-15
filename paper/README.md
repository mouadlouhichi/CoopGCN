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
| `CoopGCN_Array_preview.pdf` | Rendered preview (21 pp., two-column) |

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
