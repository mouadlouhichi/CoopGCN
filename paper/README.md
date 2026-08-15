# CoopGCN manuscript

The authoritative manuscript source is [`coopgcn_cas.tex`](coopgcn_cas.tex),
typeset with Elsevier's `cas-dc` class. The scientific revisions in the current
source are journal-independent: measured evidence is separated from design
history, and unsupported claims have been removed rather than hidden in
limitations.

## Files

| File | Purpose |
|---|---|
| `coopgcn_cas.tex` | Authoritative manuscript source |
| `references.bib` | Bibliography |
| `cas/` | Elsevier CAS class and bibliography files |
| `build_preview_pdf.py` | No-LaTeX preview renderer |
| `CoopGCN_CAS_preview.pdf` | Generated preview; regenerate after source edits |
| `measured_tradeoff.csv` | Generated NDCG/Coverage rank-correlation sensitivity output |
| `../data/measured_results.csv` | Machine-readable single-run measured table |
| `../scripts/analyze_measured_tradeoff.py` | Standard-library sensitivity analysis generator |
| `coopgcn_array.tex` | Older manuscript retained only for history; do not submit |
| `figures/Figure_1.png` | Fallback rendering of the architecture figure |

## Evidence policy in the current revision

The manuscript's main results now use only the retained **single-run measured
record** summarized in `specs/CoopGCN_Empirical_Review_Theme.md`:

- 5 datasets: ML-100K, ML-1M, Gowalla, Yelp2018 and Amazon-Book;
- 10 models;
- NDCG@20, Tail Recall@20 and Coverage@20;
- one point estimate per model–dataset cell, with no variance estimate.

The projected CSVs under `main_results/expected_*.csv` are design-history
artifacts, not experimental evidence. Their leaderboards, component ablation,
noise curves, gain table and derived figures have been removed from the
manuscript's results section because executed CoopGCN values contradict them.
They must not be cited as findings.

The current measured evidence supports only a preliminary, regime-dependent
accuracy–exposure trade-off. It does **not** establish:

- universal ranking-accuracy superiority;
- measured component attribution;
- random-noise or adversarial robustness;
- attribution faithfulness/explainability;
- statistical significance;
- zero latency or a wall-clock training budget.

## Main peer-review corrections

1. **Measured evidence is primary.** The projection-led Tables 8–10/12 and
   Figures 2–7 were removed from the results narrative.
2. **Measured scope is consistent.** The manuscript now presents the complete
   retained 5-dataset × 10-model table instead of describing five/ten while
   displaying only three/seven.
3. **GAT-CF status is consistent.** It is measured, but the central comparison
   is only partly answered: clean tail/coverage are available, comparative
   noise resilience and tuning traces are not.
4. **GAT-CF collapse is not over-interpreted.** The text explicitly identifies
   sparse-data collapse as potentially caused by tuning or implementation and
   requests learning-rate sweeps and training curves.
5. **No stale Amazon-Book gain remains.** Both 12.1% and projection-dependent
   26.3% headline claims were removed.
6. **Ablation and robustness claims were withdrawn.** The projected 62.2% G1
   attribution and 5.4–5.7% degradation figures are named only to explain why
   they cannot be treated as findings.
7. **Theory is narrowed.** Proposition 2 is a per-edge, per-layer Channel-A
   bound relative to a non-zero reference—not a full-model or NDCG robustness
   theorem. The degree normalization uses `sqrt(d_u d_i)` consistently.
8. **Characteristic-function boundaries are defined.** Every game defines
   `v(empty)=0`; diversity is zero for coalitions smaller than two.
9. **Axioms are scoped to credit.** Efficiency/additivity are properties of
   exact Shapley credit, not sigmoid-transformed or distilled deployed weights.
   Full propagation weights retain degree normalization.
10. **Serving claims are narrowed.** Inference performs no Monte-Carlo Shapley
    sampling, but residual attention latency is unmeasured.
11. **Statistical limits are prominent.** The abstract, results, conclusion,
    limitations and ethics statement all identify the evidence as single-run
    and non-significant.
12. **Unexecuted XAI is not claimed.** Deletion, insertion, stability and
    explainer comparisons remain a pre-specified protocol.

## Experiments still required

These issues cannot be fixed honestly by editing prose or inventing values:

1. at least five independent seeds or user-level bootstrap intervals;
2. a documented and matched tuning budget, especially for GAT-CF;
3. measured `w/o G1/G2/G3/CL/L_game` ablations;
4. independently retrained 0/5/10/20% random-injection experiments;
5. training time, inference latency, memory and distillation-gap measurements;
6. deletion/insertion/stability attribution evaluation.

Until those runs exist, the manuscript deliberately leaves RQ3–RQ5 open.

## Building

With a TeX distribution, from `paper/`:

```bash
pdflatex coopgcn_cas
bibtex coopgcn_cas
pdflatex coopgcn_cas
pdflatex coopgcn_cas
```

Without TeX, install the Python dependencies (`matplotlib` and `reportlab`) and
run:

```bash
python build_preview_pdf.py
```

The preview renderer parses `coopgcn_cas.tex`; the LaTeX source remains the
submission artifact.
