# CoopGCN preliminary methods manuscript

The authoritative source is [`coopgcn_cas.tex`](coopgcn_cas.tex). The current
revision takes the narrow-scope route requested by peer review: two descriptive
research questions, no robustness/XAI/efficiency claim, and an explicit
separation between ideal Shapley theory and the deterministic proxies used by
the retained checkpoints.

## Evidence status

- Five datasets and ten configurations.
- One retained run per model–dataset cell (seed 42).
- No seed variance, confidence interval, component ablation, corruption study,
  attribution-faithfulness result, or runtime/memory benchmark.
- MovieLens uses global temporal splits; Gowalla/Yelp/Amazon use a pinned
  LightGCN snapshot without timestamps.
- Main measured values: [`../data/measured_results.csv`](../data/measured_results.csv).
- Split/hyperedge manifest: [`../data/dataset_protocol.csv`](../data/dataset_protocol.csv).
- Descriptive cross-model association generator:
  [`../scripts/analyze_measured_tradeoff.py`](../scripts/analyze_measured_tradeoff.py).

## Critical implementation correspondence

The class and method names in the released package retain historical
“Shapley”/“TMC” terminology for checkpoint compatibility. The measured
execution path is:

- **G1:** centered edge–user alignment, tail bonus 0.05, within-neighbourhood
  standardization, EMA 0.85;
- **G2:** mean off-diagonal cosine affinity + 0.5 tail share, sigmoid
  temperature 0.5, channel mixture 0.01;
- **G3:** sigmoid score × tail factor, EMA 0.80, quantile reweighting up to
  1.01; a 5% mask is computed but not applied;
- **ranking:** weighted BPR using one effective negative, although 64 IDs are
  generated;
- **bridge:** an MLP distils the edge proxy; its training target is untempered
  while propagation uses temperature 0.5.

These quantities are deterministic cooperative-game-inspired proxies—not exact,
MC, or TMC Shapley estimators. The ideal-game propositions do not establish
properties of the measured implementation.

## Main review fixes

1. Front matter labels the paper a **preliminary methods study**.
2. Formal RQ3–RQ5 and the unmeasured ablation matrix were removed.
3. Robustness was removed from the title, abstract claims, keywords and measured
   metric set.
4. XAI positioning was reduced to a future conditional test; internal credits
   are not validated explanations.
5. The nine-baseline retained set is authoritative; SGL/SimGCL are related work
   only because no common-protocol record survives.
6. Candidate filtering, user eligibility, TR averaging, score ties, split
   construction, source commit and hyperedge generation are specified.
7. All retained CoopGCN constants and legacy implementation mismatches are
   tabulated.
8. The measured table uses separate NDCG/TR/Coverage panels.
9. Cross-model Spearman associations include average-rank tie handling and
   leave-one-model-out ranges; sparse collapsed-cluster confounding is explicit.
10. Theoretical modulation now matches released code:
    `1 + 2 lambda (sigmoid(credit/tau) - 1/2)`.
11. G3 is described as reweighting only; no pruning/denoising claim remains.
12. Reference encoding and recommendation-specific explanation coverage were
    corrected.

See [`../review/RESPONSE_TO_R4.md`](../review/RESPONSE_TO_R4.md) for the
itemized response.

## Files

| File | Purpose |
|---|---|
| `coopgcn_cas.tex` | Authoritative manuscript source |
| `references.bib` | Bibliography |
| `CoopGCN_CAS_preview.pdf` | Generated no-LaTeX preview |
| `measured_tradeoff.csv` | Generated rank-association/LOMO output |
| `build_preview_pdf.py` | Preview renderer |
| `coopgcn_array.tex` | Historical manuscript; do not submit |

## Build

With TeX:

```bash
pdflatex coopgcn_cas
bibtex coopgcn_cas
pdflatex coopgcn_cas
pdflatex coopgcn_cas
```

Without TeX, install `matplotlib` and `reportlab`, then run:

```bash
python build_preview_pdf.py
```
