# CoopGCN preliminary methods and audit manuscript

The authoritative source is [`coopgcn_cas.tex`](coopgcn_cas.tex). The current
revision does not present the retained benchmark table as a leaderboard:
sparse LightGCN cells fail calibration against verified literature references,
so the table is preserved only as diagnostic provenance.

## Evidence status

- Five datasets and ten retained configurations, one run each (seed 42).
- Sparse LightGCN NDCG is roughly 4–150× below published references. Audit
  found symmetric edge messages were accumulated twice; corrected reruns are
  pending and all retained graph comparisons are invalid.
- No seed variance, norm/component ablation, corruption study, attribution
  test, or runtime/memory benchmark.
- Ideal Shapley theory is separate from deterministic tail-aware proxies used
  by checkpoints.
- The unbounded learnable norm scale, explicit tail terms, and the 32-neighbour
  prefix are identified confounds.

## Confirmatory CSV-first workflow

Expected records are created under [`../experiments/expected/`](../experiments/expected/)
before tables or figures are built. Run:

```bash
python scripts/build_confirmatory_artifacts.py
```

The builder creates LaTeX tables in `experiments/tables/`, SVG status figures
in `experiments/figures/`, and `experiments/ARTIFACT_STATUS.md`. Pending rows
render as pending—not as results. Schemas cover:

- LightGCN calibration;
- multi-seed main runs;
- frozen norm and component/tail ablations;
- neighbour-cap sweeps;
- proxy-versus-permutation-Shapley checks;
- random-edge corruption;
- attribution faithfulness;
- runtime and memory;
- hyperedge footprint and user-degree diagnostics.

## Retained implementation correspondence

- **G1:** centered edge–user alignment, tail bonus 0.05, within-neighbourhood
  standardization, EMA 0.85, first 32 neighbours only;
- **G2:** mean cosine affinity + 0.5 tail share, mixture 0.01;
- **G3:** sigmoid score × tail factor, EMA 0.80, weights up to 1.01; diagnostic
  mask not applied;
- **ranking:** weighted BPR with one effective negative although 64 IDs are
  generated;
- **bridge:** untempered consistency target but temperature-0.5 propagation;
- **norm:** unbounded learned scalar initialized at one.

These are deterministic cooperative-game-inspired proxies—not MC/TMC Shapley.
The sparse hypergraph footprint is under 1% of catalogue slots, and non-zero G1
targets cover at most 27.6% of ML-1M edges.

## Primary review fixes

1. Baseline calibration is now the first evidence gate and fails explicitly.
2. Sparse-regime and cross-model-correlation findings were withdrawn.
3. Mechanism magnitude and norm-scale confounding are quantified.
4. The L=32 target-coverage audit is reported.
5. Tail-objective circularity and z-standardization heterogeneity are explicit.
6. W5/W7/W10 taxonomy rows now reflect the retained implementation.
7. Validation/test false-negative bias and low-history-user diagnostics are
   specified.
8. Coverage bolding was removed from the uncalibrated audit table.
9. Confirmatory records and generated artifact status are machine-readable.

See [`../review/RESPONSE_TO_R5.md`](../review/RESPONSE_TO_R5.md).

## Supporting files

| File | Purpose |
|---|---|
| `coopgcn_cas.tex` | Authoritative manuscript source |
| `retained_run_manifest.yaml` | Known checkpoint configuration and missing fields |
| `../data/measured_results.csv` | Uncalibrated retained audit record |
| `../data/dataset_protocol.csv` | Split/hyperedge protocol |
| `CoopGCN_CAS_preview.pdf` | Generated preview |
| `references.bib` | Bibliography |

## Build

```bash
cd paper
pdflatex coopgcn_cas
bibtex coopgcn_cas
pdflatex coopgcn_cas
pdflatex coopgcn_cas
```

Without TeX, install `matplotlib` and `reportlab`, then run
`python build_preview_pdf.py`.
