# Senior Reviewer Audit Report — Round 1 (CoopGCN Codebase)

**Reviewer:** Principal Research Scientist & Senior ML Systems Engineer (Best Model Rigor)  
**Date:** August 10, 2026  
**Target Architecture:** Apple Silicon Mac M4 Pro (48GB RAM, Metal MPS GPU Acceleration)  
**Status:** REQUIRED ENHANCEMENTS IDENTIFIED (Proceeding to Implementer Round 2 Fixes)

---

## 1. Executive Review Summary

We have conducted a thorough code review of the initial `coopgcn/` package (`dataset.py`, `models.py`, `shapley_data.py`, `losses.py`, `evaluator.py`, `trainer.py`, and `visualization.py`). While the core architectural logic correctly formulates the tri-level cooperative game ($\mathbf{G_1, G_2, G_3}$) and the zero-overhead inference bridge ($\mathcal{L}_{\text{game}}$), we identify **4 critical engineering and theoretical enhancements** required to reach top-tier publication and reproducibility quality matching the standards of `signalshap` and *DyHuCoG*.

---

## 2. Identified Technical Issues & Actionable Requirements

### Issue #1 (HIGH): Numerical Stability & Zero-Division Guards in Sparse Hypergraphs
* **Location:** `coopgcn/models.py` -> `ShapleyHypergraphConv.compute_group_shapley` & `MCShapleyEdgeWeighting.compute_mc_shapley`
* **Diagnosis:**
  1. In `ShapleyHypergraphConv`, when a hyperedge coalition has identical or zero-norm embeddings, cosine normalization `F.normalize(h_emb, p=2, dim=-1)` can produce `NaN` vectors if norms are 0.
  2. In `MCShapleyEdgeWeighting`, standard deviation normalization `(phi_est - phi_est.mean()) / (phi_est.std() + 1e-8)` can oscillate if `phi_est.std() == 0` across homogeneous neighbors.
* **Implementer Action Required:**
  - Add explicit zero-norm clamping: `torch.clamp(h_emb.norm(p=2, dim=-1, keepdim=True), min=1e-8)`.
  - Ensure `phi_est` normalization checks `torch.isnan(phi_est).any()` and falls back to zero-mean credit when standard deviation is below `1e-6`.

### Issue #2 (HIGH): Apple Silicon Mac M4 Pro (Metal MPS) Device Compatibility
* **Location:** `coopgcn/models.py` -> `SVDContrastiveView.compute_svd_view` & `coopgcn/trainer.py`
* **Diagnosis:**
  1. Apple Metal (`torch.device('mps')`) requires explicit tensor device marshaling when converting CPU SciPy sparse SVD results to PyTorch buffers.
  2. Certain 64-bit integer index aggregations (`torch.long` in `index_add_`) on MPS must be guaranteed to match the target device index type.
* **Implementer Action Required:**
  - Explicitly wrap SVD output conversions in `.to(device=user_embeds.device)` and assert `torch.backends.mps.is_available()` device reporting in `CoopGCNTrainer`.
  - Verify zero-overhead inference prediction (`model.predict()`) executes on MPS without CPU synchronization bottlenecks.

### Issue #3 (CRITICAL): Formal Test Suite & Shapley Axiom Verification Missing
* **Location:** `tests/`
* **Diagnosis:**
  - To match the academic rigor of `signalshap` and verify theoretical correctness, the repository must include explicit pytest unit tests mathematically asserting:
    1. **The 4 Shapley Axioms:** Efficiency ($\sum \phi_i = v(\mathcal{N})$), Symmetry (interchangeable players get equal $\phi$), Dummy Player (zero marginal contribution $\implies \phi_i=0$), and Additivity.
    2. **Proposition 1 (LightGCN & LightGCN++ Recovery):** Asserting that at $\lambda=0$, Channel A's weights match exact LightGCN symmetric normalization ($1/\sqrt{d_u d_i}$) to within machine epsilon (`1e-6`).
    3. **Proposition 2 (Adversarial Noise Immunity):** Asserting that under consistency utility $v^{\text{cons}}$, injected uncorrelated noise edges receive $\phi_{i^*} \to 0$.
    4. **Step 0.5 Leakage Audit:** Asserting zero intersection between training edges and validation/test edges.
* **Implementer Action Required:**
  - Create `tests/test_propositions.py` and `tests/test_suite.py` with full test coverage and automated AST/execution verification.

### Issue #4 (MEDIUM): Training-Only Degree Cutoff for Tail Item Mask
* **Location:** `coopgcn/dataset.py`
* **Diagnosis:**
  - To prevent evaluation leakage (**W11**), the long-tail item mask (`tail_item_mask`, bottom 80% least popular items) must be computed strictly using **training set degrees** ($d_i^{\text{train}}$), never global degrees.
* **Implementer Action Required:**
  - Explicitly document and assert in `BenchmarkDataset.__init__` that item degree quantiles are calculated exclusively on `self.train_edges`.

---

## 3. Next Steps in Multi-Role Agent Workflow
1. Implementer applies all 4 fixes to `coopgcn/` and creates the test suite in `tests/`.
2. Implementer creates `scripts/run_all.py` and updates `notebooks/coopgcn_run_all.ipynb`.
3. Reviewer conducts Round 2 re-audit and issues final verification verdict.
