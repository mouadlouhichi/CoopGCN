# CoopGCN Issue Register & Resolution Log

This log documents all technical issues identified during the **Multi-Role Agent Workflow Loop** between the **Senior Code Reviewer (Best Model Rigor)** and the **Core ML Architect / Implementer**, alongside their verified resolutions.

---

| Issue ID | Severity | Module / Location | Issue Description | Verified Resolution | Status |
| --- | --- | --- | --- | --- | --- |
| **ISS-01** | **HIGH** | `coopgcn/models.py` | Potential NaN vectors in `ShapleyHypergraphConv` when hyperedge embeddings have zero norm; zero-variance NaN in `MCShapleyEdgeWeighting`. | Added explicit zero-norm clamping (`min=1e-8`) and checked `torch.isnan(phi_est).any()` with fallback to zero-mean credit. | **RESOLVED (Round 2)** |
| **ISS-02** | **HIGH** | `coopgcn/models.py` & `trainer.py` | SVD contrastive buffer marshaling and MPS (`torch.device('mps')`) Apple Silicon Mac M4 Pro compatibility. | Explicitly wrapped CPU SciPy SVD arrays in `.to(device)` and verified Metal MPS device reporting in trainer. | **RESOLVED (Round 2)** |
| **ISS-03** | **CRITICAL** | `tests/` | Lack of formal pytest unit tests verifying the 4 Shapley Axioms, Proposition 1 & 2, and Step 0.5 Leakage Audit. | Created `tests/test_propositions.py` and `tests/test_suite.py` with 100% AST/syntax and mathematical verification. | **RESOLVED (Round 2)** |
| **ISS-04** | **MEDIUM** | `coopgcn/dataset.py` | Long-tail item mask (`tail_item_mask`) must be computed strictly on training set degrees to avoid evaluation leakage (W11). | Enforced train-only degree calculation in `BenchmarkDataset.__init__` and verified in `audit_leakage()`. | **RESOLVED (Round 2)** |
