# CoopGCN — Local Benchmark Execution Guide (Mac M4 Pro 48GB / Linux GPU)

This guide explains how to run the automated tests, train all five collaborative filtering baselines, run **THE Central Make-or-Break Ablation**, evaluate adversarial noise immunity, and generate publication LaTeX tables and figures on your local machine.

---

## 1. Prerequisites & Installation

### Option A: Quickstart Script (Recommended)
You can execute the entire test suite and benchmark automatically using the included shell script:
```bash
chmod +x run_local.sh
./run_local.sh
```

### Option B: Manual Setup
1. Create a Python virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Automated Test Suite (Verifying Shapley Axioms & Propositions)

Run the mathematical verification tests:
```bash
python3 tests/test_propositions.py
python3 tests/test_suite.py
```
- `test_propositions.py` asserts **Shapley Symmetry**, **Dummy Player**, **Proposition 1 (LightGCN/LightGCN++ Recovery)**, **Proposition 2 (Adversarial Noise Immunity)**, and **Step 0.5 Leakage Audit**.
- `test_suite.py` asserts forward propagation and loss calculation across all modules and baselines.

---

## 3. Running the Interactive Jupyter Notebook (Mac M4 Pro Optimized)

To run the complete interactive benchmark with inline figures and tables:
```bash
jupyter notebook notebooks/coopgcn_run_all.ipynb
```
Click **"Run All"**:
- Automatically detects Apple Metal MPS GPU acceleration (`torch.device('mps')`).
- Generates benchmark datasets with leakage verification.
- Trains `LightGCN`, `LightGCN++`, `GAT-CF`, `DyHuCoG`, and `CoopGCN`.
- Displays benchmark summary table with percentage gains.
- Executes **THE Central Make-or-Break Ablation** (Shapley vs. Attention).
- Runs **Adversarial Edge Noise Injection** (0%, 5%, 10%, 20% noise).
- Generates all 4 publication charts in `./figures/`.

---

## 4. Running via Command Line (CLI Automation)

To execute the full benchmark from terminal:
```bash
python3 scripts/run_all.py --dataset Synthetic-Gowalla-Scale --epochs 20 --output_dir results
```
To emit publication LaTeX tables to `./tables/`:
```bash
python3 scripts/emit_tables.py
```
