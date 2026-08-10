#!/usr/bin/env bash
# ==============================================================================
# CoopGCN — Automated Local Benchmark & Test Runner (Mac M4 Pro / Linux)
# ==============================================================================

set -e

echo "=============================================================================="
echo "                   CoopGCN Local Execution & Verification                     "
echo "=============================================================================="

# Check Python environment
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "--> Python command: $($PYTHON_CMD --version)"

# 1. Check syntax of all Python scripts
echo "--> Verifying Python AST syntax across code/ and tests/..."
$PYTHON_CMD -c "
import glob, ast
for fn in glob.glob('coopgcn/*.py') + glob.glob('tests/*.py') + glob.glob('scripts/*.py'):
    ast.parse(open(fn).read())
print('✅ All Python modules passed AST syntax verification!')
"

# 2. Run unit tests if torch is installed
if $PYTHON_CMD -c "import torch" &> /dev/null; then
    echo "--> PyTorch detected! Executing mathematical proposition & unit tests..."
    $PYTHON_CMD tests/test_propositions.py
    $PYTHON_CMD tests/test_suite.py
    echo "--> Emitting publication LaTeX tables..."
    $PYTHON_CMD scripts/emit_tables.py
    echo "--> Running short CLI benchmark (epochs=5)..."
    $PYTHON_CMD scripts/run_all.py --dataset Synthetic-Gowalla-Scale --epochs 5 --output_dir results
    echo "🏆 ALL LOCAL EXECUTION CHECKS COMPLETED SUCCESSFULLY!"
else
    echo "ℹ️ PyTorch is not yet installed in this Python environment."
    echo "   To run tests and benchmark on your Mac M4 Pro, run:"
    echo "     pip install -r requirements.txt"
    echo "     jupyter notebook notebooks/coopgcn_run_all.ipynb"
    echo "✅ Codebase syntax and directory verification complete!"
fi
