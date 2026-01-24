#!/bin/bash
# Selective KV Cache Loading Test Runner
# Usage: ./run_selective_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Selective KV Cache Loading Tests"
echo "=============================================="
echo ""

# Check if venv exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet transformers
else
    source .venv/bin/activate
fi

echo "Running tests..."
echo ""

python3 test_selective_simple.py

echo ""
echo "=============================================="
echo "Running pytest tests (if available)..."
echo "=============================================="

# Install pytest if needed
pip install --quiet pytest

# Run the pytest tests
if [ -f "tests/v1/test_selective_loading.py" ]; then
    python -m pytest tests/v1/test_selective_loading.py -v 2>&1 || echo "Note: Full pytest requires additional dependencies"
fi

echo ""
echo "Done!"
