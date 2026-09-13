#!/usr/bin/env bash
set -euo pipefail

echo "=== Mini-RAG Reader — Setup ==="

# Detect Python (python3 preferred, fall back to python)
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON="python"
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "-> Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

# Activate venv (supports Linux/macOS and Windows Git Bash)
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
fi

# Install dependencies
echo "-> Installing dependencies..."
pip install --upgrade pip
pip install -e ".[serve,dev]"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Virtual environment: .venv"
echo "Activate with:"
echo "  source .venv/bin/activate      (Linux / macOS)"
echo "  source .venv/Scripts/activate   (Windows / Git Bash)"
echo ""
echo "Next steps:"
echo "  make test         # Run tests"
echo "  make serve        # Start the web server (CPU, no model download)"
echo "  make run          # Ask a question about the sample document"
echo "  make benchmark    # Run the canonical Q&A benchmark"
echo ""
echo "To use a HuggingFace LLM (requires model access):"
echo "  RAG_HF_MODEL=your/model make serve-hf"
