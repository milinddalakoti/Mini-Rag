.PHONY: help setup install test serve serve-hf run benchmark lint clean sample-data

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	bash setup.sh

install:  ## Install package with serve + dev dependencies
	pip install -e ".[serve,dev]"

test:  ## Run tests
	python -m pytest tests/ -v

lint:  ## Run linter
	ruff check . && ruff format --check .

serve:  ## Start the web server (CPU, ExtractiveGenerator — no model download)
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

serve-hf:  ## Start server with HF model (set RAG_HF_MODEL beforehand)
	@echo "Set RAG_HF_MODEL before running, e.g.:"
	@echo "  RAG_HF_MODEL=meta-llama/Llama-3.2-3B-Instruct make serve-hf"
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

run:  ## Run CLI on the sample document
	python -m src --pdf data/sample.txt --text --question "What is the CSEP salary threshold?"

benchmark:  ## Run canonical Q&A benchmark
	python -m scripts.run_questions --report reports/metrics.json

sample-data:  ## Verify sample data loads
	python -c "from src.data import load_text; pages = load_text('data/sample.txt'); print(f'Loaded {len(pages)} page(s)')"

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mini-rag-uploads/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
