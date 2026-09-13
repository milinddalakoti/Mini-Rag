# Mini-RAG Reader

A retrieval-augmented generation system that lets you upload a PDF or text file and ask it questions. It finds relevant passages, extracts answers, and cites exactly where each answer came from.

Built because I wanted to see if I could ship a working RAG pipeline end-to-end in a week — and it turned out to be surprisingly straightforward.

## How it works

The pipeline has four stages:

1. **Load** — extracts text from PDFs (using `pypdf`, falls back to `pdfplumber` if needed) or reads plain text files.
2. **Chunk & embed** — splits text into ~500-character segments, converts each to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`, and stores them in a FAISS index.
3. **Retrieve** — when you ask a question, it embeds the question and finds the top-4 most similar chunks by cosine distance.
4. **Generate** — either extracts an answer from the retrieved chunks (`ExtractiveGenerator`, no model download) or generates one using a Hugging Face LLM (`Generator`, requires model access).

If the HF model can't load — gated repo, no network, OOM — it silently falls back to extractive mode and keeps working.

## Getting started

```bash
# Clone
git clone https://github.com/milinddalakoti/Mini-Rag.git
cd Mini-Rag

# One-command setup (creates .venv + installs everything)
bash setup.sh
# Or: make setup
# Or: pip install -e ".[serve,dev]"

# Verify it works
make test
```

Then start the web server:

```bash
# CPU — works immediately, no model download needed
make serve
# Or: uvicorn src.serve:app --host 0.0.0.0 --port 8000

# With GPU (if you have one and want HF generation)
RAG_HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct make serve-hf
```

Open `http://localhost:8000` in your browser. Upload a PDF or text file, ask a question, and get an answer with citations.

Or use the CLI:

```bash
# One-shot Q&A (ExtractiveGenerator — no model download)
python -m src --pdf data/sample.txt --text --question "What is the CSEP salary threshold?"

# With an HF model
python -m src --pdf test.pdf --hf-model Qwen/Qwen2.5-1.5B-Instruct --question "..."
```

## Architecture

```
src/
├── data/        ← PDF loading, text extraction, chunking
├── features/    ← Embedding (sentence-transformers), FAISS indexing
├── generator/   ← ExtractiveGenerator (no LLM) or Generator (HF transformers)
├── retriever/   ← FAISS wrapper, top-k retrieval
├── reader.py    ← Orchestrates the full pipeline
├── serve.py     ← FastAPI server + web UI
├── config.py    ← Pydantic config with auto GPU detection
└── __main__.py  ← CLI entry point
```

Key files:
- `configs/baseline.yaml` — default configuration (model, chunk size, top-k, etc.)
- `Makefile` — shortcuts for setup, test, serve, run, benchmark
- `setup.sh` — one-command environment setup
- `Dockerfile` + `docker-compose.yml` — containerized deployment

## API

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/` | GET | — | Web UI |
| `/health` | GET | — | Health check |
| `/upload` | POST | multipart form | Upload a PDF or text file |
| `/load` | POST | `{"pdf_path": "..."}` | Load a local PDF |
| `/ask` | POST | `{"question": "...", "top_k": 4}` | Ask a question |
| `/clear` | POST | — | Reset and clear uploads |

Example with curl:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the CSEP salary threshold?"}'
```

## Configuration

Edit `configs/baseline.yaml` or use environment variables:

- `RAG_HF_MODEL` — HuggingFace model ID for LLM generation. If unset or the model fails to load, uses `ExtractiveGenerator`. `Qwen/Qwen2.5-1.5B-Instruct` is public and needs no token.
- `MINI_RAG_UPLOAD_DIR` — Where uploaded files go (default: `~/.mini-rag-uploads`)

The system auto-detects CUDA GPUs and uses them for both embeddings and LLM inference. Falls back to CPU automatically.

**Gated models** (like `meta-llama/Llama-3.2-3B-Instruct`) require authentication:
```bash
huggingface-cli login    # paste your token
# or
export HF_TOKEN=<your-token>
```
Then open the model page on the Hub and click **Agree & access** to accept the license.

## Under the hood

- **Chunking**: sliding window with sentence-boundary snapping — chunks never cut through sentences mid-way
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- **Index**: FAISS `IndexFlatIP` (cosine similarity)
- **Extractive generator**: scores sentences by keyword overlap, extracts job title patterns (e.g. "Sales Associate – Zudio") for employment documents
- **GPU detection**: checks `torch.cuda.is_available()` at config load time; `torch_dtype` auto-selects `bfloat16` on CUDA, `float32` on CPU

## Troubleshooting

**Server starts but answers are wrong or generic.** The chunks might not contain the answer. Try increasing `top_k` in `configs/baseline.yaml` or checking that your document was loaded correctly.

**Model download fails.** If `RAG_HF_MODEL` is set but the model can't download, the system falls back to `ExtractiveGenerator` automatically. Check your internet connection and whether the model is gated.

**No GPU detected despite having a CUDA card.** Make sure `torch` and `torchvision` are installed with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

**UnicodeDecodeError on text files.** The loader uses UTF-8 by default. If your file has a different encoding, convert it first.

## Dependencies

- Python 3.11+
- `sentence-transformers>=3.0` — embeddings
- `faiss-cpu>=1.8` — vector index
- `pdfplumber>=0.10` — PDF fallback extraction
- `pypdf>=4.0` — primary PDF extraction
- `pydantic>=2.6`, `pyyaml>=6.0`, `numpy>=1.26` — config and math
- `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `python-multipart>=0.0.9`, `aiofiles>=23` — web server
- `pytest>=8`, `pytest-cov>=5`, `ruff>=0.6` — dev tools
- `torch>=2.0`, `transformers>=4.42` — optional HF generation (`pip install -e ".[hf]"`)

## License

MIT
