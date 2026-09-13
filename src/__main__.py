"""CLI runner: ingest a PDF (or text), ask a question, print answer with citations.

Usage:
    python -m src --pdf data/sample.txt --text --question "What is the CSEP salary threshold?"
    python -m src --pdf data/sample.pdf --question "..."
    python -m src --config configs/baseline.yaml --pdf data/sample.txt --question "..."

Defaults to `ExtractiveGenerator` (no model download). Pass `--hf-model <name>`
to use the Hugging Face pipeline instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.config import Config, load_default_config
from src.data import chunk_text, load_pdf, load_text
from src.features import Embedder, build_faiss_index, embed_chunks
from src.generator import ExtractiveGenerator
try:
    from src.generator import Generator
except ImportError:
    Generator = None
from src.reader import Reader
from src.retriever import Retriever


def _measure(name: str, t0: float) -> float:
    dt = time.perf_counter() - t0
    print(f"[{name}] {dt*1000:.1f} ms", file=sys.stderr)
    return dt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-RAG Reader")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/baseline.yaml bundled with package)",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to a PDF file (or plain-text file if --text is passed)",
    )
    parser.add_argument("--text", action="store_true", help="Treat input as plain text, not PDF")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override config.chunking.chunk_size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Override config.chunking.chunk_overlap")
    parser.add_argument("--top-k", type=int, default=None, help="Override config.retrieval.top_k")
    parser.add_argument(
        "--hf-model",
        default=os.environ.get("RAG_HF_MODEL") or None,
        help="Use HuggingFace Generator with this model name (downloads the model). "
        "Falls back to ExtractiveGenerator if the model cannot be loaded. "
        "(Can also be set via the RAG_HF_MODEL env var.)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write metrics JSON (e.g. reports/metrics.json)",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    """Load config from --config flag or fall back to the bundled default."""
    if args.config:
        return Config.from_yaml(args.config)
    return load_default_config()


def main() -> int:
    parser = _build_parser()

    # Pre-parse --config to load defaults.
    pre_args, _ = parser.parse_known_args()
    config = _resolve_config(pre_args)

    # Apply config values as argparse defaults (CLI args override).
    parser.set_defaults(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        top_k=config.retrieval.top_k,
        embedding_model=config.embedding.model,
    )

    args = parser.parse_args()

    in_path = Path(args.pdf)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    timings: dict[str, float] = {}

    # 1) Load + chunk
    t0 = time.perf_counter()
    pages = load_text(in_path) if args.text else load_pdf(in_path)
    chunks = chunk_text(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        source_path=str(in_path),
    )
    timings["chunk_ms"] = _measure("chunk", t0) * 1000
    print(f"Loaded {len(pages)} page(s), {len(chunks)} chunk(s).", file=sys.stderr)

    # 2) Embed
    t0 = time.perf_counter()
    embedder = Embedder(model_name=args.embedding_model)
    vectors = embed_chunks(embedder, chunks)
    timings["embed_ms"] = _measure("embed", t0) * 1000
    timings["embed_dim"] = embedder.dim

    # 3) FAISS
    t0 = time.perf_counter()
    index = build_faiss_index(vectors)
    retriever = Retriever(index=index, chunks=chunks)
    timings["index_ms"] = _measure("index", t0) * 1000

    # 4) Generator — HF model if requested & available, else extractive.
    t0 = time.perf_counter()
    if args.hf_model and Generator is not None:
        try:
            generator = Generator(model_name=args.hf_model)
            generator_kind = "HF"
        except Exception as exc:
            print(
                f"WARNING: HuggingFace model '{args.hf_model}' failed to load "
                f"({type(exc).__name__}: {exc}). "
                "Falling back to ExtractiveGenerator (no model download needed).",
                file=sys.stderr,
            )
            generator = ExtractiveGenerator()
            generator_kind = "Extractive"
    else:
        if args.hf_model and Generator is None:
            print(
                f"WARNING: --hf-model '{args.hf_model}' was requested but "
                "transformers/torch is not installed. "
                "Falling back to ExtractiveGenerator (no model download needed).",
                file=sys.stderr,
            )
        generator = ExtractiveGenerator()
        generator_kind = "Extractive"
    timings["generator_init_ms"] = _measure("generator_init", t0) * 1000

    # 5) Ask
    t0 = time.perf_counter()
    q_vec = embedder.embed([args.question])
    retrieved = retriever.retrieve(q_vec, top_k=args.top_k)
    text = generator.generate(args.question, retrieved)
    timings["ask_ms"] = _measure("ask", t0) * 1000

    # Output
    print("=" * 72)
    print(f"Q: {args.question}")
    print("=" * 72)
    print(text)
    print("=" * 72)
    print("Top citations:")
    for rc in retrieved:
        print(
            f"  - [{rc.chunk.chunk_id}] page {rc.chunk.page_number} "
            f"score={rc.score:.3f}: {rc.chunk.text[:80].strip()}\u2026"
        )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": "0.1.0",
            "week": 1,
            "project": "mini-rag-reader",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_file": str(in_path),
            "input_kind": "text" if args.text else "pdf",
            "question": args.question,
            "generator": generator_kind,
            "embedding_model": args.embedding_model,
            "metrics": {
                "pages_loaded": len(pages),
                "chunks_total": len(chunks),
                "embedding_dim": embedder.dim,
                "top_k": args.top_k,
                "retrieved_chunk_ids": [rc.chunk.chunk_id for rc in retrieved],
                "retrieval_scores": [round(rc.score, 4) for rc in retrieved],
                "answer_char_count": len(text),
                "answer_has_citations": any(
                    f"[{rc.chunk.chunk_id}]" in text for rc in retrieved
                ),
            },
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
