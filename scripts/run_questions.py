"""Run a small benchmark of canonical questions to populate metrics.json with
multiple Q&A rows + sanity-check answers.

Usage:
    python -m scripts.run_questions --report reports/metrics.json
    python -m scripts.run_questions --config configs/baseline.yaml

The canonical questions are tuned to the sample CSEP guide text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as `python -m scripts.run_questions` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_default_config  # noqa: E402
from src.data import chunk_text, load_text  # noqa: E402
from src.features import Embedder, build_faiss_index, embed_chunks  # noqa: E402
from src.generator import ExtractiveGenerator  # noqa: E402
from src.retriever import Retriever  # noqa: E402


CANONICAL_QUESTIONS = [
    {
        "question": "What is the CSEP salary threshold from 1 March 2026?",
        "expect_keywords": ["40,904", "March 2026"],
    },
    {
        "question": "What is the General Employment Permit threshold?",
        "expect_keywords": ["36,605"],
    },
    {
        "question": "How long is the Stamp 1G post-study work permit?",
        "expect_keywords": ["24 months", "12 months"],
    },
    {
        "question": "Which Dublin companies hire AI engineers in Ireland?",
        "expect_keywords": ["Stripe", "Workday", "Intercom"],
    },
]


def _resolve_config(config_path: str | None) -> Config:
    if config_path:
        return Config.from_yaml(config_path)
    return load_default_config()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--pdf", default="data/sample.txt")
    parser.add_argument("--text", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--report", default="reports/metrics.json")
    args = parser.parse_args()

    config = _resolve_config(args.config)
    top_k = args.top_k if args.top_k is not None else config.retrieval.top_k

    in_path = Path(args.pdf)
    if not in_path.exists():
        print(f"ERROR: {in_path}", file=sys.stderr)
        return 2

    pages = load_text(in_path) if args.text else None
    chunks = chunk_text(
        pages,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        source_path=str(in_path),
    )
    print(f"Loaded {len(pages)} page(s), {len(chunks)} chunk(s).")

    embedder = Embedder(
        model_name=config.embedding.model,
        device=config.embedding.device,
    )
    vectors = embed_chunks(embedder, chunks)
    index = build_faiss_index(vectors)
    retriever = Retriever(index=index, chunks=chunks)
    generator = ExtractiveGenerator()

    rows: list[dict] = []
    passed = 0
    for q in CANONICAL_QUESTIONS:
        t0 = time.perf_counter()
        q_vec = embedder.embed([q["question"]])
        retrieved = retriever.retrieve(q_vec, top_k=top_k)
        text = generator.generate(q["question"], retrieved)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hit = all(k.lower() in text.lower() for k in q["expect_keywords"])
        passed += int(hit)
        mark = "+" if hit else "x"
        print(f"  {mark} ({elapsed_ms:.0f} ms) Q: {q['question']}")
        print(f"      Keywords expected: {q['expect_keywords']}")
        print(f"      Top retrieved: {[rc.chunk.chunk_id for rc in retrieved]}")
        print(f"      Answer[:160]: {text[:160].strip()}.")
        rows.append(
            {
                "question": q["question"],
                "expect_keywords": q["expect_keywords"],
                "passed": hit,
                "top_k": top_k,
                "retrieved_chunk_ids": [rc.chunk.chunk_id for rc in retrieved],
                "retrieval_scores": [round(rc.score, 4) for rc in retrieved],
                "answer_char_count": len(text),
                "elapsed_ms": round(elapsed_ms, 2),
                "answer": text,
            }
        )

    summary = {
        "total_questions": len(CANONICAL_QUESTIONS),
        "passed": passed,
        "pass_rate": round(passed / len(CANONICAL_QUESTIONS), 3),
    }

    report = {
        "schema_version": "0.1.0",
        "week": 1,
        "project": "mini-rag-reader",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_file": str(in_path),
        "input_kind": "text" if args.text else "pdf",
        "embedding_model": config.embedding.model,
        "embedding_dim": embedder.dim,
        "chunks_total": len(chunks),
        "generator": "Extractive",
        "summary": summary,
        "results": rows,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")
    print(f"Pass rate: {passed}/{len(CANONICAL_QUESTIONS)} ({summary['pass_rate']*100:.0f}%)")
    return 0 if passed == len(CANONICAL_QUESTIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
