"""Mini-RAG Reader package.

Modular layout:
- src.data: PDF loading + chunking
- src.features: embedding + FAISS indexing
- src.retriever: top-k retrieval over FAISS index
- src.generator: Hugging Face LLM streaming
- src.reader: end-to-end orchestrator
- src.serve: FastAPI HTTP wrapper
- src.config: Pydantic config loader
"""

__version__ = "0.1.0"
