"""Pydantic config loader for Mini-RAG Reader.

Loads `configs/baseline.yaml` into a typed config object. All pipeline
components (Reader, CLI, serve) accept a `Config` instance.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _detect_device() -> str:
    """Auto-detect the best available accelerator.

    Preference order: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.
    Safe to call without torch installed: returns "cpu".
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _default_torch_dtype(device: str | None = None) -> str:
    """Pick a torch dtype that is valid for the detected device.

    bfloat16 is only supported on CUDA; CPU and MPS fall back to float32
    so text-generation pipelines don't crash on accelerator-less machines.
    """
    if device is None:
        device = _detect_device()
    return "bfloat16" if device == "cuda" else "float32"


class EmbeddingConfig(BaseModel):
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dim: int = Field(default=384)
    device: str = Field(default_factory=_detect_device)


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=4)
    similarity: str = Field(default="ip")


class LLMConfig(BaseModel):
    model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens: int = Field(default=512)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    torch_dtype: str = Field(default_factory=_default_torch_dtype)


class ServeConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")


class Config(BaseModel):
    """Top-level config loaded from baseline.yaml."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    data_dir: str = Field(default="data")
    sample_pdf: str = Field(default="data/sample.pdf")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a YAML config file (e.g. configs/baseline.yaml)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return _from_flat_yaml(raw)


def _from_flat_yaml(data: dict) -> Config:
    """Map the flat baseline.yaml keys into the nested Config model.

    Device- and dtype-sensitive keys auto-detect the local accelerator
    (CUDA/MPS/CPU) when absent from the YAML, so a fresh clone uses the GPU
    without hand-editing baseline.yaml.
    """
    device = _detect_device()
    return Config(
        embedding=EmbeddingConfig(
            model=data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            dim=data.get("embedding_dim", 384),
            device=data.get("embedding_device") or device,
        ),
        chunking=ChunkingConfig(
            chunk_size=data.get("chunk_size", 500),
            chunk_overlap=data.get("chunk_overlap", 50),
        ),
        retrieval=RetrievalConfig(
            top_k=data.get("top_k", 4),
            similarity=data.get("similarity", "ip"),
        ),
        llm=LLMConfig(
            model=data.get("llm_model", "Qwen/Qwen2.5-1.5B-Instruct"),
            max_new_tokens=data.get("llm_max_new_tokens", 512),
            temperature=data.get("llm_temperature", 0.1),
            top_p=data.get("llm_top_p", 0.95),
            device_map=data.get("device_map", "auto"),
            torch_dtype=data.get("torch_dtype") or _default_torch_dtype(device),
        ),
        serve=ServeConfig(
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 8000),
            log_level=data.get("log_level", "info"),
        ),
        data_dir=data.get("data_dir", "data"),
        sample_pdf=data.get("sample_pdf", "data/sample.pdf"),
    )


def load_default_config() -> Config:
    """Load the bundled baseline.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    return Config.from_yaml(config_path)
