"""Tests for the YAML config loader."""

from pathlib import Path

import yaml

from src.config import (
    Config,
    EmbeddingConfig,
    LLMConfig,
    _default_torch_dtype,
    _detect_device,
    load_default_config,
)


def test_config_yaml_is_loadable():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    assert cfg_path.exists(), f"missing config: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    # Required keys
    for k in (
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "llm_model",
        "port",
    ):
        assert k in cfg, f"missing key: {k}"


def test_config_chunk_overlap_lt_size():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["chunk_overlap"] < cfg["chunk_size"]


def test_config_top_k_positive():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["top_k"] >= 1


def test_detect_device_returns_valid_accelerator():
    assert _detect_device() in {"cuda", "mps", "cpu"}


def test_default_torch_dtype_is_device_aware():
    # bfloat16 is only safe on CUDA.
    assert _default_torch_dtype("cuda") == "bfloat16"
    # CPU / MPS must never claim bfloat16.
    assert _default_torch_dtype("cpu") == "float32"
    assert _default_torch_dtype("mps") == "float32"
    # No-arg form adapts to the local machine.
    assert _default_torch_dtype() == _default_torch_dtype(_detect_device())


def test_llm_config_default_dtype_matches_local_device():
    assert LLMConfig().torch_dtype == _default_torch_dtype()


def test_embedding_config_default_device_matches_detection():
    assert EmbeddingConfig().device == _detect_device()


def test_default_llm_model_is_not_gated():
    # Llama-3.2 is gated behind a license request; the shipped default must
    # be a publicly pullable model so a fresh clone works without HF login.
    assert not LLMConfig().model.startswith("meta-llama/")
    assert load_default_config().llm.model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_default_config_auto_detects_gpu_and_dtype():
    cfg = load_default_config()
    # baseline.yaml no longer hard-codes "cpu" / "bfloat16": a GPU machine
    # actually gets cuda + bfloat16, and CPU machines get float32.
    assert cfg.embedding.device == _detect_device()
    assert cfg.llm.torch_dtype == _default_torch_dtype(_detect_device())


def test_yaml_explicit_device_overrides_autodetect(tmp_path):
    cfg_file = tmp_path / "explicit.yaml"
    cfg_file.write_text(
        "embedding_model: sentence-transformers/all-MiniLM-L6-v2\n"
        "embedding_dim: 384\n"
        "embedding_device: cpu\n"
        "llm_model: Qwen/Qwen2.5-1.5B-Instruct\n"
        "device_map: auto\n"
        "torch_dtype: float32\n"
    )
    cfg = Config.from_yaml(cfg_file)
    # Explicit values are honoured, not clobbered by auto-detection.
    assert cfg.embedding.device == "cpu"
    assert cfg.llm.torch_dtype == "float32"
    assert cfg.llm.device_map == "auto"


def test_yaml_without_device_autodetects(tmp_path):
    cfg_file = tmp_path / "no_device.yaml"
    cfg_file.write_text(
        "llm_model: Qwen/Qwen2.5-1.5B-Instruct\n"
        "chunk_size: 500\n"
        "chunk_overlap: 50\n"
        "top_k: 4\n"
        "port: 8000\n"
    )
    cfg = Config.from_yaml(cfg_file)
    assert cfg.embedding.device == _detect_device()
    assert cfg.llm.torch_dtype == _default_torch_dtype(_detect_device())
