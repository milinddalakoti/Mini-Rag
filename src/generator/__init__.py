"""src.generator — answer generators + prompt builder.

Public API:
    from src.generator import Generator, ExtractiveGenerator, build_prompt, SYSTEM_PROMPT

- `Generator`            : Hugging Face transformers pipeline (needs model download).
- `ExtractiveGenerator` : deterministic, no-LLM, fast — great for Week-1 demo.
"""

from src.generator.extractive import ExtractiveGenerator
from src.generator.generator import SYSTEM_PROMPT, build_prompt

__all__ = [
    "ExtractiveGenerator",
    "build_prompt",
    "SYSTEM_PROMPT",
]

def __getattr__(name: str):
    if name == "Generator":
        from src.generator.generator import Generator
        return Generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")