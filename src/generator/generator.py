"""Hugging Face generator + prompt builder.

Wraps the HF pipeline so the rest of the code never imports transformers
directly. Defaults to a small open-weights model that runs on CPU.
"""

from __future__ import annotations

from src.retriever import RetrievedChunk


SYSTEM_PROMPT = (
    "You are a precise retrieval-augmented assistant. Answer the user's question "
    "ONLY using the provided context. If the answer is not in the context, say "
    "'I don't know based on the provided documents.' Cite chunk IDs in square "
    "brackets, e.g. [c00012]."
)


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Build a chat prompt with system + context + question."""
    context_parts = []
    for rc in retrieved:
        ctx = rc.chunk
        context_parts.append(
            f"[{ctx.chunk_id}] (page {ctx.page_number})\n{ctx.text}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer with citations to chunk IDs in square brackets. "
        "If the context doesn't contain the answer, say so explicitly."
    )
    return user_prompt


class Generator:
    """Thin wrapper around HF text-generation pipeline."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except ImportError as e:
            raise ImportError(
                "transformers + torch are required. `pip install transformers torch`."
            ) from e

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[torch_dtype]

        self.pipe = pipeline(
            task="text-generation",
            model=model_name,
            device_map=device_map,
            dtype=dtype,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        user_prompt = build_prompt(question, retrieved)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        out = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()