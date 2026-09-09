"""E5-Mistral-7B embedder with instruction template.

This is the special embedder that requires:
- Heavy 14 GB model load (mps or cuda strongly preferred)
- instruction template 'query: {q}\\npassage: {p}'
- last-token pooling (T5 / Mistral convention)
- L2 normalisation

Heavy imports (torch, transformers) deferred to first use.

Memory gate: refuses to load if estimated peak exceeds 80% of
available RAM (or the explicit max_memory_gb override).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from trivium.embeddings.base import Embedder


class E5(Embedder):
    """E5-Mistral-7B-instruct embedding model.

    Args:
        slug: Identifier (typically 'e5-mistral-7b').
        model_id: HuggingFace model id (default 'intfloat/e5-mistral-7b-instruct').
        dimension: 4096.
        prompt_prefix_query: Defaults to 'query: '.
        prompt_prefix_doc: Defaults to 'passage: '.
        max_memory_gb: Override the 80% free-RAM safety cap.
    """

    DEFAULT_MODEL = "intfloat/e5-mistral-7b-instruct"
    DEFAULT_DIM = 4096
    DEFAULT_QUERY_PREFIX = "query: "
    DEFAULT_DOC_PREFIX = "passage: "

    def __init__(
        self,
        slug: str = "e5-mistral-7b",
        model_id: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIM,
        prompt_prefix_query: str = DEFAULT_QUERY_PREFIX,
        prompt_prefix_doc: str = DEFAULT_DOC_PREFIX,
        batch_size: int = 8,
        max_seq_length: int = 256,
        max_memory_gb: float | None = None,
        device: str = "mps",
    ) -> None:
        self.slug_value = slug
        self.model_id_value = model_id
        self.dimension = dimension
        self.prompt_query = prompt_prefix_query
        self.prompt_doc = prompt_prefix_doc
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_memory_gb = max_memory_gb
        self.device = device
        self.model = None
        self.tokenizer = None

    @property
    def slug(self) -> str:
        return self.slug_value

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def warmup(self, sample_texts: Sequence[str] = ("warmup",)) -> None:
        if self.model is None:
            self.ensure_model()
        with self.inference_mode():
            self.tokenize_pair([self.prompt_query + sample_texts[0]])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_prompted([self.prompt_doc + t for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_prompted([self.prompt_query + t for t in texts])

    def encode_prompted(self, prompted: list[str]) -> np.ndarray:
        self.ensure_model()
        import torch

        all_vectors = []
        with self.inference_mode():
            for start in range(0, len(prompted), self.batch_size):
                batch = prompted[start : start + self.batch_size]
                inputs = self.tokenize_pair(batch)
                outputs = self.model(**inputs, output_hidden_states=False)
                last_hidden = (
                    outputs.last_hidden_state
                    if hasattr(outputs, "last_hidden_state")
                    else outputs[0]
                )
                attention_mask = inputs["attention_mask"]
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled = last_hidden[torch.arange(last_hidden.size(0)), sequence_lengths]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vectors.append(pooled.cpu().float().numpy())
        return np.concatenate(all_vectors, axis=0).astype(np.float32)

    def tokenize_pair(self, texts: list[str]) -> dict:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tok.items()}

    def inference_mode(self):
        """Return a torch.inference_mode() context manager."""
        import torch

        return torch.inference_mode()

    def ensure_model(self) -> None:
        """Lazy-load the model; gates on free memory."""
        if self.model is not None:
            return
        self.check_memory()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_value)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_id_value,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    def check_memory(self) -> None:
        if self.max_memory_gb is not None:
            self.validate_within(self.max_memory_gb)
            return
        try:
            free_gb = linux_available_memory_gb() if os.name == "posix" else 16.0
            peak_estimate = 14.0
            if peak_estimate > 0.8 * free_gb:
                raise RuntimeError(
                    f"E5-Mistral-7B at ~14 GB exceeds 80% of available "
                    f"{free_gb:.1f} GB RAM. Pass max_memory_gb to override."
                )
        except FileNotFoundError:
            pass

    @staticmethod
    def validate_within(limit_gb: float) -> None:
        free_gb = linux_available_memory_gb() if os.name == "posix" else 16.0
        if limit_gb > free_gb:
            raise RuntimeError(f"max_memory_gb={limit_gb} > available {free_gb:.1f}")


def linux_available_memory_gb() -> float:
    """Best-effort free memory in GB on Linux. Returns inf on failure."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError):
        pass
    return float("inf")
