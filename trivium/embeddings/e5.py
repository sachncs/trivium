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
        self._slug = slug
        self._model_id = model_id
        self._dimension = dimension
        self._prompt_query = prompt_prefix_query
        self._prompt_doc = prompt_prefix_doc
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._max_memory_gb = max_memory_gb
        self._device = device
        self._model = None
        self._tokenizer = None

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    def warmup(self, sample_texts: Sequence[str] = ("warmup",)) -> None:
        if self._model is None:
            self._ensure_model()
        with self._inference_mode():
            inputs = self._tokenize([self._prompt_query + sample_texts[0]])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self._prompt_doc + t for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self._prompt_query + t for t in texts])

    def _encode(self, prompted: list[str]) -> np.ndarray:
        self._ensure_model()
        import torch

        all_vectors = []
        with self._inference_mode():
            for start in range(0, len(prompted), self._batch_size):
                batch = prompted[start : start + self._batch_size]
                inputs = self._tokenize(batch)
                outputs = self._model(**inputs, output_hidden_states=False)
                last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                # last-token pooling: index the final non-pad position per row
                attention_mask = inputs["attention_mask"]
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled = last_hidden[torch.arange(last_hidden.size(0)), sequence_lengths]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vectors.append(pooled.cpu().float().numpy())
        return np.concatenate(all_vectors, axis=0).astype(np.float32)

    def _tokenize(self, texts: list[str]) -> dict:
        tok = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in tok.items()}

    def _inference_mode(self):
        import torch

        return torch.inference_mode()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._check_memory()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._model = AutoModel.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            device_map=self._device,
        )
        self._model.eval()

    def _check_memory(self) -> None:
        if self._max_memory_gb is not None:
            self._validate_within(self._max_memory_gb)
            return
        try:
            free_gb = _linux_available_memory_gb() if os.name == "posix" else 16.0
            peak_estimate = 14.0
            if peak_estimate > 0.8 * free_gb:
                raise RuntimeError(
                    f"E5-Mistral-7B at ~14 GB exceeds 80% of available "
                    f"{free_gb:.1f} GB RAM. Pass max_memory_gb to override."
                )
        except FileNotFoundError:
            pass

    @staticmethod
    def _validate_within(limit_gb: float) -> None:
        free_gb = _linux_available_memory_gb() if os.name == "posix" else 16.0
        if limit_gb > free_gb:
            raise RuntimeError(f"max_memory_gb={limit_gb} > available {free_gb:.1f}")


def _linux_available_memory_gb() -> float:
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
