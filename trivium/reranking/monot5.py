"""MonoT5 seq2seq reranker.

MonoT5 prompts: 'Query: {q} Document: {d} Relevant:'. The model
generates one token; we read the logit of 'true' vs 'false' and
softmax-normalise to a probability. The cross-attention is in the
encoder; the seq2seq decoder just produces one decision token.

Memory: T5-3B is 11 GB in fp16; gate via max_memory_gb.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.reranking.base import Reranker


class Monot5(Reranker):
    """MonoT5-{base,2b,3b} seq2seq reranker.

    Default: castorini/monot5-3b-msmarco-10k (T5-3B, 11 GB).
    """

    def __init__(
        self,
        slug: str = "monot5-3b",
        model_id: str = "castorini/monot5-3b-msmarco-10k",
        batch_size: int = 4,
        max_length: int = 512,
        device: str = "cpu",
        fp16: bool = True,
    ) -> None:
        self._slug = slug
        self._model_id = model_id
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._fp16 = fp16
        self._model = None
        self._tokenizer = None
        self._true_id: int | None = None
        self._false_id: int | None = None

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def model_id(self) -> str:
        return self._model_id

    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        top_k: int,
    ) -> SearchResult:
        if not candidates:
            return SearchResult.empty()
        self._ensure_model()
        prompts = [f"Query: {query} Document: {c.body} Relevant:" for c in candidates]
        scores = self._score_batch(prompts)
        order = np.argsort(-scores)[:top_k]
        return SearchResult(
            hits=[
                (c, float(s))
                for c, s in zip(
                    [candidates[i] for i in order],
                    [scores[i] for i in order],
                    strict=False,
                )
            ]
        )

    def _score_batch(self, prompts: list[str]) -> np.ndarray:
        import torch

        all_scores = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self._batch_size):
                batch = prompts[start : start + self._batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                decoder_input_ids = torch.full(
                    (enc.input_ids.size(0), 1),
                    self._model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self._device,
                )
                outputs = self._model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits[:, 0, :]
                true_logit = logits[:, self._true_id]
                false_logit = logits[:, self._false_id]
                probs = torch.softmax(torch.stack([false_logit, true_logit], dim=1), dim=1)[:, 1]
                all_scores.append(probs.cpu().float().numpy())
        return np.concatenate(all_scores, axis=0)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        dtype = torch.float16 if self._fp16 and self._device != "cpu" else torch.float32
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()
        self._true_id = self._tokenizer("true", add_special_tokens=False).input_ids[0]
        self._false_id = self._tokenizer("false", add_special_tokens=False).input_ids[0]
