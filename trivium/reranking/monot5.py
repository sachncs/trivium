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
        self.slug_value = slug
        self.model_id_value = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.fp16 = fp16
        self.model = None
        self.tokenizer = None
        self.true_token_id: int | None = None
        self.false_token_id: int | None = None

    @property
    def slug(self) -> str:
        return self.slug_value

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        top_k: int,
    ) -> SearchResult:
        if not candidates:
            return SearchResult.empty()
        self.ensure_model()
        prompts = [f"Query: {query} Document: {c.body} Relevant:" for c in candidates]
        scores = self.score_batch(prompts)
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

    def score_batch(self, prompts: list[str]) -> np.ndarray:
        import torch

        all_scores = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                decoder_input_ids = torch.full(
                    (enc.input_ids.size(0), 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = self.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits[:, 0, :]
                true_logit = logits[:, self.true_token_id]
                false_logit = logits[:, self.false_token_id]
                probs = torch.softmax(torch.stack([false_logit, true_logit], dim=1), dim=1)[:, 1]
                all_scores.append(probs.cpu().float().numpy())
        return np.concatenate(all_scores, axis=0)

    def ensure_model(self) -> None:
        """Lazy-load the T5 model and tokenize the true/false decision tokens."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_value)
        dtype = torch.float16 if self.fp16 and self.device != "cpu" else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_id_value,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.true_token_id = self.tokenizer("true", add_special_tokens=False).input_ids[0]
        self.false_token_id = self.tokenizer("false", add_special_tokens=False).input_ids[0]
