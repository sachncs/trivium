"""Cross-encoder reranker. ms-marco-MiniLM-L-6-v2 by default. Batch scoring on CPU."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _doc_text(c: dict) -> str:
    """Concatenate title + text the same way the BM25 index does."""
    title = (c.get("title") or "").strip()
    text = (c.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 256, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[dict], top_k: int = 10) -> list[tuple[dict, float]]:
        """candidates: list of dicts with at least 'id', 'title', 'text'. Returns top_k (cand, score)."""
        if not candidates:
            return []
        pairs = [(query, _doc_text(c)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [(candidates[i], float(scores[i])) for i in order]
