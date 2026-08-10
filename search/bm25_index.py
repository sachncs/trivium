"""bm25s wrapper. mmap-mode for scale-friendly RAM behavior."""
from __future__ import annotations

from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer


class BM25Index:
    def __init__(self, k1: float = 0.9, b: float = 0.4, method: str = "lucene"):
        self.k1 = k1
        self.b = b
        self.method = method
        self.bm25 = None
        self.tokenizer = None

    def build(self, texts: list[str], show_progress: bool = True) -> None:
        self.tokenizer = Tokenizer(stopwords="en", stemmer=None)
        corpus_tokens = self.tokenizer.tokenize(texts, return_as="tuple")
        self.bm25 = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self.bm25.index(corpus_tokens, show_progress=show_progress)

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(dir_path), corpus=None)
        self.tokenizer.save_vocab(save_dir=str(dir_path))
        self.tokenizer.save_stopwords(save_dir=str(dir_path))

    def load(self, dir_path: str | Path, mmap: bool = True) -> None:
        dir_path = Path(dir_path)
        self.bm25 = bm25s.BM25.load(str(dir_path), mmap=mmap)
        self.tokenizer = Tokenizer(stopwords="en", stemmer=None)
        self.tokenizer.load_vocab(save_dir=str(dir_path))
        self.tokenizer.load_stopwords(save_dir=str(dir_path))

    def query(self, query_text: str, k: int = 100) -> tuple[list[int], list[float]]:
        q_tokens = self.tokenizer.tokenize([query_text], update_vocab=False, return_as="tuple", show_progress=False)
        indices, scores = self.bm25.retrieve(q_tokens, k=k, show_progress=False)
        return indices[0].tolist(), scores[0].tolist()
