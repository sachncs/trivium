"""BM25 retriever (bm25s backend)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.retrieval.base import Retriever


class Bm25(Retriever):
    """Sparse lexical retriever backed by bm25s.

    Args:
        k1, b: BM25 hyperparameters (Anserini defaults).
        method: 'lucene' (default) / 'atire' / 'bm25l'.
        stopwords, stemmer: passed to bm25s.tokenization.Tokenizer.
    """

    def __init__(
        self,
        k1: float = 0.9,
        b: float = 0.4,
        method: str = "lucene",
        stopwords: str = "en",
        stemmer: str | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.method = method
        self.stopwords = stopwords
        self.stemmer = stemmer
        self._bm25 = None
        self._tokenizer = None
        self._doc_ids: list[str] = []
        self._id_to_pos: dict[str, int] = {}

    @property
    def slug(self) -> str:
        return "bm25"

    def add_documents(self, documents: Sequence[Document], vectors: np.ndarray | None = None) -> None:
        if not documents:
            raise RuntimeError("Bm25.add_documents requires at least one document")
        self._doc_ids = [d.doc_id for d in documents]
        self._id_to_pos = {did: i for i, did in enumerate(self._doc_ids)}

        from bm25s import BM25
        from bm25s.tokenization import Tokenizer

        self._tokenizer = Tokenizer(stopwords=self.stopwords, stemmer=self.stemmer)
        corpus_tokens = self._tokenizer.tokenize(
            [d.body for d in documents],
            return_as="tuple",
        )
        self._bm25 = BM25(method=self.method, k1=self.k1, b=self.b)
        self._bm25.index(corpus_tokens, show_progress=False)

    def search(self, query_vectors: np.ndarray, k: int) -> list[SearchResult]:
        """Encode the raw query text from `query_vectors` (we treat it as strings).

        bm25s has no notion of pre-computed query vectors; the input
        is treated as raw query strings. Shape: (N,) dtype U.
        """
        self._ensure_built()
        queries = self._coerce_query_strings(query_vectors)
        query_tokens = self._tokenizer.tokenize(
            queries,
            update_vocab=False,
            return_as="tuple",
            show_progress=False,
        )
        indices, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)
        indices = np.asarray(indices)
        scores = np.asarray(scores)
        results: list[SearchResult] = []
        for row_idx in range(indices.shape[0]):
            ids = indices[row_idx]
            sc = scores[row_idx]
            results.append(
                SearchResult.from_pairs(
                    [
                        (self._doc_ids[int(ids[i])], float(sc[i]))
                        for i in range(ids.shape[0])
                        if int(ids[i]) != -1
                    ]
                )
            )
        return results

    def set_search_params(self, **params) -> None:
        """BM25 has no search-time params. We accept and ignore."""
        return

    def size_bytes(self) -> int:
        if self._bm25 is None:
            return 0
        bm = self._bm25
        vocab = getattr(bm, "vocab", None)
        scores_size = 0
        try:
            scores_size = int(np.asarray(bm.scores["idf"]).nbytes)
        except Exception:
            pass
        return scores_size

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(dir_path), corpus=None)
        self._tokenizer.save_vocab(save_dir=str(dir_path))
        self._tokenizer.save_stopwords(save_dir=str(dir_path))

    def load(self, dir_path: str | Path, mmap: bool = True) -> None:
        from bm25s import BM25
        from bm25s.tokenization import Tokenizer

        dir_path = Path(dir_path)
        self._bm25 = BM25.load(str(dir_path), mmap=mmap)
        self._tokenizer = Tokenizer(stopwords=self.stopwords, stemmer=self.stemmer)
        self._tokenizer.load_vocab(save_dir=str(dir_path))
        self._tokenizer.load_stopwords(save_dir=str(dir_path))

    def ensure_doc_positions(self) -> None:
        """Call this after load() to populate doc_ids metadata."""
        # The legacy mmap save doesn't persist doc_ids. The caller
        # must provide them externally if they're needed.
        pass

    def _ensure_built(self) -> None:
        if self._bm25 is None or self._tokenizer is None:
            raise RuntimeError("Bm25.search called before add_documents(); call add_documents or load first")

    @staticmethod
    def _coerce_query_strings(input_) -> list[str]:
        # Accept list[str], numpy array of strings, or a single string.
        if isinstance(input_, str):
            return [input_]
        if isinstance(input_, np.ndarray):
            return input_.astype("U").tolist()
        return list(input_)
