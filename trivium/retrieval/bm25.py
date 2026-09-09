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
        self.bm25 = None
        self.tokenizer = None
        self.doc_ids: list[str] = []
        self.id_to_pos: dict[str, int] = {}

    @property
    def slug(self) -> str:
        return "bm25"

    def add_documents(
        self, documents: Sequence[Document], vectors: np.ndarray | None = None
    ) -> None:
        if not documents:
            raise RuntimeError("Bm25.add_documents requires at least one document")
        self.doc_ids = [d.doc_id for d in documents]
        self.id_to_pos = {did: i for i, did in enumerate(self.doc_ids)}

        from bm25s import BM25
        from bm25s.tokenization import Tokenizer

        self.tokenizer = Tokenizer(stopwords=self.stopwords, stemmer=self.stemmer)
        corpus_tokens = self.tokenizer.tokenize(
            [d.body for d in documents],
            return_as="tuple",
        )
        self.bm25 = BM25(method=self.method, k1=self.k1, b=self.b)
        self.bm25.index(corpus_tokens, show_progress=False)

    def search(self, query_vectors: np.ndarray, k: int) -> list[SearchResult]:
        """Encode the raw query text from `query_vectors` (we treat it as strings).

        bm25s has no notion of pre-computed query vectors; the input
        is treated as raw query strings. Shape: (N,) dtype U.
        """
        self.ensure_built()
        queries = coerce_query_strings(query_vectors)
        query_tokens = self.tokenizer.tokenize(
            queries,
            update_vocab=False,
            return_as="tuple",
            show_progress=False,
        )
        indices, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)
        indices = np.asarray(indices)
        scores = np.asarray(scores)
        results: list[SearchResult] = []
        for row_idx in range(indices.shape[0]):
            ids = indices[row_idx]
            sc = scores[row_idx]
            results.append(
                SearchResult.from_pairs(
                    [
                        (self.doc_ids[int(ids[i])], float(sc[i]))
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
        if self.bm25 is None:
            return 0
        bm = self.bm25
        scores_size = 0
        from contextlib import suppress

        with suppress(Exception):
            scores_size = int(np.asarray(bm.scores["idf"]).nbytes)
        return scores_size

    def save(self, dir_path: str | Path) -> None:
        """Persist the BM25 index, tokenizer vocab, stopwords, and document IDs.

        bm25s' on-disk format has no place for the original document IDs,
        so we round-trip them alongside the index. A load() with no
        matching doc_ids.npy file will leave self.doc_ids empty (matching
        the historical behaviour) and emit a warning.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(dir_path), corpus=None)
        self.tokenizer.save_vocab(save_dir=str(dir_path))
        self.tokenizer.save_stopwords(save_dir=str(dir_path))
        np.save(dir_path / "doc_ids.npy", np.asarray(self.doc_ids, dtype=object))

    def load(self, dir_path: str | Path, mmap: bool = True) -> None:
        """Reload a saved BM25 index; doc_ids are restored if doc_ids.npy is present."""
        from bm25s import BM25
        from bm25s.tokenization import Tokenizer

        dir_path = Path(dir_path)
        self.bm25 = BM25.load(str(dir_path), mmap=mmap)
        self.tokenizer = Tokenizer(stopwords=self.stopwords, stemmer=self.stemmer)
        self.tokenizer.load_vocab(save_dir=str(dir_path))
        self.tokenizer.load_stopwords(save_dir=str(dir_path))

        ids_path = dir_path / "doc_ids.npy"
        if ids_path.is_file():
            loaded = np.load(ids_path, allow_pickle=True).tolist()
            self.doc_ids = [str(d) for d in loaded]
            self.id_to_pos = {did: i for i, did in enumerate(self.doc_ids)}
        else:
            # Historical fallback: no doc_ids on disk. Caller can use
            # ensure_doc_positions() if it has them elsewhere.
            self.doc_ids = []
            self.id_to_pos = {}

    def ensure_doc_positions(self) -> None:
        """Hook for callers that load a saved index and need to repopulate doc_ids."""
        # The legacy mmap save doesn't persist doc_ids. The caller
        # must provide them externally if they're needed.
        pass

    def ensure_built(self) -> None:
        if self.bm25 is None or self.tokenizer is None:
            raise RuntimeError(
                "Bm25.search called before add_documents(); call add_documents or load first"
            )


def coerce_query_strings(input_) -> list[str]:
    """Coerce bm25s query input into a list[str].

    Accepts: a single str, list[str], numpy array of strings (any shape).
    """
    if isinstance(input_, str):
        return [input_]
    if isinstance(input_, np.ndarray):
        return input_.astype("U").tolist()
    return list(input_)
