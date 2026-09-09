"""Per-encoder embedding using EmbeddingCache.

Each encoder slug is content-hashed; re-running with a new
encoder does not destroy previous embeddings (the legacy code
overwrote vectors.npz).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.domain.document import Document
from trivium.io.embeddings import EmbeddingCache, encoder_revision


def embed_corpus_per_encoder(
    documents: Sequence[Document],
    embedder,
    cache: EmbeddingCache,
    cache_dir,
    batch_size: int = 128,
    max_seq_length: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Encode `documents` with `embedder`; reuse cached file if present.

    Returns float32 (N, dim). New encoders (different slug or
    different revision) get their own npz; old files are kept
    for cross-encoder comparison runs.
    """
    cfg = {
        "model_id": getattr(embedder, "model_id", embedder.slug),
        "max_seq_length": max_seq_length,
        "normalize": normalize,
        "prompt_prefix_doc": "",
        "prompt_prefix_query": "",
    }
    revision = encoder_revision(cfg)

    cached = cache.get(embedder.slug, revision)
    if cached is not None:
        vectors, ids = cached
        return vectors

    docs_to_encode = list(documents)
    seed_ids = {d.doc_id for d in docs_to_encode if d.doc_id.startswith("scifact:")}
    rest_ids = [d.doc_id for d in docs_to_encode if d.doc_id not in seed_ids]
    seed_docs = [d for d in docs_to_encode if d.doc_id in seed_ids]
    rest_docs = [d for d in docs_to_encode if d.doc_id in rest_ids]

    chunks = []
    chunk_ids = []

    if seed_docs:
        vec = embedder.encode_documents([d.body for d in seed_docs])
        chunks.append(vec)
        chunk_ids.extend([d.doc_id for d in seed_docs])
    if rest_docs:
        vec = embedder.encode_documents([d.body for d in rest_docs])
        chunks.append(vec)
        chunk_ids.extend([d.doc_id for d in rest_docs])

    vectors = np.concatenate(chunks, axis=0).astype(np.float32)
    ids_arr = np.array(chunk_ids)
    cache.put(embedder.slug, revision, vectors, ids_arr, doc_count=len(ids_arr))
    return vectors
