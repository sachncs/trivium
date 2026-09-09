"""SHA + MinHash LSH deduplication against the seed set.

Fixes two latent bugs from the legacy data/prepare.py:dedup:

1. The seed LSH index used to be built lazily through the same
   loop that iterated over the full corpus, mixing decisions;
   we now build it on a separate pass first.
2. seed_lsh.query(m) used to be treated as a boolean; we now
   use len(query) > 0 for cleaner semantics.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from trivium.domain.document import Document


def _text_hash(doc: Document) -> str:
    return hashlib.sha256(f"{doc.title}\n{doc.text}".encode("utf-8")).hexdigest()


def _minhash_lsh(documents: Iterable[Document], jaccard_threshold: float):
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=64)
    seed_signatures: list[tuple[str, MinHash]] = []
    for d in documents:
        m = MinHash(num_perm=64)
        for tok in set(f"{d.title}\n{d.doc_text}".lower().split()) if hasattr(d, "doc_text") else set(d.body.lower().split()):
            m.update(tok.encode("utf-8"))
        lsh.insert(d.doc_id, m)
        seed_signatures.append((d.doc_id, m))
    return lsh, seed_signatures


def deduplicate(corpus: list[Document], jaccard_threshold: float = 0.85) -> list[Document]:
    """Exact-text + MinHash near-duplicate dedup.

    Args:
        corpus: Full corpus (seed + distractors). Seed docs are
            preserved regardless of text hash collisions.
        jaccard_threshold: MinHash LSH threshold; 0.85 is the
            Anserini/BEIR default.
    """
    seed_docs = [d for d in corpus if d.doc_id.startswith("scifact:")]

    lsh, _ = _minhash_lsh(seed_docs, jaccard_threshold)

    seed_text_hashes = {_text_hash(d) for d in seed_docs}
    seen: set[str] = set()
    out: list[Document] = []
    for d in corpus:
        h = _text_hash(d)
        if h in seen:
            continue
        seen.add(h)
        if d.doc_id.startswith("scifact:"):
            out.append(d)
            continue
        if h in seed_text_hashes:
            continue
        m = _build_minhash(d, num_perm=64)
        if len(lsh.query(m)) > 0:
            continue
        out.append(d)
    return out


def _build_minhash(d: Document, num_perm: int):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    for tok in set(d.body.lower().split()):
        m.update(tok.encode("utf-8"))
    return m
