"""BEIR scientific + encyclopedic distractors (no web/MS-MARCO; domain-mismatch)."""
from __future__ import annotations

from collections.abc import Iterator

from trivium.domain.document import Document


def beir_corpus_iter(repo_id: str) -> Iterator[tuple[str, str, str]]:
    """Yield (id, title, text) from a BeIR/<dataset> corpus via `datasets`."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, "corpus", split="corpus")
    for row in ds:
        yield row["_id"], row.get("title", "") or "", row.get("text", "") or ""


def load_distractor_pool(sources: list[str]) -> list[Document]:
    """Scientific + encyclopedic distractors, no web (MS MARCO).

    Args:
        sources: BEIR source slugs (e.g. ['scidocs', 'trec-covid', 'nfcorpus']).
    """
    pool: list[Document] = []
    for src in sources:
        n_before = len(pool)
        for did, title, text in beir_corpus_iter(f"BeIR/{src}"):
            pool.append(Document(doc_id=f"distractor:{did}", title=title, text=text))
        print(f"  {src}: {len(pool) - n_before:,} docs", flush=True)
    return pool
