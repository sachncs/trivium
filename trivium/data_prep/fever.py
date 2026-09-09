"""FEVER sampler for padding past the BEIR scientific corpus ceiling."""
from __future__ import annotations

import random

from trivium.data_prep.distractors import beir_corpus_iter
from trivium.domain.document import Document


def sample_fever(n_target: int, seed: int) -> list[Document]:
    """Sample FEVER at a fixed seed to pad the distractor pool.

    Args:
        n_target: Number of docs to keep (caps at full FEVER size).
        seed: Random seed for deterministic shuffle-then-slice.
    """
    rng = random.Random(seed)
    docs = [
        Document(doc_id=f"distractor:{did}", title=title, text=text)
        for did, title, text in beir_corpus_iter("BeIR/fever")
    ]
    rng.shuffle(docs)
    out = docs[:n_target]
    print(f"  fever: {len(out):,} sampled", flush=True)
    return out
