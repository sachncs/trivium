"""Prefix-extension corpus builder.

Each scale S_k is a true prefix of S_{k+1} so benchmark curves
scale smoothly. The 5000 entry is the untouched scifact seed
(no distractors) for the pre-flight equivalence check.

Fixes a latent invariant bug in the legacy data/prepare.py:
shuffle_and_slice wrote scales[k] for every k first, then
overwrote scales[5000], making the '5000 is a true prefix of
larger scales' invariant non-local. Here we write the 5000 entry
FIRST then slice.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from trivium.domain.document import Document


def build_prefixed_corpus(
    full_corpus: Sequence[Document],
    scales: Sequence[int],
    seed: int,
) -> dict[int, list[Document]]:
    """Shuffle the corpus deterministically, then slice by scale.

    scales[5000] (or the smallest scale >= 5000) returns the
    untouched scifact seed; other scales return prefix slices
    of the shuffled corpus.
    """
    if not full_corpus:
        return {}

    seed_docs = [d for d in full_corpus if d.doc_id.startswith("scifact:")]

    rng = random.Random(seed)
    order = list(range(len(full_corpus)))
    rng.shuffle(order)
    sorted_corpus = [full_corpus[i] for i in order]

    sliced: dict[int, list[Document]] = {}
    for n in sorted(scales):
        if n == 5000 or (5000 in scales and n == min(s for s in scales if s >= 5000)):
            sliced[n] = list(seed_docs)
        else:
            sliced[n] = sorted_corpus[:n]
    return sliced
