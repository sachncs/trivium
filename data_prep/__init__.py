"""Data preparation: corpus builders + embedding per encoder.

Each module focuses on one slice of the pipeline:
- seed.py: scifact seed loader (BEIR canonical URL)
- distractors.py: BEIR scientific pool
- fever.py: FEVER sample for padding past the BEIR ceiling
- dedup.py: SHA + MinHash LSH deduplication
- corpus.py: prefix-extension corpus builder (1M = superset of 500K = ...)
- embed.py: per-encoder embedding using EmbeddingCache
"""
from data_prep.corpus import build_prefixed_corpus
from data_prep.dedup import deduplicate
from data_prep.distractors import load_distractor_pool
from data_prep.embed import embed_corpus_per_encoder
from data_prep.fever import sample_fever
from data_prep.seed import load_scifact_seed

__all__ = [
    "build_prefixed_corpus",
    "deduplicate",
    "embed_corpus_per_encoder",
    "load_distractor_pool",
    "load_scifact_seed",
    "sample_fever",
]
