#!/usr/bin/env python3
"""Prepare the BEIR-augmented scifact corpus and embeddings.

CLI over the trivium.data_prep package. Replaces data/prepare.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from trivium.config.loader import load_config
from trivium.io.embeddings import EmbeddingCache
from trivium.io.manifest import Manifest, ManifestEncoders, ManifestScales

from data_prep.corpus import build_prefixed_corpus
from data_prep.dedup import deduplicate
from data_prep.distractors import load_distractor_pool
from data_prep.embed import embed_corpus_per_encoder
from data_prep.fever import sample_fever
from data_prep.seed import load_scifact_queries_and_qrels, load_scifact_seed

CACHE = Path(__file__).parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    p.add_argument("--max-scales", default=None, help="Cap scales, e.g. '5000,100000'")
    p.add_argument("--skip-embeddings", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_scales:
        cfg.scales = [int(x) for x in args.max_scales.split(",")]

    max_scale = max(cfg.scales)
    os.environ["PYTHONHASHSEED"] = str(cfg.runtime.python_hash_seed)

    print("=== Build corpus ===", flush=True)
    seed = load_scifact_seed()
    print(f"  scifact seed: {len(seed):,}", flush=True)

    pool = load_distractor_pool(cfg.corpus.distractors)
    if max_scale > 5000:
        pool += sample_fever(cfg.corpus.fever_sample, cfg.corpus.seed_random)
    print(f"  raw distractor pool: {len(pool):,}", flush=True)

    full = seed + pool
    full = deduplicate(full, cfg.corpus.dedup_jaccard_threshold)
    print(f"  after dedup: {len(full):,}", flush=True)

    sliced = build_prefixed_corpus(full, cfg.scales, cfg.corpus.seed_random)
    for n in sorted(sliced):
        print(f"  scale {n:>9,}: {len(sliced[n]):>9,} docs", flush=True)

    write_jsonl(CACHE / "scifact_seed.jsonl", [
        {"id": d.doc_id, "title": d.title, "text": d.text} for d in sliced.get(5000, seed)
    ])
    print(f"  wrote {CACHE / 'scifact_seed.jsonl'}", flush=True)

    onem = sliced[max_scale]
    write_jsonl(CACHE / "corpus.jsonl", [
        {"id": d.doc_id, "title": d.title, "text": d.text} for d in onem
    ])
    print(f"  wrote {CACHE / 'corpus.jsonl'} ({len(onem):,} rows)", flush=True)

    queries, qrels = load_scifact_queries_and_qrels()
    write_jsonl(CACHE / "queries.jsonl", queries)
    write_jsonl(CACHE / "qrels.jsonl", qrels)

    if not args.skip_embeddings:
        cache = EmbeddingCache(CACHE)
        for entry in cfg.encoders:
            print(f"=== Embedding corpus with {entry.slug} ({entry.model_id}) ===", flush=True)
            from trivium.embeddings.sentence import Sentence

            embedder = Sentence(
                slug=entry.slug,
                model_id=entry.model_id,
                dimension=entry.dimension,
                prompt_prefix_doc=entry.prompt_prefix_doc,
                prompt_prefix_query=entry.prompt_prefix_query,
                batch_size=entry.batch_size,
                max_seq_length=entry.max_seq_length,
                normalize=entry.normalize,
                device=entry.device,
            )
            embed_corpus_per_encoder(
                onem,
                embedder,
                cache,
                CACHE,
                batch_size=entry.batch_size,
                max_seq_length=entry.max_seq_length,
                normalize=entry.normalize,
            )

    manifest = Manifest(
        scales=ManifestScales(counts={str(k): len(v) for k, v in sliced.items()}),
        encoders=ManifestEncoders(entries={e.slug: e.model_id for e in cfg.encoders}),
        corpus_sha256=hashlib.sha256((CACHE / "corpus.jsonl").read_bytes()).hexdigest(),
        n_queries=len(queries),
        n_qrels=len(qrels),
    )
    manifest.save(CACHE / "manifest.json")
    print(f"  manifest: {CACHE / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
