"""Data preparation: build nested 1M corpus + embeddings for scifact stress test.

Built once per machine; all subsequent stages reuse the cache.

Layout:
  data/cache/
    corpus.jsonl       full nested corpus (5K seed + 1M seed+distractors), title+text+id
    vectors.npz        float32 embeddings, l2-normalized, shape (N, 384)
    queries.jsonl      scifact test queries {id, text}
    qrels.jsonl        scifact test qrels {qid, did, rel}
    manifest.json      corpus hash, embedding model revision, sizes per scale
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def normalize_text(title: str, text: str) -> str:
    """BEIR-style concatenation: title + newline + text. Whitespace-collapsed."""
    title = re.sub(r"\s+", " ", (title or "").strip())
    text = re.sub(r"\s+", " ", (text or "").strip())
    if title and text:
        return f"{title}\n{text}"
    return title or text


def beir_corpus_iter(repo_id: str):
    """Yield (id, title, text) from a BeIR/<dataset> corpus via `datasets`."""
    ds = load_dataset(repo_id, "corpus", split="corpus")
    for row in ds:
        yield row["_id"], row.get("title", "") or "", row.get("text", "") or ""


def load_seed_corpus() -> list[dict]:
    """Untouched scifact 5K — the equivalence-check baseline. Uses canonical BEIR zip."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip", "data/cache/beir_raw")
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    out = []
    for did, payload in corpus.items():
        out.append({"id": f"scifact:{did}", "title": payload.get("title", "") or "",
                    "text": payload.get("text", "") or ""})
    return out


def load_distractor_pool(cfg: dict) -> list[dict]:
    """Scientific + encyclopedic distractors. No web (MS MARCO) — domain mismatch."""
    pool: list[dict] = []
    for src in cfg["corpus"]["distractors"]:
        repo = f"BeIR/{src}"
        n_before = len(pool)
        for did, title, text in beir_corpus_iter(repo):
            pool.append({"id": f"distractor:{did}", "title": title, "text": text})
        print(f"  {src}: {len(pool) - n_before:,} docs", flush=True)
    return pool


def sample_fever(n_target: int, seed: int) -> list[dict]:
    """Sample FEVER at a fixed seed to pad the distractor pool."""
    rng = random.Random(seed)
    out = []
    for did, title, text in beir_corpus_iter("BeIR/fever"):
        out.append({"id": f"distractor:{did}", "title": title, "text": text})
    rng.shuffle(out)
    out = out[:n_target]
    print(f"  fever: {len(out):,} sampled", flush=True)
    return out


def dedup(corpus: list[dict], seed_text_hashes: set, jaccard_threshold: float) -> list[dict]:
    """Exact-text + near-duplicate dedup against seed using MinHash LSH.

    Bug-fix: the seed docs are KEPT regardless of their hash appearing in
    seed_text_hashes (we check seed_text_hashes only for distractor docs).
    MinHash scales to millions of documents; O(n) insert + O(1) query.
    """
    from datasketch import MinHash, MinHashLSH

    seen: set[str] = set()
    out = []

    seed_lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=64)
    for d in corpus:
        if not d["id"].startswith("scifact:"):
            continue
        text = d["title"] + "\n" + d["text"]
        m = MinHash(num_perm=64)
        for tok in set(text.lower().split()):
            m.update(tok.encode("utf-8"))
        seed_lsh.insert(d["id"], m)

    for d in corpus:
        text = d["title"] + "\n" + d["text"]
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if d["id"].startswith("scifact:"):
            out.append(d)
            continue
        if h in seed_text_hashes:
            continue
        m = MinHash(num_perm=64)
        for tok in set(text.lower().split()):
            m.update(tok.encode("utf-8"))
        if seed_lsh.query(m):
            continue
        out.append(d)
    return out


def shuffle_and_slice(corpus: list[dict], seed: int, scales: list[int]) -> dict[int, list[dict]]:
    """Seeded shuffle. Each scale is a true prefix of the larger one."""
    rng = random.Random(seed)
    order = list(range(len(corpus)))
    rng.shuffle(order)
    sorted_corpus = [corpus[i] for i in order]
    sliced = {}
    for n in sorted(scales):
        sliced[n] = sorted_corpus[:n]
    sliced[5000] = [d for d in corpus if d["id"].startswith("scifact:")]
    return sliced


def load_queries() -> list[dict]:
    """scifact queries + qrels from the canonical BEIR zip (300 test queries).

    Uses beir.util.download_and_unzip to get the exact BEIR-1.0.0-scifact test split,
    which is what every published baseline number was measured against.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip", "data/cache/beir_raw")
    split = "test"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    qids = list(queries.keys())
    qid_set = set(qids)
    out_queries = [{"id": qid, "text": queries[qid]} for qid in qids]
    out_qrels = []
    for qid, qd in qrels.items():
        for did, rel in qd.items():
            out_qrels.append({"qid": qid, "did": f"scifact:{did}", "rel": int(rel)})
    return out_queries, out_qrels


def embed(texts: list[str], model_name: str, batch_size: int, max_seq_length: int, normalize: bool) -> np.ndarray:
    """Sentence-transformers batched encode. l2-normalize if requested."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(vecs, dtype=np.float32)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    p.add_argument("--skip-embeddings", action="store_true", help="Use cached vectors.npz if present")
    p.add_argument("--max-scales", default=None, help="Cap scales for fast iteration, e.g. '5000,100000'")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_scales:
        cfg["scales"] = [int(x) for x in args.max_scales.split(",")]
    max_scale = max(cfg["scales"])

    os.environ["PYTHONHASHSEED"] = str(cfg["runtime"]["python_hash_seed"])

    print(f"=== Build corpus ===", flush=True)
    seed = load_seed_corpus()
    print(f"  scifact seed: {len(seed):,}", flush=True)
    seed_text_hashes = {hashlib.sha256((d['title'] + '\n' + d['text']).encode('utf-8')).hexdigest() for d in seed}

    pool = load_distractor_pool(cfg)
    if max_scale > 5000:
        pool += sample_fever(cfg["corpus"]["fever_sample"], cfg["corpus"]["seed_random"])
    print(f"  raw distractor pool: {len(pool):,}", flush=True)

    full = seed + pool
    full = dedup(full, seed_text_hashes, cfg["corpus"]["dedup_jaccard_threshold"])
    print(f"  after dedup: {len(full):,}", flush=True)

    scaled = shuffle_and_slice(full, cfg["corpus"]["seed_random"], cfg["scales"] + [max_scale])
    for n in sorted(scaled.keys()):
        print(f"  scale {n:>9,}: {len(scaled[n]):>9,} docs", flush=True)

    # Persist the full scifact seed separately so 5K scale can always recover it,
    # even when scifact docs are scattered through the larger shuffled corpus.
    write_jsonl(CACHE / "scifact_seed.jsonl", seed)
    print(f"  wrote {CACHE / 'scifact_seed.jsonl'} ({len(seed):,} rows)", flush=True)

    # Persist only the requested max scale (smaller scales are prefixes).
    # Sampling is seeded and deterministic, so re-running with a larger max_scale
    # gives a superset of the smaller scale's prefix.
    onem = scaled[max_scale]
    write_jsonl(CACHE / "corpus.jsonl", onem)
    print(f"  wrote {CACHE / 'corpus.jsonl'} ({len(onem):,} rows)", flush=True)

    queries, qrels = load_queries()
    write_jsonl(CACHE / "queries.jsonl", queries)
    write_jsonl(CACHE / "qrels.jsonl", qrels)
    print(f"  queries: {len(queries):,}; qrels: {len(qrels):,}", flush=True)

    emb_path = CACHE / "vectors.npz"
    if args.skip_embeddings and emb_path.exists():
        print(f"  reusing cached embeddings: {emb_path}", flush=True)
    else:
        # Embed the full scifact seed first (so 5K scale always has all 5183 docs),
        # then the rest of the 100K corpus (excluding scifact-prefixed docs already seen).
        seed_ids = {d["id"] for d in seed}
        embedded_ids = set()
        chunks = []

        print(f"=== Embedding {len(seed):,} seed docs (scifact) ===", flush=True)
        seed_texts = [normalize_text(d["title"], d["text"]) for d in seed]
        seed_vecs = embed(
            seed_texts,
            cfg["embedding"]["model"],
            cfg["embedding"]["batch_size"],
            cfg["embedding"]["max_seq_length"],
            cfg["embedding"]["normalize"],
        )
        chunks.append((seed_vecs, [d["id"] for d in seed]))
        embedded_ids.update(d["id"] for d in seed)

        rest = [d for d in onem if d["id"] not in embedded_ids]
        if rest:
            print(f"=== Embedding {len(rest):,} distractor docs ===", flush=True)
            rest_texts = [normalize_text(d["title"], d["text"]) for d in rest]
            rest_vecs = embed(
                rest_texts,
                cfg["embedding"]["model"],
                cfg["embedding"]["batch_size"],
                cfg["embedding"]["max_seq_length"],
                cfg["embedding"]["normalize"],
            )
            chunks.append((rest_vecs, [d["id"] for d in rest]))

        all_vecs = np.concatenate([c[0] for c in chunks], axis=0)
        all_ids = np.concatenate([np.array(c[1]) for c in chunks], axis=0)
        np.savez_compressed(emb_path, vectors=all_vecs, ids=all_ids)
        print(f"  wrote {emb_path} shape={all_vecs.shape} dtype={all_vecs.dtype}", flush=True)

    manifest = {
        "scales": {n: len(scaled[n]) for n in sorted(scaled.keys())},
        "embedding_model": cfg["embedding"]["model"],
        "embedding_dim": cfg["embedding"]["dim"],
        "corpus_sha256": hashlib.sha256(open(CACHE / "corpus.jsonl", "rb").read()).hexdigest(),
        "scifact_seed_count": len(seed),
        "distractor_count": len(onem) - len(seed),
        "n_queries": len(queries),
        "n_qrels": len(qrels),
    }
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {CACHE / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
