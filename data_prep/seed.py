"""Load untouched scifact 5K from BEIR canonical zip.

Replaces data/prepare.py:load_seed_corpus. The scifact zip is
fetched once from the BEIR UKP server and cached locally.
"""
from __future__ import annotations

from trivium.domain.document import Document


def load_scifact_seed() -> list[Document]:
    """Untouched scifact 5K (the equivalence-check baseline)."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "data/cache/beir_raw",
    )
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    return [
        Document(
            doc_id=f"scifact:{did}",
            title=payload.get("title", "") or "",
            text=payload.get("text", "") or "",
        )
        for did, payload in corpus.items()
    ]


def load_scifact_queries_and_qrels() -> tuple[list[dict], list[dict]]:
    """scifact queries + qrels from the canonical BEIR zip.

    Returns (queries_rows, qrels_rows) as plain dicts so the
    caller can persist them through CorpusCache.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "data/cache/beir_raw",
    )
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    qids = list(queries.keys())
    out_queries = [{"id": qid, "text": queries[qid]} for qid in qids]
    out_qrels = []
    for qid, qd in qrels.items():
        for did, rel in qd.items():
            out_qrels.append({"qid": qid, "did": f"scifact:{did}", "rel": int(rel)})
    return out_queries, out_qrels
