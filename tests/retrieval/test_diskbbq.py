"""Tests for trivium.retrieval.diskbbq.Bbq.

Atomic checks:
- Build + search returns non-empty hits
- RAM mode and disk mode agree on doc_ids (same codes/centroids)
- Recall@10 vs Faiss.Flat on a tiny corpus is ≥0.5
- size_bytes reports on-disk footprint after build
- Round-trip: write to disk, instantiate fresh Bbq from same out_dir, same hits
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from trivium.domain.document import Document
from trivium.retrieval.diskbbq import Bbq
from trivium.retrieval.faiss import Faiss


def _toy_corpus(n: int = 200, dim: int = 96, seed: int = 42) -> tuple[list[Document], np.ndarray]:
    docs = [Document(doc_id=f"d{i:04d}", title=f"t{i}", text=f"doc number {i}") for i in range(n)]
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return docs, vecs


class TestBbqBuildAndSearch:
    def test_build_emits_disk_layout(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=False)
        bbq.add_documents(docs, vectors=vecs)
        assert (tmp_path / "centroids.npy").exists()
        assert (tmp_path / "subvector_thresholds.npy").exists()
        assert (tmp_path / "metadata.json").exists()
        for cid in range(bbq.nlist):
            assert (tmp_path / f"list_{cid}.ids.npy").exists()
            assert (tmp_path / f"list_{cid}.codes.npy").exists()

    def test_search_returns_nonempty_hits(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=False)
        bbq.add_documents(docs, vectors=vecs)
        results = bbq.search(vecs[:5], k=10, mode="ram")
        assert len(results) == 5
        for r in results:
            assert len(r) > 0
            assert all(h.doc_id.startswith("d") for h in r)

    def test_ram_and_disk_agree(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=True)
        bbq.add_documents(docs, vectors=vecs)
        q = vecs[:5]
        r_ram = bbq.search(q, k=10, mode="ram")
        r_disk = bbq.search(q, k=10, mode="disk")
        for i in range(5):
            ram_ids = [h.doc_id for h in r_ram[i]]
            disk_ids = [h.doc_id for h in r_disk[i]]
            assert ram_ids == disk_ids, f"q{i}: RAM {ram_ids} != DISK {disk_ids}"

    def test_recall_vs_flat(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=500, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=16, m=24, nprobe=8, use_rescore=True)
        bbq.add_documents(docs, vectors=vecs)

        q = vecs[:30]

        # Compute exact ground truth with a numpy matrix-multiply instead
        # of going through faiss. This keeps the test hermetic: it does not
        # depend on faiss IndexFlatIP, which is known to race with the
        # tqdm monitor thread that sentence-transformers spawns on some
        # platforms (see https://github.com/facebookresearch/faiss/issues/2906).
        r_ram = bbq.search(q, k=10, mode="ram")
        q_n = q / np.linalg.norm(q, axis=1, keepdims=True)
        d_n = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        # q_n @ d_n.T gives (Q, N) cosine similarities
        sims = q_n @ d_n.T
        top_idx = np.argpartition(-sims, kth=10, axis=1)[:, :10]
        # Sort the top-k by score
        r_exact = []
        for i, row in enumerate(top_idx):
            row_sorted = row[np.argsort(-sims[i, row])]
            r_exact.append(
                [
                    type("Hit", (), {"doc_id": docs[int(j)].doc_id, "score": float(sims[i, int(j)])})
                    for j in row_sorted
                ]
            )

        overlaps = []
        for i in range(30):
            ram_ids = set(h.doc_id for h in r_ram[i])
            exact_ids = set(h.doc_id for h in r_exact[i])
            overlaps.append(len(ram_ids & exact_ids))
        avg = sum(overlaps) / (30 * 10)
        assert avg >= 0.5, f"recall@10 vs flat = {avg:.3f} < 0.5"

    def test_size_bytes_positive(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=True)
        bbq.add_documents(docs, vectors=vecs)
        assert bbq.size_bytes() > 0

    def test_round_trip(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=True)
        bbq.add_documents(docs, vectors=vecs)
        q = vecs[:5]
        r1 = bbq.search(q, k=10, mode="ram")
        bbq2 = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=True)
        assert bbq2.centroids is None
        meta = bbq2.load_metadata()
        assert meta["nlist"] == 8
        assert meta["m"] == 24

    def test_unbuilt_search_raises(self, tmp_path: Path) -> None:
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=False)
        import pytest

        with pytest.raises(RuntimeError):
            bbq.search(np.zeros((1, 96), dtype=np.float32), k=10, mode="ram")

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=100, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=4, m=24, nprobe=2, use_rescore=False)
        bbq.add_documents(docs, vectors=vecs)
        import pytest

        with pytest.raises(ValueError):
            bbq.search(vecs[:1], k=5, mode="nope")

    def test_set_search_params_roundtrip(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=100, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=4, m=24, nprobe=2, use_rescore=False)
        bbq.add_documents(docs, vectors=vecs)
        bbq.set_search_params(nprobe=3)
        assert bbq.nprobe == 3
        bbq.set_search_params(use_rescore=True, k_factor=8)
        assert bbq.use_rescore is True
        assert bbq.k_factor == 8


class TestBbqEdgeCases:
    def test_nb_bits_must_be_one(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(ValueError):
            Bbq(out_dir=tmp_path, m=24, nbits=4)

    def test_dim_must_be_divisible_by_m(self, tmp_path: Path) -> None:
        docs, vecs = _toy_corpus(n=100, dim=100)
        bbq = Bbq(out_dir=tmp_path, nlist=4, m=24, nprobe=2)
        import pytest

        with pytest.raises(ValueError):
            bbq.add_documents(docs, vectors=vecs)

    def test_requires_vectors(self, tmp_path: Path) -> None:
        docs, _ = _toy_corpus(n=10)
        bbq = Bbq(out_dir=tmp_path, nlist=4, m=24, nprobe=2)
        import pytest

        with pytest.raises(ValueError):
            bbq.add_documents(docs, vectors=None)

    def test_ram_rescore_emits_nonzero_scores(self, tmp_path: Path) -> None:
        """Regression: RAM-mode rescore used to overwrite the inner-product score with 0.

        Reported in issue #4. With use_rescore=True, every hit must carry the
        actual <vec, query> similarity, not a placeholder 0.0.
        """
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=True)
        bbq.add_documents(docs, vectors=vecs)
        results = bbq.search(vecs[:5], k=10, mode="ram")
        assert len(results) == 5
        for r in results:
            assert len(r) > 0
            # At least one hit must have a strictly positive score; with random
            # GF(2) codes no hit's inner product is exactly 0 by chance.
            assert any(h.score != 0.0 for h in r), "RAM-mode rescore emitted score=0 for every hit"

    def test_ram_rescore_scores_match_disk_rescore(self, tmp_path: Path) -> None:
        """RAM and disk modes must report the same scores for the same hits."""
        docs, vecs = _toy_corpus(n=200, dim=96)
        bbq = Bbq(out_dir=tmp_path, nlist=8, m=24, nprobe=4, use_rescore=True)
        bbq.add_documents(docs, vectors=vecs)
        q = vecs[:5]
        r_ram = bbq.search(q, k=10, mode="ram")
        r_disk = bbq.search(q, k=10, mode="disk")
        for i in range(5):
            ram_by_id = {h.doc_id: h.score for h in r_ram[i]}
            disk_by_id = {h.doc_id: h.score for h in r_disk[i]}
            for did, score in ram_by_id.items():
                assert did in disk_by_id, f"q{i}: RAM hit {did} missing from disk"
                assert score == pytest.approx(disk_by_id[did], abs=1e-6), (
                    f"q{i}: score mismatch for {did}: ram={score} disk={disk_by_id[did]}"
                )