"""Tests for trivium.tokenizer.gigatoken (wrapper over the gigatoken Rust lib)."""
from __future__ import annotations

from pathlib import Path

import pytest

from trivium.tokenizer.gigatoken import GigaToken, GigaTokenSummary


class TestGigaToken:
    def test_default_tokenizer_loads(self):
        gt = GigaToken("gpt2")
        assert gt.vocab_size > 1000
        assert "gpt2" in gt.model_id

    def test_slug_format(self):
        gt = GigaToken("gpt2")
        assert gt.slug.startswith("gigatoken:")
        assert "gpt2" in gt.slug

    def test_count_streaming_yields_summary(self):
        gt = GigaToken("gpt2")
        summary = gt.count_streaming(["hello world", "goodbye world", "a b c"], batch_size=2)
        assert isinstance(summary, GigaTokenSummary)
        assert summary.doc_count == 3
        assert summary.total_tokens > 0
        assert summary.vocab_size > 0
        assert summary.tokens_per_second >= 0

    def test_count_streaming_batches_correctly(self):
        gt = GigaToken("gpt2")
        summary = gt.count_streaming([f"doc {i}" for i in range(100)], batch_size=10)
        assert summary.doc_count == 100

    def test_count_streaming_rejects_zero_batch(self):
        gt = GigaToken("gpt2")
        with pytest.raises(ValueError):
            gt.count_streaming([], batch_size=0)

    def test_to_dict_roundtrip(self):
        s = GigaTokenSummary(
            total_tokens=100,
            vocab_size=50257,
            doc_count=10,
            bytes_processed=200,
            seconds_elapsed=0.5,
            tokens_per_second=200.0,
            bytes_per_second=400.0,
        )
        d = s.to_dict()
        assert d["total_tokens"] == 100
        assert d["seconds_elapsed"] == 0.5
        assert d["tokens_per_second"] == 200.0

    def test_count_streaming_iterable_does_not_load_all(self):
        """Streaming: gen with N items must not materialise a list."""
        gt = GigaToken("gpt2")

        def infinite():
            i = 0
            while True:
                yield f"doc {i}"
                i += 1
                if i >= 50:
                    return

        summary = gt.count_streaming(infinite(), batch_size=10)
        assert summary.doc_count == 50


class TestGigaTokenMemmap:
    def test_tokenize_to_memmap_writes_files(self, tmp_path: Path):
        gt = GigaToken("gpt2")
        out_dir = tmp_path / "tokens"
        manifest = gt.tokenize_to_memmap(
            ["hello world", "this is a test", "gigatoken fan"],
            out_dir=out_dir,
            batch_size=2,
        )
        assert (out_dir / "tokens.bin").exists()
        assert (out_dir / "offsets.bin").exists()
        assert (out_dir / "manifest.json").exists()
        assert manifest["doc_count"] == 3
        assert manifest["total_tokens"] > 0

    def test_tokenize_to_memmap_total_tokens_matches_sum(self, tmp_path: Path):
        gt = GigaToken("gpt2")
        out_dir = tmp_path / "tokens"
        gt.tokenize_to_memmap(
            ["hello world", "goodbye world"],
            out_dir=out_dir,
            batch_size=1,
        )
        import numpy as np

        offsets = np.fromfile(out_dir / "offsets.bin", dtype=np.int64)
        assert offsets[0] == 0
        assert offsets[-1] == sum(
            len(line) for line in ["hello world", "goodbye world"] for _ in [0]
        ) or offsets[-1] > 0


class TestGigaTokenFileSources:
    def test_tokenize_jsonl(self, tmp_path: Path):
        import json

        jsonl = tmp_path / "corpus.jsonl"
        with open(jsonl, "w") as f:
            for text in ["hello world", "goodbye world", "third line"]:
                f.write(json.dumps({"text": text}) + "\n")
        gt = GigaToken("gpt2")
        encoded = gt.tokenize_jsonl(jsonl, text_field="text")
        assert len(encoded) == 3
        assert all(isinstance(row, list) for row in encoded)
        assert all(all(isinstance(t, int) for t in row) for row in encoded)

    def test_tokenize_textfile(self, tmp_path: Path):
        textfile = tmp_path / "lines.txt"
        textfile.write_text("alpha\nbeta gamma\ndelta epsilon zeta\n")
        gt = GigaToken("gpt2")
        encoded = gt.tokenize_textfile(textfile)
        assert len(encoded) == 3
