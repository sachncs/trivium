"""GigaToken: trivium wrapper over the gigatoken Rust tokeniser.

gigatoken (https://github.com/marcelroed/gigatoken) provides a
Rust-backed BPE/SentencePiece tokeniser with GB/s throughput and
awkward-array batch outputs. This module wraps it in the
trivium.* style — frozen dataclass summaries, single-word class
names, no semi-private helpers — so the rest of the package can
track token throughput without taking a hard dependency on
gigatoken's API.

Public surface:
- count_streaming(texts, batch_size=8192): bounded-memory token
  counting over an iterable. Returns GigaTokenSummary.
- tokenize_to_memmap(texts, out_path): persist a 1B-token corpus
  to a uint32 memmap so the index never loads the full token
  stream into RAM.
- tokenize_jsonl(path): gigatoken.JsonlFileSource streaming.
- tokenize_textfile(path): gigatoken.TextFileSource streaming.
- train_bpe(...): wrapper around gigatoken.train_bpe.
- GigaTokenSummary: frozen dataclass with tokens/sec and
  bytes/sec.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import gigatoken as external_gigatoken
import numpy as np


@dataclass(frozen=True)
class GigaTokenSummary:
    """A frozen bundle of throughput statistics from one GigaToken run."""

    total_tokens: int
    vocab_size: int
    doc_count: int
    bytes_processed: int
    seconds_elapsed: float
    tokens_per_second: float
    bytes_per_second: float

    def to_dict(self) -> dict:
        return {
            "total_tokens": int(self.total_tokens),
            "vocab_size": int(self.vocab_size),
            "doc_count": int(self.doc_count),
            "bytes_processed": int(self.bytes_processed),
            "seconds_elapsed": round(float(self.seconds_elapsed), 3),
            "tokens_per_second": round(float(self.tokens_per_second), 1),
            "bytes_per_second": round(float(self.bytes_per_second), 1),
        }


class GigaToken:
    """Thin trivium-style wrapper around gigatoken.Tokenizer.

    Args:
        tokenizer: Either a HuggingFace tokenizer id (e.g. 'gpt2'),
            a path to a local tokenizer.json, or an already-constructed
            gigatoken.Tokenizer instance. Internally we always wrap
            with gigatoken.Tokenizer for the Rust pipeline.
    """

    DEFAULT_BUILTIN_TOKENIZER = "gpt2"

    def __init__(self, tokenizer: str | Path | object = DEFAULT_BUILTIN_TOKENIZER) -> None:
        self.gigatoken: external_gigatoken.Tokenizer = external_gigatoken.Tokenizer(tokenizer)
        self.vocab_size: int = int(self.gigatoken.vocab_size)
        self.model_id: str = str(tokenizer) if isinstance(tokenizer, str | Path) else "<instance>"

    @property
    def slug(self) -> str:
        """Short identifier used in CSV rows."""
        return f"gigatoken:{self.model_id}"

    @property
    def native(self) -> external_gigatoken.Tokenizer:
        """Access the underlying gigatoken.Tokenizer for advanced use."""
        return self.gigatoken

    def count_streaming(
        self,
        texts: Iterable[str],
        batch_size: int = 8192,
    ) -> GigaTokenSummary:
        """Tokenise an iterable of strings; return throughput stats.

        Memory: O(batch_size * max_seq_len) during a batch, then
        discarded. Uses gigatoken's encode_batch with parallel=True
        for gigatoken's native multi-thread throughput.

        Args:
            texts: Iterable of str (generator, file handle, list).
            batch_size: Number of docs per gigatoken batch. gigatoken
                scales to 8K-32K on modern hardware.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        total = 0
        doc_count = 0
        bytes_processed = 0
        t0 = time.perf_counter()
        for batch in batched_texts(texts, batch_size):
            bytes_processed += sum(len(t.encode("utf-8")) for t in batch)
            encoded = self.gigatoken.encode_batch(batch, parallel=True)
            total += int(sum(len(row.tolist()) for row in encoded))
            doc_count += len(batch)
        elapsed = time.perf_counter() - t0
        return GigaTokenSummary(
            total_tokens=total,
            vocab_size=self.vocab_size,
            doc_count=doc_count,
            bytes_processed=bytes_processed,
            seconds_elapsed=elapsed,
            tokens_per_second=total / elapsed if elapsed > 0 else 0.0,
            bytes_per_second=bytes_processed / elapsed if elapsed > 0 else 0.0,
        )

    def tokenize_to_memmap(
        self,
        texts: Iterable[str],
        out_dir: str | Path,
        batch_size: int = 8192,
        dtype: type = np.uint32,
    ) -> dict:
        """Stream-tokenise to a uint32 memmap layout.

        Layout:
            out_dir/tokens.bin        concatenated uint32 token IDs
            out_dir/offsets.bin       int64: doc_id -> start
                                     (length = doc_count + 1)
            out_dir/manifest.json     GigaTokenSummary.to_dict()

        Returns the manifest dict.
        """
        if dtype != np.uint32 and dtype != np.int64:
            raise ValueError(f"gigatoken emits uint32; dtype must be uint32 or int64, got {dtype}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = out_dir / "tokens.bin"
        offsets_path = out_dir / "offsets.bin"

        bytes_processed = 0
        t0 = time.perf_counter()
        offsets: list[int] = [0]
        with open(tokens_path, "wb") as tokens_f:
            for batch in batched_texts(texts, batch_size):
                bytes_processed += sum(len(t.encode("utf-8")) for t in batch)
                encoded = self.gigatoken.encode_batch(batch, parallel=True)
                for row in encoded:
                    arr = np.asarray(row.tolist(), dtype=np.uint32)
                    arr.tofile(tokens_f)
                    offsets.append(offsets[-1] + arr.size)
                del encoded

        offsets_arr = np.asarray(offsets, dtype=np.int64)
        with open(offsets_path, "wb") as offsets_f:
            offsets_arr.tofile(offsets_f)

        elapsed = time.perf_counter() - t0
        total_tokens = int(offsets_arr[-1])
        doc_count = len(offsets_arr) - 1
        manifest = GigaTokenSummary(
            total_tokens=total_tokens,
            vocab_size=self.vocab_size,
            doc_count=doc_count,
            bytes_processed=bytes_processed,
            seconds_elapsed=elapsed,
            tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
            bytes_per_second=bytes_processed / elapsed if elapsed > 0 else 0.0,
        )
        out_dir.joinpath("manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
        return manifest.to_dict()

    def tokenize_jsonl(
        self,
        jsonl_path: str | Path,
        text_field: str = "text",
    ) -> list[list[int]]:
        """Tokenise a JSONL file using gigatoken's native JsonlFileSource.

        Memory: the underlying file source streams line-by-line; only
        the awkward-array encoding is held in RAM. For 1B tokens,
        chunk the result.
        """
        source = external_gigatoken.JsonlFileSource([str(jsonl_path)], field=text_field)
        encoded = self.gigatoken.encode_files(source, parallel=True)
        return [list(row.tolist()) for row in encoded]

    def tokenize_textfile(self, txt_path: str | Path) -> list[list[int]]:
        """Tokenise a plain text file (one doc per line) via gigatoken's TextFileSource.

        With the default ``separator=None`` each file is one document;
        pass ``separator=...`` to split on a literal pattern.
        """
        source = external_gigatoken.TextFileSource([str(txt_path)], separator="\n")
        encoded = self.gigatoken.encode_files(source, parallel=True)
        return [list(row.tolist()) for row in encoded]

    @staticmethod
    def train_bpe(
        corpus_paths: list[str | Path],
        vocab_size: int,
        special_tokens: dict[str, int] | None = None,
        tie_breaking: str = "huggingface",
    ) -> object:
        """Train a BPE tokenizer on gigatoken's Rust backend.

        Returns the bytes of the resulting tokenizer.json. Persist
        with Path(out).write_bytes(...) and reload via GigaToken(path).
        """
        return external_gigatoken.train_bpe(
            in_data=[str(p) for p in corpus_paths],
            vocab_size=int(vocab_size),
            special_tokens=special_tokens or {},
            tie_breaking=tie_breaking,
        )


def batched_texts(iterable: Iterable[str], batch_size: int):
    """Yield lists of up to batch_size items from the iterable."""
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
