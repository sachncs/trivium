"""Pydantic v2 schema for configs/default.yaml.

The legacy YAML loader was a `yaml.safe_load(open(path))` followed
by deep dict access (`cfg["vector"]["nprobe"][2]`). Any typo or
missing field would silently fall through and only manifest at
runtime deep in a benchmark row. This schema:

- Forces every field to be present or have a default.
- Validates types at load time.
- Replaces the brittle `cfg["vector"]["nlist"][cfg["scales"].index(scale)]`
  pattern with `config.vector.nlist_for_scale(scale)` (looked up
  in a dict by scale, raises KeyError if scale not configured).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: str = "scifact"
    distractors: list[str] = Field(default_factory=lambda: ["scidocs", "trec-covid", "nfcorpus"])
    fever_sample: int = 0
    seed_random: int = 42
    dedup_jaccard_threshold: float = 0.85


class EncoderEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    model_id: str
    dimension: int
    prompt_prefix_doc: str = ""
    prompt_prefix_query: str = ""
    device: str = "cpu"
    batch_size: int = 128
    max_seq_length: int = 256
    normalize: bool = True


class Bm25Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "lucene"
    k1: float = 0.9
    b: float = 0.4
    candidate_pool: int = 100
    stopwords: str = "en"
    stemmer: str | None = None


class VectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # nlist keyed by scale so lookups by scale are O(1) instead of
    # `cfg["scales"].index(scale)` (legacy ValueError-prone pattern).
    nlist_by_scale: dict[int, int] = Field(default_factory=dict)
    nprobe_sweep: list[int] = Field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    m: int = 48
    nbits: int = 4
    use_opq: bool = True
    use_rflat: bool = True
    k_factor: int = 4
    # Replaces the hardcoded `scale <= 5000` in legacy run_vector.
    min_scale_for_ivfpq: int = 5000
    train_size_strategy: Literal["sqrt_n", "50_x_nlist", "fixed_N"] = "50_x_nlist"
    train_size_fixed: int = 30_000

    def nlist_for_scale(self, scale: int) -> int:
        if scale not in self.nlist_by_scale:
            raise KeyError(
                f"nlist not configured for scale={scale}; configure vector.nlist_by_scale in default.yaml"
            )
        return self.nlist_by_scale[scale]


class HybridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rrf_k: int = 60
    rrf_k_sweep: list[int] = Field(default_factory=lambda: [10, 30, 60, 100, 200])
    rrf_weight_sweep: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]
    )
    candidate_pool: int = 100


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str = "encoder"
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 256
    batch_size: int = 32
    candidate_pool: int = 50


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warmup: int = 20
    iters: int = 3
    top_k_eval: list[int] = Field(default_factory=lambda: [10, 100])


class PreflightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_ndcg10: float = 0.6789
    tolerance: float = 0.03
    min_observed: float = 0.62


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    random_seed: int = 42
    python_hash_seed: int = 42
    faiss_omp_threads: int = 1


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    scales: list[int] = Field(default_factory=lambda: [5_000, 100_000, 500_000, 1_000_000])
    encoders: list[EncoderEntry] = Field(default_factory=list)
    bm25: Bm25Config = Field(default_factory=Bm25Config)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
