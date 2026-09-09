"""Load + validate configs/default.yaml into the pydantic Config model.

The pydantic model in schema.py is the source of truth. This loader
translates the YAML nesting into the flat-but-namespaced pydantic tree.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from trivium.config.schema import (
    BenchmarkConfig,
    Bm25Config,
    Config,
    CorpusConfig,
    EncoderEntry,
    HybridConfig,
    PreflightConfig,
    RerankConfig,
    RuntimeConfig,
    VectorConfig,
)


def default_config_path() -> Path:
    """Locate the bundled `configs/default.yaml` regardless of install mode.

    Prefers a repo-root `configs/default.yaml` if one exists next to the
    current working directory (so contributors can edit the canonical
    file in place). Falls back to the file shipped inside the installed
    `trivium` package.
    """
    repo_local = Path.cwd() / "configs" / "default.yaml"
    if repo_local.is_file():
        return repo_local
    return Path(str(resources.files("trivium.configs").joinpath("default.yaml")))


def list_pairs(value: Any) -> list[tuple[float, float]]:
    """Convert the YAML list-of-pairs shape into a typed list."""
    if value is None:
        return []
    return [(float(a), float(b)) for a, b in value]


def to_int_keys(d: Any) -> dict[int, int]:
    """Convert YAML string-keyed mapping to int-keyed mapping."""
    if d is None:
        return {}
    return {int(k): int(v) for k, v in d.items()}


def load_config(path: str | Path | None = None) -> Config:
    """Parse YAML at `path` and return a validated Config.

    When `path` is None the bundled default config is used.
    """
    if path is None:
        path = default_config_path()
    with open(path) as f:
        raw = yaml.safe_load(f)

    corpus = CorpusConfig(**raw.get("corpus", {}))
    scales = [int(s) for s in raw.get("scales", [5000, 100000, 500000, 1000000])]
    encoders = [EncoderEntry(**e) for e in raw.get("encoders", [])]

    raw_bm25 = raw.get("bm25", {})
    bm25 = Bm25Config(**raw_bm25)

    raw_vector = raw.get("vector", {})
    if "nlist" in raw_vector and isinstance(raw_vector["nlist"], list):
        legacy_nlist = [int(n) for n in raw_vector["nlist"]]
        if len(legacy_nlist) != len(scales):
            raise ValueError(
                f"vector.nlist has {len(legacy_nlist)} entries but scales has {len(scales)}"
            )
        nlist_by_scale = dict(zip(scales, legacy_nlist, strict=False))
    else:
        nlist_by_scale = to_int_keys(raw_vector.get("nlist_by_scale", {}))
    vector = VectorConfig(
        nlist_by_scale=nlist_by_scale,
        nprobe_sweep=[
            int(x)
            for x in raw_vector.get(
                "nprobe", raw_vector.get("nprobe_sweep", [8, 16, 32, 64, 128, 256])
            )
        ],
        m=int(raw_vector.get("m", 48)),
        nbits=int(raw_vector.get("nbits", 4)),
        use_opq=bool(raw_vector.get("use_opq", True)),
        use_rflat=bool(raw_vector.get("use_rflat", True)),
        k_factor=int(raw_vector.get("k_factor", 4)),
        min_scale_for_ivfpq=int(raw_vector.get("min_scale_for_ivfpq", 5000)),
        train_size_strategy=raw_vector.get("train_size_strategy", "50_x_nlist"),
        train_size_fixed=int(raw_vector.get("train_size_fixed", 30000)),
    )

    raw_hybrid = raw.get("hybrid", {})
    hybrid = HybridConfig(
        rrf_k=int(raw_hybrid.get("rrf_k", 60)),
        rrf_k_sweep=[int(x) for x in raw_hybrid.get("rrf_k_sweep", [10, 30, 60, 100, 200])],
        rrf_weight_sweep=list_pairs(raw_hybrid.get("rrf_weight_sweep")),
        candidate_pool=int(raw_hybrid.get("candidate_pool", 100)),
    )

    raw_rerank = raw.get("rerank", {})
    rerank = RerankConfig(
        **{
            k: raw_rerank[k]
            for k in ("slug", "model_id", "max_length", "batch_size", "candidate_pool")
            if k in raw_rerank
        }
    )

    raw_benchmark = raw.get("benchmark", {})
    benchmark = BenchmarkConfig(
        warmup=int(raw_benchmark.get("warmup", 20)),
        iters=int(raw_benchmark.get("iters", 3)),
        top_k_eval=[int(x) for x in raw_benchmark.get("top_k_eval", [10, 100])],
    )

    raw_preflight = raw.get("preflight", {})
    preflight = PreflightConfig(**raw_preflight)

    raw_runtime = raw.get("runtime", {})
    runtime = RuntimeConfig(**raw_runtime)

    return Config(
        corpus=corpus,
        scales=scales,
        encoders=encoders,
        bm25=bm25,
        vector=vector,
        hybrid=hybrid,
        rerank=rerank,
        benchmark=benchmark,
        preflight=preflight,
        runtime=runtime,
    )
