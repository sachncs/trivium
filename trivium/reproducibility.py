"""Reproducibility manifest: per-row provenance of versions and env."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ReproducibilityManifest:
    """A snapshot of the runtime environment for a benchmark row."""

    python: str = ""
    platform: str = ""
    numpy: str = ""
    faiss: str = ""
    sentence_transformers: str = ""
    torch: str = ""
    bm25s: str = ""
    pydantic: str = ""
    faiss_omp_threads: int = 0
    git_sha: str = ""
    device: str = ""
    extras: dict = field(default_factory=dict)

    @classmethod
    def gather(cls, device: str = "") -> ReproducibilityManifest:
        """Capture the current process environment."""
        versions = ReproducibilityManifest.safe_versions()
        return cls(
            python=sys.version.split()[0],
            platform=platform.platform(),
            numpy=np.__version__,
            faiss=versions.get("faiss", ""),
            sentence_transformers=versions.get("sentence_transformers", ""),
            torch=versions.get("torch", ""),
            bm25s=versions.get("bm25s", ""),
            pydantic=versions.get("pydantic", ""),
            faiss_omp_threads=omp_threads(),
            git_sha=git_sha(),
            device=device,
            extras={},
        )

    @staticmethod
    def safe_versions() -> dict[str, str]:
        out: dict[str, str] = {}
        for name in ("faiss", "sentence_transformers", "torch", "bm25s", "pydantic"):
            # Use a subprocess so a faulty native import (for example a torch
            # wheel that crashes on import on this OS) cannot abort the
            # benchmark process.
            try:
                import json as _json
                import subprocess as _sp

                r = _sp.run(
                    [sys.executable, "-c", f"import {name}; print({name}.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if r.returncode == 0 and r.stdout.strip():
                    out[name] = r.stdout.strip()
            except Exception:
                continue
        return out

    def to_dict(self) -> dict:
        base = {
            "python": self.python,
            "platform": self.platform,
            "numpy": self.numpy,
            "faiss": self.faiss,
            "sentence_transformers": self.sentence_transformers,
            "torch": self.torch,
            "bm25s": self.bm25s,
            "pydantic": self.pydantic,
            "faiss_omp_threads": self.faiss_omp_threads,
            "git_sha": self.git_sha,
            "device": self.device,
        }
        for k, v in sorted(self.extras.items()):
            if k not in base:
                base[k] = v
        return base


def omp_threads() -> int:
    # Run the faiss probe in a subprocess so a faulty native import or
    # threading crash on this OS cannot take down the benchmark process.
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import faiss; print(faiss.omp_get_max_threads())"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip().isdigit():
            return int(r.stdout.strip())
    except Exception:
        pass
    return 0


def git_sha() -> str:
    try:
        from pathlib import Path

        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""
