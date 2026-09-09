"""Tests that vector.nlist_by_scale matches the 4*sqrt(N) recipe.

For each configured scale, nlist must equal the smallest power of
two >= 4 * sqrt(N). This pins the doc-comment to the on-disk value
so a hand-edit cannot drift the formula.
"""
from __future__ import annotations

import pytest

from trivium.config.loader import load_config


@pytest.mark.parametrize(
    "scale,expected",
    [
        (5_000, 512),
        (100_000, 2048),
        (500_000, 4096),
        (1_000_000, 4096),
    ],
)
def test_nlist_matches_formula(scale: int, expected: int) -> None:
    cfg = load_config()
    assert cfg.vector.nlist_by_scale[scale] == expected, (
        f"nlist@{scale} drifted from 4*sqrt(N) recipe: got "
        f"{cfg.vector.nlist_by_scale[scale]}, expected {expected}"
    )


def test_all_scales_have_nlist() -> None:
    cfg = load_config()
    for scale in cfg.scales:
        assert scale in cfg.vector.nlist_by_scale, (
            f"nlist missing for scale {scale}; loader would KeyError at runtime"
        )
