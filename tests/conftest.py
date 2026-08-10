"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
CONFIG_PATH = ROOT / "configs" / "default.yaml"

sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    return CACHE


@pytest.fixture(scope="session")
def default_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def config_path() -> Path:
    return CONFIG_PATH


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark tests in tests/golden/ as 'golden' so they can be selected via -m golden."""
    for item in items:
        if "golden" in str(item.fspath):
            item.add_marker(pytest.mark.golden)
