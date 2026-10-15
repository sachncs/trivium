# Contributing

## Development setup

Trivium targets Python 3.12 or newer. Create an editable development
environment with:

```bash
uv venv
uv pip install -e ".[dev]"
```

The documentation site is an Astro project in `site/`:

```bash
cd site
npm ci
npm run build
```

## Verification

Run the same checks used by CI before opening a pull request:

```bash
ruff check .
mypy
pytest -q -m "not golden" --cov=trivium --cov-branch --cov-report=term-missing --cov-fail-under=90
```

The golden tests are explicit regression anchors and may require the
benchmark data cache. Run them separately when that cache is available:

```bash
pytest -q -m golden
```

## Design rules

Keep the public contracts small: embedders, retrievers, fusion strategies,
rerankers, and benchmark pipelines implement their corresponding abstract
base class. Add a concrete implementation to the relevant registry rather
than introducing a forwarding wrapper or a compatibility alias. Use direct,
descriptive public names for variables, functions, classes, and files; Python
dunder methods remain reserved for language protocols.

Persistence boundaries must validate shape, cardinality, IDs, and metadata,
and cache writes must be atomic. Changes to benchmark behavior should include
focused tests and update the published result metadata when appropriate.
