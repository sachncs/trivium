---
layout: page
title: Contributing
subtitle: Set up the project locally, run the tests, and submit a pull request.
section: Community
permalink: /docs/contributing/
prev:
  title: API reference
  url: /docs/api/
next:
  title: Changelog
  url: /changelog/
---

Thanks for your interest in trivium. This page covers the
development setup, the test suite, and the PR workflow.

## Set up locally

```bash
git clone https://github.com/sachncs/trivium
cd trivium
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The editable install exposes the `trivium` package on your
`$PYTHONPATH`. Source edits take effect immediately.

## Run the tests

```bash
# Non-golden suite (no data cache required)
pytest -q

# Add coverage
pytest -q --cov=trivium --cov-fail-under=55

# Golden BM25 pre-flight suite (requires data/cache to be populated)
trivium-prepare --max-scales 5000
pytest -q -m golden
```

The golden test pins BM25 nDCG@10 on the 5K scifact seed to
`0.65685 ± 0.005` so a refactor that changes the tokenizer or
qrels prefix will fail the suite.

## Lint, format, type-check

```bash
ruff check .
ruff format --check .
mypy trivium
```

## Add a new pipeline

1. Decorate the new class with `@register_pipeline("your-slug")`.
2. Subclass `BenchmarkPipeline` and implement `name`, `encoders`,
   and `run`.
3. Compose retrievers from `trivium.retrieval`, fusion strategies
   from `trivium.fusion`, and rerankers from `trivium.reranking`.
4. Add a unit test in `tests/pipelines/`.

The runner discovers your mode automatically through the
registry. No CLI changes required.

## Add a new embedder

1. Decorate a factory with `@register_embedder("your-slug")`.
2. Use the `Sentence` backend for any sentence-transformers
   model, or subclass `Embedder` for custom backends.
3. Pin `model_id`, `dimension`, and (for BGE-style encoders) the
   prompt prefixes.
4. Add a unit test in `tests/embeddings/test_registry.py` that
   stubs the heavy backend via `sys.modules`.

## Add a new reranker

1. Decorate a factory with `@register_reranker("your-slug")`.
2. Use the `Encoder` backend for any CrossEncoder model, the `Bge`
   backend for BGE rerankers, or subclass `Reranker` for custom
   backends.
3. Add a unit test in `tests/reranking/test_registry.py` that
   stubs the heavy backend.

## The BM25 pre-flight invariant

Before submitting any change that touches BM25 retrieval, rerank,
or evaluation, run:

```bash
trivium-prepare --max-scales 5000
pytest -q -m golden
```

The golden test pins:

- BM25 nDCG@10 on the 5K scifact seed to `0.65685 ± 0.005`
  (drift band: `0.6789 ± 0.03`).
- BM25 Recall@10 to `0.7784 ± 0.005`.

If your change breaks the golden test, the diff almost certainly
introduced a tokenizer or qrels-prefix change. Investigate
before relaxing the test.

## Code style

- Python 3.12+ syntax (PEP 695, PEP 604 union types).
- Type hints on every public function.
- Module-level docstrings with the module's purpose.
- Class-level docstrings with the contract and usage.
- One blank line between top-level definitions.
- Imports grouped: stdlib, third-party, first-party; sorted
  alphabetically within each group.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(pipelines): add hybrid_colbert pipeline
fix(bm25): round trip doc_ids across save/load
docs: rewrite README hero, add LICENSE + CHANGELOG
ci: enforce real CI gate; drop || true on every step
refactor(retrieval): split Faiss.Flat and Faiss.Ivpq
test(reranking): cover the new bge-large factory
```

The prefix is one of `feat`, `fix`, `docs`, `ci`, `refactor`,
`test`, `chore`, `perf`. Add a scope in parentheses.

## Pull request flow

1. Fork the repository.
2. Create a topic branch off `main` (linear history).
3. Make focused commits with clear messages.
4. Ensure `pytest -q`, `ruff check .`, `ruff format --check .`,
   and `mypy trivium` all pass.
5. Use the [PR template](https://github.com/sachncs/trivium/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
6. Push the branch and open a PR targeting `main`.

By submitting a PR, you agree to follow the [Code of Conduct](https://github.com/sachncs/trivium/blob/main/CODE_OF_CONDUCT.md).

## Releases

Tags follow `vX.Y.Z`. Cutting a release:

```bash
git tag -a vX.Y.Z -m "DiskBBQ retriever + hybrid_bbq pipeline"
git push origin vX.Y.Z
gh release create vX.Y.Z --notes-file CHANGELOG.md
```

## Next steps

- Read [Changelog](../../changelog/) for what landed in each
  release.
- Read [License](../../license/) for the Apache-2.0 terms.
