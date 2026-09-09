## Summary

<!-- One-paragraph description of what this PR changes and why. -->

## Linked issues

<!-- `Fixes #123`, `Closes #456`, etc. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Pre-flight

- [ ] If your change touches BM25 retrieval, the BM25 pre-flight
      golden test still passes:
      `pytest -q -m golden`
- [ ] The coverage gate is still met:
      `pytest -q --cov=trivium --cov-fail-under=55 -m "not golden"`
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)

## Tests

- [ ] I have added tests for my change (or explained why none are needed)
- [ ] I have updated existing tests if behaviour shifted
- [ ] All new and existing tests pass locally

## Documentation

- [ ] I have updated the docs (or explained why no doc change is needed)
- [ ] If you added a new pipeline / embedder / reranker slug, the
      public-facing docs mention it

## Dependencies

- [ ] The change does not introduce a new dependency without justification
