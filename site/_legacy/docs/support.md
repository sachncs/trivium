---
layout: page
title: Support
subtitle: How to get help with trivium.
section: Community
permalink: /support/
---

trivium is a personal project maintained on a best-effort basis.

## Questions

For "how do I…" questions, see the [Documentation](../). The
[Architecture page](../docs/architecture/), [API reference](../docs/api/),
and [How-to guides](../docs/guides/) cover most patterns.

If the docs don't answer your question, open a [Discussion](https://github.com/sachncs/trivium/discussions) on GitHub.

## Bug reports

Open a [bug report](https://github.com/sachncs/trivium/issues/new?template=bug.yml).

Include:

- The trivium version (`pip show trivium`).
- Python, OS, faiss, torch, and sentence-transformers versions.
- The exact commands that reproduce the bug.
- The full output of the failing run.

If the bug is about nDCG or latency, attach
`results/benchmark.csv` and the snippet of `data/cache/manifest.json`.

## Security issues

See [Security](../security/).

## Maintainer availability

This is a personal project. There is no SLA on response time outside
of the security policy. Issues and discussions are answered on a
best-effort basis.

## Commercial support

This project does not offer commercial support. Custom work on top
of trivium is best addressed through a freelance engagement; reach
out via email for details.
