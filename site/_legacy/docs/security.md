---
layout: page
title: Security
subtitle: How to report a vulnerability.
section: Community
permalink: /security/
---

trivium is a personal project maintained on a best-effort basis. The
following policy applies to all published versions.

## Supported versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |
| older   | :x:                |

Only the latest released version receives security fixes. Please
upgrade before reporting an issue.

## Reporting a vulnerability

**Please do not file a public issue.** Use GitHub's [private
vulnerability reporting][private-report] for this repository, or
email [sachncs@gmail.com](mailto:sachncs@gmail.com) directly. Both
channels reach the same maintainer.

Include:

- The trivium version (`pip show trivium`).
- A reproducer (commands, configuration, environment).
- The observed behaviour versus expected.
- Any relevant log output.

## Response SLA

- Acknowledgement within **3 business days**.
- Triage and severity assessment within **7 business days**.
- Fix timeline negotiated based on severity and exploitability.

Thank you for helping keep trivium and its users safe.

[private-report]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability
