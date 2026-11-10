# Changelog

## Unreleased

- Added a "Run it in 60 seconds" section to the README (install + two `erag ask` commands on the mock LLM). The documented command now uses `python3` rather than `python3.11`.
- Added GitHub issue and pull request templates.
- Added a test that the `/ask` API refuses a finance question when the `x-user-role` header is `employee`, and that the `admin` role can still retrieve the Q3 figure.
- Marked the synthetic compensation-band sentence in the demo corpus as a reviewed privacy-scan false positive. Those figures are fictional band ranges, not anyone's pay.

## 0.1.0 - 2026-10-14

Initial public release: hybrid BM25 + vector retrieval with RRF, document-level RBAC, two-gate grounded refusal, a FastAPI `/ask` endpoint, and a 17-case eval that runs on a mock LLM.
