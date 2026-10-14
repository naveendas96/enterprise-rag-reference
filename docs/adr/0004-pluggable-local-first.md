# ADR 0004: Local-first, pluggable models
**Decision:** `LLM_PROVIDER=mock|ollama|openai|anthropic`, `EMBEDDER=hash|st`, `VECTOR_STORE=numpy|chroma`. Defaults need no downloads or keys, so CI is deterministic and free.
**Consequences:** The default hashing embedder is lexical-ish. Use `EMBEDDER=st` (all-MiniLM-L6-v2) for real semantic recall.
