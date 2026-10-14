# ADR 0001: Hybrid BM25 + vector retrieval fused with RRF
**Status:** Accepted
**Context:** Enterprise questions mix exact terms (band codes, policy numbers) and paraphrases. Pure vector search misses exact tokens; pure BM25 misses paraphrases.
**Decision:** Run both and fuse with Reciprocal Rank Fusion (k=60), then rerank the top 12.
**Consequences:** RRF needs no score calibration between retrievers. It costs two lookups per query, which is negligible at this scale.
