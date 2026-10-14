# ADR 0003: Refuse when the answer is not grounded
**Status:** Accepted
**Decision:** There are two gates. (1) A pre-generation gate: if the best reranked chunk covers less than 50% of the query's content terms, refuse without calling the LLM. (2) A post-generation gate: an answer without a valid `[n]` citation is replaced with the refusal.
**Consequences:** Some recall is lost on heavily paraphrased questions, in exchange for zero confident wrong answers on the golden set. The threshold is configurable.
