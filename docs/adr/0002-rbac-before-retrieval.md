# ADR 0002: Enforce document ACLs before ranking, not after generation
**Status:** Accepted
**Context:** If a restricted chunk reaches the prompt, it can leak through paraphrase even when the citation is hidden.
**Decision:** Every chunk carries the `roles` from its document front matter. Both retrievers filter by role (a Chroma `where` filter or a numpy mask) before fusion. The eval counts `rbac_leaks` and CI fails on any leak.
**Consequences:** You need an index-time ACL model. In production, sync it from the IdP or SharePoint groups.
