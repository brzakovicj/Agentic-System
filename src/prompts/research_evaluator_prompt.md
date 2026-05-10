Evaluate whether the retrieved documents are sufficient for answering the query.

Query is:
'$query'

Retrieved documents is:
'$documents'

---

Based on the documents above, determine whether they are sufficient to answer the query.

Return a JSON object with a single field:
- **need_web_search** – `true` if the documents are insufficient or irrelevant, `false` if they adequately cover the query.

## Examples

| Situation | need_web_search |
|---|---|
| Documents directly answer the query | `false` |
| Documents are empty or off-topic | `true` |
| Documents are partially relevant but lack key details | `true` |
