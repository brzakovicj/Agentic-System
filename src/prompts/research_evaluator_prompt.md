Evaluate whether the retrieved documents are sufficient for answering the query.

Query is:
$query

Retrieved documents is:
$documents

You MUST respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.
The JSON must have exactly these fields:
{
  "need_web_search": true or false,
  "reason": "your explanation here"
}