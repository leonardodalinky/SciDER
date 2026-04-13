# Semantic Scholar API Cookbook

How to verify a candidate paper via the Semantic Scholar Graph API.

Base: `https://api.semanticscholar.org/graph/v1`

Reference: <https://api.semanticscholar.org/api-docs/graph>

## API key (optional)

The pipeline uses the **public, unauthenticated endpoint** by default — no key
required.  If you have a Semantic Scholar API key you can pass it via the
`x-api-key` header to get higher rate limits (useful for large batches).

Get a free key at <https://api.semanticscholar.org/> then export it once:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"
```

The bundled `scripts/s2_search.py` helper picks this up automatically.  If the
variable is not set the script falls back to the unauthenticated endpoint — the
pipeline works fine either way; just keep to ≤1 QPS on live requests.

```bash
# check whether the key is configured
python skills/literature-review-agent/scripts/s2_search.py --check-key

# search by title (key used automatically if set)
python skills/literature-review-agent/scripts/s2_search.py \
    --query "Attention is All You Need" --limit 5

# print the raw S2 JSON
python skills/literature-review-agent/scripts/s2_search.py \
    --query "BERT pre-training" --raw
```

The repo never commits a key.  Key management is your responsibility (shell
environment, 1Password, doppler, etc.).

## Endpoint 1 — Search by title

```
GET /paper/search
    ?query=<URL-encoded title>
    &limit=5
    &fields=title,abstract,year,authors,venue,externalIds
```

Example:

```
GET https://api.semanticscholar.org/graph/v1/paper/search?query=Attention%20Is%20All%20You%20Need&limit=5&fields=title,abstract,year,authors,venue,externalIds
```

Response (truncated):

```json
{
  "total": 12345,
  "data": [
    {
      "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
      "title": "Attention is All you Need",
      "abstract": "The dominant sequence transduction models are based on...",
      "year": 2017,
      "venue": "NeurIPS",
      "authors": [{"name": "Ashish Vaswani"}, ...],
      "externalIds": {
        "DBLP": "conf/nips/VaswaniSPUJGKP17",
        "ArXiv": "1706.03762",
        "DOI": "10.5555/3295222.3295349"
      }
    },
    ...
  ]
}
```

## Endpoint 2 — Get a specific paper by ID

```
GET /paper/<paperId>?fields=title,abstract,year,authors,venue,externalIds,citationCount
```

## Useful identifiers

You can pass these as `<paperId>`:

- S2 internal: `204e3073870fae3d05bcbc2f6a8e263d9b72e776`
- DOI: `DOI:10.18653/v1/N18-3011`
- ArXiv: `ARXIV:1706.03762`
- Corpus ID: `CorpusId:13756489`
- URL: `URL:https://arxiv.org/abs/1706.03762`

## Rate limits

- Unauthenticated: ~1 QPS sustained. Bursts will get 429.
- Per the paper, "the strict throughput limits of the Semantic Scholar API
  (1 query per second)" — App. B.

If you get HTTP 429, sleep 5 seconds before retrying. Don't loop tightly.

## Fields cheat sheet

| Field | Type | Required by our pipeline? |
|---|---|---|
| `paperId` | string | yes (dedup key) |
| `title` | string | yes (Levenshtein match) |
| `abstract` | string | yes (rule 2: must exist) |
| `year` | int | yes (cutoff check) |
| `authors[].name` | string | yes (BibTeX author field) |
| `venue` | string | recommended (BibTeX journal/booktitle) |
| `externalIds.DOI` | string | recommended (dedup fallback, BibTeX doi) |
| `externalIds.ArXiv` | string | recommended (dedup fallback) |
| `publicationDate` | string `YYYY-MM-DD` | optional (more precise cutoff check) |
| `citationCount` | int | optional (could inform tie-breaking) |

Always pass `fields=...` explicitly — the default response is minimal and
will not include the abstract.

## Error handling

| Status | Meaning | What to do |
|---|---|---|
| 200 | OK | proceed |
| 400 | bad query syntax | URL-encode the title properly; retry once |
| 404 | not found | discard the candidate |
| 429 | rate limited | sleep 5s, retry |
| 500-503 | S2 down | sleep 30s, retry up to 3 times, then give up |

## Polite use

The S2 API is a public service. Do not hammer it. If you have many candidates:

- Throttle to 1 QPS.
- Cache hits (the dedup script already serves as a deduplication cache).
- Do not parallelize. Verification is sequential by design.
