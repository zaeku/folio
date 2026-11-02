# 2026-11-02 — what folio-sent prefixes change for EmbeddingGemma 2

`google/embeddinggemma-2` against `gte-modernbert-base`, both Q8_0 GGUF behind
`llama-server`, with folio itself sending the model card's retrieval prefixes:
`task: search result | query: ` before a query and `title: none | text: `
before a section, set through `FOLIO_QUERY_PREFIX` and `FOLIO_DOCUMENT_PREFIX`.
It relies on `2026-10-11-what-embeddinggemma-2-changes-on-rust-book.md`, whose
prefixed arms embedded outside folio.

**With its prefixes, EmbeddingGemma 2 through `folio query` beats gte.** The
merge ratio is still the 0.9 read off gte's scale, so this is the gain left
after the loss that measurement attributes to the merge rule.

| Arm | top 1 | top 5 | top 10 | top 20 | MRR |
|---|---|---|---|---|---|
| gte, through folio, no prefix | 94 | 128 | 139 | 144 | 0.750 |
| Gemma, through folio, no prefix | 67 | 119 | 133 | 147 | 0.607 |
| Gemma, through folio, prefixed | 101 | 142 | 146 | 146 | 0.799 |

Of 147 queries on the pinned `rust-lang/book`. The two unprefixed arms
reproduce the 2026-10-11 figures exactly. These describe this query set and the
difference between arms, not folio, for the reasons `D-01M1VNK3CJHMPR` gives
for the shape.

**Cost.** From `benchmarks/run.py`, both corpora, dim 768 for both models, the
Gemma run prefixed:

| | gte | Gemma |
|---|---|---|
| Index rust-book, 555 sections | 43.5 s | 82.0 s |
| Index mdn, 119,565 sections | 2076.0 s | 3106.8 s |

Gemma's query latency was 27 ms median on rust-book and 140 ms on mdn; gte's
report was overwritten before it was read, so no latency comparison is made
here.

**Method.**

- *Query set.* `benchmarks/queries/rust-book-seed55.json`, scored by
  `benchmarks/queries/score.py` with no prefix argument, since folio now adds
  the query prefix itself. Each arm indexed the corpus with `folio index
  --rebuild` first.
- *Serving.* gte: `llama-server` 0.8.0 build 11960, `--pooling cls -c 8192 -b
  8192 -ub 8192`. Gemma: the same binary, `-hf unsloth/embeddinggemma-2-GGUF:Q8_0
  --no-mmproj --pooling mean` with the same context and batch flags. 0.8.0
  loads the model without a source build.
