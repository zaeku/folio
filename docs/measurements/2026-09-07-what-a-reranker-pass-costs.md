# 2026-09-07 — what a reranker pass over the top k costs

A cross-encoder scores each `(query, section)` pair, so its cost is set by the
text in the top k rather than by k. That makes the distribution of section sizes
the first thing to report.

Sampled 3000 of the 119,565 sections in the `mdn/content` index — the index the
2026-09-06 column of
[the public benchmarks over dividing and merging](2026-09-06-the-public-benchmarks-over-dividing-and-merging.md)
describes, which is where the query figure below comes from:

| | Characters |
|---|---|
| Median | 199 |
| p75 | 562 |
| p90 | 1,210 |
| p99 | 3,512 |
| Largest sampled | 7,914 |

So folio's `--max-chars 8000` budget is not the size a reranker would usually be
handed. Four target lengths, real section text at each, four runs per cell with
the first discarded and the median of the remaining three:

| Reranker | Section size | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| jina-reranker-v1-tiny-en `Q8_0`, 33M | 199 | 8 ms | 15 ms | 24 ms |
| | 1,210 | 30 ms | 62 ms | 111 ms |
| | 3,512 | 110 ms | 235 ms | 463 ms |
| | 8,000 | 332 ms | 789 ms | 1,540 ms |
| bge-reranker-v2-m3 `Q8_0`, 568M | 199 | 102 ms | 195 ms | 377 ms |
| | 1,210 | 448 ms | 912 ms | 1,689 ms |
| | 3,512 | 1,452 ms | 3,176 ms | 6,146 ms |
| | 8,000 | 5,490 ms | 9,673 ms | 17,908 ms |

The query a pass would be added to is 135 ms median and 170 ms worst on this
corpus. One embedding round trip measured 10 ms in the same session, against the
8 ms recorded in [what a query reads](2026-09-05-what-a-query-reads.md).

**There is a cheap corner and it is not where an estimate put it.** The small
reranker over the top 20 at MDN's median section is 24 ms, about two embedding
round trips, and 18% of the query it would join. Before running this, 0.34 s for
a top-20 pass was estimated from the 0.53 s per 32 embedding inputs recorded on
2026-09-06. That is sixteen times too high at the median and about right at p90
to p99, because the estimate carried a per-input cost from a workload whose
inputs were whole sections at the budget.

**The size of the top k decides, not k.** For the small reranker at k=20, 199
characters per section is 24 ms and 8,000 is 1,540 ms: forty times the
characters for sixty-four times the time, slightly superlinear as attention
within a document should be. Doubling k roughly doubles a cell; going from the
median section to the budget multiplies it by sixty.

**That variance is the finding, and it is a property a query does not have
today.** The same reranker and the same k costs 24 ms or 1,540 ms depending on
which sections happened to rank, so a query's latency would move with the
content of its own answer. A folio query is flat in document size: it ranks
vectors and stats the files it returns without reading one.

**The large reranker has no cheap corner.** At the median section and k=20 it is
377 ms, already about three times the whole query, and 6.1 s at p99. Nothing in
the sizes this corpus actually produces makes it affordable.

**Method.** `llama-server` b10809-5266f24da with `--rerank`, `-c 8192 -b 8192
-ub 8192`, one process per reranker on ports 8083 and 8084:
`gpustack/jina-reranker-v1-tiny-en-GGUF` and `gpustack/bge-reranker-v2-m3-GGUF`,
both at `Q8_0`, which is the quantization folio's own endpoint runs. Requests to
`/v1/rerank` carried one query and k documents.

Documents were real section text drawn from the corpus, within 15% of each
target length, 20 per length, and the same 20 across every reranker and k.
Generated filler would have been wrong rather than merely artificial: repeated
filler tokenizes several times more cheaply than prose, which
[characters are not tokens](2026-09-05-characters-are-not-tokens.md) measured,
so a filler document of 8,000 characters is a smaller job than a real one.

**What this does not measure is whether any of it improves an answer.**
`D-01M1VNK3CJHMPR` declines to publish a ranking-quality score and gives its
reasons. Whether the exception it allows can even be pointed at a reranker is a
separate open question: a query drawn from a passage hands a cross-encoder an
exact substring of the text it is scoring, which inflates the arm under test
rather than both arms alike.
