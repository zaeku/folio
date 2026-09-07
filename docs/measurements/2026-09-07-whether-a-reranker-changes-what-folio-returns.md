# 2026-09-07 — whether a reranker changes what folio returns

A cross-encoder rescoring folio's top k, against folio's own order, on the query
set
[whether a query set can be drawn from the corpus](2026-09-07-whether-a-query-set-can-be-drawn-from-the-corpus.md)
built: `mdn/content` anchor text pointing at a fragment, kept only where the
anchor is three words or more and is not a substring of the section it names, so
that no arm is handed a literal copy of the passage it scores.

**How often the question arises, before any answer.** 7,169 prose-anchored
fragment links resolved to 6,763 indexed sections; the two filters and
deduplication left 941 pairs across 487 target documents. Of those 941, the
named section was inside folio's top 20 for 565 (60%) — the only pairs a
reranker over the top 20 can affect at all — and 189 of the 565 (33%) were
already at rank 1, where nothing can improve.

**The arms.** One index, one embedding model, one corpus, one seed. The
reranker reorders the same 20 candidates folio returned, so coverage at 20 is
identical by construction and only the order inside it can move. Two arms differ
only in how much of each candidate is scored, which
[what a reranker pass over the top k costs](2026-09-07-what-a-reranker-pass-costs.md)
found to be the variable the cost follows.

| Arm | top 1 | top 5 | top 10 | Mean rank | MRR |
|---|---|---|---|---|---|
| folio alone | 33.5% | 72.4% | 88.7% | 4.52 | 0.505 |
| Reranked, candidates cut to 500 characters | 31.5% | 69.2% | 84.6% | 5.05 | 0.485 |
| Reranked, candidates cut to 2,000 characters | 29.4% | 73.6% | 88.8% | 4.37 | 0.483 |

Paired bootstrap over the 565 pairs, 10,000 resamples, against folio alone:

| Arm | top 1 | top 5 | top 10 | MRR |
|---|---|---|---|---|
| 500 characters | -1.9 [-6.4, +2.7] | -3.2 [-7.8, +1.4] | **-4.1 [-7.6, -0.5]** | -0.020 [-0.056, +0.015] |
| 2,000 characters | -4.1 [-8.7, +0.5] | +1.2 [-3.4, +5.7] | +0.2 [-3.2, +3.5] | -0.022 [-0.057, +0.013] |

**No arm improves anything, and the one interval that excludes zero is a
loss.** Cutting candidates to 500 characters costs 4.1 points of coverage at
top 10, and a sign test over the 422 pairs whose rank moved reads 190 up against
232 down (p = 0.041). At 2,000 characters every interval spans zero: 212 up
against 225 down (p = 0.534). Reordering is happening either way — three
quarters of the pairs move — and it is not moving the named section up.

**The two results together say why.** The cheap configuration is cheap because
it throws text away, and the text it throws away is where the match was. Raising
the cut to 2,000 characters recovers the loss and buys nothing: 67 ms rather
than 48 ms on top of a query that was 135 ms median on this corpus, for a
difference indistinguishable from none. So the configuration that is affordable
is the one that hurts, and the configuration that does not hurt does not help.

**What was not tested, and why.** One reranker, and a small one:
jina-reranker-v1-tiny-en at 33M parameters, `Q8_0`, English only. The 568M
bge-reranker-v2-m3 was not run over this set because the cost measurement had
already put it at 377 ms for 20 candidates at MDN's median section length,
about three times the whole query it would join, and 6.1 s at the 99th
percentile. So this measures the reranker a query could afford. A larger one is
untested rather than shown not to help, and it is untested on cost.

**The level in the first table is not folio's retrieval quality and may not be
quoted as one.** `D-01M1VNK3CJHMPR` admits this shape of measurement for a
difference between two configurations of folio and refuses a published score,
and the reasons apply here in full. The query set is drawn from one corpus's
link structure, it is filtered to exactly the pairs a literal match cannot
answer, and one link names one section while other sections of the same document
may answer as well without being credited. The 60% and the 33.5% describe this
set, not folio, and they exist so the difference beside them can be read.

**Method.** `folio query <anchor> -l 20 --paths-only --no-refresh`, one process
per query, from the corpus root, median 163 ms including process start against
the 135 ms the harness measures in-process. A candidate counts as the answer
when the range folio returned contains the named section's range, which is the
containment a reader following the reference would experience — folio merges
adjacent sections, so a returned row can cover several. Reranking called
`/v1/rerank` on `llama-server` b10809-5266f24da with `--rerank`, `-c 8192
-b 8192 -ub 8192`, one query and 20 documents per request, medians of 48 ms and
67 ms for the two cuts.
