# 2026-09-05 — public corpora

From `benchmarks/run.py`, which fetches a corpus pinned by commit and drives the
`folio` command on PATH. Reproducible by anyone: the corpus is public and the
harness is in the repository. `benchmarks/reports/` is scratch output and is not
committed; a run worth keeping is recorded here, dated, with what produced it.

`rust-lang/book` `src/` at `1500248d`, 112 files and 1.2 MB, embedded with
`gte-modernbert-base` Q8 through llama.cpp on Metal. This corpus carries no
frontmatter at all, so it measures heading splitting, ranking and cost with the
filtering path unused.

| Fact | Value |
|---|---|
| Sections | 548 from 112 files |
| Index | 38.7 s |
| Vectors | 1.7 MB at dim 768, exactly dim x 4 x sections |
| Section records | 0.12 MB `index.db` |
| Query | 15 ms median, 31 ms worst, over 400 queries at top 10 |
| Re-index after one changed file | 0.06 s, one file re-embedded |
| Matrix growth from that re-index | 6,144 bytes, 2 rows |

Two things the numbers say that the private corpora could not. Embedding cost
tracks tokens rather than files: 548 prose sections took 40 s where 116 short
records took 2.2 s, which is 3.8x the time per section for sections carrying
book paragraphs rather than one fact each. And the flat-matrix invariant that
`no-ann-index` asserts on a fixture holds on a real corpus, which the report
checks on every run rather than taking on faith.

`mdn/content` `files/en-us/` at `f4c14731`, 14,616 files and 59.6 MB, same model.
This corpus carries `status` as a list containing `deprecated`, so it measures
the filtering path and the contamination the filters exist to remove.

A first index writes the same bytes whichever way the matrix is stored. Four
runs took 1908 s, 1991 s, 1917 s and 1893 s, which is the spread of a 32-minute
embedding job on a laptop and not a difference any of the storage changes
made.

| Fact | Value |
|---|---|
| Sections | 119,359 from 14,616 files |
| Index | 1893 s |
| Vectors | 366.7 MB at dim 768, exactly dim x 4 x sections |
| Section records | 47.1 MB `index.db` |
| Query | 132 ms median, 179 ms worst, over 400 queries at top 10 |
| Re-index after one changed file | 0.47 s, one file re-embedded |
| Matrix growth from that re-index | 24,576 bytes, 8 rows |
| Deprecated sections | 4,256, or 3.57% of the corpus |
| Deprecated share of the top ten | 3.0% |
| Deprecated share once `--where status!=deprecated` is passed | 0.0% |

**Where a query's 132 ms goes.** The same query on `rust-lang/book` — 548
sections, same endpoint, same binary — is 15 ms, and that is everything that
does not scale: process start, the embedding round trip, and a scan too small
to see. The remaining 117 ms is what 119,359 sections cost, of which reading
the live row list is 10 ms and scanning the mapped matrix about 102 ms.

That decomposition is the strongest thing the corpus said about the no-ANN
decision, and it says it in the decision's favour. About three quarters of a
query is now the exhaustive scan itself, which is the part an approximate index
would replace — and it would replace 102 ms with a graph to build, a recall
parameter to defend, and bytes to load beside the matrix. What it removed was
never the arithmetic.

Reading records was 180 ms of this before the query stopped reading records it
does not decide on. `folio status` still pays it, and should: which frontmatter
keys the corpus carries is a question about every record.

**Checking a result against its files is free.** A query stats the file behind
each row it returns before returning it. With nothing changed that is 0.12 s,
which is what the query cost before the check existed. When one of the returned
files has moved, folio re-indexes and answers again: inserting two lines above
the top hit's section turned `40-55` into `42-57` at the same score in 0.75 s,
with no `folio index` run by anyone.

**Contamination is small and the filter is exact.** Deprecated material is
3.57% of the corpus and 3.0% of what the top ten returns over 400 queries. The
filter takes it to zero.

**Reading that figure took more queries than it was being measured with.** Four
runs at 40 queries gave 4.0%, 2.5%, 5.75% and — measured separately over the
same index — a 95% interval 9.5 points wide. The queries are fixed by seed and
the corpus is pinned by commit, so the run-to-run movement is the index: each
run re-embeds, llama.cpp on Metal does not reduce in a fixed order, and sections
a thousandth of a cosine apart change places at the top-ten boundary.

But 400 hits are not 400 draws. They arrive in 40 clusters, because one query
about a retired API contributes several deprecated hits at once, so a handful of
queries landing differently moves the whole figure. Resampling queries whole
against one unchanged index, on 2026-09-05:

| Queries | Estimate | 95% interval | Width |
|---|---|---|---|
| 40 | 5.75% | 1.75 – 11.25 | 9.50 pp |
| 80 | 5.50% | 2.38 – 9.25 | 6.88 pp |
| 160 | 4.50% | 2.25 – 7.12 | 4.88 pp |
| 240 | 4.54% | 2.75 – 6.54 | 3.79 pp |
| 400 | 4.20% | 2.80 – 5.80 | 3.00 pp |

It narrows as 1/√n: 40 to 400 is √10, and 9.50/3.00 is 3.17. `QUERY_COUNT` is
now 400, which costs about 200 s on top of a 1900 s index.

**What that bought is a statement, not a smaller number.** At 400 the interval
covers the corpus's own 3.57%, so retired material is not over-represented in
results — it is indistinguishable from proportional. No single run of 40 could
have said that, and the two 400-query figures measured so far, 4.20% and 3.0%,
agree with it and with each other.

The queries were drawn from section titles at random with a fixed seed and
without reference to `status`; a query set aimed at deprecated pages would have
produced a larger and meaningless number.

**Two runs failed before this one, each differently, and both are the reason
the character budget moved.** The first died with `input (8216 tokens) is too
large to process. increase the physical batch size`, which folio reported as if
the endpoint were absent, because it could not tell a server that answered from
one that was not there. The second, after that was fixed and the server's batch
raised, died with `input (8216 tokens) is larger than the max context size
(8192 tokens)` — the model's trained context, which `llama-server` caps `-c` at
and says so. No server flag raises it, so `--max-chars` came down to 8000 and
its default with it. A character budget cannot bound a token count in general;
that folio truncates on characters rather than chunking on a token estimate is
a gap this corpus found and the private ones could not, having never passed
1,100 tokens in a section.

**What a re-index actually costs, and what a vector database would not fix.**
Re-indexing MDN after one changed file took 2.74 s, of which walking the tree
was 822 ms, reading and hashing all 14,616 files 1294 ms, loading the index
350 ms and rewriting it 470 ms. Change detection was 77% of it and the index
write 17%, so an embedded vector database — which replaces the write and the
scan — addresses the smaller share.

Listing a file instead of reading it costs 33 ms against 1294 ms. `git status
--porcelain` on the same corpus takes 450 ms and `jj diff --name-only` 250 ms,
so a version control system is 7.6x to 14x slower than the stat it would
replace, and `git status` stats every file anyway before comparing its index.
With a stat prefilter in place, a single-file re-index is **1.24 s**.

Walking the tree in one thread took 449-804 ms depending on how warm the cache
was, and 172 ms across ten. With that threaded too, a re-index with nothing
changed is **0.895 s** and one changed file is **1.19 s**. The walk gained less
in place than on the bench, because in a real run it was already warm.

What is left of a re-index is loading the index at 350 ms and rewriting it at
470 ms — 69% of it — and the rewrite is the term that stays proportional to the
corpus rather than to the change. Both terms are gone; *the storage layout*
above is where they went.

**Where a query's time goes, in Rust.** Of the 350 ms to load the MDN index:
parsing 119,359 JSON records is 128 ms, reading the 367 MB matrix 76 ms,
converting those bytes to `f32` 76 ms, and splitting them into one `Vec` per row
80 ms. Mapping the matrix costs 0 ms and a full scan over the map with no
allocation costs 102 ms.

With the matrix mapped rather than read, a query over 119,359 sections is
**240 ms** against 446 ms, and `folio status` **150 ms** against 350 ms. What is
left of a query is reading the records, which is now its largest term and still
is: moving them into a database changed the way they are read and not how many
of them folio reads.
