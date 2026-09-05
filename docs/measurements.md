# Measurements

Every number folio reports about itself. `../AGENTS.md` requires a measured
fact to carry how it was measured and when, so this file is where one
goes; a number quoted anywhere else should point here.

**Environment.** Apple M1 Pro, 32 GB, macOS 26. Corpora as named in
[references.md](references.md).

**The retrieval scores below are a smoke test, not evidence.** Seven questions
were written against a private corpus after watching the tool answer, which is
the leakage the zvec-grep benchmark documentation warns about, and 7/7 against
6/7 is a difference of one question with no repeated trials and no variance
estimate. They are kept because they are what the model choice was made on, and
they are labelled so nobody quotes them as a result. `benchmarks/` is where a
reproducible number will come from.

## 2026-09-05 — what a query reads, and the daemon it replaced

On `mdn/content`, 119,359 sections at dim 768, against the index the run below
produced. Three warm runs each; the first run after an index is discarded.

A query was 267 ms: 180 ms reading records, 8 ms for the embedding round trip,
about 80 ms of scan. Splitting the 180 ms, measured through the `sqlite3` CLI
as an indicator of the read rather than folio's own path:

| Reading | Cost |
|---|---|
| The `slot` column alone | 10 ms |
| Every column, raw | 50 ms |
| Every column as folio deserialised them — a `String` per field and a JSON parse of `fm` and `breadcrumb` per row | 180 ms |

So a query that carries no filter needed 10 ms of that 180. It ranks on the
vectors, and the only record fact it needs before choosing a winner is which
rows are still live.

| | before | after |
|---|---|---|
| Query, no filter | 267 ms | **120 ms** |
| Query, `--where status!=deprecated` | 267 ms | **220 ms** |
| Re-index, nothing changed | 0.38 s | **0.27 s** |
| `folio status` | 0.18 s | 0.17 s |

`folio status` does not move, and should not: it reports which frontmatter keys
the corpus carries, which is a question about every record.

**A daemon was the alternative and the numbers rejected it.** A resident process
removes the reading and the paging, leaving 8 ms of embedding and about 80 ms of
scan: a floor near 88 ms, against the 120 ms above. Ten milliseconds, for which
it would have to be told when the index changed — a threaded stat walk of this
corpus is 172 ms, more than the reading it would replace — hold the database
open while `folio index` writes to it, which is the concurrent access the
default journal mode was chosen against, and keep 367 MB resident where the
mapping is page cache the kernel can reclaim. Its one real advantage is a cold
page cache, which is also what the page cache is for.

## 2026-09-05 — the storage layout

On `mdn/content`, the largest corpus folio is measured against: 14,616 files,
119,359 sections, dim 768. Measured against the index that run produced, by
re-running `folio index` over an unchanged corpus and then over one with a
single file edited. The vectors were carried across rather than re-embedded, so
these compare two storage layouts over the same numbers.

| Fact | Records in a JSONL file, matrix rewritten | Records in SQL, matrix appended |
|---|---|---|
| Re-index, nothing changed | 1.19 s | 0.38 s |
| Re-index, one file edited | 2.74 s | 0.48 s |
| Bytes written to the matrix per edit | 406 MB | 27 KB |
| Query | 240 ms | 260 ms |
| `folio status` | — | 0.18 s |
| Record store on disk | 39 MB JSONL + 2.6 MB state | 47 MB `index.db` |

The edit replaced eight sections and added one, and the matrix grew by exactly
nine rows: 27,648 bytes, which is 9 x 768 x 4. `benchmarks/run.py` makes the
same measurement on its own edit and reports it, so the figure is checked on
every run rather than taken once.

The record store on its own did not pay for itself. Measured in between, with
the records in SQL and the matrix still rewritten whole, an unchanged re-index
took 1.48 s — slower than the 1.19 s it replaced, because the index path still
read every record and every vector and now paid a database for the records too.
The two changes are one change: what the database bought was the ability to
change eight records without touching the rest, and that is only worth anything
once the matrix stops being rewritten beside them.

Query did not move. The database can return ten records without reading
119,349 others, but folio's query does not want ten: `no-ann-index` ranks every
section and `anti-join-reads-the-whole-index` collects pointers from every
record, so the query path reads the whole record set either way. Reading it out
of SQL costs about what parsing it out of JSONL cost. The 101 ms figure quoted
in `D-01M1QHKG8KGB4Q` is the cost of finding ten records among many, which is a
thing folio never does.

## 2026-09-05 — public corpora

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
| Index | 39.1 s |
| Vectors | 1.7 MB at dim 768, exactly dim x 4 x sections |
| Section records | 0.12 MB `index.db` |
| Query | 14 ms median, 24 ms worst, over 40 queries at top 10 |
| Re-index after one changed file | 0.05 s, one file re-embedded |
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

A first index writes the same bytes whichever way the matrix is stored. Three
runs took 1908 s, 1991 s and 1917 s, which is the spread of a 32-minute
embedding job on a laptop and not a difference any of the storage changes
made.

| Fact | Value |
|---|---|
| Sections | 119,359 from 14,616 files |
| Index | 1917 s |
| Vectors | 366.7 MB at dim 768, exactly dim x 4 x sections |
| Section records | 47.2 MB `index.db` |
| Query | 134 ms median, 156 ms worst, over 40 queries at top 10 |
| Re-index after one changed file | 0.48 s, one file re-embedded |
| Matrix growth from that re-index | 24,576 bytes, 8 rows |
| Deprecated sections | 4,256, or 3.57% of the corpus |
| Deprecated share of the top ten | 5.75%, and see below |
| Deprecated share once `--where status!=deprecated` is passed | 0.0% |

**Where a query's 134 ms goes.** The same query on `rust-lang/book` — 548
sections, same endpoint, same binary — is 14 ms, and that is everything that
does not scale: process start, the embedding round trip, and a scan too small
to see. The remaining 120 ms is what 119,359 sections cost, of which reading
the live row list is 10 ms and scanning the mapped matrix about 102 ms.

That decomposition is the strongest thing the corpus said about the no-ANN
decision, and it says it in the decision's favour. About 76% of a query is now
the exhaustive scan itself, which is the part an approximate index would
replace — and it would replace 102 ms with a graph to build, a recall parameter
to defend, and bytes to load beside the matrix. What it removed was never the
arithmetic.

Reading records was 180 ms of this before the query stopped reading records it
does not decide on. `folio status` still pays it, and should: which frontmatter
keys the corpus carries is a question about every record.

**Contamination is small, the filter is exact, and the unfiltered figure is
noisier than one number can show.** Three runs of the same 40 seeded queries
over the same pinned corpus returned 4.0%, 2.5% and 5.75%. The filtered figure
was 0.0% every time.

The spread is not the query path. Measured against one unchanged index, two
sweeps of those 40 queries returned byte-identical rankings, and a filter that
keeps every record returned the same rankings again — so ranking is
deterministic, and the two read paths agree over 400 real hits and not only
over a fence's fixture. What differs between runs is the index: each run
re-embeds the corpus, llama.cpp on Metal does not reduce in a fixed order, and
sections separated by a thousandth of a cosine change places at the top-ten
boundary.

It is also a small-sample estimator. 400 hits sounds like 400 draws but they
arrive in 40 clusters, because one query about a retired API contributes
several deprecated hits at once; two or three queries landing differently move
the figure by the whole spread above. **So the unfiltered number is a range,
2.5% to 5.75% over three runs, not the point value the report prints.** What
the report prints is one run, which is what a report is.

None of that touches the number the corpus is here for. Deprecated material is
3.57% of the corpus, it is not over-represented in results at any of the three
figures, and the filter took it to zero in all three. The queries were drawn
from section titles at random with a fixed seed and without reference to
`status`; a query set aimed at deprecated pages would have produced a larger
and meaningless number.

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

## 2026-09-04 — private corpora

Re-check a fact before building on it.

| Fact | Value |
|---|---|
| Best local embedding model tested | `gte-modernbert-base` Q8, 768 dim, 8192 tokens, CLS pooling, at 2.2 s to index against 7.2 s for `Qwen3-Embedding-0.6B`, a 768 dim index against 1024, and a 300 MB model against 610 MB. On the smoke test both placed the answering document in the top three for all seven questions; gte ranked six of them first and Qwen3 five, which is one question and not a result |
| Pooling is per model and the default is wrong for both | `--pooling cls` for gte-modernbert, `--pooling last` for the Qwen3-Embedding family |
| An encoder needs its whole input in one physical batch | `-b` and `-ub` must both be at least your longest section. gte-modernbert returned HTTP 500 on a 1106 token section at the default 512, with an explicit message. A decoder does not: `Qwen3-Embedding-0.6B` processed a 1068 token section at the same setting, chunking the prefill, so no earlier measurement here was silently truncated |
| GGUF conversion quality varies by publisher | `keisuke-miyako/gte-modernbert-base-gguf` loads. `cstr/gte-modernbert-base-GGUF` does not: `key not found in model: bert.context_length`. Test a conversion before trusting it |
| Gain from `Qwen3-Embedding-4B` over `0.6B` | None. Same 6/7 on the retrieval set, same rank on the hard query, for 6.7x the parameters and 2.5x the index |
| Why the 4B gains nothing | Its extra parameters are MLP capacity. 0.6B spends 26% of itself on the 151.7k vocab table, but only ~440M parameters do the retrieval work and those were trained for it |
| `gte-modernbert-base` runs on Metal, through llama.cpp | 2.2 s to index 116 files, against 8.8 s for the same server at `-ngl 0`: a 4.0x GPU speedup. Its ONNX q4 path is the thing that fails, on WebGPU (`MatMulNBits` dimension mismatch), falling back to CPU at 5m10s. The runtime was the variable, never the model |
| Result granularity | The heading section, always. Indexing the same corpus at context 8192, 2048 and 256 returned the identical range for the same query |
| What actually drives pointer precision | Section size, not model choice. In one corpus a file with 22 `###` headings returned 12–16 line ranges while a file with 2 returned a 133 line range |
| Long documents | Not a truncation risk. A 10,275 byte section reported zero truncated fragments even at context 256, because the indexer chunks before embedding |
| Brute-force cosine cost | 348 sections, microseconds. 35,000, ~10 ms. 100,000, 410 MB at f32 and ~30 ms |
| folio against zvec-grep, same corpus and model | 7.2 s against 19 s to index, and 45 ms to re-index one changed file. The one question zvec-grep placed outside the top three is the synonym bridge above; it is the observation the lexical-route decision rests on, and one question is not a score |
| `mlxcel` 0.6.0 | Serves no embeddings. `/v1/embeddings`, `/embeddings` and `/embedding` all return 404, and `mlxcel arch` lists no encoder family. Given an embedding checkpoint it loads it as a causal LM |

Two things are unmeasured. Do not assume either.

- Where retrieval quality starts to fall as a corpus grows. Every number above
  comes from corpora of 116 and 5 files.
- Whether a reranker over the top k earns its latency.

**The 2026-09-04 measurements have no committed artifact.** They were taken in a
session and written down here, which the rule permits and does not make
reproducible. They are the record of how the model and the design were chosen,
and they stay for that reason. Anything reported as a folio number from here on
comes from `benchmarks/run.py`.
