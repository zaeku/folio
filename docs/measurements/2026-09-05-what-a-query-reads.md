# 2026-09-05 — what a query reads, and the daemon it replaced

On `mdn/content`, 119,359 sections at dim 768, against the index the run in
[public corpora](2026-09-05-public-corpora.md)
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
