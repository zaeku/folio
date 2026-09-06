# 2026-09-05 — the storage layout

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
