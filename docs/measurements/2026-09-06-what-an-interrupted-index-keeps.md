# 2026-09-06 — what an interrupted index keeps

Before this, `reindex` embedded every section into memory and called `append`
and the record write once each, at the end. Measured the same day by accident: a
33-minute rebuild of `mdn/content` was killed partway and `vectors.f32` came
back 0 bytes. An index now commits in batches of 512 sections rounded up to a
file boundary.

**A killed run keeps its finished files.** On a fixture of 300 files and 1,800
sections, `folio index` was killed after five seconds:

| | |
|---|---|
| Files stamped | 86 of 300 |
| Records | 516 |
| Rows in the matrix | 516 — exactly the records, no dead rows |
| Dimension recorded | 768 |

The next run reported **214 re-embedded**, took the matrix to 1,800 rows, and
left no dead rows: not one file was embedded twice. A killed `--rebuild`
behaves the same way.

**The batch is whole files because the stamp is what resumes.** A stamp says the
index is current for its file, so writing one for a file whose sections are only
half committed would make the next run skip the other half for good. `fresh` is
already built one file at a time, so batching only chooses where to end.

**A dimension and a fingerprint have to be recorded from the first batch, not
the last.** Written at the end, a killed first index leaves `dim` at 0, the next
run reads the rows as belonging to no vector space, and discards every one of
them. That was measured before it was fixed: a run that had kept 516 rows threw
them away and re-embedded all 1,800. The fingerprint vector's length is the
dimension, so one request settles both before anything commits.

**It costs nothing on the paths that were already fast.** On `mdn/content`,
14,616 files: a re-index with nothing changed is 0.30 s against the 0.29 s it
was, and one changed file is 0.44 s against 0.47 s. Both are inside the noise.

**What a kill costs now** is the batch in flight: 512 sections, about eight
seconds of embedding at the 0.53 s per 32 inputs that
[the public benchmarks over dividing and merging](2026-09-06-the-public-benchmarks-over-dividing-and-merging.md)
works out to, 1,976 s over 3,737 requests.
