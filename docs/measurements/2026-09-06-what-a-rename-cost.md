# 2026-09-06 — what a rename cost before and after it kept its vectors

`mdn/content` at its pinned commit, 14,616 files and 119,565 sections,
`gte-modernbert-base-Q8_0`, `--max-chars 8000`. One file of seven sections was
moved from `web/api/window/afterprint_event/` to a sibling directory and
`folio index` was timed, first with the binary that pairs a departed path with
an arriving hash and then with the same binary with that pairing disabled.

| | records its path | delete and create |
|---|---|---|
| `folio index` | 0.33 s | 1.11 s |
| Bytes added to the matrix | 0 | 21,504 |
| Rows left dead | 0 | 7 |
| Sections re-embedded | 0 | 7 |

**The saving is the round trip, not the bytes.** Seven rows are 21 KB against a
367 MB matrix, and the dead ones would be reclaimed by a compaction that is
already rare. What a move avoids is asking the endpoint to embed text it has
already embedded: 0.78 s of the 1.11 s, for one file of seven sections. A
directory rename moves every file under it at once, and each one of them used to
pay that.

**Renaming is not free, only cheap.** 0.33 s is the walk — every file is still
listed and stamped — so a move costs what a re-index with nothing changed
costs, plus one lookup by hash and one path rewrite per moved file.
