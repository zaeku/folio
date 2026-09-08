# 2026-09-08 — what re-embedding a subtree costs

A corpus may now be indexed one directory at a time, and a subtree of an indexed
corpus is covered twice. The question this answers is whether the second index
should embed those sections again or take them from the first.

`mdn/content`, pinned by the commit `benchmarks/run.py` uses, indexed at
`gte-modernbert` on a local `llama-server` at the default 8,000-character
budget. Each subtree was indexed from scratch into its own `.folio/`, with the
parent's index present above it.

| Subtree | Files | Sections | Wall clock | Index |
|---|---|---|---|---|
| `web/svg` | 300 | 2,177 | 43.8 s | 7.9 MB |
| `web/javascript` | 1,336 | 13,735 | 209.7 s | 47 MB |
| whole corpus, 2026-09-06 | 14,616 | 119,565 | 1,976 s | 396 MB |

**It is linear in sections, at 15 to 20 ms each.** 20.1 ms for `web/svg`, 15.3 ms
for `web/javascript`, 16.5 ms for the whole corpus. Files are the wrong unit:
these two subtrees differ by 4.5 times in files and 6.3 times in sections, and
`glossary` holds twice the files of `learn_web_development` in a sixth of the
text.

**Reading the same rows out of the parent costs a fraction of a second.** The
rows `web/svg` re-embedded in 43.8 s are 2,177 rows of the parent's record store,
which `sqlite3` returns whole in 0.15 s; the 13,735 rows behind `web/javascript`
come back in 0.26 s. Their vectors are a slice of the parent's matrix — 6.4 MB
and 40.2 MB — copied rather than computed. That is a floor rather than a plan: it
excludes proving the parent's fingerprint against the endpoint answering now,
matching content hashes, and every case where the two indexes do not correspond.

**So the ratio is about three hundred to one, and the absolute is under a
minute.** Re-embedding a plausible subtree costs seconds, once, at a moment the
caller chose by running the command. Borrowing from the parent instead would put
a second index's data path inside the write path, where a budget that differs
divides sections differently, a file the parent ignored has no row to borrow,
and the guarantee that an interrupted run keeps its finished files has to hold
throughout. folio embeds again.

**The number that does not fit that conclusion is the endpoint's time.** The 43.8
s and the 209.7 s are not folio's; they are a server holding the same model for
the whole run. Extrapolating by share of markdown bytes — 59.6 MB across the
corpus — `web/api` is about 750 s. Where a subtree is carried to a machine that
cannot run that model, or is kept past the point where the model can be run at
all, the cost of embedding again is not seconds but the model's availability,
and that is a different question from this one.
