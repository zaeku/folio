# Benchmarks

Public, reproducible measurements of folio. Two corpora pinned by commit, fetched
from GitHub, indexed through the `folio` command on PATH.

```sh
cargo install --path ..
llama-server -hf keisuke-miyako/gte-modernbert-base-gguf \
  --hf-file gte-modernbert-base-Q8_0.gguf \
  --embeddings --pooling cls -c 16384 -b 16384 -ub 16384 --port 8080
export FOLIO_ENDPOINT=http://127.0.0.1:8080/v1/embeddings

./run.py --model gte-modernbert
```

`reports/latest.md` and `reports/latest.json` are the output. Neither is
committed: a report belongs to the machine and the model that produced it, and a
committed one would go stale without saying so. A run worth keeping goes into
`../docs/measurements/` under its own date, with the corpus commit, the model
and the machine named beside it.

## The model's context is the limit, and no flag raises it

`run.py` sets `--max-chars 8000`, below the 8192-token context of the model
above. That budget is on folio's side because the limit is the model's own:
llama-server caps `-c` at the trained context and says so —

```
the slot context (16384) exceeds the training context of the model (8192) - capping
```

A character budget cannot bound a token count. A byte-level tokenizer can spend
more than one token on a multi-byte character, and MDN has a section that
reached 8216 tokens inside 12000 characters where the private corpora never
passed 1100. 8000 characters is far under the limit for this corpus at its
observed density, and it is not a proof. Chunking a long section on a token
estimate rather than truncating it on a character count is the fix folio does
not have yet; `zvec-grep` solves the same problem with a density heuristic.

Two MDN runs died on this before the budget moved, each with a different
message. Both are in `../docs/measurements/`, because a limit discovered by
hitting it is worth writing down where the next reader will look.

## Three kinds of measurement, and only two of them are here

| Kind | Needs | Here |
|---|---|---|
| **Cost** — index time, query latency, index size, re-index after an edit and what that edit writes | a large corpus pinned by commit | yes |
| **Contamination** — how much retired material a query surfaces | a corpus that marks its own lifecycle | yes, on MDN |
| **Ranking quality** — whether the ordering is good | queries with gold answers | **no** |
| **Retrievability** — whether a section can be found at all | queries drawn from the corpus itself | only to compare folio with folio, see below |

Ranking quality is absent on purpose. Measuring it needs a labelled query set,
and the two ways to get one are both wrong here. Writing questions against a
corpus you have already watched the tool answer is the leakage every retrieval
benchmark warns about — `../docs/measurements/` carries seven such questions
and labels them as a smoke test rather than a result. Adopting BEIR or MTEB
instead would measure the embedding model: those are passage collections, while
folio's unit is a heading section carrying frontmatter, so the score would move
with the model and stand still with folio.

## The one quality-shaped question that is allowed

A change to how folio cuts or splits a section cannot be judged by cost or by
contamination, and refusing to measure it means choosing between prefix and
split by argument. So one narrow measurement is admissible, and its boundaries
are what keep it from becoming the thing refused above.

**It is retrievability, not preference.** Take the text truncation removes — the
tail of a section past the budget. Draw a query from it. The section it came
from is the answer by construction, so nothing is annotated by anyone who has
watched the tool run. Measure the share of those queries whose source section is
returned in the top k, under one folio and then another.

**It compares folio to folio and nothing else.** One model, one corpus, one
seed, two configurations. The absolute figure is inflated because a query drawn
from a passage shares its wording, and that inflation is identical in both arms,
which is why the difference is readable and the level is not. It may never be
quoted as folio's ranking quality, and never compared across models: that is the
number this file refuses, and drawing queries from the corpus does not make it
available.

**It reports how often the question arises, first.** Measured 2026-09-06 on the
pinned corpora: 142 of MDN's 119,359 sections are cut, and 6 of the 548 in
`rust-lang/book`. Within the cut sections 43.5% of the text is past the budget
and unreachable by any query — 873,797 characters, in pages like
`web/html/reference/attributes/index.md`, which is a reference list whose tail
is more entries. A rate of 0.12% is small; a page whose second half cannot be
found is not, and both halves of that sentence belong in the report.

`folio status` prints the cut count and `folio status --truncated` names them,
so the harness reports it without measuring anything itself.

## Contamination is the number that measures folio

A vector index ranks a superseded statement and the statement that replaced it
side by side, because they are semantically alike. Everything folio adds beyond
ranking exists to separate them. So the number worth publishing is not a
preference score but a rate:

> what share of the top ten results carry `status: deprecated`, with the filter
> and without it

It needs no annotation, it is objective, and it moves with folio rather than
with the model. The queries are drawn from the titles of randomly sampled
sections with a fixed seed, deliberately without reference to `status`: a query
set chosen from deprecated pages would guarantee contamination instead of
estimating it.

The filtered figure should be zero. It is reported rather than asserted, because
a filter that quietly stopped working would otherwise look like a corpus that
had cleaned itself up.

**The unfiltered figure needed 400 queries to mean anything.** At 40 its 95%
interval was 9.5 points wide around a value near 4%, and four runs of the same
seeded queries over the same pinned corpus returned 4.0%, 2.5%, 5.75% and 3.0%.
400 hits are not 400 draws: they arrive in clusters, one per query, because a
query about a retired API contributes several deprecated hits at once. The
interval narrows as 1/sqrt(n), so `QUERY_COUNT` is 400 and the interval is 3.0
points, which is finally narrower than the thing being measured.
`../docs/measurements/` carries the table. The filtered figure was zero in
every run, and that is the one the filter is judged on.

## The two corpora, and why both

Counted over the subpath each one is measured on, not the whole repository.

| | `rust-lang/book` `src/` | `mdn/content` `files/en-us/` |
|---|---|---|
| Markdown files | 112 | 14,616 |
| Bytes | 1.2 MB | 59.6 MB |
| Frontmatter | **none** | `title`, `slug`, `page-type`, `status`, `browser-compat` |
| Exercises | heading sections alone | the frontmatter path, including a lifecycle field |

They are a contrast, not a pair of samples. `rust-lang/book` has no frontmatter
whatsoever, so it isolates the part of folio that has nothing to do with
filtering: heading splitting, ranking, and cost. `mdn/content` is 49x its size
and carries `status` as a scalar list containing `deprecated`, which is both the
shape folio's list-membership filter was built for and the only public corpus
found that marks retirement in place rather than deleting it.

Neither is the corpus folio was developed against. Those two are private and
named in `../docs/references.md`, and one of them turned out not to need the
anti-join at all because the project it comes from removes a decision rather
than marking it — which is why a public corpus that keeps retired material was
worth finding.

## Pinning

The commits in `run.py` are fixed so that two runs measure the same bytes. Bump
them deliberately, in a change that says why, and expect the numbers to move:
MDN grows, and a corpus that grew is not the corpus the last report measured.

## What the harness may not do

It reads no folio source and imports nothing from the crate. It sees the `folio`
command, its stdout, and the files it writes under `.folio/` — the same surface
the decision layer's fences are held to, for the same reason. A harness that
reached inside would measure an implementation rather than a tool. Reading the
section records means reading `.folio/index.db` with `sqlite3`, which is on that
surface: it is a file folio writes, in a format it did not invent.

Python is used because the contamination figure is a join between query output
and section records, and because the harness is not part of what folio ships.
