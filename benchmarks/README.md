# Benchmarks

Public, reproducible measurements of folio. Two corpora pinned by commit, fetched
from GitHub, indexed through the `folio` command on PATH.

```sh
cargo install --path ..
llama-server -hf keisuke-miyako/gte-modernbert-base-gguf \
  --hf-file gte-modernbert-base-Q8_0.gguf \
  --embeddings --pooling cls -c 8192 -b 8192 -ub 8192 --port 8080
export FOLIO_ENDPOINT=http://127.0.0.1:8080/v1/embeddings

./run.py --model gte-modernbert
```

`reports/latest.md` and `reports/latest.json` are the output. Neither is
committed: a report belongs to the machine and the model that produced it, and a
committed one would go stale without saying so. A run worth keeping goes into
`../docs/measurements.md` under its own date, with the corpus commit, the model
and the machine named beside it.

## Three kinds of measurement, and only two of them are here

| Kind | Needs | Here |
|---|---|---|
| **Cost** — index time, query latency, index size, re-index after an edit | a large corpus pinned by commit | yes |
| **Contamination** — how much retired material a query surfaces | a corpus that marks its own lifecycle | yes, on MDN |
| **Ranking quality** — whether the ordering is good | queries with gold answers | **no** |

Ranking quality is absent on purpose. Measuring it needs a labelled query set,
and the two ways to get one are both wrong here. Writing questions against a
corpus you have already watched the tool answer is the leakage every retrieval
benchmark warns about — `../docs/measurements.md` carries seven such questions
and labels them as a smoke test rather than a result. Adopting BEIR or MTEB
instead would measure the embedding model: those are passage collections, while
folio's unit is a heading section carrying frontmatter, so the score would move
with the model and stand still with folio.

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
reached inside would measure an implementation rather than a tool.

Python is used because the contamination figure is a join between query output
and section records, and because the harness is not part of what folio ships.
