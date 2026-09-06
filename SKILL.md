---
name: folio
description: >-
  Ranks markdown sections by meaning and returns `path:start-end` references to
  read. Use it to find a section when you do not know the words the corpus uses
  for it. Use it to rank sections filtered on frontmatter, such as a status or
  a supersedes. Do not use it for exact strings, for regexes, or for code
  symbols.
---

# folio

folio ranks markdown heading sections by meaning and names where they sit. It
returns references, never bodies: reading the range it names is your next step,
and it reads the file, so a result is a candidate rather than a quotation.

## Install

```sh
cargo install folio-cli
```

The crate is `folio-cli` and the command it installs is `folio`.

`folio index` checks its character budget against the endpoint before it embeds
anything, and lowers it for that run if the endpoint refuses the longest section.
A section past the budget is divided into consecutive records that each fit, so
a long section's tail is still reachable; a single line too long to divide is cut
and marked. So a corpus in a language that
packs more tokens into a character indexes without being told a number.

`folio index` writes the index to `.folio/` under the root it indexes. Add
`.folio/` to that corpus's ignore file before you index a repository. An index
is derived from the corpus and belongs to whoever built it, never to history.

## Before it can answer

folio contains no inference code. It calls an OpenAI-compatible embeddings
endpoint, which must be running:

```sh
folio config set endpoint http://127.0.0.1:8080/v1/embeddings
```

That writes the user's config. A corpus needing its own model — one in a
language the default model was not trained for — carries its own instead:

```sh
folio config set --project model bge-m3
```

`folio.yaml` at the corpus root is committed with the corpus, so everyone who
indexes it embeds it the same way. `FOLIO_ENDPOINT` and `--endpoint` override
both for one shell or one command. Run `folio config` to see which answered.

Only `folio index` reads any of them. A query uses the endpoint and model the
index recorded, so a corpus indexed against one server keeps answering from it.

Without a reachable endpoint, `index` and `query` fail. Nothing else does. If a
command reports it cannot reach the endpoint, say so — do not read that as an
empty corpus.

A query also fails when the endpoint answers with different weights than the
index was built on, naming a fingerprint below the threshold. The server was
restarted on another model. Report it and let the user decide; `folio index
--rebuild` is what fixes it, and it re-embeds the whole corpus.

## The commands

```sh
folio index                 # every .md under the working directory
folio query "does unfinished work count as a failure"
folio status                # files, sections, model, frontmatter keys carried
folio status --truncated    # the sections that were cut before embedding
folio config                # the endpoint and model a new index would use
folio doctor                # ask the endpoint what ranking depends on
```

Do not raise `--max-chars` to explore. The server allocates for the largest
input it is asked to embed and keeps that allocation until it restarts, so a
probe larger than the budget you index with leaves it holding memory for nothing.

Run `folio doctor` when results look wrong rather than absent. It reports
whether the endpoint answers, whether your longest section fits its batch, and
whether a paraphrase still ranks above an unrelated sentence. It exits 1 when
one of those fails.

`index` takes a positional root and `query` takes `--root`; both default to `.`.
The index lives in `.folio/` under that root. `query` prints five rows by
default, `-l N` for more, each a score, a `path:start-end`, and the heading
trail beneath it:

```
#1  0.693  s21-does-an-incomplete-session-count-as-a-failure/README.md:6-6
        Does an incomplete session count as a failure?
```

A row can cover several sections, and says so as `(N sections)`. Sections that
touch and rank alike are returned as the one range that covers them, because
that is the read they describe. A section beside a neighbour that ranks much
lower keeps its own narrower range.

Read those line ranges. A score is a ranking, not a verdict — the answer may be
in the third row, and it may be in none of them.

`--paths-only` prints one `path:start-end` per line and nothing else, at about a
quarter the size. Use it when you are feeding the ranges to a file reader rather
than judging them. A row whose file has moved is reported on stderr in that
mode, so read stderr too.

## Filtering on frontmatter

No key is built in, so the corpus's own schema is what you filter on. If you do
not know which keys the corpus carries, run `folio status` first.

```sh
folio query "who owns a decision's status" --where status=live
folio query "the workspace layout"         --where supersedes
folio query "current guidance"             --where type!=deprecated
```

`key=value` matches and reads as membership on a list. Bare `key` tests
presence. `key!=value` also passes when the key is absent, so a filter never
silently drops what nobody annotated.

When the pointer sits on the successor rather than on the record to drop, a
predicate cannot reach it and an anti-join can:

```sh
folio query "when is revenue recognised" --exclude-pointed-by supersedes
```

Any section whose `id` appears in another section's `supersedes` stops being a
candidate, and the count dropped is printed. `--identity` names the key holding
a record's own identity if the corpus does not spell it `id`.

## Staleness

A query stats the file behind each row before returning it, and re-indexes if
one has moved, because a shifted line range makes you read the wrong lines.
`--no-refresh` turns off the writing, not the checking. folio still marks a
moved row `(stale)`. Treat that row's range as approximate. Locate its heading
yourself before you read.

A query that refreshes declines rather than waits when another folio holds the
write lock, and refreshes at most once.

## What to use instead

| Question | Tool |
|---|---|
| An exact string, identifier, or regex | `rg` — exhaustive where folio is not |
| Symbols, callers, call graphs | CodeGraph |
| Anything not markdown | folio skips PDF, office documents and source files |
