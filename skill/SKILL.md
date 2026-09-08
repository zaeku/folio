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
returns references, never bodies. Reading the range it names is your next step,
and you read the file itself. So what folio names is a candidate rather than a
quotation.

folio explains its own refusals — a filter it cannot express, a predicate that
emptied a result, an index whose weights no longer match. What is here is what it
cannot tell you at the moment you need it.

## What to use instead

| Question | Tool |
|---|---|
| An exact string, identifier, or regex | `rg` — exhaustive where folio is not |
| Symbols, callers, call graphs | CodeGraph |
| Anything not markdown | folio skips PDF, office documents and source files |

## The commands

```sh
folio index                 # every .md under the working directory
folio query "does unfinished work count as a failure"
folio status                # files, sections, model, frontmatter keys carried
folio config                # the configuration a new index would use
folio doctor                # ask the endpoint what ranking depends on
```

`index` takes a positional root and defaults to `.`. It reads the nearest
`folio.yaml` at or above that root, so a subtree embeds the way its corpus
declared; it says so when the root is already inside another index, and
`folio status` prints `declared` beside `model` when the two disagree.
`query`, `status` and `doctor` default to the corpus around you: folio walks
upward from the working directory to the first `.folio/` or `folio.yaml`, so a
question can be asked from anywhere inside a corpus without moving first. A
reference then prints a path that opens from where you are.
The index lives in `.folio/` under that root. **Add `.folio/` to that root's
ignore file before you index a repository.** folio will not do it for you. An
index is derived from the corpus and belongs to whoever built it, never to
history.

`--root` is repeatable, and folio ranks several indexes as one answer, each
reference naming the root it came from. folio returns a section two of them hold
once. It refuses by name any root it cannot show belongs with the others, so a
query that answers has answered from all of them.

Run `folio doctor` when results look wrong rather than absent. It exits 1 when
the endpoint fails something ranking depends on, and names which.

## Reading a result

```
#1  0.693  s21-does-an-incomplete-session-count-as-a-failure/README.md:6-6
        Does an incomplete session count as a failure?
```

A query prints five rows by default, or `-l N` for more. Each row is a score, a
`path:start-end`, and the heading trail beneath it.

**Read those line ranges.** A score orders candidates rather than judging them —
the answer may be in the third row, and it may be in none of them.

folio marks a row `(N sections)` when it covers several sections that touch and
rank alike, because that is the one read they describe.

A row marked `(stale)` sits in a file that moved after folio indexed it. Treat
its range as approximate. Locate its heading yourself before reading.

`--paths-only` prints one `path:start-end` per line and nothing else. Use it
when you are feeding ranges to a file reader rather than judging them. folio
reports a stale row on stderr in that mode, so read stderr too.

## The endpoint

folio contains no inference code and calls an OpenAI-compatible embeddings
endpoint, which must be running:

```sh
folio config set endpoint http://127.0.0.1:8080/v1/embeddings
```

Run `folio config` to see where that answer came from. A corpus needing its own
model — one in a language the default was not trained for — carries `folio.yaml`
at its root, committed with it, so everyone indexes it the same way. For an
endpoint that needs an API key, run `folio config set api_key <token>` to store
it in user config, or export `FOLIO_API_KEY` (or `OPENAI_API_KEY` for
`api.openai.com`). Credentials never live in `folio.yaml`.

**An unreachable endpoint is not an empty corpus.** Only `index` and `query`
need one, and both say so plainly when it is missing. Report what folio said
rather than the absence of results.

folio also refuses a query when the endpoint answers with different weights than
the index was built on, naming what it measured. The server was restarted on
another model. `folio index --rebuild` is the fix and it re-embeds everything, so
report it and let the user decide.

## Filtering on frontmatter

No key is built in, so you filter on the corpus's own schema. Run `folio status`
first if you do not know which keys it carries.

```sh
folio query "who owns a decision's status" --where status=live
folio query "the workspace layout"         --where supersedes
folio query "current guidance"             --where type!=deprecated
```

| Predicate | Passes when |
|---|---|
| `key=value` | the value matches, or is a member of a list |
| `key` | the key is present |
| `!key` | the key is absent |
| `key!=value` | the value differs, **and also when the key is absent** |
| `a\|b` | either side does, inside one argument |
| repeated `--where` | every flag does |

`key!=value` passing on absence is the one to remember: a filter never silently
drops what nobody annotated, and `!key` is how you ask for those on purpose.

folio refuses a filter it cannot express instead of guessing. It names the
predicate that emptied a result, so an empty answer means nothing matched.

When the pointer sits on the successor rather than on the record to drop, a
predicate cannot reach it and an anti-join can:

```sh
folio query "when is revenue recognised" --exclude-pointed-by supersedes
```

Any section whose `id` appears in another section's `supersedes` stops being a
candidate, and folio prints the count it dropped. `--identity` names the key holding
a record's own identity if the corpus does not spell it `id`.
