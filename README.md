# folio

Section-scoped semantic search over a markdown corpus. The index holds
references, never bodies.

A *folio* is a numbered leaf reference — a pointer to where text sits, not the
text. That is what this index stores: for every markdown heading section, a path,
a line range, the heading trail that names it, and the document's frontmatter.
Ask it a question and it ranks the sections you should read. Reading them is
your next step, and it reads the file, so an index that has fallen behind costs
you a wasted candidate rather than a wrong quotation — and where that would cost
more than a candidate, a query notices and catches the index up first.

## Why it exists

`grep` fails on the words you did not think of. Searching a corpus for
`independent` finds nothing when every document writes `independence`; searching
for `commit` returns six directories and buries the one that answers the
question among five plausible decoys. Picking wrong there is how an agent reads
the wrong document and misunderstands a project.

Semantic ranking fixes that much. What it does not fix — and makes worse — is
precedence. A superseded statement and the statement that replaced it are
semantically alike, so a vector index surfaces them side by side: in a two-line
fixture the retired definition of revenue ranks at 0.763 directly under the live
one at 0.842. Corpora that record their own lifecycle already carry the answer in
frontmatter, as a `status` or a `supersedes`. folio indexes those fields as
filters, so a query can say which of two similar passages still holds.

## Install

```sh
cargo install folio-cli
```

The crate is `folio-cli`, because the name `folio` is held on crates.io by a
placeholder. The command it installs is `folio`. To build from a clone instead,
run `cargo install --path .`.

folio calls an OpenAI-compatible embeddings endpoint and contains no inference
code, so the model is yours to choose. Any server exposing `/v1/embeddings`
works. One that has been measured:

```sh
llama-server -hf keisuke-miyako/gte-modernbert-base-gguf \
  --hf-file gte-modernbert-base-Q8_0.gguf \
  --embeddings --pooling cls -c 8192 -b 8192 -ub 8192 --port 8080
```

Two flags there are not optional. `--pooling cls` is what this model wants, and
the default is wrong for it — the Qwen3-Embedding family wants `--pooling last`
instead. And `-b`/`-ub` must be at least your longest section: an encoder needs
its whole input in one physical batch, so at the default 512 a longer section
comes back as an HTTP 500 rather than a truncated vector.

### Keeping the server up

Preparing a server to run a command-line tool is a strange shape for a CLI, and
folio does not manage one. It prints a service file for whatever manages
services on your machine, and installing it is yours to do:

```sh
folio unit > ~/Library/LaunchAgents/dev.folio.embeddings.plist
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/dev.folio.embeddings.plist
```

```sh
folio unit --systemd > ~/.config/systemd/user/folio-embeddings.service
systemctl --user enable --now folio-embeddings
```

`launchctl bootout gui/$UID/dev.folio.embeddings` stops it again. On macOS
before Ventura, `launchctl load` and `unload` are the pair to use instead.

`folio unit` writes nothing and starts nothing. It fills in the port your
configuration already points at, and the absolute path to `llama-server`,
because a service manager starts a job with a bare environment and will not find
an unqualified name. The model and its pooling arrive as flags:

```sh
folio unit --hf Qwen/Qwen3-Embedding-0.6B-GGUF \
  --hf-file Qwen3-Embedding-0.6B-Q8_0.gguf --pooling last
```

The service runs from login until you stop it. `llama-server` binds its own
socket rather than accepting one, so neither launchd nor systemd can start it on
demand, and it is not free while it waits.

What it holds is set by the longest section folio sends it, not by the model
file. Measured on the machine in `docs/measurements.md`: 69 MB once loaded,
330 MB after a section at the default 8,000-character budget, and 1,899 MB after
one of 8,000 tokens. It does not give any of that back until it restarts, so a
corpus of shorter sections is a smaller server as well as a more precise index.
The first query after several idle hours costs 231 ms against 9-10 ms warm,
because macOS has swapped most of it out by then.

If you already run an embeddings endpoint, ignore all of this and name it:
`folio config set endpoint http://your-host:port/v1/embeddings`.

## Use

```sh
folio config set endpoint http://127.0.0.1:8080/v1/embeddings

folio index                      # every .md under the working directory
folio query "does unfinished work count as a failure"
folio status
```

Only `folio index` needs to be told where the endpoint is. A query reads the
endpoint and model the index recorded, because a vector space belongs to one of
each and the recorded pair is the only correct answer for that corpus.

`--endpoint` beats `FOLIO_ENDPOINT`, which beats `folio.yaml` beside the corpus,
which beats the user's `~/.config/folio/config.yaml`, which beats
`http://127.0.0.1:8080/v1/embeddings`. `folio config` prints what won and where
each file is. Both are YAML with two keys, so editing one by hand is fine.

### A corpus that needs its own model

Every number here was measured on English. A corpus in another language wants a
model trained for it, and that choice belongs to the corpus rather than to the
machine indexing it:

```sh
folio config set --project model bge-m3
folio config set --project endpoint http://127.0.0.1:8081/v1/embeddings
```

Check the budget too. folio counts characters because it cannot see the model's
tokenizer, and the default 8,000 was set below an 8,192-token context on
English, which runs about 3.7 characters per token. Korean prose on the same
tokenizer runs 0.66, so 8,192 tokens is about 5,400 characters and the default
sends sections the server refuses. `folio doctor` asks that question with your
own longest section.

That writes `folio.yaml` at the corpus root. Commit it, and everyone who indexes
that corpus embeds it the same way. It is deliberately not inside `.folio/`: the
index there is derived and disposable, while which model a corpus needs is
neither.

Changing the model or the endpoint discards the index rather than mixing vector
spaces, and `folio index` says so before it re-embeds:

```
$ FOLIO_MODEL=other-model folio index
the index was built by bge-m3 at http://127.0.0.1:8081/v1/embeddings, and this
run uses other-model at http://127.0.0.1:8081/v1/embeddings — re-embedding
every section
```

Output names sections, with the heading trail beneath each:

```
#1  0.693  s21-does-an-incomplete-session-count-as-a-failure/README.md:6-6
        Does an incomplete session count as a failure?
```

### Checking the endpoint

Two ways a server disappoints folio are silent. A section longer than the
server's physical batch comes back as an HTTP error rather than a short vector,
and a pooling mode the model was not trained for returns vectors that rank badly
while looking like vectors.

```
$ folio doctor
  endpoint   http://127.0.0.1:8080/v1/embeddings  (user config)
  model      default
  reachable  yes, 768 dimensions, 43 ms for one input
  long input 8000 characters accepted, the longest section in ./docs/measurements.md, cut to the budget
  structure  paraphrase 0.900, unrelated 0.352 — ok
```

The batch question is asked with your own longest section, because characters
are not tokens: repeated filler tokenizes several times more cheaply than prose,
and a corpus that is not written in English packs more tokens into the same
characters. The structure question is asked with an English triple, so it says
less about a corpus in another language; it catches a space that is inverted or
collapsed, not one that is merely mediocre.

`folio doctor` exits 1 when a question fails, and prints the server's own
sentence with it.

Leave `--max-chars` alone unless you are testing a budget you mean to index
with. The server allocates for the largest input it is asked to embed and keeps
that allocation, so a probe larger than your budget makes it hold memory that
indexing would never have needed.

### Filtering on frontmatter

Frontmatter is flattened to dotted keys and stored as written — no field is
built in, so any producer's schema is queryable.

```sh
folio query "who owns a decision's status" --where status=live
folio query "the workspace layout"         --where supersedes
folio query "current guidance"             --where type!=deprecated
```

`key=value` matches, and reads as membership when the value is a list, so
`--where tags=alpha` works. `key` alone tests presence. `key!=value` also passes
when the key is absent, so a filter never silently drops the documents nobody has
annotated yet.

### Dropping what something else replaced

A `--where` predicate reads one record. Precedence does not live in one record:
the pointer sits on the successor, and the record you want gone is the one it
points at. That needs a join.

```sh
folio query "when is revenue recognised" --exclude-pointed-by supersedes
```

Any section whose `id` appears in any other section's `supersedes` stops being a
candidate, and the count of what went is printed so the drop is never silent.
Neither key is built in — `--exclude-pointed-by` names the pointer and
`--identity` names the key holding a record's own identity, which defaults to
`id` only because most schemas spell it that way.

The values are collected from the whole index rather than from what the other
filters leave, because a superseded record is superseded whether or not the
record that replaced it also answers this query.

Defaults belong to the query, not the index. The
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
reads an absent `status` as `stable`; folio stores what the file says and leaves
that reading to you.

### Keeping it current

`folio index` lists every file and re-embeds only the ones whose contents
changed. Listing is what makes it cheap — a file whose length and modification
time are what the index recorded is never opened — so on 14,616 files a re-index
with nothing changed is 0.27 s, and one changed file is 0.47 s.

There is no watcher and no daemon. A query checks itself instead. Before
returning a row it stats the file behind it, and if that file has moved since it
was indexed, folio brings the index up to date and answers again:

```
$ folio query "when is revenue recognised"
(1 of the files behind this result had changed; 1 file(s) re-embedded before answering)
#1  0.812  finance/revenue.md:42-57
        Recognition > Timing
```

This is the one place a stale index does harm rather than waste. A line range is
not a candidate; it is an instruction to read lines 40 to 55, and once two lines
are inserted above that section, following the instruction reads the wrong
lines.

It costs one stat per returned row, which is nothing: over 119,359 sections a
query with nothing changed is 0.12 s either way. It checks the rows it returns
and not the corpus — a file that changed without surfacing still costs you a
candidate, which is the trade this index makes everywhere else too. Walking the
whole tree to close that would cost 0.27 s on every query, twice what the query
costs.

`--no-refresh` turns off the writing, not the checking. A row whose file has
moved is still marked, because the caller who asked to be answered from the
index as it stands is the one who most needs to know where it does not:

```
$ folio query "when is revenue recognised" --no-refresh
#1  0.812  finance/revenue.md:40-55  (stale)
        Recognition > Timing
```

A refreshing query takes the same write lock `folio index` takes, and declines
rather than waits when another folio holds it, since that one is already
producing an index at least as fresh. It refreshes at most once: a row still
stale afterwards means the files are moving while folio reads them, and saying
so beats looping.

### From an agent

`SKILL.md` is the same surface written for an agent to act on: the commands, the
filters, and when to reach for `rg` instead. Point a skill-loading agent at it,
or copy it into wherever that agent keeps skills.

## What it does not do

- **Exact matching.** Use `rg`. It is exhaustive and folio is not, and a lexical
  route mixed into the ranking measurably buried correct answers.
- **Code structure.** folio indexes prose sections, not symbols or call graphs.
- **Anything but markdown.** PDF, office documents and source files are skipped.

## Sizing

One `f32` matrix, mapped and scanned end to end. No approximate index and no
recall parameter: a query over 548 sections is 15 ms and one over 119,359 is
132 ms, of which about 102 ms is the scan itself. The rest is one embedding
round trip and reading the list of rows that are still live.

Three quarters of a query is therefore the exhaustive arithmetic — which is the
part an approximate index replaces, and it would replace 102 ms with a graph to
build, a recall parameter to defend, and more bytes to load beside the matrix.

Headings buy two things, and the second one is memory: a corpus of shorter
sections asks the server for smaller batches, and the server sizes itself to the
largest batch it is asked for.

Pointer precision is set by your headings, not by the model. A file whose long
sections carry `###` subheadings returns 12-line ranges; the same content under
one `##` returns a 133-line range. If a result feels too coarse, add a heading
before you change models.

## License

MIT. See [LICENSE](LICENSE).
