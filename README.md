# folio

Section-scoped semantic search over a markdown corpus. The index holds
references, never bodies.

A *folio* is a numbered leaf reference — a pointer to where text sits, not the
text. That is what this index stores: for every markdown heading section, a path,
a line range, the heading trail that names it, and the document's frontmatter.
Ask it a question and it ranks the sections you should read. Reading them is
your next step, and it reads the file, so an index that has fallen behind costs
you a wasted candidate rather than a wrong quotation.

## Why it exists

`grep` fails on the words you did not think of. Searching a corpus for
`independent` finds nothing when every document writes `independence`; searching
for `commit` returns six directories and buries the one that answers the
question among five plausible decoys. Picking wrong there is how an agent reads
the wrong document and misunderstands a project.

Semantic ranking fixes that much. What it does not fix — and makes worse — is
precedence. A superseded statement and the statement that replaced it are
semantically alike, so a vector index surfaces them side by side. Corpora that
record their own lifecycle already carry the answer in frontmatter, as a `status`
or a `supersedes`. folio indexes those fields as filters, so a query can say
which of two similar passages still holds.

## Install

```sh
cargo install --path .
```

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

## Use

```sh
export FOLIO_ENDPOINT=http://127.0.0.1:8080/v1/embeddings

folio index                      # every .md under the working directory
folio query "does unfinished work count as a failure"
folio status
```

Output names sections, with the heading trail beneath each:

```
#1  0.693  s21-does-an-incomplete-session-count-as-a-failure/README.md:6-6
        Does an incomplete session count as a failure?
```

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

Defaults belong to the query, not the index. The
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
reads an absent `status` as `stable`; folio stores what the file says and leaves
that reading to you.

### Keeping it current

`folio index` hashes every file and re-embeds only the ones that changed. There
is no watcher and no background daemon, because the files are the truth and
hashing them is cheap: re-indexing one changed file out of 116 takes 45 ms.

## What it does not do

- **Exact matching.** Use `rg`. It is exhaustive and folio is not, and a lexical
  route mixed into the ranking measurably buried correct answers.
- **Code structure.** folio indexes prose sections, not symbols or call graphs.
- **Anything but markdown.** PDF, office documents and source files are skipped.

## Sizing

One `f32` matrix, scanned end to end. No approximate index, no recall parameter:
348 sections rank in microseconds, 35,000 in about 10 ms, 100,000 in about 30 ms
over 410 MB.

Pointer precision is set by your headings, not by the model. A file whose long
sections carry `###` subheadings returns 12-line ranges; the same content under
one `##` returns a 133-line range. If a result feels too coarse, add a heading
before you change models.

## License

MIT
