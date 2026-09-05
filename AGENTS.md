# Agent Guidelines

Guidelines for an agent working on folio. It holds rules, verified facts, and the
boundaries this project defends. It is not a description of the product — read
`README.md` for that, and read it first if you have not.

**Terms.** *Section* means one markdown heading span, the unit folio indexes and
returns. *Reference* means the `path:start-end` triple plus frontmatter that a
section record holds; folio never stores the section's text. *Endpoint* means
the OpenAI-compatible `/v1/embeddings` service folio calls. *Corpus* means the
tree folio indexes.

This table decides where new code goes.

| Work | Belongs in | Why |
|---|---|---|
| Heading split, frontmatter flattening | `src/sections.rs` | The only non-trivial logic in the project, so it is where the tests live |
| Anything issuing SQL | `src/store.rs` | Every statement folio makes is there, so the engine behind it can be replaced without the rest knowing |
| CLI, ranking, filters, HTTP call | `src/main.rs` | Small enough that splitting it further would buy indirection, not clarity |
| Embedding inference | Outside the binary, behind the endpoint | Changing the model must not mean rebuilding folio |
| Reading a document's body | The caller, after folio names a range | The index holds references only |
| Exact string or regex matching | `rg`, not folio | Measured: a lexical route drowned the correct vector hit |
| Symbol and call-graph questions about code | CodeGraph, not folio | folio indexes prose sections, not symbols |

## Hard rules

**Date every measured fact you rely on.** Embedding models and their runtimes
turn over quickly. Record the number, how it was measured, and the date, in the
same place you use it. An undated number is a guess to the next reader.

## Decision layer

`decisions/` is a separate, unpublished repository. It holds every rule of this
project that a check can reach: each rule is a document paired with a fence, and
the fence fails when the rule is broken. That repository carries its own
`AGENTS.md`, so nothing about working in it is repeated here.

A rule a check can reach belongs there rather than in this file. What stays here
is what no check reaches: the hard rule above stays for good, because nothing
executable can test whether a number was written down honestly.

## Measurements

`docs/measurements.md` holds every number folio reports about itself, with how it
was measured and when. The rule on dating a measured fact governs what goes
there, and a number quoted anywhere else — `README.md` included — should be
traceable to an entry in it.

**A folio number comes from `benchmarks/run.py`.** It fetches two public corpora
pinned by commit and drives the `folio` command on `PATH`, so anyone can rerun
it. `benchmarks/README.md` says what those numbers are and, at more length, what
they are not: contamination is measured and ranking quality deliberately is not.
`.gitignore` excludes `benchmarks/reports/`, which is scratch. A run worth
keeping goes into `docs/measurements.md` under its own date.

The 2026-09-04 entries predate that harness and have no committed artifact. They
are the record of how the model and the design were chosen, and they say so.

Two things are still unmeasured and named in that file. Do not assume either.

## Toolchain

Nix declares the tools on this machine. Do not install a tool globally to make a
task work. Add it to the flake that needs it, so the next reader gets it.

**The code layer has no flake yet, and `llama-server` sits outside Nix.** It was
installed with `brew install llama.cpp` on 2026-09-04, and the rebuild of the
machine's Nix bridge failed and rolled back, so the binary at
`/opt/homebrew/bin/llama-server` is not in the declarative configuration and may
not survive the next rebuild. Write the flake when it declares something real:
`rustc`, `cargo`, a C compiler for the bundled SQLite, and whichever embedding
server the project settles on.

**The endpoint is a runtime dependency, not a build one.** folio compiles and its
tests pass with no server running. Only `index` and `query` need one:

```sh
llama-server -hf keisuke-miyako/gte-modernbert-base-gguf \
  --hf-file gte-modernbert-base-Q8_0.gguf \
  --embeddings --pooling cls -c 8192 -b 8192 -ub 8192 --port 8080
export FOLIO_ENDPOINT=http://127.0.0.1:8080/v1/embeddings
```

Any OpenAI-compatible embeddings endpoint substitutes for that one without a
change to folio. That is the whole reason the boundary is HTTP.

## Version control

**Jujutsu, colocated.** `.jj` and `.git` coexist. Measured on **jj 0.44.0**.

Read the [`use-jujutsu-safely`](https://github.com/zaeku/skills/tree/main/plugins/version-control/skills/use-jujutsu-safely)
skill before an unfamiliar command, or `jj help <command>` if you do not have
it. The rules that cost something here:

- **Record finished work with `jj commit -m`, not `jj describe`.** `jj commit`
  describes `@` and opens a new empty change, so your next edit lands somewhere
  new. `jj describe` leaves `@` described, and the next edit amends the change
  you just described. Use `jj describe -r <id>` to correct a description, or to
  describe a change that is not `@`.
- **Pass `-r` to `jj new` every time.** `@` is per workspace, and the default
  parent is wrong as soon as a second workspace exists.
- **`jj describe` on a change that already has a description replaces it
  silently.** Read `jj log` first. Resolve a revset to a commit id before you
  pass it to `-r`.
- **`jj util snapshot` when a session in a workspace ends.** It records the
  working copy and does nothing else; `jj status` also snapshots, but as a side
  effect of a command that may reset the working copy in the same breath.
- Do not discard existing changes. Existing changes belong to the user unless a
  task identifies them as agent changes.

## Language

Write documentation, project artifacts, code comments, and change descriptions
in English. Reply to the user in the language of their prompt.

## Reference paths

`docs/references.md` names the prior art this project was measured against, the
Open Knowledge Format specification behind the frontmatter rules, and the
corpora — the two public ones the benchmarks pin, and the two private ones the
design was chosen on. The private pair are Jujutsu working copies. Add an
ignore entry before you write an index into one.
