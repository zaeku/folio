# Agent Guidelines

Guidelines for an agent working on folio. It holds rules, verified facts, and the
boundaries this project defends. It is not a description of the product — read
`README.md` for that, and read it first if you have not.

**Terms.** *Section* means one markdown heading span, the unit folio indexes and
returns. *Reference* means the `path:start-end` triple plus frontmatter that a
section record holds; the section's text is never stored. *Endpoint* means the
OpenAI-compatible `/v1/embeddings` service folio calls. *Corpus* means the tree
folio indexes.

This table decides where new code goes.

| Work | Belongs in | Why |
|---|---|---|
| Heading split, frontmatter flattening | `src/sections.rs` | The only non-trivial logic in the project, so it is where the tests live |
| CLI, store format, filters, HTTP call | `src/main.rs` | Small enough that splitting it would buy indirection, not clarity |
| Embedding inference | Outside the binary, behind the endpoint | Changing the model must not mean rebuilding folio |
| Reading a document's body | The caller, after folio names a range | The index holds references only |
| Exact string or regex matching | `rg`, not folio | Measured: a lexical route drowned the correct vector hit |
| Symbol and call-graph questions about code | CodeGraph, not folio | folio indexes prose sections, not symbols |

## Hard rules

**Retrieval names candidates; the caller reads the file.** This is what makes a
stale index harmless: the cost of staleness is a wasted candidate, never a wrong
quotation. Do not add a mode that answers from stored text.

**Do not add a lexical retrieval route.** Measured on 2026-09-04: with a BM25
component fused in, the one query whose answer needed a synonym bridge
(`library` against a corpus that only writes `dependency`) fell out of the top
eight; with the vector route alone the same query ranked it first. `rg` already
covers exact matching, exhaustively. A fusion layer here subtracts.

**Do not add an approximate-nearest-neighbour index.** A flat `f32` matrix
scanned end to end is correct at this project's scale — see the table below. HNSW
would add a dependency, a build step, and a recall parameter to answer a question
that costs milliseconds without them.

**The incremental unit is the file, never the section.** Editing one line
shifts every later section's range without changing its content. Re-embedding the
whole changed file is both the smaller code and the correct answer; tracking
section identity across edits is the dirtiest code this project could acquire.

**Date every measured fact you rely on.** Embedding models and their runtimes
turn over quickly. Record the number, how it was measured, and the date, in the
same place you use it. An undated number is a guess to the next reader.

## Decision layer

`decisions/` is a separate repository carrying its own toolchain, so that a clone
of it alone can verify itself. `decisions/SPEC.md` is its charter: it defines
what a decision document is, what a fence is, and the checks that enforce both.
Read §7 and §12 before you add a decision or a fence.

**What belongs there rather than here.** A hard rule above is a rule someone has
to remember. A decision in `live/` with a fence in `fences/` is a rule that fails
a check when it is broken. Anything mechanically checkable should end up there,
and this file should end up holding only what cannot be.

**Adopting a rule means writing its fence in the same change** (§8 Adopt). When
that happens, delete the rule from this file and point at the decision id
instead. A rule kept in both places drifts, and the copy without the fence is
the one that goes stale.

**A rule is identified by its opening sentence, not by a number.** Rules leave
this file one at a time as they are adopted, so a position would shift under
every adoption and every reference to one would rot silently.

**Every rule above is a proposal in `decisions/intake/`, except the last.** A
proposal stays there until it has a fence, because §8 Adopt puts the decision
and its fence in the same change. Adopting one means deleting the rule from this
file and leaving only its row here.

| Rule | Decision | State |
|---|---|---|
| Truncation is recorded, never silent | `D-01M1PP6HJWFT2Q` | **live**, fenced by `truncation-is-recorded` |
| One model per vector space | `D-01M1PP6HKHF97G` | **live**, fenced by `one-model-per-vector-space` |
| The index holds references, never bodies | `D-01M1PP6HMS8Q54` | **live**, fenced by `references-never-bodies` |
| No frontmatter key is privileged | `D-01M1PP6HHJEQ4C` | **live**, fenced by `frontmatter-is-generic` |
| Defaults at query time | `D-01M1PP6HJ7H3BM` | **live**, fenced by `frontmatter-is-generic` |
| Retrieval names candidates | `D-01M1PP6HFHY7FW` | intake |
| No lexical retrieval route | `D-01M1PP6HG6GSGK` | intake |
| No approximate-nearest-neighbour index | `D-01M1PP6HGWJMQC` | intake |
| The incremental unit is the file | `D-01M1PP6HM6WGT4` | intake |

`D-01M1PP6HFHY7FW` will adopt as `fence: none`: no black-box test proves a mode
was never added. §4 asks that the reason be written down rather than typed as a
line, and the proposal carries it.

**Date every measured fact you rely on** is not a decision. It is a
documentation discipline, and nothing executable checks that a number was
written down honestly. It stays in this file for good.

**How a fence reaches folio.** §5 requires a fence to test the code layer as a
black box, through whatever it exposes to anyone else. For folio that is the
`folio` command on `PATH`, its exit codes and output, and the files it writes
under `.folio/`. A fence that reads `src/`, names a Cargo target, or calls a Rust
function is not a black-box test whatever its exit code says; that test belongs
in the code layer.

**A folio fence needs a corpus and an endpoint.** folio does nothing without
both, so a fence builds its own fixture corpus in a temporary directory, and
exits 2 rather than 1 when no embeddings endpoint is reachable (§5). Exit 2 is
the difference between "the decision is broken" and "nothing was running to
ask", and a fence that confuses them reports an outage as a violation.

## Measurements

`docs/measurements.md` holds every number folio reports about itself, with how
it was measured and when. The rule on dating a measured fact governs
what goes there. Two things are
still unmeasured and named in that file; do not assume either.

Those numbers have no committed artifact yet. They were taken in a session and
written down, which that rule permits and does not make reproducible. Anything
reported as a folio number from here on should come from a script in the
repository.

## Toolchain

Nix declares the tools on this machine. Do not install a tool globally to make a
task work; add it to the flake that needs it, so the next reader gets it.

**The decision layer carries its own flake.** `decisions/flake.nix` declares
`cargo`, `rustc`, `jujutsu` and `git`. It stays independent of this layer so
that a clone of `decisions/` alone can still verify itself. Enter it with `nix
develop` from inside `decisions/`, or run `direnv allow` there once.

**The code layer has no flake yet, and `llama-server` sits outside Nix.** It was
installed with `brew install llama.cpp` on 2026-09-04, and the machine's Nix
bridge rebuild failed and rolled back, so the binary at
`/opt/homebrew/bin/llama-server` is not in the declarative configuration and may
not survive the next rebuild. Write the flake when it declares something real:
`rustc`, `cargo`, and whichever embedding server the project settles on.

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

Read the `use-jujutsu-safely` skill before an unfamiliar command. The rules that
cost something:

- **Record finished work with `jj commit -m`, not `jj describe`.** `jj commit`
  describes `@` and opens a new empty change, so your next edit lands somewhere
  new. `jj describe` leaves `@` described, and the next edit amends the change
  you just described. Use `jj describe -r <id>` to correct a description, or to
  describe a change that is not `@`.
- **Pass `-r` to `jj new` every time.** `@` is per workspace, and the default
  parent is wrong as soon as a second workspace exists.
- **`jj describe` on a change that already has a description replaces it
  silently.** Check `jj log` first, and resolve a revset to a commit id before
  passing it to `-r`.
- **`jj util snapshot` when a session in a workspace ends.** It records the
  working copy and does nothing else; `jj status` also snapshots, but as a side
  effect of a command that may reset the working copy in the same breath.
- Do not discard existing changes. Existing changes belong to the user unless a
  task identifies them as agent changes.

`.folio/` is ignored. An index is derived from the corpus and belongs to whoever
built it, never to history.

## Language

English for documentation, project artifacts, code comments, and change
descriptions. Reply to the user in the language of their prompt.

## Reference paths

`docs/references.md` names the prior art this project was measured against, the
Open Knowledge Format specification behind the frontmatter rules, and the two
corpora every measurement used. Both corpora are Jujutsu working copies; never
write an index into one without an ignore entry first.
