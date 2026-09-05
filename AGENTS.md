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

`decisions/` is a separate repository carrying its own toolchain, so that a clone
of it alone can verify itself. `decisions/SPEC.md` is its charter: it defines
what a decision document is, what a fence is, and the checks that enforce both.
Read §7 and §12 before you add a decision or a fence.

**What belongs there rather than here.** A hard rule above is a rule someone has
to remember. A decision in `live/` with a fence in `fences/` is a rule that fails
a check when it is broken. Anything mechanically checkable should end up there,
and this file should end up holding only what cannot be.

**Adopting a rule means writing its fence in the same change** (§8 Adopt), and
deleting the rule from this file. A rule kept in both places drifts, and the
copy without the fence is the one that goes stale.

**A rule is identified by its opening sentence, not by a number.** Rules leave
this file one at a time as they are adopted, so a position would shift under
every adoption and every reference to one would rot silently.

**`intake/` is empty and every rule that could become a decision has.** One
entry is left above, and it is the one nothing executable can check: whether a
number was written down honestly.

**No list of decisions is kept anywhere.** `decisions/live/` is the list: one
file per decision, named by its id and title, and `cargo run --bin check` there
derives the rest. §2 allows the code layer to point at the decision layer and
forbids the reverse, and it says the decision → fence relationship is derived by
scanning rather than stored — so a written index would be a stored reverse
reference, and a generated committed one would be the same thing rebuilt on
every run. A table of them used to live in this file and was already wrong twice
over by the time it was removed.

A decision does not have to pass through this file to exist. A rule someone
wrote here does have to leave it when it is adopted, and how to find which
decision a departed rule became is §9's question, answered by
`jj log -r 'diff_contains("D-...")' -p` in whichever layer you are standing in:
the change that deletes the rule is the mapping.

**What nothing can execute is held by bytes instead.** Some decisions carry
`fence: none`, because no black-box test reaches them — an absence cannot be
proved by running something, and a saving that shows only as latency does not
show in a fence's fixture at all. §4 requires each to record in `verified:` the
SHA-256 of its body at adoption, so the check reports when a decision drifts out
from under what someone read. It does not judge the prose, and it does not print
the hash that would silence it —
`cargo run --bin hash <path>` computes one, and running it deliberately is the
act of re-verifying.

**Date every measured fact you rely on** is not a decision. It is a
documentation discipline, and nothing executable checks that a number was
written down honestly. It stays in this file for good.

**How a fence reaches folio.** §5 requires a fence to test the code layer as a
black box, through whatever it exposes to anyone else. For folio that is the
`folio` command on `PATH`, its exit codes and output, and the files it writes
under `.folio/`. A fence that reads `src/`, names a Cargo target, or calls a Rust
function is not a black-box test whatever its exit code says; that test belongs
in the code layer.

**`check` tests what you published, not what you built.** §5 reaches folio
through the `folio` command on `PATH`; `target/release/` is not on it. Run
`cargo install --path .` before `cargo run --bin check`, or a fence will
correctly report that the binary someone could actually run does not hold the
decision you just wrote. This has happened: a `folio` eight hours old failed
eight fences, and the fences were right.

**A fence names the binary it tested.** Every `fail` leads with
`$(command -v folio)`, because the failure that matters is often about which
binary answered rather than about the code in front of you. It leads rather than
trails so that a multi-line message from the tool cannot push it out of sight.

**A fence that cannot read the record store exits 2, not 1.** Reading
`.folio/index.db` is how most fences see what folio indexed, and a folio that
writes something else has not violated *their* decision — it has made them
unevaluable. `vectors-out-of-the-database` and `no-ann-index` are the fences
that guard the store's shape, and they are the ones that fail. That division is
what keeps a changed store format from reporting as eight simultaneous
violations.

**A folio fence needs a corpus and an endpoint.** folio does nothing without
both, so a fence builds its own fixture corpus in a temporary directory, and
exits 2 rather than 1 when no embeddings endpoint is reachable (§5). Exit 2 is
the difference between "the decision is broken" and "nothing was running to
ask", and a fence that confuses them reports an outage as a violation.

## Measurements

`docs/measurements.md` holds every number folio reports about itself, with how it
was measured and when. The rule on dating a measured fact governs what goes
there, and a number quoted anywhere else — `README.md` included — should be
traceable to an entry in it.

**A folio number comes from `benchmarks/run.py`.** It fetches two public corpora
pinned by commit and drives the `folio` command on `PATH`, so anyone can rerun
it. `benchmarks/README.md` says what those numbers are and, at more length, what
they are not: contamination is measured and ranking quality deliberately is not.
`benchmarks/reports/` is scratch and is not committed; a run worth keeping goes
into `docs/measurements.md` under its own date.

The 2026-09-04 entries predate that harness and have no committed artifact. They
are the record of how the model and the design were chosen, and they say so.

Two things are still unmeasured and named in that file. Do not assume either.

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
Open Knowledge Format specification behind the frontmatter rules, and the
corpora — the two public ones the benchmarks pin, and the two private ones the
design was chosen on. The private pair are Jujutsu working copies; never write
an index into one without an ignore entry first.
