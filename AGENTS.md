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
| Where a setting comes from, which corpus a command stands in, what may be sent where | `src/config.rs` | Resolution, the upward walks, and the transport rule are one subject |
| The request to the endpoint, and the arithmetic on what returns | `src/embed.rs` | folio runs no model; this is the whole of what it says to one |
| The `--where` grammar | `src/filters.rs` | A predicate reads one record and nothing else, so it needs nothing else |
| The service file `folio unit` prints | `src/unit.rs` | It starts nothing and reads only configuration |
| The command line: flags, and the sentences `--help` prints | `src/cli.rs` | Declaration and help text, and no work |
| Keeping an index current | `src/index.rs` | The write path, and the file is its unit |
| Answering a question: ranking, merging, what a result names | `src/query.rs` | The read path, and it embeds one input rather than a corpus |
| `main`, the small commands, and the index files under `.folio/` | `src/main.rs` | Dispatch, `doctor`, `status`, `config`, `extract`, and the three functions that open what a command reads |
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

## Board

`kanban/` is a third repository, ignored here, holding what is next rather than
what is true. Run `kanban-md board` from the repository root. It is local: it
orders one person's work and records what blocks what, and nothing in it is an
agreement with anyone else. Work that has to be coordinated with someone on
another machine belongs on the forge, not here.

Three layers, three rhythms. This one changes per work item, `decisions/`
changes when a rule does, and the board changes hourly.

**A card's tag is its kind, and the kind says what has to exist before the card
is claimed.** The loop this project is built on runs a measurement, then a
decision, then code, and a card belongs to one of those three.

- A `measurement` card needs nothing to exist first. It produces what the next
  one reads.
- A `decision` card carries its decision's id as a second tag before it leaves
  `backlog`. Having an id means an `intake/` document exists, and that document
  is the thing someone can argue with. It may be a question rather than a claim:
  `decisions/SPEC.md` §5 lets a fence be adopted while it exits 2, so a
  candidate can stand and report where the work is until the code catches up.
- A card whose work changes a rule stands that rule's fence up first.
  `cargo run --bin check` in `decisions/` then says where the work stands, and
  the fence names the surface it cannot find yet.

**A card's kind is learned when someone picks it up, not when it is filed**, so
this is asked at the claim and never at creation. Two of the cards worked on
2026-09-07 had the wrong kind on them: one asked for a measurement a live
decision had already refused, and one proposed a feature whose worth could not
be judged before two runs that did not exist yet. Correcting the kind was the
most useful part of both reviews, and a rule enforced at filing would have
demanded the answer at the moment it was least available.

Not every card touches a rule. Re-running a benchmark or measuring two backends
wants no decision, and telling those apart is a judgement rather than a
condition to check.

**Nothing enforces any of this.** No fence reaches the board — a fence tests the
code layer as a black box, and the board is neither the code layer nor
published — so it is a rule someone has to remember, which is what this file is
for.

## Measurements

Every number folio reports about itself is one file under `docs/measurements/`,
named by the date it was taken and the question it answers. `docs/measurements.md`
is the preamble and says how to read them.

**`ls docs/measurements/` is the index, and no other exists.** A new measurement
is a new file, never a line added to a list. Each carries its own date in its
heading as well as in its name, and one that corrects or relies on another names
it by filename, never by position. A fence in the decision layer fails on any of
those, so the reasons live there and what is here is how to write one.

A number quoted anywhere else — `README.md` included — has to be traceable to one
of those files, and `cargo test` fails when it is not: `tests/numbers.rs` reads
the prose of this file, `README.md` and `skill/SKILL.md`, and asks whether
every number carrying a unit appears under `docs/measurements/`. It skips fenced blocks,
because a transcript or a flag's value is an illustration rather than a claim.

**It checks existence, not agreement.** A number used in the wrong sense still
passes. What it catches is a value changed in one place and not the other, which
is the failure that happens. Do not answer it by removing the number: an undated
number is a guess to the next reader, and no number is not an improvement on a
guess. Measure it and record one.

**A folio number comes from `benchmarks/run.py`.** It fetches two public corpora
pinned by commit and drives the `folio` command on `PATH`, so anyone can rerun
it. `benchmarks/README.md` says what those numbers are and, at more length, what
they are not — read it before adding a number to the harness, because what it
declines to measure is a decision and not an omission.
`.gitignore` excludes `benchmarks/reports/`, which is scratch. A run worth
keeping becomes a file under `docs/measurements/` on its own date, and that
judgement is a person's — the harness never writes there.

**One table in `README.md` belongs to `run.py`.** It is fenced by HTML comments
naming its owner, and a full run replaces it. Do not edit it by hand and do not
quote its figures in the prose beside it; change `run.py` or rerun it.
Everything outside those markers is hand-written, and a fence fails on both.

The 2026-09-04 measurement predates that harness and has no committed artifact.
It is the record of how the model and the design were chosen, and it says so.

`docs/measurements/2026-09-04-private-corpora.md` names two things as unmeasured
and both have since been answered, in opposite ways: whether a reranker over the
top k earns its latency was measured on 2026-09-07 and decided, and where
retrieval quality falls as a corpus grows is refused rather than pending — a
live decision declines to publish that score and gives its reasons. That file
still reads as it did, because it records what was true on its own date; this one
does not, so it says so here.

## Toolchain

Nix declares the tools on this machine. Do not install a tool globally to make a
task work. Add it to the flake that needs it, so the next reader gets it.

**Work inside this layer's flake.** `flake.nix` declares `rustc`, `cargo`,
`clippy`, `rustfmt`, `jujutsu` and `git`, and `mkShell`'s stdenv carries the C
compiler `rusqlite`'s bundled SQLite needs. `.envrc` is `use flake`, so
`direnv allow` once and the shell arrives with the directory; `nix develop` does
the same by hand.

All four rust tools come from one nixpkgs on purpose. Two installations answering
at once is what `E0514: found crate serde compiled by an incompatible version of
rustc` reports, and outside this shell that is the state of this machine:
`rustc` and `cargo` resolve to the system's Nix profile while `cargo-clippy` and
`rustfmt` resolve to an older rustup toolchain, so `cargo clippy` cannot run at
all. `cargo +stable` is not a way around it either — the `cargo` in front does
not implement `+toolchain`.

**No embedding server is declared, and both are declared outside this project.**
`llama-server` and `text-embeddings-router` are in the machine's Nix
configuration by way of its Homebrew bridge, so they survive a rebuild, but that
is the machine's declaration and not this project's. Which of the two folio
settles on is undecided, and the next paragraph is why the shell does not need
one.

**`cargo test`, `cargo fmt --check` and `cargo clippy` run on push and on a pull request**, on
Linux and on macOS, from `.github/workflows/ci.yml`. Neither installs anything
nor starts a server, for the reason in the next paragraph. They block: a branch
clippy or rustfmt disagrees with does not merge.

`rustfmt.toml` carries the settings that make that bearable. rustfmt's defaults
expand a packed struct literal or match arm, and packed is why a 2,600-line
`main.rs` stays readable enough that the table above declines to split it, so
`use_small_heuristics = "Max"` keeps a construct on one line wherever it fits.
Run `cargo fmt` before you commit rather than formatting against it by hand.

The workflow uses whatever `stable` the runner image carries and pins nothing,
which is why it prints `rustc --version` before it does anything else. The
fences in `decisions/` are not run there — that repository is unpublished, so a
workflow cannot reach it, and `cargo run --bin check` stays something a person
runs.

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

**Jujutsu, colocated.** `.jj` and `.git` coexist. Measured on **jj 0.45.1**.

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
  Snapshot before anything rewrites what that workspace has checked out. Once
  its working copy is stale, `jj util snapshot` refuses as well, and the `jj
  workspace update-stale` that clears the stale state is what removes its
  files.
- Do not discard existing changes. Existing changes belong to the user unless a
  task identifies them as agent changes.
- **Sign before you push.** GitHub rejects an unsigned push to `main`.

  1. Run `jj sign`. It takes the `revsets.sign` revset, set here to
     `reachable(@-, mutable())`. That revset reaches in both directions, so it
     includes `@` as a descendant of `@-` whenever anything is signed at all.
  2. Run `jj unsign -r @`. Add `--ignore-working-copy` when the repository will
     no longer open; that flag skips the snapshot and not the revset, so it
     prevents nothing else here.
  3. Push.

  Do not leave `@` signed. `behavior = "keep"` preserves a signature through a
  rewrite, every jj command rewrites `@`, and a signed `@` asks 1Password on
  `jj status`. Nothing is signed as it is made, because jj signs the
  working-copy commit on every snapshot and each signature is a prompt.
  Unsigning after the push is safe, since `@` is neither pushed nor a parent of
  anything.

## Language

Write documentation, project artifacts, code comments, and change descriptions
in English. Reply to the user in the language of their prompt.

## Reference paths

`docs/references.md` names the prior art this project was measured against, the
Open Knowledge Format specification behind the frontmatter rules, and the
corpora — the two public ones the benchmarks pin, and the two private ones the
design was chosen on. The private pair are Jujutsu working copies. Add an
ignore entry before you write an index into one.
