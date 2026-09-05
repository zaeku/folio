# Reference paths

**`zvec-grep`, the prior art this project was measured against.** Source:
https://github.com/zvec-ai/zvec-grep (Apache 2.0). A local checkout sits at
`~/workspace/References/zvec-grep`. Treat it as read-only. It is not
installed on this machine and does not need to be.

Read these files when you need the design folio departs from:

| File | Holds |
|---|---|
| `docs/04-pipeline.md` | What it indexes, and its freshness model |
| `docs/03-mcp.md` | Its agent-facing tool surface |
| `src/engine/pipeline/indexing/input-budget.ts` | The pre-embedding chunker, and why a long section is split rather than truncated |
| `src/engine/extraction/markdown/extractor.ts` | Its markdown extractor, which reads no frontmatter — the gap folio exists to close |
| `src/engine/models/catalog.ts` | Its fixed model catalog, the constraint folio replaces with an endpoint |

**The Open Knowledge Format.** Source:
https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md. A
local copy sits at `~/.claude/skills/okf-spec/references/okf/SPEC.md`, reached
through the `okf-spec` skill. Rules 5 and 6 come from it. Read
"Concept documents" for the frontmatter contract and "Provenance, trust, and
lifecycle" for the `status`, `stale_after`, `verified` and `generated` families
before you touch filtering. The spec is the only source of truth for OKF; never
author against a remembered template.

**The decision layer.** `decisions/` is a separate repository. `decisions/SPEC.md`
is its charter. Read §4 for a decision document's four frontmatter keys, §5 for
what a fence and a tripwire are and the three exit values they share, §7 for the
checks, and §8 for the six lifecycle transitions. `cargo run --bin check` from
inside `decisions/` runs the §7 checks; `cargo test` there drives one fixture
layer carrying every §7 defect at once.

**The public corpora, which every current number comes from.** Fetched and
pinned by `benchmarks/run.py`, unpacked under `benchmarks/corpora/`, which is
ignored. `benchmarks/README.md` says why these two and not others.

| Corpus | Shape |
|---|---|
| `rust-lang/book` `src/` at `1500248d` | 112 files, 1.2 MB, no frontmatter at all |
| `mdn/content` `files/en-us/` at `f4c14731` | 14,616 files, 59.6 MB, `status` as a list carrying `deprecated` |

**The private corpora the design was chosen on.** They carry the 2026-09-04
measurements and nothing since. Both are read-only to this project, and both are
Jujutsu working copies — never write an index into one without an ignore entry
first.

| Path | Shape |
|---|---|
| `(a private decision surface)` | 116 files, 476 KB, one fact per file, `id` / `question` / `verified` / `supersedes` frontmatter |
| `(a private prose surface)` | 5 files, 146 KB, long prose sections up to 10 KB |

The first private corpus carries `supersedes` on 11 sections, and every one of those 11
points at an id that is not in the corpus. That is not a defect: the project it
comes from removes a decision when it stops holding and keeps the text in
Jujutsu history, so its surface holds only what is still true. It solves the
problem folio's anti-join solves, one layer earlier and more thoroughly.

So the anti-join is not for corpora shaped like that one. It is for corpora that
keep a superseded statement in place — an OKF bundle, where `deprecated` means
"kept for links and history" — and for those, `--exclude-pointed-by` is the
filter a `--where` predicate cannot express, because the pointer sits on the
successor rather than on the record to drop.

No corpus here exercises it. MDN marks retirement in place, which is the shape
the anti-join was built for, but it marks it with `status` rather than with a
pointer from the successor — so `--where status!=deprecated` reaches it and
`--exclude-pointed-by` has nothing to join on. The measurement for the anti-join
is therefore a fixture rather than a number from any of these.
