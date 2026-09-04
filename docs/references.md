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

**The corpora every number here was measured on.** Both are read-only to this
project, and both are Jujutsu working copies — never write an index into one
without an ignore entry first.

| Path | Shape |
|---|---|
| `(a private decision surface)` | 116 files, 476 KB, one fact per file, `id` / `question` / `verified` / `supersedes` frontmatter |
| `(a private prose surface)` | 5 files, 146 KB, long prose sections up to 10 KB |

The first corpus already carries `supersedes` on 11 sections. folio can filter on
its presence but cannot yet exclude the records it points at, because that is a
join across sections rather than a predicate on one. That is the next feature and
the reason this project exists: a superseded statement and a live one are
semantically alike, so a vector index ranks them together unless something tells
it not to.
