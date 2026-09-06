# 2026-09-06 — what dividing does to one large page

The page `README.md` uses to show dividing, measured so the figure it quotes has
a source. `mdn/content` at the commit `benchmarks/run.py` pins,
`web/html/reference/attributes/index.md`, `--max-chars 8000`,
`gte-modernbert-base-Q8_0`.

| | |
|---|---|
| The page | 52,862 characters |
| Its largest heading section, `## Attribute list` | 46,529 characters |
| What folio indexes it as | 11 records, 0 truncated |

**The section counts are not folio's own.** folio reports records, not the
sections it split before dividing them, so the 46,529 was counted outside it: a
section is a heading line plus every line until the next heading of any level,
which is the rule `src/sections.rs` splits on, and the count includes the
heading line and the newlines. `README.md` carried 46,528 for this before it was
measured, which is the same section under some other counting of its edges.

**Across the corpus, by that same rule outside folio:** 104,967 heading
sections, of which 134 hold more than 8,000 characters, and the largest is
50,659 characters — `## HTML elements` in
`web/api/html_sanitizer_api/default_sanitizer_configuration/index.md`.

Those totals do not match folio's. folio indexes 119,565 records over 14,616
files, and 104,967 plus one span per file for what precedes its first heading is
119,583 — near enough to place the difference there rather than in the heading
rule. The pre-dividing run counted 142 sections at the budget where this rule
counts 134. So the corpus-wide figures here are the right order and not folio's
numbers; the per-page row above is, because folio produced it.

**It does cross-check the benchmark.** Dividing took MDN from 119,359 records to
119,565, which is 206 more. Around 134 over-budget sections becoming about 340
pieces is 206, so the two accounts of the same event agree at that scale.
