# 2026-11-03 — whether splitting a long function at syntax nodes finds it

A function longer than folio's 8,000-character budget is embedded cut at the
budget, so whatever it does past that point is invisible. Does splitting it at
syntax nodes instead, each piece carrying the function's header, find it when a
question asks for it, and what does it cost the other questions?

**It finds the long function and costs the rest little, but the long
functions are too few to move the score.** Of five questions whose target is
over budget, splitting raises two from far down to the top three and changes
little for the other three. No question loses first place, and MRR over the
whole set rises by 0.002 on each corpus.

| Question | Cut at 8,000 | Split |
|---|---|---|
| daq4xss 99 | 28 | 2 |
| daq4xss 102 | 3 | 2 |
| minisqlite 7 | 33 | 31 |
| minisqlite 28 | 1 | 1 |
| minisqlite 64 | 17 | 3 |

| Corpus | Long functions | Chunks | Questions moved down | MRR cut | MRR split |
|---|---|---|---|---|---|
| daq4xss | 36 of 1,095 | 82 | 9, by 1 to 7 places | 0.659 | 0.661 |
| minisqlite | 32 of 9,769 | 83 | 3, by 1 or 2 places | 0.561 | 0.563 |

ripgrep has one function over budget, at 8,811 characters, and no question on
it, so it was not run. The MRR when cut equals the one the code-prefix,
filename-title arm reached in
`2026-11-02-whether-embeddinggemma-2-finds-functions-from-a-question.md` and
`2026-11-03-whether-embeddinggemma-2-finds-functions-in-a-denser-call-graph.md`,
so both columns rank the same questions on the same units.

**What it does not show.** Five questions are anecdotes, not a rate. A chunk is only as good as the node it ends on: 8 of
the 165 chunks are still cut at the budget, because a single statement with no
block or arm inside it is longer than that.

**Method.**

- *Split.* `benchmarks/queries/chunked.py`. A span over budget is replaced by the
  outermost spans inside it that the Tree-sitter query matches — a block's
  child statements and a `match` arm in Rust, a compound statement's children
  and a `case` in C++, a block's children in Python — through `ast-editor
  inspect --query`, recursing into any that are still over budget. Adjacent
  pieces are then merged while they fit. Every chunk starts with the lines of
  the function before its first piece, its doc comment and signature, at most
  1,500 characters.
- *Ranking.* Gemma, `task: code retrieval | query: ` and `title: <filename> |
  text: `, served as in the measurements above. A function ranks by the best of
  its chunks; the other functions are embedded as before.
- *Sets.* `benchmarks/queries/daq4xss-seed58.json` and
  `benchmarks/queries/minisqlite-seed58.json`.
