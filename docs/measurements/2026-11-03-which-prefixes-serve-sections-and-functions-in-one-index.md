# 2026-11-03 — which prefixes serve sections and functions in one index

An index holding both markdown sections and source functions has one query
prefix for every question, and EmbeddingGemma 2's model card gives a different
one for each: `task: search result | query: ` for search and `task: code
retrieval | query: ` for code. Which serves both, and does naming the file in
the document's title help both kinds of unit?

**The search-result prefix serves both, and the filename title helps functions
and costs sections.** Section questions rank best behind the search prefix
with `title: none`. Function questions rank the same behind either query
prefix and gain from the filename, which section questions lose by about as
much.

| Query prefix | Title | Sections, 147 | Functions, 127 | All, 274 |
|---|---|---|---|---|
| search result | `none` | 0.874 | 0.364 | 0.638 |
| search result | filename | 0.855 | 0.384 | 0.637 |
| code retrieval | `none` | 0.853 | 0.354 | 0.622 |
| code retrieval | filename | 0.840 | 0.385 | 0.629 |

MRR. The section column's first row is within 0.001 of the 0.875 that
`2026-10-11-what-embeddinggemma-2-changes-on-rust-book.md` measured on rust-book
alone, and the function column's best is within 0.005 of the 0.390 that
`2026-11-03-whether-embeddinggemma-2-finds-functions-in-a-denser-call-graph.md`
measured on ripgrep alone, so sharing an index cost either kind little.

**What the gaps mean.**

- *Query prefix.* Search beats code retrieval by 0.015 to 0.021 on sections and
  is within 0.010 of it on functions, either way. On the private corpus code
  retrieval led by 0.038 on functions alone; on ripgrep alone search already led.
- *Title.* The filename gains functions 0.020 or 0.031 and loses sections 0.019
  or 0.013, so one document prefix for both kinds trades one against the other,
  and a prefix per kind of unit would take the gain without the loss.
- *Crossing over.* A section question's first result was a function twice in 588
  rankings, both behind the code-retrieval prefix. A function question's first result was a section 12 times of 127
  with `title: none` and 4 or 5 with the filename, mostly ripgrep's own guide
  describing the same feature, which a reader may well want.

**What it does not show.** The corpus is two repositories put side by side, and
each query set was written for its own; a question written about a tree that
documents its own code was not asked. Sections ranked one by one, without the
merge of adjacent results `folio query` performs, so the section column is not
what `folio query` returns.

**Method.**

- *Corpus.* `rust-lang/book` at `1500248d8f230566e4ec9f27fcbb8fe9e2898ab1`, its
  `src` at the root as `benchmarks/run.py` pins it, and `BurntSushi/ripgrep` at
  `91f5d22ef48b` under `ripgrep/`. Sections are the 656 rows `folio index
  --max-chars 8000` recorded for the tree's markdown, 101 of them ripgrep's own
  documents. Functions are the 2,905 units of `benchmarks/queries/functions.py`.
- *Questions.* `benchmarks/queries/rust-book-seed55.json` and
  `benchmarks/queries/ripgrep-seed58.json`.
- *Arms.* `benchmarks/queries/mixed.py`, EmbeddingGemma 2 Q8_0 on
  `llama-server` 0.8.0 with `--pooling mean`. A section question hits the unit
  containing its line, as `benchmarks/queries/score.py` counts; a function
  question hits a unit overlapping its function by half of the shorter.
