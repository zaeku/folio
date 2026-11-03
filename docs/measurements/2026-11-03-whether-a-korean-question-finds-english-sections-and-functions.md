# 2026-11-03 — whether a Korean question finds English sections and functions

It was reported that asking about English documents in Korean finds nothing.
With EmbeddingGemma 2, how much does a Korean question lose against the same
question in English, on English prose and on source code?

**On prose a Korean question finds nearly what an English one does; on code it
finds about half.** The same targets asked for in Korean rank sections within
0.020 of English when technical terms stay in English, and within 0.091 when
they are translated too. Functions lose 0.140 and 0.185 of an English MRR of
0.380.

| Questions | Sections, MRR | Sections in top 20 | Functions, MRR | Functions in top 20 |
|---|---|---|---|---|
| English | 0.870 | 147 of 147 | 0.380 | 94 of 127 |
| Korean, technical terms in English | 0.850 | 145 | 0.240 | 69 |
| Korean, technical terms translated | 0.779 | 143 | 0.195 | 59 |

Behind the search query prefix, with `title: none` for the sections column and
the filename title for the functions column, which are the best arm for each in
`2026-11-03-which-prefixes-serve-sections-and-functions-in-one-index.md`. Behind
the code-retrieval query prefix functions fell the same way, to 0.210 and
0.164. These describe these query sets and the difference between arms, for
the reasons `D-01M1VNK3CJHMPR` gives.

**What the gaps mean.** A section is English prose, and a question that means
the same in another language lands near it. A function is identifiers and
English comments, and a Korean question has less of either to meet: the gap
narrows when the question keeps the English terms, which are the words the code
itself uses. So the report holds for code and not, under this model, for prose.
The report may come from the production endpoint before this one, which served
gte-modernbert-base, an English model; that model is no longer on this machine,
so the explanation is not tested here.

**What it does not show.** Every Korean question is a translation by Gemini 3.8
Flash of an English one, not a question a Korean speaker wrote. The endpoint is
shared, so a vector may move slightly under other traffic: the English column
is within 0.004 of the same arms measured on a local server in
`2026-11-03-which-prefixes-serve-sections-and-functions-in-one-index.md`.
Korean documents were not asked about, in either language.

**Method.**

- *Corpus.* The mixed tree of that measurement: `rust-lang/book` at
  `1500248d8f230566e4ec9f27fcbb8fe9e2898ab1` with `BurntSushi/ripgrep` at
  `91f5d22ef48b` under it, 656 sections and 2,905 functions.
- *Questions.* `benchmarks/queries/rust-book-seed55.json` and
  `benchmarks/queries/ripgrep-seed58.json`, and their rewrites by
  `benchmarks/queries/translate.py`: `*-ko.json` keeping the terms a Korean
  developer writes in English, `*-ko-all.json` translating those too. Each pair
  keeps its target.
- *Arms.* `benchmarks/queries/mixed.py` against `google/embeddinggemma-2` at 768
  dimensions on the shared endpoint at `10.4.208.6:8080`.
