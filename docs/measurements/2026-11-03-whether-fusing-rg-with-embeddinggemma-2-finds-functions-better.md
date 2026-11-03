# 2026-11-03 — whether fusing rg with EmbeddingGemma 2 finds functions better

The code-query sets drop every question that names a piece of its function,
which is what `rg` matches on. On the questions as the model wrote them, names
included, does fusing `rg`'s ranking with EmbeddingGemma 2's find functions
better than either alone?

**No: fusion is far below Gemma alone on every corpus, named questions
included.** `rg`'s ranking is mostly noise on these questions, and fusing it at
equal weight pushes Gemma's correct answers down.

| Corpus | Questions | Gemma | `rg` | Fused |
|---|---|---|---|---|
| daq4xss | 149 | 0.694 | 0.078 | 0.298 |
| ripgrep | 150 | 0.383 | 0.070 | 0.216 |
| minisqlite | 150 | 0.591 | 0.137 | 0.401 |

MRR. Split by whether the question names a piece of the function:

| Corpus | Named | Gemma | `rg` | Fused | Unnamed | Gemma | `rg` | Fused |
|---|---|---|---|---|---|---|---|---|
| daq4xss | 40 | 0.788 | 0.098 | 0.362 | 109 | 0.659 | 0.070 | 0.274 |
| ripgrep | 23 | 0.345 | 0.063 | 0.236 | 127 | 0.390 | 0.071 | 0.212 |
| minisqlite | 73 | 0.622 | 0.169 | 0.430 | 77 | 0.561 | 0.107 | 0.374 |

The unnamed columns equal the code-prefix, filename-title and `rg` arms of
`2026-11-02-whether-embeddinggemma-2-finds-functions-from-a-question.md` and
`2026-11-03-whether-embeddinggemma-2-finds-functions-in-a-denser-call-graph.md`.

**What the gaps mean.** A named question here names one word of the function,
such as `exceeds` or `parse`, and that word is common across the tree, so `rg`
gains little from it: its MRR on named questions is at most 0.169. Gemma gains
more from the same word on two corpora of three.

**What it does not show.** No question names a whole identifier. That is the
case `rg` answers outright and no ranking needs to be fused for, so this says
nothing against `rg` once the name is known; it says that fusing at equal
weight hurts while it is not. A weighted fusion, or one that admits `rg` only
when a term is rare, was not tried. The questions are what Gemini 3.8 Flash
wrote, not a sample of what developers ask.

**Method.**

- *Set.* Every question generation produced: those kept, and those it dropped
  for containing a piece of the function's name, which are
  `benchmarks/queries/*-seed58-dropped.json`. No question was dropped as copied,
  so this is the whole of what the model wrote.
- *Arms.* `benchmarks/queries/hybrid.py`. Gemma as in the measurements above.
  `rg` as in `benchmarks/queries/code-arms.py`. Fusion takes each arm's top 100
  and scores a function `1/(60 + rank)` summed over the arms that rank it.
