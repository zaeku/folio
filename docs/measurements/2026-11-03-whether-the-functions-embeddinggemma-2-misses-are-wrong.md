# 2026-11-03 — whether the functions EmbeddingGemma 2 misses are wrong

The two code-query measurements score a hit only when the drawn function comes
first, yet a question written for one function can fit another as well. Of the
questions the best arm misses, how many had an equally good or better function
ranked above the target?

**About one miss in five; the other four are real.** Counting those as found
raises MRR by 0.06 to 0.07 on each corpus and leaves the order of the corpora
as it was.

| Corpus | Misses | True miss | Ambiguous | False miss | Bad query | top 1 | MRR |
|---|---|---|---|---|---|---|---|
| daq4xss | 51 | 38 | 9 | 4 | 0 | 58 → 65 of 109 | 0.659 → 0.721 |
| ripgrep | 96 | 80 | 9 | 6 | 1 | 31 → 41 of 127 | 0.390 → 0.457 |
| minisqlite | 45 | 35 | 7 | 3 | 0 | 32 → 39 of 77 | 0.561 → 0.630 |

- *True miss*: only the target is what the question describes.
- *Ambiguous*: the target and a function above it both are, such as one
  Prometheus text encoder copied into several services, or one event-mask check
  in Rust and in Python.
- *False miss*: a function above it is, and the target is less so. One question
  describes expanding `SELECT *`, and the drawn target is the `UPDATE …
  RETURNING` variant.
- *Bad query*: no candidate is.

The arrow is from the strict score in
`2026-11-02-whether-embeddinggemma-2-finds-functions-from-a-question.md` and
`2026-11-03-whether-embeddinggemma-2-finds-functions-in-a-denser-call-graph.md`
to a lenient one. The lenient one ranks a miss at the first function above it
scored 2, and is an upper bound on what the judge allows rather than a new score
for the arm.

By the target's rank, the ambiguous misses sit near the top: 20 of the 25 are at
ranks 2 to 5, where 75 of 100 misses are true. At ranks 6 to 20, 39 of 48 are
true, and past rank 20, 39 of 44.

**What it does not show.**

- The judges read at most three functions above each target. A target at rank 5
  or lower has functions between rank 4 and itself that nobody judged, so the
  number of ambiguous and false misses is a lower bound.
- The judges mostly read the first 25 to 40 lines of each span and its doc
  comment, and rarely the callers or owning type; six of the seven judges
  said so.
- A model judged a model's ranking, on questions a model wrote. A handful of
  verdicts were read by hand and found sound; that is not a review of all 192.
- The lexical arms were not judged, so this does not say whether CodeGraph or
  `rg` misses would turn out ambiguous more often.

**Method.**

- *Arm.* EmbeddingGemma 2 with `task: code retrieval | query: ` before the
  question and `title: <filename> | text: ` before the function, served as in the
  two measurements above, recomputed for this run. Its first-place counts
  match theirs. `benchmarks/queries/misses.py` is that script, tidied to write
  the key the joining step reads, and has not been rerun in that form.
- *Blinding.* For each miss the candidates are the target and the functions
  ranked above it, at most three, shuffled under labels A to D. A judge was told
  neither the rank nor which was the target.
- *Judges.* Seven Claude Opus 5.5 subagents run in parallel, two for daq4xss,
  three for ripgrep and two for minisqlite, each scoring every candidate 2 (what
  the question describes), 1 (related, such as a caller, callee or variant) or
  0. Several candidates could score 2.
- *Joining.* `benchmarks/queries/judged.py`. The joined verdicts, with each
  judge's one-sentence reason, are `benchmarks/queries/*-seed58-misses.json`.
