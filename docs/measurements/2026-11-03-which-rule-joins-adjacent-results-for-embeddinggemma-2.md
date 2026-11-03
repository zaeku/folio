# 2026-11-03 — which rule joins adjacent results for EmbeddingGemma 2

`folio query` joins touching sections into one result while each scores at
least 0.9 times the best of them, a ratio read off gte-modernbert-base. A report
from a production corpus said that under EmbeddingGemma 2, whose scores sit in
a narrow band, the ratio admits nearly every section of a relevant file. Does
another rule rank better on the public benchmarks?

**A threshold set from the query's own scores ranks sections best, and code is
indifferent to the rule.** Joining a piece while it scores at least the best
minus a quarter of the way from the best to the median of the hydrated window
raised section MRR by 0.041 in English and by 0.072 and 0.125 in Korean, and
narrowed the first result. A fixed ratio of 0.95 came within 0.013 of it. On
functions every rule landed within 0.010 of every other.

| Rule | rust-book | rust-book, Korean | rust-book, Korean terms | ripgrep | ripgrep, Korean | ripgrep, Korean terms |
|---|---|---|---|---|---|---|
| ratio 0.90, 0.7.1 | 0.857 | 0.804 | 0.684 | 0.370 | 0.229 | 0.188 |
| ratio 0.93 | 0.891 | 0.859 | 0.742 | 0.371 | 0.226 | 0.187 |
| ratio 0.95 | 0.893 | 0.868 | 0.796 | 0.377 | 0.226 | 0.185 |
| ratio 0.97 | 0.891 | 0.873 | 0.792 | 0.380 | 0.225 | 0.185 |
| no joining | 0.874 | 0.851 | 0.774 | 0.377 | 0.224 | 0.179 |
| median, k = 0.25 | 0.898 | 0.876 | 0.809 | 0.379 | 0.225 | 0.181 |
| median, k = 0.5 | 0.886 | 0.856 | 0.812 | 0.377 | 0.223 | 0.185 |

MRR, of 147 section questions and 127 function questions per column. The
English rust-book question's first result averaged 82.0 lines under 0.90, 50.7
under the median rule at 0.25 and 42.5 with no joining.

**What the gaps mean.** A cosine's spread is the model's: the ratio of two
scores means one thing on gte and another on EmbeddingGemma 2. The median of the
window is where the query's background sits, so a quarter of the way from the
best toward it is a distance measured in that query's own spread, which a
ratio is not. No joining at all ranks below both, so joining earns its place;
the threshold was what was wrong. Functions rarely touch at similar scores, so
the rule seldom has anything to decide there.

**What it does not show.** Two corpora, both English, with questions a model
wrote, and one model. The report's own 48 queries on a production corpus, half
of them asked of Korean documents, put 0.95 ahead of 0.90 by 0.03 and did not
try the median rule. The median depends on the window it is taken over, and
the window is 128 rows or the limit. A corpus smaller than the window has no
background to take a median of; there folio joins at 0.95 times the best, and
that case was not measured beyond a six-section fence fixture.

**Method.**

- *Corpus.* The mixed tree of
  `2026-11-03-which-prefixes-serve-sections-and-functions-in-one-index.md`,
  indexed by folio 0.7.1 with `code: true` and the search prefixes, 656
  sections and 2,921 functions.
- *Replay.* `benchmarks/queries/merge-replay.py` scores every unit from the
  endpoint, keeps the top 128, cuts touching pieces into runs around their best
  piece under each rule, and ranks runs by their line-weighted mean. Under ratio
  0.90 it reproduces `folio query` through `benchmarks/queries/score.py` exactly:
  112 at rank 1 and MRR 0.857 on rust-book, 28 and 0.370 on ripgrep.
- *Questions.* `benchmarks/queries/rust-book-seed55.json`,
  `benchmarks/queries/ripgrep-seed58.json`, and their `-ko` and `-ko-all`
  rewrites.
- *Serving.* EmbeddingGemma 2 Q8_0 on a local `llama-server` 0.8.0 with
  `--pooling mean`.
