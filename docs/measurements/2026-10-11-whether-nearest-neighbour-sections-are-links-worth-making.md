# 2026-10-11 — whether nearest-neighbour sections are links worth making

A wiki's links are only as complete as whoever writes them can remember the rest
of the corpus. The question this answers is whether folio's index, which already
holds one vector per section, can propose those links without anyone holding the
whole corpus in mind — and how its proposals compare with the usual route of
extracting entities and relations and joining sections through them.

**Corpus.** `rust-lang/book`, `src/`, at the commit `benchmarks/run.py` pins:
112 files, 555 sections, indexed by folio 0.6.1 on a local `llama-server`
serving `gte-modernbert-base` Q8_0, at the default 8,000-character budget.

**Graphs.** Each joins every section to its five best-scoring other sections.

- *folio k-NN*: the dot product of the stored vectors. Nothing is embedded.
- *LLM entity*: `gemini-3.7-flash-low` through `agy` 1.3.3 read every section
  in 32 batches and named the entities each discusses and up to eight triples
  between them. Sections are joined by tf-idf cosine over shared entity names,
  keeping names in 2 to 20% of sections. 765 distinct entities, 1,931 triples,
  18 sections with no usable entity. 0.76 M tokens, 4 to 5 minutes at four
  batches in parallel.
- *LLM relation*: the same, over the triples rather than the entity lists.
- *heuristic entity*: the same, over backtick spans and CamelCase words, as a
  floor that costs nothing.

**Reference.** The 256 file pairs the book's authors linked to one another. An
edge counts as found when its two sections' files are such a pair.

## Against the authors' links

| Graph | Precision | Recall | Same-file edges | Cross-chapter edges | Max in-degree |
|---|---|---|---|---|---|
| folio k-NN | 0.23 | 0.41 | 53% | 24% | 38 |
| LLM entity | 0.16 | 0.32 | 43% | 40% | 13 |
| LLM relation | 0.15 | 0.28 | 44% | 38% | 16 |
| heuristic entity | 0.14 | 0.29 | 32% | 48% | 18 |
| folio ∪ LLM entity | 0.17 | 0.52 | 42% | 36% | 40 |

The folio and LLM entity graphs share a fifth of their edges, Jaccard 0.20. Of
the authors' pairs, 52 are found by folio alone, 28 by the LLM alone, and 54 by
both. At ten neighbours the order holds: folio recalls 0.57, the LLM entity
graph 0.40, their union 0.68.

**Extraction by a model barely beats extraction by regex.** Recall 0.32 against
0.29. What shapes the graph is joining sections through shared names, not how
well the names were found. The triples add nothing as edges: 1,057 distinct
predicates over 1,931 triples is nearly one predicate per triple.

## Judged blind

Precision against the authors' links counts every pair they did not link as
wrong. To see what those pairs are, 208 of them were scored 0 to 2 by
`gemini-3.7-flash-medium` through `agy` 1.3.3, ten per call, shuffled, with
nothing to say which graph proposed them. Each section was shown up to 3,000
characters. 2 means an editor should add the link, 1 that a link would be
marginal, 0 that the two are unrelated or share only a word. 0.51 M tokens.

| Group | n | 2 | 1 | 0 | Mean |
|---|---|---|---|---|---|
| control: pairs the authors linked | 30 | 63% | 20% | 17% | 1.47 |
| folio k-NN, not linked by the authors | 100 | 45% | 38% | 17% | 1.28 |
| LLM entity only, not linked by the authors | 50 | 26% | 36% | 38% | 0.88 |
| control: random cross-file pairs | 28 | 4% | 18% | 79% | 0.25 |

The controls bracket the judge: it rejects random pairs and is strict even with
the authors' own links.

**A folio proposal the authors did not make is rejected no more often than one
they did.** Both are scored 0 at 17%. Weighting the two groups by their share of
folio's cross-file edges puts about half of all its proposals at 2 and over four
fifths at 1 or better.

**Its rejections are hubs.** The pairs scored 0 join a section to a chapter
summary, an introduction, or the keyword appendix: broad, shallow sections that
lie near everything. The folio graph's highest in-degree is `SUMMARY.md`, 38
against the LLM entity graph's 13. Many of the pairs scored 2 are consecutive
steps of one worked example — the `minigrep` refactors in chapter 12 — which a
"next section" link may already cover.

## What this does not show

One corpus, written by authors who already link carefully, which is kinder than a
wiki. One judge, one pass, no person checking a sample of its scores. Entity
names were lowercased and not otherwise merged.

## Reproducing it

The harness is `benchmarks/links/`, kept apart from `run.py` because a judged
score is the kind of number `D-01M1VNK3CJHMPR` refuses; whether this one falls
under that refusal is the open proposal `D-01M4KE7EN1MV2J`. The model outputs
behind the tables above are not committed. Replayed through that harness
against a fresh index of the same commit, every figure above came out the same
except the heuristic graph's highest in-degree, 17 rather than 18, where two
neighbours tie.
