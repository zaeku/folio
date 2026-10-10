# 2026-10-11 — what EmbeddingGemma 2 changes on rust-book

`google/embeddinggemma-2` against `gte-modernbert-base`, both Q8_0 GGUF behind
`llama-server`, on the pinned `rust-lang/book` at `1500248d8f23`. One corpus,
one generated query set, one seed.

**EmbeddingGemma 2 ranks better and folio hides it.** Ranked section by section
outside folio, it answers more queries at rank 1 than gte does, and with the
model card's retrieval prefixes it answers far more. Through `folio query` it
answers fewer, because the merge rule's ratio was read off gte's score scale.

| Arm | top 1 | top 5 | top 20 | MRR |
|---|---|---|---|---|
| gte, through folio | 94 | 128 | 144 | 0.750 |
| Gemma, through folio | 67 | 119 | 147 | 0.607 |
| gte, sections, no prefix | 96 | 129 | 144 | 0.762 |
| gte, sections, prefixed | 99 | 131 | 144 | 0.770 |
| Gemma, sections, no prefix | 91 | 130 | 145 | 0.732 |
| Gemma, sections, prefixed | 119 | 143 | 147 | 0.875 |

Of 147 queries. These describe this query set and the difference between arms,
not folio, for the reasons `D-01M1VNK3CJHMPR` gives for the shape.

**Why folio costs Gemma 0.125 of MRR and gte almost none.** A piece joins a
merged run while its score is at least 0.9 of the run's best. Gemma's scores sit
close together — the three best sections for one query scored 0.848, 0.817 and
0.815 — so nearly every touching neighbour passes, runs grow to whole files, and
the line-weighted mean dilutes the answer. `ch03-03-how-functions-work.md` came
back as lines 1-253 at rank 4 where the section starting at line 1 ranked first
on its own. The stored vectors, the query vector, batch position and the
fingerprint riding in the query's request were each ruled out: every one
reproduced at cosine 1.0 for both models.

**The prefixes are the model card's.** Query `task: search result | query: `,
document `title: none | text: `. folio sends neither, so the prefixed arms embed
every section outside folio. gte gains 0.008 from the same strings, which is
noise; Gemma gains 0.143.

**Cost.** From `benchmarks/run.py rust-book`, 555 sections at dim 768 for both:

| | gte | Gemma |
|---|---|---|
| Index | 41.3 s | 75.0 s |
| Query median | 32 ms | 22 ms |
| Query worst | 74 ms | 43 ms |
| Longest section | 2697 tokens | 2713 tokens |
| Characters per token | 3.70 | 3.81 |

No section passes 8192 tokens under either tokenizer, so the 8000 character
`--max-chars` truncates nothing for Gemma that it keeps for gte.

**Method.**

- *Query set.* 150 sections of at least 400 characters drawn from folio's index
  with `random.Random(55)`, sent to `agy --model gemini-3.7-flash-low` in 15
  calls of 10, each asked for one 5 to 15 word query a learner would type
  without having read the section, copying no five-word phrase and not quoting
  the heading. 3 that copied five words anyway were dropped, leaving 147.
  Queries share the generating model's habits; batching and the random draw add
  some variety and do not remove that.
- *Through folio.* `folio query <q> -l 20 --paths-only --no-refresh` from the
  corpus root. A hit is a returned range containing the section's first line.
- *Sections.* Every section's text embedded directly, ranked by dot product,
  hit when the exact section is in the top 20. No merging, so these arms read
  the model and not what a caller receives.
- *Serving.* gte: `llama-server` 0.6.0 build 11429, `--pooling cls -c 8192 -b
  8192 -ub 8192`. Gemma: `unsloth/embeddinggemma-2-GGUF:Q8_0`, `--no-mmproj`
  so only the text encoder loads, `--pooling mean` with the same context and
  batch flags, on `llama-server` built from `ggml-org/llama.cpp` at
  `aa94f2086`. 0.6.0 refuses the file with `unknown model architecture:
  'gemma-embedding2'`; support arrived in ggml-org/llama.cpp#30054, merged
  2026-10-06.

The scripts and the 147 queries are `benchmarks/queries/`.
