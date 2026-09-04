# Measurements

Every number folio reports about itself. `../AGENTS.md` requires a measured
fact to carry how it was measured and when, so this file is where one
goes; a number quoted anywhere else should point here.

**Environment.** Apple M1 Pro, 32 GB, macOS 26. Corpora as named in
[references.md](references.md).

**The retrieval scores below are a smoke test, not evidence.** Seven questions
were written against a private corpus after watching the tool answer, which is
the leakage the zvec-grep benchmark documentation warns about, and 7/7 against
6/7 is a difference of one question with no repeated trials and no variance
estimate. They are kept because they are what the model choice was made on, and
they are labelled so nobody quotes them as a result. `benchmarks/` is where a
reproducible number will come from.

## 2026-09-05 — public corpora

From `benchmarks/run.py`, which fetches a corpus pinned by commit and drives the
`folio` command on PATH. Reproducible by anyone: the corpus is public and the
harness is in the repository. `benchmarks/reports/` is scratch output and is not
committed; a run worth keeping is recorded here, dated, with what produced it.

`rust-lang/book` `src/` at `1500248d`, 112 files and 1.2 MB, embedded with
`gte-modernbert-base` Q8 through llama.cpp on Metal. This corpus carries no
frontmatter at all, so it measures heading splitting, ranking and cost with the
filtering path unused.

| Fact | Value |
|---|---|
| Sections | 548 from 112 files |
| Index | 40.0 s |
| Vectors | 1.7 MB at dim 768, exactly dim x 4 x sections |
| Query | 16 ms median, 17 ms worst, over 40 queries at top 10 |
| Re-index after one changed file | 0.06 s, one file re-embedded |

Two things the numbers say that the private corpora could not. Embedding cost
tracks tokens rather than files: 548 prose sections took 40 s where 116 short
records took 2.2 s, which is 3.8x the time per section for sections carrying
book paragraphs rather than one fact each. And the flat-matrix invariant that
`no-ann-index` asserts on a fixture holds on a real corpus, which the report
checks on every run rather than taking on faith.

`mdn/content` is the second corpus and its numbers, including the contamination
rate, are not in yet.

## 2026-09-04 — private corpora

Re-check a fact before building on it.

| Fact | Value |
|---|---|
| Best local embedding model tested | `gte-modernbert-base` Q8, 768 dim, 8192 tokens, CLS pooling, at 2.2 s to index against 7.2 s for `Qwen3-Embedding-0.6B`, a 768 dim index against 1024, and a 300 MB model against 610 MB. On the smoke test both placed the answering document in the top three for all seven questions; gte ranked six of them first and Qwen3 five, which is one question and not a result |
| Pooling is per model and the default is wrong for both | `--pooling cls` for gte-modernbert, `--pooling last` for the Qwen3-Embedding family |
| An encoder needs its whole input in one physical batch | `-b` and `-ub` must both be at least your longest section. gte-modernbert returned HTTP 500 on a 1106 token section at the default 512, with an explicit message. A decoder does not: `Qwen3-Embedding-0.6B` processed a 1068 token section at the same setting, chunking the prefill, so no earlier measurement here was silently truncated |
| GGUF conversion quality varies by publisher | `keisuke-miyako/gte-modernbert-base-gguf` loads. `cstr/gte-modernbert-base-GGUF` does not: `key not found in model: bert.context_length`. Test a conversion before trusting it |
| Gain from `Qwen3-Embedding-4B` over `0.6B` | None. Same 6/7 on the retrieval set, same rank on the hard query, for 6.7x the parameters and 2.5x the index |
| Why the 4B gains nothing | Its extra parameters are MLP capacity. 0.6B spends 26% of itself on the 151.7k vocab table, but only ~440M parameters do the retrieval work and those were trained for it |
| `gte-modernbert-base` runs on Metal, through llama.cpp | 2.2 s to index 116 files, against 8.8 s for the same server at `-ngl 0`: a 4.0x GPU speedup. Its ONNX q4 path is the thing that fails, on WebGPU (`MatMulNBits` dimension mismatch), falling back to CPU at 5m10s. The runtime was the variable, never the model |
| Result granularity | The heading section, always. Indexing the same corpus at context 8192, 2048 and 256 returned the identical range for the same query |
| What actually drives pointer precision | Section size, not model choice. In one corpus a file with 22 `###` headings returned 12–16 line ranges while a file with 2 returned a 133 line range |
| Long documents | Not a truncation risk. A 10,275 byte section reported zero truncated fragments even at context 256, because the indexer chunks before embedding |
| Brute-force cosine cost | 348 sections, microseconds. 35,000, ~10 ms. 100,000, 410 MB at f32 and ~30 ms |
| folio against zvec-grep, same corpus and model | 7.2 s against 19 s to index, and 45 ms to re-index one changed file. The one question zvec-grep placed outside the top three is the synonym bridge above; it is the observation the lexical-route decision rests on, and one question is not a score |
| `mlxcel` 0.6.0 | Serves no embeddings. `/v1/embeddings`, `/embeddings` and `/embedding` all return 404, and `mlxcel arch` lists no encoder family. Given an embedding checkpoint it loads it as a causal LM |

Two things are unmeasured. Do not assume either.

- Where retrieval quality starts to fall as a corpus grows. Every number above
  comes from corpora of 116 and 5 files.
- Whether a reranker over the top k earns its latency.

**The 2026-09-04 measurements have no committed artifact.** They were taken in a
session and written down here, which the rule permits and does not make
reproducible. They are the record of how the model and the design were chosen,
and they stay for that reason. Anything reported as a folio number from here on
comes from `benchmarks/run.py`.
