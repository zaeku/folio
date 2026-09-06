# 2026-09-04 — private corpora

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

- Where retrieval quality starts to fall as a corpus grows. Cost and
  contamination are now measured on 119,359 sections; quality is not measured
  anywhere, for the reasons `../benchmarks/README.md` gives.
- Whether a reranker over the top k earns its latency.

**The 2026-09-04 measurements have no committed artifact.** They were taken in a
session and written down here, which the rule permits and does not make
reproducible. They are the record of how the model and the design were chosen,
and they stay for that reason. Anything reported as a folio number from here on
comes from `benchmarks/run.py`.
