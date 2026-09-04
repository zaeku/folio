# Measurements

Every number folio reports about itself. Hard rule 10 in `../AGENTS.md` requires
a measured fact to carry how it was measured and when, so this file is where one
goes; a number quoted anywhere else should point here.

**Environment.** Apple M1 Pro, 32 GB, macOS 26. Corpora as named in
[references.md](references.md).

**Retrieval set.** Seven questions with a known answering document, scored on
whether that document appears in the top three. The questions are conceptual
rather than lexical — they deliberately avoid the words the target documents
use — because that is the case `rg` already fails.

## 2026-09-04

Re-check a fact before building on it.

| Fact | Value |
|---|---|
| Best local embedding model tested | `gte-modernbert-base` Q8, 768 dim, 8192 tokens, CLS pooling. 7/7 top-3 with 6 ranked first, against 7/7 with 5 first for `Qwen3-Embedding-0.6B`, at 2.2 s to index against 7.2 s, a 768 dim index against 1024, and a 300 MB model against 610 MB |
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
| folio against zvec-grep, same corpus and model | 7/7 against 6/7 top-3; 7.2 s against 19 s to index; 45 ms to re-index one changed file |
| `mlxcel` 0.6.0 | Serves no embeddings. `/v1/embeddings`, `/embeddings` and `/embedding` all return 404, and `mlxcel arch` lists no encoder family. Given an embedding checkpoint it loads it as a causal LM |

Two things are unmeasured. Do not assume either.

- Where retrieval quality starts to fall as a corpus grows. Every number above
  comes from corpora of 116 and 5 files.
- Whether a reranker over the top k earns its latency.

**These measurements have no committed artifact yet.** They were taken in a
session and written down here, which rule 10 permits and does not make
reproducible. Anything reported as a folio number from here on should come from
a script in the repository.
