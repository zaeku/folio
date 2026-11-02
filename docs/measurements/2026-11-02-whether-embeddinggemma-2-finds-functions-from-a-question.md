# 2026-11-02 — whether EmbeddingGemma 2 finds functions from a question

For a natural-language question that does not name the function, which ranks
the right function highest: EmbeddingGemma 2, gte-modernbert-base, CodeGraph's
search, or `rg` on the question's words? One private corpus, one generated query
set, one seed. Symbol and call-graph questions are out of scope; CodeGraph
answers those structurally and nothing here measures that.

**EmbeddingGemma 2 finds the function; the lexical arms do not.** Behind the
model card's code-search prefixes it puts the right function first for 58 of 109
questions and in the top 20 for 102. CodeGraph's `query` and `rg` find it in the
top 20 for 19 and 27, because the questions were filtered to share no piece of
the function's name, which is what both match on.

| Arm | top 1 | top 5 | top 10 | top 20 | MRR |
|---|---|---|---|---|---|
| gte, no prefix | 32 | 63 | 80 | 88 | 0.429 |
| Gemma, search prefix, `title: none` | 47 | 84 | 96 | 102 | 0.585 |
| Gemma, code prefix, `title: none` | 53 | 86 | 93 | 102 | 0.623 |
| Gemma, code prefix, `title: <filename>` | 58 | 90 | 98 | 102 | 0.659 |
| CodeGraph `query` | 2 | 9 | 12 | 19 | 0.045 |
| `rg` | 4 | 11 | 18 | 27 | 0.070 |

Of 109 questions. MRR by language:

| Arm | Rust (45) | Python (44) | C++ (20) |
|---|---|---|---|
| gte, no prefix | 0.341 | 0.503 | 0.464 |
| Gemma, search prefix, `title: none` | 0.631 | 0.527 | 0.612 |
| Gemma, code prefix, `title: none` | 0.671 | 0.564 | 0.645 |
| Gemma, code prefix, `title: <filename>` | 0.630 | 0.659 | 0.722 |
| CodeGraph `query` | 0.051 | 0.043 | 0.037 |
| `rg` | 0.108 | 0.053 | 0.024 |

These describe this query set and the difference between arms, not any tool's
quality, for the reasons `D-01M1VNK3CJHMPR` gives for the shape. Twenty C++
questions are too few to read a language difference from.

**What the gaps mean.** The code-retrieval query prefix gains 0.038 over the
search-result one, and naming the file in the document's title gains 0.036 more.
The second is something folio cannot send: its `document_prefix` is one fixed
string. Gemma's lead over gte, 0.230 here, is larger than on rust-book prose in
`2026-11-02-what-folio-sent-prefixes-change-for-embeddinggemma-2.md`.

**What it does not show.** The questions were written by Gemini 3.8 Flash, a
Google model, for a Google embedder; `2026-10-11-what-embeddinggemma-2-changes-on-rust-book.md`
names the same risk. The leak filter is what makes the lexical arms lose, by
design: a question that says the function's name is one `rg` answers, and this
set has none. The corpus has a simple call structure, which favours a unit
embedded alone; the card's second corpus, with a more complex graph, is not
measured.

**Method.**

- *Corpus.* `daq_for_xss/daq4xss_kube` at `550c4fb`, exported with `git
  archive`, the whole tree including `archive/`.
- *Units.* `benchmarks/queries/functions.py`: every span `ast-editor inspect
  --template functions` returns, widened to the comment, attribute and decorator
  lines directly above it. 533 Rust, 338 Python and 224 C++, 1,095 in all. 25
  C++ spans whose first line has no `(` were a namespace or class returned by
  error recovery and are left out. 36 units over 8,000 characters are cut there,
  folio's own budget. Tests are candidates.
- *Questions.* `benchmarks/queries/generate-code.py`, seed 58: 60 Rust, 60 Python
  and 30 C++ functions of at least 200 characters, tests left out, in 15 calls of
  10 to `agy --model gemini-3.8-flash-low`. `gemini-3.7-flash-low`, which wrote
  the rust-book set, is no longer offered. A question containing a piece of four
  or more characters of the function's name, split on snake and camel case and
  less a few function words, is dropped: 40 were, leaving 109. The set is
  `benchmarks/queries/daq4xss-seed58.json`.
- *Arms.* `benchmarks/queries/code-arms.py`. A hit is the first ranked unit
  whose whitespace-normalised body equals the target's, so either copy of a
  duplicated function counts. Embeddings rank by dot product over every unit.
  CodeGraph 1.5.0 `query -j -l 100`, kinds `function` and `method` kept, mapped
  to the unit containing the returned start line. `rg -i -o` on the question's
  words at a word start, stopwords dropped, units ranked by distinct words
  matched and then by total matches.
- *Serving.* gte: `llama-server` 0.8.0 build 11960, `--pooling cls -c 8192 -b
  8192 -ub 8192`. Gemma: the same binary, `-hf unsloth/embeddinggemma-2-GGUF:Q8_0
  --no-mmproj --pooling mean` with the same context and batch flags. The
  prefixes are the model card's, read 2026-11-02: code search is `task: code
  retrieval | query: ` before a question and `title: {title or filename} | text: `
  before code.
