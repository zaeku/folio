# 2026-09-07 — what changes when TEI serves the endpoint

Hugging Face's `text-embeddings-inference` in place of `llama-server`, behind the
same `/v1/embeddings`. It is Rust on Candle, it lists ModernBERT and Alibaba GTE
among its architectures, and it reads safetensors rather than GGUF, so
`Alibaba-NLP/gte-modernbert-base` loads directly and the conversion hazard in
[private corpora](2026-09-04-private-corpora.md) does not arise.

**folio needed no change, which is what the HTTP boundary is for.** `index`,
`query` and `status` all worked against it unaltered. Its `/v1/embeddings` takes
`input` and `model`, and the `default` folio sends for a model name is accepted
rather than validated against the model actually served.

**The fingerprint held across the swap.** An index built by `llama-server` on
`gte-modernbert-base-Q8_0` answered from TEI with nothing re-embedded. The
similarities, vectors normalized so each figure is a dot product:

| Probe | llama.cpp `Q8_0` against TEI Candle `float16` |
|---|---|
| The sentence folio ships as its fingerprint | 0.999823 |
| A Korean sentence | 0.999469 |
| A line of Rust | 0.999849 |
| A sentence from `mdn/content` | 0.999845 |

This is a third implementation after the two in
[whether the runtime changes an embedding](2026-09-06-whether-the-runtime-changes-an-embedding.md),
and a stiffer case than either: it changes the framework, the kernels and the
precision at once, `Q8_0` against `float16`. It still clears 0.999 on every
probe.

**It also answers, on that stiffer case, the question the fingerprint's own
weakness raises.** The Korean probe reads 0.999469 where the English one reads
0.999823, so the sharper probe keeps about half the margin — and it keeps it.
The concern recorded in
[what moves an endpoint's fingerprint](2026-09-06-what-moves-an-endpoints-fingerprint.md),
that a corpus in another language moves more than the English fingerprint
reports, is measured here at 3.5e-4 of margin rather than at a threshold
crossing.

**One default breaks folio silently, and it is the only thing in this swap that
does.** TEI's `--auto-truncate` is `true` out of the box. A 40,009-character
section is 12,683 tokens and the model's maximum input is 8,192, so it does not
fit either server:

- `llama-server` answers HTTP 500 and names the token count. folio's budget
  calibration sees the refusal, halves the budget to 20,000 characters, says so,
  and divides the section into three records.
- TEI answers 200 with a vector of a prefix. The full text and the full text
  with a sentence appended come back **identical to 1.00000000**, so the tail is
  in no vector, and the index records `truncated = 0`. folio's calibration keeps
  the budget it was given, because nothing refused anything.

That is the failure folio's truncation accounting exists to prevent, arriving
where folio cannot see it. `--auto-truncate false` restores the refusal exactly:
the same corpus then reports the same lowered budget and the same three records
as `llama-server`. Disabling it requires `--max-batch-tokens` to be at least the
model's maximum input length, or the server refuses to start and says why.

**The batching rule is not the same rule.** `llama-server` needs `-b` and `-ub`
at least as large as the longest section, and a batch that does not fit is an
HTTP 500. TEI splits one client request across scheduling batches: 40 sections
of about 7,500 characters, sent 32 to a request and so roughly 60,000 tokens at
once, embedded with `--max-batch-tokens 8192`. What is hard there is per input,
`max_input_length`, and the count of inputs, `max_client_batch_size` — whose
default of 32 is exactly folio's batch size, with nothing to spare.

**Pooling came out right without being configured, but not for the documented
reason.** TEI logged `Could not find a Sentence Transformers config` for this
model and reported `pooling: cls`, which is the correct choice — as its own
fallback rather than as a reading of the model's configuration. A model wanting
`last`, as the Qwen3-Embedding family does, would still need to be told.

**So the recipe has two flags that are not optional**, the same shape as
`llama-server`'s and one of them more dangerous, because its failure is a
silent prefix rather than an error:

```sh
text-embeddings-router --model-id Alibaba-NLP/gte-modernbert-base --port 8081 \
  --auto-truncate false --max-batch-tokens 8192
```

**Method.** `text-embeddings-router` 1.9.3 from Homebrew, which builds with
`-F metal` on Apple Silicon; the log confirms `Starting ModernBert model on
Metal`. `llama-server` b10809-5266f24da on port 8080 with the flags
`README.md` prescribes, TEI on 8081. Fixtures were real `mdn/content` section
text with heading and fence lines removed, so that one file is one section of a
chosen length. The truncation proof appended a sentence about canaries, which
appears in neither corpus.

The one part of folio the swap does touch is `folio unit`, which prints a
service file for `llama-server` whatever is running.
