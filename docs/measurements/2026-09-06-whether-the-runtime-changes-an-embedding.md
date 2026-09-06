# 2026-09-06 — whether the runtime changes an embedding once the backend is controlled

`Alibaba-NLP/gte-modernbert-base` at revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`,
CLS pooling, which `1_Pooling/config.json` in that repository sets as the model's
own. Four probes: English prose, Korean prose, a line of code, and 7,752
characters of English that tokenize to 1,371 tokens. Vectors normalized, so
every figure is a dot product, and the column reported is the worst probe.

**The reference** is the same weights read directly from `model.safetensors` by
transformers 5.16.1 on torch 2.14.0, fp32 compute on CPU. Every other row is one
runtime measured against it, rather than runtimes against each other, so a cell
that is wrong for its own reasons shows as one row and not a smear.

| Cell | Worst probe | Distance from reference |
|---|---|---|
| fastembed-rs 5.17.4, ONNX Runtime, CPU, fp32 | 0.99999985 | **2.9e-7** |
| llama.cpp `b10809`, F16 GGUF, CPU (`-ngl 0`) | 0.99999438 | **5.6e-6** |
| llama.cpp `b10809`, F16 GGUF, Metal | 0.99999316 | **6.8e-6** |
| llama.cpp `b10809`, Q8_0 GGUF, Metal | 0.99936703 | 6.3e-4 |

**The runtime is a performance choice.** ONNX Runtime reproduces PyTorch to 3e-7,
which is float32 accumulation noise, and llama.cpp's own C++ kernels over a GGUF
conversion reproduce it to 6e-6. Two implementations sharing no code agree four
orders of magnitude inside the 0.04 that separates adjacent results in a ranked
answer. Control the weights and the precision and the runtime does not move a
vector enough to matter.

**Quantization dominates it by two to five orders of magnitude.** Q8_0 costs
6.3e-4 against the reference and Q4_k_m costs 0.024
([what moves an endpoint's fingerprint](2026-09-06-what-moves-an-endpoints-fingerprint.md))
— 100× and
80,000× the runtime effect. What a vector space is made of is the weights and
their precision, not the program that multiplies them.

**This corrects the backend figure in
[what moves an endpoint's fingerprint](2026-09-06-what-moves-an-endpoints-fingerprint.md).**
That measurement took
Metal against CPU on Q8_0 and found 1.2e-4. On F16 the same comparison is 1.5e-6,
eighty times smaller. So 1.2e-4 was not the cost of the backend; it was the cost
of two dequantization paths for Q8_0 disagreeing. The backend alone is nearly
free, and it is quantization that makes it visible. The earlier number stands for
the configuration folio ships, which is Q8_0.

**The risk is configuration, not arithmetic.** Two traps fired while measuring
this, and each would have been reported as a runtime effect by anyone comparing
endpoints without a reference:

- fastembed-rs defaults a user-defined model to 512 tokens whatever the model
  declares, and this model declares 8,192. The long probe was silently truncated
  and the cell read 0.979 — squarely in "this reorders answers". With
  `with_max_length(8192)` it reads 0.99999996. A folio section at the
  8,000-character budget would have lost most of itself in silence.
- llama.cpp's `/tokenize` reports without the `[CLS]`/`[SEP]` pair, so its token
  ids differ from the tokenizer's on every input. The tokenization is identical:
  `reference[1:-1] == llama.cpp` on every probe, and the embeddings agree to 6e-6,
  which they could not if the special tokens were missing from the forward pass.

Both were found by checks placed before the arithmetic was believed: a probe long
enough to hit a context limit, and a token-id comparison. Neither would have been
visible in the vectors alone.

**Korean is the worst probe in every llama.cpp row**, as it was for the
fingerprint measurement. It is the most sensitive of the four to anything that
perturbs the arithmetic.

**Not measured.** The Metal column has one runtime. vLLM lists ModernBERT in
neither its supported-models nor its pooling tables, and its documented route is
the Transformers backend, which would make the cell a wrapper around the
reference rather than an independent runtime. vllm-mlx names
`mlx-community/ModernBERT-base-mlx`, the base masked-LM model rather than this
embedding fine-tune, and documents no pooling mode. rvLLM was withdrawn from
publication before it could be tested.

**Method.** A scratch uv workspace outside the repository, one probe file read by
every cell, one script per cell writing the same JSON shape, and a comparator
that does not know which runtime produced a row. fastembed-rs is a small Rust
binary rather than a server, which is fine for a measurement and does not touch
folio's endpoint boundary.
