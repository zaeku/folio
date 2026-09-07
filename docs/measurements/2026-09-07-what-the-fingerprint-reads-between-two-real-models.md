# 2026-09-07 — what the fingerprint reads between two real models

[What moves an endpoint's fingerprint](2026-09-06-what-moves-an-endpoints-fingerprint.md)
records 0.013 for "different weights entirely", and its method note says that
row is a stub endpoint returning deterministic pseudo-random vectors rather than
a model. So the number folio's whole model-swap defence rests on had never been
read between two models. Dimension cannot stand in for it: `D-01M1PP6HKHF97G`
names gte-modernbert-base, bge-base, e5-base and all-mpnet-base-v2 as sharing
768.

Three servers at once, `Q8_0` throughout, each model under its own pooling.
Vectors are normalized, so every figure is a dot product.

| Probe | gte-modernbert against bge-base | gte-modernbert against e5-base | bge-base against e5-base |
|---|---|---|---|
| folio's own, English | 0.058674 | 0.056492 | 0.560277 |
| Korean | -0.040856 | -0.045724 | 0.516274 |
| A line of Rust | -0.016926 | -0.039711 | 0.579297 |

**No pair comes near the threshold, and the margin is not close.** The nearest
any two models read is 0.579297, which is 0.419703 below the 0.999 line — about
sixty times the 0.006789 that separates the nearest discarded quantization from
it, and six hundred times the 0.000631 the nearest kept case has to spare. A
model swap is not a borderline case for this probe and no probe would make it
one.

**But 0.013 understates how close two real models get.** bge-base-en-v1.5 and
e5-base-v2 read 0.52 to 0.58 against each other, forty times the stub's figure,
because both are BERT-base retrieval models fine-tuned from related starting
points. gte-modernbert-base against either is 0.06 or below, so the stub is a
fair stand-in for an unrelated model and not for a sibling. Nothing in the
2026-09-06 table changes; the impression that any two models are seven orders
from the line does.

**A sharper probe does not help here, which is the answer this run was for.**
For every cross-model pair the Korean and Rust probes read *lower* than the
English one, not higher: the separation this check depends on is already four
hundred times the margin it has to preserve, and a more sensitive fingerprint
can only move the cases the threshold keeps. Whatever the case for a
multi-text fingerprint is, catching a model swap the English sentence misses is
not it, because no measured swap is within reach of the line on any probe.

**Method.** `llama-server` b10809-5266f24da, three processes: gte-modernbert-base
`Q8_0` with `--pooling cls` on port 8080, `CompendiumLabs/bge-base-en-v1.5-gguf`
at `q8_0` with `--pooling cls` on 8081, `ChristianAzinn/e5-base-v2-gguf` at
`Q8_0` with `--pooling mean` on 8082. Each model's own pooling, because a wrong
one produces a degraded vector and would exaggerate the very separation this
run was checking. Every embedding came back at dimension 768.

The probe texts were sent raw, with no instruction prefix, because that is what
folio sends. e5 expects `query:` and `passage:` prefixes and this measurement
gives it neither, so its vectors are off the distribution it was trained for —
which is the state folio would find it in. Adding the prefix would move e5's
vectors and cannot carry 0.06 to 0.999.

The Korean and Rust probes are recorded here rather than left to a description:
`결정이 살아 있는지는 그 문서가 아니라 울타리가 답한다.` and
`let near = same_space(&vectors[at], &o.meta.fingerprint_vec);`. The English one
is the sentence folio ships.
