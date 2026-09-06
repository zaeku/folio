# 2026-09-06 — how the name-based rule mixed two models in one index

The vectors two weight sets return were compared with a stub `/v1/embeddings`
that answers deterministically from a named weight set, because a second
768-dimension model was not on the machine and the question did not need one.

**What the old rule let through.** With identity recorded as the model and
endpoint strings, the stub was restarted on other weights at the same URL and
under the same `--model`. One file was edited and re-indexed. The result: slot 0
from the first weights, slot 2 from the second, in one matrix, ranked against
each other, with `folio status` reporting a single identity and no command
saying anything. Dimension was checked and matched, which is why nothing fired.

**What it costs to ask.** One request of 17 ms, and only when a run is about to
embed. A no-op re-index of `mdn/content` is 0.29 s against the 0.27 s it was.
A query pays nothing: folio batches 32 inputs and a query sends one, so the
fingerprint rides in the request the query was making anyway.
