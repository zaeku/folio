# 2026-09-06 — Metal against the CPU for embedding throughput

`gte-modernbert-base-Q8_0` under llama.cpp `b10809`, `--pooling cls -c 8192 -b 8192
-ub 8192`, one server at a time so they never compete for the machine, the idle
one confirmed at 0.0% CPU. One request of 32 inputs, warmed, then the median of
seven to nine trials.

| Load | Metal (default) | `-ngl 0` | Ratio |
|---|---|---|---|
| 32 sections of real MDN prose, mean 1,200 characters | **1,191 ms** | 20,790 ms | **17.5×** |
| 32 short sentences | **114 ms** | 1,822 ms | **16.0×** |

**Metal wins by more than an order of magnitude, at every input size tried.**
On the real sections that is 27 sections per second against 1.4. Indexing
`mdn/content` on the CPU would take days rather than the 33 minutes it takes now.

**This corrects a number that was used to justify looking.** An earlier reading
the same day had the CPU at 264 ms against Metal's 372 ms and suggested folio's
documented invocation was wasting a GPU that could not help a 149M-parameter
encoder. That reading was one request per arm with no warm-up, and it inverted
under replication. The figure never reached this file; it reached a board card,
which is corrected.

**27 sections per second reads low against the 60 the MDN index achieves**
(119,565 sections in 1,976 s) because these 32 inputs average 1,200 characters
while an MDN section averages about 500. The ratio between the arms is what this
entry is for; neither column is a corpus-wide rate.
