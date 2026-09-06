# 2026-09-05 — characters are not tokens, and the endpoint itself

Measured with `gte-modernbert-base-Q8_0` under `llama-server`, `--pooling cls`,
`-c 8192 -b 8192 -ub 8192`, through its `/tokenize` endpoint and through
`folio doctor`.

**A character budget is a token budget only for the language it was set on.**
folio budgets characters because it cannot see the model's tokenizer, and
`--max-chars 8000` was set below an 8,192-token context on English.

| Text | Chars per token |
|---|---|
| `"section "` repeated | 7.98 |
| One English sentence repeated | 6.10 |
| English prose, this project's README | 3.66 |
| Korean prose | 0.66 |

At 0.66, 8,192 tokens is about 5,400 characters, so the default budget sends a
Korean section that this server refuses. Measured on a 8,011-character Korean
fixture, `folio index` now finds 4,000 by halving: one refusal, one acceptance,
and the run proceeds with what it cut recorded. That is the number a rule of
thumb would have had to guess.

That spread is also why `folio doctor` probes with the corpus's own longest
section. Synthetic filler tokenizes at nearly twice the rate of real prose:
40,000 characters of `"section "` passed a batch that 18,007 tokens of repeated
English prose overflowed, and the server said so — `input (18007 tokens) is too
large to process. increase the physical batch size (current batch size: 8192)`.

**What `folio doctor` reports on this machine.** 768 dimensions. One input costs
142 ms cold and 9-10 ms warm over five runs, which agrees with the 8 ms round
trip in [what a query reads](2026-09-05-what-a-query-reads.md). Its similarity probe — a paraphrase pair against an
unrelated sentence — gives 0.900 and 0.352.

**The server grows to the largest input it has been asked to embed, and keeps
it.** Measured 2026-09-06 with a second server on port 8090, same model and
flags (`--pooling cls -c 8192 -b 8192 -ub 8192`), one input at a time:

| Largest input served so far | `phys_footprint` |
|---|---|
| None; just loaded | 69 MB |
| A short sentence | 55 MB |
| 8,000 characters, about 2,000 tokens | 330 MB |
| 44,000 characters, about 8,000 tokens | 1,899 MB |

A short input after the 8,000-token one leaves it at 1,899 MB, and ten seconds
of idling does not lower it. So `-ub 8192` sets a ceiling and costs nothing by
itself; what is paid is decided by the longest thing folio actually sends. The
model file is 153 MB, and the growth is faster than linear — four times the
tokens for 5.75 times the memory — which is the shape of an attention buffer.

**A reading of 2,305 MB, taken the same day from the server this project runs
against, was self-inflicted.** That server had been sent an 18,007-token probe
by `folio doctor --max-chars 120000`. The request failed with a 500 and the
reservation stayed. For folio's default 8,000-character budget on English prose,
which is about 2,200 tokens, the figure to expect is the 330 MB row.

An earlier entry read 488 MB and 315 MB from `ps` RSS on that same server and
called them the cost of running it. RSS counts resident pages, so it reports how
much has not been paged out rather than what the process holds; `phys_footprint`
is the figure that does not move with memory pressure, and it is what Activity
Monitor shows.

That also means a `folio doctor` run with a `--max-chars` larger than the budget
you index with inflates the server past anything indexing would ask of it, until
the server is restarted.

**Waking it costs 231 ms.** The first `folio doctor` probe after several idle
hours took 231 ms for one input against 9-10 ms warm. That is the swapped-out
2.1 GB being read back, so the idle cost does not disappear — it moves from
memory into the latency of the next query.
