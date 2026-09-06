# 2026-09-06 — the public benchmarks over dividing and merging

`benchmarks/run.py`, both corpora at their pinned commits, `gte-modernbert-base-Q8_0`,
`--max-chars 8000`, 400 queries at top 10. The comparison is the 2026-09-05 run
in [public corpora](2026-09-05-public-corpora.md), which measured the same
corpora while a long section was cut and results
were one row per section.

| | 2026-09-05 | 2026-09-06 |
|---|---|---|
| `rust-lang/book` sections | 548 | 555 |
| `mdn/content` sections | 119,359 | 119,565 |
| **Sections cut at the budget** | **142** | **0** |
| MDN index | 1893 s | 1976 s |
| MDN vectors | 366.7 MB | 367.3 MB |
| MDN query | 132 ms median, 179 worst | 135 ms median, 170 worst |
| `rust-lang/book` query | 15 ms | 25 ms |
| Deprecated share of hits | 3.0% | 3.6% |
| The same, filtered | 0.0% | 0.0% |

**Nothing is cut any more.** Not one section on either corpus contains a line
longer than the budget, so every over-budget section divided at a boundary. The
873,797 characters that
[how much text truncation puts out of reach](2026-09-06-how-much-text-truncation-puts-out-of-reach.md)
found in no vector are in vectors now,
for 206 more records on MDN and 7 on `rust-lang/book` — 0.17% more rows.

**Index time moved less than the job's own spread.** 1893 s to 1976 s is 4.4%,
and four runs of the identical job in 2026-09-05 ranged 1893 s to 1991 s. This
is inside that.

**A small corpus pays for merging and a large one does not.** MDN's median query
moved 132 ms to 135 ms, within the noise of a measurement whose worst case
improved. `rust-lang/book` went 15 ms to 25 ms, and the cause is that merging
needs neighbours: a query hydrates `max(limit x 8, 32)` records rather than
`limit`, so at 555 sections it reads 32 rows where it used to read 10. On MDN
the same 32 rows disappear into a 100 ms scan.

**Contamination did not move.** 3.0% to 3.6% of the top ten, against a corpus
share of 3.57%, with a 3.0-point interval at 400 queries. Both figures say the
same thing [public corpora](2026-09-05-public-corpora.md) said: retired material
is indistinguishable from
proportional in the results, and the filter takes it to zero.
