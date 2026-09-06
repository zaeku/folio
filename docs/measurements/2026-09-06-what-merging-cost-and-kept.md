# 2026-09-06 — what merging cost and kept

Measured on a six-document fixture, `gte-modernbert-base-Q8_0`, before and after
adjacent ranked sections were returned as one row (D-01M1TCTX5HN2E7).

| Query | Before | After |
|---|---|---|
| Six sibling sections, one subject | five rows of one file, 0.780 to 0.741 | one row, `facilities.md:3-25`, 0.763 |
| A distinctive tail beside background | `operations.md:5-7`, 0.766 | unchanged at 0.766 |
| The same, joining on contiguity alone | — | `operations.md:1-7`, 0.618 |

The third row is why joining requires more than adjacency. Contiguity alone
folded the answering section into the background beside it, and its lead over a
document that does not contain the answer fell from 0.199 to 0.051. Requiring
the pieces to rank within a tenth of the run's best leaves the tight pointer
alone and still folds the siblings.

**Every score recorded before this date predates merging.** The same query
returns a different number afterwards wherever its rows touch, so the figures in
every measurement before it is a reading of the ranking rather than of what a
caller sees.
