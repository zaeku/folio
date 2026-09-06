# 2026-09-06 — whether splitting a cut section finds its tail

The measurement `../benchmarks/README.md` allows: retrievability, on queries the
corpus labels by construction, comparing folio with folio. `mdn/content` at the
pinned commit, `gte-modernbert-base-Q8_0`, `--max-chars 8000`, top 10.

**Method.** Of MDN's 142 cut sections, 54 have a prose sentence in the text past
the budget; the other 88 are tables, lists or markup and yield no usable query.
Each sentence is used verbatim as a query, and a hit counts only when a returned
range actually contains that sentence. The split arm is a copy of the corpus
where every cut section has a deeper heading inserted at paragraph breaks, so
the tail becomes its own record; the index was re-embedded incrementally, 107
files.

| Tails | prefix | split |
|---|---|---|
| The split embedded them (12) | 3 found, mean rank 6.0 | **6 found, mean rank 1.5** |
| The split did not reach them (42) | 3 found, mean rank 3.0 | 3 found, mean rank 3.3 |

**Where it applies it works, and it applies to a fifth of the cases.** Giving a
tail its own vector doubled the hit rate and moved the mean rank from sixth to
between first and second. The second row is the control: nothing changed for the
tails the split did not reach, which is what it should say.

**The proposed boundary rule is the finding.** D-01M1T9G1416ACW asks for a
paragraph break inside the budget, and on this corpus that reached 12 of 54
tails. MDN's long sections are tables and lists with no blank lines to break at,
so 42 tails stayed past a budget even after splitting, and the split corpus
still reported 148 cut sections against the original's 142. A rule that leaves
four fifths of the cases untouched needs a fallback: a line boundary, and past
that any boundary at all, because a piece cut mid-sentence still has a vector
and a truncated tail has none.

**Sample.** 54 queries, 12 in the subset that carries the effect. Directional,
not conclusive: three hits against six. The queries share wording with the text
they were drawn from, so the level is inflated in both arms and only the
difference is readable — which is the boundary `../benchmarks/README.md` sets.
