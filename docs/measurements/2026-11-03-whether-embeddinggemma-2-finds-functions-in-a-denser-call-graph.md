# 2026-11-03 — whether EmbeddingGemma 2 finds functions in a denser call graph

The same question as
`2026-11-02-whether-embeddinggemma-2-finds-functions-from-a-question.md`, asked
of two public Rust trees whose call graphs are denser than that corpus's: does a
unit embedded alone still find the function when much of what it does lives in
the functions it calls?

**It does, and the order of the arms holds.** On both trees every EmbeddingGemma
2 arm ranks above gte, and both embedders rank far above CodeGraph's `query` and
`rg`. The absolute scores are lower than on the first corpus, and the two
gains that measurement read from the prefixes do not both carry over.

| Arm | ripgrep: top 1 | top 5 | top 10 | top 20 | MRR | minisqlite: top 1 | top 5 | top 10 | top 20 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| gte, no prefix | 20 | 50 | 63 | 72 | 0.257 | 16 | 38 | 44 | 49 | 0.320 |
| Gemma, search prefix, `title: none` | 31 | 66 | 76 | 88 | 0.375 | 23 | 52 | 55 | 61 | 0.465 |
| Gemma, code prefix, `title: none` | 27 | 70 | 78 | 93 | 0.359 | 34 | 54 | 59 | 63 | 0.562 |
| Gemma, code prefix, `title: <filename>` | 31 | 77 | 87 | 100 | 0.390 | 32 | 54 | 62 | 67 | 0.561 |
| CodeGraph `query` | 0 | 1 | 4 | 8 | 0.008 | 1 | 3 | 5 | 15 | 0.032 |
| `rg` | 5 | 13 | 20 | 28 | 0.071 | 4 | 13 | 19 | 22 | 0.107 |

Of 127 questions on ripgrep and 77 on minisqlite. These describe these query
sets and the difference between arms, not any tool's quality, for the reasons
`D-01M1VNK3CJHMPR` gives for the shape.

**What the gaps mean.** Gemma's lead over gte is 0.133 on ripgrep and 0.242 on
minisqlite, against 0.230 on the first corpus. The code-retrieval query prefix
gains 0.097 over the search-result one on minisqlite and loses 0.016 on
ripgrep. The filename title gains 0.031 on ripgrep and nothing on minisqlite,
where it still finds four more functions in the top 20. Neither prefix gain is
consistent across the three corpora. The lead over gte and over the lexical arms is.

**What it does not show.** The questions were written by Gemini 3.8 Flash for a
Google embedder, as before. The leak filter dropped 23 of 150 questions on
ripgrep and 73 of 150 on minisqlite, whose function names are database terms
(`index`, `parse`, `table`, `frame`) that a description of the function can
hardly avoid. The minisqlite set is therefore smaller and leans toward functions
whose names are not their domain's vocabulary. Graph density was compared and
not varied, so this does not say density is why the scores are lower: the trees
also differ in size, in their share of tests and in what they do.

**Method.** As in the 2026-11-02 measurement, with these differences.

- *Corpora.* `BurntSushi/ripgrep` at `91f5d22ef48b` (Unlicense) and
  `cursor/minisqlite` at `4a5c1341fdde` (no license), each fetched as a GitHub
  tarball. Only the query sets are committed, as `benchmarks/queries/ripgrep-seed58.json`
  and `benchmarks/queries/minisqlite-seed58.json`; they hold questions and
  `path:line` references, no code.
- *Graph.* CodeGraph 1.5.0 `init`: 3,923 nodes and 16,574 edges for ripgrep,
  13,753 and 65,254 for minisqlite, which is 4.2 and 4.7 edges a node against
  2.1 for the first corpus.
- *Units.* 2,905 on ripgrep and 9,769 on minisqlite, against 2,920 and 9,812
  lines an `rg` pattern for `fn` counts. Tests are 573 and 6,738 of them, and
  candidates. 1 and 32 units are cut at 8,000 characters.
- *Questions.* `generate-code.py` seed 58 with `rust=150`: 150 functions from
  1,233 eligible on ripgrep and from 2,362 on minisqlite.
- *Arms.* `code-arms.py` reads `rg` output with `--null`, and skips the line
  `rg` prints in its output when it stops reading a binary file.
- *Time.* The six arms took 10 minutes on ripgrep and 28 on minisqlite.
