# Link proposals

Whether the nearest neighbours of a section, by folio's stored vectors, are
links worth adding to a wiki — compared with joining sections through entities a
model extracted, and judged blind by a model.

This harness stands apart from `run.py` on purpose. Its judged score is
preference rather than cost or contamination, which `../README.md` declines to
measure, and whether it falls under that refusal is an open proposal in the
decision layer, `D-01M4KE7EN1MV2J`. Until that is settled it feeds nothing:
not the table in `../../README.md`, and not any claim about folio's ranking
quality.

## Running it

```sh
cargo install --path ../..
export FOLIO_ENDPOINT=http://127.0.0.1:8080/v1/embeddings   # as in ../README.md

./extract.py     # fetch and index rust-lang/book, then one agy run per batch
./compare.py     # each graph against the authors' links; ./compare.py 10 for ten neighbours
./judge.py       # sample pairs and have a model score them blind
./verdict.py     # tally the scores by group
```

It needs `python3` with `numpy`, the `folio` command, and a signed-in `agy`.
The corpus is the `rust-book` entry of `run.py`, fetched into `../corpora/` and
reused if already there. Everything written goes to `../reports/links/`, which
is not committed. A run worth keeping becomes a file under `../../docs/measurements/`, by a person.

`extract.py` and `judge.py` keep any batch already written and run only the
missing ones, so an interrupted run resumes and a rerun after deleting one file
redoes only that batch.

## What a rerun reproduces

`compare.py` and `verdict.py` are deterministic over what the two model steps
left. The model steps are not: a second extraction names different entities,
and a second judgement scores differently. So a rerun reproduces the method,
not the figures. To reproduce the figures, replay the recorded outputs —
`extract/` and `judge/` under `../reports/links/` — rather than calling the
model again.

Results are keyed by a section's file and first line, never by the index's slot
number, because two indexes of the same corpus number their sections
differently. A replay against a fresh index of the same commit therefore lands on
the same sections. Exact ties between neighbours may fall the other way, which
moves an in-degree by one.

The judged sample is drawn with a fixed seed from the graphs, so it is the same
sample only when the extraction is the same extraction.

## What it is not

One corpus, chosen because its authors link carefully, which is kinder than a
wiki. One judge per pair and one pass, with no person checking a sample of its
scores. The authors' links are a lower bound on what is worth linking, not the
set of it — which is why the judge is there at all.
