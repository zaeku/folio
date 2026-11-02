# Generated queries

A query set drawn from random sections of the pinned `rust-lang/book` and
written by a model, for comparing two configurations of folio — two models, two
merge settings — on the same questions. Its scores describe this set and the
difference between arms, never folio's ranking quality, as `D-01M1VNK3CJHMPR`
bounds that shape. It feeds nothing in `../../README.md`.

## Running it

```sh
export FOLIO_ENDPOINT=http://127.0.0.1:8080/v1/embeddings   # as in ../README.md

./generate.py ../reports/queries          # 150 sections, 15 agy calls of 10, seed 55
./score.py ../corpora/rust-book rust-book-seed55.json        # through `folio query`
./sections.py "$FOLIO_ENDPOINT" rust-book-seed55.json        # section by section, with and without prefixes
```

The corpus and its index are the ones `../links/` prepares and reads. Index it
with the model under test first: `score.py` asks whatever index is there.
`rust-book-seed55.json` is the set the 2026-10-11 measurement used;
`generate.py` writes a new one rather than reproducing it, because the model's
output is not deterministic.

## Functions

A second set asks for functions rather than sections, on a private corpus the
person running it has checked out. `daq4xss-seed58.json` is the set the
2026-11-02 measurement used.

```sh
git -C <checkout> archive 550c4fb | tar -x -C <export>
(cd <export> && codegraph init .)

./functions.py <export> > units.json
./generate-code.py <export> units.json ../reports/code-queries     # 60 Rust, 60 Python, 30 C++, seed 58
./code-arms.py <export> units.json daq4xss-seed58.json \
    http://127.0.0.1:8080/v1/embeddings http://127.0.0.1:8090/v1/embeddings
```

The first endpoint serves gte-modernbert-base and the second EmbeddingGemma 2.
