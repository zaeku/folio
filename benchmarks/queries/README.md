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

`ripgrep-seed58.json` and `minisqlite-seed58.json` are the sets the 2026-11-03
measurement used, on two public trees fetched as tarballs at a pinned commit.
Each was drawn with `rust=150` as the fifth argument of `generate-code.py`.

```sh
curl -sL https://github.com/BurntSushi/ripgrep/archive/91f5d22ef48b.tar.gz | tar -xz -C <export> --strip-components 1
curl -sL https://github.com/cursor/minisqlite/archive/4a5c1341fdde.tar.gz | tar -xz -C <export> --strip-components 1
./generate-code.py <export> units.json ../reports/code-queries 58 rust=150
```

`misses.py` writes the questions the best Gemma arm does not rank first, with the
units ranked above each target, in shuffled batches that hide which is the
target. A judge scores every candidate against the question, and `judged.py`
joins those scores with the key. `*-seed58-misses.json` are the joined
judgements the 2026-11-03 miss measurement used.

```sh
./misses.py <export> units.json ripgrep-seed58.json http://127.0.0.1:8090/v1/embeddings ripgrep ../reports/misses 3
./judged.py ../reports/misses/key-ripgrep.json verdicts.json 127 31 > ripgrep-seed58-misses.json
```

`chunked.py` ranks the same set twice, once with each function over 8,000
characters cut there and once with it split at syntax nodes. `hybrid.py` ranks
Gemma, `rg` and their reciprocal-rank fusion on the kept questions together with
the ones generation dropped for naming the function, which are
`*-seed58-dropped.json`.

```sh
./chunked.py <export> units.json minisqlite-seed58.json http://127.0.0.1:8090/v1/embeddings
./hybrid.py <export> units.json minisqlite-seed58.json minisqlite-seed58-dropped.json http://127.0.0.1:8090/v1/embeddings
```

`unit-arms.py` ranks the same questions over four other cuts of the tree — fixed
windows, regex declarations, and Tree-sitter functions led by their container's
declaration with and without the file's module comment — and scores each by
line overlap with the target and by lines read to reach it.

```sh
./unit-arms.py <export> units.json ripgrep-seed58.json http://127.0.0.1:8090/v1/embeddings
```

`mixed.py` ranks section questions and function questions together in one
index, under each pairing of the search or code-retrieval query prefix with a
`title: none` or filename document title. The sections are the rows `folio
index` records for the tree's markdown, dumped from `.folio/index.db`.

```sh
sqlite3 <root>/.folio/index.db "SELECT path, start, stop FROM sections" | tr '|' '\t' > sections.tsv
./mixed.py <root> sections.tsv units.json ripgrep rust-book-seed55.json ripgrep-seed58.json http://127.0.0.1:8090/v1/embeddings
```
