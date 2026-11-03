# 2026-11-03 — which unit to cut source code into

The code-query measurements cut a tree into functions with Tree-sitter. Is a
function the unit, and does it need a parser? Five ways of cutting the same three
trees were ranked by EmbeddingGemma 2 on the same questions.

**A function is the unit, and on Rust and Python a regex finds it as well as a
parser does.** Fixed windows fall to about half the MRR. Cutting at declaration
lines by regex matches Tree-sitter to within 0.002 on all three corpora.
Leading a function with the declaration of its `impl` or class gains a little on
two corpora and loses on the third. Leading it with the file's module comment
as well loses heavily on the corpus where every file has one.

| Unit | daq4xss | ripgrep | minisqlite |
|---|---|---|---|
| window, 40 lines every 20 | 0.376 | 0.295 | 0.328 |
| regex declaration to closing brace | 0.657 | 0.387 | 0.555 |
| Tree-sitter function | 0.659 | 0.386 | 0.555 |
| function and its container's declaration | 0.684 | 0.392 | 0.535 |
| that and the file's `//!` lines | 0.674 | 0.393 | 0.336 |

MRR, of 109, 127 and 77 questions. Questions whose function is reached within
500 lines read:

| Unit | daq4xss | ripgrep | minisqlite |
|---|---|---|---|
| window | 77 | 79 | 51 |
| regex | 100 | 106 | 68 |
| Tree-sitter function | 101 | 106 | 68 |
| container's declaration | 102 | 109 | 69 |
| and `//!` lines | 96 | 109 | 53 |

**What the gaps mean.**

- *Windows.* A window lands inside the right function often enough to be found
  eventually, but ranks it lower and costs more reading to reach.
- *Regex.* The regex found exactly the Rust and Python functions Tree-sitter
  found: 533 and 338 on daq4xss, 2,904 against 2,905 on ripgrep, 9,767 against
  9,769 on minisqlite. On C++ it found 222 of 224 and 165 spans that are not
  functions, which cost nothing here on 20 C++ questions. It relies on a
  declaration keyword starting a line and on balanced braces, not on a
  formatter: `rustfmt --check` with default settings reports 117 differences in
  ripgrep's Rust, 1,592 in daq4xss's and 7,697 in minisqlite's, mostly import
  order and line width.
- *Module comments.* minisqlite opens nearly every file with a `//!` comment of
  around 400 characters, and putting it in front of every function made the
  functions of one file alike to the embedder.

**What it does not show.** Every question was written to describe one function,
which favours the function unit; a question whose answer spans several
functions was not asked. The regex was written for Rust, Python and C++ and
tried on nothing else, and the languages it would fail on — those with no
function keyword, such as Java methods or JavaScript arrow functions — are not
here. The containers were found with Tree-sitter, not by regex.

**Method.**

- *Arms.* `benchmarks/queries/unit-arms.py`, Gemma behind `task: code retrieval
  | query: ` and `title: <filename> | text: `, every unit cut at 8,000
  characters. The Tree-sitter units are those of
  `benchmarks/queries/functions.py`. Containers are the `impls` and `traits`
  templates of `ast-editor inspect` for Rust and `classes` for Python and C++.
- *Scoring.* A question's target is its function's line span. A ranked unit hits
  when it overlaps the target by at least half of the shorter of the two, so a
  window inside a long function finds it. This differs from the body-hash
  scoring of the earlier measurements, which also counted a duplicate copy
  elsewhere, and so the Tree-sitter row reads 0.386 and 0.555 here where those
  read 0.390 and 0.561. Reading cost is the number of distinct lines in the
  units ranked up to and including the first hit.
- *Sets.* `benchmarks/queries/*-seed58.json`, the questions without a piece of
  the function's name.
