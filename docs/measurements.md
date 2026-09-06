# Measurements

Every number folio reports about itself is one file under
[measurements/](measurements/), named by the date it was taken and the question
it answers. `../AGENTS.md` requires a measured fact to carry how it was measured
and when; these are where those live, and a number quoted anywhere else should
point at one of them.

`ls measurements/` is the index. There is no other, and there will not be: an
index written by hand can disagree with what it indexes, because the two are
separate facts, while a filename cannot disagree with its own file.

**Reading them.** Names sort by date, so `ls` is newest last and
`ls measurements/ | grep 2026-09` is a month. Within a day they sort by title
rather than by when each was taken; the order that matters is which measurement
corrects which, and each says so by name. For the history of a single number,
`jj log -r 'diff_lines(substring:"1.2e-4")' -p` in this repository.

**Environment.** Apple M1 Pro, 32 GB, macOS 26. Corpora as named in
[references.md](references.md).

**The seven-question retrieval score is a smoke test, not evidence.** It lives in
[2026-09-04-private-corpora.md](measurements/2026-09-04-private-corpora.md) with
that label on it. The questions were written against a private corpus after
watching the tool answer, which is the leakage the zvec-grep benchmark
documentation warns about, and 7/7 against 6/7 is a difference of one question
with no repeated trials and no variance estimate. It is kept because it is what
the model choice was made on. `benchmarks/` is where a reproducible number comes
from.

**Where a new one goes.** A measurement worth keeping becomes a file here under
its own date. A run of `benchmarks/run.py` worth keeping becomes one too — see
[benchmarks/README.md](../benchmarks/README.md) for what that harness will and
will not measure.
