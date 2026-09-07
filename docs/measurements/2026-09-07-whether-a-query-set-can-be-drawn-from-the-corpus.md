# 2026-09-07 — whether a query set can be drawn from the corpus

`D-01M1VNK3CJHMPR` allows one quality-shaped measurement and refuses the two
ordinary ways to get a labelled query set. Its exception draws the query from the
text whose section is the answer by construction, which removes the annotator
and the leakage at once. That trick cannot be pointed at a reranker: a query
lifted from a passage is an exact substring of what a cross-encoder scores, and
an exact match carries a cross-encoder to the top almost regardless of meaning,
so the artifact would inflate the arm under test rather than both arms alike.

This is a search for a query source with the same by-construction labelling and
no substring relationship to the passage. Both pinned corpora were checked.

**Frontmatter is not it.** `rust-lang/book` carries none at all. `mdn/content`
carries `browser-compat`, `page-type`, `short-title`, `sidebar`, `slug`,
`spec-urls`, `status` and `title`, and no description. A `title` is a name
rather than a question — `Array.prototype.at()` is the lexical lookup that
belongs to `rg` — so a set built from titles would measure the route folio does
not take.

Two structural facts came out of the check and both hold. A section's text runs
from its heading line down, so frontmatter is in no section's text: the field is
outside what either arm would score. And MDN documents carry no H1, so a
document's first section is headingless and its `title` appears nowhere in the
index except as frontmatter.

**Anchor text with a fragment is it.** An MDN author writing
`[enforcing trusted types](/en-US/docs/Web/API/Trusted_Types_API#using_a_csp_to_enforce_trusted_types)`
has named a section, in prose, in a different document, years before folio
existed. `mdn/content` holds 11,152 such links, 7,169 of them with a prose
anchor rather than a macro or a code span.

Sampling 1,200 prose anchors, resolved by matching the `slug` frontmatter folio
already flattened to a path and the fragment to a heading:

| | |
|---|---|
| Resolved to a section folio indexed | 1,133 of 1,200 (94%) |
| Anchor of one word | 26% |
| Anchor of two words | 35% |
| Anchor of three words or more | 38% |
| Anchor appearing verbatim in its target section | 65% |
| Three words or more **and** not a substring of the target | 21% |

**Raw anchor text has the same disease in milder form.** Two thirds of it is a
literal substring of the section it points at, and 61% of it is one or two
words, which is a lexical lookup again. Keeping only anchors of three words or
more that are not substrings of their target leaves 21% — about 1,400 pairs over
the corpus before duplicates are removed, against the 400 queries
[the public benchmarks over dividing and merging](2026-09-06-the-public-benchmarks-over-dividing-and-merging.md)
runs. The set is small and it is large enough.

**Filtering that way biases the set, and the bias runs toward what folio is
for.** What is removed is every pair a literal match would have answered, so
what remains is the pairs where the words someone used are not the words the
document uses. That is the failure folio exists to fix rather than an
unrepresentative convenience, and the specific asymmetry that ruled out the
drawn-query trick — an exact substring pinning a cross-encoder to the top — is
gone by construction rather than by argument. What survives is ordinary topical
overlap, which every retrieval set has and both a bi-encoder and a cross-encoder
exploit.

**What it still is.** One link names one section, so this measures whether the
named section ranks, and other sections of the same document may answer as well
without being credited. That is retrievability rather than preference, which is
the shape `D-01M1VNK3CJHMPR` admits, and it is not a ranking-quality score. The
route is also one corpus: `rust-lang/book` holds nine fragment links across 112
files, most of them within a page, so nothing like this can be built there.

**Method.** The link pattern was `[anchor](/en-US/docs/<slug>#<fragment>)` with
anchors containing `{`, `}`, backtick or `|` discarded as macros or code. A
fragment was matched to a section by lowercasing both it and the heading and
reducing every run of non-alphanumerics to a single underscore, which is how MDN
builds its own anchors. Files were walked in a seeded shuffle; the two samples
stopped at 1,200 anchors and 1,100 resolved pairs. Substring tests were
case-insensitive over the section's whole text, heading line included.
