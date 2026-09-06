# 2026-09-06 — how much text truncation puts out of reach

Read from the indexes the last benchmark run left on disk, both pinned corpora,
`--max-chars 8000`.

| | `rust-lang/book` | `mdn/content` |
|---|---|---|
| Sections | 548 | 119,359 |
| Cut at the budget | 6 | 142 |
| Share cut | 1.1% | 0.12% |

Within MDN's 142, 1,136,000 characters were embedded and **873,797 were not**:
43.5% of that text is in no vector and cannot be reached by any query. The
largest losses are reference pages whose tail is more of the same list —
`web/api/html_sanitizer_api/default_sanitizer_configuration/index.md` loses
42,658 characters, `web/html/reference/attributes/index.md` 38,528, and
`web/api/keyboardevent/keycode/index.md` 35,148.

Both halves matter to D-01M1T9G1416ACW. One section in 840 is cut, which is
small; the cut ones lose half their text, and they are the pages someone
searches for one entry in, which is not.
