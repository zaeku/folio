#!/usr/bin/env python3
"""Replay folio query's merge of adjacent results under other joining rules.

usage: merge-replay.py <corpus-dir> <sections.tsv> <units.json> <sub> <gemma-endpoint> <set.json>... > out.json
Units are embedded as folio embeds them, sections behind `title: none | text: `
and functions behind `title: <filename> | text: `, questions behind
`task: search result | query: `. For each question every unit is scored, the
top 128 are kept as folio 0.7.1 hydrates them, touching pieces of one file are
cut into runs around their best piece, and runs rank by the mean of their
pieces weighted by lines. A rule decides which neighbours join the best piece:
`ratio r` keeps a piece scoring at least r times the best, and `median k`
keeps one scoring at least best - k * (best - median of the 128). A question
hits the first run containing its line, as score.py counts. A set's pairs are
those of mixed.py: sections relative to <corpus-dir>, functions to <sub>.
"""
import json, sys, urllib.request
from pathlib import Path

import numpy as np

BUDGET, WINDOW = 8000, 128
root, sub, url = Path(sys.argv[1]), sys.argv[4], sys.argv[5]
units = [(p, int(a), int(b)) for p, a, b in (l.split('\t') for l in open(sys.argv[2]).read().splitlines())]
units += [(f"{sub}/{u['file']}", u['start'], u['end']) for u in json.load(open(sys.argv[3]))]
src = {}


def text(f, a, b):
    if f not in src:
        src[f] = (root / f).read_text(errors='replace').split('\n')
    return '\n'.join(src[f][a - 1:b])


def embed(texts):
    o = []
    for j in range(0, len(texts), 16):
        r = urllib.request.Request(url, json.dumps({"input": texts[j:j + 16]}).encode(), {"Content-Type": "application/json"})
        o += [d["embedding"] for d in json.load(urllib.request.urlopen(r))["data"]]
    X = np.array(o, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def title(f):
    return "none" if f.endswith(".md") else Path(f).name


D = embed([f"title: {title(f)} | text: " + text(f, a, b)[:BUDGET] for f, a, b in units])


def runs(top, s, keep):
    by = {}
    for i in sorted(top, key=lambda i: (units[i][0], units[i][1])):
        t = by.setdefault(units[i][0], [[]])
        if t[-1] and units[t[-1][-1]][2] + 1 != units[i][1]:
            t.append([])
        t[-1].append(i)
    out = []
    for groups in by.values():
        pending = [g for g in groups if g]
        while pending:
            g = pending.pop()
            b = max(range(len(g)), key=lambda k: s[g[k]])
            lo = hi = b
            while lo > 0 and keep(s[g[lo - 1]], s[g[b]]):
                lo -= 1
            while hi + 1 < len(g) and keep(s[g[hi + 1]], s[g[b]]):
                hi += 1
            run = g[lo:hi + 1]
            lines = [units[i][2] - units[i][1] + 1 for i in run]
            out.append((sum(s[i] * n for i, n in zip(run, lines)) / sum(lines),
                        units[run[0]][0], units[run[0]][1], units[run[-1]][2]))
            pending += [x for x in (g[:lo], g[hi + 1:]) if x]
    return sorted(out, key=lambda r: -r[0])


RULES = {f"ratio {r}": (lambda r: lambda s, best, med: s >= r * best)(r) for r in (0.90, 0.93, 0.95, 0.97, 1.0)}
RULES.update({f"median {k}": (lambda k: lambda s, best, med: s >= best - k * (best - med))(k) for k in (0.25, 0.5)})

report = {}
for path in sys.argv[6:]:
    pairs = json.load(open(path))
    for p in pairs:
        if not p['file'].endswith('.md'):
            p['file'] = f"{sub}/{p['file']}"
    Q = embed(["task: search result | query: " + p['q'] for p in pairs])
    res = {name: ([], []) for name in RULES}
    for p, q in zip(pairs, Q):
        s = D @ q
        top = list(np.argsort(-s)[:WINDOW])
        med = float(np.median(s[top]))
        for name, rule in RULES.items():
            ranked = runs(top, s, lambda x, best: rule(x, best, med))[:20]
            r = next((k for k, (_, f, a, b) in enumerate(ranked, 1) if f == p['file'] and a <= p['line'] <= b), None)
            res[name][0].append(r)
            res[name][1].append(ranked[0][3] - ranked[0][2] + 1)
    report[Path(path).name] = {name: {
        "n": len(rs), "top1": sum(1 for r in rs if r == 1), "top3": sum(1 for r in rs if r and r <= 3),
        "top10": sum(1 for r in rs if r and r <= 10), "mrr": round(sum(1 / r for r in rs if r) / len(rs), 3),
        "first_lines": round(sum(ls) / len(ls), 1)} for name, (rs, ls) in res.items()}
print(json.dumps(report, indent=1))
