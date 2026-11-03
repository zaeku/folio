#!/usr/bin/env python3
"""Rank sections and functions in one index under one query prefix, outside folio.

usage: mixed.py <corpus-dir> <sections.tsv> <units.json> <sub> <section-pairs.json> <function-pairs.json> <gemma-endpoint> > out.json
<sections.tsv> is `path start stop` per row, as folio's index records them.
<units.json> and <function-pairs.json> are relative to <corpus-dir>/<sub>.
Every unit is embedded as its lines, cut at 8,000 characters, behind
`title: none | text: ` or `title: <filename> | text: `, and every question
behind `task: search result | query: ` or `task: code retrieval | query: `.
A section question hits a unit containing its line, as score.py counts; a
function question hits a unit overlapping its function by half of the shorter.
Sections rank one by one, without folio's merge of adjacent results.
"""
import json, sys, urllib.request
from pathlib import Path

import numpy as np

BUDGET = 8000
root, url, sub = Path(sys.argv[1]), sys.argv[7], sys.argv[4]
units = [(p, int(a), int(b), 'section') for p, a, b in (l.split('\t') for l in open(sys.argv[2]).read().splitlines())]
units += [(f"{sub}/{u['file']}", u['start'], u['end'], 'function') for u in json.load(open(sys.argv[3]))]
end_of = {(f"{sub}/{u['file']}", u['start']): u['end'] for u in json.load(open(sys.argv[3]))}
pairs = [{**p, "kind": "section"} for p in json.load(open(sys.argv[5]))]
pairs += [{"q": p['q'], "file": f"{sub}/{p['file']}", "line": p['line'], "kind": "function",
           "end": end_of[(f"{sub}/{p['file']}", p['line'])]} for p in json.load(open(sys.argv[6]))]
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


def hit(u, p):
    if u[0] != p['file']:
        return False
    if p['kind'] == 'section':
        return u[1] <= p['line'] <= u[2]
    overlap = min(u[2], p['end']) - max(u[1], p['line']) + 1
    return overlap >= 0.5 * min(u[2] - u[1] + 1, p['end'] - p['line'] + 1)


QUERY = {"search": "task: search result | query: ", "code": "task: code retrieval | query: "}
TITLE = {"none": lambda f: "none", "filename": lambda f: Path(f).name}
docs = {}
for t, name in TITLE.items():
    docs[t] = embed([f"title: {name(f)} | text: " + text(f, a, b)[:BUDGET] for f, a, b, _ in units])
qs = {k: embed([v + p['q'] for p in pairs]) for k, v in QUERY.items()}


def summary(ranks, top_kind):
    at = lambda k: sum(1 for r in ranks if r and r <= k)
    return {"n": len(ranks), "top1": at(1), "top10": at(10), "top20": at(20),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3), "first_is": dict(top_kind)}


out = {"units": {"section": sum(u[3] == 'section' for u in units), "function": sum(u[3] == 'function' for u in units)},
       "arms": {}}
for qk in QUERY:
    for tk in TITLE:
        D, Q = docs[tk], qs[qk]
        by = {"section": ([], {}), "function": ([], {})}
        for p, q in zip(pairs, Q):
            order = np.argsort(-(D @ q))[:20]
            ranks, kinds = by[p['kind']]
            ranks.append(next((r for r, i in enumerate(order, 1) if hit(units[i], p)), None))
            k = units[order[0]][3]
            kinds[k] = kinds.get(k, 0) + 1
        out["arms"][f"query {qk}, title {tk}"] = {kind: summary(r, k) for kind, (r, k) in by.items()}
print(json.dumps(out, indent=1))
