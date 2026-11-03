#!/usr/bin/env python3
"""Compare truncating an over-budget function with splitting it at syntax nodes.

usage: chunked.py <corpus-dir> <units.json> <pairs.json> <gemma-endpoint> > out.json
Arm: Gemma, code prefix, filename title. A function over BUDGET characters is
either cut at BUDGET (baseline) or split: descend into the outermost block
children / match arms / case statements inside a span until each piece fits,
merge adjacent pieces greedily up to BUDGET, and put the function's header
(the lines before its first child) in front of every chunk. A function scores
the best of its chunks.
"""
import json, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np

BUDGET = 8000
QUERY = {'rust': '[(block (_) @s) (match_arm) @s]',
         'cpp': '[(compound_statement (_) @s) (case_statement) @s]',
         'python': '(block (_) @s)'}
root, units, pairs, url = Path(sys.argv[1]), json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4]
src, spans = {}, {}


def lines(f):
    if f not in src:
        src[f] = (root / f).read_text(errors='replace').split('\n')
    return src[f]


def text(f, a, b):
    return '\n'.join(lines(f)[a - 1:b])


def nodes(u):
    f = u['file']
    if f not in spans:
        r = subprocess.run(['ast-editor', 'inspect', str(root / f), '--query', QUERY[u['lang']], '--include-code', 'false'],
                           capture_output=True, text=True, check=True)
        found = json.loads(r.stdout.split('```json', 1)[1].rsplit('```', 1)[0])
        spans[f] = sorted({(m['start_line'], m['end_line']) for m in found['matches']})
    return spans[f]


def split(u, a, b):
    """Outermost candidate spans strictly inside [a, b], recursively, until each fits."""
    inside = [(s, e) for s, e in nodes(u) if a <= s and e <= b and (s, e) != (a, b)]
    top = [x for x in inside if not any(y != x and y[0] <= x[0] and x[1] <= y[1] for y in inside)]
    if not top:
        return [(a, b)]
    out = []
    for s, e in top:
        out += [(s, e)] if len(text(u['file'], s, e)) <= BUDGET else split(u, s, e)
    return out


def chunks(u):
    pieces = split(u, u['start'], u['end'])
    head = text(u['file'], u['start'], pieces[0][0] - 1)[:1500]
    merged = []
    for s, e in pieces:
        if merged and len(text(u['file'], merged[-1][0], e)) + len(head) <= BUDGET:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e, (head + '\n' + text(u['file'], s, e))[:BUDGET]) for s, e in merged]


def embed(texts):
    o = []
    for j in range(0, len(texts), 16):
        r = urllib.request.Request(url, json.dumps({"input": texts[j:j + 16]}).encode(), {"Content-Type": "application/json"})
        o += [d["embedding"] for d in json.load(urllib.request.urlopen(r))["data"]]
    X = np.array(o, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


title = lambda u: f"title: {Path(u['file']).name} | text: "
full = [text(u['file'], u['start'], u['end']) for u in units]
long = [i for i, t in enumerate(full) if len(t) > BUDGET]
docs, owner = [title(u) + t[:BUDGET] for u, t in zip(units, full)], list(range(len(units)))
pieces = {}
for i in long:
    cs = chunks(units[i])
    pieces[i] = [(s, e, len(t)) for s, e, t in cs]
    docs += [title(units[i]) + t for _, _, t in cs]
    owner += [i] * len(cs)
owner = np.array(owner)
D = embed(docs)
Q = embed(["task: code retrieval | query: " + p['q'] for p in pairs])
base_rows = np.arange(len(units))
cut = set(long)
chunk_rows = np.array([r for r in range(len(docs)) if r >= len(units) or r not in cut])


def rank(rows, q, p):
    s = D[rows] @ q
    best = {}
    for r, v in zip(owner[rows], s):
        if v > best.get(r, -9):
            best[r] = v
    order = sorted(best, key=lambda i: -best[i])
    return next((k for k, i in enumerate(order, 1) if units[i]['hash'] == p['hash']), None)


longh = {units[i]['hash'] for i in long}
res = []
for n, (p, q) in enumerate(zip(pairs, Q)):
    res.append({"id": n, "long_target": p['hash'] in longh, "base": rank(base_rows, q, p), "chunked": rank(chunk_rows, q, p)})
mrr = lambda key, keep: round(sum(1 / r[key] for r in res if keep(r) and r[key] <= 20) / max(1, sum(keep(r) for r in res)), 3)
print(json.dumps({"long_units": len(long), "chunks": sum(len(v) for v in pieces.values()),
                  "chunk_sizes": {f"{units[i]['file']}:{units[i]['start']}": [x[2] for x in v] for i, v in pieces.items()},
                  "mrr_all": {"base": mrr('base', lambda r: True), "chunked": mrr('chunked', lambda r: True)},
                  "long_targets": [r for r in res if r['long_target']],
                  "changed": [r for r in res if r['base'] != r['chunked']]}, indent=1))
