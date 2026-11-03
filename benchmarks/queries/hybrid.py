#!/usr/bin/env python3
"""Gemma alone, rg alone, and the two fused by reciprocal rank, on a mixed query set.

usage: hybrid.py <corpus-dir> <units.json> <pairs.json> <dropped.json> <gemma-endpoint> > out.json
The set is every question generation kept plus every one it dropped for naming a
piece of the function, so it is what the model wrote before the leak filter.
Gemma: code prefix, filename title. rg: as in code-arms.py. Fusion: each unit
scores 1/(K + rank) summed over the arms that rank it, K = 60.
"""
import json, re, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np

K, BUDGET = 60, 8000
STOP = set("""a an and are as at be by for from in into is it its of on or that the this to with
without when which while using use via each all any by over under per""".split())
root, units, url = Path(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[5]
start = {(u['file'], u['start']): u['hash'] for u in units}
pairs = [{**p, "named": False} for p in json.load(open(sys.argv[3]))]
pairs += [{"q": d['q'], "hash": start[(d['file'], d['line'])], "named": True} for d in json.load(open(sys.argv[4]))]
src = {}


def text(u):
    if u['file'] not in src:
        src[u['file']] = (root / u['file']).read_text(errors='replace').split('\n')
    return '\n'.join(src[u['file']][u['start'] - 1:u['end']])


def embed(texts):
    o = []
    for j in range(0, len(texts), 16):
        r = urllib.request.Request(url, json.dumps({"input": texts[j:j + 16]}).encode(), {"Content-Type": "application/json"})
        o += [d["embedding"] for d in json.load(urllib.request.urlopen(r))["data"]]
    X = np.array(o, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


at = {}
for i, u in enumerate(units):
    for line in range(u['start'], u['end'] + 1):
        at.setdefault((u['file'], line), i)


def rg(q):
    terms = sorted({w for w in re.findall(r'[a-z0-9]+', q.lower()) if w not in STOP and len(w) > 1})
    out = subprocess.run(['rg', '-i', '-n', '-o', '--no-heading', '--null', '-g', '!.codegraph',
                          r'\b(' + '|'.join(terms) + ')', '.'], cwd=root, capture_output=True, text=True).stdout
    hits = {}
    for row in out.splitlines():
        if '\0' not in row:
            continue
        f, rest = row.split('\0', 1)
        line, word = rest.split(':', 1)
        i = at.get((f.removeprefix('./'), int(line)))
        if i is not None:
            d, t = hits.get(i, (set(), 0))
            hits[i] = (d | {word.lower()}, t + 1)
    return sorted(hits, key=lambda i: (-len(hits[i][0]), -hits[i][1], i))


D = embed([f"title: {Path(u['file']).name} | text: " + text(u)[:BUDGET] for u in units])
Q = embed(["task: code retrieval | query: " + p['q'] for p in pairs])


def rank_of(order, h):
    return next((r for r, i in enumerate(order[:20], 1) if units[i]['hash'] == h), None)


ranks = {"Gemma": [], "rg": [], "fused": []}
for p, q in zip(pairs, Q):
    g = [int(i) for i in np.argsort(-(D @ q))[:100]]
    r = rg(p['q'])[:100]
    s = {}
    for order in (g, r):
        for k, i in enumerate(order, 1):
            s[i] = s.get(i, 0) + 1 / (K + k)
    fused = sorted(s, key=lambda i: -s[i])
    for name, order in (("Gemma", g), ("rg", r), ("fused", fused)):
        ranks[name].append(rank_of(order, p['hash']))


def summary(rs):
    at = lambda k: sum(1 for r in rs if r and r <= k)
    return {"n": len(rs), "top1": at(1), "top5": at(5), "top10": at(10), "top20": at(20),
            "mrr": round(sum(1 / r for r in rs if r) / len(rs), 3)}


subsets = {"all": lambda p: True, "named": lambda p: p['named'], "unnamed": lambda p: not p['named']}
print(json.dumps({name: {s: summary([r for r, p in zip(rs, pairs) if keep(p)]) for s, keep in subsets.items()}
                  for name, rs in ranks.items()}, indent=1))
