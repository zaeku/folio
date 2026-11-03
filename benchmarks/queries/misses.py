#!/usr/bin/env python3
"""Write each question the best Gemma arm does not rank first, blinded for a judge.

usage: misses.py <corpus-dir> <units.json> <pairs.json> <gemma-endpoint> <name> <out-dir> [batches=2]
The arm is the code-retrieval query prefix with the filename as the document title.
For every pair whose target is not first, the candidates are the target and the
units ranked above it, at most three. <out-dir>/batch-<name>-<k>.json holds them
in shuffled order under labels A-D, with no rank and no mark on the target;
<out-dir>/key-<name>.json says which label is the target and in what order the
others ranked.
"""
import json, random, sys, urllib.request
from pathlib import Path

import numpy as np

root, units, pairs = Path(sys.argv[1]), json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
url, name, out = sys.argv[4], sys.argv[5], Path(sys.argv[6])
k = int(sys.argv[7]) if len(sys.argv) > 7 else 2
src = {}


def text(u):
    if u['file'] not in src:
        src[u['file']] = (root / u['file']).read_text(errors='replace').split('\n')
    return '\n'.join(src[u['file']][u['start'] - 1:u['end']])


def embed(texts):
    o = []
    for j in range(0, len(texts), 16):
        r = urllib.request.Request(url, json.dumps({"input": texts[j:j + 16]}).encode(),
                                   {"Content-Type": "application/json"})
        o += [d["embedding"] for d in json.load(urllib.request.urlopen(r))["data"]]
    X = np.array(o, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


span = lambda u: f"{u['file']}:{u['start']}-{u['end']}"
D = embed([f"title: {Path(u['file']).name} | text: " + text(u)[:8000] for u in units])
Q = embed(["task: code retrieval | query: " + p['q'] for p in pairs])
rng, blind, key = random.Random(7), [], {}
for n, (p, q) in enumerate(zip(pairs, Q)):
    order = np.argsort(-(D @ q))
    rank = next(r for r, i in enumerate(order, 1) if units[i]['hash'] == p['hash'])
    if rank == 1:
        continue
    target = span(units[order[rank - 1]])
    above = [span(units[i]) for i in order[:min(rank - 1, 3)]]
    cands = [target] + above
    rng.shuffle(cands)
    labels = dict(zip('ABCD', cands))
    case = f"{name}/{n}"
    key[case] = {"rank": rank, "target": next(l for l, s in labels.items() if s == target),
                 "above": [next(l for l, s in labels.items() if s == a) for a in above]}
    blind.append({"case": case, "q": p['q'], "candidates": labels})
size = -(-len(blind) // k)
for j in range(k):
    (out / f"batch-{name}-{j}.json").write_text(json.dumps({"root": str(root), "cases": blind[j * size:(j + 1) * size]}, indent=1))
(out / f"key-{name}.json").write_text(json.dumps(key, indent=1))
print(f"{len(pairs)} pairs, {len(blind)} not first, in {k} batches")
