#!/usr/bin/env python3
"""Rank function units for each question by embeddings, CodeGraph and rg, outside folio.

usage: code-arms.py <corpus-dir> <units.json> <pairs.json> <gte-endpoint> <gemma-endpoint> > arms.json
Every unit of the tree is a candidate, tests included. A hit is the first ranked
unit whose body hash equals the target's, so either copy of a duplicated function
counts. A unit is embedded at most 8,000 characters long, folio's own budget.
`codegraph init` must already have run in <corpus-dir>.
"""
import json, re, subprocess, sys, urllib.request
from pathlib import Path

import numpy as np

BUDGET = 8000
STOP = set("""a an and are as at be by for from in into is it its of on or that the this to with
without when which while using use via each all any by over under per""".split())

root, units, pairs = Path(sys.argv[1]), json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
gte, gemma = sys.argv[4], sys.argv[5]
src = {}


def text(u):
    if u['file'] not in src:
        src[u['file']] = (root / u['file']).read_text(errors='replace').split('\n')
    return '\n'.join(src[u['file']][u['start'] - 1:u['end']])


body = [text(u)[:BUDGET] for u in units]
truncated = sum(len(text(u)) > BUDGET for u in units)


def embed(url, texts):
    out = []
    for k in range(0, len(texts), 16):
        req = urllib.request.Request(url, json.dumps({"input": texts[k:k + 16]}).encode(),
                                     {"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]]
    X = np.array(out, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def rank_of(order, p):
    return next((r for r, i in enumerate(order[:20], 1) if units[i]['hash'] == p['hash']), None)


def embedding_arm(url, doc, qry):
    D = embed(url, [doc(u) + b for u, b in zip(units, body)])
    Q = embed(url, [qry + p['q'] for p in pairs])
    return [rank_of(np.argsort(-(D @ q)), p) for p, q in zip(pairs, Q)]


def codegraph_arm():
    ranks = []
    for p in pairs:
        r = json.loads(subprocess.run(['codegraph', 'query', p['q'], '-j', '-l', '100', '-p', str(root)],
                                      capture_output=True, text=True, check=True).stdout or '[]')
        order = []
        for h in r:
            n = h['node']
            if n['kind'] not in ('function', 'method'):
                continue
            i = next((i for i, u in enumerate(units) if u['file'] == n['filePath']
                      and u['start'] <= n['startLine'] <= u['end']), None)
            if i is not None and i not in order:
                order.append(i)
        ranks.append(rank_of(order, p))
    return ranks


def rg_arm():
    """The question's words, stopwords dropped, matched at a word start; a unit ranks by the
    number of distinct words it contains, then by total matches."""
    at = {}
    for i, u in enumerate(units):
        for line in range(u['start'], u['end'] + 1):
            at.setdefault((u['file'], line), i)
    ranks = []
    for p in pairs:
        terms = sorted({w for w in re.findall(r'[a-z0-9]+', p['q'].lower()) if w not in STOP and len(w) > 1})
        out = subprocess.run(['rg', '-i', '-n', '-o', '--no-heading', '-g', '!.codegraph',
                              r'\b(' + '|'.join(terms) + ')', '.'],
                             cwd=root, capture_output=True, text=True).stdout
        hits = {}
        for row in out.splitlines():
            f, line, word = row.split(':', 2)
            i = at.get((f.removeprefix('./'), int(line)))
            if i is not None:
                distinct, total = hits.get(i, (set(), 0))
                hits[i] = (distinct | {word.lower()}, total + 1)
        order = sorted(hits, key=lambda i: (-len(hits[i][0]), -hits[i][1], i))
        ranks.append(rank_of(order, p))
    return ranks


def summary(ranks, keep=lambda p: True):
    rs = [r for r, p in zip(ranks, pairs) if keep(p)]
    at = lambda k: sum(1 for r in rs if r and r <= k)
    return {"n": len(rs), "top1": at(1), "top5": at(5), "top10": at(10), "top20": at(20),
            "mrr": round(sum(1 / r for r in rs if r) / len(rs), 3)}


arms = {
    "gte, no prefix": embedding_arm(gte, lambda u: "", ""),
    "Gemma, search prefix, title none": embedding_arm(gemma, lambda u: "title: none | text: ",
                                                      "task: search result | query: "),
    "Gemma, code prefix, title none": embedding_arm(gemma, lambda u: "title: none | text: ",
                                                    "task: code retrieval | query: "),
    "Gemma, code prefix, title filename": embedding_arm(gemma, lambda u: f"title: {Path(u['file']).name} | text: ",
                                                        "task: code retrieval | query: "),
    "CodeGraph query": codegraph_arm(),
    "rg": rg_arm(),
}
print(json.dumps({"units": len(units), "truncated": truncated, "pairs": len(pairs),
                  "arms": {name: {"all": summary(r),
                                  **{lang: summary(r, lambda p, l=lang: p['lang'] == l)
                                     for lang in ('rust', 'python', 'cpp')},
                                  "ranks": r}
                           for name, r in arms.items()}}, indent=1))
