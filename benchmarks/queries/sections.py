#!/usr/bin/env python3
"""Section-level ranking outside folio, with and without the model card's retrieval prefixes.

usage: sections.py <endpoint> <pairs.json>
Reads folio's sections from the rust-book index, embeds every section's text itself.
"""
import json, sys, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'links'))
import corpus  # noqa: E402

url, pairs = sys.argv[1], json.load(open(sys.argv[2]))
c = corpus.Corpus()


def embed(texts):
    out = []
    for k in range(0, len(texts), 16):
        req = urllib.request.Request(url, json.dumps({"input": texts[k:k + 16]}).encode(), {"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]]
    X = np.array(out, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def arm(doc, qry):
    D = embed([doc + t for t in c.text])
    Q = embed([qry + p["q"] for p in pairs])
    ranks = []
    for p, q in zip(pairs, Q):
        order = np.argsort(-(D @ q))
        r = next((r for r, i in enumerate(order[:20], 1) if c.key[i] == (p["file"], p["line"])), None)
        ranks.append(r)
    at = lambda k: sum(1 for r in ranks if r and r <= k)
    return {"top1": at(1), "top5": at(5), "top10": at(10), "top20": at(20),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3)}


print(json.dumps({"n": len(pairs),
                  "none": arm("", ""),
                  "prefixed": arm("title: none | text: ", "task: search result | query: ")}))
