#!/usr/bin/env python3
"""Score a query set through `folio query`, run from the corpus root.

usage: score.py <corpus-dir> <pairs.json> [query-prefix] > ranks.json
A hit is a returned range containing the section's first line.
"""
import json, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])


def score(pairs, prefix=''):
    ranks = []
    for p in pairs:
        out = subprocess.run(['folio', 'query', prefix + p['q'], '-l', '20', '--paths-only', '--no-refresh'],
                             cwd=root, capture_output=True, text=True, check=True).stdout.split()
        rank = None
        for r, row in enumerate(out, 1):
            f, rng = row.rsplit(':', 1)
            a, b = map(int, rng.split('-'))
            if f == p['file'] and a <= p['line'] <= b:
                rank = r
                break
        ranks.append(rank)
    n = len(ranks)
    at = lambda k: sum(1 for r in ranks if r and r <= k)
    print(json.dumps({'n': n, 'top1': at(1), 'top5': at(5), 'top10': at(10), 'top20': at(20),
                      'mrr': round(sum(1 / r for r in ranks if r) / n, 3), 'ranks': ranks}))


score(json.load(open(sys.argv[2])), sys.argv[3] if len(sys.argv) > 3 else '')
