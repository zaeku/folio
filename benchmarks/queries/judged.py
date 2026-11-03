#!/usr/bin/env python3
"""Join a judge's verdicts on blinded misses with their key, and recount the arm.

usage: judged.py <key.json> <verdicts.json> <pairs> <top1> > joined.json
<verdicts.json> is [{case, scores: {label: 0|1|2}, why}], 2 meaning the candidate
is what the question describes. A miss is
  ambiguous  - the target and a unit above it both score 2
  false miss - a unit above scores 2 and the target less
  true miss  - only the target scores 2
  bad query  - nothing scores 2
The lenient rank of a miss is that of the first unit above it scoring 2, so a
question several functions answer counts as found. Summaries go to stderr.
"""
import collections, json, sys

key, verdicts = json.load(open(sys.argv[1])), {v['case']: v for v in json.load(open(sys.argv[2]))}
pairs, top1 = int(sys.argv[3]), int(sys.argv[4])
joined, kinds, strict, lenient = [], collections.Counter(), [], []
for case, k in key.items():
    v = verdicts[case]
    s = v['scores']
    up = [s.get(l, 0) for l in k['above']]
    kind = ('ambiguous' if s[k['target']] == 2 and 2 in up else 'false miss' if 2 in up
            else 'true miss' if s[k['target']] == 2 else 'bad query')
    kinds[kind] += 1
    eff = next((j for j, x in enumerate(up, 1) if x == 2), k['rank'])
    strict.append(k['rank'])
    lenient.append(eff)
    joined.append({**k, "case": case, "kind": kind, "lenient_rank": eff, "scores": s, "why": v['why']})
mrr = lambda rs: round((top1 + sum(1 / r for r in rs if r <= 20)) / pairs, 3)
print(json.dumps(joined, indent=1))
print(dict(kinds), f"top1 {top1} -> {top1 + lenient.count(1)} of {pairs}, MRR {mrr(strict)} -> {mrr(lenient)}",
      file=sys.stderr)
