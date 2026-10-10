#!/usr/bin/env python3
"""Tally the judge's scores by group, with examples of what folio got right and wrong."""

import json
from collections import Counter, defaultdict

import corpus

GROUPS = ["control_author_linked", "folio_unlinked", "entity_only_unlinked", "control_random"]


def main():
    c = corpus.Corpus()
    out = corpus.OUT / "judge"
    pairs = {p["pid"]: p for p in json.loads((out / "pairs.json").read_text())}
    score, reason, usage = {}, {}, Counter()
    for f in sorted(out.glob("judge-*.json")):
        d = json.loads(f.read_text())
        usage.update({k: v for k, v in (d.get("usage") or {}).items() if isinstance(v, int)})
        for j in d["judgements"]:
            if j["pid"] in pairs:
                score[j["pid"]], reason[j["pid"]] = j["score"], j["reason"]
    print(f"judged {len(score)}/{len(pairs)} usage={dict(usage)}")

    by = defaultdict(list)
    for pid, s in score.items():
        by[pairs[pid]["group"]].append(s)
    print(f"{'group':24} {'n':>4} {'2':>5} {'1':>5} {'0':>5} {'mean':>5}")
    for g in GROUPS:
        s = by[g]
        if s:
            n = Counter(s)
            print(f"{g:24} {len(s):4} {n[2]/len(s):5.0%} {n[1]/len(s):5.0%} {n[0]/len(s):5.0%} {sum(s)/len(s):5.2f}")

    for want in (2, 0):
        print(f"\n-- folio_unlinked scored {want}")
        for pid in [p for p in score if pairs[p]["group"] == "folio_unlinked" and score[p] == want][:5]:
            a, b = c.at[tuple(pairs[pid]["a"])], c.at[tuple(pairs[pid]["b"])]
            print(f"  {c.path[a]}#{c.head[a][:40]!r} <-> {c.path[b]}#{c.head[b][:40]!r}\n     {reason[pid]}")


if __name__ == "__main__":
    main()
