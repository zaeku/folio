#!/usr/bin/env python3
"""Score each graph against the file pairs the book's authors linked."""

import sys
from collections import Counter

import corpus


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else corpus.K
    c = corpus.Corpus()
    ents, rels, usage = corpus.extraction(c)
    G = {"folio kNN": c.folio_graph(k), "heuristic ent": c.bag_graph(c.heuristic_bags(), k)[0]}
    G["LLM entity"], usable, empty = c.bag_graph([list(e) for e in ents], k)
    G["LLM relation"] = c.bag_graph(
        [[f"{s}|{o}" for s, _, o in r] + [s for s, _, _ in r] + [o for _, _, o in r] for r in rels], k)[0]
    G["folio ∪ LLM"] = G["folio kNN"] | G["LLM entity"]

    print(f"sections={c.n} author-linked file pairs={len(c.truth)} k={k}")
    print(f"LLM: distinct entities={len(set().union(*ents))} usable={usable} sections without one={empty} "
          f"triples={sum(map(len, rels))} distinct predicates={len({p for r in rels for _, p, _ in r})}")
    print(f"LLM usage: {dict(usage)}")
    print(f"{'graph':14} {'edges':>6} {'same-file':>9} {'x-chapter':>9} {'P':>5} {'R':>5} {'max-in':>6}")
    F = {}
    for name, es in G.items():
        f = F[name] = c.files(es)
        hit = f & c.truth
        same = sum(c.path[i] == c.path[j] for i, j in es) / len(es)
        cross = sum(c.path[i][:4] != c.path[j][:4] for i, j in es) / len(es)
        indeg = max(Counter(j for _, j in es).values())
        print(f"{name:14} {len(es):6} {same:9.0%} {cross:9.0%} {len(hit)/len(f):5.2f} {len(hit)/len(c.truth):5.2f} {indeg:6}")

    names = ["folio kNN", "heuristic ent", "LLM entity", "LLM relation"]
    print("\nsection-edge Jaccard")
    for a in names:
        print(f"  {a:14}" + "".join(f"{len(G[a] & G[b]) / len(G[a] | G[b]):7.2f}" for b in names))
    fa, fl = F["folio kNN"], F["LLM entity"]
    print(f"authors' pairs found only by folio={len((fa - fl) & c.truth)} "
          f"only by LLM={len((fl - fa) & c.truth)} both={len(fa & fl & c.truth)}")
    for name in ("folio kNN", "LLM entity"):
        hubs = Counter(j for _, j in G[name]).most_common(3)
        print(f"{name} hubs:", [(c.path[j], c.head[j], d) for j, d in hubs])


if __name__ == "__main__":
    main()
