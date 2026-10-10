#!/usr/bin/env python3
"""Have a model score sampled section pairs, blind to which graph proposed them."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import corpus

MODEL = "gemini-3.7-flash-medium"
PER_CALL, WORKERS, CAP = 10, 4, 3000
SEED = 7
INSTR = """You are reviewing a wiki built from "The Rust Programming Language" book. For each PAIR below, decide whether an editor should add a cross-reference link between section A and section B.
score 2 = worth linking: a reader of one would clearly benefit from being pointed to the other (prerequisite, deeper treatment, same concept from another angle, direct contrast).
score 1 = loosely related: same broad area, but a link would be marginal.
score 0 = not worth linking: unrelated, or related only by a common word.
Judge only from the two texts. reason: one short sentence. Return ONLY the JSON object, one entry per pid. Do not use tools.

"""
# Gemini's API refuses an integer `enum`, so the range is bounded instead.
SCHEMA = {"type": "object", "properties": {"judgements": {"type": "array", "items": {"type": "object", "properties": {
    "pid": {"type": "integer"}, "score": {"type": "integer", "minimum": 0, "maximum": 2}, "reason": {"type": "string"}},
    "required": ["pid", "score", "reason"]}}}, "required": ["judgements"]}


def main():
    c = corpus.Corpus()
    ents, _, _ = corpus.extraction(c)
    rng = np.random.default_rng(SEED)
    ok = lambda i, j: c.path[i] != c.path[j] and len(c.text[i]) > 200 and len(c.text[j]) > 200
    undirected = lambda es: {tuple(sorted(e)) for e in es if ok(*e)}
    folio = undirected(c.folio_graph())
    entity = undirected(c.bag_graph([list(e) for e in ents])[0])

    def sample(pool, k):
        pool = sorted(pool)
        return [pool[i] for i in rng.choice(len(pool), min(k, len(pool)), replace=False)]

    randoms = {tuple(sorted(map(int, rng.choice(c.n, 2, replace=False)))) for _ in range(400)}
    groups = {
        "folio_unlinked": sample({e for e in folio if not c.linked(*e)}, 100),
        "entity_only_unlinked": sample({e for e in entity - folio if not c.linked(*e)}, 50),
        "control_author_linked": sample({e for e in folio | entity if c.linked(*e)}, 30),
        "control_random": sample({e for e in randoms if ok(*e)} - folio - entity, 30),
    }
    pairs = [(g, i, j) for g, es in groups.items() for i, j in es]
    pairs = [pairs[k] for k in rng.permutation(len(pairs))]

    out = corpus.OUT / "judge"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps([{"pid": k, "group": g, "a": c.key[i], "b": c.key[j]} for k, (g, i, j) in enumerate(pairs)]))
    schema = out / "schema.json"
    schema.write_text(json.dumps(SCHEMA))
    print({g: len(es) for g, es in groups.items()}, MODEL, flush=True)

    cut = lambda t: t if len(t) <= CAP else t[:CAP] + "\n[...]"

    def one(b):
        f = out / f"judge-{b:03}.json"
        if f.exists():
            return b, "kept", 0
        t0 = time.time()
        chunk = range(b * PER_CALL, min((b + 1) * PER_CALL, len(pairs)))
        prompt = INSTR + "\n\n".join(
            f"=== PAIR {k} ===\n--- A: {c.path[pairs[k][1]]}\n{cut(c.text[pairs[k][1]])}\n"
            f"--- B: {c.path[pairs[k][2]]}\n{cut(c.text[pairs[k][2]])}" for k in chunk)
        try:
            obj, usage = corpus.agy(MODEL, schema, prompt)
        except RuntimeError as ex:
            return b, f"FAIL {ex}", time.time() - t0
        obj["usage"] = usage
        f.write_text(json.dumps(obj))
        return b, f"ok {len(obj['judgements'])}/{len(chunk)}", time.time() - t0

    with ThreadPoolExecutor(WORKERS) as ex:
        for b, status, dt in ex.map(one, range((len(pairs) + PER_CALL - 1) // PER_CALL)):
            print(f"call {b:3}: {status} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
