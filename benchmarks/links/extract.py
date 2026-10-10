#!/usr/bin/env python3
"""Ask a model for the entities and triples of every section, in batches, through agy."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import corpus

MODEL = "gemini-3.7-flash-low"
BUDGET = 40_000  # characters per batch
WORKERS = 4
INSTR = """You build a knowledge graph from sections of "The Rust Programming Language" book.
For EVERY section below (each starts with <<<SECTION id>>>), return:
- entities: the Rust concepts, language features, types, traits, macros, crates, tools and example programs the section substantively discusses (not passing mentions). Use short canonical lowercase names, singular, without backticks, e.g. "ownership", "borrow checker", "vec<t>", "iterator trait", "cargo", "guessing game".
- relations: up to 8 (subject, predicate, object) triples between those entities that the section states, predicate in a few lowercase words.
Return ONLY the JSON object matching the schema, with one entry per section id. Do not use tools.

"""
SCHEMA = {"type": "object", "properties": {"sections": {"type": "array", "items": {
    "type": "object", "properties": {
        "id": {"type": "integer"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "s": {"type": "string"}, "p": {"type": "string"}, "o": {"type": "string"}},
            "required": ["s", "p", "o"]}}},
    "required": ["id", "entities", "relations"]}}}, "required": ["sections"]}


def main():
    corpus.prepare()
    c = corpus.Corpus()
    out = corpus.OUT / "extract"
    out.mkdir(parents=True, exist_ok=True)
    schema = out / "schema.json"
    schema.write_text(json.dumps(SCHEMA))

    batches, cur, size = [], [], 0
    for i, t in enumerate(c.text):
        t = t[:8000]
        if cur and size + len(t) > BUDGET:
            batches.append(cur)
            cur, size = [], 0
        cur.append((i, t))
        size += len(t)
    batches.append(cur)

    def one(b):
        f = out / f"batch-{b:03}.json"
        if f.exists():
            return b, "kept", 0
        t0 = time.time()
        prompt = INSTR + "\n\n".join(f"<<<SECTION {i}>>>\n{t}" for i, t in batches[b])
        try:
            obj, usage = corpus.agy(MODEL, schema, prompt)
        except RuntimeError as ex:
            return b, f"FAIL {ex}", time.time() - t0
        obj["usage"] = usage
        obj["keys"] = {str(i): c.key[i] for i, _ in batches[b]}
        f.write_text(json.dumps(obj))
        return b, f"ok {len(obj['sections'])}/{len(batches[b])}", time.time() - t0

    print(f"{c.n} sections in {len(batches)} batches, {MODEL}", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for b, status, dt in ex.map(one, range(len(batches))):
            print(f"batch {b:3}: {status} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
