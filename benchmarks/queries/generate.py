#!/usr/bin/env python3
"""Draw random rust-book sections, have agy write one question per section, in batches.

usage: generate.py <out-dir> [n=150] [batch=10] [seed=55]
Writes <out-dir>/pairs.json as [{q, file, line}] with line = the section's start.
"""
import json, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'links'))
import corpus  # noqa: E402

MODEL = "gemini-3.7-flash-low"
SCHEMA = {"type": "object", "required": ["queries"], "properties": {"queries": {"type": "array", "items": {
    "type": "object", "required": ["id", "q"], "properties": {"id": {"type": "integer"}, "q": {"type": "string"}}}}}}
INSTR = """For each section below, write ONE search query a Rust learner would type to find that section,
without having read it. Describe the need in your own words; do not copy any phrase of five or more words
from the section, and do not quote its heading. 5 to 15 words. One query per section id.
Return ONLY the JSON object matching the schema. Do not use tools.

"""

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n, size, seed = (int(a) for a in (sys.argv[2:] + ['150', '10', '55'][len(sys.argv) - 2:]))
c = corpus.Corpus()
pool = [i for i in range(c.n) if len(c.text[i]) >= 400]
picked = random.Random(seed).sample(pool, n)
batches = [picked[k:k + size] for k in range(0, n, size)]
(out / "schema.json").write_text(json.dumps(SCHEMA))
words = lambda t: re.findall(r"\w+", t.lower())


def copied(q, text, k=5):
    q, t = words(q), " ".join(words(text))
    return any(" ".join(q[j:j + k]) in t for j in range(len(q) - k + 1))


def call(batch):
    prompt = INSTR + "\n\n".join(f"<<<SECTION {i}>>>\n{c.text[i]}" for i in batch)
    obj, _ = corpus.agy(MODEL, out / "schema.json", prompt)
    return batch, obj["queries"]


pairs, dropped = [], 0
with ThreadPoolExecutor(4) as ex:
    for batch, qs in ex.map(call, batches):
        for e in qs:
            i = e["id"]
            if i not in batch:
                continue
            if copied(e["q"], c.text[i]):
                dropped += 1
                continue
            pairs.append({"q": e["q"], "file": c.path[i], "line": c.key[i][1]})
(out / "pairs.json").write_text(json.dumps(pairs, indent=1))
print(f"{len(pairs)} pairs from {n} sections in {len(batches)} calls, {dropped} dropped as copied, {MODEL} seed {seed}")
