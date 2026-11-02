#!/usr/bin/env python3
"""Draw random functions, have agy write one question per function, in batches.

usage: generate-code.py <corpus-dir> <units.json> <out-dir> [seed=58] [rust=60,python=60,cpp=30]
Draws that many functions per language, each of at least 200 characters, tests
left out, and writes <out-dir>/pairs.json as [{q, file, line, lang, hash}], line
being the unit's start. A question that contains a piece of the function's name
is dropped, or the lexical arms win on a leak rather than on search; a piece of
a parameter or type name is an ordinary word as often as not, and is allowed.
The dropped questions go to <out-dir>/dropped.json with the piece that matched.
"""
import json, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'links'))
import corpus  # noqa: E402

MODEL = "gemini-3.8-flash-low"
STRATA = {'rust': 60, 'python': 60, 'cpp': 30}
SCHEMA = {"type": "object", "required": ["queries"], "properties": {"queries": {"type": "array", "items": {
    "type": "object", "required": ["id", "q"], "properties": {"id": {"type": "integer"}, "q": {"type": "string"}}}}}}
INSTR = """For each function below, write ONE search query a developer on this codebase would type to find
it without having read it. Describe what it does in your own words. Do not use the function's name, its
parameter names, its type names, or any other identifier from it, and do not copy any phrase of five or more
words from it. 5 to 15 words, in English. One query per function id.
Return ONLY the JSON object matching the schema. Do not use tools.

"""
root, units = Path(sys.argv[1]), json.load(open(sys.argv[2]))
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 58
if len(sys.argv) > 5:
    STRATA = {lang: int(n) for lang, n in (s.split('=') for s in sys.argv[5].split(','))}
src = {}


def text(u):
    if u['file'] not in src:
        src[u['file']] = (root / u['file']).read_text(errors='replace').split('\n')
    return '\n'.join(src[u['file']][u['start'] - 1:u['end']])


def above(u):
    """The comment, attribute and decorator lines the unit's span took in."""
    lines = text(u).split('\n')
    n = 0
    while n < len(lines) - 1 and lines[n].strip().startswith(('//', '#', '@')):
        n += 1
    return lines[:n], '\n'.join(lines[n:])


def name(u):
    attrs, t = above(u)
    m = re.search(r'\b(?:fn|def)\s+(\w+)', t) if u['lang'] != 'cpp' else None
    if m:
        return m.group(1)
    return re.findall(r'[~A-Za-z_][A-Za-z0-9_]*', t.split('(', 1)[0])[-1]


def is_test(u):
    attrs, _ = above(u)
    return (any('test' in a for a in attrs if a.strip().startswith('#[')) or name(u).startswith('test')
            or '/tests/' in f"/{u['file']}" or Path(u['file']).name.startswith('test_'))


# Function words a name is built with and a question is written with alike.
GLUE = {'from', 'into', 'with', 'that', 'this', 'when', 'then', 'each', 'over', 'only', 'does', 'have'}


def pieces(u):
    return {w.lower() for w in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+', name(u))
            if len(w) >= 4 and w.lower() not in GLUE}


words = lambda t: re.findall(r"\w+", t.lower())


def leaks(q, u):
    return next((p for w in words(q) for p in sorted(pieces(u)) if w.startswith(p)), None)


def copied(q, t, k=5):
    q, t = words(q), " ".join(words(t))
    return any(" ".join(q[j:j + k]) in t for j in range(len(q) - k + 1))


rng = random.Random(seed)
picked = []
for lang, n in STRATA.items():
    pool = [i for i, u in enumerate(units) if u['lang'] == lang and len(text(u)) >= 200 and not is_test(u)]
    picked += rng.sample(pool, n)
batches = [picked[k:k + 10] for k in range(0, len(picked), 10)]
(out / "schema.json").write_text(json.dumps(SCHEMA))


def call(batch):
    prompt = INSTR + "\n\n".join(f"<<<FUNCTION {i}>>>\n{text(units[i])}" for i in batch)
    obj, _ = corpus.agy(MODEL, out / "schema.json", prompt)
    return batch, obj["queries"]


pairs, dropped, leaked, copies = [], [], 0, 0
with ThreadPoolExecutor(4) as ex:
    for batch, qs in ex.map(call, batches):
        for e in qs:
            i = e["id"]
            if i not in batch:
                continue
            u = units[i]
            hit = leaks(e["q"], u)
            if hit:
                leaked += 1
                dropped.append({"q": e["q"], "piece": hit, "file": u['file'], "line": u['start']})
                continue
            if copied(e["q"], text(u)):
                copies += 1
                continue
            pairs.append({"q": e["q"], "file": u['file'], "line": u['start'], "lang": u['lang'], "hash": u['hash']})
(out / "pairs.json").write_text(json.dumps(pairs, indent=1))
(out / "dropped.json").write_text(json.dumps(dropped, indent=1))
print(f"{len(pairs)} pairs from {len(picked)} functions in {len(batches)} calls, {leaked} dropped as leaking "
      f"an identifier, {copies} as copied, {MODEL} seed {seed}")
