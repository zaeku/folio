#!/usr/bin/env python3
"""Rewrite a query set's questions in another language, keeping everything else.

usage: translate.py <pairs.json> <language> <out.json> [--all-terms]
Each question is rewritten as a developer who speaks <language> would type it,
keeping the technical terms such a developer writes in English, or with
--all-terms translating those too, an identifier excepted. The model is
the one generate-code.py uses, and the rest of each pair is copied unchanged.
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'links'))
import corpus  # noqa: E402

MODEL = "gemini-3.8-flash-low"
SCHEMA = {"type": "object", "required": ["queries"], "properties": {"queries": {"type": "array", "items": {
    "type": "object", "required": ["id", "q"], "properties": {"id": {"type": "integer"}, "q": {"type": "string"}}}}}}
pairs, lang, out = json.load(open(sys.argv[1])), sys.argv[2], Path(sys.argv[3])
TERMS = ("Translate technical terms too, into the word {lang} uses or written in its script; keep only an "
         "identifier that appears in code as it is." if '--all-terms' in sys.argv[4:] else
         "Keep a technical term in English where such a developer would write it in English; translate the rest.")
INSTR = f"""Rewrite each search query below in {lang}, the way a developer who speaks {lang} would type it
into a search box. {TERMS.format(lang=lang)} Keep the meaning; do not add or drop detail. One query per id.
Return ONLY the JSON object matching the schema. Do not use tools.

"""
schema = out.with_suffix('.schema.json')
schema.write_text(json.dumps(SCHEMA))


def call(ids):
    prompt = INSTR + "\n".join(f"{i}: {pairs[i]['q']}" for i in ids)
    obj, _ = corpus.agy(MODEL, schema, prompt)
    return {e["id"]: e["q"] for e in obj["queries"] if e["id"] in ids}


batches = [list(range(k, min(k + 25, len(pairs)))) for k in range(0, len(pairs), 25)]
done = {}
with ThreadPoolExecutor(4) as ex:
    for got in ex.map(call, batches):
        done.update(got)
missing = [i for i in range(len(pairs)) if i not in done]
if missing:
    sys.exit(f"{len(missing)} question(s) came back untranslated, starting with {missing[0]}")
out.write_text(json.dumps([{**p, "q": done[i]} for i, p in enumerate(pairs)], ensure_ascii=False, indent=1))
schema.unlink()
print(f"{len(pairs)} questions in {lang}, {MODEL}")
