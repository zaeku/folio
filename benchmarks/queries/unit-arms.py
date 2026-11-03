#!/usr/bin/env python3
"""Compare four ways of cutting a source tree into units, scored by line coverage.

usage: unit-arms.py <corpus-dir> <units.json> <pairs.json> <gemma-endpoint> > out.json
Arms, each embedded with Gemma behind the code-retrieval prefix and the filename
as title, each unit cut at 8,000 characters:
  window    40 lines every 20, no parsing
  pattern   a declaration line found by regex, with the comment and attribute lines
            above it, to its closing brace by counting braces, or in Python to
            the last line indented under it
  function  the Tree-sitter units of functions.py (<units.json>)
  parent    those units, each led by the declaration line and doc comment of the
            impl, trait, class or struct around it and the file's `//!` lines
  container the same without the file's `//!` lines
A question's target is the function span in <units.json>. A unit hits when it
overlaps the target by at least half of the shorter of the two, so landing
inside a long function is finding it. Reading cost is the number of distinct
lines in the ranked units up to and including the first hit.
"""
import json, re, subprocess, sys, urllib.request
from pathlib import Path

import numpy as np

BUDGET, WIN, STRIDE = 8000, 40, 20
LANG = {'.rs': 'rust', '.py': 'python', '.cpp': 'cpp', '.cc': 'cpp', '.h': 'cpp', '.hpp': 'cpp'}
ABOVE = ('//', '#[', '#', '@', '/*', '*')
DECL = {'rust': re.compile(r'^\s*(pub(\([^)]*\))?\s+)?(default\s+)?(const\s+)?(async\s+)?(unsafe\s+)?(extern\s+"[^"]*"\s+)?fn\s+\w'),
        'python': re.compile(r'^\s*(async\s+)?def\s+\w'),
        'cpp': re.compile(r'^[ \t]*(?![ \t])(?!(if|for|while|switch|return|else|do|case|catch|delete|new)\b)[\w:<>,*&~\s]*\w[\w:<>~]*\s*\([^;{}]*\)?\s*(const)?\s*(noexcept)?\s*(override|final)?\s*(\{.*)?$')}
CONTAINER = {'rust': ['impls', 'traits'], 'python': ['classes'], 'cpp': ['classes']}
root, funcs, pairs, url = Path(sys.argv[1]), json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4]
files = [p for p in sorted(root.rglob('*')) if p.suffix in LANG and p.is_file() and '.codegraph' not in p.parts]
src = {str(p.relative_to(root)): p.read_text(errors='replace').split('\n') for p in files}
lang = {f: LANG[Path(f).suffix] for f in src}


def text(f, a, b):
    return '\n'.join(src[f][a - 1:b])


def window_units():
    out = []
    for f, ls in src.items():
        for a in range(1, max(2, len(ls) - WIN + STRIDE + 1), STRIDE):
            out.append((f, a, min(a + WIN - 1, len(ls)), ''))
    return out


def pattern_units():
    out = []
    for f, ls in src.items():
        rx = DECL[lang[f]]
        for i, line in enumerate(ls):
            if not rx.match(line):
                continue
            if lang[f] == 'python':
                ind = len(line) - len(line.lstrip())
                j = i + 1
                while j < len(ls) and (not ls[j].strip() or len(ls[j]) - len(ls[j].lstrip()) > ind):
                    j += 1
                end = j
                while end > i + 1 and not ls[end - 1].strip():
                    end -= 1
            else:
                depth, opened, end = 0, False, None
                for j in range(i, min(len(ls), i + 3000)):
                    s = re.sub(r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])\'|//.*', '', ls[j])
                    if not opened and ';' in s and '{' not in s:
                        break
                    depth += s.count('{') - s.count('}')
                    opened = opened or '{' in s
                    if opened and depth <= 0:
                        end = j + 1
                        break
                if end is None:
                    continue
            start = i + 1
            while start > 1 and ls[start - 2].strip().startswith(ABOVE):
                start -= 1
            out.append((f, start, end, ''))
    return out


def containers(f):
    spans = []
    for t in CONTAINER[lang[f]]:
        r = subprocess.run(['ast-editor', 'inspect', str(root / f), '--template', t, '--include-code', 'false'],
                           capture_output=True, text=True)
        if r.returncode or '```json' not in r.stdout:
            continue
        spans += [(m['start_line'], m['end_line']) for m in json.loads(r.stdout.split('```json', 1)[1].rsplit('```', 1)[0])['matches']]
    return spans


def parent_units(module=True):
    out, cache = [], {}
    for u in funcs:
        f = u['file']
        if f not in cache:
            ls = src[f]
            mod = '\n'.join(l for l in ls[:40] if l.lstrip().startswith('//!'))[:400]
            cache[f] = (mod, containers(f))
        mod, spans = cache[f]
        around = [(a, b) for a, b in spans if a <= u['start'] and u['end'] <= b and (a, b) != (u['start'], u['end'])]
        head = []
        if around:
            a, _ = max(around)
            k = a
            while k > 1 and src[f][k - 2].strip().startswith(ABOVE):
                k -= 1
            decl = text(f, a, a + 3).split('{', 1)[0]
            head.append((text(f, k, a - 1) + '\n' + decl).strip()[:800])
        if mod and module:
            head.insert(0, mod)
        out.append((f, u['start'], u['end'], '\n'.join(head)))
    return out


def embed(texts):
    o = []
    for j in range(0, len(texts), 16):
        r = urllib.request.Request(url, json.dumps({"input": texts[j:j + 16]}).encode(), {"Content-Type": "application/json"})
        o += [d["embedding"] for d in json.load(urllib.request.urlopen(r))["data"]]
    X = np.array(o, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


end_of = {(u['file'], u['start']): u['end'] for u in funcs}
targets = [(p['file'], p['line'], end_of[(p['file'], p['line'])]) for p in pairs]
Q = embed(["task: code retrieval | query: " + p['q'] for p in pairs])


def hit(u, t):
    if u[0] != t[0]:
        return False
    overlap = min(u[2], t[2]) - max(u[1], t[1]) + 1
    return overlap >= 0.5 * min(u[2] - u[1] + 1, t[2] - t[1] + 1)


def score(units):
    docs = [f"title: {Path(f).name} | text: " + ((h + '\n') if h else '') + text(f, a, b) for f, a, b, h in units]
    D = embed([d[:BUDGET + 200] for d in docs])
    ranks, cost = [], []
    for t, q in zip(targets, Q):
        order = np.argsort(-(D @ q))[:200]
        seen, r = set(), None
        for k, i in enumerate(order, 1):
            f, a, b, _ = units[i]
            seen.update((f, x) for x in range(a, b + 1))
            if hit(units[i], t):
                r = k
                break
        ranks.append(r)
        cost.append(len(seen) if r else None)
    at = lambda k: sum(1 for r in ranks if r and r <= k)
    within = lambda n: sum(1 for c in cost if c and c <= n)
    sizes = sorted(b - a + 1 for _, a, b, _ in units)
    return {"units": len(units), "median_lines": sizes[len(sizes) // 2],
            "top1": at(1), "top5": at(5), "top10": at(10), "top20": at(20),
            "mrr": round(sum(1 / r for r in ranks if r and r <= 20) / len(ranks), 3),
            "within_lines": {n: within(n) for n in (100, 200, 500, 1000, 2000)},
            "ranks": ranks, "cost": cost}


arms = {"window": window_units(), "pattern": pattern_units(),
        "function": [(u['file'], u['start'], u['end'], '') for u in funcs], "parent": parent_units(),
        "container": parent_units(module=False)}
if len(sys.argv) > 5 and sys.argv[5].startswith('--only='):
    arms = {k: arms[k] for k in sys.argv[5][7:].split(',')}
if len(sys.argv) > 5 and sys.argv[5] == '--count':
    print({k: (len(v), sum(any(hit(u, t) for u in v if u[0] == t[0]) for t in targets)) for k, v in arms.items()})
    sys.exit()
print(json.dumps({"pairs": len(pairs), "arms": {k: score(v) for k, v in arms.items()}}, indent=1))
