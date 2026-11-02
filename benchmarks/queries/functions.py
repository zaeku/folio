#!/usr/bin/env python3
"""Cut a source tree into function units with `ast-editor inspect --template functions`.

usage: functions.py <corpus-dir> > units.json
Writes [{file, lang, start, end, hash}]. A unit's span takes in the comment,
attribute and decorator lines directly above the definition. A C++ span whose
first line has no "(" is a namespace or class that error recovery returned as a
function, and is left out. `hash` is of the whitespace-normalised body, so two
copies of one function share it.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

LANG = {'.rs': 'rust', '.py': 'python', '.cpp': 'cpp', '.cc': 'cpp', '.h': 'cpp', '.hpp': 'cpp'}
ABOVE = {'rust': ('//', '#['), 'cpp': ('//',), 'python': ('@', '#')}

root = Path(sys.argv[1])
units, dropped = [], 0
for p in sorted(root.rglob('*')):
    lang = LANG.get(p.suffix)
    if not lang or not p.is_file():
        continue
    r = subprocess.run(['ast-editor', 'inspect', str(p), '--template', 'functions', '--include-code', 'false'],
                       capture_output=True, text=True, check=True)
    found = json.loads(r.stdout.split('```json', 1)[1].rsplit('```', 1)[0])
    lines = p.read_text(errors='replace').split('\n')
    for m in found['matches']:
        start, end = m['definition']['start_line'], m['definition']['end_line']
        if lang == 'cpp' and '(' not in lines[start - 1]:
            dropped += 1
            continue
        body = ' '.join('\n'.join(lines[start - 1:end]).split())
        while start > 1 and lines[start - 2].strip().startswith(ABOVE[lang]):
            start -= 1
        units.append({'file': str(p.relative_to(root)), 'lang': lang, 'start': start, 'end': end,
                      'hash': hashlib.sha1(body.encode()).hexdigest()[:16]})
json.dump(units, sys.stdout)
counts = {k: sum(u['lang'] == k for u in units) for k in ('rust', 'python', 'cpp')}
print(f"{counts}, {dropped} C++ container span(s) left out", file=sys.stderr)
