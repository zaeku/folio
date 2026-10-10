"""What every step reads: the pinned corpus, its index, the authors' links, and the graphs."""

import json
import math
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402  the pinned corpora and how to fetch them

NAME = "rust-book"
ROOT = HERE.parent / "corpora" / NAME
OUT = HERE.parent / "reports" / "links"
K = 5


def prepare():
    """Fetch the pinned corpus and index it, unless both are already there."""
    run.fetch(NAME, run.CORPORA[NAME], keep=True)
    if not (ROOT / ".folio" / "index.db").exists():
        subprocess.run(["folio", "index", "--max-chars", str(run.MAX_CHARS)], cwd=ROOT, check=True)
    OUT.mkdir(parents=True, exist_ok=True)


class Corpus:
    def __init__(self):
        db = sqlite3.connect(ROOT / ".folio" / "index.db")
        rows = db.execute("select slot,path,start,stop,heading from sections order by slot").fetchall()
        dim = int(db.execute("select v from meta where k='dim'").fetchone()[0])
        vectors = np.fromfile(ROOT / ".folio" / "vectors.f32", dtype="<f4").reshape(-1, dim)
        self.X = vectors[[r[0] for r in rows]]
        self.n = len(rows)
        self.path = [r[1] for r in rows]
        self.head = [r[4] or "" for r in rows]
        # A slot is an index's own numbering; a section is named by where it starts.
        self.key = [(r[1], r[2]) for r in rows]
        self.at = {k: i for i, k in enumerate(self.key)}
        lines = {p: (ROOT / p).read_text().splitlines() for p in set(self.path)}
        self.text = ["\n".join(lines[p][s - 1:e]) for _, p, s, e, _ in rows]
        self.truth = self._links(set(lines))

    def _links(self, files):
        """File pairs the book's authors linked, from inline links and reference definitions."""
        link = re.compile(r"\]\(([\w\-]+)\.(?:html|md)(?:#[^)]*)?\)|^\[[^\]]+\]:\s*([\w\-]+)\.(?:html|md)", re.M)
        truth = set()
        for i, t in enumerate(self.text):
            for a, b in link.findall(t):
                target = (a or b) + ".md"
                if target != self.path[i] and target in files:
                    truth.add(frozenset((self.path[i], target)))
        return truth

    def linked(self, i, j):
        return frozenset((self.path[i], self.path[j])) in self.truth

    def files(self, edges):
        return {frozenset((self.path[i], self.path[j])) for i, j in edges if self.path[i] != self.path[j]}

    def knn(self, S, k=K):
        """Directed edges from each section to its k best-scoring others."""
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        idx = np.argsort(-S, axis=1)[:, :k]
        return {(i, int(j)) for i in range(self.n) for j in idx[i] if S[i, j] > 0}

    def folio_graph(self, k=K):
        return self.knn(self.X @ self.X.T, k)

    def bag_graph(self, bags, k=K):
        """tf-idf cosine over named things, keeping names in 2 to 20% of sections."""
        df = Counter(e for b in bags for e in set(b))
        vocab = {e: c for c, e in enumerate(e for e, d in df.items() if 2 <= d <= self.n * 0.2)}
        M = np.zeros((self.n, len(vocab)), dtype=np.float32)
        for i, b in enumerate(bags):
            for e, c in Counter(b).items():
                if e in vocab:
                    M[i, vocab[e]] = (1 + math.log(c)) * math.log(self.n / df[e])
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        empty = int((norm == 0).sum())
        norm[norm == 0] = 1
        M /= norm
        return self.knn(M @ M.T, k), len(vocab), empty

    def heuristic_bags(self):
        names = re.compile(r"`([^`\n]{2,40})`|\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
        return [[a or b for a, b in names.findall(t)] for t in self.text]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower().strip("`"))


def extraction(c):
    """Entities and triples per section, as extract.py left them."""
    n = c.n
    ents, rels, usage = [set() for _ in range(n)], [[] for _ in range(n)], Counter()
    files = sorted((OUT / "extract").glob("batch-*.json"))
    if not files:
        sys.exit("no extraction under reports/links/extract; run extract.py first")
    for f in files:
        d = json.loads(f.read_text())
        usage.update({k: v for k, v in (d.get("usage") or {}).items() if isinstance(v, int)})
        keys = d["keys"]
        for sec in d["sections"]:
            i = c.at.get(tuple(keys.get(str(sec["id"]), ())))
            if i is not None:
                ents[i] |= {norm(e) for e in sec["entities"] if e.strip()}
                for r in sec["relations"]:
                    s, o = norm(r["s"]), norm(r["o"])
                    rels[i].append((s, norm(r["p"]), o))
                    ents[i] |= {s, o}
    return ents, rels, usage


def agy(model, schema, prompt):
    """One structured print run. Returns (object, usage) or raises with what came back."""
    r = subprocess.run(
        ["timeout", "900", "agy", "--model", model, "--disable-slash-commands",
         "--output-format", "json", "--json-schema", str(schema), "--print-timeout", "14m", "-p", prompt],
        capture_output=True, text=True,
    )
    try:
        meta = json.loads(r.stdout)
        obj = meta.get("structured_output") or json.loads(meta["response"][meta["response"].index("{"):])
    except Exception as ex:
        raise RuntimeError(f"rc={r.returncode} {r.stdout[-400:]} {r.stderr[-400:]}") from ex
    return obj, meta.get("usage")
